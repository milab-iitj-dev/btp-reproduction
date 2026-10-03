#!/usr/bin/env python3
"""
E3 - Feature-similarity probe.

Tests the core claim of Phase 3 directly, without running a benchmark:

  Text patches are NEAR-DUPLICATES in feature space (high cosine similarity) even though
  each one carries unique information. BTP's diversity selector measures only feature
  similarity, so it treats a line of text as a redundant cluster and deduplicates it.

For each image the script reports:
  1. mean pairwise cosine similarity within text patches, within background, and across
  2. the share of the diversity selector's picks that land on text, vs text's share of the image
  3. the same for an attention-based selector

Prediction if the hypothesis holds:
  sim(text,text) > sim(bg,bg)            text is the TIGHTER cluster
  div_pick_rate(text) < text_share       diversity under-samples text
  attn_pick_rate(text) > text_share      attention over-samples text

Usage
  python feature_similarity_probe.py --images img1.jpg img2.jpg --out probe.json
  python feature_similarity_probe.py --image-dir ./probe_images --layer 4 --out probe.json

Requires: transformers (Qwen2.5-VL), torch, easyocr, pillow.
NOTE: untested on GPU yet. Run on ONE image first.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image


# ----------------------------------------------------------------------------- selectors
def farthest_point_select(feats, k):
    """Greedy farthest-point sampling on cosine distance. Mirrors BTP's div_prune."""
    x = F.normalize(feats.float(), p=2, dim=1)
    d = 1.0 - torch.matmul(x, x.t())
    d.fill_diagonal_(float('inf'))
    chosen = []
    for _ in range(min(k, feats.shape[0])):
        if not chosen:
            row_min, _ = d.min(dim=1)
            idx = int(row_min.argmax().item())
        else:
            sel = d[chosen, :]
            min_vals, _ = torch.min(sel, dim=0)
            idx = int(min_vals.argmax().item())
        chosen.append(idx)
        d[:, idx] = float('-inf')
    return chosen


def attention_select(attn_vec, k):
    take = min(k, attn_vec.shape[0])
    _, idx = torch.topk(attn_vec.float(), take)
    return idx.tolist()


def mean_pairwise_cos(feats):
    if feats.shape[0] < 2:
        return float('nan')
    x = F.normalize(feats.float(), p=2, dim=1)
    s = torch.matmul(x, x.t())
    n = s.shape[0]
    off = (s.sum() - s.diagonal().sum()) / (n * n - n)
    return float(off.item())


def mean_cross_cos(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float('nan')
    xa = F.normalize(a.float(), p=2, dim=1)
    xb = F.normalize(b.float(), p=2, dim=1)
    return float(torch.matmul(xa, xb.t()).mean().item())


# ----------------------------------------------------------------------------- text mask
def build_text_mask(image_path, grid_h, grid_w, reader, conf=0.3):
    """Return a boolean list of length grid_h*grid_w, True where the patch overlaps OCR text."""
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    mask = [False] * (grid_h * grid_w)
    results = reader.readtext(np.array(img))
    boxes = []
    for box, _text, score in results:
        if score is None or score < conf:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    for (x0, y0, x1, y1) in boxes:
        c0 = max(0, int(x0 / W * grid_w))
        c1 = min(grid_w - 1, int(x1 / W * grid_w))
        r0 = max(0, int(y0 / H * grid_h))
        r1 = min(grid_h - 1, int(y1 / H * grid_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mask[r * grid_w + c] = True
    return mask, boxes


# ----------------------------------------------------------------------------- main probe
def probe_image(model, processor, image_path, question, layer, reader, budget_frac, conf):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    # where the image tokens sit in the sequence
    image_token_id = model.config.image_token_id
    ids = inputs["input_ids"][0]
    img_pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if img_pos.numel() == 0:
        raise RuntimeError("no image tokens found; check image_token_id for this checkpoint")
    img_start = int(img_pos[0].item())
    n_img = int(img_pos.numel())

    # merged patch grid (Qwen merges 2x2 patches)
    thw = inputs["image_grid_thw"][0].tolist()   # [t, h, w] in raw patches
    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
    grid_h, grid_w = thw[1] // merge, thw[2] // merge
    if grid_h * grid_w != n_img:
        print("  warning: grid {}x{}={} but {} image tokens".format(
            grid_h, grid_w, grid_h * grid_w, n_img))

    # capture hidden states entering the pruning layer, and that layer's attention
    grabbed = {}

    def hook(mod, args, kwargs, output):
        hs = output[0] if isinstance(output, tuple) else output
        grabbed["hidden"] = hs.detach()[0]
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            grabbed["attn"] = output[1].detach()

    target_layer = model.model.layers[max(0, layer - 1)]
    h = target_layer.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(**inputs, output_attentions=True)
    h.remove()

    hidden = grabbed["hidden"]
    feats = hidden[img_start:img_start + n_img].float().cpu()

    # attention from the last token onto the image block
    attn_vec = None
    if "attn" in grabbed:
        a = grabbed["attn"]                       # [b, heads, q, k]
        last = a[:, :, -1, :]                     # [b, heads, k]
        attn_vec = last[:, :, img_start:img_start + n_img].mean(dim=1).squeeze().float().cpu()

    mask, boxes = build_text_mask(image_path, grid_h, grid_w, reader, conf)
    mask = mask[:n_img] + [False] * max(0, n_img - len(mask))
    tmask = torch.tensor(mask[:n_img], dtype=torch.bool)

    text_feats = feats[tmask]
    bg_feats = feats[~tmask]
    text_share = float(tmask.sum().item()) / max(1, n_img)

    k = max(1, int(round(budget_frac * n_img)))
    div_pick = farthest_point_select(feats.clone(), k)
    div_on_text = sum(1 for i in div_pick if bool(tmask[i])) / max(1, len(div_pick))

    attn_on_text = None
    if attn_vec is not None and attn_vec.numel() == n_img:
        attn_pick = attention_select(attn_vec, k)
        attn_on_text = sum(1 for i in attn_pick if bool(tmask[i])) / max(1, len(attn_pick))

    return {
        "image": os.path.basename(image_path),
        "n_image_tokens": n_img,
        "grid": [grid_h, grid_w],
        "n_ocr_boxes": len(boxes),
        "text_share_of_image": round(text_share, 4),
        "budget_k": k,
        "cos_text_text": round(mean_pairwise_cos(text_feats), 4),
        "cos_bg_bg": round(mean_pairwise_cos(bg_feats), 4),
        "cos_text_bg": round(mean_cross_cos(text_feats, bg_feats), 4),
        "diversity_picks_on_text": round(div_on_text, 4),
        "attention_picks_on_text": (round(attn_on_text, 4) if attn_on_text is not None else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--question", default="What does the text in the image say?")
    ap.add_argument("--layer", type=int, default=4, help="BTP's first pruning layer")
    ap.add_argument("--budget-frac", type=float, default=0.5, help="stage-1 keeps half")
    ap.add_argument("--ocr-conf", type=float, default=0.3)
    ap.add_argument("--out", default="feature_similarity_probe.json")
    args = ap.parse_args()

    paths = list(args.images)
    if args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(args.image_dir, f))
    if not paths:
        raise SystemExit("no images given; use --images or --image-dir")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import easyocr

    print("loading model:", args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    reader = easyocr.Reader(["en"], gpu=False)

    rows = []
    for p in paths:
        print("probing", p)
        try:
            rows.append(probe_image(model, processor, p, args.question,
                                    args.layer, reader, args.budget_frac, args.ocr_conf))
        except Exception as e:
            print("  FAILED:", repr(e))

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    # ---- summary -----------------------------------------------------------
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float('nan')

    print("\n================ SUMMARY over {} images ================".format(len(rows)))
    print("mean cosine similarity  text-text : {:.4f}".format(avg("cos_text_text")))
    print("mean cosine similarity      bg-bg : {:.4f}".format(avg("cos_bg_bg")))
    print("mean cosine similarity    text-bg : {:.4f}".format(avg("cos_text_bg")))
    print("text share of image               : {:.4f}".format(avg("text_share_of_image")))
    print("diversity picks landing on text   : {:.4f}".format(avg("diversity_picks_on_text")))
    print("attention picks landing on text   : {:.4f}".format(avg("attention_picks_on_text")))
    print("\nHypothesis holds if text-text > bg-bg, and diversity picks < text share.")


if __name__ == "__main__":
    main()

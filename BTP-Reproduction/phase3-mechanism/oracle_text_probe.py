#!/usr/bin/env python3
"""
E2 - Oracle text preservation.

E1 showed that HOW tokens are chosen does not matter: random selection scores the same as
BTP's attention-plus-diversity selection. This script asks the remaining question:

  If we cheat and force EVERY text token to survive, spending the rest of the same budget
  on background, does accuracy come back?

Two possible outcomes, both informative:

  RECOVERS  -> the constraint is "text tokens must survive". Selection does matter after all,
               just far more aggressively than attention can manage. Content-aware retention
               is then a viable direction.
  DOES NOT  -> the constraint is the total amount of surviving context, not its identity.
               H2 is confirmed, and text-aware pruning cannot rescue this task at 12.5%.

This is a DIAGNOSTIC that uses ground-truth OCR at inference. It is not a proposed method,
and the paper must say so.

How it works: the `oracle_text` variant of the modeling file reads the text-token indices for
the current image from the environment variable BTP_TEXT_IDX. This driver sets that variable
per image before each forward pass, which lmms-eval cannot do, hence a standalone script.

Usage
  python oracle_text_probe.py --n 100 --out results/phase3/e2_oracle.json
  python oracle_text_probe.py --n 5 --out results/phase3/e2_smoke.json     # smoke test

Before running, install the oracle variant:
  MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
  python patch_selector_variants.py --src ${MODFILE}.BTP --out-dir variants --modes oracle_text,btp
  cp variants/modeling_qwen2_5_vl.py.SEL_oracle_text $MODFILE

NOTE: needs a GPU. Run via sbatch.
"""

import argparse
import hashlib
import io
import json
import os
import re
import string

os.environ.setdefault('BTP_RETAIN', '0.125')

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MID = "Qwen/Qwen2.5-VL-7B-Instruct"
SHORT = "\nAnswer the question using a single word or phrase."
_ARTICLES = {"a", "an", "the"}


def normalize(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return " ".join(t for t in s.split() if t not in _ARTICLES)


def as_float(s):
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def is_correct(pred, refs):
    if not refs:
        return None
    p = normalize(pred)
    if not p:
        return False
    nrefs = [normalize(r) for r in refs]
    if p in nrefs:
        return True
    pf = as_float(p)
    if pf is not None:
        for r in refs:
            rf = as_float(r)
            if rf is not None and abs(pf - rf) < 1e-6:
                return True
    for r in nrefs:
        if r and (r == p or r in p.split() or (len(r) > 2 and r in p)):
            return True
    return False


def text_token_indices(img, grid_h, grid_w, reader, conf=0.3):
    """Indices of merged patches that overlap OCR text."""
    W, H = img.size
    idx = set()
    for box, _txt, score in reader.readtext(np.array(img)):
        if score is None or score < conf:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        c0 = max(0, int(min(xs) / W * grid_w))
        c1 = min(grid_w - 1, int(max(xs) / W * grid_w))
        r0 = max(0, int(min(ys) / H * grid_h))
        r1 = min(grid_h - 1, int(max(ys) / H * grid_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                idx.add(r * grid_w + c)
    return sorted(idx)


def load_image(raw):
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        try:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dataset", default="lmms-lab/textvqa")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="results/phase3/e2_oracle.json")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--ocr-conf", type=float, default=0.3)
    ap.add_argument("--max-scan", type=int, default=4000)
    ap.add_argument("--no-oracle", action="store_true",
                    help="control run: do NOT set BTP_TEXT_IDX, so the variant falls back")
    ap.add_argument("--retention", type=float, default=0.125,
                    help="token budget as a fraction. Needs a BTP_PARAM-derived variant, "
                         "which reads BTP_RETAIN. Default 0.125 is the shipped setting.")
    args = ap.parse_args()

    # must be set before the model file computes its per-stage factor
    os.environ["BTP_RETAIN"] = str(args.retention)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    proc = AutoProcessor.from_pretrained(MID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", device_map="cuda").eval()

    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    rows = []
    seen = set()
    scanned = 0
    for ex in ds:
        if len(rows) >= args.n or scanned >= args.max_scan:
            break
        scanned += 1
        img = load_image(ex.get("image"))
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        md5 = hashlib.md5(buf.getvalue()).hexdigest()
        if md5 in seen:
            continue
        seen.add(md5)

        question = str(ex.get("question") or "")
        refs = ex.get("answers") or ex.get("answer") or []
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        refs = [str(r) for r in refs]

        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question + SHORT}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msg)
        inp = proc(text=[text], images=imgs, videos=vids,
                   padding=True, return_tensors="pt").to("cuda")

        thw = inp["image_grid_thw"][0].tolist()
        gh, gw = thw[1] // merge, thw[2] // merge

        tidx = text_token_indices(img, gh, gw, reader, args.ocr_conf)
        n_tok = gh * gw
        budget = int(round(args.retention * n_tok))

        # The oracle can only force-keep as many tokens as the budget allows. If the text
        # alone exceeds the budget, record that: it means preservation is impossible here.
        over_budget = len(tidx) > budget
        if args.no_oracle:
            os.environ["BTP_TEXT_IDX"] = ""
        else:
            os.environ["BTP_TEXT_IDX"] = ",".join(str(i) for i in tidx[:budget])

        try:
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
            new = gen[0][inp["input_ids"].shape[1]:]
            pred = proc.tokenizer.decode(new, skip_special_tokens=True).strip()
        except Exception as e:
            print("fail:", repr(e))
            continue

        ok = is_correct(pred, refs)
        rows.append({
            "question": question,
            "prediction": pred,
            "references": refs,
            "correct": ok,
            "n_image_tokens": n_tok,
            "budget": budget,
            "n_text_tokens": len(tidx),
            "text_share": round(len(tidx) / max(1, n_tok), 4),
            "text_over_budget": over_budget,
        })
        if len(rows) % 10 == 0 or len(rows) <= 3:
            print("  {}/{}  correct={}  text_tokens={} budget={} over={}".format(
                len(rows), args.n, ok, len(tidx), budget, over_budget))

    json.dump(rows, open(args.out, "w"), indent=1)

    scored = [r for r in rows if r["correct"] is not None]
    acc = sum(1 for r in scored if r["correct"]) / len(scored) if scored else float("nan")
    fits = [r for r in scored if not r["text_over_budget"]]
    acc_fits = sum(1 for r in fits if r["correct"]) / len(fits) if fits else float("nan")

    print("\n================ E2 ORACLE TEXT PRESERVATION ================")
    print("mode                        : {}".format("CONTROL (no oracle)" if args.no_oracle else "ORACLE"))
    print("retention (token budget)    : {}".format(args.retention))
    print("images scored               : {}".format(len(scored)))
    print("accuracy                    : {:.3f}".format(acc))
    print("images where text fits budget: {}".format(len(fits)))
    print("accuracy on those           : {:.3f}".format(acc_fits))
    print("mean text share of image    : {:.3f}".format(
        sum(r["text_share"] for r in rows) / len(rows) if rows else float("nan")))
    print("\nReference points: unpruned baseline 82.7, BTP at 12.5% about 0.18.")
    print("If accuracy stays near 0.18, preserving text does not help and H2 stands.")
    print("If it climbs well above, the constraint really is text-token survival.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 3 capture, replaces capture_tokens3.py.

Fixes three problems found in the Phase 2 capture:

  1. DUPLICATE IMAGES. TextVQA / ChartQA / AI2D store several questions per image, and the
     old loop walked consecutive streamed rows, so the same picture was captured more than
     once. Four of twelve records were duplicates. Now images are deduplicated by md5.

  2. GENERIC QUESTION. The old script asked a fixed question ("What text is shown in the
     image?") instead of the dataset's own question, so the model's answer could not be
     scored. Now the real question is used.

  3. NO CORRECTNESS. The generated answer was discarded. Now the prediction, the reference
     answers and a correctness flag are stored, so per-word coverage can be correlated
     against right/wrong answers. That is what turns E4 into evidence rather than a
     description.

Requires the BTP_VIZ3 modeling file to be active (it populates M._BTP_STAGES / M._BTP_META).

Usage
  python capture_tokens4.py --n-per-cat 100 --out results/phase3/capture4.json
  python capture_tokens4.py --n-per-cat 5 --out results/phase3/capture4_smoke.json   # smoke test

NOTE: needs a GPU. Run via sbatch.
"""

import argparse
import hashlib
import io
import json
import os
import re
import string

os.environ['BTP_VIZ'] = '1'
os.environ.setdefault('BTP_RETAIN', '0.125')

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M
from qwen_vl_utils import process_vision_info

MID = "Qwen/Qwen2.5-VL-7B-Instruct"

# dataset, split, image key, category, question key, answer key(s)
PLAN = [
    ("lmms-lab/textvqa", "validation", "image", "text_heavy", "question", ["answers", "answer"]),
    ("lmms-lab/ChartQA", "test",       "image", "chart",      "question", ["answer", "answers"]),
    ("lmms-lab/ai2d",    "test",       "image", "diagram",    "question", ["answer", "answers"]),
]

_ARTICLES = {"a", "an", "the"}


def normalize(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def get_refs(ex, keys):
    for k in keys:
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            return [str(v)]
    return []


def is_correct(pred, refs):
    """VQA-style: correct if the prediction matches any reference after normalization."""
    if not refs:
        return None
    p = normalize(pred)
    if not p:
        return False
    norm_refs = [normalize(r) for r in refs]
    if p in norm_refs:
        return True
    # allow the reference to appear inside a short generated sentence
    for r in norm_refs:
        if r and (r == p or r in p.split() or (len(r) > 2 and r in p)):
            return True
    return False


def load_image(raw):
    if raw is None:
        return None
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
    ap.add_argument("--n-per-cat", type=int, default=100)
    ap.add_argument("--out", default="results/phase3/capture4.json")
    ap.add_argument("--image-dir", default="results/phase3/images")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-scan", type=int, default=4000,
                    help="give up on a dataset after this many streamed rows")
    args = ap.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    proc = AutoProcessor.from_pretrained(MID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", device_map="cuda").eval()

    def run_one(img, question):
        M._BTP_STAGES.clear()
        M._BTP_META[0] = None
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msg)
        inp = proc(text=[text], images=imgs, videos=vids,
                   padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        new = gen[0][inp["input_ids"].shape[1]:]
        pred = proc.tokenizer.decode(new, skip_special_tokens=True).strip()

        if M._BTP_META[0] is None or len(M._BTP_STAGES) < 3:
            return None, pred
        s = [np.array(x) for x in M._BTP_STAGES[:3]]
        final = s[0][s[1]][s[2]]
        meta = dict(M._BTP_META[0])
        meta["stage1_kept"] = s[0].tolist()
        meta["final_kept"] = [int(x) for x in final.tolist()]
        return meta, pred

    out = []
    for ds_name, split, ikey, cat, qkey, akeys in PLAN:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
        except Exception as e:
            print("SKIP", ds_name, e)
            continue

        seen_md5 = set()
        got = 0
        scanned = 0
        dup_skipped = 0
        for ex in ds:
            if got >= args.n_per_cat or scanned >= args.max_scan:
                break
            scanned += 1

            img = load_image(ex.get(ikey))
            if img is None:
                continue

            # ---- deduplicate by image content, THIS is the Phase 2 bug ----
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            md5 = hashlib.md5(buf.getvalue()).hexdigest()
            if md5 in seen_md5:
                dup_skipped += 1
                continue
            seen_md5.add(md5)

            question = ex.get(qkey) or "What does the text in the image say?"
            refs = get_refs(ex, akeys)

            try:
                rec, pred = run_one(img, str(question))
            except Exception as e:
                print("fail", ds_name, got, repr(e))
                continue
            if not rec:
                print("no capture", ds_name, got)
                continue

            tag = "{}_{}_{}".format(cat, ds_name.split("/")[-1], got)
            ip = os.path.join(args.image_dir, tag + ".png")
            img.save(ip)

            rec.update({
                "tag": tag,
                "category": cat,
                "dataset": ds_name,
                "image_path": ip,
                "image_md5": md5,
                "question": str(question),
                "prediction": pred,
                "references": refs,
                "correct": is_correct(pred, refs),
            })
            out.append(rec)
            got += 1
            if got % 10 == 0 or got <= 3:
                print("  {} {}/{}  correct={}  img_num={}".format(
                    cat, got, args.n_per_cat, rec["correct"], rec.get("img_num")))

        print("{}: captured {} unique images ({} duplicate rows skipped, {} rows scanned)".format(
            cat, got, dup_skipped, scanned))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    # ---- summary -----------------------------------------------------------
    print("\n================ CAPTURE SUMMARY ================")
    print("total records      :", len(out))
    print("unique images      :", len(set(r["image_md5"] for r in out)))
    by = {}
    for r in out:
        by.setdefault(r["category"], []).append(r)
    for c, rs in sorted(by.items()):
        scored = [r for r in rs if r["correct"] is not None]
        acc = (sum(1 for r in scored if r["correct"]) / len(scored)) if scored else float("nan")
        print("  {:11s} n={:4d}  scored={:4d}  accuracy_under_BTP={:.3f}".format(
            c, len(rs), len(scored), acc))
    print("\nSaved to", args.out)
    print("Next: word_coverage.py --capture {} --stage final_kept".format(args.out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
E4 - Per-word patch coverage.

Phase 2 measured the fraction of TEXT TOKENS removed and found text was removed less than
background (75% vs 90%). That metric hides the failure, because reading a word needs its
patches to survive TOGETHER. A word can lose its meaning while the global removal rate
looks favourable.

This script computes, for every OCR word in an image, the fraction of that word's patches
that survived pruning, then summarises how many words fall below a readability threshold.

Input: a capture JSON produced by the Phase 2 instrumentation, a list of records like
  {"image": "x.jpg", "grid": [h, w], "img_num": 888, "kept": [3, 7, 11, ...]}
where "kept" are indices into the image-token block AFTER all pruning stages.

Usage
  python word_coverage.py --capture ../text-failure-study/results/viz3/capture.json \
                          --image-dir ../text-failure-study/results/viz3/images \
                          --out word_coverage.json

NOTE: untested. Check one record first with --limit 1.
"""

import argparse
import json
import os

from PIL import Image


def ocr_words(image_path, reader, conf=0.3):
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    out = []
    for box, text, score in reader.readtext(np.array(img)):
        if score is None or score < conf:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({"text": text, "box": (min(xs), min(ys), max(xs), max(ys))})
    return out, W, H


def word_patches(box, W, H, grid_h, grid_w):
    x0, y0, x1, y1 = box
    c0 = max(0, int(x0 / W * grid_w))
    c1 = min(grid_w - 1, int(x1 / W * grid_w))
    r0 = max(0, int(y0 / H * grid_h))
    r1 = min(grid_h - 1, int(y1 / H * grid_h))
    idx = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            idx.append(r * grid_w + c)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", default="word_coverage.json")
    ap.add_argument("--ocr-conf", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="a word counts as readable if at least this fraction of its patches survive")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    records = json.load(open(args.capture))
    if isinstance(records, dict):
        records = records.get("records", records.get("images", []))
    if args.limit:
        records = records[:args.limit]

    rows = []
    for rec in records:
        name = rec.get("image") or rec.get("name")
        grid = rec.get("grid")
        kept = rec.get("kept") or rec.get("kept_rel") or []
        if not name or not grid:
            print("skipping record without image/grid:", list(rec.keys()))
            continue
        grid_h, grid_w = int(grid[0]), int(grid[1])
        path = os.path.join(args.image_dir, name)
        if not os.path.exists(path):
            cands = [f for f in os.listdir(args.image_dir) if f.startswith(os.path.splitext(name)[0])]
            if not cands:
                print("image not found:", name)
                continue
            path = os.path.join(args.image_dir, cands[0])

        kept_set = set(int(i) for i in kept)
        words, W, H = ocr_words(path, reader, args.ocr_conf)

        per_word = []
        for w in words:
            idx = word_patches(w["box"], W, H, grid_h, grid_w)
            if not idx:
                continue
            surv = sum(1 for i in idx if i in kept_set)
            frac = surv / len(idx)
            per_word.append({"text": w["text"], "patches": len(idx),
                             "survived": surv, "coverage": round(frac, 4)})

        readable = sum(1 for w in per_word if w["coverage"] >= args.threshold)
        rows.append({
            "image": os.path.basename(path),
            "n_words": len(per_word),
            "mean_word_coverage": round(sum(w["coverage"] for w in per_word) / len(per_word), 4) if per_word else None,
            "readable_words": readable,
            "readable_fraction": round(readable / len(per_word), 4) if per_word else None,
            "words": per_word,
        })
        print("{:28s} words={:3d} mean_cov={:.3f} readable={}/{}".format(
            rows[-1]["image"], rows[-1]["n_words"],
            rows[-1]["mean_word_coverage"] or 0.0, readable, len(per_word)))

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    covs = [r["mean_word_coverage"] for r in rows if r["mean_word_coverage"] is not None]
    reads = [r["readable_fraction"] for r in rows if r["readable_fraction"] is not None]
    print("\n================ SUMMARY ================")
    print("images analysed              : {}".format(len(rows)))
    if covs:
        print("mean per-word coverage       : {:.3f}".format(sum(covs) / len(covs)))
    if reads:
        print("fraction of words readable   : {:.3f}  (threshold {})".format(
            sum(reads) / len(reads), args.threshold))
    print("\nCompare this against the 25% of text tokens that survive globally.")
    print("If readable-word fraction is far below that, removal rate was hiding the failure.")


if __name__ == "__main__":
    main()

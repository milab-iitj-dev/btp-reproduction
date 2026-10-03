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
    ap.add_argument("--image-dir", default=None,
                    help="only needed if records lack an image_path field")
    ap.add_argument("--out", default="word_coverage.json")
    ap.add_argument("--ocr-conf", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="a word counts as readable if at least this fraction of its patches survive")
    ap.add_argument("--stage", default="final_kept",
                    help="which kept-index field to score: final_kept or stage1_kept")
    ap.add_argument("--merge-size", type=int, default=2,
                    help="Qwen merges 2x2 raw patches into one token")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dedup", action="store_true",
                    help="skip records whose image content (md5) was already seen")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    records = json.load(open(args.capture))
    if isinstance(records, dict):
        records = records.get("records", records.get("images", []))
    if args.limit:
        records = records[:args.limit]

    rows = []
    seen_md5 = set()
    n_skipped = 0
    for rec in records:
        # ---- locate the image -------------------------------------------------
        path = rec.get("image_path")
        name = rec.get("tag") or rec.get("image") or rec.get("name")
        if path and not os.path.exists(path) and args.image_dir:
            path = os.path.join(args.image_dir, os.path.basename(path))
        if not path and args.image_dir and name:
            cands = [f for f in os.listdir(args.image_dir) if f.startswith(str(name))]
            path = os.path.join(args.image_dir, cands[0]) if cands else None
        if not path or not os.path.exists(path):
            print("image not found for record:", name)
            continue

        if args.dedup:
            import hashlib
            md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            if md5 in seen_md5:
                print("skip duplicate image:", name)
                n_skipped += 1
                continue
            seen_md5.add(md5)

        # ---- merged patch grid ------------------------------------------------
        # capture stores grid as [[t, h, w]] in RAW patches; Qwen merges 2x2 -> tokens
        grid = rec.get("grid")
        if not grid:
            print("skipping record without grid:", name)
            continue
        g = grid[0] if isinstance(grid[0], (list, tuple)) else grid
        if len(g) == 3:
            raw_h, raw_w = int(g[1]), int(g[2])
        else:
            raw_h, raw_w = int(g[0]), int(g[1])
        grid_h, grid_w = raw_h // args.merge_size, raw_w // args.merge_size

        n_expected = rec.get("img_num")
        if n_expected and grid_h * grid_w != int(n_expected):
            print("  warning: grid {}x{}={} but img_num={}".format(
                grid_h, grid_w, grid_h * grid_w, n_expected))

        # ---- surviving token indices -----------------------------------------
        kept = rec.get(args.stage)
        if kept is None:
            kept = rec.get("final_kept") or rec.get("kept") or rec.get("kept_rel") or []

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
        n_tok = int(rec.get("img_num") or (grid_h * grid_w))
        rows.append({
            "image": os.path.basename(path),
            "category": rec.get("category"),
            "correct": rec.get("correct"),
            "question": rec.get("question"),
            "prediction": rec.get("prediction"),
            "global_kept_fraction": round(len(kept_set) / max(1, n_tok), 4),
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
    glob = [r["global_kept_fraction"] for r in rows if r.get("global_kept_fraction") is not None]
    print("\n================ SUMMARY ({}) ================".format(args.stage))
    print("images analysed              : {}{}".format(
        len(rows), "  (deduplicated, {} skipped)".format(n_skipped) if args.dedup else ""))
    print("images with any OCR words    : {}".format(sum(1 for r in rows if r["n_words"])))
    if glob:
        print("global tokens kept           : {:.3f}".format(sum(glob) / len(glob)))
    if covs:
        print("mean per-word coverage       : {:.3f}".format(sum(covs) / len(covs)))
    if reads:
        print("fraction of words readable   : {:.3f}  (threshold {})".format(
            sum(reads) / len(reads), args.threshold))

    # per-category breakdown, text-heavy vs charts vs diagrams
    cats = {}
    for r in rows:
        c = r.get("category") or "unknown"
        cats.setdefault(c, []).append(r)
    print("\nby category:")
    for c, rs in sorted(cats.items()):
        rr = [x["readable_fraction"] for x in rs if x["readable_fraction"] is not None]
        cc = [x["mean_word_coverage"] for x in rs if x["mean_word_coverage"] is not None]
        print("  {:12s} n={:2d}  mean_coverage={:.3f}  readable={:.3f}".format(
            c, len(rs),
            sum(cc) / len(cc) if cc else float('nan'),
            sum(rr) / len(rr) if rr else float('nan')))

    # ---- the evidence link: does coverage predict correctness? -------------
    scored = [r for r in rows if r.get("correct") is not None and r["mean_word_coverage"] is not None]
    if scored:
        ok = [r for r in scored if r["correct"]]
        bad = [r for r in scored if not r["correct"]]

        def m(rs, key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else float('nan')

        print("\n---------------- coverage vs correctness ----------------")
        print("answered CORRECTLY   n={:4d}  mean_coverage={:.3f}  readable={:.3f}".format(
            len(ok), m(ok, "mean_word_coverage"), m(ok, "readable_fraction")))
        print("answered INCORRECTLY n={:4d}  mean_coverage={:.3f}  readable={:.3f}".format(
            len(bad), m(bad, "mean_word_coverage"), m(bad, "readable_fraction")))
        gap = m(ok, "mean_word_coverage") - m(bad, "mean_word_coverage")
        print("coverage gap (correct - incorrect): {:+.3f}".format(gap))
        print("\nA clearly POSITIVE gap means surviving text coverage predicts whether the")
        print("model still answers correctly. That is the causal link E4 is meant to show.")
        print("A gap near zero means coverage does NOT explain the failure, and the")
        print("diversity hypothesis needs rethinking.")
    else:
        print("\n(no correctness labels in this capture; rerun with capture_tokens4.py")
        print(" to get question/prediction/correct fields)")

    print("\nIf readable-word fraction sits far BELOW the global kept fraction,")
    print("then removal rate was hiding the failure and coverage is the right metric.")


if __name__ == "__main__":
    main()

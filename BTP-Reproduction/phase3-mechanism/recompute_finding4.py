#!/usr/bin/env python3
"""
Recompute Phase 2's Finding 4 (text vs non-text removal rate) on DEDUPLICATED images.

The original figure, 74.8% of text patches removed against 90.2% of non-text, was computed
over 12 capture records, four of which were duplicate images. This rescoring drops the
duplicates so the averages are not weighted toward four repeated pictures.

Runs on CPU. No GPU, no model, only the saved captures plus OCR.

Usage
  python recompute_finding4.py --capture results/viz3/capture3.json --dedup
  python recompute_finding4.py --capture results/phase3/capture4.json          # new capture
"""

import argparse
import hashlib
import json
import os

from PIL import Image


def text_mask(image_path, grid_h, grid_w, reader, conf):
    """Boolean list over merged patches, True where the patch overlaps OCR text."""
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    mask = [False] * (grid_h * grid_w)
    n_boxes = 0
    for box, _text, score in reader.readtext(np.array(img)):
        if score is None or score < conf:
            continue
        n_boxes += 1
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        c0 = max(0, int(min(xs) / W * grid_w))
        c1 = min(grid_w - 1, int(max(xs) / W * grid_w))
        r0 = max(0, int(min(ys) / H * grid_h))
        r1 = min(grid_h - 1, int(max(ys) / H * grid_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mask[r * grid_w + c] = True
    return mask, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--stage", default="final_kept")
    ap.add_argument("--merge-size", type=int, default=2)
    ap.add_argument("--ocr-conf", type=float, default=0.3)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--out", default="finding4_recomputed.json")
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    records = json.load(open(args.capture))
    rows, seen = [], set()

    for rec in records:
        path = rec.get("image_path")
        tag = rec.get("tag")
        if not path or not os.path.exists(path):
            print("image missing:", tag)
            continue
        if args.dedup:
            md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            if md5 in seen:
                print("skip duplicate:", tag)
                continue
            seen.add(md5)

        g = rec["grid"][0] if isinstance(rec["grid"][0], (list, tuple)) else rec["grid"]
        raw_h, raw_w = (g[1], g[2]) if len(g) == 3 else (g[0], g[1])
        gh, gw = raw_h // args.merge_size, raw_w // args.merge_size
        n_tok = int(rec.get("img_num") or gh * gw)

        kept = set(int(i) for i in (rec.get(args.stage) or []))
        mask, n_boxes = text_mask(path, gh, gw, reader, args.ocr_conf)
        mask = (mask + [False] * n_tok)[:n_tok]

        text_idx = [i for i in range(n_tok) if mask[i]]
        non_idx = [i for i in range(n_tok) if not mask[i]]
        if not text_idx:
            print("{:26s} no OCR text found, excluded".format(tag))
            continue

        text_removed = sum(1 for i in text_idx if i not in kept) / len(text_idx)
        non_removed = sum(1 for i in non_idx if i not in kept) / max(1, len(non_idx))

        rows.append({
            "tag": tag,
            "category": rec.get("category"),
            "n_tokens": n_tok,
            "n_text_patches": len(text_idx),
            "ocr_boxes": n_boxes,
            "text_removed_pct": round(100 * text_removed, 1),
            "nontext_removed_pct": round(100 * non_removed, 1),
        })
        print("{:26s} text={:5.1f}%  non-text={:5.1f}%  (text patches {})".format(
            tag, 100 * text_removed, 100 * non_removed, len(text_idx)))

    json.dump(rows, open(args.out, "w"), indent=2)

    if not rows:
        print("\nno usable images")
        return

    def avg(k, rs=None):
        rs = rs if rs is not None else rows
        return sum(r[k] for r in rs) / len(rs)

    print("\n================ FINDING 4, RECOMPUTED ================")
    print("images used (with OCR text) : {}".format(len(rows)))
    print("TEXT patches removed        : {:.1f}%".format(avg("text_removed_pct")))
    print("NON-TEXT patches removed    : {:.1f}%".format(avg("nontext_removed_pct")))
    print("difference                  : {:+.1f} points".format(
        avg("text_removed_pct") - avg("nontext_removed_pct")))

    by = {}
    for r in rows:
        by.setdefault(r["category"] or "unknown", []).append(r)
    print("\nby category:")
    for c, rs in sorted(by.items()):
        print("  {:11s} n={:2d}  text={:5.1f}%  non-text={:5.1f}%".format(
            c, len(rs), avg("text_removed_pct", rs), avg("nontext_removed_pct", rs)))

    print("\nPhase 2 reported 74.8% text vs 90.2% non-text over 12 records,")
    print("four of which were duplicate images. Use the numbers above instead,")
    print("and state the true image count in the report and appendix.")


if __name__ == "__main__":
    main()

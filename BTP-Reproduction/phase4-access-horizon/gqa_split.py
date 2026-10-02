#!/usr/bin/env python3
"""
Split the GQA horizon by question type. No GPU: uses the saved per-sample results.

  python gqa_split.py --results ../../../horizon --types ../../../horizon/gqa_types.json \
      --tags qwen7b,qwen3b,ivl2b --names "Qwen2.5-VL-7B,Qwen2.5-VL-3B,InternVL2-2B" \
      --out ../../../horizon

QUESTION
  GQA's curve rises in two steps in every model: part of the image's contribution arrives
  with the perception tasks, the rest only at the reading-task cliff. Is that because GQA
  mixes question types with different horizons?

For each model and each grouping (semantic, structural), every group with enough samples
gets its own floor, baseline, g(l) curve and floor-normalised horizon h_0.5 with a 95%
bootstrap interval. Groups are reported only if they have at least MIN_N samples and the
image adds at least MIN_CONTRIB to their accuracy; otherwise the horizon is noise.

CAUTION  verify and logical questions are yes/no, so they carry the partial-access bias
flip seen on POPE (the model says "no" to everything at mid depth). Their horizons are
aggregate-level only and should be read with that in mind.

Writes gqa_split.txt and gqa_split_<grouping>.png.
"""

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_access_horizon import curve_from, sustained_horizon

MIN_N = 25
MIN_CONTRIB = 0.10
PAL = ["#1E2761", "#A3312F", "#2E7D6B", "#8A6D1F", "#6A4C93", "#5A5A5A"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})


def group_stats(L, pl, idx, B=1000, seed=0):
    acc = curve_from(pl, idx)
    floor, base = acc[0], acc[L]
    h = sustained_horizon(acc, L, 0.5)
    rng = random.Random(seed)
    hs = []
    for _ in range(B):
        b = [idx[rng.randrange(len(idx))] for _ in idx]
        x = sustained_horizon(curve_from(pl, b), L, 0.5)
        if x is not None:
            hs.append(x)
    hs.sort()
    ci = (hs[int(0.025 * len(hs))], hs[int(0.975 * len(hs)) - 1]) if len(hs) > 0.9 * B else (None, None)
    g = [(acc[l] - floor) / (base - floor) if base > floor else float("nan") for l in range(L + 1)]
    return {"n": len(idx), "floor": floor, "base": base, "contrib": base - floor,
            "h": h, "ci": ci, "g": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--types", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    types = json.load(open(a.types))
    tags, names = a.tags.split(","), a.names.split(",")
    lines = []

    runs = []
    for tag in tags:
        r = json.load(open(os.path.join(a.results, tag + "_gqa_n300.json")))
        runs.append((r["n_layers"], {int(k): v for k, v in r["per_layer"].items()}, r["sample_ids"]))
    if len({tuple(s) for _, _, s in runs}) == 1:
        lines.append("All models were measured on the same 300 GQA questions.")
    else:
        lines.append("WARNING: sample sets differ between models.")
    lines.append("groups shown only with n >= {} and image contribution >= {:.2f}".format(MIN_N, MIN_CONTRIB))
    lines.append("")

    for grouping in ("semantic", "structural"):
        labels = sorted({types[s][grouping] for s in runs[0][2] if s in types and types[s][grouping]})
        lines.append("=" * 78)
        lines.append("grouping: " + grouping)
        lines.append("=" * 78)
        head = "{:<10} {:>4}".format("group", "n")
        for nm in names:
            head += "  | {:^26}".format(nm)
        lines.append(head)
        lines.append("{:<10} {:>4}".format("", "") + "  | {:^26}".format("floor base  h0.5 [CI] depth") * len(names))

        fig, axes = plt.subplots(1, len(tags), figsize=(4.0 * len(tags), 3.4), sharey=True)
        if len(tags) == 1:
            axes = [axes]
        for gi, lab in enumerate(labels):
            row = None
            for k, (L, pl, ids) in enumerate(runs):
                idx = [i for i, s in enumerate(ids) if types.get(s, {}).get(grouping) == lab]
                if row is None:
                    row = "{:<10} {:>4}".format(lab, len(idx))
                if len(idx) < MIN_N:
                    row += "  | {:^26}".format("too few samples")
                    continue
                st = group_stats(L, pl, idx)
                if st["contrib"] < MIN_CONTRIB:
                    row += "  | {:^26}".format("image adds {:.2f}, skip".format(st["contrib"]))
                    continue
                ci = "[{}-{}]".format(*st["ci"]) if st["ci"][0] is not None else "[unst]"
                depth = "{:.0f}%".format(100 * st["h"] / L) if st["h"] is not None else "n/a"
                row += "  | {:.2f} {:.2f} {:>3} {:<7} {:>4}".format(st["floor"], st["base"],
                                                                  str(st["h"]), ci, depth)
                xs = [100 * l / L for l in range(L + 1)]
                axes[k].plot(xs, st["g"], "-", color=PAL[gi % len(PAL)], lw=1.6,
                             label="{} (n={})".format(lab, len(idx)))
            lines.append(row)
        for k, nm in enumerate(names):
            axes[k].axhline(0.5, color="#8A8A8A", ls="--", lw=0.8)
            axes[k].set_title(nm, fontsize=10)
            axes[k].set_xlabel("Relative depth with access (%)")
            axes[k].set_xlim(0, 100)
            axes[k].set_ylim(-0.4, 1.3)
        axes[0].set_ylabel("Share of image contribution recovered")
        axes[-1].legend(frameon=False, fontsize=7.5, loc="upper left")
        fig.suptitle("GQA by {} question type".format(grouping), fontsize=11)
        fig.tight_layout()
        p = os.path.join(a.out, "gqa_split_{}.png".format(grouping))
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        lines.append("")

    txt = "\n".join(lines)
    open(os.path.join(a.out, "gqa_split.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

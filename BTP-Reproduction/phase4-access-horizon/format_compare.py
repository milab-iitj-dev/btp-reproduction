#!/usr/bin/env python3
"""
Format control: does the horizon follow the answer format or the need to read?

  python format_compare.py --results ../../../horizon --tags qwen7b,ivl2b \
      --names "Qwen2.5-VL-7B,InternVL2-2B" --out ../../../horizon

For each model and each reading task (textvqa, chartqa), compares the open-ended run with
the multiple-choice run on the SAME questions and images. The difference in horizon,
delta = h_open - h_mc, gets a PAIRED bootstrap interval: each resample draws one set of
question indices and applies it to both formats, so question difficulty cancels.

Reference points on the same model: MMBench (the earliest closed-form task) and the
open-ended reading horizon itself.

Reading the result
  delta near 0, MC stays late          reading drives the horizon, format does not
  MC moves to MMBench's depth          format drives it
  in between                           both contribute; report the share of the gap
Share of the gap closed = (h_open - h_mc) / (h_open - h_mmbench).
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

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})
NAVY, CRIMS, GREY = "#1E2761", "#A3312F", "#8A8A8A"
N_OF = {"textvqa": 200, "chartqa": 200, "mmbench": 300}
PAIRS = [("textvqa", "textvqa_mc"), ("chartqa", "chartqa_mc"), ("textvqa", "textvqa_hard"),
         ("mmbench_fopen", "mmbench_fmc"), ("ai2d_fopen", "ai2d_fmc")]
TITLE = {"textvqa_mc": "TextVQA, MC", "chartqa_mc": "ChartQA, MC", "textvqa_hard": "TextVQA, MC same-image distractors",
         "mmbench_fmc": "MMBench (filtered), open vs MC", "ai2d_fmc": "AI2D (filtered), open vs MC"}


def load(results, tag, task):
    import glob as _g
    c = sorted(_g.glob(os.path.join(results, "{}_{}_n*.json".format(tag, task))))
    if not c:
        return None
    p = c[0]
    r = json.load(open(p))
    L = r["n_layers"]
    pl = {int(k): v for k, v in r["per_layer"].items()}
    if any(l not in pl for l in range(L + 1)):
        return None
    return L, pl, r["sample_ids"]


def g_curve(L, pl):
    acc = curve_from(pl)
    f, b = acc[0], acc[L]
    return [(acc[l] - f) / (b - f) for l in range(L + 1)], f, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    tags, names = a.tags.split(","), a.names.split(",")
    os.makedirs(a.out, exist_ok=True)
    import glob as _g
    tasks = [p for p in PAIRS if any(_g.glob(os.path.join(a.results, "{}_{}_n*.json".format(t, p[1]))) for t in tags)]
    lines = []
    fig, axes = plt.subplots(len(tags), len(tasks), figsize=(4.2 * len(tasks), 3.2 * len(tags)),
                             squeeze=False, sharey=True)

    for ti, (tag, nm) in enumerate(zip(tags, names)):
        mmb = load(a.results, tag, "mmbench")
        h_mmb = sustained_horizon(curve_from(mmb[1]), mmb[0], 0.5) if mmb else None
        for kj, (task, variant) in enumerate(tasks):
            op, mc = load(a.results, tag, task), load(a.results, tag, variant)
            ax = axes[ti][kj]
            if not op or not mc:
                lines.append("{} {}: missing or incomplete run".format(nm, variant))
                continue
            L = op[0]
            if op[2] != mc[2]:
                lines.append("{} {}: WARNING sample ids differ, pairing invalid".format(nm, task))
                continue
            h_op = sustained_horizon(curve_from(op[1]), L, 0.5)
            h_mc = sustained_horizon(curve_from(mc[1]), L, 0.5)
            n = len(op[2])
            rng = random.Random(0)
            ds = []
            for _ in range(a.B):
                idx = [rng.randrange(n) for _ in range(n)]
                x = sustained_horizon(curve_from(op[1], idx), L, 0.5)
                y = sustained_horizon(curve_from(mc[1], idx), L, 0.5)
                if x is not None and y is not None:
                    ds.append(x - y)
            ds.sort()
            lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]
            go, fo, bo = g_curve(L, op[1])
            gm, fm, bm = g_curve(L, mc[1])
            share = ((h_op - h_mc) / (h_op - h_mmb)) if (h_mmb is not None and h_op != h_mmb) else float("nan")
            lines.append("{:<14} {:<13} open: floor {:.2f} base {:.2f} h {:>2} ({:.0f}%) | "
                         "MC: floor {:.2f} base {:.2f} h {:>2} ({:.0f}%) | delta {:+d} [{:+d}, {:+d}] | "
                         "MMBench h {} | share of gap closed {:.0%}".format(
                             nm, variant, fo, bo, h_op, 100 * h_op / L, fm, bm, h_mc, 100 * h_mc / L,
                             h_op - h_mc, lo, hi, h_mmb, share))
            hid = os.path.join(a.results, "textvqa_hard_ids.json")
            if variant == "textvqa_hard" and os.path.exists(hid):
                # Robustness: only questions whose three wrong options all came from the image.
                hard = json.load(open(hid))
                sub = [i for i, sid in enumerate(op[2]) if hard.get(sid) == 3]
                so = sustained_horizon(curve_from(op[1], sub), L, 0.5)
                sm = sustained_horizon(curve_from(mc[1], sub), L, 0.5)
                rs = random.Random(1)
                dd = []
                for _ in range(a.B):
                    ix = [sub[rs.randrange(len(sub))] for _ in sub]
                    x = sustained_horizon(curve_from(op[1], ix), L, 0.5)
                    y = sustained_horizon(curve_from(mc[1], ix), L, 0.5)
                    if x is not None and y is not None:
                        dd.append(x - y)
                dd.sort()
                am = curve_from(mc[1], sub)
                lines.append("{:<14} {:<13} fully-hard subset n={}: open h {} | MC h {} ({:.0f}%), "
                             "floor {:.2f} base {:.2f} | delta {:+d} [{:+d}, {:+d}]".format(
                                 nm, variant, len(sub), so, sm, 100 * sm / L, am[0], am[L],
                                 so - sm, dd[int(0.025 * len(dd))], dd[int(0.975 * len(dd)) - 1]))
            xs = list(range(L + 1))
            ax.plot(xs, go, "-o", color=CRIMS, ms=2.5, lw=1.6, label="open-ended")
            ax.plot(xs, gm, "-s", color=NAVY, ms=2.5, lw=1.6, label="multiple choice")
            if h_mmb is not None:
                ax.axvline(h_mmb, color=GREY, ls=":", lw=1)
                ax.text(h_mmb + 0.3, -0.3, "MMBench h", fontsize=7, color=GREY)
            ax.axhline(0.5, color=GREY, ls="--", lw=0.8)
            ax.set_title("{}: {}".format(nm, TITLE[variant]), fontsize=9)
            ax.set_xlim(0, L)
            ax.set_ylim(-0.4, 1.25)
            ax.set_xlabel("Layers with access")
            if kj == 0:
                ax.set_ylabel("Share of image contribution")
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Format control: same questions, open-ended vs multiple choice", fontsize=11)
    fig.tight_layout()
    p = os.path.join(a.out, "format_control.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "format_control.txt"), "w").write(txt + "\n")
    print(txt)
    print("wrote", p)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Is the task ordering of visual access horizons reproducible across models?

  python compare_models.py --results ../../../horizon --tags qwen7b,qwen3b \
      --names "Qwen2.5-VL-7B,Qwen2.5-VL-3B" --out ../../../horizon

For each pair of models, over the tasks both have:
  - Spearman rank correlation of the floor-normalised horizon h_alpha (alpha 0.5 and 0.9)
  - a 95% bootstrap interval, resampling samples within every task of both models, so the
    interval reflects the measurement noise in each horizon, not just in the ranking
  - an exact permutation p-value for the observed correlation

Writes <tag1>_vs_<tag2>_relative.png, horizons on relative depth side by side, and
<tag1>_vs_<tag2>_rank.txt.
"""

import argparse
import glob
import itertools
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_access_horizon import curve_from, sustained_horizon

LABEL = {"textvqa": "TextVQA", "chartqa": "ChartQA", "docvqa": "DocVQA", "pope": "POPE",
         "gqa": "GQA", "mmbench": "MMBench", "scienceqa": "ScienceQA", "ai2d": "AI2D"}
READING = {"textvqa", "chartqa", "docvqa"}
COL = ["#1E2761", "#A3312F", "#2E7D6B", "#8A6D1F"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})


def load(results, tag):
    d = {}
    for f in glob.glob(os.path.join(results, tag + "_*.json")):
        r = json.load(open(f))
        L = r["n_layers"]
        pl = {int(k): v for k, v in r["per_layer"].items()}
        if all(l in pl for l in range(L + 1)):
            d[r["task"]] = (L, pl)
    return d


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def horizon(L, pl, alpha, idx=None):
    return sustained_horizon(curve_from(pl, idx), L, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    tags, names = a.tags.split(","), a.names.split(",")
    data = [load(a.results, t) for t in tags]
    tasks = sorted(set.intersection(*[set(d) for d in data]))
    rng = random.Random(0)
    lines = ["tasks: " + ", ".join(tasks), ""]

    rel = {}
    for (i, j) in itertools.combinations(range(len(tags)), 2):
        for alpha in (0.5, 0.9):
            hi = [horizon(*data[i][t], alpha) / data[i][t][0] for t in tasks]
            hj = [horizon(*data[j][t], alpha) / data[j][t][0] for t in tasks]
            rho = spearman(hi, hj)
            boot = []
            for _ in range(a.B):
                bi, bj = [], []
                for t in tasks:
                    for d, out in ((data[i][t], bi), (data[j][t], bj)):
                        L, pl = d
                        n = len(pl[0])
                        idx = [rng.randrange(n) for _ in range(n)]
                        h = horizon(L, pl, alpha, idx)
                        out.append(h / L if h is not None else 1.0)
                boot.append(spearman(bi, bj))
            boot = sorted(x for x in boot if not math.isnan(x))
            lo, up = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
            # exact permutation test over task labels
            perms = list(itertools.permutations(hj)) if len(tasks) <= 9 else None
            if perms:
                ge = sum(1 for p in perms if spearman(hi, list(p)) >= rho - 1e-12)
                pval = ge / len(perms)
            else:
                pval = float("nan")
            lines.append("{} vs {}, alpha {}: Spearman rho = {:.3f}  95% bootstrap CI [{:.3f}, {:.3f}]"
                         "  one-sided permutation p = {:.4f}".format(names[i], names[j], alpha, rho, lo, up, pval))
            for t, x, y in zip(tasks, hi, hj):
                lines.append("   {:<10} {:5.1f}%  {:5.1f}%".format(t, 100 * x, 100 * y))
            lines.append("")
            if alpha == 0.5:
                rel[(i, j)] = (hi, hj)

    # ---- figure: h0.5 and h0.9 on relative depth, models side by side
    order = sorted(tasks, key=lambda t: horizon(*data[0][t], 0.5) / data[0][t][0])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ys = list(range(len(order)))[::-1]
    off = [0.15 - 0.3 * k / max(1, len(tags) - 1) for k in range(len(tags))]
    for k, (d, nm) in enumerate(zip(data, names)):
        for y, t in zip(ys, order):
            L, pl = d[t]
            h5, h9 = horizon(L, pl, 0.5), horizon(L, pl, 0.9)
            x5, x9 = 100 * h5 / L, 100 * h9 / L
            ax.plot([x5, x9], [y + off[k]] * 2, "-", color=COL[k], lw=1.2, alpha=0.6)
            ax.plot([x5], [y + off[k]], "o", color=COL[k], ms=6, label=nm if y == ys[0] else None)
            ax.plot([x9], [y + off[k]], "D", color=COL[k], mfc="white", ms=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[t] for t in order])
    for lab in ax.get_yticklabels():
        if lab.get_text() in {LABEL[t] for t in READING}:
            lab.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative depth with access to the visual tokens (%)")
    ax.plot([], [], "o", color="#8A8A8A", label="50% of contribution")
    ax.plot([], [], "D", color="#8A8A8A", mfc="white", label="90%")
    ax.set_title("Task ordering of visual access horizons across models", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    base = "_vs_".join(tags)
    p = os.path.join(a.out, base + "_relative.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    with open(os.path.join(a.out, base + "_rank.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", p)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figures and summary table for the Phase 4 access-horizon results.

  python make_horizon_figure.py --results ../../../horizon --tag qwen7b --out ../../../horizon

Writes
  <tag>_curves.png     g(l) for every task on one axis, reading tasks highlighted
  <tag>_horizons.png   h_0.5 and h_0.9 per task with 95% bootstrap intervals
  <tag>_summary.csv    one row per task

g(l) = (acc(l) - floor) / (baseline - floor), the share of the image's contribution
recovered when l layers had access. Definitions as in measure_access_horizon.py.
"""

import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_access_horizon import (curve_from, sustained_horizon, bootstrap_ci,
                                    per_sample_horizons)

READING = {"textvqa", "chartqa", "docvqa"}
LABEL = {"textvqa": "TextVQA", "chartqa": "ChartQA", "docvqa": "DocVQA", "pope": "POPE",
         "gqa": "GQA", "mmbench": "MMBench", "scienceqa": "ScienceQA", "ai2d": "AI2D"}
NAVY, CRIMS, GREY = "#1E2761", "#A3312F", "#8A8A8A"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})


def load(results, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(results, tag + "_*.json"))):
        r = json.load(open(f))
        L = r["n_layers"]
        pl = {int(k): v for k, v in r["per_layer"].items()}
        if any(l not in pl for l in range(L + 1)):
            print("skipping incomplete", f)
            continue
        acc = curve_from(pl)
        floor, base = acc[0], acc[L]
        row = {"task": r["task"], "L": L, "n": len(r["sample_ids"]), "floor": floor,
               "baseline": base, "contribution": base - floor,
               "g": [(acc[l] - floor) / (base - floor) for l in range(L + 1)]}
        for a in (0.5, 0.9):
            h = sustained_horizon(acc, L, a)
            lo, hi, _ = bootstrap_ci(pl, L, a)
            row["h%s" % a], row["ci%s" % a] = h, (lo, hi)
        ps, prior = per_sample_horizons(pl, L)
        ps = sorted(h for h in ps if h is not None)
        row["ps_n"], row["ps_prior"] = len(ps), prior
        row["ps_med"] = ps[len(ps) // 2] if ps else None
        row["ps_q25"] = ps[len(ps) // 4] if ps else None
        row["ps_q75"] = ps[(3 * len(ps)) // 4] if ps else None
        out.append(row)
    return sorted(out, key=lambda r: (r["h0.5"] if r["h0.5"] is not None else 99, r["task"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", default="qwen7b")
    ap.add_argument("--model-name", default="Qwen2.5-VL-7B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.results, a.tag)
    if not rows:
        raise SystemExit("no complete results for tag " + a.tag)
    L = rows[0]["L"]

    # ---- curves
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    other = [r for r in rows if r["task"] not in READING]
    greys = plt.cm.Blues([0.45 + 0.5 * i / max(1, len(other) - 1) for i in range(len(other))])
    ci = 0
    for r in rows:
        xs = list(range(L + 1))
        if r["task"] in READING:
            ax.plot(xs, r["g"], "-", color=CRIMS, lw=1.8, alpha=0.9,
                    marker={"textvqa": "o", "chartqa": "s", "docvqa": "^"}[r["task"]], ms=3.2,
                    label=LABEL[r["task"]])
        else:
            ax.plot(xs, r["g"], "-", color=greys[ci], lw=1.8, marker="o", ms=2.8,
                    label=LABEL[r["task"]])
            ci += 1
    ax.axhline(0.5, color=GREY, ls="--", lw=0.9)
    ax.axhline(0.9, color=GREY, ls=":", lw=0.9)
    ax.text(L - 0.3, 0.52, "50% of the image's contribution", fontsize=8, color=GREY, ha="right")
    ax.text(L - 0.3, 0.84, "90%", fontsize=8, color=GREY, ha="right")
    ax.set_xlim(0, L)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xlabel("Layers with access to the visual tokens (all removed after this point)")
    ax.set_ylabel("Share of the image's contribution recovered, g(l)")
    ax.set_title("{}: visual access horizon by task".format(a.model_name), fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    p1 = os.path.join(a.out, a.tag + "_curves.png")
    fig.savefig(p1, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- horizons with intervals
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        col = CRIMS if r["task"] in READING else NAVY
        for key, mk, off in (("0.5", "o", 0.12), ("0.9", "D", -0.12)):
            h, (lo, hi) = r["h" + key], r["ci" + key]
            if h is None:
                continue
            if lo is not None:
                ax.plot([lo, hi], [y + off] * 2, "-", color=col, lw=1.4)
            ax.plot([h], [y + off], mk, color=col, ms=6 if mk == "o" else 5,
                    mfc=col if key == "0.5" else "white")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[r["task"]] for r in rows])
    ax.set_xlim(0, L)
    ax.set_xlabel("Layers with access (95% bootstrap interval)")
    ax.plot([], [], "o", color=GREY, label="50% of contribution recovered")
    ax.plot([], [], "D", color=GREY, mfc="white", label="90%")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("{}: horizon per task".format(a.model_name), fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(a.out, a.tag + "_horizons.png")
    fig.savefig(p2, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- table
    p3 = os.path.join(a.out, a.tag + "_summary.csv")
    with open(p3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n", "layers", "floor", "baseline", "image_contribution",
                    "h50", "h50_ci", "h50_depth_pct", "h90", "h90_ci",
                    "per_sample_n", "excluded_correct_without_image",
                    "per_sample_q25", "per_sample_median", "per_sample_q75"])
        for r in rows:
            w.writerow([r["task"], r["n"], r["L"], "%.4f" % r["floor"], "%.4f" % r["baseline"],
                        "%.4f" % r["contribution"], r["h0.5"], "%s-%s" % r["ci0.5"],
                        "%.1f" % (100.0 * r["h0.5"] / L) if r["h0.5"] is not None else "",
                        r["h0.9"], "%s-%s" % r["ci0.9"], r["ps_n"], r["ps_prior"],
                        r["ps_q25"], r["ps_med"], r["ps_q75"]])
    for p in (p1, p2, p3):
        print("wrote", p)


if __name__ == "__main__":
    main()

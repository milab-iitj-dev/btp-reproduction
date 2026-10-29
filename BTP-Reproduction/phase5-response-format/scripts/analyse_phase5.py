#!/usr/bin/env python3
"""
Analyse the Phase 5 runs: one table and one figure per experiment, plus a summary.

  python analyse_phase5.py --results ../results/raw --out ../results/summary --figs ../figures

Two depth measures per condition, both "sustained" (the condition must hold at every later
layer too, so one noisy point cannot set them):
  safe withdrawal layer   first l with accuracy >= 95% of the full-image accuracy
  h50                     first l keeping >= 50% of what the image adds over the floor
                          (the Phase 4 horizon)
Intervals are paired bootstrap over questions: one resample of question indices is applied to
both conditions, so question difficulty cancels in the difference.
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "phase4-access-horizon", "scripts"))
from measure_access_horizon import curve_from, sustained_horizon  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})
NAVY, CRIMS, TEAL, GREY, GOLD = "#1E2761", "#A3312F", "#1D6B6B", "#8A8A8A", "#B07D1A"
NAME = {"open": "open-ended", "fill": "fill-in-the-blank", "mc2": "2 options", "mc4": "4 options",
        "mc8": "8 options", "mc4_type": "4 options, same-type distractors", "yesno": "yes / no"}
COLOR = {"open": CRIMS, "fill": GOLD, "mc8": TEAL, "mc4": NAVY, "mc2": GREY,
         "mc4_type": "#5B4B8A", "yesno": "#3F7F3F"}


def load(d, cond):
    p = os.path.join(d, "qwen7b_textvqa_{}.json".format(cond))
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    L = r["n_layers"]
    if any(str(l) not in r["per_layer"] for l in range(L + 1)):
        print("incomplete:", cond)
        return None
    return r


def depth(r, idx=None):
    L = r["n_layers"]
    acc = curve_from({int(k): v for k, v in r["per_layer"].items()}, idx)
    return (sustained_horizon(acc, L, 0.95, normalise=False),
            sustained_horizon(acc, L, 0.5, normalise=True), acc)


def paired(ra, rb, B=2000, seed=0):
    """Bootstrap of depth(a) - depth(b) on the same questions."""
    n = len(ra["sample_ids"])
    rng = random.Random(seed)
    ds, dh = [], []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sa, ha, _ = depth(ra, idx)
        sb, hb, _ = depth(rb, idx)
        if sa is not None and sb is not None:
            ds.append(sa - sb)
        if ha is not None and hb is not None:
            dh.append(ha - hb)
    q = lambda x: (sorted(x)[int(.025 * len(x))], sorted(x)[int(.975 * len(x)) - 1]) if x else (None, None)
    return q(ds), q(dh)


def ci(r, B=2000, seed=1):
    n = len(r["sample_ids"])
    rng = random.Random(seed)
    s_, h_ = [], []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        s, h, _ = depth(r, idx)
        if s is not None:
            s_.append(s)
        if h is not None:
            h_.append(h)
    q = lambda x: (sorted(x)[int(.025 * len(x))], sorted(x)[int(.975 * len(x)) - 1]) if x else (None, None)
    return q(s_), q(h_)


def stable_layer(seq, final):
    """First index from which seq equals final at every later position."""
    s = None
    for l in range(len(seq) - 1, -1, -1):
        if seq[l] == final:
            s = l
        else:
            break
    return s


def per_sample_horizon(r, i):
    L = r["n_layers"]
    col = [r["per_layer"][str(l)][i] for l in range(L + 1)]
    if not col[L] or col[0]:
        return None          # wrong at baseline, or right without the image
    return stable_layer(col, 1)


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[o[k]] = (i + j) / 2
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else float("nan")


def _normtext(x):
    import re as _re
    t = _re.sub(r"[^a-z0-9 ]", " ", str(x or "").lower())
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def norm_curve(c):
    L = len(c) - 1
    d = c[L] - c[0]
    return [(x - c[0]) / d if d else float("nan") for x in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.figs, exist_ok=True)
    R = {c: load(a.results, c) for c in NAME}
    R = {c: r for c, r in R.items() if r}
    ids = {tuple(r["sample_ids"]) for r in R.values()}
    assert len(ids) == 1, "conditions were run on different question sets"
    L = next(iter(R.values()))["n_layers"]
    n = len(next(iter(R.values()))["sample_ids"])
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("Phase 5, Qwen2.5-VL-7B, TextVQA, {} questions (same questions in every condition), "
        "{} decoder layers".format(n, L))
    say("safe = first layer keeping >= 95% of full-image accuracy for good; "
        "h50 = Phase 4 horizon. Brackets: 95% bootstrap intervals.\n")

    rows = {}
    for c, r in R.items():
        s, h, acc = depth(r)
        (slo, shi), (hlo, hhi) = ci(r, a.B)
        noimg = mean(r["noimage"]["correct"]) if "noimage" in r else float("nan")
        rows[c] = dict(safe=s, h50=h, acc=acc, floor=acc[0], base=acc[L], noimg=noimg,
                       sci=(slo, shi), hci=(hlo, hhi))

    def table(conds, title):
        say(title)
        say("  {:<34} {:>7} {:>9} {:>7} {:>14} {:>12}".format(
            "condition", "full", "no image", "floor", "safe layer", "h50"))
        for c in conds:
            if c not in rows:
                say("  {:<34} not run".format(NAME[c]))
                continue
            x = rows[c]
            say("  {:<34} {:>7.3f} {:>9.3f} {:>7.3f} {:>4} [{:>2},{:>2}] {:>4} [{:>2},{:>2}]".format(
                NAME[c], x["base"], x["noimg"], x["floor"], x["safe"], *x["sci"], x["h50"], *x["hci"]))

    def diffs(pairs):
        for c1, c2 in pairs:
            if c1 in R and c2 in R:
                (s0, s1), (h0, h1) = paired(R[c1], R[c2], a.B)
                say("  {} minus {}: safe {:+d} [{}, {}]   h50 {:+d} [{}, {}]".format(
                    NAME[c1], NAME[c2], rows[c1]["safe"] - rows[c2]["safe"], s0, s1,
                    rows[c1]["h50"] - rows[c2]["h50"], h0, h1))

    def curves(conds, fname, title):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for c in conds:
            if c not in rows:
                continue
            acc = [rows[c]["acc"][l] for l in range(L + 1)]
            ax[0].plot(range(L + 1), acc, "-o", ms=2.4, lw=1.5, color=COLOR[c], label=NAME[c])
            ax[1].plot(range(L + 1), norm_curve(acc), "-o", ms=2.4, lw=1.5, color=COLOR[c])
            ax[0].axvline(rows[c]["safe"], color=COLOR[c], ls=":", lw=1)
        ax[0].set_ylabel("accuracy")
        ax[1].set_ylabel("share of what the image adds")
        ax[1].axhline(.5, color=GREY, ls="--", lw=.8)
        for x in ax:
            x.set_xlabel("layers that kept the image")
            x.set_xlim(0, L)
        ax[0].legend(frameon=False, fontsize=8)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.figs, fname), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ---- Experiment 1
    table(["open", "mc8", "mc4", "mc2"], "EXPERIMENT 1: number of choices")
    diffs([("open", "mc8"), ("mc8", "mc4"), ("mc4", "mc2"), ("open", "mc2")])
    curves(["open", "mc8", "mc4", "mc2"], "exp1_number_of_choices.png",
           "Experiment 1: fewer options, earlier withdrawal?")
    say()

    # ---- Experiment 2
    table(["mc4", "mc4_type"], "EXPERIMENT 2: same-type distractors and the no-image control")
    if "mc4_type" in R:
        tiers = [m.get("type_tiers") or [] for m in R["mc4_type"]["meta"]]
        say("  same-category distractors for all 3 options: {} of {} questions".format(
            sum(1 for t in tiers if t == [0, 0, 0]), len(tiers)))
        say("  chance for 4 options: 0.250")
    diffs([("open", "mc4_type"), ("mc4_type", "mc4")])
    curves(["open", "mc4_type", "mc4"], "exp2_same_type_distractors.png",
           "Experiment 2: do the options give the answer away?")
    say()

    # ---- Experiment 3
    table(["yesno", "mc4", "fill", "open"], "EXPERIMENT 3: four response modes")
    diffs([("open", "fill"), ("fill", "mc4"), ("mc4", "yesno"), ("open", "yesno")])
    curves(["yesno", "mc4", "fill", "open"], "exp3_response_modes.png",
           "Experiment 3: same visual fact, four ways of answering")
    say()

    # ---- Experiment 4
    say("EXPERIMENT 4: when does the decision become stable?")
    say("  Image-dependent questions only: right with the image, wrong when it is withdrawn at")
    say("  layer 0. (Questions the options already give away have a trivial decision layer of 0.)")
    say("  decision = median first withdrawal layer from which the model's choice no longer changes")
    say("  (closed: top option; open/fill: the answer text). p90 = 90th percentile of the same.")
    say("  conf. 90% = first layer from which mean confidence in the correct answer keeps 90% of its")
    say("  full-image gain. lens = median first layer at which the correct answer is the unmodified")
    say("  model's internal top choice and stays so (logit lens).")
    say("  {:<34} {:>5} {:>6} {:>9} {:>5} {:>10} {:>6}".format(
        "condition", "n", "safe", "decision", "p90", "conf. 90%", "lens"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    hist_data = {}
    for c in ["yesno", "mc2", "mc4", "mc8", "mc4_type", "fill", "open"]:
        if c not in R:
            continue
        r = R[c]
        closed = c not in ("open", "fill")
        dep = [i for i in range(n) if r["per_layer"][str(L)][i] and not r["per_layer"]["0"][i]]
        dec = []
        for i in dep:
            if closed:
                seq = [r["conf_top"][str(l)][i] for l in range(L + 1)]
            else:
                seq = [_normtext(r["predictions"][str(l)][i]) for l in range(L + 1)]
            dec.append(stable_layer(seq, seq[L]))
        hist_data[c] = [per_sample_horizon(r, i) for i in dep]
        dsort = sorted(x for x in dec if x is not None)
        p90 = dsort[int(.9 * (len(dsort) - 1))] if dsort else None
        cp = [mean(r["conf_p"][str(l)]) for l in range(L + 1)]
        g = norm_curve(cp)
        c90 = None
        for l in range(L, -1, -1):
            if g[l] >= .9:
                c90 = l
            else:
                break
        lens_layer = []
        lens = r.get("lens", [])
        for i in dep:
            if i >= len(lens):
                continue
            s = stable_layer([row[1] for row in lens[i]], 1)
            lens_layer.append(None if s is None else s + 1)     # layers counted from 1
        say("  {:<34} {:>5} {:>6} {:>9} {:>5} {:>10} {:>6}".format(
            NAME[c], len(dep), rows[c]["safe"], median(dec), p90, c90, median(lens_layer)))
        ax[0].plot(range(L + 1), g, "-o", ms=2.2, lw=1.4, color=COLOR[c], label=NAME[c])
        if r.get("lens"):
            lp = [mean([row[l][0] for row in r["lens"]]) for l in range(L)]
            ax[1].plot(range(1, L + 1), lp, "-", lw=1.5, color=COLOR[c], label=NAME[c])
    say("  Per-question layer from which each image-dependent question stays correct:")
    for c, v in hist_data.items():
        v = [x for x in v if x is not None]
        say("  {:<34} beyond 16: {:>3} of {:>3}   at 22 or later: {:>3}".format(
            NAME[c], sum(1 for x in v if x > 16), len(v), sum(1 for x in v if x >= 22)))
    sel = [c for c in ("mc4_type", "yesno", "fill", "open") if c in hist_data]
    for k, c in enumerate(sel):
        v = [x for x in hist_data[c] if x is not None]
        ax[2].hist(v, bins=range(0, L + 2), alpha=.55, color=COLOR[c], label=NAME[c])
    ax[2].set_xlabel("layer from which the question stays correct")
    ax[2].set_ylabel("image-dependent questions")
    ax[2].legend(frameon=False, fontsize=7)
    ax[0].set_xlabel("layers that kept the image")
    ax[0].set_ylabel("confidence in the correct answer\n(share of full-image confidence)")
    ax[0].axhline(.9, color=GREY, ls="--", lw=.8)
    ax[1].set_xlabel("decoder layer (full image, logit lens)")
    ax[1].set_ylabel("probability of the correct answer")
    ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Experiment 4: confidence under withdrawal, and inside the unmodified model", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figs, "exp4_decision_stability.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    open(os.path.join(a.out, "phase5_summary.txt"), "w").write("\n".join(lines) + "\n")
    json.dump({c: {k: v for k, v in x.items() if k != "acc"} for c, x in rows.items()},
              open(os.path.join(a.out, "phase5_depths.json"), "w"), indent=1)
    print("wrote", a.out, a.figs)


if __name__ == "__main__":
    main()

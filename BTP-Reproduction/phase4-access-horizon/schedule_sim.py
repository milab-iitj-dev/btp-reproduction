#!/usr/bin/env python3
"""
Step 2 of the access-horizon study: can a pruning schedule use the answer format to stop
carrying visual tokens once they are no longer needed?

  python schedule_sim.py --results ../../../horizon --out ../../../horizon

No GPU needed. Every horizon run already stores, for every sample, whether the answer was
correct when all visual tokens were deleted after layer c, for every c from 0 to L. So the
accuracy of any "delete after layer c" schedule is read off exactly, not estimated.

A schedule assigns each sample a cut c (layers that keep image access; c = L is the
unmodified model). Policies compared, all calibrated to the same rule:
    every task in a group must keep >= tau of its full-model accuracy at the cut and at
    every later layer (the sustained rule used for horizons)

  full            c = L for everything
  released BTP    c = 22 on Qwen2.5-VL-7B (the released end_layer); not calibrated
  uniform         one cut for all samples, safe for every task (format and task unknown)
  format-aware    one cut for open-ended samples, one for closed-form samples
                  (multiple choice, yes/no, pick-one); format is read from the prompt
  task oracle     one cut per task (needs the task identity; upper bound for task-level)

Cuts are chosen on a random half of the samples and scored on the other half, repeated over
many splits. Cost is reported two ways:
  visual layers saved  mean of 1 - c/L, the share of decoder layers that no longer carry
                       visual tokens
  measured time        run time at the cut / run time of the full model, from the
                       seconds_per_layer each job logged (end-to-end, same GPU type)

Held-out check: the format cuts learned on the 8 natural tasks are applied, unchanged, to
the format-control sets (TextVQA and ChartQA as multiple choice, MMBench and AI2D
open-ended), which the calibration never saw.
"""

import argparse
import glob
import json
import os
import random

NATURAL = ["mmbench", "pope", "scienceqa", "ai2d", "gqa", "chartqa", "docvqa", "textvqa"]
CLOSED_TASKS = {"mmbench", "pope", "scienceqa", "ai2d"}
OPEN_TASKS = {"chartqa", "docvqa", "textvqa"}
GQA_CLOSED = {"verify", "logical", "choose", "compare"}
HELD_OUT = {"textvqa_mc": "closed", "chartqa_mc": "closed", "textvqa_hard": "closed",
            "mmbench_fmc": "closed", "ai2d_fmc": "closed",
            "mmbench_fopen": "open", "ai2d_fopen": "open"}
RELEASED_CUT = {"qwen7b": 22}


def load(results, tag, task):
    c = sorted(glob.glob(os.path.join(results, "{}_{}_n*.json".format(tag, task))))
    if not c:
        return None
    r = json.load(open(c[0]))
    L = r["n_layers"]
    pl = {int(k): v for k, v in r["per_layer"].items()}
    sec = {int(k): v for k, v in r.get("seconds_per_layer", {}).items()}
    if any(l not in pl for l in range(L + 1)):
        return None
    return {"L": L, "pl": pl, "ids": r["sample_ids"], "sec": sec}


def acc(run, idx, c):
    v = run["pl"][c]
    return sum(v[i] for i in idx) / len(idx) if idx else float("nan")


def safe_cut(units, L, tau):
    """Smallest c such that every (run, idx) unit keeps >= tau of its full accuracy at every
    layer from c to L. units: list of (run, idx)."""
    best = L
    for c in range(L, -1, -1):
        ok = True
        for run, idx in units:
            full = acc(run, idx, L)
            if full <= 0:
                continue
            if acc(run, idx, c) < tau * full:
                ok = False
                break
        if not ok:
            break
        best = c
    return best


def split(n, rng):
    p = list(range(n))
    rng.shuffle(p)
    return sorted(p[: n // 2]), sorted(p[n // 2:])


def time_ratio(run, c):
    s = run["sec"]
    if c in s and run["L"] in s and s[run["L"]] > 0:
        return s[c] / s[run["L"]]
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default="qwen7b,qwen3b,ivl2b")
    ap.add_argument("--names", default="Qwen2.5-VL-7B,Qwen2.5-VL-3B,InternVL2-2B")
    ap.add_argument("--taus", default="0.95,0.98")
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    gqa_types = json.load(open(os.path.join(a.results, "gqa_types.json")))
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    for tag, nm in zip(a.tags.split(","), a.names.split(",")):
        runs = {t: load(a.results, tag, t) for t in NATURAL}
        runs = {t: r for t, r in runs.items() if r}
        L = next(iter(runs.values()))["L"]

        # Each sample's format. GQA is split per question by its structural type.
        fmt = {}
        for t, r in runs.items():
            if t == "gqa":
                fmt[t] = ["closed" if gqa_types.get(s, {}).get("structural") in GQA_CLOSED else "open"
                          for s in r["ids"]]
            else:
                fmt[t] = ["closed" if t in CLOSED_TASKS else "open"] * len(r["ids"])

        for tau in [float(x) for x in a.taus.split(",")]:
            say("=" * 96)
            say("{}   L = {}   rule: every task keeps >= {:.0%} of full accuracy   {} splits".format(
                nm, L, tau, a.splits))
            rng = random.Random(0)
            pols = ["full", "uniform", "format-aware", "task oracle"]
            if tag in RELEASED_CUT:
                pols.insert(1, "released BTP")
            ret = {p: {t: [] for t in runs} for p in pols}
            saved = {p: [] for p in pols}
            tim = {p: [] for p in pols}
            cuts = {p: [] for p in pols}
            for _ in range(a.splits):
                cal, test = {}, {}
                for t, r in runs.items():
                    cal[t], test[t] = split(len(r["ids"]), rng)
                # Calibration units, by policy.
                c_uni = safe_cut([(runs[t], cal[t]) for t in runs], L, tau)
                c_fmt = {}
                for f in ("open", "closed"):
                    units = [(runs[t], [i for i in cal[t] if fmt[t][i] == f]) for t in runs]
                    units = [(r, i) for r, i in units if len(i) >= 10]
                    c_fmt[f] = safe_cut(units, L, tau)
                c_task = {t: safe_cut([(runs[t], cal[t])], L, tau) for t in runs}

                def cut_for(p, t, i):
                    if p == "full":
                        return L
                    if p == "released BTP":
                        return RELEASED_CUT[tag]
                    if p == "uniform":
                        return c_uni
                    if p == "format-aware":
                        return c_fmt[fmt[t][i]]
                    return c_task[t]

                for p in pols:
                    n_all, sv, tm = 0, 0.0, 0.0
                    for t, r in runs.items():
                        idx = test[t]
                        full = acc(r, idx, L)
                        got = sum(r["pl"][cut_for(p, t, i)][i] for i in idx) / len(idx)
                        ret[p][t].append(got / full if full > 0 else float("nan"))
                        for i in idx:
                            c = cut_for(p, t, i)
                            sv += 1 - c / L
                            tm += time_ratio(r, c)
                            n_all += 1
                    saved[p].append(sv / n_all)
                    tim[p].append(tm / n_all)
                cuts["uniform"].append(c_uni)
                cuts["format-aware"].append((c_fmt["open"], c_fmt["closed"]))
                cuts["task oracle"].append(tuple(c_task[t] for t in runs))

            def mean(x):
                x = [v for v in x if v == v]
                return sum(x) / len(x) if x else float("nan")

            def q05(x):
                x = sorted(v for v in x if v == v)
                return x[int(0.05 * len(x))] if x else float("nan")

            say("{:<14} {:>10} {:>10} {:>12} {:>14}   {}".format(
                "policy", "vis saved", "time", "worst task", "worst (5% q.)", "typical cut"))
            for p in pols:
                worst = [min(ret[p][t][k] for t in runs) for k in range(a.splits)]
                if p == "uniform":
                    cs = "c = {:.1f}".format(mean(cuts[p]))
                elif p == "format-aware":
                    cs = "open {:.1f}, closed {:.1f}".format(mean([c[0] for c in cuts[p]]),
                                                             mean([c[1] for c in cuts[p]]))
                elif p == "task oracle":
                    cs = ", ".join("{} {:.0f}".format(t, mean([c[j] for c in cuts[p]]))
                                   for j, t in enumerate(runs))
                elif p == "released BTP":
                    cs = "c = {}".format(RELEASED_CUT[tag])
                else:
                    cs = "c = {}".format(L)
                say("{:<14} {:>9.1%} {:>9.0%} {:>12.1%} {:>14.1%}   {}".format(
                    p, mean(saved[p]), mean(tim[p]), mean(worst), q05(worst), cs))
            say("  per-task accuracy kept (mean over splits):")
            say("  {:<14} ".format("") + " ".join("{:>9}".format(t[:9]) for t in runs))
            for p in pols:
                say("  {:<14} ".format(p) + " ".join("{:>8.1%} ".format(mean(ret[p][t])) for t in runs))

            # Held-out: format cuts learned on ALL natural samples, applied to unseen sets.
            full_idx = {t: list(range(len(r["ids"]))) for t, r in runs.items()}
            c_fmt_all = {}
            for f in ("open", "closed"):
                units = [(runs[t], [i for i in full_idx[t] if fmt[t][i] == f]) for t in runs]
                c_fmt_all[f] = safe_cut([(r, i) for r, i in units if len(i) >= 10], L, tau)
            ho = [(t, f, load(a.results, tag, t)) for t, f in HELD_OUT.items()]
            ho = [(t, f, r) for t, f, r in ho if r]
            if ho:
                say("  held-out sets, format cuts from the natural tasks (open {}, closed {}):".format(
                    c_fmt_all["open"], c_fmt_all["closed"]))
                for t, f, r in ho:
                    idx = list(range(len(r["ids"])))
                    c = c_fmt_all[f]
                    full = acc(r, idx, L)
                    say("    {:<14} {:<6} full {:.3f}  at cut {:.3f}  kept {:.1%}  vis saved {:.0%}  time {:.0%}".format(
                        t, f, full, acc(r, idx, c), acc(r, idx, c) / full, 1 - c / L, time_ratio(r, c)))
    p = os.path.join(a.out, "schedule_sim.txt")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()

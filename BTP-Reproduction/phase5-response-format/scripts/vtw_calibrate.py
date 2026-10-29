"""VTW's own choice of withdrawal layer, applied to Qwen2.5-VL-7B on every task we measured.

Visual Tokens Withdrawal (Lin et al., AAAI 2025) removes all vision tokens from layer K on
and picks K per target dataset: sample 20 inputs, scan K from 5 to N, and take the first K
whose KL divergence between the standard and the withdrawn output logits is below
eta = 0.003. This script reproduces that search on the same tasks as Phases 4 and 5, so the
chosen K can be scored against our stored per-layer accuracies without new sweeps.

Convention. VTW's layer K is 1-based and the first layer WITHOUT the image, so the image is
kept by layers 1..K-1. In our sweeps l = number of layers that kept the image, so l = K - 1.

Two divergences are logged per task and layer, both KL(standard || withdrawn) averaged over
the 20 samples:
  kl_first   on the first output token (the next-token logits after the prompt)
  kl_answer  over every token of the standard model's greedy answer, teacher-forced
VTW's paper states the criterion on the output logits without fixing the position; the first
is the literal reading for single-token benchmarks, the second the natural one for longer
answers. Per-sample values are saved so any other threshold can be applied afterwards.

  python vtw_calibrate.py --out results/phase5/vtw_qwen7b.json
"""

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(ROOT, "phase3-mechanism", "scripts"),
          os.path.join(ROOT, "phase4-access-horizon", "scripts"), HERE):
    sys.path.insert(0, p)

from measure_access_horizon import SHORT_SUFFIX, TASKS, save  # noqa: E402

# Phase 4 tasks (natural format and the format controls) and Phase 5 conditions.
P4 = ["mmbench", "scienceqa", "ai2d", "pope", "textvqa_mc", "chartqa_mc", "mmbench_fmc",
      "ai2d_fmc", "gqa", "chartqa", "docvqa", "textvqa", "mmbench_fopen", "ai2d_fopen"]
P5 = ["p5_yesno", "p5_mc4_type", "p5_mc2", "p5_mc8", "p5_fill", "p5_open"]
MAX_NEW = {"docvqa": 32}


_P5 = {}


def build(task, n_cal, seed):
    if task.startswith("p5_"):
        if not _P5:
            from measure_response_modes import build_all
            _P5.update(build_all(200, 0))
        pool = _P5[task[3:]]
    else:
        pool = TASKS[task]["build"](200 if task not in ("pope",) else 300, 0)
    rng = random.Random(seed)
    return rng.sample(pool, min(n_cal, len(pool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--tasks", default=",".join(P4 + P5))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from measure_visual_depth import VisualTokenDeleter, pick_loader
    model, layers, run, _ = pick_loader(args.model)(args.model)
    L = len(layers)
    deleter = VisualTokenDeleter(model, layers, 0)

    res = {"model": args.model, "n_layers": L, "n": args.n, "seed": args.seed,
           "k_min": args.k_min, "eta": 0.003, "tasks": {}}
    if os.path.exists(args.out):
        res = json.load(open(args.out))
        print("resuming, tasks done:", sorted(res["tasks"]))

    def logits_for(inp, ans):
        """Logits at the positions that predict the answer tokens (first row = first token)."""
        full = dict(inp)
        full["input_ids"] = torch.cat([inp["input_ids"], ans], 1)
        full["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(ans)], 1)
        with torch.inference_mode():
            out = model(**full, use_cache=False)
        # From the end: with tokens withdrawn the output is shorter than the input.
        return torch.log_softmax(out.logits[0, -ans.shape[1] - 1:-1].float(), -1)

    def kl(lp, lq):
        return (lp.exp() * (lp - lq)).sum(-1)          # per position, KL(p || q)

    for task in [t for t in args.tasks.split(",") if t]:
        if task in res["tasks"]:
            continue
        t0 = time.time()
        samples = build(task, args.n, args.seed)
        max_new = MAX_NEW.get(task, 16)
        first = {str(K): [] for K in range(args.k_min, L + 1)}
        answer = {str(K): [] for K in range(args.k_min, L + 1)}
        for s in samples:
            inp, positions, proc = run(s["image"], s["question"],
                                       suffix=s.get("suffix", SHORT_SUFFIX),
                                       max_pixels=s.get("max_pixels"))
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
            ans = gen[:, inp["input_ids"].shape[1]:]
            if ans.shape[1] == 0:
                continue
            ref = logits_for(inp, ans)
            for K in range(args.k_min, L + 1):
                deleter.layer_index = K - 1        # layers 1..K-1 keep the image
                deleter.arm()
                try:
                    deleter.begin_sample(positions)
                    d = kl(ref, logits_for(inp, ans))
                finally:
                    deleter.disarm()
                first[str(K)].append(d[0].item())
                answer[str(K)].append(d.mean().item())
            torch.cuda.empty_cache()

        def pick(curve, eta):
            for K in range(args.k_min, L + 1):
                v = curve[str(K)]
                if v and sum(v) / len(v) < eta:
                    return K
            return None

        res["tasks"][task] = {"ids": [s["id"] for s in samples], "kl_first": first,
                              "kl_answer": answer,
                              "K_first": pick(first, 0.003), "K_answer": pick(answer, 0.003)}
        save(res, args.out)
        print("  {:<14} K(first token) = {}   K(answer) = {}   {:4.0f} s".format(
            task, res["tasks"][task]["K_first"], res["tasks"][task]["K_answer"],
            time.time() - t0), flush=True)
    print("saved to", args.out)


if __name__ == "__main__":
    main()

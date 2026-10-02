#!/usr/bin/env python3
"""
Record which textvqa_hard samples got three distractors from their own image. CPU only.

The horizon runs store per-sample correctness but not how each question's options were
built. Rebuilding the samples is deterministic (same selection rule, same seed), so this
reproduces the builder and writes {sample_id: number_of_same_image_distractors}.

Run on the HPC login node:
  python phase4-access-horizon/hard_ids.py --out results/horizon/textvqa_hard_ids.json

format_compare.py then reports the horizon on the fully hard subset as well.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_access_horizon as m   # noqa: E402  (also puts phase3-mechanism on the path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples = m.build_textvqa_hard(a.n, a.seed)
    d = {s["id"]: int(s.get("same_image_distractors", -1)) for s in samples}
    json.dump(d, open(a.out, "w"), indent=0)
    full = sum(1 for v in d.values() if v == 3)
    print("wrote", a.out, " samples:", len(d), " fully hard:", full)
    os._exit(0)   # skip the datasets streaming teardown crash on the login node


if __name__ == "__main__":
    main()

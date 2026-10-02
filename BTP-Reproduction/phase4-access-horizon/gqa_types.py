#!/usr/bin/env python3
"""
Fetch GQA question-type labels for the samples used in the horizon runs. CPU only, no GPU.

GQA labels every question two ways:
  structural  query, verify, logical, choose, compare   (the form of the answer)
  semantic    obj, attr, cat, rel, global               (what the question is about)
and a fine-grained 'detailed' type (e.g. relS, existAttrC, weather).

Run on the HPC login node:
  python phase4-access-horizon/gqa_types.py results/horizon/*gqa*.json \
      --out results/horizon/gqa_types.json

Then copy gqa_types.json back and run gqa_split.py locally.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    want = set()
    for f in a.results:
        for sid in json.load(open(f))["sample_ids"]:
            want.add(sid[len("gqa_"):] if sid.startswith("gqa_") else sid)
    print("sample ids to label:", len(want))

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev",
                      streaming=True)
    out = {}
    for ex in ds:
        qid = str(ex["id"])
        if qid in want:
            t = ex.get("types") or {}
            out["gqa_" + qid] = {"structural": t.get("structural"), "semantic": t.get("semantic"),
                                 "detailed": t.get("detailed"), "question": ex["question"],
                                 "answer": ex["answer"]}
            if len(out) == len(want):
                break
    missing = len(want) - len(out)
    json.dump(out, open(a.out, "w"), indent=1)
    print("labelled", len(out), " missing", missing, " wrote", a.out)
    import os
    os._exit(0)   # skip the streaming-thread teardown crash seen on the login node


if __name__ == "__main__":
    main()

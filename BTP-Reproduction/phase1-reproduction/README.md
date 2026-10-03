# Phase 1: reproduction

Question: does Balanced Token Pruning (BTP) reproduce on the models the authors released
code for?

**Answer: yes, within about one point.** LLaVA-1.5-7B, LLaVA-1.5-13B and Qwen2.5-VL-7B, full
datasets, baseline and BTP arms. LLaVA-1.6 is not reproducible: its patch was never released.
All numbers: [`../RESULTS.md`](../RESULTS.md#phase-1-reproduction).

One early signal mattered later: Qwen2.5-VL-7B TextVQA fell from 82.7 to 23.1 under BTP.
The paper reports no reading task, and Phases 2 to 4 follow from this number.

## Files

```
phase1-reproduction/
  scripts/   SLURM jobs: bench_* (LLaVA suite), sqa_* (LLaVA ScienceQA), qwen_* (Qwen suite)
             each in a _baseline and a _btp version
  results/   lmms-eval scores, one file per model, task and arm:
             <model>_<task>_<baseline|btp>.json
             INDEX.csv lists every file with its score
  report/    BTP_Reproduction_Deck.pptx, BTP_Reproduction_Results.xlsx
```

Environment setup and the pitfalls that cost time are in the top-level
[`README.md`](../../README.md).

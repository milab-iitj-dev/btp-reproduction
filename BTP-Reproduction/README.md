# BTP Reproduction — LLaVA-v1.5-7B

Reproduction of the NeurIPS 2025 paper **"Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization"** ([arXiv:2505.22038](https://arxiv.org/abs/2505.22038)) on LLaVA-v1.5-7B.

- **Author:** P. S. Kedar (Research Intern)
- **Supervisor:** Prof. Divya Saxena, SAIDE, IIT Jodhpur
- **Hardware:** IIT Jodhpur HPC — NVIDIA A100-SXM4-40GB, SLURM scheduler
- **Date:** June 2026

---

## What this project does

- The paper proposes **BTP (Balanced Token Pruning)** — a method to make vision-language models faster by removing image tokens the model does not really need.
- LLaVA converts every image into 576 tokens. Processing all of them is slow. BTP keeps only the useful ones (128 in the default setting) and drops the rest in stages.
- Our goal: run the original model (baseline) and the BTP version on the same benchmarks, and check if we get the same numbers the paper reports.

---

## How BTP is applied (in simple terms)

- BTP does not change the model weights. It changes only **one file** inside the `transformers` library: `modeling_llama.py`.
- The authors give a modified copy of this file. You replace the original with their copy, and the pruning logic runs inside the model's forward pass.
- This means switching between baseline and BTP is just swapping one file. We kept three copies to make this safe:
  - `modeling_llama.py.BASELINE` — original file (no pruning)
  - `modeling_llama.py.BTP` — authors' file (with pruning)
  - `modeling_llama.py` — whichever one is active right now
- Every job script prints which mode is active before running, so no result can be mixed up.

---

## Results — comparison with the paper

All runs use the **full datasets** (no sampling), batch size 1, FlashAttention-2, on one A100.

### Benchmarks reported in the paper (Table 1, LLaVA-1.5-7B)

| Benchmark | Paper Baseline | Our Baseline | Paper BTP (128 tokens) | Our BTP | Gap (BTP) |
|---|---|---|---|---|---|
| GQA | 62.0 | 61.98 | 59.0 | **58.98** | 0.02 |
| MME (Perception) | 1510.7 | 1507.5 | 1487.0 | **1497.2** | 10.2 |
| MMBench-EN (dev) | 64.3 | 64.00 | 62.7 | **63.49** | 0.79 |
| POPE | 85.8 | 86.98 | 85.6 | **85.36** | 0.24 |
| SQA (ScienceQA-IMG) | 69.4 | 69.41 | 69.1 | **69.21** | 0.11 |
| MM-Vet | 29.0 | — | 29.1 | — | pending |

- Every completed benchmark matches the paper within about 1 point.
- MME uses a 2000-point scale, so a 10-point gap there is less than 1%.

### Extra benchmarks (not in the paper's table)

We also ran these because they are standard LLaVA evaluations:

| Benchmark | Our Baseline | Our BTP | Note |
|---|---|---|---|
| TextVQA (val) | 46.11 | 40.02 | See TextVQA note below |
| SEED-Bench (image) | 66.23 | 63.99 | Image split only — LLaVA-1.5 is not a video model |
| MME (Cognition) | 355.7 | 351.4 | Paper reports only the Perception score |

**TextVQA note:** LLaVA's official number is 58.2, but that setup adds OCR text (words detected in the image) into the prompt. Our evaluation tool (`lmms-eval`) does not add OCR text by default, and the known score for that setting is about 46 — which is exactly what we got. So the difference is the prompt format, not a model problem.

---

## Environment

The paper's code is sensitive to library versions. These exact versions are required:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.3.0 (CUDA 12.1) |
| Transformers | 4.40.0 (exact — the BTP file is written for this version) |
| FlashAttention | 2.7.4.post1 (the authors' specified wheel) |
| datasets | 3.2.0 |
| numpy | 1.26.4 |
| GPU | Ampere or newer (A100 used here) — FlashAttention-2 needs this |

Install order matters: LLaVA → torch → flash-attn wheel → lmms-eval → then re-pin transformers/numpy/datasets last, because each install quietly upgrades the previous one's versions.

---

## Problems we faced and how we solved them

- **BTP gave 36% on POPE (on Kaggle T4 GPUs) instead of ~85%.**
  - The model answered almost everything wrongly, and many outputs were not even valid yes/no answers.
  - Cause: BTP removes image tokens in the middle of generation. The bookkeeping for this (token positions, attention cache) is written for FlashAttention-2. Kaggle's T4 GPU cannot run FlashAttention-2, so the model fell back to a different attention method, the bookkeeping went wrong, and the output text got corrupted.
  - Fix: run on an A100 with FlashAttention-2 and the exact library versions above. The same code then reproduced the paper's numbers.
  - Lesson: a wrong result is not always a wrong method — sometimes it is a wrong environment.

- **A corrupted `modeling_llama.py` backup (505 lines instead of 1566).**
  - We accidentally made our "original" backup while a newer transformers version was installed, so the backup was the wrong file.
  - Fix: force-reinstall transformers 4.40.0 and re-make the backup. We now check the file's line count and grep for the pruning function to confirm which file is active.

- **`cache_position` error when starting evaluation.**
  - LLaVA's code was written for transformers 4.37, but version 4.40 passes one extra argument (`cache_position`) that LLaVA's forward function did not accept.
  - Fix: add `cache_position=None` to the forward function's arguments in `llava_llama.py`. It is safely ignored.

- **Jobs stuck in queue / rejected.**
  - The cluster requires memory to be requested per CPU core (`--mem-per-cpu`), not as a total.
  - Shorter time limits (`--time`) get scheduled faster.
  - Baseline and BTP jobs must never run at the same time (they share the swapped file), so the BTP job is submitted with `--dependency` so it starts only after the baseline job finishes.

- **Evaluation needed a Hugging Face account token** to download benchmark datasets — created a free token and logged in once on the cluster.

---

## Repository contents

```
scripts/        SLURM job scripts (baseline + BTP, per benchmark)
results/        Raw results JSON files from lmms-eval + summary
logs/           SLURM output logs for every job (with mode check lines)
README.md       This file
```

- Every results file in `results/` can be traced to a job log in `logs/` — the log records which mode (baseline/BTP) was active.

---

## Pending work

- **MM-Vet benchmark** — needs GPT-4 as an automatic judge (open-ended answers cannot be string-matched). Waiting on an OpenAI API key. The generation step can be run on the HPC any time; only the judging step needs the key.
- **TextVQA with OCR tokens** — rerun with the OCR-included prompt to match LLaVA's official 58.2 setting.
- **Efficiency measurements** — the paper reports TFLOPS (3.82 → 0.85 with BTP). We observed BTP slightly *slower* in wall-clock time at batch size 1, because extracting attention scores for pruning has a cost and short answers (1 token) give pruning little room to pay off. A proper FLOPs/latency measurement is planned.
- **LLaVA-1.5-13B / other models** — the paper also covers 13B, LLaVA-1.6, and Qwen2.5-VL; reproducing those is a possible next step.

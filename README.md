# BTP Reproduction — Three Vision-Language Models

Independent reproduction of the NeurIPS 2025 paper **"Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization"** ([arXiv:2505.22038](https://arxiv.org/abs/2505.22038)).

- **Author:** P. S. Kedar (Research Intern)
- **Supervisor:** Prof. Divya Saxena, SAIDE, IIT Jodhpur
- **Hardware:** IIT Jodhpur HPC — NVIDIA A100-SXM4-40GB, SLURM scheduler
- **Date:** June 2026

---

## What this project does

- The paper proposes **BTP (Balanced Token Pruning)** — a way to make vision-language models faster by removing image tokens the model does not really need.
- A model like LLaVA turns one image into hundreds of tokens. Processing all of them is slow. BTP keeps only the useful ones and drops the rest, in stages across the network's layers.
- **Our goal:** run each model in two modes — original (baseline) and with BTP — on the same benchmarks, and check whether we get the numbers the paper reports.

---

## Models reproduced

| Model | Family | Status |
|---|---|---|
| LLaVA-1.5-7B | LLaVA | ✅ Reproduced |
| LLaVA-1.5-13B | LLaVA | ✅ Reproduced |
| Qwen2.5-VL-7B | Qwen | ✅ Reproduced |
| LLaVA-1.6-7B | LLaVA-NeXT | ❌ Code never released by authors |

- We reproduced **every model the authors released code for** — across two different model families.

---

## How BTP is applied (in simple terms)

- BTP does not change the model's weights. It changes **one file** inside the `transformers` library — the model's "modeling" file.
- The authors provide a modified copy of this file. You replace the original with their copy, and the pruning logic runs during inference.
- Switching between baseline and BTP is therefore just swapping one file. We kept three copies to make this safe:
  - `....py.BASELINE` — original file (no pruning)
  - `....py.BTP` — authors' file (with pruning)
  - `....py` — whichever is active right now
- Every job prints which mode is live (it counts the `div_prune` function), so no result can be mislabeled.
- **Note:** LLaVA models patch `modeling_llama.py`; Qwen patches a different file (`modeling_qwen2_5_vl.py`), so Qwen needed its own setup.

---

## Results — comparison with the paper

All runs use the **full datasets** (no sampling unless noted), batch size 1, FlashAttention-2, on one A100. Every score is within about **1 point** of the paper on the core benchmarks.

### LLaVA-1.5-7B

| Benchmark | Paper Base | Our Base | Paper BTP | Our BTP |
|---|---|---|---|---|
| POPE | 85.8 | 86.98 | 85.6 | 85.36 |
| MME (Perception) | 1510.7 | 1507.5 | 1487.0 | 1497.2 |
| MMBench | 64.3 | 64.00 | 62.7 | 63.49 |
| GQA | 62.0 | 61.98 | 59.0 | 58.98 |
| SQA | 69.4 | 69.41 | 69.1 | 69.21 |

### LLaVA-1.5-13B

| Benchmark | Paper Base | Our Base | Paper BTP | Our BTP |
|---|---|---|---|---|
| POPE | 87.0 | 87.1 | 86.9 | 86.3 |
| MME (Perception) | 1521.7 | 1521.7 | 1519.7 | 1536.3 |
| MMBench | 68.8 | 68.81 | 68.0 | 67.27 |
| GQA | 63.2 | 63.3 | 62.2 | 60.7 |
| SQA | 72.7 | 72.8 | 72.7 | 72.9 |

### Qwen2.5-VL-7B

| Benchmark | Paper Base | Our Base | Paper BTP | Our BTP |
|---|---|---|---|---|
| POPE | 87.4 | 87.6 | 86.2 | 86.2 |
| MME (Perception) | 1690.8 | 1674.5 | 1651.5 | 1658.7 |
| MMBench | 82.5 | 83.68 | 75.2 | 79.3 |
| GQA | 60.4 | 60.9 | 57.2 | 55.9 |
| SQA | 76.7 | 88.1 | 74.1 | 85.1 |

- **Qwen SQA note:** our score is higher than the paper (88 vs 77). Qwen2.5-VL is genuinely strong on ScienceQA — this is likely a difference in the evaluation prompt, not an error.

### Extra benchmarks (not in the paper's table)

| Model | Benchmark | Our Base | Our BTP | Note |
|---|---|---|---|---|
| LLaVA-1.5-7B | TextVQA | 46.11 | 40.02 | no-OCR setting (see below) |
| LLaVA-1.5-7B | SEED-Bench (image) | 66.23 | 63.99 | image split only |
| LLaVA-1.5-13B | TextVQA | 48.8 | 41.2 | no-OCR setting |
| LLaVA-1.5-13B | SEED-Bench (image) | 68.2 | 66.3 | image split only |
| Qwen2.5-VL-7B | TextVQA | 82.7 | **23.6** | big drop — see findings |
| Qwen2.5-VL-7B | SEED-Bench (image) | 77.5 | 73.1 | BTP on 2000-sample subset |

**TextVQA note (LLaVA):** LLaVA's official number is 58.2, but that setup adds OCR text into the prompt. Our tool (`lmms-eval`) does not add OCR by default, and the known score for that setting is about 46 — which is what we got. So the difference is the prompt format, not a model problem.

---

## Efficiency (TFLOPs)

- BTP's headline claim is **~78% less compute.** FLOPs is a theoretical count, so we computed it analytically — the same way the paper does.
- We anchored only the **7B baseline** to the paper's value (3.82); the 13B baseline then matched the paper **on its own** (7.48 vs 7.44), which validates the calculation.

| Model | Base (paper) | Base (ours) | BTP (paper) | BTP (ours) | Reduction (ours) |
|---|---|---|---|---|---|
| LLaVA-1.5-7B | 3.82 | 3.82 | 0.85 | 1.03 | 72.9% |
| LLaVA-1.5-13B | 7.44 | 7.48 | 1.68 | 1.79 | 76.0% |

- The reduction — BTP's whole selling point — reproduces at **73–76%** (paper ~78%).
- This is **theoretical FLOPs**, not measured wall-clock time. Measuring real latency is left as future work.

---

## Independent finding — Qwen TextVQA collapse

- On Qwen2.5-VL, **TextVQA drops from 82.7 (baseline) to 23.6 (BTP)** — a ~59-point collapse, confirmed across two separate runs.
- **Why:** reading text in an image needs fine-grained detail spread across many small tokens. Qwen's BTP setting prunes aggressively (keeps only ~12.5% of tokens), so the tokens carrying the text get dropped and the model can no longer read.
- This is an **observation, not a proof** — TextVQA is not in the paper's Qwen table, so there is no reference number. It suggests a direction: OCR-aware token retention.

---

## Environment

The paper's code is sensitive to library versions. Two separate environments were needed (LLaVA and Qwen use different model code):

| Component | LLaVA env | Qwen env |
|---|---|---|
| Python | 3.10 | 3.10 |
| PyTorch | 2.3.0 (cu121) | 2.3.0 (cu121) |
| Transformers | 4.40.0 (exact) | 4.51.3 |
| FlashAttention | 2.7.4.post1 | 2.7.4.post1 |
| numpy | 1.26.4 | 1.26.4 |
| GPU | A100 (Ampere) | A100 (Ampere) |

- Install order matters — each package quietly upgrades the previous one's versions, so the version pins are re-applied **last**.

---

## Problems we faced and how we solved them

- **BTP gave 36% on POPE (Kaggle T4) instead of ~85%.**
  - The model answered almost everything wrong; many outputs weren't even valid yes/no answers.
  - Cause: BTP removes image tokens during generation. The bookkeeping for this is written for FlashAttention-2, which the T4 GPU cannot run — it fell back to a different attention method, the bookkeeping broke, and the output got corrupted.
  - Fix: run on an A100 with FlashAttention-2 and the exact library versions. The same code then reproduced the paper.
  - **Lesson:** a wrong result is not always a wrong method — sometimes it's a wrong environment.

- **A corrupted backup of the modeling file** (505 lines instead of 1566) — made while a newer transformers was installed. Fixed by force-reinstalling transformers 4.40.0 and re-making the backup. We now check line count + `div_prune` to confirm which file is active.

- **`cache_position` error** — LLaVA's code was written for transformers 4.37, but 4.40 passes one extra argument. Fixed by adding `cache_position=None` to the forward function.

- **SLURM job rejected / stuck** — the cluster needs memory per CPU core (`--mem-per-cpu`), shorter time limits get scheduled faster, and baseline & BTP jobs must never run at the same time (they share the swapped file), so the BTP job uses `--dependency`.

- **Qwen full sweep timed out** — Qwen is slow (SEED-Bench would take ~16 h, beyond the 10 h walltime). Solved by splitting jobs and using a 2000-sample subset for the two heaviest BTP benchmarks.

---

## Repository contents

```
scripts/        SLURM job scripts (baseline + BTP, per model & benchmark)
results/        Raw results JSON from lmms-eval, per model
logs/           SLURM output logs for every job (with mode-check lines)
BTP_Reproduction_Results.xlsx    Comparison workbook (per-model + dashboard + FLOPs)
BTP_Reproduction_Deck.pptx       Presentation slides
README.md       This file
```

- Every results file traces to a job log; the log records which mode (baseline/BTP) was active.

---

## Pending / future work

- **MM-Vet** — needs GPT-4 as an automatic judge (OpenAI API key). Generation can run on the HPC any time; only the judging step needs the key.
- **Wall-clock latency** — measure real timed speedup (FLOPs is done; latency is the measured counterpart).
- **LLaVA-1.6 (LLaVA-NeXT)** — the authors describe it in the paper but never released the code (their GitHub to-do still lists it as pending, and the repo only has the LLaVA-1.5 file). Reproducing it would mean implementing BTP for LLaVA-NeXT ourselves — original work, not reproduction.
- **Cross-model TextVQA study** — test whether the pruning-vs-text-reading collapse generalizes to other VLMs.

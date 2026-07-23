# Balanced Token Pruning: Reproduction and Failure-Mode Study

Work on the NeurIPS 2025 paper **"Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization"** ([arXiv:2505.22038](https://arxiv.org/abs/2505.22038)), in two phases:

- **Phase 1 (Reproduction):** reproduce the paper's results across every model it released code for. Result: three models reproduced within about a point of the paper.
- **Phase 2 (Failure-mode study):** investigate a text-reading collapse we discovered in Phase 1, and work out why it happens. Full write-up, figures, scripts and raw results are in [`text-failure-study/`](text-failure-study/).

- **Author:** P. S. Kedar (Research Intern)
- **Supervisor:** Prof. Divya Saxena, SAIDE, IIT Jodhpur
- **PhD mentor:** Aditya Sharma
- **Hardware:** IIT Jodhpur HPC, NVIDIA A100-SXM4-40GB, SLURM scheduler
- **Date:** June-July 2026

---

## What this project does

- The paper proposes **BTP (Balanced Token Pruning)** — a way to make vision-language models faster by removing image tokens the model does not really need.
- A model like LLaVA turns one image into hundreds of tokens. Processing all of them is slow. BTP keeps only the useful ones and drops the rest, in stages across the network's layers.
- **Our goal:** run each model in two modes — original (baseline) and with BTP — on the same benchmarks, and check whether we get the numbers the paper reports.

---

## Getting started

The paper's code is sensitive to library versions, and LLaVA and Qwen need **two separate environments** (they patch different model code). The exact pins matter: they are what avoid the FlashAttention fallback that corrupts BTP's output (see "Problems we faced").

| Component | LLaVA env | Qwen env |
|---|---|---|
| Python | 3.10 | 3.10 |
| PyTorch | 2.3.0 (cu121) | 2.3.0 (cu121) |
| Transformers | 4.40.0 (exact) | 4.51.3 |
| FlashAttention | 2.7.4.post1 | 2.7.4.post1 |
| numpy | 1.26.4 | 1.26.4 |
| GPU | A100 (Ampere) | A100 (Ampere) |

```bash
# 1. Clone
git clone https://github.com/milab-iitj-dev/btp-reproduction.git
cd btp-reproduction

# 2. Create the LLaVA environment (repeat with the Qwen pins for the Qwen env)
conda create -n llava_btp python=3.10 -y
conda activate llava_btp

# 3. Install — apply the version pins LAST; each step can silently upgrade the previous one
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e repos/LLaVA
pip install -e repos/lmms-eval
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install transformers==4.40.0 numpy==1.26.4          # re-pin last

# 4. Locate the modeling file inside the ACTIVE env (don't hard-code a path — resolve it)
MODFILE=$(python -c "import transformers.models.llama.modeling_llama as m; print(m.__file__)")
#   e.g. .../site-packages/transformers/models/llama/modeling_llama.py
#   Qwen env instead: transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
# Keep two copies alongside it once: $MODFILE.BASELINE (original) and $MODFILE.BTP (authors' pruning file)

# 5. Switch mode by copying one file over $MODFILE
cp ${MODFILE}.BTP      $MODFILE     # pruning ON  (div_prune count = 1)
# cp ${MODFILE}.BASELINE $MODFILE   # pruning OFF (baseline, div_prune count = 0)

# 6. Run a benchmark — a ready-made SLURM script (recommended on the HPC)...
sbatch scripts/pope_btp.sh
#   ...or directly with lmms-eval:
accelerate launch -m lmms_eval \
  --model llava --model_args pretrained=liuhaotian/llava-v1.5-7b \
  --tasks pope --batch_size 1
```

The exact modeling-file location is inside the active environment's `site-packages`, so the scripts resolve it with the `python -c "import ...; print(m.__file__)"` line above rather than assuming a fixed path. Ready-made SLURM scripts for every model and benchmark (baseline + BTP, each with the file swap and a `div_prune` mode-check baked in) are in [`scripts/`](scripts/) — for example `sbatch scripts/pope_baseline.sh`, `scripts/pope_btp.sh`, `scripts/qwen_btp.sh`, or `scripts/sqa_btp.sh`.

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
| POPE | 85.8 | 87.0 | 85.6 | 85.4 |
| MME (Perception) | 1510.7 | 1507.5 | 1487.0 | 1497.2 |
| MMBench | 64.3 | 64.0 | 62.7 | 63.5 |
| GQA | 62.0 | 62.0 | 59.0 | 59.0 |
| SQA | 69.4 | 69.4 | 69.1 | 69.2 |

### LLaVA-1.5-13B

| Benchmark | Paper Base | Our Base | Paper BTP | Our BTP |
|---|---|---|---|---|
| POPE | 87.0 | 87.1 | 86.9 | 86.3 |
| MME (Perception) | 1521.7 | 1521.7 | 1519.7 | 1536.3 |
| MMBench | 68.8 | 68.8 | 68.0 | 67.3 |
| GQA | 63.2 | 63.3 | 62.2 | 60.7 |
| SQA | 72.7 | 72.8 | 72.7 | 72.9 |

### Qwen2.5-VL-7B

| Benchmark | Paper Base | Our Base | Paper BTP | Our BTP |
|---|---|---|---|---|
| POPE | 87.4 | 87.6 | 86.2 | 86.2 |
| MME (Perception) | 1690.8 | 1674.5 | 1651.5 | 1658.7 |
| MMBench | 82.5 | 83.7 | 75.2 | 79.3 |
| GQA | 60.4 | 60.9 | 57.2 | 55.9 |
| SQA | 76.7 | 88.1 | 74.1 | 85.1 |

- **Note on Qwen SQA:** our score is higher than the paper (88 vs 77). Qwen2.5-VL is strong on ScienceQA, so this is likely a difference in the evaluation prompt, not an error.

### Extra benchmarks (not in the paper's table)

| Model | Benchmark | Our Base | Our BTP | Note |
|---|---|---|---|---|
| LLaVA-1.5-7B | TextVQA | 46.1 | 40.0 | no-OCR setting (see below) |
| LLaVA-1.5-7B | SEED-Bench (image) | 66.2 | 64.0 | image split only |
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
- This is an **observation, not a proof**: TextVQA is not in the paper's Qwen table, so there is no reference number. **This collapse became the subject of Phase 2 below.**

---

## Phase 2 — Text-Failure Study

The Qwen TextVQA collapse became its own investigation: *why* does BTP break text, and can we do anything about it? Four experiments, all on Qwen2.5-VL-7B. (Full report, deck, figures, scripts and raw results live in [`text-failure-study/`](text-failure-study/).)

### What we tested

| Experiment | Question |
|---|---|
| 1. Benchmark sweep | Does the collapse hit other text tasks, or only TextVQA? |
| 2. Retention sweep | Does accuracy fall gradually as more tokens are pruned? |
| 3. Token visualization | Which image regions does BTP actually throw away? |
| 4. Text-region analysis | Are text regions removed more often than the rest? |

### What we found

**1. It spreads to every text-heavy task.**

| Dataset | Baseline | BTP | Drop |
|---|---|---|---|
| TextVQA | 86.2 | 23.3 | −62.9 |
| DocVQA (ANLS) | 94.7 | 19.2 | −75.5 |
| ChartQA | 76.8 | 29.0 | −47.8 |
| AI2D | 86.4 | 79.6 | −6.8 |

TextVQA, DocVQA and ChartQA all collapse. AI2D barely moves, because it reads diagram layout rather than dense text. That contrast pins the failure to reading text specifically.

**2. It is a cliff, not a slope.** Even keeping 90% of tokens already breaks text (TextVQA 86 down to 23), and it stays flat all the way to 12.5%. There is no safe pruning level for text.

**3. Almost nothing survives.** At the 12.5% setting BTP ships with, 87.5% of image patches are removed. The overlay figures show the text buried under the discarded regions.

**4. The twist: BTP does not target text.** It removes text *less* often than non-text (74.8% vs 90.2%), and on charts it clearly protects the text. The token selection is doing its job.

### Why it happens

The failure is not bad token selection. The real reason is that **text has almost no redundancy**. In a photo, one patch of sky looks like the next, so dropping most of them costs nothing, which is exactly why AI2D survives. But every character on a page is unique. Lose three-quarters of the text patches and the words fall apart, no matter which quarter you keep. BTP's core assumption, that visual tokens are redundant and so pruning is safe, simply does not hold for text.

### What we could do about it

- **Content-aware retention:** detect the text-dense regions first and shield them from pruning, so only the redundant background gets cut.
- **Task-adaptive pruning:** prune hard on photos, go gentle on documents and charts, instead of one fixed level for everything.

Both keep BTP's speed where it is safe and stop it firing where it hurts.

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
scripts/                         SLURM job scripts (baseline + BTP, per model & benchmark)
results/                         Raw results JSON from lmms-eval, per model
logs/                            SLURM output logs for every job (with mode-check lines)
BTP_Reproduction_Results.xlsx    Comparison workbook (per-model + dashboard + FLOPs)
BTP_Reproduction_Deck.pptx       Phase 1 presentation slides
text-failure-study/              Phase 2: report, deck, figures, scripts, raw results
README.md                        This file
```

- Every results file traces to a job log; the log records which mode (baseline/BTP) was active.
- Phase 2's `tex

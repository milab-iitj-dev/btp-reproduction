# Phase 5: response format and visual access

Four experiments requested by Prof. Saxena after the Phase 4 review, to pin down why answer
format controls how long a model needs the image:

1. **Number of choices.** Does the safe withdrawal layer move earlier as the options shrink
   (open-ended, 8, 4, 2)?
2. **Same-type distractors.** Are multiple-choice options giving the answer away? Distractors
   of the same kind as the answer, plus a true no-image control.
3. **Response modes.** The same visual fact asked as yes/no, multiple choice,
   fill-in-the-blank and open-ended.
4. **Decision stability.** Does the safe withdrawal layer coincide with the layer at which the
   model's answer stops changing?

## Setup

- Qwen2.5-VL-7B (28 decoder layers), TextVQA validation, the 189 questions of Phase 4's 200
  whose answer is not yes or no. **The same 189 questions in every condition**, so every
  comparison is paired.
- Withdrawal as in Phase 4: all visual tokens deleted before decoder layer `l`, for every `l`
  from 0 to 28, by forward pre-hooks on the unmodified model. `l` = layers that kept the image.
- **Safe layer** = first `l` from which accuracy stays at or above 95% of full-image accuracy.
  **h50** = the Phase 4 horizon (half of what the image adds over the floor). 95% intervals
  by bootstrap over questions; differences by paired bootstrap.
- Distractors:
  - nested: 7 answers of the same kind (number or text) from other images; the 2 options are
    a subset of the 4, which are a subset of the 8;
  - same-type: 3 answers of the same category as the gold answer (brand, year, number, time,
    name, colour and so on), drawn from 4000 other TextVQA questions; 185 of 189 questions got
    all three from the exact category.
- Fill-in-the-blank gives the number of words and the first letter. Yes/no asks
  `Is the answer "X"?`, true for half the questions and a same-type distractor for the rest.
- Each run also records the model's confidence in the correct answer at every withdrawal
  layer (option probability for closed formats, teacher-forced answer probability for
  open ones), a no-image run (text-only prompt), and a logit lens on the unmodified model.

## Results

### Experiment 1: number of choices

| Condition | Full image | No image | Safe layer | h50 |
|---|---|---|---|---|
| Open-ended | 0.910 | 0.053 | 27 [27, 28] | 23 [23, 23] |
| 8 options | 0.989 | 0.545 | 15 [10, 15] | 5 [5, 7] |
| 4 options | 1.000 | 0.672 | 14 [12, 15] | 7 [5, 9] |
| 2 options | 1.000 | 0.720 | 13 [12, 15] | 9 [6, 12] |

- Open-ended to any multiple choice: safe layer **12 to 14 layers earlier** (open minus 8
  options +12 [12, 17]; open minus 2 options +14 [12, 16]).
- 8 to 4 to 2 options: +1 [-3, 3] and +1 [-2, 2]. **The number of options does not move the
  safe layer.** Fewer options only raise no-image guessing (0.55, 0.67, 0.72).

### Experiment 2: same-type distractors

| Condition | Full image | No image | Safe layer | h50 |
|---|---|---|---|---|
| 4 options, distractors from other images | 1.000 | 0.672 | 14 [12, 15] | 7 [5, 9] |
| 4 options, same-type distractors | 0.989 | 0.582 | 15 [13, 15] | 8 [6, 9] |

- Same-type distractors cut no-image accuracy from 0.67 to 0.58. Chance is 0.25, so the
  question and options together still carry a strong language prior.
- The safe layer moves by +1 [-1, 3]. Of the 87 questions that need the image, **only one needs
  it past layer 16** under same-type distractors. Option leakage is real, but it does not
  explain why multiple choice can drop the image early.

### Experiment 3: four response modes

| Condition | Full image | No image | Safe layer | h50 |
|---|---|---|---|---|
| Yes / no | 0.958 | 0.534 | 15 [15, 15] | 9 [8, 13] |
| 4 options | 1.000 | 0.672 | 14 [12, 15] | 7 [5, 9] |
| Fill-in-the-blank | 0.783 | 0.159 | 28 [27, 28] | 23 [23, 23] |
| Open-ended | 0.910 | 0.053 | 27 [27, 28] | 23 [23, 23] |

- **Two regimes, not a gradient.** Verifying or selecting an answer needs the image up to
  layer 15; producing it needs it to 27-28. Open minus yes/no: +12 [12, 13].
- Fill-in-the-blank hints (word count and first letter) do not help: fill and open-ended
  have the same h50, difference 0 [0, 0]. Fill is also less accurate with the full image
  (0.78), mostly from partial answers such as "Bryan" for "bryan owens".
- Withdrawn after layer 22, open-ended answers have the right type and the wrong value
  ("samsung" for "dakota", "33" for "22"). The model knows what kind of answer is needed
  early and copies the exact text from the image only at layers 22 to 23.

### Experiment 4: decision stability

Image-dependent questions only (right with the image, wrong when it is withdrawn at layer 0);
questions the options already give away have a trivial decision layer of 0.

| Condition | n | Safe layer | Decision layer (median) | Decision p90 | Confidence 90% | Logit lens |
|---|---|---|---|---|---|---|
| Yes / no | 78 | 15 | 10 | 15 | 15 | 4 * |
| 2 options | 50 | 13 | 8 | 15 | 15 | 19 |
| 4 options | 74 | 14 | 7 | 14 | 15 | 22 |
| 8 options | 92 | 15 | 6 | 15 | 15 | 24 |
| 4 options, same-type | 87 | 15 | 8 | 15 | 15 | 22 |
| Fill-in-the-blank | 118 | 28 | 23 | 28 | 28 | 26 |
| Open-ended | 159 | 27 | 23 | 27 | 27 | 27 |

\* The yes/no lens compares two labels and sits near 0.5 at every layer; it is not informative.

- **Confidence tracks the safe layer exactly.** Mean confidence in the correct answer reaches
  90% of its full-image value at layer 15 for every closed format and at 27-28 for open ones.
- **Per question, closed formats have a hard ceiling at layer 15.** Questions needing the image
  past layer 16: 0 or 1 in each closed condition, against 156 of 159 open-ended and 110 of
  118 fill-in-the-blank. All five closed conditions jump at the same step, from 14 to 15
  layers; open and fill jump from 22 to 23.
- **In closed formats the image is read before the answer is formed.** In the unmodified
  model, the correct option becomes the top internal choice at layers 19-24, four to nine
  layers after the image stops being needed. In open-ended answering it emerges only at
  layer 27, so the image is needed almost to the end.

### The same split in the Phase 4 data

Re-reading the Phase 4 sweeps with the same 95% safe-layer rule (no new runs) gives the same two
clusters in every model and task, as a share of decoder depth:

| Model | Closed-form tasks and controls | Open-ended tasks and controls |
|---|---|---|
| Qwen2.5-VL-7B (28 layers) | 50-57% (TextVQA MC 14; MMBench, POPE, ScienceQA 15; AI2D, ChartQA MC 16) | 82-96% (MMBench open 23; ChartQA, DocVQA, GQA 24; AI2D open 25; TextVQA 27) |
| Qwen2.5-VL-3B (36 layers) | 42-61% (ScienceQA 15; MMBench, POPE, AI2D 22) | 89-92% (ChartQA, GQA 32; DocVQA, TextVQA 33) |
| InternVL2-2B (24 layers) | 46-58% (TextVQA MC 11, MMBench 13, ChartQA MC 13, AI2D 14, POPE 14) | 79-83% (GQA, AI2D open 19; TextVQA, DocVQA, ChartQA, MMBench open 20) |

Within open-ended TextVQA the depth does not depend on answer length or type: one-word numbers,
one-word text and multi-word answers all have a median per-question layer of 23.

## What it means

- Answer format splits into two regimes: **select** (yes/no, any number of options, easy or
  hard distractors) and **produce** (open-ended, fill-in-the-blank). Within a regime the
  safe withdrawal layer barely moves; between them it moves 12-14 layers.
- Withdrawing at the depth that is safe for closed formats (about layer 15) leaves open-ended
  TextVQA at **0.11** against 0.91. A withdrawal layer calibrated on closed-form tasks does not
  transfer to open-ended ones.
- This refines Phase 4: the format effect is about whether the model must copy content out
  of the image, not about how many options it compares or how hard they are.

## Limitations

- One model and one task. Qwen2.5-VL-7B on TextVQA; Phase 4 showed the task order carries
  across models, but these four controls have not been run on InternVL.
- Closed formats still score 0.53-0.72 without the image, so only 50-92 questions per closed
  condition are image-dependent.
- The logit lens uses the first answer token for open formats and is unreliable for yes/no.
- The first open-ended run lost its withdrawal layers to a confidence bug (output length
  changes when tokens are deleted); it was rerun with the fix. Baseline, no-image and lens
  passes were unaffected and kept.

## Files

```
phase5-response-format/
  README.md                this file
  scripts/
    measure_response_modes.py   builds all seven conditions, runs one withdrawal sweep
    analyse_phase5.py           tables, paired intervals and the four figures
    p5_job.sh                   SLURM job for one condition (COND=...)
    submit_p5.sh                submits all seven as two dependency chains
  results/raw/             qwen7b_textvqa_<cond>.json, cond in open, fill, mc2, mc4, mc8,
                           mc4_type, yesno (per-question answers, confidence, lens)
  results/summary/         phase5_summary.txt, phase5_depths.json
  figures/                 exp1_number_of_choices.png, exp2_same_type_distractors.png,
                           exp3_response_modes.png, exp4_decision_stability.png
```

Run on the HPC from the workspace root, then analyse locally from `scripts/`:

```bash
bash phase5-response-format/scripts/submit_p5.sh
python analyse_phase5.py --results ../results/raw --out ../results/summary --figs ../figures
```

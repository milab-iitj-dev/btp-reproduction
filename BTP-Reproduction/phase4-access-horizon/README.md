# Phase 4: visual access horizons

Question from Prof. Saxena:

> Do different multimodal tasks have different visual-information access horizons, and can
> this be used to design better pruning schedules? First establish whether the horizon is
> task-dependent and reproducible across models; only then move to the algorithmic question.

## Method

For each task and model, all visual tokens are deleted before decoder layer `l`, for every
`l` from 0 (no image at all) to `L` (unmodified model). Per-sample correctness is stored at
every layer, so confidence intervals and any later schedule can be computed without rerunning.

- **Floor** = accuracy with no image (layer 0). **Image contribution** = baseline minus floor.
- **Horizon h50** = the first layer after which the model keeps at least half of the image
  contribution at every later layer. Normalising by the floor matters: multiple-choice tasks
  score well from the question alone.
- 95% intervals by bootstrap over samples; cross-model agreement by Spearman rank correlation
  with a permutation test.
- Deletion uses forward pre-hooks (no source patching). The tool reproduces the Phase 3
  InternVL2-2B TextVQA threshold to within 0.1 of a layer (18.05 against 17.95).

Models: Qwen2.5-VL-7B (28 layers), Qwen2.5-VL-3B (36), InternVL2-2B (24).
Tasks: MMBench, POPE, ScienceQA-IMG, AI2D, GQA (300 samples each), ChartQA, DocVQA, TextVQA (200).

## Results

### 1. The task order is reproducible across models and families

h50 as a share of decoder depth:

| Task | Qwen 7B | Qwen 3B | InternVL 2B |
|---|---|---|---|
| MMBench | 36% | 39% | 38% |
| POPE | 50% | 56% | 50% |
| ScienceQA | 54% | 39% | 50% |
| AI2D | 54% | 61% | 50% |
| GQA | 68% | 61% | 58% |
| ChartQA | 82% | 78% | 75% |
| DocVQA | 82% | 86% | 75% |
| TextVQA | 82% | 89% | 79% |

| Pair | Spearman rho (h50) [95% CI] | permutation p |
|---|---|---|
| Qwen 7B vs Qwen 3B (same family) | 0.895 [0.76, 0.97] | 0.003 |
| Qwen 7B vs InternVL 2B | 0.962 [0.86, 0.98] | 0.0009 |
| Qwen 3B vs InternVL 2B | 0.932 [0.81, 0.99] | 0.001 |

The order carries across models; the absolute layer does not, and must be measured per model.

### 2. The order is driven mainly by answer format, not by reading

Every early task in the suite is closed-form (multiple choice, yes/no) and every late task is
open-ended, so the two explanations were confounded. Three controls separate them.

**GQA split by question type:** open "query" questions sit at 71-81% of depth in all three
models, close to the reading tasks; verify, logical and choose questions sit at 38-56%.

**Same questions, format changed (h50 in layers, paired 95% intervals):**

| Model | Task | Open-ended | Multiple choice | Shift |
|---|---|---|---|---|
| Qwen 7B | TextVQA | 23 | 8 | +15 [+14, +18] |
| Qwen 7B | TextVQA, distractors from the same image | 23 | 10 | +13 [+10, +14] |
| Qwen 7B | ChartQA | 23 | 15 | +8 [+8, +10] |
| Qwen 7B | MMBench (filtered) | 20 | 10 | +10 [+4, +12] |
| Qwen 7B | AI2D (filtered) | 23 | 15 | +8 [+5, +9] |
| InternVL 2B | TextVQA | 19 | 9 | +10 [+8, +10] |
| InternVL 2B | TextVQA, distractors from the same image | 19 | 11 | +8 [+7, +10] |
| InternVL 2B | ChartQA | 18 | 12 | +6 [+5, +6] |
| InternVL 2B | MMBench (filtered) | 17 | 9 | +8 [+6, +10] |
| InternVL 2B | AI2D (filtered) | 19 | 14 | +5 [+4, +7] |

Removing the options moves closed tasks late; adding options moves reading tasks early. The
effect holds in both directions and in both families. Content still matters, less: ChartQA
and AI2D keep part of the gap in multiple-choice form.

### 3. A format-aware schedule (offline, exact)

`schedule_sim.py` scores "delete all visual tokens after layer c" schedules directly from the
stored per-sample results. Cuts are calibrated on a random half and scored on the other half,
200 splits. Rule: every task keeps at least 98% of its full accuracy.

| Model | One cut for all: visual layers saved | Format-aware: visual layers saved | Format-aware worst task |
|---|---|---|---|
| Qwen 7B | 0.2% | 17.5% (open 28, closed 20) | 98.5% |
| Qwen 3B | 5.5% | 22.8% | 98.4% |
| InternVL 2B | 7.9% | 23.0% | 97.9% |

On Qwen 7B the released BTP deletion (after layer 22 for everything) saves 21.4% but keeps
only 22% on the worst task. Format cuts learned on the eight natural tasks transfer unchanged
to the format-control sets (98.3-100.6% kept). Measured end-to-end time falls to 87-93%.

### 4. Format-aware BTP on GPU

Released BTP with one change: the final "delete all remaining image tokens" step runs after
layer 20 for closed-form tasks and never for open-ended ones. The open-ended arm is identical
to corrected BTP from Phase 3, so only closed-form tasks needed new runs. Qwen2.5-VL-7B,
same limits as the earlier arms (AI2D 500, the rest full).

| Task | Format | Baseline | Released BTP | Corrected BTP | Format-aware BTP |
|---|---|---|---|---|---|
| POPE | closed | 87.6 | 86.2 | 86.3 | **86.3** |
| MME (perception) | closed | 1674.5 | 1658.7 | 1651.2 | **1664.1** |
| MMBench-EN | closed | 83.7 | 79.3 | 79.0 | **79.6** |
| ScienceQA-IMG | closed | 88.1 | 85.1 | 85.0 | **85.1** |
| AI2D | closed | 86.4 | 79.6 | 79.2 | **79.6** |
| TextVQA | open | 86.2 | 23.3 | 80.5 | **80.5** |
| DocVQA (ANLS) | open | 94.7 | 19.2 | 76.8 | **76.8** |
| ChartQA | open | 76.8 | 29.0 | 45.4 | **45.4** |
| GQA | mixed, run as open | 60.9 | 55.9 | 58.8 | **58.8** |

- Closed-form tasks lose nothing when the image is dropped two layers earlier than the
  released code does; all five match or beat both released and corrected BTP.
- Open-ended tasks keep corrected BTP's recovery: reading tasks at 77.9% of baseline against
  28.4% for released BTP.
- So the format rule keeps the deletion where it is safe and removes it where it is not.
  Inside BTP the extra compute saved is small, because only 12.5% of visual tokens remain
  after layer 16; the larger saving is the deletion-only schedule in section 3.
- ChartQA stays at 59% of baseline under any BTP arm: there the graded pruning, not the
  deletion, is the main cost (Phase 3).

## Limitations

- Three models, two families, 200-300 samples per task.
- Multiple-choice reading controls still score 0.51-0.61 without the image (0.66 on the fully hard subset); open-ended MMBench and
  AI2D score only 0.28-0.40, so those curves are noisier.
- POPE scores below chance under partial access, so per-sample horizons on yes/no tasks are
  unreliable; task-level horizons are used instead.
- ChartQA and DocVQA images capped at 2048 tokens on Qwen and 6 tiles on InternVL, so
  baselines differ slightly from official numbers.
- The schedule result in section 3 is deletion only; section 4 checks it inside BTP on one model.

## Files

```
phase4-access-horizon/
  README.md                this file
  scripts/                 all code (run from the workspace root on the HPC)
  results/raw/             per-sample deletion sweeps: <model>_<task>_n<N>.json
                           (<task>_mc, _hard, _fmc, _fopen are the format controls)
                           plus gqa_types.json and textvqa_hard_ids.json
  results/summary/         horizon tables (*_summary.csv), rank correlations,
                           GQA split, format control, schedule simulation
  results/btp_format/      lmms-eval scores of format-aware BTP (closed-form tasks)
  figures/                 curves, horizons, cross-model and format-control figures
  report/                  8-slide summary deck (pdf, and the html source it is built from)
```

| Script | Purpose |
|---|---|
| `measure_access_horizon.py` | Deletion sweep and analysis (`--analyse`); all task loaders and format variants |
| `internvl_adapter.py` | InternVL2 support for the hook tool |
| `compare_models.py` | Cross-model rank correlation and relative-depth figure |
| `make_horizon_figure.py` | Per-model curves and horizon figures |
| `gqa_types.py`, `gqa_split.py` | GQA question types and the per-type split |
| `hard_ids.py` | Lists TextVQA questions whose distractors all come from the same image |
| `format_compare.py` | Paired open-ended vs multiple-choice comparison |
| `schedule_sim.py` | Offline pruning-schedule evaluation |
| `p4_pilot_job.sh`, `p4_ivl_job.sh` | SLURM jobs for the sweeps (Qwen, InternVL) |
| `p4_btp_format_job.sh` | Format-aware BTP on one closed-form task |
| `submit_p4_*.sh` | Submitters: pilot, full, ivl, format, hard, mirror, btp_format |

Analysis commands, run from `scripts/`:

```bash
python measure_access_horizon.py --analyse ../results/raw/qwen7b_textvqa_n200.json
python format_compare.py --results ../results/raw --tags qwen7b,ivl2b --names "Qwen2.5-VL-7B,InternVL2-2B" --out ../results/summary
python schedule_sim.py --results ../results/raw --out ../results/summary
```

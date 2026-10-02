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

In progress: released BTP with the final deletion moved to layer 21 for closed-form tasks;
open-ended tasks use corrected BTP (Phase 3 E9). See `p4_btp_format_job.sh`.

## Limitations

- Three models, two families, 200-300 samples per task.
- Multiple-choice controls still score 0.55-0.66 without the image; open-ended MMBench and
  AI2D score only 0.28-0.40, so those curves are noisier.
- POPE scores below chance under partial access, so per-sample horizons on yes/no tasks are
  unreliable; task-level horizons are used instead.
- ChartQA and DocVQA images capped at 2048 tokens on Qwen and 6 tiles on InternVL, so
  baselines differ slightly from official numbers.
- The schedule result is deletion only; its interaction with graded pruning is what step 4 tests.

## Files

| File | Purpose |
|---|---|
| `measure_access_horizon.py` | Deletion sweep and analysis (`--analyse`); all task loaders and format variants |
| `internvl_adapter.py` | InternVL2 support for the hook tool |
| `compare_models.py` | Cross-model rank correlation and relative-depth figures |
| `make_horizon_figure.py` | Per-model curves and horizon figures |
| `gqa_types.py`, `gqa_split.py` | GQA question types and the per-type split |
| `hard_ids.py` | Lists TextVQA questions whose distractors all come from the same image |
| `format_compare.py` | Paired open-ended vs multiple-choice comparison |
| `schedule_sim.py` | Offline pruning-schedule evaluation |
| `p4_pilot_job.sh`, `p4_ivl_job.sh` | SLURM jobs (Qwen, InternVL) |
| `submit_p4_*.sh` | Submitters for pilot, full suite, InternVL, format, hard, mirror and BTP runs |
| `p4_btp_format_job.sh` | Format-aware BTP on the closed-form suite |
| `results/` | Raw per-sample results (`<model>_<task>_n<N>.json`) and summary tables |
| `figures/` | Curves, horizons, cross-model (`qwen7b_vs_qwen3b_vs_ivl2b_relative.png`) and format-control figures |

Analysis commands, run from this folder:

```bash
python measure_access_horizon.py --analyse results/qwen7b_textvqa_n200.json
python format_compare.py --results results --tags qwen7b,ivl2b --names "Qwen2.5-VL-7B,InternVL2-2B" --out results
python schedule_sim.py --results results --out results
```

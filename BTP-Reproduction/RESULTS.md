# Results

Every number in this file comes from a logged run. Raw aggregate scores are under
`results/` and `text-failure-study/results/`; the full run-by-run record with methodology
is `phase3-mechanism/RESULTS_LOG.md`.

Subject paper: Balanced Token Pruning, NeurIPS 2025, arXiv:2505.22038.
Code examined: `NeurIPS2025-Balanced-Token-Pruning` at commit `9682db0`.

---

## Headline

- BTP's released Qwen2.5-VL code deletes **every remaining image token at layer 23**, which
  the paper's Appendix 7.3 specifies only for LLaVA. Disabling that one condition moves
  TextVQA from **0.1725 to 0.7860**.
- There is a **visual token access threshold**: a depth past which deleting all visual
  tokens costs little, and before which the loss is severe. Measured at **80.3%** of decoder
  depth in Qwen2.5-VL-7B and **74.8%** in InternVL2-2B.
- The released constant leaves Qwen **22 layers** of access against a threshold of **22.5**.
  One layer near the threshold is worth about **47.7 percentage points** on TextVQA.
- **The standard benchmark suite does not show any of this.** With the discrepancy present,
  the code still retains **96.1%** of baseline across five of the paper's six benchmarks.

---

## Phase 1: reproduction

Three models, full datasets, matching the paper within about one point.

| Model | Benchmark | Baseline | BTP |
|---|---|---|---|
| LLaVA-1.5-7B | POPE | 86.98 | 85.36 |
| | MME-P | 1507.5 | 1497 |
| | MMBench-EN | 64.0 | 63.49 |
| | GQA | 61.98 | 58.98 |
| | SQA-I | 69.41 | 69.21 |
| LLaVA-1.5-13B | POPE | 87.1 | 86.3 |
| | MME-P | 1521.7 | 1536 |
| | MMBench-EN | 68.81 | 67.27 |
| | GQA | 63.3 | 60.7 |
| | SQA-I | 72.8 | 72.9 |
| Qwen2.5-VL-7B | POPE | 87.6 | 86.2 |
| | MME-P | 1674.5 | 1658.7 |
| | MMBench-EN | 83.68 | 79.3 |
| | GQA | 60.9 | 55.9 |
| | SQA-I | 88.1 | 85.1 |

FLOPs reproduced analytically: 7B 3.82 to 1.03, 13B 7.48 to 1.79. Reductions of 72.9% and
76.0% against the paper's roughly 78%.

**Not reproducible:** LLaVA-1.6. The authors never released the LLaVA-Next patch.

---

## Phase 2: where it breaks

Text-intensive benchmarks, which the paper does not report. Qwen2.5-VL-7B, limit 500.

| Benchmark | Baseline | BTP | Change |
|---|---|---|---|
| TextVQA | 86.2 | 23.3 | -62.9 |
| DocVQA (ANLS) | 94.7 | 19.2 | -75.5 |
| ChartQA | 76.8 | 29.0 | -47.8 |
| AI2D (control) | 86.4 | 79.6 | -6.8 |

- The collapse is a cliff, not a slope. Retaining 90% of tokens already gives the full drop;
  retaining 12.5% gives the same.
- BTP removes text **less** than background: 76.3% against 89.9% on seven unique images. It
  is not biased against text.
- Replicates on Qwen2.5-VL-3B from a strong baseline: TextVQA 78.7 to 10.8.

---

## Phase 3: cause, threshold, and the blind spot

### Four alternative explanations, all eliminated

| Hypothesis | Test | Result |
|---|---|---|
| Selector picks badly | Ablated to diversity, attention, random at fixed budget | 17.3 to 19.4, standard errors 2.6 to 2.7 |
| Text patches are near-duplicates | Feature similarity at layer 4 | text-text 0.379, background-background 0.438 |
| Too many tokens removed | Retention swept 100% to 12.5% | 90% retention already gives 86.2 to 23.3 |
| Wrong tokens kept | Ground-truth OCR oracle forces all text to survive | 0.197 oracle, 0.197 control |

The control that broke it open: at **100% retention**, where nothing is pruned, the score is
still 0.178. The loss was happening after the pruning stages.

### The implementation discrepancy

`qwen-2.5-vl/modeling_qwen2_5_vl.py`, class `Qwen2_5_VLModel` (line 1115), constants at
lines 1140 to 1144, branch at line 1385:

```python
elif layer_index == self.end_layer:          # self.end_layer = 23
    remain_sys_index  = torch.tensor(list(range(self.img_start_idx)))
    remain_text_index = torch.tensor(list(range(img_end, total_length)))
    #  no remain_img_index in the concatenation below
    all_remained_index = torch.cat((remain_sys_index, remain_text_index))
    hidden_states = hidden_states[:, all_remained_index, :]
```

Causal test, job 416735, 200 samples, one condition changed:

| Arm | TextVQA | AI2D |
|---|---|---|
| No pruning | 0.8565 | 0.8750 |
| BTP as released | 0.1725 | 0.8150 |
| Deletion disabled | **0.7860** | 0.8100 |

Appendix 7.3 of the paper specifies complete deletion for the LLaVA family, and for
Qwen2.5-VL specifies retaining 12.5% "to preserve model performance". The released code
applies the LLaVA branch regardless of model.

### Full suite, Qwen2.5-VL-7B

Corrected arm keeps the 12.5% pruning and prevents the deletion.

| Benchmark | Baseline | Released | Corrected | Kept | Pruning cost | Deletion cost |
|---|---|---|---|---|---|---|
| TextVQA | 86.2 | 23.3 | 80.5 | 93.4% | 5.7 | 57.2 |
| DocVQA (ANLS) | 94.7 | 19.2 | 76.8 | 81.1% | 17.9 | 57.6 |
| ChartQA | 76.8 | 29.0 | 45.4 | 59.1% | **31.4** | **16.4** |
| AI2D (control) | 86.4 | 79.6 | 79.2 | 91.7% | 7.2 | 0.4 |
| GQA | 60.9 | 55.9 | 58.8 | 96.5% | 2.1 | 2.9 |
| POPE | 87.6 | 86.2 | 86.3 | 98.5% | 1.3 | 0.1 |
| MME (perception) | 1674.5 | 1658.7 | 1651.2 | 98.6% | 23.3 | 7.5 |
| MMBench-EN | 83.7 | 79.3 | 79.0 | 94.5% | 4.7 | 0.3 |
| ScienceQA-Image | 88.1 | 85.1 | 85.0 | 96.5% | 3.1 | 0.1 |

MME is scored out of roughly 2000, so its last two figures are 1.4% and 0.4% of its own
baseline. Negative deletion costs are given as magnitudes.

**ChartQA reverses the pattern.** Pruning costs 31.4 there against the deletion's 16.4, so
the discrepancy is not the dominant problem on every reading task.

Averaged across the paper's five benchmarks we ran: released **96.1%**, corrected **96.9%**,
a gap of 0.8 points. On the three reading tasks: **28.4%** and **77.9%**.

### Visual token access threshold

All visual tokens deleted at a given depth, graded pruning disabled, TextVQA limit 500.
Reported as layers that still had access, counting from the first.

**Qwen2.5-VL-7B**, 28 layers, baseline 0.8624:

| Layers with access | 1 | 7 | 13 | 19 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|
| TextVQA | 0.072 | 0.099 | 0.135 | 0.142 | 0.158 | 0.234 | **0.645** | 0.763 | 0.781 | 0.782 |

**InternVL2-2B**, 24 layers, baseline 0.7156:

| Layers with access | 2 | 8 | 12 | 14 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| TextVQA | 0.078 | 0.081 | 0.104 | 0.138 | 0.158 | 0.262 | 0.363 | 0.574 | **0.723** |

| Model | Decoder layers | Threshold | Relative depth |
|---|---|---|---|
| InternVL2-2B | 24 | 18.0 | 74.8% |
| Qwen2.5-VL-7B | 28 | 22.5 | 80.3% |

Threshold is the half-of-baseline crossing, interpolated between the bracketing layers.

Qwen never fully recovers, plateauing at 90.7% even with 26 of 28 layers. InternVL reaches
101% with 20 of 24.

**Counting convention.** BTP's `end_layer` is one-based and deletes before the named layer,
so `end_layer = 23` leaves 22 layers with access. Our sweeps and tool are zero-based. An
earlier draft conflated the two and reported Qwen as 83.9%.

### Instrument validation

`phase3-mechanism/measure_visual_depth.py` repeats the measurement with a forward hook and
no source patching. Job 423959, different sample set:

| Layers with access | Source patch | Forward hook |
|---|---|---|
| 13 | 15.7% | 17.0% |
| 22 | 27.2% | 27.5% |
| 23 | 74.8% | 75.8% |
| 24 | 88.5% | 86.3% |

### Cross-family: InternVL2-2B, pruning only

Our own implementation, no discrepancy present, graded pruning to 12.5%.

| Benchmark | Baseline | Pruned | Retained |
|---|---|---|---|
| TextVQA | 0.716 | 0.468 | 65.4% |
| DocVQA (ANLS) | 0.835 | 0.303 | 36.3% |
| ChartQA | 0.586 | 0.398 | 67.9% |
| AI2D (control) | 0.746 | 0.728 | 97.6% |
| GQA | 0.592 | 0.546 | 92.2% |
| POPE | 0.888 | 0.848 | 95.5% |
| MMBench-EN | 83.4 | 81.0 | 97.1% |

Reading tasks average 56.5%. GQA, POPE and MMBench average 94.9%. In points lost that is
43.5 against 5.1, about 8.6 to one. No discrepancy anywhere, and the suite still reports
95% while half the reading ability is gone.

Deleting all tokens at layer 20 of InternVL costs nothing (0.7234 against a 0.7156
baseline), because layer 20 falls above its threshold.

---

## Conclusions

**Implementation.** The released BTP code deletes all remaining image tokens just below the
measured threshold for Qwen2.5-VL, while the paper's appendix asks for retention on that
model. This says nothing about author intent, and nothing about Qwen2.5-VL as a model.

**Model.** Visual token access has a measurable depth threshold in both models tested, and
the two differ by more than five percentage points of decoder depth. That is enough for a
constant transplanted between architectures to land on the wrong side.

**Evaluation.** The benchmark suite did not reveal the degradation in text reading, with the
discrepancy present or absent. A suite containing no reading task cannot distinguish a
method that preserves reading from one that destroys it.

**Separability.** Pruning and deletion have separable costs, and which dominates depends on
the benchmark.

---

## Claims withdrawn

- *"Text has almost no redundancy."* Phase 2's explanation. Our own measurement refutes the
  premise: text patches are **less** alike than background patches, 0.379 against 0.438.
- *"Surviving text coverage predicts a correct answer."* Held at n=11, vanished at n=250.
  The original result came from mixing in AI2D, which does not require reading.
- *"The effect is specific to Qwen."* Concluded from one measurement in InternVL. The depth
  sweep showed it wrong: both models have a threshold, at different depths.
- *"Qwen's threshold is at 83.9% depth."* A counting error; it is 80.3%.

## Known limitations

- Two architectures only.
- Thresholds measured on TextVQA only. ChartQA's decomposition suggests it may differ there.
- The InternVL implementation is ours, with the diversity selector only and layers
  transplanted from Qwen rather than calibrated.
- The OCR oracle ran at one budget after an unexplained slowdown ended the sweep.
- Phase 2 and 3 comparisons use 500-sample subsets, matched across arms.

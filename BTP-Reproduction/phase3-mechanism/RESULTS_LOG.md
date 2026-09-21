# Phase 3 Results Log

Running record of every Phase 3 run: what was executed, what came back, what it means, and
what is still open. Written as the work happens so the final report can be assembled from
facts rather than memory.

Conventions: **VERIFIED** = observed output. **HYPOTHESIS** = not yet tested.
Anything not marked VERIFIED must not go into a report as a result.

---

## 0. Starting point (carried over from Phase 2)

| Fact | Value | Status |
|---|---|---|
| Qwen2.5-VL-7B TextVQA, no pruning | 82.7 | VERIFIED (Phase 1) |
| Qwen2.5-VL-7B TextVQA, BTP 12.5% | 23.6 | VERIFIED (Phase 1) |
| Follow-up run, TextVQA | 86.2 to 23.3 | VERIFIED (Phase 2, different sample size) |
| DocVQA | 94.7 to 19.2 | VERIFIED |
| ChartQA | 76.8 to 29.0 | VERIFIED |
| AI2D (control) | 86.4 to 79.6 | VERIFIED |
| Collapse is a cliff, not a slope | breaks already at 90% retention | VERIFIED |
| Retention schedule is correct | per-stage counts logged | VERIFIED (job 373054) |
| LLaVA-1.5-7B TextVQA | 46.1 to 40.0, no collapse | VERIFIED |

Phase 2's stated explanation was "text has almost no redundancy". Phase 3 revises this.

---

## 1. Source-code reading (2026-09-13)

**VERIFIED by reading `modeling_qwen2_5_vl.py`:**

- `div_prune()` (line ~1145) is **greedy farthest-point sampling on cosine distance**:
  normalize features, compute cosine similarity, convert to distance, then repeatedly pick
  the token whose minimum distance to the already-selected set is largest. It deliberately
  keeps dissimilar tokens and discards near-duplicates.
- Stage 1 (layer 4): `atten_select_num = int(k * 0.1)`, extended 1.2x. So only about **10%
  of the budget is attention-driven; roughly 90% comes from the diversity selector.**
- Stage 2 (layer 7): `int(k * 0.4)` attention, remainder diversity.
- Stage 3 (layer 16): inline `topk` on attention, no diversity call.
- **Position embeddings ARE sliced correctly** with `all_remained_index` (lines ~1308-1312,
  ~1343-1347), keeping each surviving token's original position. So the Phase 2 speculation
  about "position rewriting" causing the failure is **not supported**.
- `attention_mask` and `cache_position` are **not** re-sliced after pruning. This is the
  likely reason FlashAttention-2 is mandatory, and why the T4 fallback produced 36% on POPE.

**HYPOTHESIS (H1), not yet tested.** Text patches are *similar to each other in feature
space* (same font, stroke, colour) while each carries unique information. BTP measures
redundancy as feature similarity, so a farthest-point sampler treats a line of text as a
redundant cluster and thins it. The failure would then be a property of the **selection
objective**, not of token count.

Competing, still open: H2 pure information loss, H3 structural/positional damage,
H4 resolution dependence.

---

## 2. Duplicate-image bug in the Phase 2 capture (2026-09-14)

**VERIFIED by md5 over `results/viz3/capture3.json`:**

`capture_tokens3.py` walked consecutive streamed dataset rows, but TextVQA, ChartQA and AI2D
store several questions per image, so the same picture was captured repeatedly.

| | |
|---|---|
| records claimed | 12 |
| unique images | **8** |
| usable (OCR finds text) | **7** |
| duplicate pairs | textvqa_3=_4, chart_0=_1, chart_2=_3, diagram_0=_1 |

Also verified: the old capture asked a **fixed generic question** instead of the dataset's
own question, and discarded the model's answer, so correctness was never measurable.

**Impact:** every Phase 2 average over these images double-counts four of them. The report,
Appendix B.3, README and deck all state "12 images (5 TextVQA, 4 ChartQA, 3 AI2D)" and need
correcting.

---

## 3. Finding 4 recomputed on unique images (2026-09-14)

Script: `recompute_finding4.py --capture results/viz3/capture3.json --dedup`

| metric | Phase 2 (12 records) | corrected (7 usable) |
|---|---|---|
| text patches removed | 74.8% | **76.3%** |
| non-text patches removed | 90.2% | **89.9%** |
| difference | −15.4 | **−13.6** |

Per category (n is tiny, treat as anecdote):

| category | n | text removed | non-text removed |
|---|---|---|---|
| chart | 2 | 73.7% | 92.8% |
| diagram | 2 | 74.7% | 87.8% |
| text_heavy | 3 | 79.2% | 89.5% |

**VERIFIED.** The conclusion is unchanged: BTP removes text *less* than background, so it
is not biased against text. One image inverts the pattern (`diagram_ai2d_2`, text 89.3% vs
non-text 87.2%).

---

## 4. E4, per-word coverage (2026-09-14, CPU only)

Script: `word_coverage.py --capture results/viz3/capture3.json --dedup`

| metric | final_kept (12.5%) | stage1_kept (50%) |
|---|---|---|
| global image tokens kept | 0.124 | 0.500 |
| mean per-word coverage | **0.244** | 0.544 |
| words readable (>=50% of patches) | **0.215** | 0.638 |

By category at 12.5%:

| category | n | mean coverage | readable |
|---|---|---|---|
| diagram | 2 | 0.264 | **0.400** |
| text_heavy | 4 | 0.255 | 0.150 |
| chart | 2 | 0.207 | 0.127 |

**VERIFIED.** Two honest readings:

1. A prediction we made **failed**: readable-word fraction is *not* below the global kept
   fraction (0.215 vs 0.124). Text coverage is about **2x** the global rate, which confirms
   Finding 4 rather than overturning it. Coverage alone does not expose the failure.
2. What does carry signal is the **ordering**: the surviving task (AI2D) keeps words readable
   about 2.7x more often than the failing ones, matching the accuracy ordering.

**Limitations:** n=2 per category for chart and diagram. No correctness labels in this
capture, so this is descriptive, not causal.

---

## 5. Jobs run on GPU (2026-09-15)

Both COMPLETED, exit 0. Backfill picked them up at 02:10 on 15 Sep, far earlier than the
16 Sep estimate.

### 5.1 Probe, job 410228 — INVALID RUN

Ran on `text_heavy_textvqa_0.png`, which is the one image in the set where OCR finds **no
text at all**. Output: `text share of image = 0.0000`, text-text similarity `nan`.

Only usable number: **background-background cosine similarity = 0.4349**.

Not a result. Re-queued on an image that has text.

### 5.2 Capture4 smoke, job 410229 — DEDUP CONFIRMED, SCORING BROKEN

**VERIFIED: the dedup fix works.** Duplicate rows skipped while collecting 5 unique images
per category: text_heavy 1 skipped in 6 rows, chart 4 in 9, diagram 13 in 18. 15 records,
15 unique images.

**But accuracy came back 0.000 for all three categories**, including AI2D, which scores 79.6
under BTP in Phase 2. That is a scoring bug, not a model failure. Inspecting the records
showed two causes:

1. The model answered in prose ("The brand of the camera is...") while references are short
   ("dakota"). The benchmark prompt suffix ("answer using a single word or phrase") was
   missing.
2. AI2D references are **indices into an options list** ('1', '0', '3'), not answer text, so
   they could never match.

Also noted: `instrument_viz3.py` raised `IndexError` because the job called it without its
modeling-file argument. The pre-existing `.BTP_VIZ3` file was used instead, so the capture
still ran, but the job script is now fixed to pass `$MODFILE`.

**Fixes applied to `capture_tokens4.py`:** short-answer suffix for TextVQA and ChartQA,
proper multiple-choice prompt for AI2D with index-to-letter mapping, numeric tolerance in
scoring.

---

## 5A. Re-runs on 2026-09-16

### 5A.1 Capture4 with fixed scoring, job 411314 — VALID

| category | accuracy under BTP | matching Phase 2 benchmark |
|---|---|---|
| diagram (AI2D) | **0.800** | 79.6 |
| text_heavy (TextVQA) | 0.200 | 23.3 |
| chart (ChartQA) | 0.200 | 29.0 |

**VERIFIED.** The scoring fix is validated: AI2D at 0.800 lands almost exactly on the known
79.6, and the two text tasks reproduce their collapse. Dedup also held (15 records, 15 unique
images). The run ends with a `PyGILState_Release` abort during interpreter teardown, but this
happens *after* the JSON is written, so the data is intact.

### 5A.2 Feature-similarity probe, job 411315 — REFUTES THE H1 PREMISE

Image: `text_heavy_textvqa_1.png` (one image only).

| measurement | value | prediction under H1 |
|---|---|---|
| cosine similarity, text-text | **0.3789** | should be HIGHER than bg-bg |
| cosine similarity, bg-bg | **0.4380** | |
| cosine similarity, text-bg | 0.3746 | |
| text share of image | 0.2883 | |
| diversity picks landing on text | 0.2590 | below share (held, weakly) |
| attention picks landing on text | 0.3604 | above share (held) |

**VERIFIED, and it contradicts H1.** Text patches are *less* similar to one another (0.379)
than background patches are to one another (0.438). Text is therefore **not** a tight feature
cluster, so the claim that a diversity selector treats a line of text as a redundant cluster
has no basis. The H1 premise as originally stated is wrong.

What does hold: attention over-samples text (0.360 vs a 0.288 share) while diversity slightly
under-samples it (0.259). That explains why BTP overall removes text *less* than background,
which is Finding 4.

Caveat: one image. Needs repeating across the larger capture.

### 5A.3 Coverage versus correctness, on capture4 (CPU)

All 15 images, 11 with OCR words:

| | n | mean coverage | readable |
|---|---|---|---|
| answered correctly | 4 | 0.256 | 0.225 |
| answered incorrectly | 7 | 0.199 | 0.081 |
| gap | | +0.057 | +0.144 |

Looks supportive, **but it is an artifact**. AI2D answers do not depend on reading dense
text, and AI2D has both high accuracy (0.800) and low readability. Restricting to the
text-dependent tasks only (TextVQA and ChartQA):

| | n | mean coverage | readable |
|---|---|---|---|
| answered correctly | 2 | 0.239 | 0.083 |
| answered incorrectly | 6 | 0.209 | **0.094** |

The coverage gap shrinks to +0.030 and the readability gap **reverses**. So within the tasks
that actually require reading, there is currently **no evidence that surviving text coverage
predicts correctness**. n is far too small (2 versus 6) to conclude either way, but the
earlier positive gap must not be quoted, since it came from mixing in a task that does not
need text.

---

## 5B. E1 selector ablation, job 411681 (2026-09-17) — THE DECISIVE RESULT

TextVQA, 200 samples per mode, **identical 12.5% token budget**, only the selection rule
changes. Unpruned baseline for reference: 82.7.

| mode | selection rule | exact_match | stderr |
|---|---|---|---|
| `btp` | attention seed + diversity fill (unmodified) | 0.1725 | 0.0261 |
| `div` | 100% diversity | 0.1845 | 0.0268 |
| `attn` | 100% attention | **0.1940** | 0.0272 |
| `random` | uniform random | 0.1820 | 0.0263 |

**VERIFIED.** All four sit within about two points of each other, well inside the error bars.
**Uniform random selection performs as well as BTP's attention-plus-diversity selection.**

Note: an earlier attempt (job 411389) failed because the run omitted
`attn_implementation="flash_attention_2"`. Without it the model loads with SDPA, which does
not support `output_attentions`, so `pre_layer_atten` was None and BTP crashed. The working
Phase 1 and 2 scripts always passed that argument.

### What E1 settles

- **H1 is dead.** The diversity objective is not the cause. Removing diversity entirely
  (`attn`) or replacing it with noise (`random`) changes nothing.
- **Selection quality is irrelevant on this task.** The 64-point gap from 82.7 to ~18 is not
  recovered by any selection strategy at this budget.
- Finding 4 now makes sense in context: BTP does protect text relative to background, and
  protecting it is simply not enough.

### The remaining explanation (H2)

Combine E1 with the Phase 2 retention sweep, where the collapse already happens between 100%
and 90% retention. Together they say: **text reading requires near-complete token coverage.
Remove a meaningful fraction and it fails, regardless of which tokens are removed.** The
failure is a property of how sensitive the task is to token removal, not a flaw in BTP's
algorithm.

Caveat: n=200 per mode, stderr ~0.026. The 2-point spread is not significant, but a small
real effect cannot be excluded at this sample size.

---

## 5C. Coverage versus correctness at n=300 (job 411682) — NEGATIVE

Full capture, 300 images, 250 with OCR words.

| | n | mean coverage | readable |
|---|---|---|---|
| answered correctly | 62 | 0.276 | 0.229 |
| answered incorrectly | 188 | 0.287 | 0.266 |
| gap | | **−0.011** | **−0.037** |

Text-dependent tasks only (TextVQA + ChartQA):

| | n | mean coverage | readable |
|---|---|---|---|
| correct | 26 | 0.313 | 0.263 |
| wrong | 163 | 0.285 | 0.259 |
| gap | | +0.028 | **+0.004** |

**VERIFIED negative.** Surviving text coverage does **not** predict whether the answer is
correct. The +0.057 gap seen earlier at n=11 was noise plus the AI2D confound.

This does not contradict H2. At the 12.5% operating point essentially every image is already
below the readability threshold, so there is no variance left for coverage to explain. It is
a floor effect. The evidence for H2 comes from the retention sweep, not from within-condition
coverage.

Per-category coverage at n=100 each: text_heavy 0.417 coverage / 0.413 readable,
diagram 0.270 / 0.250, chart 0.174 / 0.123.

---

## 5D. E2 oracle text preservation, job 412401 (2026-09-18) — NULL, AND DECISIVE

TextVQA, 100 unique images, BTP at the shipped 12.5% budget. The oracle arm force-keeps every
OCR text token (up to the budget) and spends whatever is left on BTP's own rule. The control
arm is the identical script with the oracle disabled, so the comparison is internal.

| arm | overall accuracy | accuracy where text fits the budget (n=61) |
|---|---|---|
| ORACLE | 0.160 | **0.197** |
| CONTROL | 0.180 | **0.197** |

**VERIFIED, and the oracle really did apply:** 21 of 100 predictions differ between the arms.
The identical subset accuracy is not a plumbing failure.

**Forcing perfect text preservation does not restore reading.** On the fair subset the two
arms are identical. Overall the oracle is slightly *worse* (0.160 vs 0.180), because on the
39 over-budget images it keeps only text and discards all surrounding context.

### The budget is the size of the text

| quantity | value |
|---|---|
| mean text share of image | **0.127** |
| BTP budget at shipped setting | 0.125 |
| images where text alone exceeds the whole budget | **39 / 100** |

At 12.5% retention the budget is essentially the same size as the text region. "Keep the
text" is therefore not a spare design choice, it consumes the entire budget, and for 39% of
images it is not even geometrically possible.

### Failure mode

The model does not abstain. It produces plausible text-shaped guesses: `boston` where the
answer was `copenhagen`, `james bond` where it was `joe b.`. Both arms confabulate.

---

## 5E. Synthesis, what Phase 3 establishes

Putting the runs together:

1. **The collapse is real and reproducible** (Phase 1 and 2, two independent runs).
2. **BTP does not target text.** It removes text less than background, 76.3% vs 89.9%.
3. **The selection rule is irrelevant.** Random scores the same as BTP's
   attention-plus-diversity: 0.182 vs 0.1725, well inside the error bars (E1).
4. **Even a ground-truth oracle that preserves all text does not help**, 0.197 against a
   0.197 control (E2).
5. **The collapse begins at the very first cut**, already at 90% retention (Phase 2 sweep).
6. **At the shipped budget there is no room to be clever**: the budget is about the size of
   the text region itself.

**Conclusion.** The failure is not a flaw in *which* tokens BTP chooses. It is a property of
the task: reading text inside an image needs near-complete visual context, and once a large
fraction of tokens is removed the ability is gone regardless of what survives. No selection
policy at this budget recovers it.

This also **rules out the "content-aware retention" idea** we had listed as future work, at
least at this budget. E2 is exactly that idea with a perfect oracle, and it does not work.
Saying so is more useful than proposing a fix that cannot succeed.

Open: whether a much larger budget (say 50% or 75%) plus text preservation would recover,
and whether this generalises beyond Qwen (E5).

### 5H. THE MECHANISM, found and proven (2026-09-19) — jobs 416734, 416735

**The cause is a single unconditional line, not the pruning budget and not the selector.**

At `self.end_layer = 23` the BTP modeling file executes:

```python
remain_sys_index  = torch.tensor(list(range(self.img_start_idx)))
remain_text_index = torch.tensor(list(range(img_end, total_length)))
all_remained_index = torch.cat((remain_sys_index, remain_text_index))
hidden_states = hidden_states[:, all_remained_index, :]
```

`remain_img_index` is absent. **Every remaining image token is deleted at layer 23, always.**
The attention-based variant that would have kept some is commented out in the shipped file.

#### Step 1, the anomaly (job 416734, n=50)

| config | TextVQA |
|---|---|
| BASELINE | 0.876 |
| BTP_PARAM at retention **1.00** (zero tokens removed upstream) | 0.178 |
| SEL_btp variant at retention 1.00 | 0.178 |
| BTP_PARAM at retention 0.125 | 0.178 |

Identical at every budget. `_BTP_F = 1.0` was verified to propagate into the accelerate
worker, so retention really was 1.00 and nothing was pruned at layers 4, 7 or 16. The score
still collapsed, which pointed at something downstream of the budget.

#### Step 2, the causal test (job 416735, n=200)

One line changed, `self.end_layer = 23` to `999`, so the branch never fires on a 28-layer
model. Everything else identical, including the 12.5% staged pruning.

| run | TextVQA | AI2D |
|---|---|---|
| BASELINE | 0.8565 | 0.8750 |
| BTP as shipped | **0.1725** | 0.8150 |
| BTP with layer-23 drop disabled | **0.7860** | 0.8100 |

**VERIFIED.** Disabling that one line restores TextVQA from 0.1725 to 0.7860, about 92% of
baseline. The staged pruning to 12.5% costs only about 7 points on its own. The layer-23 wipe
accounts for roughly 61 of the 68-point collapse.

#### What this explains

Every earlier negative result was downstream of this line:

- **E1, selector irrelevant.** btp, div, attn and random all end at layer 23 with no image
  tokens. There was nothing left for the selector to influence.
- **E2, oracle failed.** Preserved text tokens are deleted at layer 23 anyway.
- **Phase 2 cliff at 90% retention.** Never about the amount. Any configuration ends the same.
- **Coverage did not predict correctness.** A floor effect, every sample was equally wiped.
- **AI2D survives.** Diagram answers rely on features already abstracted into the text stream
  before layer 23; reading needs image patches still present at that depth.
- **Replicates on the 3B.** Same architecture, same hardcoded layer behaviour.

#### Corrected conclusion

> BTP's collapse on text-in-image tasks is not caused by how many visual tokens it removes,
> nor by which ones it selects. The implementation deletes **all** remaining image tokens at
> layer 23. Tasks that must still read the image at that depth fail completely, while tasks
> answerable from earlier abstraction survive. The graded pruning BTP is built around is
> nearly harmless by comparison, costing about 7 points at a 12.5% budget.

Note for the write-up: this is a property of the released implementation. Whether the paper
intends this final drop, or whether it is a leftover from an ablation (the alternative branch
is commented out directly above it), should be checked against the paper text before making
any claim about the authors' intent.

### 5G. E5 cross-model, Qwen2.5-VL-3B, job 414662 (2026-09-18) — REPLICATES

Same modeling class as the 7B, so the existing BTP patch ran unchanged. Only the checkpoint
differs. TextVQA plus the AI2D control, 200 samples each.

| task | baseline | BTP | change |
|---|---|---|---|
| TextVQA | **0.7865** | **0.1075** | **−68.0** |
| AI2D (control) | 0.8050 | 0.7300 | −7.5 |

For comparison, the 7B: TextVQA 82.7 → 23.6 (−59), AI2D 86.4 → 79.6 (−6.8).

**VERIFIED.** The pattern replicates closely, and the collapse on the 3B is if anything more
severe. The control again survives.

**Why this one matters.** The 3B baseline reads text well (0.7865), so this is not the "weak
baseline, little to lose" caveat that limits the LLaVA comparison. A model that could read
before pruning loses that ability entirely afterwards. The failure is therefore **not a quirk
of the 7B checkpoint and not a function of model size**; it follows the architecture.

Still untested: models outside the Qwen2.5-VL family. Phi-3.5-Vision or InternVL2-2B would be
the decisive test, high resolution but not Qwen's 2D position scheme.

### 5F. Budget sweep with oracle — ATTEMPTED, PARKED (2026-09-18)

Tried to repeat E2 at retention 0.125 / 0.25 / 0.50 / 0.75 to see whether text preservation
helps once there is room to spare. Two problems, one fixed, one not:

1. **Fixed.** `modeling_qwen2_5_vl.py.BTP_PARAM` line 29 reads the budget at *module import*:
   `_BTP_F = float(_os.environ.get('BTP_RETAIN','0.125')) ** (1.0/3.0)`. Setting the variable
   inside `main()` is too late, so every pass would silently have run at 0.125 and the sweep
   would have been fake. Fixed by exporting `BTP_RETAIN=$R` in front of the python call in
   the job script. Worth remembering for any future parameterised run.
2. **Not fixed.** With the oracle variant built on `BTP_PARAM`, throughput collapsed to about
   20 minutes per image (4 generate calls in 91 minutes), against ~9 seconds per image when
   the same oracle was built on plain `.BTP` (job 412401). Cause not identified. Cancelled
   rather than spend more GPU on it.

**Consequence for the claim.** E2 stands as tested **at the shipped 12.5% budget only**. The
honest statement is: preserving text does not help at the operating point BTP ships with;
whether it would help at a much larger budget is untested. Do not imply otherwise.

---

## 6. Queued, not yet run (as of 2026-09-16)

Serialized with `--dependency` because all three swap the same modeling file, and running
two at once would silently corrupt results.

| job | what it does | what it decides |
|---|---|---|
| 411314 capture4 | 5 images per category, dedup, real questions, correctness | gives coverage-vs-correctness data; validates the scoring fix (AI2D should be ~0.6-0.8, not 0) |
| 411315 probe | baseline model, one image with text, layer-4 features + OCR mask | tests the H1 premise: is text a tighter feature cluster than background? |
| 411316 E1 | selector ablation, `btp` / `div` / `attn` / `random` at the same 12.5% budget, TextVQA | **the causal test.** If `attn` recovers much of the gap, the selection objective is the cause. If all four sit near 23, H1 is dead. |

Chain: 411314 → 411315 → 411316.

---

## 7. Open questions

1. Does the diversity objective cause the collapse? (E1, queued)
2. Is text really a tighter feature cluster? (E3 probe, queued)
3. Does forcing text tokens to survive restore accuracy? (E2, not yet queued)
4. Does coverage predict correctness within a run? (needs capture4 output)
5. Is this Qwen-specific or resolution-driven? (E5, not yet queued; cheapest first step is
   Qwen2.5-VL-3B, which reuses the same modeling class with no code changes)

---

## 8. What can honestly be claimed today

**Can claim:**
- BTP's diversity selector is farthest-point sampling and supplies about 90% of stage-1 kept
  tokens. Verified in source.
- BTP removes text *less* than background (76.3% vs 89.9% on 7 unique images).
- At the shipped setting, roughly 78% of OCR words fall below half coverage, and the
  surviving task keeps words readable far more often than the failing ones.
- The Phase 2 image count was wrong; the real sample was 8 unique images, 7 usable.

- The corrected capture reproduces the known scores (AI2D 0.800 against 79.6), so the
  measurement pipeline is now trustworthy.

**Refuted, do not repeat:**
- "Text patches are near-duplicates in feature space." Measured the other way round:
  text-text 0.379 versus background-background 0.438. H1's premise fails.
- "Surviving text coverage predicts correctness." Once AI2D is excluded, the gap disappears
  and the readability gap reverses.

**Cannot claim yet:**
- That the diversity objective *causes* the text collapse. E1 has not run.
- Any per-category conclusion from n=2.

**Where this leaves the explanation.** With the feature-cluster story refuted, the remaining
candidate is closer to the original Phase 2 intuition, but stated in information terms rather
than feature terms: reading text needs near-complete coverage, so removing 87.5% of tokens
destroys it regardless of *which* tokens are chosen. That is H2. E1 tests it directly: if
attention-only, diversity-only and random all sit near 23, selection does not matter and H2
is correct. If attention-only recovers, selection does matter after all.


---

## 5I. What the paper actually specifies for the final stage (VERIFIED against arXiv:2505.22038v2)

Source: arXiv HTML, section 7.3 "Experiment Settings" (appendix), read 2026-09-19.

Verbatim (quoted for the record, two sentences):

> "For LLaVA-v1.5-7B, LLaVA-v1.5-13B, and LLaVA-v1.6-7B, we divide the pruning
> process into five stages ... In each stage, except for the last one, we retain
> 50% of the tokens from the previous stage. In the final stage, all tokens are
> discarded to maximize inference speed. For Qwen2.5-VL, since its image token
> processing can be clearly divided into two stages, we retain 25% of the tokens
> in the fourth stage and 12.5% in the final stage to preserve model performance."

### Reading

1. Total discard at a deep layer is a REAL part of BTP, not an ablation leftover.
   It is specified, and the stated purpose is inference speed.
2. It is specified for the LLaVA family ONLY.
3. For Qwen2.5-VL the paper specifies the opposite: the final stage RETAINS 12.5%,
   and the paper states the reason explicitly, "to preserve model performance".
4. Therefore the released implementation's unconditional
   `elif layer_index == self.end_layer:` block (self.end_layer = 23), which deletes
   every remaining image token inside the `our_method` branch with no model-family
   guard, applies the LLaVA schedule to Qwen2.5-VL. It contradicts the paper's own
   appendix specification for that model.

### Supporting details from the paper

- Section 5, "Baselines and models": VTW is described as the baseline that
  "discards all image tokens at a specific transformer layer". The code carries VTW
  separately as `self.vtw` / `self.vtw_layer = 16`. So the layer-23 wipe inside the
  BTP branch is not the VTW baseline code leaking in either.
- Section 4.2 defines pruning at exactly three layers l1 < l2 < l3, with a retained
  subset P at each. No fourth "empty set" stage appears in the method section.
- Table 7 gives lambda = (0.2, 0.5, 0.8, 1.0) for qwen-2.5-vl-7b, i.e. four stages.
  The code has three halvings (layers 4, 7, 16) reaching 12.5% plus the layer-23
  wipe. Stage count matches; the content of the last stage does not.
- Table 1 reports Qwen2.5-VL-7B BTP at roughly 22 to 25% token retention holding
  about 96 to 98% of original average performance. A configuration that ends with
  zero image tokens cannot produce those numbers on the text-heavy benchmarks.

### Consequence for our claim

The framing is now precise and defensible:

  The released BTP implementation applies the LLaVA final-stage total-discard
  schedule to Qwen2.5-VL, whereas the paper's appendix specifies a 12.5% retention
  final stage for that model. This single line accounts for the collapse we measured
  on text-heavy tasks.

Evidence chain, all ours, all reproducible:
  - job 416734: accuracy identical (0.178) at every retention ratio from 1.00 to 0.125.
  - `_BTP_F = 1.0` propagation confirmed, so the sweep was real.
  - job 416735, n = 200 TextVQA:
      BASELINE          0.8565
      BTP_NORMAL        0.1725
      BTP_NO_LAYER23    0.7860   (one-line change, self.end_layer 23 -> 999)
  - Staged pruning to 12.5% therefore costs about 7 points. The layer-23 wipe costs
    the remaining 61 of the 68-point collapse.

Status: PROVEN, and now anchored to the paper text rather than to inference.

### Open question, to raise with Aditya sir

Whether to report this as (a) an implementation defect in the released code, or
(b) a reproducibility finding: the published Qwen2.5-VL numbers are not reproducible
from the released code as shipped. Option (b) is the stronger and more careful claim,
since we cannot know which configuration produced the paper's Table 1.

---

## 5J. E9 - spec-compliant BTP, full suite (job chain 416896-416904)

Arm definition: `.BTP` with `self.end_layer = 23` replaced by `self.end_layer = -1`, so the
layer-23 total-discard branch can never fire. Nothing else changed. Justification is the
paper's own Appendix 7.3, which specifies a 12.5% retention final stage for Qwen2.5-VL and
reserves total discard for the LLaVA family. See section 5I.

Baseline and shipped-BTP arms are NOT re-run; the Phase 1 and Phase 2 numbers are reused,
with limits matched (text set 500, paper set full).

Verified in the job log before trusting any number:
  variant built OK: line 1143 `self.end_layer = -1`
  sanity: live file div_prune count = 1   (BTP active, not baseline)
  no FATAL, FlashAttention-2 loaded, A100-40GB

### TextVQA (limit 500), job 416896, VERIFIED

| arm                  | exact_match |
|----------------------|-------------|
| baseline             | 0.862       |
| BTP as shipped       | 0.233       |
| BTP spec-compliant   | **0.805**   |

- Recovers 57.2 of the 62.9-point collapse, i.e. 91% of the damage was the layer-23 wipe.
- Residual cost of genuine staged pruning to 12.5%: 5.7 points. This is the real price of
  BTP on a text-heavy task, and it is a reasonable one.
- 0.805 / 0.862 = 93.4% of baseline.
- Consistent with the n=200 pilot (0.786), so not a small-sample artifact.

Remaining in chain: docvqa_val, chartqa, ai2d (limit 500), then pope, mme, mmbench_en_dev,
gqa, scienceqa_img (full).

Scripts: phase3-mechanism/e9_job.sh, phase3-mechanism/submit_e9.sh

### E9 full suite result (jobs 416896-416904, 417968), VERIFIED

Fixed arm only. Baseline and shipped reused from Phase 1 (full) and Phase 2 exp1 (limit 500),
limits matched per task.

| benchmark      | baseline | BTP shipped | BTP spec-compliant | % of baseline |
|----------------|----------|-------------|--------------------|---------------|
| TextVQA        | 86.2     | 23.3        | 80.5               | 93.4%         |
| DocVQA (ANLS)  | 94.7     | 19.2        | 76.8               | 81.1%         |
| ChartQA        | 76.8     | 29.0        | 45.4               | 59.1%         |
| AI2D           | 86.4     | 79.6        | 79.2               | 91.7%         |
| POPE (acc)     | 87.6     | 86.2        | 86.3               | 98.5%         |
| MME perception | 1674.5   | 1658.7      | 1651.2             | 98.6%         |
| MMBench-EN     | 83.68    | 79.3        | 79.0               | 94.5%         |
| SQA-I          | 88.1     | 85.1        | 85.0               | 96.5%         |
| GQA            | 60.9     | 55.9        | pending            | job 417968    |

(416903 = GQA died OUT_OF_MEMORY at 4 cpus x 4G. Resubmitted as 417968 with 8 x 8G.)

#### Finding A: the paper's benchmark suite is blind to the defect

On POPE, MME, MMBench and SQA the fixed arm and the shipped arm are IDENTICAL within noise
(86.3 vs 86.2; 1651 vs 1659; 79.0 vs 79.3; 85.0 vs 85.1). Those tasks do not require reading
the image after layer 23, so deleting every image token there costs nothing measurable.

This explains why the defect was never caught. The paper evaluates on GQA, MME, MMBench,
POPE, SQA and MM-Vet. None of them is a text-reading task. Add TextVQA, DocVQA or ChartQA and
the same code loses 63 to 76 points.

The paper's headline claim REPRODUCES on the paper's own suite: mean about 97% of baseline
across POPE, MME, MMBench and SQA, against the stated 96 to 98%. The claim is not wrong. It
is just not measuring the failure mode.

#### Finding B: graded pruning has a real cost that scales with text density

After removing the layer-23 wipe, a residual gap remains and it tracks how much reading the
task requires:

  POPE / MME        1 to 2 points      no reading required
  TextVQA           5.7 points         short scene text
  DocVQA            17.9 points        dense document text
  ChartQA           31.4 points        dense text plus fine-grained values

So the earlier single-line story must NOT be oversold. Layer 23 is the dominant cause, worth
roughly 76 to 91% of the collapse depending on the task, but pruning to 12.5% genuinely does
damage text-dense tasks on top of that.

#### Combined statement for the write-up

Two separable effects, both measured:
  1. An implementation defect: the LLaVA final-stage total-discard schedule is applied to
     Qwen2.5-VL, contradicting the paper's Appendix 7.3. Invisible on the paper's benchmark
     suite, catastrophic on text-reading tasks.
  2. A genuine method limitation: even spec-compliant, pruning to 12.5% costs 18 to 31 points
     on document and chart understanding, while costing 1 to 2 points on tasks that do not
     require reading.

OPERATIONAL NOTE: an OOM kill is SIGKILL, so the bash EXIT trap does not fire and the live
modeling file is left as the BTP variant. Any job that assumes a clean baseline must copy
${MODFILE}.BASELINE at START, not only at exit.

### GQA result (job 417968, full, 8 cpus x 8G), VERIFIED

GQA exact_match: baseline 60.9, BTP shipped 55.9, BTP spec-compliant 58.78. Completed in
1h58m. Live modeling file confirmed restored to BASELINE afterwards (div_prune count = 0).

Note GQA does show a small shipped-vs-fixed gap (2.9 points) unlike POPE / MME / MMBench /
SQA, which were identical within noise. Consistent with GQA being mildly more grounded in
image detail, but the gap is an order of magnitude smaller than on the text tasks.

### E9 AGGREGATE, the headline result

Mean percentage of baseline retained:

| suite                                        | BTP shipped | BTP spec-compliant | difference |
|----------------------------------------------|-------------|--------------------|------------|
| paper's suite: GQA, MME, MMBench, POPE, SQA   | 96.1%       | 96.9%              | 0.8 pts    |
| text suite: TextVQA, DocVQA, ChartQA          | 28.4%       | 77.9%              | 49.5 pts   |

Per-task percentages used:
  shipped  GQA 91.8, MME 99.0, MMBench 94.8, POPE 98.4, SQA 96.6
           TextVQA 27.0, DocVQA 20.3, ChartQA 37.8
  fixed    GQA 96.5, MME 98.6, MMBench 94.5, POPE 98.5, SQA 96.5
           TextVQA 93.4, DocVQA 81.1, ChartQA 59.1

THE DECISIVE POINT: the SHIPPED code, defect included, scores 96.1% on the paper's own
benchmark suite. The paper claims 96 to 98%. So the shipped code reproduces the paper's
headline claim while deleting every image token at layer 23.

The defect is therefore undetectable on that suite in aggregate, not merely per task. It
becomes visible only when the evaluation includes a task that requires reading text from
the image.

Framing for the write-up: the contribution is not "we found a bug". It is that a defect
which destroys 63 to 76 points of text-reading ability is invisible to the standard
benchmark suite used to validate VLM token-pruning methods. The evaluation protocol, not
just this implementation, is what failed.

PHASE 3 QWEN ANALYSIS COMPLETE. Remaining: cross-family check (InternVL2-2B) and the report.

---

## 6. E10 - cross-family test on InternVL2-2B

### Why this model

InternVL2-2B differs from Qwen2.5-VL-7B on every axis that could otherwise explain our
findings:

| axis      | Qwen2.5-VL-7B            | InternVL2-2B                       |
|-----------|--------------------------|------------------------------------|
| backbone  | Qwen2.5, 28 layers       | InternLM2-1.8B, 24 layers          |
| positions | M-RoPE (multimodal RoPE) | standard 1D RoPE                   |
| visual    | variable merged patch grid | 448px dynamic tiling, 256 tok/tile |

If the pattern reproduces here, position handling and patch layout are ruled out as causes.

### Implementation

The BTP authors released patches for LLaVA and Qwen2.5-VL only, so we implemented the
pruning ourselves: phase3-mechanism/make_internvl_btp.py injects into the two remote-code
files InternVL2 ships on the Hub. Arms are selected by the IVL_MODE env var:

  unpatched   pristine files, true baseline
  off         patched files, pruning disabled. Inertness control, MUST equal unpatched
  prune       graded pruning at layers 3, 6, 14 down to 12.5%, no final discard
  wipe        no graded pruning, delete all image tokens at layer 20. Isolates Finding A
  prune_wipe  both, the analogue of BTP as shipped for Qwen

Pruning layers are a proportional transfer of Qwen's 4/7/16 and wipe at 23, scaled from 28
to 24 layers. This is NOT the paper's calibration procedure (Section 4.3), and is reported
as a limitation.

Selector is the diversity component only, farthest-point sampling on cosine distance. The
attention component is omitted because InternLM2's FlashAttention-2 path does not expose
attention scores without further surgery. The InternVL arm therefore tests the PRUNING
REGIME, not a faithful reimplementation of BTP's selector. Stated plainly in the write-up.

### Bug found and fixed during bring-up (worth keeping, it is a real result about the method)

First run died with `CUDA error: device-side assert triggered`, reported at an innocent line
inside the pruning helper. Cause: InternLM2 sizes its rotary cache from the LIVE sequence
length and then gathers cos[position_ids]. Pruning shortens the sequence while deliberately
preserving the ORIGINAL position ids, exactly as the released Qwen code does, so the gather
indexed past the end of a 962-row cache with ids up to 1857. CUDA reported it asynchronously
at the next syncing op.

Fix: record the pre-pruning length and grow the rotary cache to span it. Inert when pruning
is off, since the recorded value then equals the live length.

NOTE FOR THE PAPER: this is not merely our bug. Any implementation that prunes tokens while
preserving original position ids must handle this, and the released Qwen code only avoids it
because Qwen computes position embeddings outside the attention module. It is a portability
hazard of the BTP approach and worth one sentence.

### Smoke test (job 418973, MODE=prune_wipe, n=5), VERIFIED

    [BTP-IVL] layer  3 PRUNE  seq 1858 ->  962   img 1792 ->  896
    [BTP-IVL] layer  6 PRUNE  seq  962 ->  514   img  896 ->  448
    [BTP-IVL] layer 14 PRUNE  seq  514 ->  290   img  448 ->  224
    [BTP-IVL] layer 20 WIPE   seq  290 ->   66   img  224 -> 0

Exact halving, final retention 224/1792 = 12.5%, then total discard. Image token counts vary
per sample (1792, 1280, 768) confirming dynamic tiling is handled correctly.

### Results

Stage A pending. Scripts: make_internvl_btp.py, e10_job.sh, submit_e10.sh.

### E10 Stage A results (jobs 418974-418982), VERIFIED

TextVQA and MMBench-EN, limit 500, all five arms. InternVL2-2B.

| arm        | TextVQA | vs baseline | MMBench-EN | vs baseline |
|------------|---------|-------------|------------|-------------|
| unpatched  | 0.7156  | baseline    | 83.4       | baseline    |
| off        | 0.7156  | 0.0         | (not run)  |             |
| prune      | 0.4678  | -24.8       | 81.0       | -2.4        |
| wipe       | 0.7234  | +0.8        | 83.4       | 0.0         |
| prune_wipe | 0.4682  | -24.7       | 81.0       | -2.4        |

#### 1. Inertness control PASSED

off = 0.7156 = unpatched, identical to four decimal places. The patched file with pruning
disabled behaves exactly like the pristine file. Every other number in this table is
therefore attributable to the pruning itself and not to the injection.

#### 2. Finding A does NOT replicate cross-family

Deleting every image token at layer 20 costs InternVL2-2B nothing: TextVQA 0.7234 against a
0.7156 baseline, MMBench unchanged at 83.4. Confirmed independently by prune_wipe (0.4682)
being equal to prune (0.4678), so the wipe adds nothing on top of graded pruning either.

Relative depth is comparable (Qwen wipes at 23 of 28 = 82%, InternVL at 20 of 24 = 83%), so
this is not explained by the wipe happening earlier. On Qwen, a wipe alone with NO upstream
pruning drove TextVQA to 0.178 (job 416734 at retention 1.00). On InternVL the same
operation is harmless.

CONCLUSION: the layer-23 catastrophe is specific to Qwen2.5-VL, not a general consequence of
discarding visual tokens at depth. Qwen apparently still needs image tokens present at 82%
depth to read; InternVL has already absorbed what it needs into the text stream by then.
Why the two families differ is an open question and a good one for the paper.

CAVEAT PENDING: DEBUG defaults to 0 in submit_e10.sh, so these runs printed no per-layer
trace. Job 418983 re-runs MODE=wipe at n=5 with DEBUG=1 to confirm the branch actually
fires. A null result is only meaningful once the intervention is shown to have happened.

#### 3. Finding B DOES replicate, and the blind spot holds

Graded pruning to 12.5% with no wipe at all:

  TextVQA   0.7156 -> 0.4678   = 65.4% of baseline, a loss of 24.8 points
  MMBench   83.4   -> 81.0     = 97.1% of baseline, a loss of  2.4 points

Different backbone, different positional scheme, different visual tiling, same asymmetry.
Reading degrades roughly eight times as much as the standard suite reports.

#### What this does to the overall argument

The specific defect does not generalise. The EVALUATION claim does, and it arguably gets
stronger:

On InternVL2-2B there is no bug at all. The pruning is honest and behaves exactly as
designed. The standard benchmark suite still reports 97% of baseline while the model has
lost a third of its reading ability. The blind spot is therefore not an artefact of one
defective implementation. It is structural: the suite does not contain a task that measures
the capability most sensitive to removing visual evidence.

Revised framing for the write-up:
  - Qwen2.5-VL: an implementation defect, invisible to the suite (Finding A).
  - InternVL2-2B: no defect, correct pruning, still invisible to the suite (Finding B).
  - The common thread is the evaluation protocol, not the code.

Stage B (DocVQA, ChartQA, AI2D, POPE, GQA across four arms) would strengthen the Finding B
claim on InternVL. Worth running given how clean Stage A is.

### Wipe branch verified firing (job 418983), VERIFIED

MODE=wipe at n=5 with DEBUG=1:

    [BTP-IVL] mode=wipe prune_layers=(3, 6, 14) wipe_layer=20 retain=0.125
    [BTP-IVL] layer 20 WIPE   seq 1858 ->   66   img 1792 -> 0
    [BTP-IVL] layer 20 WIPE   seq 1345 ->   65   img 1280 -> 0
    [BTP-IVL] layer 20 WIPE   seq  834 ->   66   img  768 -> 0

The branch fires, every image token is removed, and no PRUNE lines appear (correct for wipe
mode). The Stage A null result is therefore real, not a no-op.

InternVL2-2B scores 0.72 on TextVQA with ZERO image tokens present after layer 20.

---

## 7. E11 - visual information depth sweep (jobs 419005-419012, queued)

### The question

Stage A leaves two isolated data points:
  Qwen2.5-VL-7B    wipe at layer 23 of 28 (82% depth)  ->  catastrophic, 0.876 to 0.178
  InternVL2-2B     wipe at layer 20 of 24 (83% depth)  ->  harmless, 0.7156 to 0.7234

Same relative depth, opposite outcomes. That contrast is only useful if we can say WHERE
each architecture stops needing the image, so we sweep the wipe layer instead of fixing it.

### What it measures

The depth at which visual information has been fully absorbed into the text stream. Wipe
before that depth and the task should collapse; wipe after it and nothing should happen. The
knee in the curve is the result.

### Why it matters for the paper

Every visual token pruning method implicitly assumes visual information is still needed at
the layer it prunes. That assumption holds at different depths in different architectures,
and nobody measures it. This converts our null result into a positive one, and it explains
the Qwen defect properly: layer 23 is catastrophic there not because deleting tokens at depth
is inherently harmful, but because Qwen has not finished reading by then.

### Design

InternVL2-2B, MODE=wipe, TextVQA limit 500, wipe layer in {2, 4, 6, 8, 10, 12, 16, 20}.
Baseline for comparison is 0.7156 (job 418974). e10_job.sh now honours an incoming
IVL_WIPE_LAYER rather than hardcoding 20. Script: submit_e11.sh.

### Outcomes and what each would mean

  Collapse shallow, recovery by 12 to 16   InternVL finishes reading about halfway through.
                                           The knee is the headline number.
  Flat all the way down to layer 2         The vision encoder is doing the reading and the
                                           language model barely consults the tokens. A
                                           surprising architectural result in its own right.
  Gradual decline, no clean knee           Information is spread across depth rather than
                                           absorbed at a point. Less tidy, still reportable.

All three are worth writing up, which is what makes this a good experiment.

### STILL NEEDED: the matching sweep on Qwen

The InternVL curve alone is half the story. The same sweep on Qwen2.5-VL, varying
self.end_layer over roughly {4, 8, 12, 16, 20, 23, 27}, would give the second curve and turn
this into a single figure comparing where two architectures stop needing visual evidence.
That figure is the strongest artefact Phase 3 could produce. Not yet written.

### E10 Stage B results (jobs 418984-419003), VERIFIED

InternVL2-2B, limit 500, four arms per benchmark. Raw scores:

| benchmark | unpatched | wipe   | prune  | prune_wipe |
|-----------|-----------|--------|--------|------------|
| TextVQA   | 0.7156    | 0.7234 | 0.4678 | 0.4682     |
| DocVQA    | 0.8353    | 0.8245 | 0.3031 | 0.3033     |
| ChartQA   | 0.586     | 0.576  | 0.398  | 0.402      |
| AI2D      | 0.746     | 0.750  | 0.728  | 0.718      |
| POPE      | 0.8880    | 0.8840 | 0.8480 | 0.8540     |
| GQA       | 0.592     | 0.594  | 0.546  | 0.546      |
| MMBench   | 83.4      | 83.4   | 81.0   | 81.0       |

As a percentage of the unpatched baseline:

| benchmark | wipe  | prune |
|-----------|-------|-------|
| TextVQA   | 101.1 | 65.4  |
| DocVQA    |  98.7 | 36.3  |
| ChartQA   |  98.3 | 67.9  |
| AI2D      | 100.5 | 97.6  |
| POPE      |  99.5 | 95.5  |
| GQA       | 100.3 | 92.2  |
| MMBench   | 100.0 | 97.1  |

#### Finding A does not generalise. Settled.

The wipe is inert on all seven benchmarks, spanning 98.3 to 101.1 per cent of baseline.
Deleting every image token at layer 20 of InternVL2-2B costs nothing, anywhere, including
on the three tasks that require reading. prune_wipe equals prune throughout, which is the
second independent confirmation. The wipe branch was verified firing in job 418983.

The layer-23 catastrophe is therefore specific to Qwen2.5-VL. It is not a general property
of discarding visual tokens at depth. E12 sweeps the wipe layer on Qwen to locate where
that model does still need the image.

#### Finding B replicates, and the blind spot is structural

Mean percentage of baseline retained under graded pruning to 12.5 per cent, no wipe:

  text suite  (TextVQA, DocVQA, ChartQA)   56.5 %
  paper suite (GQA, POPE, MMBench)         94.9 %
  gap                                      38.4 points

AI2D again behaves as a control at 97.6 per cent, consistent with Qwen.

This is the result that matters. On InternVL2-2B there is no defect at all. The pruning is
correct and behaves exactly as designed. The standard benchmark suite still reports about
95 per cent of baseline while the model has lost roughly half its reading ability. The blind
spot is not an artefact of one defective implementation; it is a property of the evaluation
protocol.

DocVQA is the worst case at 36.3 per cent, a loss of 53 points, larger than the 17.9 points
that spec-compliant BTP costs Qwen on the same benchmark.

#### Honest caveat, must appear in the write-up

Our InternVL selector implements the diversity component only. The attention component is
omitted because InternLM2's FlashAttention-2 path does not expose attention scores. Real BTP
seeds roughly 10 per cent of the stage-one budget from attention, and attention is known to
over-sample text (probe job 411315: attention picks land on text 36.0 per cent of the time
against a 28.8 per cent share).

So the InternVL pruning damage is an UPPER BOUND on what BTP itself would cost, not a
like-for-like measurement. The direction and the text-versus-suite asymmetry hold; the
magnitude is inflated. Do not compare the 56.5 per cent figure directly against Qwen's
spec-compliant numbers.

A fairer future comparison would require exposing attention scores in InternLM2, which is
feasible but was out of scope here.

---

## 8. THE DEPTH SWEEPS, and a correction to Section 6

**This section supersedes the claim in Section 6 that "Finding A does not generalise".**
It does generalise. The earlier conclusion was drawn from a single wipe layer per model, and
a single layer is not enough to characterise the behaviour.

### 8.1 Qwen2.5-VL-7B, 28 layers (jobs 421205-421218), VERIFIED

Graded pruning disabled, so only the wipe varies. TextVQA, limit 500, unpruned control at
end_layer = -1.

| end_layer | TextVQA |
|-----------|---------|
| none (-1) | 0.8624  |
| 2         | 0.0718  |
| 4         | 0.0780  |
| 6         | 0.0914  |
| 8         | 0.0990  |
| 10        | 0.1094  |
| 12        | 0.1264  |
| 14        | 0.1354  |
| 16        | 0.1452  |
| 18        | 0.1372  |
| 20        | 0.1418  |
| 22        | 0.1584  |
| **24**    | **0.6454** |
| 26        | 0.7806  |

Flat floor from layer 2 to 22, then a cliff. The transition sits between 22 and 24.

### 8.2 InternVL2-2B, 24 layers (jobs 419005-419012), VERIFIED

Same protocol, MODE=wipe, baseline 0.7156.

| wipe layer | TextVQA |
|------------|---------|
| 2          | 0.0780  |
| 4          | 0.0766  |
| 6          | 0.0690  |
| 8          | 0.0808  |
| 10         | 0.0868  |
| 12         | 0.1036  |
| 16         | 0.1576  |
| **20**     | **0.7234** |

Identical shape. Flat floor from 2 to 16, then a cliff to full baseline. The transition sits
between 16 and 20.

### 8.3 The finding

Both architectures behave the same way:

| model            | layers | floor     | cliff between | knee depth |
|------------------|--------|-----------|---------------|------------|
| InternVL2-2B     | 24     | 0.07-0.16 | 16 and 20     | about 75%  |
| Qwen2.5-VL-7B    | 28     | 0.07-0.16 | 22 and 24     | about 82%  |

There is a sharp, architecture-specific depth below which discarding all visual tokens
destroys text reading, and above which it costs nothing. The transition is abrupt, not
gradual: on Qwen, 0.158 at layer 22 against 0.645 at layer 24.

### 8.4 Why this reverses Section 6

Section 6 concluded that the layer-23 catastrophe was Qwen-specific because a wipe at layer
20 of InternVL cost nothing. That was correct as an observation and wrong as a conclusion.
Layer 20 simply happens to fall ABOVE InternVL's knee, while layer 23 falls BELOW Qwen's.
The same operation, opposite outcomes, decided by which side of an unmeasured cliff the
hardcoded constant lands on.

### 8.5 The consequence for BTP

BTP's released Qwen configuration uses end_layer = 23. Our measurements:

    wipe at 22  ->  0.1584
    wipe at 23  ->  0.2344   (from the earlier stages-active build, to be re-confirmed)
    wipe at 24  ->  0.6454

The released constant sits one layer below the threshold. Had it been 24, TextVQA would have
scored about 0.645 instead of 0.234. A single layer is worth roughly 41 points.

This also explains the paper's own Appendix 7.3, which specifies total discard for LLaVA but
12.5 per cent retention for Qwen2.5-VL. The authors evidently knew total discard was unsafe
on Qwen. The released code does not implement what the appendix specifies.

### 8.6 Revised claim for the write-up

> Total-discard pruning schedules hardcode a layer index. There exists a sharp,
> architecture-specific depth below which discarding visual tokens destroys text reading and
> above which it is free. The released BTP implementation selects layer 23 for Qwen2.5-VL,
> one layer below that threshold. This depth is never measured in the literature, and the
> standard benchmark suite cannot detect the consequence of getting it wrong.

This is a stronger and more useful claim than the earlier one. It is not a bug report. It
identifies a quantity that pruning methods depend on, that varies by architecture, and that
nobody currently measures.

### 8.7 Pending

Jobs 421231-421233 re-run Qwen at 23, 25 and 27 on the current build so every point is
directly comparable. Jobs 421234-421237 fill the InternVL gap at 14, 17, 18 and 19 to
localise its knee to a single layer.

### 8.8 COMPLETE CURVES (all jobs finished), VERIFIED

Both sweeps delete every visual token at the stated layer and change nothing else. Graded
pruning is disabled throughout, so depth is the only variable.

**Qwen2.5-VL-7B**, 28 layers, baseline 0.8624:

| layer | score  | % of baseline |
|-------|--------|---------------|
| 2     | 0.0718 |  8.3          |
| 4     | 0.0780 |  9.0          |
| 6     | 0.0914 | 10.6          |
| 8     | 0.0990 | 11.5          |
| 10    | 0.1094 | 12.7          |
| 12    | 0.1264 | 14.7          |
| 14    | 0.1354 | 15.7          |
| 16    | 0.1452 | 16.8          |
| 18    | 0.1372 | 15.9          |
| 20    | 0.1418 | 16.4          |
| 22    | 0.1584 | 18.4          |
| **23**| **0.2344** | **27.2**  |
| **24**| **0.6454** | **74.8**  |
| 25    | 0.7632 | 88.5          |
| 26    | 0.7806 | 90.5          |
| 27    | 0.7820 | 90.7          |

**InternVL2-2B**, 24 layers, baseline 0.7156:

| layer | score  | % of baseline |
|-------|--------|---------------|
| 2     | 0.0780 | 10.9          |
| 4     | 0.0766 | 10.7          |
| 6     | 0.0690 |  9.6          |
| 8     | 0.0808 | 11.3          |
| 10    | 0.0868 | 12.1          |
| 12    | 0.1036 | 14.5          |
| 14    | 0.1378 | 19.3          |
| 16    | 0.1576 | 22.0          |
| 17    | 0.2616 | 36.6          |
| 18    | 0.3628 | 50.7          |
| 19    | 0.5740 | 80.2          |
| **20**| **0.7234** | **101.1** |

#### Half-of-baseline crossing

    InternVL2-2B   layer 18.0 of 24  =  74.9% depth
    Qwen2.5-VL-7B  layer 23.5 of 28  =  83.9% depth

#### Three observations

1. **The shapes match.** A flat floor around 10 to 18 per cent, then an abrupt rise. The
   same phenomenon in both architectures at a different threshold. This is a property of
   vision language models generally, not a quirk of one model.

2. **The shipped constant lands one layer short.** Layer 23 retains 27.2 per cent, layer 24
   retains 74.8 per cent. One layer is worth 47.6 points. BTP's released configuration sits
   on the last layer at which total discard is still catastrophic for Qwen2.5-VL.

3. **Qwen never fully recovers; InternVL does.** Qwen plateaus at 90.7 per cent even when
   the deletion is placed at layer 27 of 28. InternVL reaches 101.1 per cent at layer 20 of
   24, with four layers still to run. So Qwen continues to draw on visual tokens right to
   the end of the stack for roughly nine per cent of cases, while InternVL has genuinely
   finished with them. This was not predicted and is a real architectural difference worth
   reporting.

Figures are generated by reports/make_depth_figures.py, which contains the measured values
and computes the half-crossings directly.


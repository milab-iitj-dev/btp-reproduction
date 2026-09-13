# Phase 3 — Mechanism: why BTP destroys text reading

Phase 2 established *that* BTP collapses on text-in-image tasks and ruled out the naive
explanation (BTP does not preferentially delete text). Phase 3 establishes *why*, with
interventions rather than correlations, and tests whether the failure is Qwen-specific.

---

## 1. The corrected hypothesis

Phase 2 concluded "text has almost no redundancy". Reading the BTP source shows this is
imprecise, and a reviewer will attack it.

`div_prune()` (in `modeling_qwen2_5_vl.py`, line ~1145) is **greedy farthest-point sampling
on cosine distance**: it repeatedly keeps the token *most dissimilar* to everything already
kept, and discards near-duplicates. At the first pruning stage only ~10% of the budget is
chosen by attention; the remaining ~90% comes from this diversity selector.

Now note what text looks like in feature space:

| | feature similarity between patches | information carried per patch |
|---|---|---|
| Sky / wall | high (near-duplicates) | low, interchangeable |
| Characters in a word | **high** (same font, stroke, colour) | **high, each patch unique** |

So text patches are **feature-redundant but informationally unique**. A diversity sampler
cannot tell those apart, because it only measures feature similarity. It therefore sees a
line of text as a redundant cluster and deduplicates it down to a few representatives.

> **H1 (selection-objective hypothesis).** The collapse is caused by BTP's diversity
> criterion, which assumes feature similarity implies informational redundancy. That
> assumption holds for natural image regions and is violated for text.

This reframes Phase 2's Finding 4 instead of contradicting it. The attention component
rescues *some* text tokens, so the removal **rate** for text looks favourable (75% vs 90%),
while the diversity component still destroys **within-word completeness**. Removal rate was
the wrong metric; per-word coverage is the right one.

### Competing hypotheses to rule out

- **H2 (pure information loss).** Losing any large fraction of text tokens breaks reading,
  regardless of *which* ones. Predicts random pruning ≈ BTP pruning.
- **H3 (structural/positional).** The act of pruning disturbs the spatial layout that
  Qwen's 2D position encoding relies on. Predicts collapse even at very low prune rates and
  even when text tokens are fully preserved.
- **H4 (resolution dependence).** Only models that read text through many fine patches can
  collapse. Explains Qwen vs LLaVA and predicts the cross-model results.

H1 and H4 are compatible and are the expected outcome. H2 and H3 must be tested, not assumed.

---

## 2. Experiments and what each one discriminates

| # | Experiment | Script | Discriminates |
|---|---|---|---|
| E1 | Selector ablation at matched budget | `patch_selector_variants.py` | H1 vs H2 |
| E2 | Oracle text-preservation | `patch_selector_variants.py` (`oracle_text`) | H1/H2 vs H3 |
| E3 | Feature-similarity probe | `feature_similarity_probe.py` | direct evidence for H1 |
| E4 | Per-word coverage vs correctness | `word_coverage.py` | replaces removal-rate metric |
| E5 | Cross-model sweep | `run_cross_model.sh` | H4 |

### E1 — Selector ablation (the causal test)

Hold the token budget **identical** (12.5%) and change only *how* tokens are chosen:

| Mode | Selection rule | H1 predicts |
|---|---|---|
| `btp` | attention 10% + diversity 90% (unmodified) | collapse (baseline ~23) |
| `div` | 100% diversity | collapse, equal or worse |
| `attn` | 100% attention | **substantial recovery** |
| `random` | uniform random | in between; if ≈ `btp`, H1 is weakened |

The decisive comparison is `attn` vs `div` at the same budget. If attention-only recovers a
large part of the gap, the failure is the *selection objective*, not the token count. If all
four collapse equally, H2 or H3 is correct instead and H1 is dead.

### E2 — Oracle text-preservation (causal, diagnostic only)

Force-keep every OCR-identified text patch, then fill the remaining budget with BTP's own
rule, so the budget is unchanged.

- Accuracy recovers → loss of text tokens is genuinely causal (H1/H2 confirmed, H3 rejected).
- Accuracy stays low → pruning damages something beyond token identity (H3), and the whole
  redundancy story needs rewriting.

This is a **diagnostic probe, not a proposed method.** It uses ground-truth OCR at inference,
which no real system has. It exists to test causality, and the paper must say so.

### E3 — Feature-similarity probe (the elegant one)

No benchmark run needed, which makes it cheap. For each image, capture image-token hidden
states at the pruning layer, label each patch text/background using OCR boxes, then report:

1. mean pairwise cosine similarity within text, within background, and across;
2. what fraction of the diversity selector's picks land on text vs background;
3. the same for the attention selector.

H1 predicts text patches form a **tighter** cluster than background, and that the diversity
selector under-samples them relative to their share of the image.

### E4 — Per-word coverage

For each OCR word, compute the fraction of its patches that survive, then correlate with
whether that question was answered correctly. Predicts a sharp readability threshold and
gives a far better metric than global removal rate.

### E5 — Cross-model sweep

Vary two axes independently to separate architecture family from resolution:

| Model | Resolution | Position scheme | Prediction (H4) |
|---|---|---|---|
| LLaVA-1.5-7B | fixed low (576 tok) | 1D | no collapse (confirmed, 46 → 40) |
| Qwen2.5-VL-7B | dynamic high | 2D M-RoPE | collapse (confirmed, 82.7 → 23.6) |
| Qwen2.5-VL-3B | dynamic high | 2D M-RoPE | collapse (tests size, not family) |
| InternVL2-2B | dynamic tiling | 1D | collapse if resolution drives it |
| Phi-3.5-Vision | dynamic crops | 1D | collapse if resolution drives it |
| SmolVLM-2B | low/moderate | 1D | little or no collapse |

If the high-resolution 1D models also collapse, resolution (H4) is the driver and the 2D
position story (H3) is dead. If only Qwen collapses, H3 deserves a closer look. Either
result is publishable, which is what makes this a good design.

---

## 3. Status and honesty notes

- The scripts here are **written but not yet executed**; they need a smoke test on one image
  before any full run. Nothing in this folder should be cited as a result until it has run.
- E2 uses ground-truth OCR and is a diagnostic, not a method.
- Report the budget explicitly in every ablation, since the whole argument depends on the
  budget being held constant.

## 4. Run order

1. `feature_similarity_probe.py` (cheap, no benchmark, validates the premise)
2. `patch_selector_variants.py` → generate variants → run TextVQA for each mode
3. `word_coverage.py` on the captured indices
4. `run_cross_model.sh` last, it is the most compute-hungry

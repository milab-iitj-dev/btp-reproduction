# Phase 3: the cause, the threshold, the blind spot

Question: what actually destroys text reading under BTP on Qwen2.5-VL, and is it specific
to one model?

## What was found

1. **Four explanations eliminated.** Changing how tokens are selected, text patches being
   near-duplicates, how many tokens are removed, and protecting every text token with OCR all
   leave TextVQA at 17-20%.
2. **The control that broke it open.** At 100% retention, where nothing is pruned, TextVQA is
   still 17.8%. The loss happens after the pruning stages.
3. **The cause.** The released Qwen code deletes every remaining image token at layer 23
   (`self.end_layer = 23`). The paper's appendix specifies that step for LLaVA only and asks
   Qwen to keep 12.5%. Disabling that one condition: TextVQA 0.1725 to 0.7860.
4. **A visual token access threshold.** Deleting all image tokens costs little above a certain
   depth and a lot below it: 80.3% of depth in Qwen2.5-VL-7B, 74.8% in InternVL2-2B. The
   released constant sits just below Qwen's threshold.
5. **The benchmark blind spot.** With the discrepancy present, the paper's suite still shows
   96.1% of baseline while reading tasks are at 28.4%. InternVL2-2B with our own pruning and
   no discrepancy: 94.9% on the suite, 56.5% on reading.

All numbers: [`../RESULTS.md`](../RESULTS.md#phase-3-cause-threshold-and-the-blind-spot).
Run-by-run record, including invalid runs and withdrawn claims:
[`RESULTS_LOG.md`](RESULTS_LOG.md).

## Files

```
phase3-mechanism/
  RESULTS_LOG.md   every run, marked VERIFIED or HYPOTHESIS
  scripts/
    patch_selector_variants.py, e1_job.sh     selector ablation (E1)
    oracle_text_probe.py                      OCR text-token protection (E2)
    feature_similarity_probe.py               are text patches near-duplicates? (E3)
    word_coverage.py, capture_tokens4.py,
    recompute_finding4.py                     per-word coverage analyses
    run_cross_model.sh                        Qwen2.5-VL-3B replication (E5)
    e9_job.sh, submit_e9.sh                   corrected BTP on the full suite (E9)
    make_internvl_btp.py, e10_job.sh,
    submit_e10.sh, submit_e11.sh              BTP-style pruning on InternVL2 (E10, E11)
    e12_job.sh, submit_e12.sh                 Qwen deletion-depth sweep (E12)
    measure_visual_depth.py, e13_job.sh       hook-based threshold tool and its check (E13)
  figures/         report figures (fig0 to fig9)
  report/          BTP_Phase3_Report (docx, pdf), BTP_Phase3_Deck (pptx, pdf), build scripts
```

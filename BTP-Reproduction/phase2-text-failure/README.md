# Phase 2: where BTP breaks

Question: why does BTP collapse on Qwen2.5-VL-7B when the task needs reading text in the image?

## What was found

- **The collapse is specific to reading tasks** (500-sample subsets): TextVQA 86.2 to 23.3,
  DocVQA 94.7 to 19.2, ChartQA 76.8 to 29.0. AI2D, the control, only 86.4 to 79.6.
- **It is a cliff, not a slope.** Keeping 90% of visual tokens already gives the full drop;
  keeping 12.5% gives the same. A verified token count at every stage rules out a
  configuration error.
- **BTP does not target text.** It removes text patches less often than background
  (76.3% against 89.9%).
- **It replicates on Qwen2.5-VL-3B** (TextVQA 78.7 to 10.8).

The explanation offered at the time, "text has low redundancy", was later **withdrawn**:
Phase 3 measured text patches as *less* alike than background, and found the real cause.

## Files

```
phase2-text-failure/
  scripts/   exp1_* benchmark arms, exp2_* retention sweep, capture/visualisation scripts
  results/
    exp1_benchmarks/        baseline and BTP on TextVQA, DocVQA, ChartQA, AI2D (+ summary.txt)
    exp2_retention_sweep/   TextVQA and DocVQA at retention 100% down to 12.5%
                            (+ retention_verification.txt: tokens kept per stage)
    exp3_token_visualisation/  kept-token capture and OCR overlap analysis
    INDEX.csv               every result file with its score
  figures/   retention curve, text-removal rates, kept-token overlays
  report/    BTP_TextFailure_Report.docx, BTP_TextFailure_Deck.pptx
```

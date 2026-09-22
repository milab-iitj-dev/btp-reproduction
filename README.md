# Balanced Token Pruning: reproduction and failure-mode study

Reproduction and analysis of **"Balanced Token Pruning: Accelerating Vision Language Models
Beyond Local Optimization"** (NeurIPS 2025, [arXiv:2505.22038](https://arxiv.org/abs/2505.22038)).

- **Author:** P. S. Kedar, Research Intern
- **Supervisor:** Prof. Divya Saxena, SAIDE, IIT Jodhpur
- **Mentor:** Aditya Sharma, PhD Scholar
- **Hardware:** IIT Jodhpur HPC, A100-SXM4-40GB, SLURM
- **Period:** June 2026 to present

---

## What we found

**All numbers and their provenance: [`BTP-Reproduction/RESULTS.md`](BTP-Reproduction/RESULTS.md)**

- BTP reproduces on LLaVA-1.5-7B, LLaVA-1.5-13B and Qwen2.5-VL-7B, within about a point.
- On tasks that require reading text inside an image, Qwen2.5-VL collapses: TextVQA 86.2 to
  23.3, DocVQA 94.7 to 19.2. The paper reports no such task.
- Cause: the released code deletes **every remaining image token at layer 23**, which the
  paper's Appendix 7.3 specifies only for the LLaVA family. Disabling that one condition
  takes TextVQA from 0.1725 to 0.7860.
- There is a **visual token access threshold**, a depth past which deletion costs little and
  before which it is severe. 80.3% of decoder depth in Qwen2.5-VL-7B, 74.8% in InternVL2-2B.
  The released constant lands just below Qwen's. One layer is worth about 47.7 points.
- **None of this is visible on the paper's benchmark suite.** With the discrepancy present
  the code still retains 96.1% of baseline across five of its six benchmarks. InternVL2-2B,
  with no discrepancy at all, still loses half its reading ability while the same suite
  reports 95%.

---

## Repository layout

| Path | Contents |
|---|---|
| [`BTP-Reproduction/RESULTS.md`](BTP-Reproduction/RESULTS.md) | Every result, all three phases, with conclusions and withdrawn claims |
| `BTP-Reproduction/scripts/` | Phase 1 SLURM job scripts |
| `BTP-Reproduction/results/` | Phase 1 aggregate scores from lmms-eval |
| `BTP-Reproduction/logs/` | Phase 1 job logs |
| `BTP-Reproduction/text-failure-study/` | Phase 2: report, deck, figures, scripts, results |
| `BTP-Reproduction/phase3-mechanism/` | Phase 3: the cause, the threshold, the tool |
| `BTP-Reproduction/phase3-mechanism/RESULTS_LOG.md` | Run-by-run record, VERIFIED and HYPOTHESIS marked separately |
| `BTP-Reproduction/phase3-mechanism/report/` | Phase 3 report, docx and pdf, plus the scripts that build it |
| `BTP-Reproduction/BTP_Reproduction_Deck.pptx` | Phase 1 slides |
| `BTP-Reproduction/BTP_Reproduction_Results.xlsx` | Phase 1 workbook |

---

## Phase 3 tooling

- `measure_visual_depth.py` measures the visual token access threshold on any Hugging Face
  vision language model, using forward hooks and no source patching. No training, no
  calibration set. A coarse sweep plus bisection finds the threshold in about seven runs.
- `make_internvl_btp.py` implements BTP-style pruning for InternVL2 by injecting into its
  Hub remote code. The authors released patches for LLaVA and Qwen only.
- `e9` to `e13` job scripts and their `submit_*.sh` chains reproduce every Phase 3 result.

---

## Environments

LLaVA and Qwen need **separate environments**; they patch different model files. The pins
matter, because a FlashAttention fallback silently corrupts BTP's output.

| | LLaVA | Qwen |
|---|---|---|
| Python | 3.10 | 3.10 |
| PyTorch | 2.3.0 cu121 | 2.3.0 cu121 |
| transformers | 4.40.0 exact | 4.51.3 |
| flash-attn | 2.7.4.post1 | 2.7.4.post1 |
| numpy | 1.26.4 | 1.26.4 |
| GPU | A100 | A100 |

```bash
git clone https://github.com/milab-iitj-dev/btp-reproduction.git
cd btp-reproduction

conda create -n llava_btp python=3.10 -y && conda activate llava_btp
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 numpy==1.26.4 flash-attn==2.7.4.post1 --no-build-isolation
pip install -e repos/LLaVA -e repos/lmms-eval
```

BTP is applied by **replacing one file** inside `transformers`, not by importing a library:

```bash
MODFILE=$(python -c "import transformers.models.llama.modeling_llama as m; print(m.__file__)")
cp $MODFILE ${MODFILE}.BASELINE                      # keep the original
cp repos/NeurIPS2025-Balanced-Token-Pruning/llava/modeling_llama.py $MODFILE
grep -c "def div_prune" $MODFILE                     # 1 means BTP is live, 0 means baseline
```

---

## Things that cost us time

- **FlashAttention-2 is mandatory.** Without it lmms-eval loads SDPA, which does not support
  `output_attentions`, so BTP's attention tensor is `None` and it crashes or degrades. A T4
  fallback once produced 36% on POPE and looked like a real result.
- **Never run two BTP jobs at once.** They swap the same modeling file. Chain them with
  `--dependency=afterany:<jobid>`.
- **An OOM kill is SIGKILL**, so a bash `EXIT` trap never fires and the modeling file is
  left dirty. Restore the baseline at job start, not only at exit.
- **`BTP_PARAM` reads its retention at module import.** Setting it inside `main()` is too
  late and the sweep runs silently at the default.
- **Python buffers stdout to log files.** Use `python -u` or progress never appears.
- **Pruning while keeping original position ids** overruns the rotary cache in models that
  size it from the live sequence length. Qwen escapes this only because it builds position
  embeddings outside the attention module.

---

## Status

- Phase 1 reproduction: **done**
- Phase 2 failure-mode study: **done**
- Phase 3 cause, threshold and tooling: **done**
- Open: thresholds on DocVQA and ChartQA, a third architecture, manuscript

**Not reproducible:** LLaVA-1.6. The authors never released the LLaVA-Next patch.

---

## Citation

```bibtex
@misc{kedar2026btp,
  author = {P. S. Kedar and Divya Saxena and Aditya Sharma},
  title  = {Balanced Token Pruning: Reproduction and Visual Token Access Thresholds},
  year   = {2026},
  note   = {IIT Jodhpur. \url{https://github.com/milab-iitj-dev/btp-reproduction}}
}
```

Subject paper:

```bibtex
@inproceedings{btp2025,
  title     = {Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization},
  booktitle = {NeurIPS},
  year      = {2025},
  note      = {arXiv:2505.22038}
}
```

MIT licensed. See [LICENSE](LICENSE).

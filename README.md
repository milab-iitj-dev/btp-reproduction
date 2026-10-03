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
- **How long a model needs the image depends mainly on the answer format.** Across 8 tasks
  and 3 models the order of tasks by access depth is reproducible (Spearman 0.90 to 0.96),
  and open-ended questions need the image far longer than multiple-choice ones. Turning
  TextVQA into multiple choice moves its depth from layer 23 to 8 on Qwen; asking MMBench
  without options moves it from 10 to 20. A schedule that reads the format from the prompt
  saves 17.5 to 23% of visual layers at 98% accuracy, where a single cut saves under 8%.
  Inside BTP, the same rule keeps the deletion on closed-form tasks with no loss and removes
  it on open-ended ones, recovering reading from 28.4% to 77.9% of baseline.

---

## Repository layout

Every phase folder has the same shape: `README.md` (question, answer, file list),
`scripts/`, `results/`, `figures/` (where a phase has figures), `report/`.

```
BTP-Reproduction/
  RESULTS.md                 every number from all four phases, in one place
  phase1-reproduction/       does BTP reproduce?               yes, within ~1 point
  phase2-text-failure/       where does it break?               reading tasks collapse
  phase3-mechanism/          why?                               a layer-23 deletion; access threshold
    RESULTS_LOG.md           run-by-run record, incl. invalid runs and withdrawn claims
  phase4-access-horizon/     does the needed depth depend on the task, and can pruning use it?
```

| Looking for | Go to |
|---|---|
| A number | `BTP-Reproduction/RESULTS.md` |
| A phase's story and file list | `BTP-Reproduction/phaseN-*/README.md` |
| Reports and decks | `BTP-Reproduction/phaseN-*/report/` |
| Raw scores | `BTP-Reproduction/phaseN-*/results/` (Phases 1 and 2 have an `INDEX.csv`) |
| Job scripts and code | `BTP-Reproduction/phaseN-*/scripts/` |

---

## Phase 3 tooling

- `measure_visual_depth.py` measures the visual token access threshold on any Hugging Face
  vision language model, using forward hooks and no source patching. No training, no
  calibration set. A coarse sweep plus bisection finds the threshold in about seven runs.
- `make_internvl_btp.py` implements BTP-style pruning for InternVL2 by injecting into its
  Hub remote code. The authors released patches for LLaVA and Qwen only.
- `e9` to `e13` job scripts and their `submit_*.sh` chains reproduce every Phase 3 result.
- All of these live in `BTP-Reproduction/phase3-mechanism/scripts/`.

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
- Phase 4 access horizons, format controls and format-aware BTP: **done**
- Open: per-question format detection inside the model, manuscript

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

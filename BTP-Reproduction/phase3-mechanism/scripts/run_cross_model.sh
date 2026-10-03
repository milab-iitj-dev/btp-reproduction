#!/bin/bash
#SBATCH --job-name=btp_crossmodel
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/crossmodel_%j.out
#
# E5 - Does the text collapse happen only on Qwen2.5-VL-7B, or on other models too?
#
# The cheapest decisive test first: Qwen2.5-VL-3B uses the SAME modeling class
# (modeling_qwen2_5_vl.py) as the 7B, so the existing BTP patch works unchanged.
# Only the checkpoint changes. That isolates MODEL SIZE from architecture.
#
# Usage:
#   sbatch run_cross_model.sh
#
# Before running, set MODFILE and the variant paths for your environment.

set -euo pipefail

source ~/.bashrc
conda activate qwen_btp

RESULTS=./results/phase3_crossmodel
mkdir -p "$RESULTS" logs

MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
echo "modeling file: $MODFILE"

# text-heavy task plus the diagram control, kept small so both modes fit in walltime
TASKS="textvqa_val,ai2d"
LIMIT=500

run_one () {
  local MODEL_PATH="$1"; local TAG="$2"; local MODE="$3"   # MODE = baseline | btp
  if [ "$MODE" = "baseline" ]; then
    cp "${MODFILE}.BASELINE" "$MODFILE"
  else
    cp "${MODFILE}.BTP" "$MODFILE"
  fi
  echo "=== $TAG / $MODE : div_prune count = $(grep -c 'def div_prune' "$MODFILE") (0=baseline, 1=BTP) ==="

  accelerate launch -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="$MODEL_PATH" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --limit $LIMIT \
    --log_samples \
    --output_path "${RESULTS}/${TAG}_${MODE}"
}

# ---------------------------------------------------------------- Tier 1: same architecture
# Same modeling file, smaller checkpoint. No code changes required.
run_one "Qwen/Qwen2.5-VL-3B-Instruct" "qwen25vl_3b" "baseline"
run_one "Qwen/Qwen2.5-VL-3B-Instruct" "qwen25vl_3b" "btp"

# Restore baseline so the next job never inherits a pruned file
cp "${MODFILE}.BASELINE" "$MODFILE"

echo "DONE. Compare the TextVQA drop for 3B against the 7B result (82.7 -> 23.6)."
echo "If 3B collapses too, the failure is architectural, not a quirk of the 7B checkpoint."

# ---------------------------------------------------------------- Tier 2 and 3 (not automated)
#
# Tier 2  Qwen2-VL-2B / 7B
#   Different file (modeling_qwen2_vl.py) but nearly the same decoder structure.
#   Port the patch by copying the three pruning blocks and div_prune/attn_prune across.
#   Tests: same family, earlier generation.
#
# Tier 3  The decisive test for hypothesis H4 (resolution vs position scheme)
#   InternVL2-2B      dynamic tiling, 1D positions
#   Phi-3.5-Vision    dynamic crops,  1D positions
#   These are HIGH RESOLUTION but do NOT use Qwen's 2D M-RoPE.
#     - if they collapse too  -> resolution / fine-token dependence drives the failure (H4)
#     - if they do NOT collapse -> Qwen's 2D position handling is implicated (H3)
#   Each needs its own pruning patch in the corresponding modeling file. Reuse the same
#   three-stage schedule and the same budget so results stay comparable.
#
# Keep the schedule identical across every model, otherwise the comparison is meaningless.

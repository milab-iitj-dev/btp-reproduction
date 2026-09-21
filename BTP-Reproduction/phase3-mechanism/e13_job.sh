#!/bin/bash
#SBATCH --job-name=btp_e13
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e13_%j.out
#
# E13 - validate the threshold measurement tool against a curve we already know.
#
# measure_visual_depth.py deletes visual tokens with a forward pre-hook rather than by
# patching model source, so that the same code works on any Hugging Face vision language
# model. That portability is worthless unless the hook reproduces what source patching
# produced, so this job checks it on Qwen2.5-VL-7B, where we already have ground truth from
# the E12 sweep.
#
# GROUND TRUTH, from E12 (source-patched, graded pruning disabled, TextVQA limit 500):
#   baseline (no deletion)  0.8624
#   layer 20                0.1418
#   layer 22                0.1584
#   layer 23                0.2344
#   layer 24                0.6454
#   layer 26                0.7806
#   half-of-baseline crossing at layer 23.5, which is 83.9% depth
#
# PASS CRITERION
#   The hook-based tool should place the crossing within about one layer of 23.5 and should
#   show the same floor-then-cliff shape. Sample sizes differ (200 here against 500 there),
#   so individual points may move by a few points; the threshold should not.
#
# If it disagrees, the hook is mishandling one of position_ids, attention_mask,
# cache_position or position_embeddings, and the tool is not usable until that is found.
#
# USAGE
#   sbatch e13_job.sh                       # validation run, explicit layers
#   MODE=bisect sbatch e13_job.sh           # the cheap path, coarse sweep plus bisection

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

MODE=${MODE:-validate}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
N=${N:-200}

mkdir -p results/depth logs

# The tool must run against the PRISTINE model file. If a previous job died from an OOM
# kill its exit trap never fired, and the BTP variant may still be installed, which would
# silently contaminate this measurement.
MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
if [ -f "${MODFILE}.BASELINE" ]; then
  cp "${MODFILE}.BASELINE" "$MODFILE"
fi
echo "sanity: div_prune count = $(grep -c 'def div_prune' "$MODFILE")   (must be 0)"
if [ "$(grep -c 'def div_prune' "$MODFILE")" -ne 0 ]; then
  echo "FATAL: a BTP variant is installed. The tool must measure the unmodified model."
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "MODE  = $MODE"
echo "MODEL = $MODEL"
echo "N     = $N"

if [ "$MODE" = "validate" ]; then
  # Same layers as the E12 sweep, so the two curves can be compared point by point.
  python -u phase3-mechanism/measure_visual_depth.py \
    --model "$MODEL" \
    --n "$N" \
    --layers 2,8,14,20,22,23,24,26 \
    --no-bisect \
    --out results/depth/qwen7b_validate.json
else
  # The cheap path the paper recommends: quarter depths, then bisect.
  python -u phase3-mechanism/measure_visual_depth.py \
    --model "$MODEL" \
    --n "$N" \
    --out results/depth/qwen7b_bisect.json
fi

RC=$?
echo "=================== DONE rc=$RC ==================="
echo "Compare against E12: crossing at layer 23.5 of 28, which is 83.9% depth."
exit $RC

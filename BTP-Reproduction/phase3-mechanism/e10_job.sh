#!/bin/bash
#SBATCH --job-name=btp_e10
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/e10_%j.out
#
# E10 - cross-family test on InternVL2-2B.
#
# Tests whether the two Qwen findings are properties of BTP's pruning REGIME or artefacts
# of Qwen2.5-VL specifically. InternVL2-2B differs on every axis that could confound:
# InternLM2-1.8B backbone with 24 layers, standard 1D RoPE rather than M-RoPE, and 448px
# dynamic tiling at 256 tokens per tile rather than a merged patch grid.
#
# ARMS (env IVL_MODE, see make_internvl_btp.py)
#   off         no pruning. MUST match the unpatched baseline, this is the control that
#               proves the patch itself is inert.
#   prune       graded pruning at layers 3, 6, 14 down to 12.5%. No final discard.
#   wipe        no graded pruning, delete all image tokens at layer 20. Isolates Finding A.
#   prune_wipe  both. Analogue of BTP as shipped for Qwen.
#
# The decisive comparison is `wipe` vs `off`. If a text task collapses while GQA / MME /
# MMBench / POPE / SQA barely move, Finding A is cross-family and the claim becomes a
# statement about the evaluation protocol rather than about one implementation.
#
# USAGE
#   MODE=wipe  TASKS=textvqa_val LIMIT=500 sbatch e10_job.sh
#   MODE=off   TASKS=textvqa_val LIMIT=500 sbatch e10_job.sh
#   Smoke test first:
#   MODE=prune_wipe TASKS=textvqa_val LIMIT=5 DEBUG=1 sbatch e10_job.sh
#
# NEVER run two of these at once. They share one set of remote-code files.

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

MODE=${MODE:-off}
TASKS=${TASKS:-textvqa_val}
LIMIT=${LIMIT:-500}
DEBUG=${DEBUG:-0}
TAG=${TAG:-$(echo "$TASKS" | tr ',' '-')}

SNAP=$(python - <<'PY'
import os, glob
h = os.environ["HF_HOME"]
c = glob.glob(os.path.join(h, "hub", "models--OpenGVLab--InternVL2-2B", "snapshots", "*"))
if len(c) != 1:
    raise SystemExit("FATAL: expected exactly one snapshot dir, found %d: %s" % (len(c), c))
print(c[0])
PY
)
if [ -z "$SNAP" ]; then
  echo "FATAL: could not resolve the InternVL2-2B snapshot directory"
  exit 1
fi

LM="$SNAP/modeling_internlm2.py"
CHAT="$SNAP/modeling_internvl_chat.py"

echo "SNAP  = $SNAP"
echo "MODE  = $MODE"
echo "TASKS = $TASKS"
echo "LIMIT = $LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p results/phase3 logs

# Build variants if they are not there yet. Safe to call repeatedly: the script refuses to
# overwrite an existing .BASELINE and always patches from the pristine copy.
if [ ! -f "${LM}.BTP" ] || [ ! -f "${CHAT}.BTP" ]; then
  python phase3-mechanism/make_internvl_btp.py --snapshot "$SNAP" || exit 1
fi

# Restore pristine files on the way out, whatever happens. Note an OOM kill is SIGKILL and
# will bypass this, which is why the install step below never assumes a clean starting state.
restore() {
  cp "${LM}.BASELINE" "$LM"
  cp "${CHAT}.BASELINE" "$CHAT"
  rm -rf "$HF_HOME/modules/transformers_modules"
  echo "restored pristine remote code"
}
trap restore EXIT INT TERM

# Install the arm. Deterministic: always copied from .BTP or .BASELINE, never mutated
# in place, so a dirty leftover from a previous killed job cannot leak in.
if [ "$MODE" = "unpatched" ]; then
  cp "${LM}.BASELINE" "$LM"
  cp "${CHAT}.BASELINE" "$CHAT"
else
  cp "${LM}.BTP" "$LM"
  cp "${CHAT}.BTP" "$CHAT"
fi

# transformers copies remote code into modules/transformers_modules at load time and will
# happily serve a stale copy. Clearing it is mandatory between arms.
rm -rf "$HF_HOME/modules/transformers_modules"

# These respect a value passed in on the sbatch command line and otherwise fall back to the
# defaults. sbatch exports the submitting environment, so
#   IVL_WIPE_LAYER=8 MODE=wipe sbatch e10_job.sh
# reaches the model file. Needed for the depth sweep in submit_e11.sh.
export IVL_MODE="$MODE"
export IVL_PRUNE_LAYERS="${IVL_PRUNE_LAYERS:-3,6,14}"
export IVL_WIPE_LAYER="${IVL_WIPE_LAYER:-20}"
export IVL_RETAIN="${IVL_RETAIN:-0.125}"
export IVL_DEBUG="$DEBUG"

echo "sanity: injected marker present = $(grep -c 'BTP-STYLE PRUNING (INJECTED)' "$LM")"
echo "sanity: IVL_MODE=$IVL_MODE IVL_PRUNE_LAYERS=$IVL_PRUNE_LAYERS IVL_WIPE_LAYER=$IVL_WIPE_LAYER"

LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

echo ""
echo "=================== INTERNVL2-2B MODE=$MODE TASKS=$TASKS ==================="

accelerate launch -m lmms_eval \
  --model internvl2 \
  --model_args pretrained=OpenGVLab/InternVL2-2B \
  --tasks "$TASKS" \
  --batch_size 1 \
  $LIMIT_ARG \
  --output_path "results/phase3/e10_${MODE}_${TAG}"

RC=$?
echo "=================== DONE rc=$RC ==================="
exit $RC

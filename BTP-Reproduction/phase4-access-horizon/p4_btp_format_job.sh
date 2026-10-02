#!/bin/bash
#SBATCH --job-name=p4b_fmt
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p4b_%j.out
#
# Step 2 confirmation: BTP with a format-aware final deletion, Qwen2.5-VL-7B.
#
# WHY
#   The released BTP file deletes every remaining image token after layer 22 for every
#   input (self.end_layer = 23; layer_index is 1-based, so 22 layers keep access). That
#   collapses open-ended reading tasks. Disabling it (Phase 3 E9, "corrected BTP") fixes
#   them but gives up the deletion everywhere.
#
#   Phase 4 found that closed-form questions (multiple choice, yes/no) stop needing the
#   image around the middle of the network, while open-ended ones need it to the end.
#   The offline schedule simulation (schedule_sim.py) puts the safe cut for closed-form
#   questions on Qwen2.5-VL-7B at 20 layers of access under a 98% rule.
#
#   A format-aware BTP therefore:
#     open-ended questions   never delete            = corrected BTP (E9), already run
#     closed-form questions  delete after 20 layers  = THIS job, end_layer = 21
#
#   The open-ended arm needs no new GPU time: it is configuration-identical to E9.
#   GQA mixes both formats per question; it stays on the safe open-ended setting (E9).
#
# WHAT THIS TESTS
#   The cut was calibrated with BTP's graded pruning switched off. Here the graded pruning
#   at layers 4, 7, 16 stays on. The question is whether closed-form tasks still lose
#   nothing when the remaining 12.5% of visual tokens is dropped two layers earlier than
#   the released code does.
#
# USAGE
#   TASKS=pope LIMIT=0 END_LAYER=21 sbatch p4_btp_format_job.sh
#   Use submit_p4_btp_format.sh, which chains the closed-form suite.
#
# NEVER run two BTP jobs at once. They swap the same modeling file. Horizon (hook) jobs
# must not run alongside either: they would load the modified file.

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

TASKS=${TASKS:-pope}
LIMIT=${LIMIT:-0}
END_LAYER=${END_LAYER:-21}
TAG=${TAG:-${TASKS}_end${END_LAYER}}

MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")

echo "MODFILE   = $MODFILE"
echo "TASKS     = $TASKS"
echo "LIMIT     = $LIMIT"
echo "END_LAYER = $END_LAYER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p results/phase4_btp logs

restore() {
  cp "${MODFILE}.BASELINE" "$MODFILE"
  echo "restored BASELINE (def div_prune count = $(grep -c 'def div_prune' "$MODFILE"))"
}
trap restore EXIT INT TERM

# Stage layers are 4, 7 and 16. An end_layer equal to one of them would collide with a
# pruning branch (Phase 3 E12), so refuse.
case "$END_LAYER" in
  4|7|16) echo "FATAL: END_LAYER $END_LAYER collides with a pruning stage"; exit 1 ;;
esac

VAR="${MODFILE}.P4FMT"
sed "s|self\.end_layer = 23|self.end_layer = ${END_LAYER}  # Phase 4 format-aware: closed-form questions keep image access for $((END_LAYER - 1)) layers|" \
  "${MODFILE}.BTP" > "$VAR"

if [ "$(grep -c "self.end_layer = ${END_LAYER} " "$VAR")" -ne 1 ]; then
  echo "FATAL: end_layer substitution did not apply"
  grep -n 'self.end_layer' "$VAR"
  exit 1
fi
cp "$VAR" "$MODFILE"
echo "sanity: div_prune count = $(grep -c 'def div_prune' "$MODFILE")"
echo "sanity: live end_layer  = $(grep -m1 'self.end_layer' "$MODFILE")"

LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

echo "=================== BTP_FORMAT end_layer=$END_LAYER : $TASKS ==================="
START=$(date +%s)
accelerate launch -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2 \
  --tasks "$TASKS" \
  --batch_size 1 \
  $LIMIT_ARG \
  --output_path "results/phase4_btp/${TAG}"
RC=$?
echo "=================== DONE rc=$RC  seconds=$(( $(date +%s) - START )) ==================="
exit $RC

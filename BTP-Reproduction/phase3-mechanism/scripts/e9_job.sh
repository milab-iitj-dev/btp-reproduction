#!/bin/bash
#SBATCH --job-name=btp_e9
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/e9_%j.out
#
# E9 - spec-compliant BTP on Qwen2.5-VL-7B, full benchmark suite.
#
# WHY THIS RUN EXISTS
#   The shipped BTP modeling file deletes every remaining image token at layer 23:
#
#       elif layer_index == self.end_layer:        # self.end_layer = 23
#           all_remained_index = cat(remain_sys_index, remain_text_index)
#           hidden_states = hidden_states[:, all_remained_index, :]
#
#   remain_img_index is absent from that concat, so no image token survives layer 23.
#
#   The paper (arXiv:2505.22038v2, Appendix 7.3) specifies total discard in the final
#   stage for the LLaVA family ONLY. For Qwen2.5-VL it specifies the opposite: the final
#   stage retains 12.5% "to preserve model performance". The branch sits inside the
#   our_method path with no model-family guard, so the LLaVA schedule is being applied
#   to Qwen.
#
#   Job 416735 (TextVQA, n=200) showed disabling that one line moves accuracy from
#   0.1725 to 0.7860 against a 0.8565 baseline. This job asks whether that holds across
#   the whole suite, i.e. whether spec-compliant BTP reproduces the paper's claim of
#   roughly 96-98% of original average at 22-25% retention.
#
#   This is NOT a new method. We remove a line that contradicts the paper's own
#   appendix for this model. Nothing is invented.
#
# WHAT THIS RUN DOES NOT DO
#   No baseline arm, no shipped-BTP arm. Those numbers already exist from Phase 1 and
#   Phase 2 and are reused. Only the fixed arm is new, which keeps GPU cost down.
#   Match the LIMIT to whatever the existing arm used, or the comparison is invalid.
#
# USAGE
#   Text set, to compare against Phase 2 exp1 which used limit 500:
#     TASKS=textvqa_val,docvqa_val,chartqa,ai2d LIMIT=500 sbatch e9_job.sh
#
#   Paper Table 1 set, to compare against Phase 1 which ran full:
#     TASKS=gqa            LIMIT=0 sbatch e9_job.sh
#     TASKS=mme            LIMIT=0 sbatch e9_job.sh
#     TASKS=mmbench_en_dev LIMIT=0 sbatch e9_job.sh
#     TASKS=pope           LIMIT=0 sbatch e9_job.sh
#     TASKS=scienceqa_img  LIMIT=0 sbatch e9_job.sh
#
#   LIMIT=0 means no limit. Full Qwen sweeps have hit the 10h walltime before, which is
#   why the paper set is submitted one task per job rather than as one loop.
#
#   NEVER run two of these at once. Every BTP job swaps the same modeling file.
#   Use submit_e9.sh, which chains them with --dependency=afterany.

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

TASKS=${TASKS:-textvqa_val}
LIMIT=${LIMIT:-500}
TAG=${TAG:-$(echo "$TASKS" | tr ',' '-')}

MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")

echo "MODFILE = $MODFILE"
echo "TASKS   = $TASKS"
echo "LIMIT   = $LIMIT"
echo "TAG     = $TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p results/phase3 logs

# Always put the file back, even if lmms-eval dies or the job is cancelled.
restore() {
  cp "${MODFILE}.BASELINE" "$MODFILE"
  echo "restored BASELINE (def div_prune count = $(grep -c 'def div_prune' "$MODFILE"))"
}
trap restore EXIT INT TERM

# ---------------------------------------------------------------------------
# Build the spec-compliant variant from the shipped BTP file.
#
# self.end_layer = -1 rather than 999. A layer index is never negative, so the branch
# can never fire, and it does not silently depend on the model having fewer than 999
# layers. The comment carries the justification so the next reader does not have to
# rediscover it.
# ---------------------------------------------------------------------------
FIXED="${MODFILE}.FIXED"

sed 's|self\.end_layer = 23|self.end_layer = -1  # disabled: paper Appendix 7.3 specifies a 12.5% retention final stage for Qwen2.5-VL; total discard is the LLaVA schedule and has no model-family guard here|' \
  "${MODFILE}.BTP" > "$FIXED"

# Refuse to run on a variant that was not actually modified. A silent sed miss would
# produce a "fixed" arm identical to the shipped one and we would draw a false
# conclusion from it.
if [ "$(grep -c 'self.end_layer = -1' "$FIXED")" -ne 1 ]; then
  echo "FATAL: end_layer substitution did not apply. Pattern changed in the source file."
  echo "found: $(grep -n 'self.end_layer' "$FIXED")"
  exit 1
fi
if grep -q 'self\.end_layer = 23' "$FIXED"; then
  echo "FATAL: an unmodified 'self.end_layer = 23' still present in the variant."
  exit 1
fi

echo "variant built OK:"
grep -n 'self.end_layer' "$FIXED"

cp "$FIXED" "$MODFILE"

# Confirm the live file is BTP and not baseline. div_prune only exists in the BTP file.
echo "sanity: live file div_prune count = $(grep -c 'def div_prune' "$MODFILE")"
echo "sanity: live file end_layer       = $(grep -m1 'self.end_layer' "$MODFILE")"

# ---------------------------------------------------------------------------
# Evaluate.
#
# attn_implementation=flash_attention_2 is MANDATORY. Without it lmms-eval loads SDPA,
# which does not support output_attentions, so pre_layer_atten comes back None and BTP
# crashes at last_token_attn = pre_layer_atten[:,:,-1,:]. This cost us job 411389.
#
# python -u equivalent: lmms-eval already flushes, but accelerate buffers, so keep the
# per-task output paths distinct to make partial results recoverable if walltime hits.
# ---------------------------------------------------------------------------
LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

echo ""
echo "=================== BTP_FIXED : $TASKS ==================="

accelerate launch -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2 \
  --tasks "$TASKS" \
  --batch_size 1 \
  $LIMIT_ARG \
  --output_path "results/phase3/e9_fixed_${TAG}"

RC=$?
echo "=================== DONE rc=$RC ==================="

echo ""
echo "E9 complete for: $TASKS"
echo "Compare against:"
echo "  baseline      Phase 1 / Phase 2 exp1 results"
echo "  BTP shipped   Phase 1 / Phase 2 exp1 results"
echo "Expectation if the paper is right: fixed arm lands within a few points of baseline."
echo "Reference point already measured: TextVQA n=200 baseline 0.8565, shipped 0.1725, fixed 0.7860."

exit $RC

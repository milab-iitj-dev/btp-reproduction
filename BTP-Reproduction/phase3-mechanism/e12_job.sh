#!/bin/bash
#SBATCH --job-name=btp_e12
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/e12_%j.out
#
# E12 - visual information depth sweep on Qwen2.5-VL-7B.
#
# The companion to E11, which sweeps the same thing on InternVL2-2B. Together the two give
# one figure: where does each architecture stop needing to look at the image?
#
# ISOLATING THE WIPE
#   A clean depth sweep must vary ONE thing, the layer at which all image tokens are
#   deleted. The shipped BTP file also prunes at layers 4, 7 and 16, which would confound
#   the curve. So this job builds its variant from ${MODFILE}.BTP_PARAM and runs it with
#   BTP_RETAIN=1.0, which keeps every token at those three stages.
#
#   Job 416734 already verified this configuration: at retention 1.00 nothing is pruned
#   upstream, yet TextVQA still scored 0.178, because the layer-23 deletion fired anyway.
#   That is exactly the isolated wipe we want, and it matches InternVL's `wipe` arm.
#
# USAGE
#   END_LAYER=16 sbatch e12_job.sh
#   END_LAYER=-1 sbatch e12_job.sh      # control: no wipe at all, should match baseline
#
#   END_LAYER=-1 disables the branch, since a layer index is never negative.
#
# NEVER run two BTP jobs at once. They share one modeling file. Chain with a dependency.

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

END_LAYER=${END_LAYER:--1}
TASKS=${TASKS:-textvqa_val}
LIMIT=${LIMIT:-500}
RETAIN=${RETAIN:-1.0}
TAG=${TAG:-endL${END_LAYER}}

MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")

echo "MODFILE   = $MODFILE"
echo "END_LAYER = $END_LAYER"
echo "RETAIN    = $RETAIN"
echo "TASKS     = $TASKS"
echo "LIMIT     = $LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p results/phase3 logs

# BTP_PARAM is the retention-parameterised variant built in Phase 2. Without it this sweep
# cannot isolate the wipe, so fail loudly rather than silently falling back to .BTP.
if [ ! -f "${MODFILE}.BTP_PARAM" ]; then
  echo "FATAL: ${MODFILE}.BTP_PARAM not found."
  echo "Rebuild it with make_param_btp.py before running this sweep."
  exit 1
fi

restore() {
  cp "${MODFILE}.BASELINE" "$MODFILE"
  echo "restored BASELINE (def div_prune count = $(grep -c 'def div_prune' "$MODFILE"))"
}
trap restore EXIT INT TERM

# An OOM kill is SIGKILL and bypasses the trap above, so never assume the live file is
# clean on entry. Everything below rebuilds from a pristine source.
SWEEP="${MODFILE}.SWEEP"

# ---------------------------------------------------------------------------
# Disable the three graded pruning stages and keep ONLY the wipe.
#
# Two reasons, both learned the hard way on the first attempt:
#
#   1. COLLISION. The wipe lives in an `elif layer_index == self.end_layer:` arm that comes
#      AFTER the pruning-stage branches. Setting end_layer to 4, 7 or 16 means the pruning
#      branch matches first and the wipe silently never fires. Jobs 419015 and 419018
#      returned exactly the control score for that reason, so those points were not
#      evidence of anything.
#
#   2. CRASH. Wiping before layer 16 removes every image token, and the stage at layer 16
#      then tries to prune image tokens that no longer exist. The empty range() yields a
#      float tensor, and indexing with it raises
#        IndexError: tensors used as indices must be long, int, byte or bool tensors
#      That killed jobs 419016 and 419017 at layers 8 and 12.
#
# Disabling the stages fixes both, makes the arm an exact analogue of InternVL's `wipe`
# mode, and runs roughly five times faster because div_prune is no longer called at k = n.
# ---------------------------------------------------------------------------
# A third problem appeared once the stages were disabled: self.hidden_length is ONLY ever
# assigned inside those three pruning branches, and the wipe needs it to compute
#   img_end = self.img_start_idx + self.hidden_length
# With the stages off it is never set, so every wipe job died in under a minute.
#
# self.model.img_num is assigned outside the decoder loop on every forward pass and holds
# the full image-token count, so we seed hidden_length from it. When the stages DO run they
# overwrite it exactly as before, so this is inert in every other configuration.
sed -e "s|self\.start_layer = 4|self.start_layer = -1        # E12: graded pruning disabled|" \
    -e "s|self\.img_sense_layer = 7|self.img_sense_layer = -1    # E12: graded pruning disabled|" \
    -e "s|self\.rel_start_layer = 16|self.rel_start_layer = -1    # E12: graded pruning disabled|" \
    -e "s|self\.end_layer = 23|self.end_layer = ${END_LAYER}        # E12 depth sweep|" \
    -e "s|self\.model\.img_num = image_embeds\.shape\[0\]|&\n                self.model.hidden_length = image_embeds.shape[0]  # E12: stages are off, so seed it here|" \
    "${MODFILE}.BTP_PARAM" > "$SWEEP"

for PAT in "self.start_layer = -1" "self.img_sense_layer = -1" "self.rel_start_layer = -1" \
           "self.end_layer = ${END_LAYER}" "self.model.hidden_length = image_embeds.shape[0]"; do
  # -F is required: one of these patterns contains [0], which grep would otherwise read as
  # a regex character class and never match. That cost a whole sweep.
  if [ "$(grep -cF -- "$PAT" "$SWEEP")" -ne 1 ]; then
    echo "FATAL: substitution failed for: $PAT"
    echo "found: $(grep -n '_layer = ' "$SWEEP" | head)"
    exit 1
  fi
done

echo "variant built OK:"
grep -n 'start_layer\|img_sense_layer\|rel_start_layer\|end_layer' "$SWEEP" | head -5

cp "$SWEEP" "$MODFILE"

echo "sanity: div_prune count = $(grep -c 'def div_prune' "$MODFILE")"

# BTP_PARAM reads the retention ratio at MODULE IMPORT, so this must be exported on its own
# line before the launcher. Setting it as a command prefix, or inside python, is too late
# and the sweep would silently run at the default 0.125.
export BTP_RETAIN="$RETAIN"
echo "sanity: BTP_RETAIN=$BTP_RETAIN"

LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

echo ""
echo "=================== QWEN DEPTH SWEEP end_layer=$END_LAYER ==================="

accelerate launch -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2 \
  --tasks "$TASKS" \
  --batch_size 1 \
  $LIMIT_ARG \
  --output_path "results/phase3/e12_${TAG}_${TASKS}"

RC=$?
echo "=================== DONE rc=$RC ==================="
echo "Reference: unpruned TextVQA baseline at limit 500 = 0.862"
exit $RC

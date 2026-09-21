#!/bin/bash
#SBATCH --job-name=btp_e1
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e1_%j.out
#
# E1 - selector ablation. The causal test.
#
# Budget is IDENTICAL across modes (BTP's shipped 12.5%). Only the selection rule changes:
#   btp    attention seed + diversity fill  (control, should reproduce ~23)
#   div    100% diversity
#   attn   100% attention
#   random uniform random
#
# If `attn` recovers a large part of the gap, the selection objective is causal.
# If all four sit near 23, selection is NOT the cause and information loss is.
#
# Modes run SEQUENTIALLY on purpose: they all swap the same modeling file.

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

LIMIT=${LIMIT:-200}
MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
echo "MODFILE=$MODFILE"
echo "LIMIT=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p variants results/phase3

python patch_selector_variants.py \
  --src ${MODFILE}.BTP \
  --out-dir variants \
  --modes btp,div,attn,random
if [ $? -ne 0 ]; then
  echo "FATAL: variant generation failed"
  cp ${MODFILE}.BASELINE $MODFILE
  exit 1
fi
ls -l variants/

for MODE in btp div attn random; do
  echo ""
  echo "=================== MODE=$MODE ==================="
  cp variants/modeling_qwen2_5_vl.py.SEL_${MODE} $MODFILE
  echo "sanity: BTP_SELECT_MODE lines = $(grep -c 'BTP_SELECT_MODE' $MODFILE)"
  echo "sanity: mode string = $(grep -m1 'BTP_SELECT_MODE = ' $MODFILE)"

  accelerate launch -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks textvqa_val \
    --batch_size 1 \
    --limit $LIMIT \
    --output_path results/phase3/e1_${MODE}

  echo "=================== MODE=$MODE DONE ==============="
done

cp ${MODFILE}.BASELINE $MODFILE
echo ""
echo "restored baseline: div_prune count = $(grep -c 'def div_prune' $MODFILE)"
echo "E1 COMPLETE. Compare exact_match across the four modes."

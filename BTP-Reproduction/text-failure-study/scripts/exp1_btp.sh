#!/bin/bash
#SBATCH --job-name=exp1_btp
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
export HF_HOME=/scratch/data/divyasaxena_rs/kedar_BTPreproduction/hf_cache

QMOD=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
cp ${QMOD}.BTP $QMOD
echo "MODE: div_prune=$(grep -c 'def div_prune' $QMOD) (expect 1) | EXP1 baseline | Qwen"

for TASK in textvqa_val docvqa_val chartqa ai2d; do
  echo "=== START $TASK $(date) ==="
  accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="Qwen/Qwen2.5-VL-7B-Instruct",attn_implementation="flash_attention_2" \
    --tasks $TASK --batch_size 1 --limit 500 \
    --log_samples --log_samples_suffix exp1_btp \
    --output_path ./results/exp1/ || echo "=== FAILED $TASK ==="
  echo "=== END $TASK $(date) ==="
done

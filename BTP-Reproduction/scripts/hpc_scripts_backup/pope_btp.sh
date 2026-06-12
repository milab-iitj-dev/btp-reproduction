#!/bin/bash
#SBATCH --job-name=pope_btp200
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -e
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/llava_btp
export HF_HOME=/scratch/data/divyasaxena_rs/kedar_BTPreproduction/hf_cache

MODFILE=$(python -c "import transformers.models.llama.modeling_llama as m; print(m.__file__)")
echo "MODE: div_prune=$(grep -c 'def div_prune' $MODFILE) (0=baseline 1=BTP)"
python -c "import torch, transformers, flash_attn; print('torch', torch.__version__, '| tf', transformers.__version__, '| fa', flash_attn.__version__)"

accelerate launch --num_processes=1 -m lmms_eval \
  --model llava \
  --model_args pretrained="liuhaotian/llava-v1.5-7b",attn_implementation="flash_attention_2" \
  --tasks pope \
  --batch_size 1 \
  --limit 200 \
  --log_samples \
  --log_samples_suffix btp_fa2 \
  --output_path ./results/lmms_logs/

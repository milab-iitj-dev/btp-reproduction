#!/bin/bash
#SBATCH --job-name=exp2_full
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
export HF_HOME=/scratch/data/divyasaxena_rs/kedar_BTPreproduction/hf_cache
QMOD=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
cp ${QMOD}.BTP_PARAM $QMOD
echo "param active: _BTP_F=$(grep -c '_BTP_F' $QMOD)"

run_eval () {
  accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="Qwen/Qwen2.5-VL-7B-Instruct",attn_implementation="flash_attention_2" \
    --tasks $1 --batch_size 1 --limit $3 \
    --log_samples --log_samples_suffix $2 \
    --output_path ./results/exp2/ || echo "=== FAILED $1 $2 ==="
}

# cliff region first (most valuable), then floor
for R in 0.90 0.75 0.60 0.25 0.20 0.125; do
  export BTP_RETAIN=$R
  PCT=$(python -c "print(int(float('$R')*1000))")
  echo "===== RETAIN=$R ====="
  echo "=== textvqa_val retain$PCT $(date) ==="; run_eval textvqa_val retain$PCT 500
  echo "=== docvqa_val retain$PCT $(date) ==="; run_eval docvqa_val retain$PCT 250
done

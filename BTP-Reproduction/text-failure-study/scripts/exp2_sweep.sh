#!/bin/bash
#SBATCH --job-name=exp2_sweep
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

run_eval () {  # $1=task  $2=suffix
  accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="Qwen/Qwen2.5-VL-7B-Instruct",attn_implementation="flash_attention_2" \
    --tasks $1 --batch_size 1 --limit 500 \
    --log_samples --log_samples_suffix $2 \
    --output_path ./results/exp2/ || echo "=== FAILED $1 $2 ==="
}

mkdir -p results/exp2

# 100% = baseline (no pruning) — use BASELINE file
cp ${QMOD}.BASELINE $QMOD
echo "===== RETAIN=100% (baseline) div_prune=$(grep -c 'def div_prune' $QMOD) ====="
for T in textvqa_val docvqa_val; do echo "=== $T retain100 $(date) ==="; run_eval $T retain100; done

# pruned levels — use parameterized BTP file
cp ${QMOD}.BTP_PARAM $QMOD
echo "param file active: _BTP_F=$(grep -c '_BTP_F' $QMOD)"
for R in 0.50 0.40 0.30 0.25 0.20 0.125; do
  export BTP_RETAIN=$R
  PCT=$(python -c "print(int(float('$R')*1000))")
  echo "===== RETAIN=$R ====="
  for T in textvqa_val docvqa_val; do echo "=== $T retain$PCT $(date) ==="; run_eval $T retain$PCT; done
done

#!/bin/bash
#SBATCH --job-name=p4_pilot
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --exclude=cn22,cn25
#SBATCH --output=logs/p4_pilot_%j.out
#
# Phase 4 pilot. Full deletion-depth curve, every layer 0..28, on Qwen2.5-VL-7B.
#
# Purpose of the pilot, in order:
#   1. Cost. seconds per layer is logged, so the full eight-task plan can be sized.
#   2. The floor. l = 0 means no layer saw the image. POPE should sit well above zero
#      there; that is why the analysis normalises by (baseline - floor).
#   3. First comparison of horizons between an object-presence task (POPE) and a reading
#      task (TextVQA). Phase 3 costs predict POPE early and TextVQA late.
#
# PASS CRITERIA
#   TextVQA should reproduce job 423959: floor-then-cliff, half crossing near l = 22.5.
#   If it does not, the pilot is void, whatever POPE shows.
#
# The run is resumable. If it hits the walltime, resubmit the same command and it continues
# from the last completed layer. It refuses to resume onto a different sample set.
#
# USAGE
#   TASK=textvqa N=200 sbatch phase4-access-horizon/p4_pilot_job.sh
#   TASK=pope    N=300 sbatch phase4-access-horizon/p4_pilot_job.sh
#   or: bash phase4-access-horizon/submit_p4_pilot.sh

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

TASK=${TASK:?set TASK=textvqa or TASK=pope}
N=${N:-200}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
TAG=${TAG:-qwen7b}
OUT=results/horizon/${TAG}_${TASK}_n${N}.json

mkdir -p results/horizon logs

# Measure the UNMODIFIED model. An OOM-killed BTP job can leave its variant installed,
# because SIGKILL skips the exit trap. Restore first, then verify.
MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
if [ -f "${MODFILE}.BASELINE" ]; then
  cp "${MODFILE}.BASELINE" "$MODFILE"
fi
NDIV=$(grep -c 'def div_prune' "$MODFILE")
echo "sanity: div_prune count = $NDIV   (must be 0)"
if [ "$NDIV" -ne 0 ]; then
  echo "FATAL: a BTP variant is installed. This must measure the unmodified model."
  exit 1
fi

for f in phase3-mechanism/measure_visual_depth.py phase4-access-horizon/measure_access_horizon.py; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# cn22 and cn25 split each A30 into four MIG slices of 6 GB (gres gpu:12 = 3 cards x 4).
# SLURM hands out one slice, Qwen 7B needs ~16 GB, so loading OOMs (jobs 437302, 437351).
# nvidia-smi reports the whole card there, which is misleading. They are excluded above;
# this check catches any other MIG node before a wasted model load.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi -L
if nvidia-smi -L | grep -q "MIG"; then
  echo "FATAL: MIG slice allocated on $(hostname). Too small for a 7B model. Exclude this node."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "TASK  = $TASK"
echo "N     = $N"
echo "MODEL = $MODEL"
echo "OUT   = $OUT"
date

python -u phase4-access-horizon/measure_access_horizon.py \
  --model "$MODEL" \
  --task "$TASK" \
  --n "$N" \
  --layers all \
  --out "$OUT"
RC=$?

date
echo "=================== DONE rc=$RC ==================="
exit $RC

#!/bin/bash
#SBATCH --job-name=p4_ivl
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
# 2 CPUs is enough for single-sample GPU inference on a 2B model. The fat nodes are
# usually CPU-bound (6 idle CPUs across cn23/26/27 on 26 Sep), so asking for 8 kept these
# jobs pending with free GPUs available.
#SBATCH --time=10:00:00
#SBATCH --exclude=cn22,cn25
#SBATCH --output=logs/p4_ivl_%j.out
#
# Phase 4 on InternVL2-2B: deletion-depth curve, every layer 0..24, one task per job.
# Cross-FAMILY test of the task ordering found on Qwen2.5-VL-7B and 3B.
#
# VALIDATION BUILT IN
#   TextVQA must reproduce E11 (source-patched sweep): half-of-baseline crossing near
#   17.95 layers with access, 74.8% depth. The hook path has never run on InternVL before,
#   so if TextVQA disagrees, no other InternVL task is trusted.
#
# PRISTINE REMOTE CODE
#   E10/E11 patched the Hub snapshot in place. This job measures the unmodified model, so
#   it restores .BASELINE if a patched file is found and refuses to run if it cannot.
#   Each job uses its own transformers module cache (HF_MODULES_CACHE), so parallel jobs
#   never delete a cache another job is importing from.
#
# USAGE
#   TASK=textvqa N=200 sbatch phase4-access-horizon/p4_ivl_job.sh
#   or: bash phase4-access-horizon/submit_p4_ivl.sh

set -uo pipefail

source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache
export HF_MODULES_CACHE=$PWD/hf_modules_p4/${SLURM_JOB_ID:-manual}
mkdir -p "$HF_MODULES_CACHE"
trap 'rm -rf "$HF_MODULES_CACHE"' EXIT

TASK=${TASK:?set TASK}
N=${N:-200}
MODEL=${MODEL:-OpenGVLab/InternVL2-2B}
TAG=${TAG:-ivl2b}
export IVL_MAX_NUM=${IVL_MAX_NUM:-6}
OUT=results/horizon/${TAG}_${TASK}_n${N}.json
mkdir -p results/horizon logs

SNAP=$(python - <<'PY'
import os, glob
c = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--OpenGVLab--InternVL2-2B", "snapshots", "*"))
print(c[0] if len(c) == 1 else "")
PY
)
if [ -n "$SNAP" ]; then
  for f in modeling_internlm2.py modeling_internvl_chat.py; do
    if grep -q "BTP-STYLE PRUNING (INJECTED)\|_BTP_MAX_POS" "$SNAP/$f" 2>/dev/null; then
      if [ -f "$SNAP/$f.BASELINE" ]; then
        cp "$SNAP/$f.BASELINE" "$SNAP/$f"
        echo "restored pristine $f from .BASELINE"
      else
        echo "FATAL: $f is patched and no .BASELINE exists"; exit 1
      fi
    fi
  done
  NPATCH=$(cat "$SNAP/modeling_internlm2.py" "$SNAP/modeling_internvl_chat.py" | grep -c "_BTP_MAX_POS\|BTP-STYLE PRUNING")
  echo "sanity: BTP markers in remote code = $NPATCH   (must be 0)"
  [ "$NPATCH" -eq 0 ] || { echo "FATAL: remote code still patched"; exit 1; }
else
  echo "no local snapshot yet, it will be downloaded pristine"
fi

for f in phase3-mechanism/measure_visual_depth.py phase4-access-horizon/measure_access_horizon.py \
         phase4-access-horizon/internvl_adapter.py; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
if nvidia-smi -L | grep -q "MIG"; then
  echo "FATAL: MIG slice allocated on $(hostname). Exclude this node."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "TASK = $TASK   N = $N   MODEL = $MODEL   IVL_MAX_NUM = $IVL_MAX_NUM"
echo "OUT  = $OUT"
date

python -u phase4-access-horizon/measure_access_horizon.py \
  --model "$MODEL" --task "$TASK" --n "$N" --layers all --out "$OUT"
RC=$?
date
echo "=================== DONE rc=$RC ==================="
exit $RC

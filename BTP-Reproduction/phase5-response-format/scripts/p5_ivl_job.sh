#!/bin/bash
#SBATCH --job-name=p5_ivl
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/p5_ivl_%j.out
#
# Phase 5 on InternVL2-2B: one response condition, full withdrawal curve (0..24) plus a
# no-image control. Answers only (no confidence, no logit lens).
#   COND=open sbatch phase5-response-format/scripts/p5_ivl_job.sh

set -uo pipefail
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache
export HF_MODULES_CACHE=$PWD/hf_modules_p5/${SLURM_JOB_ID:-manual}
mkdir -p "$HF_MODULES_CACHE"
trap 'rm -rf "$HF_MODULES_CACHE"' EXIT

COND=${COND:?set COND}
N=${N:-200}
MODEL=${MODEL:-OpenGVLab/InternVL2-2B}
export IVL_MAX_NUM=${IVL_MAX_NUM:-6}
OUT=results/phase5/ivl2b_textvqa_${COND}.json
mkdir -p results/phase5 logs

# The Phase 3 BTP injection edits the cached remote code; make sure it is pristine.
SNAP=$(python - <<'PY'
import os, glob
c = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--OpenGVLab--InternVL2-2B", "snapshots", "*"))
print(c[0] if len(c) == 1 else "")
PY
)
[ -n "$SNAP" ] || { echo "FATAL: no InternVL2-2B snapshot"; exit 1; }
for f in modeling_internlm2.py modeling_internvl_chat.py; do
  if grep -q "BTP-STYLE PRUNING (INJECTED)\|_BTP_MAX_POS" "$SNAP/$f" 2>/dev/null; then
    if [ -f "$SNAP/$f.BASELINE" ]; then
      cp "$SNAP/$f.BASELINE" "$SNAP/$f"; echo "restored pristine $f"
    else
      echo "FATAL: $f is patched and no .BASELINE exists"; exit 1
    fi
  fi
done
NPATCH=$(cat "$SNAP/modeling_internlm2.py" "$SNAP/modeling_internvl_chat.py" | grep -c "_BTP_MAX_POS\|BTP-STYLE PRUNING")
echo "sanity: BTP markers in remote code = $NPATCH   (must be 0)"
[ "$NPATCH" -eq 0 ] || { echo "FATAL: remote code still patched"; exit 1; }

if nvidia-smi -L | grep -q "MIG"; then
  echo "FATAL: MIG slice on $(hostname)"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "COND = $COND   N = $N   MODEL = $MODEL   OUT = $OUT"
date
python -u phase5-response-format/scripts/measure_response_modes.py \
  --model "$MODEL" --cond "$COND" --n "$N" --out "$OUT"
RC=$?
date
echo "=================== DONE rc=$RC ==================="
exit $RC

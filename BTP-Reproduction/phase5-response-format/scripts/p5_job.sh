#!/bin/bash
#SBATCH --job-name=p5
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --exclude=cn22,cn25
#SBATCH --output=logs/p5_%j.out
#
# Phase 5, one response condition on Qwen2.5-VL-7B + TextVQA: full withdrawal curve
# (every layer 0..28) with confidence, plus a no-image control and a logit-lens pass.
#   COND=mc4 sbatch phase5-response-format/scripts/p5_job.sh
# Conditions: open fill mc2 mc4 mc8 mc4_type yesno. Use submit_p5.sh for all of them.
# Hook-based deletion on the UNMODIFIED model, so it may run alongside other horizon jobs,
# but never alongside a BTP job (those swap the modeling file).

set -uo pipefail
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache

COND=${COND:?set COND}
N=${N:-200}
OUT=results/phase5/qwen7b_textvqa_${COND}.json
mkdir -p results/phase5 logs

MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
if [ -f "${MODFILE}.BASELINE" ]; then
  cp "${MODFILE}.BASELINE" "$MODFILE"
fi
NDIV=$(grep -c 'def div_prune' "$MODFILE")
echo "sanity: div_prune count = $NDIV   (must be 0)"
[ "$NDIV" -eq 0 ] || { echo "FATAL: a BTP variant is installed"; exit 1; }
if nvidia-smi -L | grep -q "MIG"; then
  echo "FATAL: MIG slice on $(hostname)"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "COND = $COND   N = $N   OUT = $OUT"
date
python -u phase5-response-format/scripts/measure_response_modes.py --cond "$COND" --n "$N" --out "$OUT"
RC=$?
date
echo "=================== DONE rc=$RC ==================="
exit $RC

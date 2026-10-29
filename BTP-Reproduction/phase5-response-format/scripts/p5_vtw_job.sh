#!/bin/bash
#SBATCH --job-name=p5_vtw
#SBATCH --partition=dgx_fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/p5_vtw_%j.out
#
# VTW's KL search for the withdrawal layer, on Qwen2.5-VL-7B and every Phase 4/5 task.
#   sbatch phase5-response-format/scripts/p5_vtw_job.sh

set -uo pipefail
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/data/divyasaxena_rs/kedar_BTPreproduction/envs/qwen_btp
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
export HF_HOME=$PWD/hf_cache
mkdir -p results/phase5 logs

MODFILE=$(python -c "import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m; print(m.__file__)")
[ -f "${MODFILE}.BASELINE" ] && cp "${MODFILE}.BASELINE" "$MODFILE"
NDIV=$(grep -c 'def div_prune' "$MODFILE")
echo "sanity: div_prune count = $NDIV   (must be 0)"
[ "$NDIV" -eq 0 ] || { echo "FATAL: a BTP variant is installed"; exit 1; }
if nvidia-smi -L | grep -q "MIG"; then
  echo "FATAL: MIG slice on $(hostname)"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
date
python -u phase5-response-format/scripts/vtw_calibrate.py --out results/phase5/vtw_qwen7b.json
RC=$?
date
echo "=================== DONE rc=$RC ==================="
exit $RC

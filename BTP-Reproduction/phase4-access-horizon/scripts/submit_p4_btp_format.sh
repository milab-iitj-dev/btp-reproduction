#!/bin/bash
# Format-aware BTP, closed-form suite, Qwen2.5-VL-7B. One chained job per task, because
# every BTP job swaps the same modeling file. Limits match the earlier released and
# corrected arms: AI2D 500 (Phase 2), the paper set full (Phase 1).
#
# Run from the workspace root:  bash phase4-access-horizon/scripts/submit_p4_btp_format.sh

set -uo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

RUNS=("ai2d:500" "scienceqa_img:0" "mmbench_en_dev:0" "pope:0" "mme:0")
PREV=""
for R in "${RUNS[@]}"; do
  T="${R%%:*}"; LIM="${R##*:}"
  DEP=""
  [ -n "$PREV" ] && DEP="--dependency=afterany:${PREV}"
  ID=$(TASKS="$T" LIMIT="$LIM" END_LAYER=21 sbatch --parsable $DEP phase4-access-horizon/scripts/p4_btp_format_job.sh)
  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then echo "FAILED to submit $T: $ID"; exit 1; fi
  echo "queued $ID  task=$T  limit=$LIM  after=${PREV:-none}"
  PREV="$ID"
done

#!/bin/bash
#
# Submit the full E9 sweep as a CHAIN.
#
# Every BTP job rewrites the same modeling_qwen2_5_vl.py inside the conda env. Two jobs
# running at once will corrupt each other's arm and produce numbers that look plausible
# and are wrong. So each job waits for the previous one with --dependency=afterany.
# afterany, not afterok, so a single task failing does not strand the rest of the queue.
#
# Run from the workspace root:
#   bash phase3-mechanism/submit_e9.sh
#
# Then watch:
#   squeue -u $USER
#   tail -f logs/e9_<jobid>.out

set -uo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

JOB="phase3-mechanism/e9_job.sh"

# task:limit pairs. Limit must match whatever the existing baseline and shipped arms
# used, otherwise the comparison is meaningless.
#   text set  -> 500, matching Phase 2 exp1
#   paper set -> 0 (full), matching Phase 1
RUNS=(
  "textvqa_val:500"
  "docvqa_val:500"
  "chartqa:500"
  "ai2d:500"
  "pope:0"
  "mme:0"
  "mmbench_en_dev:0"
  "gqa:0"
  "scienceqa_img:0"
)

PREV=""
for R in "${RUNS[@]}"; do
  TASK="${R%%:*}"
  LIM="${R##*:}"

  DEP=""
  if [ -n "$PREV" ]; then
    DEP="--dependency=afterany:${PREV}"
  fi

  OUT=$(TASKS="$TASK" LIMIT="$LIM" TAG="$TASK" sbatch $DEP "$JOB")
  ID=$(echo "$OUT" | awk '{print $NF}')

  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
    echo "FAILED to submit $TASK: $OUT"
    exit 1
  fi

  echo "queued $ID  task=$TASK  limit=$LIM  after=${PREV:-none}"
  PREV="$ID"
done

echo ""
echo "Chain submitted. Last job id: $PREV"
echo "Text set finishes first, so TextVQA / DocVQA / ChartQA / AI2D results arrive before the paper set."

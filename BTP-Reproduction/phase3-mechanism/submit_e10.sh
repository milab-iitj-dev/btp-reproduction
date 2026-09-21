#!/bin/bash
#
# Submit the InternVL2-2B cross-family sweep as a dependency chain.
#
# All arms rewrite the same remote-code files, so nothing may run concurrently.
#
# STAGE A (default): the decisive minimum, 9 jobs.
#   One text-reading task (TextVQA) and one paper-suite task (MMBench-EN), across all arms.
#   If Finding A is cross-family, TextVQA collapses under `wipe` while MMBench barely moves.
#   `off` vs `unpatched` on TextVQA is the inertness control: they MUST agree, otherwise the
#   patch itself is doing something and every other number is suspect.
#
# STAGE B: widen to the remaining benchmarks once Stage A confirms the pattern.
#
# Usage:
#   bash phase3-mechanism/submit_e10.sh A
#   bash phase3-mechanism/submit_e10.sh B
#
# Limits are held at 500 throughout. These numbers are compared only against each other,
# across arms of the same model, so a consistent subset is sufficient and much cheaper than
# full runs. Do NOT compare them to the Qwen table, which used different limits per task.

set -uo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

STAGE=${1:-A}
JOB="phase3-mechanism/e10_job.sh"
LIM=500

if [ "$STAGE" = "A" ]; then
  RUNS=(
    "unpatched:textvqa_val"
    "off:textvqa_val"
    "prune:textvqa_val"
    "wipe:textvqa_val"
    "prune_wipe:textvqa_val"
    "unpatched:mmbench_en_dev"
    "prune:mmbench_en_dev"
    "wipe:mmbench_en_dev"
    "prune_wipe:mmbench_en_dev"
  )
elif [ "$STAGE" = "B" ]; then
  RUNS=(
    "unpatched:docvqa_val"
    "wipe:docvqa_val"
    "prune:docvqa_val"
    "prune_wipe:docvqa_val"
    "unpatched:chartqa"
    "wipe:chartqa"
    "prune:chartqa"
    "prune_wipe:chartqa"
    "unpatched:ai2d"
    "wipe:ai2d"
    "prune:ai2d"
    "prune_wipe:ai2d"
    "unpatched:pope"
    "wipe:pope"
    "prune:pope"
    "prune_wipe:pope"
    "unpatched:gqa"
    "wipe:gqa"
    "prune:gqa"
    "prune_wipe:gqa"
  )
else
  echo "unknown stage '$STAGE', expected A or B"
  exit 1
fi

PREV=""
for R in "${RUNS[@]}"; do
  M="${R%%:*}"
  T="${R##*:}"

  DEP=""
  if [ -n "$PREV" ]; then
    DEP="--dependency=afterany:${PREV}"
  fi

  OUT=$(MODE="$M" TASKS="$T" LIMIT="$LIM" TAG="$T" sbatch $DEP "$JOB")
  ID=$(echo "$OUT" | awk '{print $NF}')

  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
    echo "FAILED to submit mode=$M task=$T: $OUT"
    exit 1
  fi

  echo "queued $ID  mode=$M  task=$T  after=${PREV:-none}"
  PREV="$ID"
done

echo ""
echo "Stage $STAGE submitted. Last job id: $PREV"
echo ""
echo "Collect with:"
echo "  for f in logs/e10_*.out; do echo \"--- \$f\"; grep -m1 'MODE  =' \$f; grep -E '^\\|(textvqa|docvqa|chartqa|ai2d|pope|mme|mmbench|gqa|scienceqa)' \$f; done"

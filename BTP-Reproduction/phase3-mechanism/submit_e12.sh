#!/bin/bash
#
# Submit the Qwen2.5-VL depth sweep as a dependency chain.
#
# Companion to submit_e11.sh, which does the same on InternVL2-2B. Between them they produce
# the comparison figure: the depth at which each architecture stops needing visual tokens.
#
# Qwen2.5-VL-7B has 28 layers. The control at -1 disables the wipe entirely and should land
# on the unpruned baseline of 0.862, since BTP_RETAIN=1.0 also prunes nothing upstream. If
# the control does NOT match the baseline, something else in BTP_PARAM is interfering and
# the whole sweep is suspect, so run it first.
#
# Usage:
#   bash phase3-mechanism/submit_e12.sh                 # chain behind whatever is queued
#   AFTER=419012 bash phase3-mechanism/submit_e12.sh    # chain behind a specific job

set -uo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

JOB="phase3-mechanism/e12_job.sh"
TASK=${TASK:-textvqa_val}
LIM=${LIM:-500}

# -1 first: it is the control, and a failure there invalidates everything after it.
#
# The graded pruning stages are now disabled inside e12_job.sh, so there is no longer any
# collision with layers 4, 7 and 16, and no crash when the wipe lands before layer 16.
# The full depth range is therefore available. Denser sampling around 16 to 24, where the
# first attempt suggests the knee lies: layer 20 scored 0.1418 and layer 23 scored 0.2344,
# against a control of 0.8624.
LAYERS=${LAYERS:-"-1 2 4 6 8 10 12 14 16 18 20 22 24 26"}

PREV=${AFTER:-$(squeue -u "$USER" -h -o "%i" | sort -n | tail -1)}
if [ -n "$PREV" ]; then
  echo "chaining after job $PREV"
else
  echo "queue is empty, starting immediately"
fi

for L in $LAYERS; do
  DEP=""
  if [ -n "$PREV" ]; then
    DEP="--dependency=afterany:${PREV}"
  fi

  OUT=$(END_LAYER="$L" TASKS="$TASK" LIMIT="$LIM" RETAIN=1.0 TAG="endL${L}" \
        sbatch $DEP "$JOB")
  ID=$(echo "$OUT" | awk '{print $NF}')

  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
    echo "FAILED to submit end_layer $L: $OUT"
    exit 1
  fi

  echo "queued $ID  end_layer=$L  task=$TASK  after=${PREV:-none}"
  PREV="$ID"
done

echo ""
echo "E12 submitted. Last job id: $PREV"
echo ""
echo "Expected shape: collapse when the wipe lands before Qwen has finished reading,"
echo "recovery once it lands after. Known points: layer 23 gives 0.178, no wipe gives 0.862."
echo ""
echo "Collect with:"
echo "  for f in logs/e12_*.out; do"
echo "    L=\$(grep -m1 'END_LAYER =' \$f | awk '{print \$3}')"
echo "    S=\$(grep -m1 '^|textvqa' \$f | awk -F'|' '{print \$6}')"
echo "    [ -n \"\$S\" ] && echo \"end_layer \$L  ->  \$S\""
echo "  done | sort -n -k2"

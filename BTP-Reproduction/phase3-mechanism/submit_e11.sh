#!/bin/bash
#
# E11 - visual information depth sweep.
#
# WHY
#   Stage A gave us a striking null: deleting EVERY image token at layer 20 of InternVL2-2B
#   costs nothing on TextVQA (0.7234 against a 0.7156 baseline, wipe branch verified firing
#   in job 418983). On Qwen2.5-VL the same operation at a comparable relative depth
#   (layer 23 of 28) drives TextVQA from 0.876 to 0.178.
#
#   Two models, two answers. That difference is only interesting if we can say WHERE each
#   architecture stops needing the image. So instead of one wipe layer, sweep it.
#
# WHAT IT MEASURES
#   The depth at which visual information has been fully absorbed into the text stream.
#   Wipe before that depth and the task collapses; wipe after it and nothing happens. The
#   knee in the curve is the answer.
#
# WHY IT MATTERS
#   Every visual token pruning method implicitly assumes visual information is still needed
#   at the layer it prunes. That assumption holds at different depths in different
#   architectures, and nobody measures it. This turns our null result into a positive one,
#   and it explains the Qwen defect properly: layer 23 is catastrophic there not because
#   deleting tokens at depth is inherently harmful, but because Qwen has not finished
#   reading by then.
#
# USAGE
#   bash phase3-mechanism/submit_e11.sh                 # chain after whatever is queued
#   AFTER=419003 bash phase3-mechanism/submit_e11.sh    # chain after a specific job
#
#   Set AFTER to the last Stage B job id so this does not collide with it. All arms share
#   the same remote-code files, so nothing may run concurrently.

set -uo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

JOB="phase3-mechanism/e10_job.sh"
TASK=${TASK:-textvqa_val}
LIM=${LIM:-500}

# Layers to test. 24-layer backbone, so this brackets the whole depth range.
LAYERS=${LAYERS:-"2 4 6 8 10 12 16 20"}

# If AFTER is not given, chain behind the last job currently queued for this user so we
# never run alongside another arm.
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

  OUT=$(IVL_WIPE_LAYER="$L" MODE=wipe TASKS="$TASK" LIMIT="$LIM" TAG="wipeL${L}" \
        sbatch $DEP "$JOB")
  ID=$(echo "$OUT" | awk '{print $NF}')

  if ! [[ "$ID" =~ ^[0-9]+$ ]]; then
    echo "FAILED to submit wipe layer $L: $OUT"
    exit 1
  fi

  echo "queued $ID  wipe_layer=$L  task=$TASK  after=${PREV:-none}"
  PREV="$ID"
done

echo ""
echo "E11 submitted. Last job id: $PREV"
echo ""
echo "Baseline for comparison: unpatched TextVQA = 0.7156 (job 418974)."
echo "Expect a collapse at shallow layers and no effect at deep ones. The knee is the result."
echo ""
echo "Collect with:"
echo "  for f in logs/e10_*.out; do"
echo "    L=\$(grep -m1 'IVL_WIPE_LAYER=' \$f | sed 's/.*IVL_WIPE_LAYER=//;s/ .*//')"
echo "    S=\$(grep -m1 '^|textvqa' \$f | awk -F'|' '{print \$6}')"
echo "    [ -n \"\$L\" ] && echo \"layer \$L  ->  \$S\""
echo "  done | sort -n -k2"

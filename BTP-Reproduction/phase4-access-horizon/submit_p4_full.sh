#!/bin/bash
# Phase 4, full task set on Qwen2.5-VL-7B. POPE and TextVQA are already done (pilot).
#
#   gqa        compositional scene questions     300 samples
#   mmbench    general multiple choice            300
#   scienceqa  science diagrams, multiple choice  300
#   ai2d       diagram reading, multiple choice   300
#   chartqa    chart reading                      200   (capped at 2048 visual tokens)
#   docvqa     document reading, ANLS             200   (capped, one question per page)
#
# Horizon jobs never swap the modeling file after loading, so they can run in parallel.
# Each task gets a dependent resume job in case it hits the walltime.
# MIG nodes cn22 and cn25 are excluded inside p4_pilot_job.sh.
#
# Run from /scratch/data/divyasaxena_rs/kedar_BTPreproduction

set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

J=phase4-access-horizon/p4_pilot_job.sh
PART=${PART:-fat}

for spec in gqa:300 mmbench:300 scienceqa:300 ai2d:300 chartqa:200 docvqa:200; do
  t=${spec%%:*}; n=${spec##*:}
  a=$(TASK=$t N=$n sbatch --parsable -p "$PART" --job-name="p4_$t" "$J")
  b=$(TASK=$t N=$n sbatch --parsable -p "$PART" --job-name="p4_$t" --dependency=afterany:$a "$J")
  printf "%-10s %s  (resume %s)\n" "$t" "$a" "$b"
done

echo
echo "watch:  squeue -u \$USER"
echo "check:  grep -hE 'task    :|balance|l  28|FATAL|Error' logs/p4_pilot_*.out | tail -30"

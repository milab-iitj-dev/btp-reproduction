#!/bin/bash
# Phase 4 on InternVL2-2B, all eight tasks.
#
# TextVQA runs first because it is the validation: it must reproduce the E11 threshold
# (about 18 of 24 layers). POPE runs alongside it. The other six wait for TextVQA to finish
# successfully (afterok), so a code-level failure costs one job, not eight.
# Passing afterok only proves it ran; still check the TextVQA threshold before trusting
# the rest.
#
# Run from /scratch/data/divyasaxena_rs/kedar_BTPreproduction

set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs
J=phase4-access-horizon/p4_ivl_job.sh

T=$(TASK=textvqa N=200 sbatch --parsable --job-name=p4i_textvqa "$J")
P=$(TASK=pope N=300 sbatch --parsable --job-name=p4i_pope "$J")
echo "textvqa    $T   (validation)"
echo "pope       $P"
for spec in gqa:300 mmbench:300 scienceqa:300 ai2d:300 chartqa:200 docvqa:200; do
  t=${spec%%:*}; n=${spec##*:}
  j=$(TASK=$t N=$n sbatch --parsable --job-name=p4i_$t --dependency=afterok:$T "$J")
  printf "%-10s %s   (after textvqa)\n" "$t" "$j"
done
echo
echo "follow validation:  tail -f logs/p4_ivl_$T.out"

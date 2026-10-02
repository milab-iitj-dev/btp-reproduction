#!/bin/bash
# Submit the Phase 4 pilot: TextVQA and POPE, every layer, Qwen2.5-VL-7B.
#
# The two jobs can run at the same time. Unlike the BTP jobs, neither swaps the modeling
# file after loading; both only restore the pristine file at start, which is idempotent.
#
# Each task gets a second, dependent job that resumes if the first hit the walltime.
# If the first finished, the second loads the model, finds every layer done, re-runs the
# analysis and exits.
#
# Run from /scratch/data/divyasaxena_rs/kedar_BTPreproduction

set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs

J=phase4-access-horizon/p4_pilot_job.sh

T1=$(TASK=textvqa N=200 sbatch --parsable "$J")
T2=$(TASK=textvqa N=200 sbatch --parsable --dependency=afterany:$T1 "$J")
P1=$(TASK=pope N=300 sbatch --parsable "$J")
P2=$(TASK=pope N=300 sbatch --parsable --dependency=afterany:$P1 "$J")

echo "textvqa  $T1  (resume $T2)"
echo "pope     $P1  (resume $P2)"
echo
echo "watch:   squeue -u \$USER"
echo "follow:  tail -f logs/p4_pilot_$T1.out"

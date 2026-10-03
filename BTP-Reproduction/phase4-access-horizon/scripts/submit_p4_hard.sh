#!/bin/bash
# Hard-distractor control: TextVQA multiple choice where the wrong options are other text
# from the SAME image. Tests whether the early MC horizon only reflects easy distractors.
# Two jobs, Qwen2.5-VL-7B and InternVL2-2B, about 20-35 min each.
#
# The builder prints how many samples got three same-image distractors; check that line.
#
# Run from /scratch/data/divyasaxena_rs/kedar_BTPreproduction

set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs
RES="--cpus-per-task=2 --mem-per-cpu=16G -p fat"
q=$(TASK=textvqa_hard N=200 TAG=qwen7b sbatch --parsable $RES --job-name=p4h_q7 phase4-access-horizon/scripts/p4_pilot_job.sh)
i=$(TASK=textvqa_hard N=200 TAG=ivl2b sbatch --parsable $RES --job-name=p4h_iv phase4-access-horizon/scripts/p4_ivl_job.sh)
echo "textvqa_hard  qwen7b $q   ivl2b $i"
echo
echo "check:  grep -hE 'hard distractors|task    :|l   0 |l  2[48] |FATAL|Traceback' logs/p4_pilot_$q.out logs/p4_ivl_$i.out"

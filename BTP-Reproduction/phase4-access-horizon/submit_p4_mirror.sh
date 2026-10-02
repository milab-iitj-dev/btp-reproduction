#!/bin/bash
# Mirror format control: MMBench and AI2D questions that make sense without options, run
# both as multiple choice and open-ended on the SAME filtered set. Qwen2.5-VL-7B and
# InternVL2-2B. Eight jobs, about 15-30 min each, two at a time.
#
# Each log prints "mirror <task> <fmt>: kept K of S scanned". The MC and open lines for a
# task must show the same K, otherwise the pairing is broken.
#
# Run from /scratch/data/divyasaxena_rs/kedar_BTPreproduction

set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs
RES="--cpus-per-task=2 --mem-per-cpu=16G -p fat"
for t in mmbench_fmc mmbench_fopen ai2d_fmc ai2d_fopen; do
  q=$(TASK=$t N=200 TAG=qwen7b sbatch --parsable $RES --job-name=p4m_q7_$t phase4-access-horizon/p4_pilot_job.sh)
  i=$(TASK=$t N=200 TAG=ivl2b sbatch --parsable $RES --job-name=p4m_iv_$t phase4-access-horizon/p4_ivl_job.sh)
  printf "%-14s qwen7b %s   ivl2b %s\n" "$t" "$q" "$i"
done
echo
echo "check:  grep -hE 'mirror |task    :|l   0 |l  2[48] .*baseline|FATAL|Traceback' \$(ls -t logs/p4_*.out | head -8)"

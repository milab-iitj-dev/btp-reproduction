#!/bin/bash
# Format control: TextVQA and ChartQA asked as multiple choice, same questions as the
# open-ended runs, on Qwen2.5-VL-7B and InternVL2-2B. Four jobs, about 20-40 min each.
#
# CPUs: the fat nodes are CPU-bound, so every job asks for 2 CPUs x 16 GB. Single-sample GPU
# inference does not need more, and 8 CPUs kept jobs pending with GPUs free (26 Sep).
#
# Afterwards, locally:
#   python format_compare.py --results <horizon dir> --tags qwen7b,ivl2b \
#       --names "Qwen2.5-VL-7B,InternVL2-2B" --out <horizon dir>
#
# Run from /scratch/data/divyasaxena_rs/kedar_BTPreproduction

set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs
RES="--cpus-per-task=2 --mem-per-cpu=16G -p fat"

for t in textvqa_mc chartqa_mc; do
  q=$(TASK=$t N=200 TAG=qwen7b sbatch --parsable $RES --job-name=p4f_q7_$t phase4-access-horizon/p4_pilot_job.sh)
  i=$(TASK=$t N=200 TAG=ivl2b sbatch --parsable $RES --job-name=p4f_iv_$t phase4-access-horizon/p4_ivl_job.sh)
  printf "%-11s qwen7b %s   ivl2b %s\n" "$t" "$q" "$i"
done
echo
echo "check:  ls -t logs/p4_*.out | head -4 | xargs grep -hE 'task    :|l  2[48] |FATAL|Traceback'"

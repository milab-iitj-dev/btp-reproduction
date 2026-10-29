#!/bin/bash
# Phase 5: all seven response conditions, as two chains (two GPUs at once). dgx_fat has its own
# per-user GPU allowance (dgxqos, 4 GPUs), separate from the 2 shared on fat/dgx (fatqos).
# Run from the workspace root:
#   bash phase5-response-format/scripts/submit_p5.sh
set -euo pipefail
cd /scratch/data/divyasaxena_rs/kedar_BTPreproduction
mkdir -p logs
RES="--cpus-per-task=2 --mem-per-cpu=16G -p dgx_fat"
chain() {
  local prev="" id
  for c in "$@"; do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    id=$(COND=$c sbatch --parsable $RES $dep --job-name=p5_$c phase5-response-format/scripts/p5_job.sh)
    printf "%-9s %s\n" "$c" "$id"
    prev=$id
  done
}
chain mc4 open mc2 mc8
chain mc4_type yesno fill

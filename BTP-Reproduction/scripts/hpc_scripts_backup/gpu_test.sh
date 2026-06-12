#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test.out

hostname
date

echo "===== GPU INFO ====="
nvidia-smi

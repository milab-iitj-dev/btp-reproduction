#!/bin/bash
#SBATCH --job-name=env_check
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --output=logs/env_check_%j.out

hostname
nvidia-smi

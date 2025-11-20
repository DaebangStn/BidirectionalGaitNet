#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --gpus=1

ulimit -u 65535
ulimit -n 65536
python -u ppo/ppo_hierarchical.py
echo "Training finished"

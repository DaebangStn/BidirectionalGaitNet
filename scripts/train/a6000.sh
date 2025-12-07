#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --gpus=1

ulimit -u 65535
ulimit -n 65536
python -u ppo/learn.py a6000 --env_file "data/env/${SLURM_JOB_NAME}.yaml"
echo "Training finished"

#!/bin/bash
#SBATCH --cpus-per-task=108
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ulimit -u 65535
ulimit -n 65536

cd ~/BidirectionalGaitNet
PYTHONUNBUFFERED=1 pixi run python -m ppo.learn --env_file "data/env/${SLURM_JOB_NAME}.yaml"
echo "Training finished"

#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ulimit -u 65535
ulimit -n 65536

cd ~/BidirectionalGaitNet
pixi run python -m ppo.learn --env_file "data/env/${SLURM_JOB_NAME}.yaml"
echo "Training finished"

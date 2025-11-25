#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=exo


ulimit -u 65535
ulimit -n 65536
python -u ppo/learn.py --num_envs 128 --num_steps 128 --muscle_batch_size 512  --num_minibatches 16 --env-file "data/env/${SLURM_JOB_NAME}.yaml"
echo "Training finished"
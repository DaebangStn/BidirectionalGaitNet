#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=exo


ulimit -u 65535
ulimit -n 65536
python -u ppo/ppo_rollout_learner.py --num_envs 128 --num_steps 128 --muscle_batch_size 512  --num_minibatches 16 --run-name "${SLURM_JOB_NAME}"
echo "Training finished"
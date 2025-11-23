#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --gpus=1

ulimit -u 65535
ulimit -n 65536
python -u ppo/ppo_hierarchical.py --num_envs 96 --num_steps 128 --muscle_batch_size 512  --num_minibatches 16 --run-name "${SLURM_JOB_NAME}"
echo "Training finished"

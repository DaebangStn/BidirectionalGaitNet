#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --gpus=1

let "num_pending_trials=(${SLURM_NNODES} * 64)"
export TUNE_MAX_PENDING_TRIALS_PG=${num_pending_trials}

python ppo/ppo_hierarchical.py
echo "Training finished"

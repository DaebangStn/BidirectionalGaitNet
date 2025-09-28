#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=exo

let "num_pending_trials=(${SLURM_NNODES} * 64)"
export TUNE_MAX_PENDING_TRIALS_PG=${num_pending_trials}

python3 -u ray_train.py --config=ppo_small_node --name="${SLURM_JOB_NAME}" --env "data/${SLURM_JOB_NAME}.xml"

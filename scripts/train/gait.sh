#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=exo


ulimit -u 65535
ulimit -n 65536

# Handle prefetch workflow flag
CLEAR_CACHE_FLAG=""
if [ -n "$NO_CLEAR_CACHE" ]; then
    CLEAR_CACHE_FLAG="--no_clear_cache True"
fi

# Use ENV_FILE if set, otherwise fall back to SLURM_JOB_NAME
ENV_FILE="${ENV_FILE:-data/env/${SLURM_JOB_NAME}.yaml}"

python -u ppo/learn.py gait --env_file "$ENV_FILE" $CLEAR_CACHE_FLAG
echo "Training finished"
#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --partition=exo


ulimit -u 65535
ulimit -n 65536

let "num_pending_trials=(${SLURM_NNODES} * 64)"
export TUNE_MAX_PENDING_TRIALS_PG=${num_pending_trials}

ray stop
ray start --head --num-cpus=${SLURM_CPUS_PER_TASK} --num-gpus=1 \
  --ray-client-server-port=19999 --min-worker-port=20000 --max-worker-port=21000

python3 -u python/ray_train.py --config=ppo_small_n1 --env "data/${SLURM_JOB_NAME}.yaml"
ray stop
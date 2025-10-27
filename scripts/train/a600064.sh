#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --gpus=1

let "num_pending_trials=(${SLURM_NNODES} * 64)"
export TUNE_MAX_PENDING_TRIALS_PG=${num_pending_trials}

ray stop
ray start --head --num-cpus=${SLURM_CPUS_PER_TASK} --num-gpus=1 \
  --ray-client-server-port=19999 --min-worker-port=20000 --max-worker-port=21000

python3 -u python/ray_train.py --config=ppo_64_a6000 --env "data/${SLURM_JOB_NAME}.yaml"

ray stop
echo "Training finished"

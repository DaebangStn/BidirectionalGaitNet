#!/bin/bash
#SBATCH --job-name=gaitnet_serr
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --gpus=1

ulimit -u 65535
ulimit -n 65536
python -u ppo/ppo_rollout_learner.py --num_envs 96 --num_steps 8 --muscle_batch_size 128  --num_minibatches 4 --run-name "${SLURM_JOB_NAME}" --use-erroneous-bootstrap
echo "Training finished"

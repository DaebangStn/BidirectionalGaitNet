#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --partition=exo

ulimit -u 65535
ulimit -n 65536

let "num_pending_trials=(${SLURM_NNODES} * 64)"
export TUNE_MAX_PENDING_TRIALS_PG=${num_pending_trials}

let "worker_num=(${SLURM_NTASKS} - 1)"
let "total_cores=${SLURM_NTASKS} * ${SLURM_CPUS_PER_TASK}"

allocated_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=$(echo "$allocated_nodes" | head -n1)
suffix='6379'
ip_head_wo_port=$(getent hosts "${head_node}" | awk '{print $1}')
ip_head=${ip_head_wo_port}:$suffix
echo "head: ${head_node} with ip: ${ip_head}"

srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL,NCCL_SOCKET_IFNAME=ib0 \
  ray start --head --block --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &

sleep 3

srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude="${head_node}" \
  --export=ALL,NCCL_SOCKET_IFNAME=ib0 ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &

sleep 3
python3 -u python/ray_train.py --config=ppo_small_n3 --env "data/env/${SLURM_JOB_NAME}.yaml"

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --export=ALL ray stop

echo "Ray stopped"

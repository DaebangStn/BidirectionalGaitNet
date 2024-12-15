#!/bin/bash
#SBATCH --job-name=gaitnet
#SBATCH --cpus-per-task=128
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --partition=all
let "worker_num=(${SLURM_NTASKS} - 1)"
let "total_cores=${SLURM_NTASKS} * ${SLURM_CPUS_PER_TASK}"
suffix='6379'
ip_head=$1:$suffix
allocated_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=$(echo "$allocated_nodes" | head -n1)

export ip_head # Exporting for latter access by trainer.py
ulimit -n 65536
srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --export=ALL,NCCL_SOCKET_IFNAME=ib0 \
  ray start --head --block --dashboard-host 0.0.0.0 --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &

sleep 3

srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude="${head_node}" \
  --export=ALL,NCCL_SOCKET_IFNAME=ib0 ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &

sleep 3
python3 -u ray_train.py --config=ppo_small_node --name=test --cluster


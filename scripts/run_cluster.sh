#!/bin/bash
# Cluster execution wrapper for optimal NUMA and threading configuration
#
# Usage: ./scripts/run_cluster.sh python3 ppo/ppo_rollout_learner.py --num-envs 16
#
# This script configures:
# 1. NUMA binding to single node (avoids cross-node penalty)
# 2. OpenMP threading limits (prevents thread explosion)
# 3. MKL threading limits (prevents BLAS thread explosion)

# Set threading limits BEFORE execution
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# Check if we're on a multi-NUMA system
NUMA_NODES=$(numactl --hardware 2>/dev/null | grep "available:" | awk '{print $2}')

if [ "$NUMA_NODES" -gt 1 ] 2>/dev/null; then
    echo "Multi-NUMA system detected ($NUMA_NODES nodes)"
    echo "Binding to NUMA node 0 for optimal performance..."
    echo "Running: numactl --cpunodebind=0 --membind=0 $@"
    echo ""

    exec numactl --cpunodebind=0 --membind=0 "$@"
else
    echo "Single NUMA node system"
    echo "Running: $@"
    echo ""

    exec "$@"
fi

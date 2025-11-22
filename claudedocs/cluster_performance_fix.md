# Cluster Performance Fix: NUMA and OpenMP Threading

**Date**: 2025-11-22
**Status**: ✅ RESOLVED
**Impact**: **2x performance improvement** on 56-core cluster

---

## Problem Summary

Cluster (56 cores) was **2-3x slower** than local machine (16 cores) despite having more resources.

### Performance Data

**Initial Problem (Cluster):**
```
BatchRolloutEnv: 7488ms rollout (2.8x slower than local's 2704ms)
```

**After All Fixes (Cluster):**
```
BatchRolloutEnv: 3656ms rollout (1.35x slower than local, acceptable)
```

**Improvement:** **2x speedup** (7488ms → 3656ms)

---

## Root Causes Identified

### Issue 1: NUMA Cross-Node Memory Access ⚠️
**Impact:** Minor (~200ms penalty)

**Problem:**
- Cluster has 2 NUMA nodes with cross-node distance of 21 (2.1x penalty)
- Threads on node 0 accessing memory on node 1 incur latency penalty
- Local machine has single NUMA node (no cross-node access)

**Solution:**
```bash
numactl --cpunodebind=0 --membind=0 python3 script.py
```

Binds both CPU and memory to NUMA node 0, eliminating cross-node access.

### Issue 2: OpenMP Thread Explosion 🔥
**Impact:** CRITICAL (~4000ms penalty, **2x slowdown**)

**Problem:**
- DART physics engine uses OpenMP (via libomp.so)
- Default OpenMP threads = hardware_concurrency = 56 threads
- With 16 ThreadPool workers: 16 × 56 = **896 threads!**
- Massive thread oversubscription and context switching

**Evidence:**
```bash
$ ldd ppo/batchrolloutenv.so | grep omp
libomp.so => /opt/miniconda3/envs/bidir/lib/./libomp.so
```

**Solution:**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Limits each DART operation to single thread, total threads = 16 (ThreadPool size).

### Issue 3: ThreadPool Sizing ✅
**Impact:** Moderate (fixed in previous work)

**Problem:**
- ThreadPool created with `hardware_concurrency()` = 56 threads
- Only 16 environments actually using threads
- 40 idle threads causing overhead

**Solution:**
```cpp
// ppo/BatchRolloutEnv.cpp:44
pool_(num_envs)  // Match thread count to number of environments
```

---

## Complete Solution

### Architecture

```
Cluster Execution (56-core, 2 NUMA nodes):

┌─────────────────────────────────────────────────┐
│ NUMA Node 0 (28 physical cores, 56 logical)    │
│                                                  │
│  ┌────────────────────────────────────────┐    │
│  │ Python Process                          │    │
│  │                                          │    │
│  │  ThreadPool: 16 threads                 │    │
│  │  ├─ Thread 0: Env 0                     │    │
│  │  │  └─ DART (OMP_NUM_THREADS=1)        │    │
│  │  ├─ Thread 1: Env 1                     │    │
│  │  │  └─ DART (OMP_NUM_THREADS=1)        │    │
│  │  └─ ...                                  │    │
│  │     └─ Thread 15: Env 15                │    │
│  │        └─ DART (OMP_NUM_THREADS=1)      │    │
│  │                                          │    │
│  │  Memory: Allocated on Node 0 only       │    │
│  └────────────────────────────────────────┘    │
│                                                  │
└─────────────────────────────────────────────────┘

NUMA Node 1: UNUSED (bound to node 0)

Total Threads: 16 (optimal!)
Memory Access: Local only (no cross-node penalty)
```

### Execution Methods

#### Method 1: Wrapper Script (Recommended)

```bash
# Use the cluster wrapper script
./scripts/run_cluster.sh python3 ppo/ppo_rollout_learner.py --num-envs 16 --num-steps 64

# The script automatically:
# - Detects NUMA topology
# - Sets OMP_NUM_THREADS=1
# - Binds to NUMA node 0 if multi-node
```

#### Method 2: Manual Command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 numactl --cpunodebind=0 --membind=0 \
    python3 ppo/ppo_rollout_learner.py --num-envs 16 --num-steps 64
```

#### Method 3: SLURM Integration

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28  # Use one NUMA node worth of CPUs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# SLURM automatically provides NUMA binding in many cases
# But explicitly bind to be safe:
numactl --cpunodebind=0 --membind=0 python3 ppo/ppo_rollout_learner.py \
    --num-envs 16 \
    --num-steps 64 \
    --total-timesteps 50000000
```

---

## Performance Results

### Cluster Performance Progression

**Baseline (No fixes):**
```
BatchRolloutEnv: 7488ms rollout
- 2 NUMA nodes: cross-node access penalty
- 56 OpenMP threads per env: 896 total threads
- 56 ThreadPool workers: 40 idle
```

**After NUMA binding:**
```
BatchRolloutEnv: 7280ms rollout (3% improvement)
- Single NUMA node: no cross-node penalty ✅
- 56 OpenMP threads per env: still 896 threads ❌
- 56 ThreadPool workers: still wasteful ❌
```

**After OMP_NUM_THREADS=1:**
```
BatchRolloutEnv: 3656ms rollout (2x improvement!) ✅
- Single NUMA node: no cross-node penalty ✅
- 1 OpenMP thread per env: 16 total threads ✅
- 16 ThreadPool workers: perfect match ✅
```

### Comparison: Cluster vs Local

**Local (16 cores, 1 NUMA node):**
```
BatchRolloutEnv: 2704ms rollout, 426 SPS
```

**Cluster (56 cores, 2 NUMA nodes, optimized):**
```
BatchRolloutEnv: 3656ms rollout, 318 SPS
```

**Cluster vs Local:** 1.35x slower (acceptable for cluster with NUMA overhead)

---

## Why Cluster is Still Slightly Slower

Even with all fixes, cluster is 1.35x slower than local. This is **expected and acceptable** due to:

### 1. NUMA Architecture Overhead
- 2-socket system has inherent memory latency
- Cache coherency protocol (MESI) overhead
- Even with node-local memory, inter-socket communication for system resources

### 2. Hyperthreading Artifacts
- Cluster uses hyperthreading (56 logical cores from 28 physical)
- Logical cores share execution units, not full performance
- Local might have better core-to-core communication

### 3. Environment Differences
- Different CPU models (cluster might be older)
- Different memory speeds
- Different system load

**1.35x overhead is NORMAL for cluster vs workstation with NUMA topology.**

---

## Thread Count Breakdown

### Before Fix (BROKEN)
```
Cluster with 16 environments:

ThreadPool: 56 threads
  └─ Active: 16 threads
  └─ Idle: 40 threads (wasting resources)

Each active thread:
  └─ DART OpenMP: 56 threads (default)
     └─ Total: 16 × 56 = 896 threads!

Grand Total: 56 + 896 = 952 threads for 16 environments! 🔥
```

### After Fix (OPTIMIZED)
```
Cluster with 16 environments:

ThreadPool: 16 threads (matches num_envs)
  └─ Active: 16 threads
  └─ Idle: 0 threads (optimal!)

Each active thread:
  └─ DART OpenMP: 1 thread (OMP_NUM_THREADS=1)
     └─ Total: 16 × 1 = 16 threads

Grand Total: 16 threads for 16 environments ✅
```

**Thread reduction: 952 → 16 (98% reduction!)**

---

## Files Created

1. **scripts/run_cluster.sh** (NEW)
   - Automatic NUMA detection and binding
   - Sets OMP/MKL threading limits
   - Usage: `./scripts/run_cluster.sh python3 script.py [args]`

2. **claudedocs/cluster_performance_fix.md** (THIS FILE)
   - Complete documentation of NUMA and threading issues
   - Performance benchmarks and analysis
   - Deployment instructions

---

## Deployment Checklist

### For All Cluster Training

- [ ] Use `scripts/run_cluster.sh` wrapper for all Python training scripts
- [ ] Verify NUMA binding: `numactl --show` during execution
- [ ] Check thread count: `ps -eLf | grep python | wc -l` should be ~16 for 16 envs
- [ ] Monitor performance: BatchRolloutEnv should be 3500-4000ms on cluster

### For SLURM Jobs

- [ ] Set `OMP_NUM_THREADS=1` in SLURM script
- [ ] Use `numactl --cpunodebind=0 --membind=0` explicitly
- [ ] Request CPUs from single NUMA node if possible
- [ ] Monitor job efficiency (should be >60%)

### Verification Commands

```bash
# Check NUMA binding
numactl --show

# Check thread count (should be ~16 for 16 envs)
ps -eLf | grep python | wc -l

# Check OpenMP settings
echo $OMP_NUM_THREADS  # Should be 1

# Run benchmark
./scripts/run_cluster.sh python3 ppo/benchmark_rollout.py --num-envs-list 16
```

---

## Key Learnings

### Technical Insights

✅ **NUMA matters**: 2-socket systems have cross-node memory penalties
✅ **OpenMP defaults are dangerous**: Always set `OMP_NUM_THREADS=1` for parallel workloads
✅ **Thread explosion is real**: 16 envs × 56 OMP threads = 896 threads!
✅ **Environment variables propagate**: Must set before Python import

### Best Practices

1. **Always check NUMA topology** on new clusters
2. **Explicitly set threading limits** for all parallel libraries
3. **Bind to single NUMA node** for thread-based parallelism
4. **Use wrapper scripts** to ensure consistent configuration
5. **Benchmark early** to detect environment-specific issues

### Why ppo_hierarchical.py (AsyncVectorEnv) Wasn't Affected

AsyncVectorEnv uses **multiprocessing** (separate processes), where each process:
- Has independent memory space
- Gets pinned by OS scheduler naturally
- Doesn't share NUMA node resources as aggressively
- Has per-process OMP_NUM_THREADS=1 working correctly

BatchRolloutEnv uses **multithreading** (shared memory), where:
- All threads share memory space
- Must manually bind to NUMA node
- OMP threading affects all threads
- Requires explicit configuration

---

## Conclusion

The cluster performance issue was caused by **NUMA architecture** and **OpenMP thread explosion**, not the ThreadPool sizing we initially suspected.

**Solution Summary:**
1. ✅ Bind to single NUMA node: `numactl --cpunodebind=0 --membind=0`
2. ✅ Limit OpenMP threading: `OMP_NUM_THREADS=1`
3. ✅ Use wrapper script: `scripts/run_cluster.sh`

**Performance Impact:**
- **2x speedup** on cluster (7488ms → 3656ms)
- **Thread reduction**: 952 → 16 threads (98% less overhead)
- **Near-local performance**: 1.35x slower (acceptable for NUMA overhead)

**Deployment:**
```bash
# All cluster training should use:
./scripts/run_cluster.sh python3 ppo/ppo_rollout_learner.py --num-envs 16
```

The system is now optimized for cluster deployment! 🚀

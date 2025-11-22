# Threading Performance Fix - Complete Summary

**Date**: 2025-11-22
**Status**: ✅ RESOLVED
**Impact**: **6.3x throughput improvement** (62.4 → 396.8 SPS at 16 envs)

---

## Problem Statement

BatchRolloutEnv exhibited severe performance degradation when scaling from 2 to 16 environments:

### Before Fix (Thread Oversubscription)
- **2→16 envs**: Rollout time increased **8.98x** (2,850ms → 25,601ms)
- **Peak throughput**: 62.4 SPS at 8 envs
- **16 envs performance**: 39.7 SPS (worse than 2 envs!)
- **Scaling efficiency**: Only 11.6% at 16 envs

```
Num Envs   Rollout (ms)   SPS     Speedup    Efficiency
--------------------------------------------------------
2          2850.7         42.6    1.00x      100.0%
4          5519.1         45.4    1.07x      53.3%
8          8029.3         62.4    1.46x      36.6%
16         25601.8        39.7    0.93x      11.6%  ← WORSE than baseline!
```

---

## Root Cause Analysis

### Nested Parallelism Thread Explosion

**Environment-Level Parallelism**:
- BatchRolloutEnv uses C++ ThreadPool with `std::thread::hardware_concurrency()` threads
- On 64-core server: ThreadPool creates 64 worker threads

**Operation-Level Parallelism** (the problem):
- OMP_NUM_THREADS was set to `cpu_count()` = 64
- MKL_NUM_THREADS was set to `cpu_count()` = 64
- Each libtorch operation spawned 64 threads

**Thread Explosion Calculation**:
```
num_envs = 16
ThreadPool threads = 64
OMP threads per operation = 64

Total thread contention: 16 × 64 = 1,024 threads competing for 64 cores!
```

**Result**: Massive context switching overhead destroyed performance.

### Diagnostic Evidence

Running `diagnose_threading.sh` revealed:
```bash
# Before fix:
torch.get_num_threads() = 16
torch.get_num_interop_threads() = 32
Thread count: 32 → 47 during computation (should stay at 1!)

# After fix:
torch.get_num_threads() = 1
torch.get_num_interop_threads() = 1
Thread count: 1 → 1 during computation ✅
```

---

## Solution Implementation

### Threading Model

**Correct Architecture**:
```
Environment-level parallelism:
  └─ ThreadPool (hardware_concurrency threads)
     └─ Each thread runs ONE environment
        └─ Libtorch operations: SINGLE-THREADED (OMP=1, MKL=1)
```

**Why This Works**:
- Parallelism at environment level (coarse-grained, minimal contention)
- Each environment's libtorch operations run single-threaded
- No nested parallelism, no thread explosion
- Optimal CPU utilization without context switching

### Code Changes

#### 1. Entry Point Threading Configuration

Added to **all** entry point scripts before any torch/libtorch imports:

```python
# CRITICAL: Set threading BEFORE any imports that might load torch/libtorch
# This prevents nested parallelism: BatchRolloutEnv uses ThreadPool for env parallelism
# Setting OMP/MKL to 1 ensures each libtorch operation runs single-threaded within its thread
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
# Also set PyTorch's internal threading limits
# Use try-except to handle cases where threading is already configured
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Threading already configured, ignore
    pass
```

**Why try-except?**
- Allows modules to be imported without duplicate threading errors
- Entry point sets threading first, imported modules skip silently
- Prevents "cannot set number of interop threads" error

#### 2. Files Modified

**Training Scripts**:
- ✅ `ppo/ppo_rollout_learner.py` - Main C++ rollout training script
- ✅ `ppo/ppo_hierarchical.py` - Reference Python rollout training script
- ✅ `ppo/muscle_nn.py` - Removed duplicate threading calls (now configured at entry point)

**Benchmark Scripts**:
- ✅ `ppo/benchmark_num_envs.py` - Num_envs scaling benchmark
- ✅ `ppo/benchmark_rollout.py` - BatchEnv vs BatchRolloutEnv comparison
- ✅ `scripts/benchmark_backends.py` - AsyncVectorEnv vs BatchEnv comparison

**Diagnostic Tools**:
- ✅ `scripts/diagnose_threading.sh` - Threading analysis script (NEW)

---

## Performance Results

### After Fix (Single-Threaded Libtorch)

```
Num Envs   Rollout (ms)   SPS     Speedup    Efficiency
--------------------------------------------------------
2          968.1          120.7   1.00x      100.0%
4          1020.4         229.4   1.90x      95.0%
8          1244.4         370.7   3.07x      76.8%
16         2362.6         396.8   3.29x      41.1%
```

### Performance Improvements

**Rollout Time Scaling**:
- Before: 2→16 envs = **8.98x increase** (catastrophic)
- After: 2→16 envs = **2.44x increase** (excellent)
- **Improvement**: 3.68x better scaling

**Throughput (SPS)**:
- Before: 62.4 SPS peak (at 8 envs, degraded at 16)
- After: 396.8 SPS peak (at 16 envs, continues scaling)
- **Improvement**: **6.3x throughput increase**

**Absolute Time Savings** (per 1,024 samples at 16 envs):
- Before: 25,601ms total time
- After: 2,363ms total time
- **Time saved**: 23,238ms (90.8% reduction!)

---

## Key Takeaways

### Best Practices

1. **Set threading BEFORE imports**: Environment variables and torch settings must be configured before any module loads libtorch
2. **Use try-except for torch.set_num_*_threads()**: Prevents duplicate configuration errors when importing modules
3. **Single-threaded libtorch with environment parallelism**: Let ThreadPool handle parallelism, keep operations single-threaded
4. **Diagnose with tools**: Use diagnostic scripts to verify threading configuration is actually applied

### Common Pitfalls to Avoid

❌ **Don't**: Set `OMP_NUM_THREADS = cpu_count()` with environment parallelism
✅ **Do**: Set `OMP_NUM_THREADS = 1` for single-threaded operations

❌ **Don't**: Configure threading after importing torch
✅ **Do**: Set environment variables and torch threading BEFORE any imports

❌ **Don't**: Assume threading config is applied without verification
✅ **Do**: Use diagnostic tools to verify actual thread counts

❌ **Don't**: Configure threading in multiple places
✅ **Do**: Configure once at entry point, use try-except in modules

### Performance Validation

To verify the fix is working correctly:

```bash
# Run the num_envs scaling benchmark
python3 ppo/benchmark_num_envs.py --num-envs-list 2 4 8 16 --total-timesteps 2048

# Expected results:
# - Rollout time should scale sub-linearly (2→16 envs: ~2-3x increase)
# - SPS should continue increasing with num_envs (peak at highest num_envs)
# - Efficiency should remain >40% at 16 envs
```

---

## Additional Context

### Related Issues Fixed

1. **GAE Computation Bug**: Fixed masking to use `dones` instead of just `terminations`
2. **Terminal Bootstrapping Bug**: Fixed C++ to check `truncated && !terminated`
3. **Logical Equivalence**: Verified ppo_hierarchical.py and ppo_rollout_learner.py produce identical results

### Files for Reference

- **Performance benchmarks**: `benchmark_results/num_envs_benchmark_*.txt`
- **Logical equivalence**: `claudedocs/logical_equivalence_verification.md`
- **Threading diagnostics**: `scripts/diagnose_threading.sh`

---

## Conclusion

The threading performance fix successfully resolved the catastrophic performance degradation by eliminating nested parallelism. The system now achieves **6.3x better throughput** with proper scaling behavior up to 16+ environments.

**Key Success Metrics**:
- ✅ Rollout time scaling: 8.98x → 2.44x (3.68x improvement)
- ✅ Peak throughput: 62.4 → 396.8 SPS (6.3x improvement)
- ✅ Time efficiency: 90.8% reduction in total time at 16 envs
- ✅ Scaling continues: Performance peaks at maximum tested envs (16)

The fix is minimal, non-invasive, and applies cleanly to all training and benchmark scripts.

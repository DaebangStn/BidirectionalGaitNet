# Threading Configuration Best Practices

Quick reference guide for proper threading configuration in BidirectionalGaitNet.

---

## TL;DR

**Always add this to the TOP of your entry point scripts** (before any torch imports):

```python
# CRITICAL: Set threading BEFORE any imports that might load torch/libtorch
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Already configured, ignore
```

---

## Why This Matters

### The Problem: Nested Parallelism

Our C++ environments (BatchEnv, BatchRolloutEnv) use **ThreadPool for environment-level parallelism**:
- Creates `hardware_concurrency()` threads (e.g., 64 threads on a 64-core server)
- Each thread runs ONE environment simulation

If libtorch operations ALSO use multiple threads (via OMP/MKL), we get **nested parallelism**:
```
ThreadPool threads (64) × OMP threads (64) = 4,096 threads on 64 cores!
```

**Result**: Catastrophic context switching overhead, performance degradation.

### The Solution: Single-Threaded Operations

Set OMP/MKL to 1 thread, so libtorch operations run **single-threaded within each ThreadPool thread**:
```
ThreadPool threads (64) × OMP threads (1) = 64 threads on 64 cores ✅
```

**Result**: Optimal CPU utilization, 6.3x throughput improvement.

---

## Configuration Template

### Entry Point Script (e.g., training script, benchmark)

```python
#!/usr/bin/env python3
"""Your script description."""

# ========================================
# THREADING CONFIGURATION - MUST BE FIRST
# ========================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Import torch AFTER setting environment variables
import torch
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Threading already configured, ignore

# Now safe to import other modules
import numpy as np
from pathlib import Path
# ... other imports ...

# Rest of your script
```

### Library Module (e.g., muscle_nn.py, utilities)

```python
"""Library module that might be imported by entry points."""

import torch
# DO NOT set threading here - it's configured at the entry point
# If you try to set it, you'll get "cannot set number of interop threads" error

# Just a note in comments:
# NOTE: PyTorch threading is configured at the entry point (ppo_rollout_learner.py, etc.)
# Do not set threading here to avoid "cannot set number of interop threads" errors

# Rest of your module
```

---

## Which Files Need This?

### ✅ Entry Point Scripts (MUST configure threading)

Training scripts:
- `ppo/ppo_rollout_learner.py`
- `ppo/ppo_hierarchical.py`

Benchmark scripts:
- `ppo/benchmark_num_envs.py`
- `ppo/benchmark_rollout.py`
- `scripts/benchmark_backends.py`

Any other scripts that directly use BatchEnv/BatchRolloutEnv.

### ❌ Library Modules (DO NOT configure threading)

- `ppo/muscle_nn.py`
- `ppo/muscle_learner.py`
- Any utility modules imported by entry points

**Why?** Threading can only be set once. Entry points configure it first, libraries inherit the settings.

---

## Verification

### Check Threading is Applied

```python
import torch
print(f"torch.get_num_threads(): {torch.get_num_threads()}")
print(f"torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

# Should output:
# torch.get_num_threads(): 1
# torch.get_num_interop_threads(): 1
```

### Check Environment Variables

```bash
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# Should output:
# OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1
```

### Monitor Thread Count During Execution

```bash
# Run your script
python3 ppo/ppo_rollout_learner.py --num-envs 16 --total-timesteps 1000 &
PID=$!

# Monitor thread count (should stay constant)
while kill -0 $PID 2>/dev/null; do
    ps -p $PID -L -o nlwp | tail -1
    sleep 0.5
done
```

### Use Diagnostic Script

```bash
# Comprehensive threading diagnostic
bash scripts/diagnose_threading.sh
```

---

## Common Errors and Fixes

### Error: "cannot set number of interop threads"

**Cause**: Trying to set threading after it was already set (or after parallel work started).

**Fix**: Use try-except block:
```python
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Already configured, ignore
```

### Error: Performance still slow despite threading config

**Diagnosis**:
```python
# Check if threading is actually applied
import torch
print(torch.get_num_threads())  # Should be 1, not 16 or 64
```

**Fix**: Ensure threading is configured BEFORE any torch imports:
```python
# WRONG ORDER:
import torch  # ❌ Threading initialized here
os.environ["OMP_NUM_THREADS"] = "1"  # ❌ Too late!

# CORRECT ORDER:
os.environ.setdefault("OMP_NUM_THREADS", "1")  # ✅ Set first
import torch  # ✅ Now threading is correct
```

### Error: Module import fails with threading error

**Cause**: Module is trying to set threading, but entry point already set it.

**Fix**: Remove threading configuration from the library module. Only entry points should configure threading.

---

## Performance Benchmarks

### Before Threading Fix
```
Num Envs   Rollout (ms)   SPS     Speedup
------------------------------------------
2          2850.7         42.6    1.00x
16         25601.8        39.7    0.93x  ← SLOWER!
```

### After Threading Fix
```
Num Envs   Rollout (ms)   SPS     Speedup
------------------------------------------
2          968.1          120.7   1.00x
16         2362.6         396.8   3.29x  ✅ 6.3x faster!
```

---

## Quick Checklist

When adding a new script that uses BatchEnv/BatchRolloutEnv:

- [ ] Add threading configuration at the TOP (before any imports)
- [ ] Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`
- [ ] Set `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` with try-except
- [ ] Verify threading is applied: `torch.get_num_threads()` should return 1
- [ ] Run benchmark to confirm good scaling performance

---

## References

- **Threading fix documentation**: `claudedocs/threading_performance_fix.md`
- **Diagnostic script**: `scripts/diagnose_threading.sh`
- **Benchmark results**: `benchmark_results/num_envs_benchmark_*.txt`
- **Logical equivalence**: `claudedocs/logical_equivalence_verification.md`

---

**Last Updated**: 2025-11-22
**Status**: ✅ Production-ready

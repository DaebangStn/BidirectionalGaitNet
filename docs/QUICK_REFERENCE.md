# Libtorch Migration - Quick Reference Card

## 🎯 What Changed

### Before
- ❌ Python MuscleNN (requires GIL)
- ❌ Sequential BatchEnv (no parallelization)
- ❌ ~234 SPS with 8 environments

### After
- ✅ C++ MuscleNN with libtorch (no GIL)
- ✅ Parallel BatchEnv with ThreadPool
- ✅ ~373 SPS with 32 environments (**1.59x speedup**)

## 🏗️ Architecture Overview

```
Python Training Code
    ↓
BatchEnv (C++)
    ↓ [GIL Released]
ThreadPool → Environment[0] → C++ MuscleNN
          → Environment[1] → C++ MuscleNN
          → Environment[N] → C++ MuscleNN
    ↓ [GIL Reacquired]
Return (obs, rewards, dones)
```

## 📁 Key Files

### C++ MuscleNN
- `sim/MuscleNN.h` - Network class definition
- `sim/MuscleNN.cpp` - Forward pass, weight loading
- 4-layer MLP: 256-256-256, LeakyReLU(0.2)

### Environment Integration
- `sim/Environment.h` - `mMuscleNN` member, `setMuscleNetworkWeight()`
- `sim/Environment.cpp` - `initialize()`, `calcActivation()`

### Viewer
- `viewer/GLFWApp.h` - `mMuscleStateDict` storage
- `viewer/GLFWApp.cpp` - `loadNetworkFromPath()`, `initEnv()`

### Parallel Execution
- `ppo/BatchEnv.cpp` - Parallel `reset()`, `step()`
- `ppo/ThreadPool.h` - Worker thread pool

## 🔧 Build Commands

```bash
# Build everything
ninja -C build/release

# Build specific targets
ninja -C build/release viewer
ninja -C build/release batchenv.so
```

## 🐍 Python Usage

```python
import sys
sys.path.insert(0, '/path/to/ppo')
import batchenv

# Load YAML config
with open('data/env/A2_sep.yaml') as f:
    yaml_content = f.read()

# Create parallel BatchEnv
env = batchenv.BatchEnv(yaml_content, num_envs=32)

# Reset (parallel)
obs = env.reset()  # (32, obs_dim) float32

# Step (parallel, GIL released)
obs, rew, done = env.step(actions)  # actions: (32, action_dim) float32
```

## ⚡ Performance

| Envs | SPS   | Speedup |
|------|-------|---------|
| 8    | 234.7 | 1.00x   |
| 16   | 327.4 | 1.40x   |
| 32   | 373.1 | 1.59x   |

## 🔍 Debugging

### Viewer (GLFW)
```bash
export DISPLAY=:0
./scripts/viewer
```

### Test Parallel BatchEnv
```bash
/opt/miniconda3/bin/micromamba run -n bidir \
    python scripts/test_parallel_batchenv.py
```

### Check Muscle Activations
If activations are zero:
1. Verify checkpoint has `muscle.pt`
2. Check `setMuscleNetworkWeight()` is called in `initEnv()`
3. Confirm `mMuscleStateDict` is not None

## 🐛 Common Issues

### Segfault in calcActivation()
**Check**: `mChildNetworks.empty()` before accessing `.back()`

### Zero activations
**Check**: Weights transferred via `setMuscleNetworkWeight()`

### Poor parallel scaling
**Check**: `pool_.wait()` after task submission

### GLFW initialization failure
**Solution**: `export DISPLAY=:0`

## 📊 Network Architecture

```
Input: muscle_tau (muscle_dofs) + tau (action_dim)
  ↓
Linear(input_dim, 256) + LeakyReLU(0.2)
  ↓
Linear(256, 256) + LeakyReLU(0.2)
  ↓
Linear(256, 256) + LeakyReLU(0.2)
  ↓
Linear(256, num_muscles)
  ↓
Output: unnormalized activations
  ↓
forward_filter(): relu(tanh(x))
  ↓
Final: muscle activations [0, 1]
```

## 🔄 Weight Loading Flow

```
1. Python: torch.load('muscle.pt') → state_dict
2. Viewer: loadNetworkFromPath() → store in mMuscleStateDict
3. Viewer: initEnv() → call setMuscleNetworkWeight()
4. Environment: convert dict → C++ format
5. MuscleNN: load_state_dict() → weights loaded
6. calcActivation() → non-zero activations ✓
```

## 📚 Documentation

- **Full Guide**: `docs/MUSCLE_NETWORK_LIBTORCH_GUIDE.md`
- **Parallelization**: `docs/THREADPOOL_PARALLELIZATION_SUMMARY.md`
- **Complete Summary**: `docs/LIBTORCH_MIGRATION_COMPLETE.md`

## 🚀 Next Steps

1. **Batch GPU Inference**: Collect all tensors → single GPU forward → 5-10x speedup
2. **Profile Bottlenecks**: Use `perf` to find DART collision hot spots
3. **Distributed Scaling**: MPI/Ray for multi-machine training

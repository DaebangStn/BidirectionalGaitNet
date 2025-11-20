# Muscle Network Dual-Device Architecture

## Overview
Migration from Ray RLlib to CleanRL with optimized device placement for hierarchical muscle control.

## Device Strategy

### GPU Training (Python MuscleLearner)
**Location**: `ppo/muscle_learner.py`
**Device**: CUDA (if available)
**Purpose**: Fast supervised learning on collected muscle tuples

```python
muscle_learner = MuscleLearner(
    num_actuator_action=num_actuator_action,
    num_muscles=num_muscles,
    num_muscle_dofs=num_muscle_dofs,
    device="cuda"  # GPU for training
)
```

**Training Process**:
1. Collect muscle tuples from all environments (CPU → GPU transfer)
2. Train MuscleNN on GPU with Adam optimizer
3. Convert weights to CPU for distribution: `get_state_dict()` → CPU tensors

### CPU Simulation (C++ Environments)
**Location**: `sim/Environment.cpp`
**Device**: CPU only (`is_cpu=True`)
**Purpose**: Thread-safe parallel rollouts without CUDA context overhead

```cpp
// Line 300, 737
mMuscleNN = py::module::import("python.ray_model")
    .attr("generating_muscle_nn")(
        character->getNumMuscleRelatedDof(),
        getNumActuatorAction(),
        character->getNumMuscles(),
        true,  // is_cpu=True - Forces CPU execution
        mUseCascading
    );
```

**Per-Environment Benefits**:
- No CUDA context creation (avoids multi-GPU overhead)
- Independent CPU-based MuscleNN per environment
- Thread-safe for AsyncVectorEnv with 96 parallel workers
- No device synchronization bottlenecks

## Weight Transfer Flow

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING PHASE (GPU)                                    │
├─────────────────────────────────────────────────────────┤
│ 1. MuscleLearner.model (GPU)                           │
│ 2. Train on muscle tuples                              │
│ 3. get_state_dict() → {k: v.cpu() for k,v in ...}     │
└─────────────────────────────────────────────────────────┘
                        ↓
                  CPU tensors
                        ↓
┌─────────────────────────────────────────────────────────┐
│ DISTRIBUTION PHASE (Python → C++)                      │
├─────────────────────────────────────────────────────────┤
│ 1. envs.call('update_muscle_weights', state_dict)     │
│ 2. GymEnvManager.update_muscle_weights(py::dict)      │
│ 3. Environment.setMuscleNetworkWeight(weights)        │
│ 4. mMuscleNN.attr("load_state_dict")(weights)         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ SIMULATION PHASE (CPU x96 environments)                │
├─────────────────────────────────────────────────────────┤
│ Each environment:                                       │
│ - mMuscleNN (CPU-only)                                 │
│ - unnormalized_no_grad_forward() on CPU               │
│ - No CUDA synchronization                             │
│ - Independent parallel execution                       │
└─────────────────────────────────────────────────────────┘
```

## Code Changes

### 1. MuscleLearner Weight Conversion (ppo/muscle_learner.py:259-266)
```python
def get_state_dict(self) -> Dict:
    """
    Get model state dict with CPU tensors for C++ environment update.

    Note: Converts GPU tensors to CPU for distribution to C++ simulation environments.
    Training happens on GPU, but C++ environments require CPU tensors.
    """
    return {k: v.cpu() for k, v in self.model.state_dict().items()}
```

**Key Change**: Added `.cpu()` conversion to ensure tensors are on CPU before distribution.

### 2. No Changes Required in C++
- `Environment.cpp` already uses `is_cpu=True` (lines 300, 737)
- `GymEnvManager.update_muscle_weights()` already exists (line 157)
- `Environment.setMuscleNetworkWeight()` accepts `py::object` (line 190)

### 3. CleanRL Integration (ppo/ppo_hierarchical.py:211-224)
```python
muscle_learner = MuscleLearner(
    num_actuator_action=num_actuator_action,
    num_muscles=num_muscles,
    num_muscle_dofs=num_muscle_dofs,
    learning_rate=args.muscle_lr,
    num_epochs=args.muscle_num_epochs,
    batch_size=args.muscle_batch_size,
    is_cascaded=use_cascading,
    device="cuda" if torch.cuda.is_available() and args.cuda else "cpu"
)

# Initialize muscle weights in all environments
state_dict = muscle_learner.get_state_dict()  # GPU → CPU conversion happens here
envs.call('update_muscle_weights', state_dict)
```

## Performance Benefits

### GPU Training
- **Batch Processing**: Large minibatches (512) on GPU
- **Fast Gradients**: Parallel backward pass across all samples
- **Optimized CUDA Kernels**: cuBLAS for matrix operations

### CPU Simulation
- **No Context Overhead**: Each environment thread doesn't need CUDA context
- **Memory Efficiency**: No GPU memory per environment (96 envs × 0 MB = 0 MB)
- **Parallelism**: True multi-core parallelism without GPU contention
- **Simplicity**: No device synchronization or tensor device management in rollout loop

## Verification Checklist

✅ **GPU Training**:
- MuscleLearner initializes on CUDA when available
- Training tensors moved to GPU in `learn()` method
- Optimizer operates on GPU tensors

✅ **CPU Simulation**:
- Environment.cpp passes `is_cpu=True` to MuscleNN
- Each environment creates independent CPU-based network
- No CUDA calls in simulation loop

✅ **Weight Transfer**:
- `get_state_dict()` converts GPU → CPU tensors
- C++ `update_muscle_weights()` accepts Python dict
- `load_state_dict()` works with CPU tensors on CPU-based MuscleNN

## Testing Commands

### Build
```bash
ninja -C build/release
```

### Run Hierarchical Training
```bash
python ppo/ppo_hierarchical.py --env_file data/env/A2_sep.yaml
```

### Verify Device Placement
```python
# In ppo_hierarchical.py, add after line 220:
print(f"MuscleLearner device: {muscle_learner.device}")
print(f"MuscleLearner model device: {next(muscle_learner.model.parameters()).device}")
# Expected: cuda:0

# In Environment.cpp, verify line 300 has:
# true,  // is_cpu=True
```

## Migration Status

| Component | Status | Notes |
|-----------|--------|-------|
| C++ Environment | ✅ No changes needed | Already CPU-only |
| Python MuscleLearner | ✅ Modified | Added CPU conversion in `get_state_dict()` |
| Weight Distribution | ✅ Compatible | Existing `update_muscle_weights()` works |
| CleanRL Integration | ✅ Complete | GPU training, CPU simulation |

## Key Insights

1. **Ray → CleanRL**: No breaking changes because CleanRL reuses `python.ray_model.MuscleNN`
2. **Device Separation**: Training (GPU) and simulation (CPU) are completely decoupled
3. **Interface Preservation**: All C++ interface methods (`unnormalized_no_grad_forward`, `forward_filter`, `load_state_dict`) remain compatible
4. **Single Code Change**: Only needed `.cpu()` in `get_state_dict()` for weight distribution

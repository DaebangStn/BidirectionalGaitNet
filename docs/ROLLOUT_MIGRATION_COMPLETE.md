# Rollout System Documentation

## Overview
Ray-based distributed rollout system with HDF5 output and viewer integration for motion visualization.

## Architecture

### Core Components
```
Environment (simulation) → RolloutEnvironment (data collection) → HDF5 output
                        → RenderEnvironment (visualization) → Viewer
```

**RolloutEnvironment** (`rollout/RolloutEnvironment.{h,cpp}`)
- Wraps Environment for data collection
- YAML-configurable field recording
- Parameter sweep support via `set_parameters()`
- Python bindings: `pyrollout` module

**RenderEnvironment** (`viewer/RenderEnvironment.{h,cpp}`)
- Wraps Environment for GLFWApp visualization
- Manages graph data buffers
- Separates rendering from simulation

**RolloutRecord** (`rollout/RolloutRecord.{h,cpp}`)
- Dynamic data buffer (Eigen::MatrixXd)
- YAML-based field configuration
- Records: angles, velocities, contacts, GRF, phase, cycle

## Ray Rollout System

### Workers
- **PolicyWorker** (1 GPU): Batched inference for all environments
- **EnvWorker** (N CPUs): Parallel simulation and data recording

### Usage
```bash
# Basic rollout
python python/ray_rollout.py \
    --checkpoint ray_results/checkpoint_dir \
    --config data/rollout/angle.yaml \
    --workers 16

# With parameter sweep
python python/ray_rollout.py \
    --checkpoint ray_results/checkpoint_dir \
    --config data/rollout/angle.yaml \
    --param-file data/U500_k0.csv \
    --workers 16
```

### Configuration (YAML)
```yaml
record:
  angle: {enabled: true, hip: true, knee: true, ankle: true}
  velocity: {enabled: true, hip: true, knee: true}
  foot: true
sample:
  cycle: 5  # Target gait cycles
```

### Parameter Sweep (CSV)
```csv
param_idx,cadence,stride
0,0.4000,0.4000
1,0.4000,0.4476
```
Column names must match environment parameter names.

## HDF5 Output Structure

```
rollout_data.h5
├── param_0/
│   ├── cycle_0/
│   │   ├── motions [59, 56]      # Full state vectors (variable steps)
│   │   ├── phase [59]
│   │   ├── angle/{HipR, KneeR, AnkleR}
│   │   ├── anvel/{...}           # Angular velocities
│   │   ├── contact/{left, right}
│   │   ├── grf/{left, right}
│   │   └── root/{x, y, z}
│   ├── cycle_1/ [320, 56]        # Different cycle length
│   └── ...
├── param_1/
└── ...
```

**Key Features:**
- Variable cycle lengths (e.g., 59, 320, 408 steps per cycle)
- Hierarchical: `param_idx → cycle_idx → data`
- Compressed (gzip-4), supports numpy/pandas/polars

## Viewer Integration

### Motion Struct
```cpp
struct Motion {
    std::string name, source_type;  // "npz", "hdf5", "simulation"
    Eigen::VectorXd param, motion;
    int frames_per_cycle = 101;     // NPZ: 101, HDF5: variable
    int num_cycles = 60;            // NPZ: 60, HDF5: variable
};
```

### Loading
- **NPZ**: `motions/*.npz` (fixed 101 frames × 60 cycles)
- **HDF5**: `sampled/**/rollout_data.h5` (variable lengths)
- Automatic on viewer initialization via `loadMotionFiles()` + `loadHDF5MotionFiles()`

### Playback
Generalized interpolation in `drawPlayableMotion()`:
- Adapts to arbitrary `frames_per_cycle` and `num_cycles`
- No hardcoded magic numbers (removed all `101` references)
- Backward compatible with NPZ files

## File Organization

### Created (11 files)
- `rollout/` - RolloutRecord, RolloutEnvironment, Python bindings, CMakeLists
- `viewer/RenderEnvironment.{h,cpp}`
- `python/ray_rollout.py`

### Modified (3 files)
- `viewer/GLFWApp.{h,cpp}` - Motion struct, HDF5 loader, generalized interpolation
- `sim/Environment.{h,cpp}` - Removed pGraphData (moved to wrappers)

## Key Design Decisions

1. **Separation of concerns**: Environment (sim) / RolloutEnvironment (data) / RenderEnvironment (viz)
2. **Single GPU context**: PolicyWorker centralizes inference
3. **Variable cycle lengths**: HDF5 supports arbitrary gait cycle durations
4. **No magic numbers**: Dynamic sizing based on Motion struct fields
5. **Dual format support**: NPZ (legacy) + HDF5 (new) coexist seamlessly

## Build & Test

```bash
ninja -C build/release                    # Build C++ components
python -c "from pyrollout import *"       # Verify Python bindings
./build/release/viewer/viewer <metadata>  # Test viewer with motions
```

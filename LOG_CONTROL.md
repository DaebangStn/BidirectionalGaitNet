# Complete Log Control Guide

## Overview
Unified log level system across C++ and Python code to control output verbosity.

## Quick Start

### C++ Log Control
Edit `viewer/Log.h` line 17:
```cpp
#define LOG_LEVEL LOG_LEVEL_WARN  // Change INFO to WARN
```

### Python Log Control
Set environment variable before running:
```bash
export LOG_LEVEL=1  # 0=SILENT, 1=WARN, 2=INFO, 3=VERBOSE
./build/release/viewer/viewer data/trained_nn/merge_no_mesh_lbs
```

## Log Levels

| Level | Value | C++ Constant | Python | What Shows |
|-------|-------|--------------|--------|------------|
| **SILENT** | 0 | `LOG_LEVEL_SILENT` | `LOG_LEVEL_SILENT` | Nothing |
| **WARN** | 1 | `LOG_LEVEL_WARN` | `LOG_LEVEL_WARN` | Warnings only |
| **INFO** | 2 | `LOG_LEVEL_INFO` | `LOG_LEVEL_INFO` | Summaries (default) |
| **VERBOSE** | 3 | `LOG_LEVEL_VERBOSE` | `LOG_LEVEL_VERBOSE` | All details |

## What Gets Logged at Each Level

### SILENT (0)
- No output
- Use for batch processing, automated tests

### WARN (1)
**C++ Warnings:**
- `[data/motion/walk.h5] Warning: No parameters in motion file`
- `[NPZ] Warning: Parameter count mismatch`
- `[HDF] Warning: Expected X DOF, got Y`
- `[Camera] Preset X is not valid`

**Python Warnings:**
- Similar warning messages from Python modules

**Output Example:**
```
[Log] Level: WARN (1)
[data/motion/walk.h5] Warning: No parameters in motion file
[data/motion/shortLleg_mean.h5] Warning: No parameters in motion file
```

### INFO (2) - DEFAULT
**All warnings, plus summaries:**
- `[HDF Single] Loading 4 extracted HDF5 files...`
- `[HDF Single] Loaded data/motion/walk.h5 with 524 frames`
- `[BVH] Loading 5 BVH files into mMotions...`
- `[NPZ] Loaded data/npz_motions/Sim_Healthy.npz with 60 frames (279 parameters)`

**Output Example:**
```
[Log] Level: INFO (2)
[HDF Single] Loading 4 extracted HDF5 files...
[HDF Single] Loaded data/motion/walk.h5 with 524 frames
[data/motion/walk.h5] Warning: No parameters in motion file
[BVH] Loading 5 BVH files into mMotions...
[NPZ] Loaded data/npz_motions/Sim_Healthy.npz with 60 frames (279 parameters)
```

### VERBOSE (3)
**All messages including detailed diagnostics:**
- `[URIResolver] Initialized with data root: /home/geon/...`
- `[Python] URIResolver initialized with data root: /home/geon/...`
- `[Config] Loading render config from: render.yaml`
- `[Config] Loaded - Window: 2560x1440, Control: 250, Plot: 350, ...`
- `[Camera] Loading camera preset 0: Frontal view`
- `[DARTHelper] Building skeleton from file : /home/geon/.../skeleton.xml`
- `[Character] Using Muscle Path: /home/geon/.../muscle_anchor_all.xml`
- `[Environment] BVH Path resolved: @data/motion/walk.bvh -> /home/geon/.../walk.bvh`
- `[Python] Loading network from ray_results/ckpt-005000-1020_130406`
- `[HDF] Loaded /home/geon/.../walk.h5 with 524 frames (56 DOF/frame, 0.00208333 s/frame)`
- All frame counts, parameter details, calibration values

**Output Example:**
```
[Log] Level: VERBOSE (3)
[Log] Python Level: VERBOSE (3)
[URIResolver] Initialized with data root: /home/geon/BidirectionalGaitNet/data
[Config] Loading render config from: render.yaml
[Config] Loaded - Window: 2560x1440, Control: 250, Plot: 350, Rollout: 5, ...
[Python] URIResolver initialized with data root: /home/geon/BidirectionalGaitNet/data
[Camera] Loading camera preset 0: Frontal view
[DARTHelper] Building skeleton from file : /home/geon/.../skeleton.xml
[Character] Using Muscle Path: /home/geon/.../muscle_anchor_all.xml
[Python] Loading network from ray_results/ckpt-005000-1020_130406
[HDF] Loaded /home/geon/.../walk.h5 with 524 frames (56 DOF/frame, 0.00208333 s/frame)
... (all details)
```

## Usage Examples

### Example 1: Quiet Mode for Demos
```bash
# Edit viewer/Log.h, change line 17 to:
#define LOG_LEVEL LOG_LEVEL_WARN

# Set Python log level
export LOG_LEVEL=1

# Run viewer - only warnings shown
ninja -C build/release && ./build/release/viewer/viewer data/trained_nn/merge_no_mesh_lbs
```

### Example 2: Normal Operation (Default)
```bash
# C++ already defaults to INFO in viewer/Log.h
# Python defaults to INFO if LOG_LEVEL not set

# Run viewer - essential summaries shown
./build/release/viewer/viewer data/trained_nn/merge_no_mesh_lbs
```

### Example 3: Debugging with Full Details
```bash
# Edit viewer/Log.h, change line 17 to:
#define LOG_LEVEL LOG_LEVEL_VERBOSE

# Set Python log level
export LOG_LEVEL=3

# Rebuild and run - all details shown
ninja -C build/release && ./build/release/viewer/viewer data/trained_nn/merge_no_mesh_lbs
```

### Example 4: Silent Mode for Batch Processing
```bash
# Edit viewer/Log.h, change line 17 to:
#define LOG_LEVEL LOG_LEVEL_SILENT

# Set Python log level
export LOG_LEVEL=0

# Run with no output (except errors to stderr)
ninja -C build/release && ./build/release/viewer/viewer data/trained_nn/merge_no_mesh_lbs
```

## Files Modified

### C++ Files
- `viewer/Log.h` - Log system header with level control
- `viewer/main.cpp` - Prints log level at startup
- `viewer/GLFWApp.cpp` - Config, Camera, motion loading logs
- `sim/DARTHelper.cpp` - Skeleton building logs
- `sim/Environment.cpp` - Path resolution logs
- `sim/Character.cpp` - Muscle path logs
- `sim/HDF.cpp`, `sim/NPZ.cpp`, `sim/HDFRollout.cpp`, `sim/BVH_Parser.cpp` - Motion loading

### Python Files
- `python/log_config.py` - Python log system (NEW)
- `python/uri_resolver.py` - URI resolution logs
- `python/ray_model.py` - Network loading logs
- `python/ray_rollout.py` - Rollout network loading logs

## Documentation
- `viewer/LOG_USAGE.md` - Original usage guide
- `viewer/LOG_CATEGORIES.md` - Detailed log categorization
- `LOG_CONTROL.md` - This file (complete guide)

## Notes

### Errors Always Shown
Error messages always appear via `stderr` regardless of log level:
```cpp
std::cerr << "[FATAL] Critical error" << std::endl;  // Always shown
```

### BVH Files
BVH "No parameters" warnings removed - BVH format doesn't support parameters by design.

### Log Level Status
At startup, both C++ and Python print their current log level:
```
[Log] Level: INFO (2)
[Log] Python Level: INFO (2)
```

### Gym Deprecation Warning
The Gym deprecation warning is not controlled by this system - it comes from the gym library itself.

# Log System Usage

## Overview
A centralized logging system for controlling output verbosity in motion loading and simulation code.

## Log Levels

The system supports 4 log levels (defined in `viewer/Log.h`):

| Level | Value | Description | What Shows |
|-------|-------|-------------|------------|
| `LOG_LEVEL_SILENT` | 0 | No logs | Nothing |
| `LOG_LEVEL_WARN` | 1 | Warnings only | Parameter mismatches, missing data |
| `LOG_LEVEL_INFO` | 2 | Normal operation (default) | File loading summaries + warnings |
| `LOG_LEVEL_VERBOSE` | 3 | Detailed debugging | Frame counts, parameters, all details |

## Changing Log Level

### Method 1: Edit Log.h (Recommended)
Edit `viewer/Log.h` line 13:
```cpp
#define LOG_LEVEL LOG_LEVEL_WARN  // Only warnings
```

### Method 2: Compiler Flag
Add to CMakeLists.txt or command line:
```bash
-DLOG_LEVEL=LOG_LEVEL_WARN
```

### Method 3: Per-file Override
In a specific .cpp file before including Log.h:
```cpp
#define LOG_LEVEL LOG_LEVEL_VERBOSE
#include "Log.h"
```

## Available Macros

```cpp
LOG_VERBOSE(msg)  // Detailed debug info (level 3+)
LOG_INFO(msg)     // Standard info (level 2+)
LOG_WARN(msg)     // Warnings (level 1+)
```

## Example Usage

```cpp
#include "Log.h"

// Verbose details (only shown at LOG_LEVEL_VERBOSE)
LOG_VERBOSE("[HDF] Loaded " << frames << " frames with " << dof << " DOF/frame");

// Standard info (shown at LOG_LEVEL_INFO and above)
LOG_INFO("[BVH] Loaded " << filename << " with " << count << " frames");

// Warnings (shown at LOG_LEVEL_WARN and above)
LOG_WARN("[NPZ] Warning: Parameter count mismatch (NPZ: " << npz_count
         << ", Environment: " << env_count << ")");

// Errors always use std::cerr (not filtered)
std::cerr << "[FATAL] Critical error occurred" << std::endl;
```

## Before/After Comparison

### Before (Always verbose):
```
[URIResolver] Initialized with data root: /home/geon/BidirectionalGaitNet/data
[Config] Loading render config from: render.yaml
[Config] Reset phase set to: 0
[Camera] Loading camera preset 0: Frontal view
[DARTHelper] Building skeleton from file : /home/geon/.../skeleton.xml
[DARTHelper] Building skeleton from file : /home/geon/.../skeleton.xml
[Character] Using Muscle Path: /home/geon/.../muscle_anchor_all.xml
[Environment] BVH Path resolved: @data/motion/walk.bvh -> /home/geon/.../walk.bvh
[HDF Single] Loading 4 extracted HDF5 files...
[HDF] Loaded data/motion/walk.h5 with 524 frames (56 DOF/frame, 0.00208333 s/frame)
[HDF Single] Loaded data/motion/walk.h5 with 524 frames
[data/motion/walk.h5] Warning: No parameters in motion file
[BVH] Loading 5 BVH files into mMotions...
[data/motion/walk_rswing.bvh] Warning: No parameters in motion file  ← REMOVED
[data/motion/backflip.bvh] Warning: No parameters in motion file      ← REMOVED
... (very verbose)
```

### After (LOG_LEVEL_WARN):
```
[data/motion/walk.h5] Warning: No parameters in motion file
[data/motion/shortLleg_mean.h5] Warning: No parameters in motion file
```

### After (LOG_LEVEL_INFO - default):
```
[HDF Single] Loading 4 extracted HDF5 files...
[HDF Single] Loaded data/motion/walk.h5 with 524 frames
[data/motion/walk.h5] Warning: No parameters in motion file
[BVH] Loading 5 BVH files into mMotions...
[NPZ] Loaded data/npz_motions/Sim_Healthy.npz with 60 frames (279 parameters)
```

### After (LOG_LEVEL_VERBOSE):
```
[Config] Loading render config from: render.yaml
[Config] Reset phase set to: 0
[Camera] Loading camera preset 0: Frontal view
[DARTHelper] Building skeleton from file : /home/geon/.../skeleton.xml
[Character] Using Muscle Path: /home/geon/.../muscle_anchor_all.xml
[Environment] BVH Path resolved: @data/motion/walk.bvh -> /home/geon/.../walk.bvh
[HDF] Loaded data/motion/walk.h5 with 524 frames (56 DOF/frame, 0.00208333 s/frame)
[BVH] Loaded data/motion/walk_rswing.bvh with 131 frames
... (all details shown)
```

**Key Improvements:**
- BVH "No parameters" warnings removed (BVH format doesn't support parameters)
- Initialization logs (Config, Camera, DARTHelper, etc.) now controlled by log level
- Clear distinction between INFO (summaries) and VERBOSE (all details)

## Files Using Log System

- `viewer/Log.h` - Log macro definitions
- `viewer/GLFWApp.cpp` - Main viewer with motion loading UI
- `sim/NPZ.cpp` - NPZ motion file loader
- `sim/HDF.cpp` - HDF5 single motion file loader
- `sim/HDFRollout.cpp` - HDF5 rollout motion loader
- `sim/BVH_Parser.cpp` - BVH motion file parser

## Recommendations

- **Development**: Use `LOG_LEVEL_VERBOSE` to see all loading details
- **Normal use**: Use `LOG_LEVEL_INFO` (default) for essential summaries
- **Production/demos**: Use `LOG_LEVEL_WARN` to minimize output
- **Silent mode**: Use `LOG_LEVEL_SILENT` when running batch processes

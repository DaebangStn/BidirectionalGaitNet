# Log Message Categories

This document shows which messages appear at each log level.

## LOG_LEVEL_SILENT (0)
**No output** - Completely silent operation

---

## LOG_LEVEL_WARN (1)
**Warnings only** - Only shows potential issues that need attention

### Motion Loading Warnings
- `[data/motion/walk.h5] Warning: No parameters in motion file`
- `[NPZ] Warning: Parameter count mismatch (NPZ: X, Environment: Y)`
- `[HDF] Warning: Expected X DOF per frame, got Y`
- `[HDF] Warning: No frames loaded`
- `[HDF] Warning: Found parameter_names but no param_state`

### System Warnings
- `[Camera] Preset X is not valid`
- `[Config] Warning: Could not load render.yaml: ...`

### Notes
- **BVH "No parameters" warning removed** - BVH format doesn't support parameters
- Errors always shown via stderr regardless of log level

---

## LOG_LEVEL_INFO (2) - DEFAULT
**Essential summaries** - Shows what's happening without overwhelming detail

### All warnings from LOG_LEVEL_WARN, plus:

### Motion Loading Summaries
- `[HDF Single] Loading 4 extracted HDF5 files...`
- `[HDF Single] Loaded data/motion/walk.h5 with 524 frames`
- `[BVH] Loading 5 BVH files into mMotions...`
- `[NPZ] Loaded data/npz_motions/Sim_Healthy.npz with 60 frames (279 parameters)`
- `[Motion] Found 3 HDF5 rollout files`

### Key Operations
- File loading confirmations (without full paths)
- Summary counts (X files loaded, Y frames total)
- Important state changes

---

## LOG_LEVEL_VERBOSE (3)
**All details** - Full diagnostic information for debugging

### All messages from LOG_LEVEL_INFO, plus:

### Initialization Details
- `[URIResolver] Initialized with data root: /home/geon/...`
- `[Config] Loading render config from: render.yaml`
- `[Config] Reset phase set to: 0`
- `[Config] Loaded - Window: 2560x1440, Control: 250, Plot: 350, ...`
- `[Camera] Loading camera preset 0: Frontal view`
- `[DARTHelper] Building skeleton from file : /home/geon/.../skeleton.xml`
- `[Character] Using Muscle Path: /home/geon/.../muscle_anchor_all.xml`
- `[Environment] BVH Path resolved: @data/motion/walk.bvh -> /home/geon/.../walk.bvh`

### Motion Loading Details
- `[HDF] Loaded /home/geon/.../data/motion/walk.h5 with 524 frames (56 DOF/frame, 0.00208333 s/frame)`
- `[HDF] Loaded 13 parameters`
- `[HDF] Matched 5 / 13 parameters (Environment has 13 parameters)`
- `[HDF] Height calibration applied: Y offset = 0.05`
- `[BVH] Loaded data/motion/walk_rswing.bvh with 131 frames`
- `[NPZ] Applied 279 parameters`

### Technical Details
- Full file paths
- Frame counts with DOF and timing info
- Parameter matching details
- Calibration values
- All diagnostic information

---

## Quick Reference

| What do you want to see? | Log Level |
|---------------------------|-----------|
| Nothing (automated/batch) | `SILENT` |
| Only warnings/problems | `WARN` |
| Essential summaries (default) | `INFO` |
| Everything for debugging | `VERBOSE` |

---

## Python Logs

**Note**: Python logs like `[Python] URIResolver initialized` and `[Python] Loading network` are **not controlled** by this C++ log system. They come from:
- `python/uri_resolver.py`
- `python/ray_model.py`
- `python/ray_rollout.py`

To control Python log verbosity, modify those Python files or add a similar log level system there.

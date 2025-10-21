# Parameter CSV Fix Summary

**Date**: October 21, 2025
**Issue**: Parameter CSV files causing memory errors and zero mass warnings
**Status**: ✅ FIXED

## Problem Description

When using parameter CSV files to modify skeleton parameters, the simulation would crash with:
- `MemoryError: std::bad_alloc`
- `Warning [BodyNode.cpp:501] A negative or zero mass [0] is set to BodyNode [FemurR]`
- Multiple warnings for various body parts (TibiaR, ArmL, ArmR, ForeArmL, ForeArmR, etc.)

This occurred even when trying to modify only specific body parts (e.g., shortening only the left leg).

## Root Cause

The `SetParameters()` function in `rollouts/RolloutEnvironment.cpp` was initializing all parameters to 0.0:

```cpp
// BEFORE (buggy):
Eigen::VectorXd param_state = Eigen::VectorXd::Zero(param_names.size());
```

When skeleton scaling parameters were set to 0.0, it meant "scale body part to zero size", resulting in zero mass. This caused the DART physics engine to fail with memory allocation errors.

## Solution

Changed the initialization to use default parameter values instead of zeros:

```cpp
// AFTER (fixed):
Eigen::VectorXd param_state = mEnv.getParamDefault();
```

This ensures unspecified parameters use their default values:
- Skeleton parameters: 1.0 (100% normal size)
- Torsion parameters: 0.0 (no torsion)
- Gait parameters: 1.0 (normal gait)

## Impact

### Before Fix
Users had to specify ALL parameters in the CSV file, even if they only wanted to modify one or two:

```csv
# Required all body parts to avoid zero mass errors
param_idx,skeleton_global,skeleton_FemurL,skeleton_TibiaL,skeleton_FemurR,skeleton_TibiaR,skeleton_ArmL,skeleton_ArmR,skeleton_ForeArmL,skeleton_ForeArmR
0,1.0,0.9,0.9,1.0,1.0,1.0,1.0,1.0,1.0
```

### After Fix
Users can specify only the parameters they want to modify:

```csv
# Only specify what you want to change
param_idx,skeleton_FemurL,skeleton_TibiaL
0,0.9,0.9
```

All other body parts automatically default to 1.0 (normal size).

## Files Changed

### Core Fix
- **`rollouts/RolloutEnvironment.cpp`** (lines 219-242)
  - Changed parameter initialization from `Zero()` to `getParamDefault()`
  - Simplified code by removing verbose debug logging after verification
  - Added clear comments explaining default value usage

### Documentation Created
- **`docs/parameter_files.md`** - Comprehensive parameter documentation
  - All 13 available parameters documented
  - CSV file format specification
  - Multiple examples for common use cases
  - Troubleshooting guide
  - Best practices

- **`data/rolloutParam/README.md`** - Quick reference guide
  - Example file descriptions
  - Template for creating custom files
  - Validation steps
  - Common troubleshooting

- **`data/rolloutParam/deprecated/README.md`** - Deprecation notices
  - Explanation of why old files were deprecated
  - Migration guide for users
  - Historical context

- **`docs/PARAMETER_FIX_SUMMARY.md`** - This file
  - Complete summary of issue, fix, and impact
  - Technical details for maintainers

### Example Files Created/Updated
- **`data/rolloutParam/short_left_leg.csv`** - Minimal example (left leg only)
- **`data/rolloutParam/short_left_leg_with_global.csv`** - With global scaling
- **`data/rolloutParam/left_leg_progression.csv`** - Multiple parameter sets for batch testing
- **`data/rolloutParam/minimal_left_leg.csv`** - Kept as template
- **`data/rolloutParam/default_params.csv`** - Kept for baseline comparisons

### Files Deprecated
Moved to `data/rolloutParam/deprecated/`:
- `short_left_leg_complete.csv` - Contains invalid parameter names
- `short_left_leg_only.csv` - Redundant
- `short_skel_L.csv` - Incomplete debug file
- `short_skel_L_fixed.csv` - Temporary debug file
- `short_left_leg_noglobal.csv` - Duplicate

## Testing Performed

### Debug Testing
Added comprehensive debug logging to trace parameter matching:
```
[DEBUG] SetParameters: Total 13 parameters available
[DEBUG] SetParameters: Received 2 parameters from CSV
[DEBUG]   CSV param: skeleton_FemurL = 0.9
[DEBUG]   CSV param: skeleton_TibiaL = 0.9
[DEBUG] SetParameters: Initialized param_state with defaults (size=13)
[DEBUG]   Unmatched: param[0] 'gait_stride' = 1 (default)
[DEBUG]   Unmatched: param[1] 'gait_cadence' = 1 (default)
[DEBUG]   Unmatched: param[2] 'skeleton_global' = 1 (default)
[DEBUG]   Matched: param[3] 'skeleton_FemurL' = 0.9
[DEBUG]   Unmatched: param[4] 'skeleton_FemurR' = 1 (default)
[DEBUG]   Matched: param[5] 'skeleton_TibiaL' = 0.9
[DEBUG]   Unmatched: param[6] 'skeleton_TibiaR' = 1 (default)
...
```

This confirmed:
- Unspecified skeleton parameters default to 1.0 (not 0.0)
- Only CSV-specified parameters are overridden
- All body parts have valid mass

### Verification Testing
Tested with multiple configurations:
1. **Minimal left leg** (`skeleton_FemurL=0.9, skeleton_TibiaL=0.9`)
   - ✅ Completed successfully
   - ✅ No zero mass warnings
   - ✅ Right leg and all other body parts at normal size

2. **With global scaling** (`skeleton_global=0.95, skeleton_FemurL=0.9, skeleton_TibiaL=0.9`)
   - ✅ Completed successfully
   - ✅ Left leg at 90%, right leg at 95%, other parts at 95%

3. **Default parameters** (empty CSV)
   - ✅ Uses `updateParamState()` for sampling from default distribution

All tests produced valid HDF5 output with expected data structure.

## Available Parameters

The system supports 13 parameters (discovered via debug output):

### Gait Parameters (default: 1.0)
- `gait_stride` - Stride length multiplier
- `gait_cadence` - Step cadence multiplier

### Skeleton Scaling (default: 1.0)
- `skeleton_global` - Global skeleton scale
- `skeleton_FemurL` - Left femur (thigh)
- `skeleton_FemurR` - Right femur (thigh)
- `skeleton_TibiaL` - Left tibia (shin)
- `skeleton_TibiaR` - Right tibia (shin)
- `skeleton_ArmL` - Left upper arm
- `skeleton_ArmR` - Right upper arm
- `skeleton_ForeArmL` - Left forearm
- `skeleton_ForeArmR` - Right forearm

### Torsion (default: 0.0)
- `torsion_FemurL` - Left femur torsion angle
- `torsion_FemurR` - Right femur torsion angle

## Usage Examples

### Shorten Only Left Leg
```bash
python python/ray_rollout.py \
    --checkpoint ./ray_results/base_anchor_all-026000-1014_092619 \
    --config @data/rollout/angle.yaml \
    --param-file data/rolloutParam/short_left_leg.csv
```

### Multiple Parameter Sets
```bash
python python/ray_rollout.py \
    --checkpoint ./ray_results/base_anchor_all-026000-1014_092619 \
    --config @data/rollout/metabolic.yaml \
    --param-file data/rolloutParam/left_leg_progression.csv \
    --workers 4
```

## Best Practices

1. **Specify Only What You Need**: Only include parameters you want to modify
2. **Use Positive Values**: Skeleton parameters should be 0.1 to 2.0 for realistic results
3. **Test Incrementally**: Start with single worker to validate before scaling up
4. **Document Your Configs**: Use descriptive filenames
5. **Check Defaults**: Remember unspecified parameters use defaults (1.0 or 0.0)

## Backward Compatibility

### Breaking Change
This fix changes the behavior of parameter CSV files. Before the fix, unspecified parameters defaulted to 0.0. After the fix, they default to 1.0 (for skeleton) or 0.0 (for torsion).

### Migration
If you have existing CSV files that worked before (by specifying all parameters), they will continue to work. However, you can now simplify them by removing parameters set to their default values.

**Example migration**:
```csv
# Before (all parameters specified to avoid zeros)
param_idx,skeleton_global,skeleton_FemurL,skeleton_TibiaL,skeleton_FemurR,skeleton_TibiaR
0,1.0,0.9,0.9,1.0,1.0

# After (only specify what changes)
param_idx,skeleton_FemurL,skeleton_TibiaL
0,0.9,0.9
```

## Related Issues

- **Zero Mass Warnings**: Fixed by using `getParamDefault()` instead of `Zero()`
- **Memory Allocation Errors**: Caused by zero mass physics failures, fixed by valid defaults
- **Parameter Confusion**: Resolved with comprehensive documentation

## Future Improvements

Potential enhancements (not implemented):
1. Parameter validation in Python before sending to C++
2. Warning when CSV contains unrecognized parameter names
3. Automatic parameter range validation (e.g., skeleton scales > 0)
4. HDF5 metadata recording which parameters were explicitly set vs defaulted

## References

- [docs/parameter_files.md](parameter_files.md) - Complete parameter documentation
- [data/rolloutParam/README.md](../data/rolloutParam/README.md) - Example files guide
- [rollouts/RolloutEnvironment.cpp:219-242](../rollouts/RolloutEnvironment.cpp#L219-L242) - Implementation

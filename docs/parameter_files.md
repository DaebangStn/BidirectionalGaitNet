# Parameter CSV Files

This document explains how to create and use parameter CSV files for rollout simulations.

## Overview

Parameter CSV files allow you to specify custom values for simulation parameters like skeleton scaling, gait characteristics, and muscle properties. You can provide parameter files to `ray_rollout.py` using the `--param-file` flag.

## Available Parameters

The system supports 13 parameters (as of this documentation):

### Gait Parameters
- `gait_stride` - Stride length multiplier (default: 1.0)
- `gait_cadence` - Step cadence multiplier (default: 1.0)

### Skeleton Scaling Parameters
- `skeleton_global` - Global skeleton scale factor (default: 1.0)
- `skeleton_FemurL` - Left femur (thigh) scale (default: 1.0)
- `skeleton_FemurR` - Right femur (thigh) scale (default: 1.0)
- `skeleton_TibiaL` - Left tibia (shin) scale (default: 1.0)
- `skeleton_TibiaR` - Right tibia (shin) scale (default: 1.0)
- `skeleton_ArmL` - Left upper arm scale (default: 1.0)
- `skeleton_ArmR` - Right upper arm scale (default: 1.0)
- `skeleton_ForeArmL` - Left forearm scale (default: 1.0)
- `skeleton_ForeArmR` - Right forearm scale (default: 1.0)

### Torsion Parameters
- `torsion_FemurL` - Left femur torsion angle (default: 0.0)
- `torsion_FemurR` - Right femur torsion angle (default: 0.0)

## CSV File Format

Parameter CSV files must have:
1. A header row with `param_idx` as the first column, followed by parameter names
2. One or more data rows with parameter index and values

**Important**: You only need to specify the parameters you want to modify. Unspecified parameters will use their default values.

### Example 1: Shorten Left Leg Only

To simulate a person with a shorter left leg (90% normal length):

```csv
param_idx,skeleton_FemurL,skeleton_TibiaL
0,0.9,0.9
```

This sets:
- Left femur: 90% of normal length
- Left tibia: 90% of normal length
- All other body parts: 100% (normal size) - **automatically defaulted**

### Example 2: Shorten Left Leg with Global Scaling

To simulate a shorter person with an even shorter left leg:

```csv
param_idx,skeleton_global,skeleton_FemurL,skeleton_TibiaL
0,0.95,0.9,0.9
```

This sets:
- Global scale: 95% (entire skeleton scaled down to 95%)
- Left femur: 90% (overrides the global 95%)
- Left tibia: 90% (overrides the global 95%)
- Right femur/tibia: 95% (from skeleton_global)
- Arms and other parts: 95% (from skeleton_global)

### Example 3: Multiple Parameter Sets

You can test multiple configurations in one rollout:

```csv
param_idx,skeleton_FemurL,skeleton_TibiaL
0,1.0,1.0
1,0.95,0.95
2,0.9,0.9
3,0.85,0.85
```

This will run 4 separate simulations with progressively shorter left legs.

### Example 4: Gait Modification

To modify gait characteristics:

```csv
param_idx,gait_stride,gait_cadence
0,1.2,0.9
```

This sets:
- Stride length: 120% longer steps
- Cadence: 90% slower step frequency
- All skeleton parts: 100% (default normal size)

## Parameter Interaction

### Global vs. Individual Scaling

When `skeleton_global` is specified:
1. It applies to ALL skeleton parts first
2. Individual part parameters (like `skeleton_FemurL`) override the global value for those specific parts

Example:
```csv
param_idx,skeleton_global,skeleton_FemurL
0,0.8,0.9
```

Results in:
- `skeleton_FemurL`: 0.9 (explicitly set)
- `skeleton_FemurR`: 0.8 (from skeleton_global)
- `skeleton_TibiaL`: 0.8 (from skeleton_global)
- `skeleton_TibiaR`: 0.8 (from skeleton_global)
- All arms: 0.8 (from skeleton_global)

## Default Values

**Critical**: The parameter system uses intelligent defaults:

- **Skeleton parameters**: Default to 1.0 (100% normal size)
- **Torsion parameters**: Default to 0.0 (no torsion)
- **Gait parameters**: Default to 1.0 (normal gait)

This means you can safely specify only the parameters you want to modify without worrying about other body parts getting zero mass or invalid physics.

## Usage Examples

### Basic Usage

```bash
python python/ray_rollout.py \
    --checkpoint ./ray_results/my_checkpoint \
    --config @data/rollout/angle.yaml \
    --param-file data/rolloutParam/short_left_leg.csv
```

### Multiple Workers

```bash
python python/ray_rollout.py \
    --checkpoint ./ray_results/my_checkpoint \
    --config @data/rollout/metabolic.yaml \
    --param-file data/rolloutParam/multiple_configs.csv \
    --workers 4
```

## Common Mistakes

### ❌ Wrong: Including Non-Existent Parameters

```csv
param_idx,skeleton_Pelvis,skeleton_FootL,skeleton_Head
0,1.0,1.0,1.0
```

These parameters don't exist in the system. Only use the 13 parameters listed above.

### ❌ Wrong: Expecting Zero-Valued Defaults

Before the fix in October 2025, unspecified parameters defaulted to 0.0, causing physics errors. Now they correctly default to 1.0 (or 0.0 for torsion).

### ✅ Correct: Minimal Specification

```csv
param_idx,skeleton_FemurL,skeleton_TibiaL
0,0.9,0.9
```

Specify only what you need to change. Everything else uses safe defaults.

## Verification

After running a rollout, you can verify the parameters were applied correctly by checking:

1. **HDF5 attributes**: Parameter values are stored in the HDF5 file
2. **Simulation behavior**: Observe if the character moves as expected
3. **Log output**: Check for any warnings about invalid masses or physics

## Technical Details

### Implementation

The parameter loading happens in:
- **Python side**: `python/ray_rollout.py` reads the CSV and passes parameters to workers
- **C++ side**: `rollouts/RolloutEnvironment.cpp` applies parameters via `SetParameters()`

The key implementation detail is that `SetParameters()` uses `mEnv.getParamDefault()` to initialize all parameters to their default values before applying CSV values, ensuring unspecified parameters have valid physics-safe defaults.

### Parameter State Vector

Internally, parameters are stored as an `Eigen::VectorXd` with 13 elements. The CSV values are mapped to this vector by parameter name.

## Troubleshooting

### Physics Errors / Zero Mass Warnings

If you see warnings like:
```
Warning [BodyNode.cpp:501] A negative or zero mass [0] is set to BodyNode [FemurR]
```

This indicates a parameter is set to 0.0 (or negative). Check your CSV file - you likely have a typo or invalid value.

### Memory Allocation Errors

```
MemoryError: std::bad_alloc
```

This can occur if invalid physics (zero mass) causes the simulation to fail. Verify all skeleton parameters are positive values (typically 0.1 to 2.0).

### No Data Generated

If rollout completes but produces no data:
1. Check the filter pipeline in your config YAML
2. Verify target cycles is appropriate (not too low)
3. Check that the simulation didn't fail early (check logs)

## Best Practices

1. **Start Simple**: Test with default parameters first (`--param-file` with empty CSV or minimal changes)
2. **Incremental Changes**: Modify one parameter at a time to understand its effect
3. **Realistic Values**: Keep skeleton scales between 0.5 and 1.5 for realistic human proportions
4. **Document Your Configs**: Use descriptive filenames like `short_left_leg_10pct.csv`
5. **Version Control**: Keep parameter files in git to track experimental configurations

## Related Documentation

- [Adding Attributes and Datasets](adding_attributes_and_datasets.md) - How to record custom data
- [Rollout Configuration](ROLLOUT_MIGRATION_COMPLETE.md) - General rollout system documentation
- [Filter Pipeline](adding_attributes_and_datasets.md#filter-pipeline) - Data processing filters

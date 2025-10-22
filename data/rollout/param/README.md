# Parameter CSV Files

This directory contains example parameter CSV files for rollout simulations.

## Quick Start

Use these files with `ray_rollout.py`:

```bash
python python/ray_rollout.py \
    --checkpoint ./ray_results/your_checkpoint \
    --config @data/rollout/angle.yaml \
    --param-file data/rolloutParam/short_left_leg.csv
```

## Example Files

### `short_left_leg.csv`
Simulates a person with shortened left leg (90% normal length).
- Left femur: 90%
- Left tibia: 90%
- Right leg: 100% (normal)
- All other body parts: 100% (default)

**Use case**: Asymmetric gait analysis, leg length discrepancy studies

### `short_left_leg_with_global.csv`
Simulates a shorter person (95%) with even shorter left leg (90%).
- Global scale: 95%
- Left femur: 90% (overrides global)
- Left tibia: 90% (overrides global)
- Right leg: 95% (from global)

**Use case**: Combined height and asymmetry analysis

### `minimal_left_leg.csv`
Same as `short_left_leg.csv` - minimal example showing only modified parameters.

**Use case**: Template for creating your own parameter files

### `short_left_leg_complete.csv` (DEPRECATED)
**Note**: This file contains invalid parameter names (Pelvis, FootL, etc.) that don't exist in the system. Do not use this file.

### `short_left_leg_only.csv` (DEPRECATED)
Earlier version with skeleton_global. Use `short_left_leg.csv` or `short_left_leg_with_global.csv` instead.

### `short_skel_L.csv` (OLD)
Incomplete file from debugging. Do not use.

### `default_params.csv`
Empty parameter set - uses all default values.
```csv
param_idx
0
```

**Use case**: Baseline comparison with default parameters

## Creating Your Own Parameter Files

### Template

```csv
param_idx,parameter_name_1,parameter_name_2
0,value1,value2
1,value1,value2
```

### Available Parameters

See [docs/parameter_files.md](../../docs/parameter_files.md) for complete documentation.

**Skeleton scaling** (default: 1.0):
- `skeleton_global` - All body parts
- `skeleton_FemurL`, `skeleton_FemurR` - Thighs
- `skeleton_TibiaL`, `skeleton_TibiaR` - Shins
- `skeleton_ArmL`, `skeleton_ArmR` - Upper arms
- `skeleton_ForeArmL`, `skeleton_ForeArmR` - Forearms

**Gait parameters** (default: 1.0):
- `gait_stride` - Stride length
- `gait_cadence` - Step frequency

**Torsion** (default: 0.0):
- `torsion_FemurL`, `torsion_FemurR` - Femur rotation

### Important Rules

1. **Only specify parameters you want to change** - unspecified parameters use safe defaults
2. **First column must be `param_idx`** - unique identifier for each parameter set
3. **Values should be positive** - especially for skeleton parameters (0.1 to 2.0 is realistic)
4. **One row per parameter set** - you can have multiple rows for batch testing

## Examples

### Single Parameter Modification

```csv
param_idx,skeleton_FemurL
0,0.85
```

Shortens only the left femur to 85%. Everything else stays at default (1.0).

### Multiple Parameter Sets

```csv
param_idx,skeleton_FemurL,skeleton_TibiaL
0,1.0,1.0
1,0.95,0.95
2,0.9,0.9
3,0.85,0.85
```

Runs 4 simulations with progressively shorter left legs.

### Bilateral Modification

```csv
param_idx,skeleton_FemurL,skeleton_FemurR,skeleton_TibiaL,skeleton_TibiaR
0,0.9,0.9,0.9,0.9
```

Shortens both legs symmetrically.

### Gait Modification

```csv
param_idx,gait_stride,gait_cadence
0,1.2,0.8
```

Longer, slower strides (120% stride length, 80% cadence).

## Validation

After creating a parameter file, test with a single worker first:

```bash
python python/ray_rollout.py \
    --checkpoint ./ray_results/your_checkpoint \
    --config @data/rollout/angle.yaml \
    --param-file data/rolloutParam/your_file.csv \
    --workers 1
```

Check for:
- No physics warnings (zero mass, invalid inertia)
- No memory allocation errors
- Reasonable simulation behavior
- Expected data output in HDF5 file

## Troubleshooting

**"Warning: A negative or zero mass [0] is set to BodyNode"**
- You likely have a skeleton parameter set to 0.0 or negative
- Check your CSV values - all skeleton scales should be positive

**"MemoryError: std::bad_alloc"**
- Physics engine failed due to invalid configuration
- Verify all skeleton parameters are reasonable (0.1 to 2.0)

**"Unknown parameter name"**
- Check parameter names match exactly (case-sensitive)
- See available parameters list above

**Simulation completes but behavior is unexpected**
- Verify parameter values are as intended
- Check if `skeleton_global` is interacting with individual parameters
- Review parameter interaction in [docs/parameter_files.md](../../docs/parameter_files.md)

## Reference

For complete documentation, see:
- [docs/parameter_files.md](../../docs/parameter_files.md) - Complete parameter documentation
- [docs/adding_attributes_and_datasets.md](../../docs/adding_attributes_and_datasets.md) - Data recording
- [docs/ROLLOUT_MIGRATION_COMPLETE.md](../../docs/ROLLOUT_MIGRATION_COMPLETE.md) - Rollout system overview

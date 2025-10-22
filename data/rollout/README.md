# Rollout System with Preset Support

This directory contains the rollout configuration system with preset support for convenient command execution.

## Quick Start

**Important**: Use `./scripts/rollout` instead of `./scripts/rollout` to ensure the correct micromamba environment with all dependencies (ray, torch, etc.) is used.

### List Available Presets
```bash
./scripts/rollout list
```

### Run a Rollout with Preset
```bash
# Using preset name
./scripts/rollout gait-metabolic --checkpoint ./ray_results/base_anchor_all-026000-1014_092619

# Using preset alias (shorter)
./scripts/rollout preset1 -c ./ray_results/base_anchor_all-026000-1014_092619

# Quick test with fewer workers
./scripts/rollout test -c ./ray_results/base_anchor_all-026000-1014_092619

# Production run with maximum workers
./scripts/rollout prod -c ./ray_results/base_anchor_all-026000-1014_092619
```

**Alternative**: If you have the micromamba environment activated:
```bash
# Activate environment first
eval "$(/opt/miniconda3/bin/micromamba shell hook --shell zsh)"
micromamba activate bidir

# Then use Python module directly
python -m python.rollout.rollout_cli list
python -m python.rollout.rollout_cli preset1 -c <checkpoint>
```

## Available Presets

### 1. `gait-metabolic` (alias: `preset1`)
Metabolic analysis with U225 gait parameters (225 variations)
- **Config**: `data/rollout/config/metabolic.yaml`
- **Workers**: 16
- **Parameters**: `data/rollout/param/gait_U225.csv`

**Example**:
```bash
./scripts/rollout preset1 -c ./ray_results/base_anchor_all-026000-1014_092619
```

### 2. `default-metabolic` (alias: `preset2`)
Metabolic analysis with default parameter set
- **Config**: `data/rollout/config/metabolic.yaml`
- **Workers**: 16
- **Parameters**: `data/rollout/param/default_params.csv`

**Example**:
```bash
./scripts/rollout preset2 -c ./ray_results/base_anchor_all-026000-1014_092619
```

### 3. `quick-test` (alias: `test`)
Quick test rollout with 4 workers and default params
- **Config**: `data/rollout/config/metabolic.yaml`
- **Workers**: 4
- **Parameters**: `data/rollout/param/default_params.csv`

**Example**:
```bash
./scripts/rollout test -c ./ray_results/my-checkpoint
```

### 4. `production` (alias: `prod`)
Production rollout with maximum workers (32) for U225 gait analysis
- **Config**: `data/rollout/config/metabolic.yaml`
- **Workers**: 32
- **Parameters**: `data/rollout/param/gait_U225.csv`

**Example**:
```bash
./scripts/rollout prod -c ./ray_results/base_anchor_all-026000-1014_092619
```

## Overriding Preset Values

You can override any preset value with command-line arguments:

```bash
# Use preset1 but with 32 workers instead of 16
./scripts/rollout preset1 -c <checkpoint> --workers 32

# Use preset1 but with a custom config file
./scripts/rollout preset1 -c <checkpoint> --config my_custom_config.yaml

# Use preset1 but with different parameters
./scripts/rollout preset1 -c <checkpoint> --param-file my_params.csv

# Use random sampling instead of parameter file
./scripts/rollout preset1 -c <checkpoint> --param-file "" --num-samples 100
```

## Preset Configuration File

Presets are defined in `data/rollout/presets.yaml`. You can edit this file to:
- Add new presets
- Modify existing presets
- Create new aliases

**Example preset definition**:
```yaml
presets:
  my-custom-preset:
    config: data/rollout/config/metabolic.yaml
    workers: 20
    param-file: data/rollout/param/my_params.csv
    sample-dir: ./sampled
    description: "My custom rollout configuration"

aliases:
  mypreset: my-custom-preset
```

## Command-Line Reference

### Required Arguments
- `preset`: Preset name or 'list' to show available presets
- `-c, --checkpoint`: Path to checkpoint directory (REQUIRED for rollout execution)

### Optional Arguments (Override Preset Values)
- `--config`: Override preset config YAML path
- `--workers`: Override preset worker count
- `--param-file`: Override preset parameter CSV file
- `--num-samples`: Number of random samples (when not using param-file)
- `--sample-dir`: Override preset sample directory

## Comparison: Before and After

### Before (Manual Command)
```bash
python python/ray_rollout.py \
  --checkpoint ./ray_results/base_anchor_all-026000-1014_092619 \
  --config data/rollout/config/metabolic.yaml \
  --workers 16 \
  --param-file data/rollout/param/gait_U225.csv
```

### After (With Preset)
```bash
./scripts/rollout preset1 -c ./ray_results/base_anchor_all-026000-1014_092619
```

**Benefits**:
- ✅ **Shorter commands**: 60% reduction in typing
- ✅ **No typos**: Preset values are validated
- ✅ **Reusable**: Same preset across different checkpoints
- ✅ **Discoverable**: `./scripts/rollout list` shows all options
- ✅ **Flexible**: Override any value when needed

## Directory Structure

```
data/rollout/
├── README.md              # This file
├── presets.yaml          # Preset definitions
├── config/               # Configuration YAML files
│   └── metabolic.yaml   # Metabolic rollout config
└── param/                # Parameter CSV files
    ├── gait_U225.csv    # U225 gait variations
    └── default_params.csv # Default parameters
```

## Output Location

Rollout results are saved in the `sample-dir` (default: `./sampled`) with the format:
```
./sampled/[checkpoint_name]+[config_name]+on_[timestamp]/rollout_data.h5
```

**Example**:
```
./sampled/base_anchor_all-026000+metabolic+on_20251023_143022/rollout_data.h5
```

## Advanced Usage

### Creating a Custom Workflow

1. Create a new parameter file in `data/rollout/param/`
2. Add a new preset in `data/rollout/presets.yaml`
3. Run with `./scripts/rollout <your-preset> -c <checkpoint>`

### Debugging Rollouts

Use the `quick-test` preset with fewer workers for faster iteration:
```bash
./scripts/rollout test -c <checkpoint> --workers 2
```

### Production Runs

Use the `production` preset for maximum throughput:
```bash
./scripts/rollout prod -c <checkpoint>
```

Monitor with Ray dashboard:
```bash
ray dashboard  # Usually at http://localhost:8265
```

## Code Organization

The rollout system is now organized in the `python/rollout/` module:

```
python/rollout/
├── __init__.py           # Module initialization and exports
├── rollout_cli.py        # CLI interface with preset support
├── ray_rollout.py        # Main rollout execution engine
├── rollout_worker.py     # Ray worker implementations
└── uniform_sampled.py    # Uniform sampling utilities
```

**Direct module usage** (for custom scripts):
```python
from python.rollout import run_rollout, create_sample_directory
from python.rollout import PolicyWorker, EnvWorker, FileWorker

# Run rollout programmatically
run_rollout(
    checkpoint_path="./ray_results/checkpoint",
    record_config_path="data/rollout/config/metabolic.yaml",
    output_path="./output/rollout_data.h5",
    ...
)
```

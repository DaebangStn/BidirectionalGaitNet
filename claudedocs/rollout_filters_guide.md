# Rollout Data Filters Guide

## Overview

The rollout pipeline now supports scalable, composable data filtering and processing. Filters can remove unwanted cycles or compute derived fields before saving to HDF5.

## Architecture

**FilterPipeline**: Orchestrates multiple filters sequentially
**DataFilter**: Base class for all filters (filter or transform cycles)
**Integration**: Applied in `FileWorker._flush()` before HDF5 write

## Available Filters

### 1. Drop Short Cycles
**Purpose**: Remove cycles with too few steps (incomplete/truncated cycles)

**Configuration**:
```yaml
filters:
  drop_short_cycles:
    enabled: true
    min_steps: 10
```

**Behavior**: Removes any cycle with `< min_steps` timesteps

---

### 2. Drop First N Cycles
**Purpose**: Remove initial transient cycles (warmup/settling behavior)

**Configuration**:
```yaml
filters:
  drop_first_n_cycles:
    enabled: true
    n: 1
```

**Behavior**: Drops first N cycles from each rollout

---

### 3. Drop Last M Cycles
**Purpose**: Remove final cycles (potentially unstable or incomplete)

**Configuration**:
```yaml
filters:
  drop_last_m_cycles:
    enabled: true
    m: 1
```

**Behavior**: Drops last M cycles from each rollout

---

### 4. Compute Travel Distance
**Purpose**: Add cumulative travel distance field from root position

**Configuration**:
```yaml
filters:
  compute_travel_distance:
    enabled: true
```

**Behavior**:
- Requires `root` position data in matrix_data
- Computes 2D distance (x-z plane) cumulative displacement
- Adds `travel_distance` field to scalar data

---

## Usage Examples

### Example 1: Enable Filters
Edit your config YAML (e.g., `data/rollout/angle.yaml`):
```yaml
sample:
  cycle: 5

filters:
  drop_short_cycles:
    enabled: true
    min_steps: 15
  drop_first_n_cycles:
    enabled: true
    n: 1
  compute_travel_distance:
    enabled: true
```

Run:
```bash
python ray_rollout.py \
  --checkpoint path/to/ckpt \
  --config data/rollout/angle.yaml \
  --workers 16
```

---

### Example 2: Different Filter Settings
Create multiple config files for different experiments:

`config_clean.yaml` (strict filtering):
```yaml
filters:
  drop_short_cycles:
    enabled: true
    min_steps: 20
  drop_first_n_cycles:
    enabled: true
    n: 2
```

`config_raw.yaml` (minimal filtering):
```yaml
filters:
  drop_short_cycles:
    enabled: true
    min_steps: 5
```

---

### Example 3: No Filters
Disable all filters (collect raw data):
```yaml
filters: {}
```

Or set all filters to `enabled: false`.

---

## Filter Execution Order

Filters execute in this sequence:
1. **drop_short_cycles**: Remove incomplete cycles
2. **drop_first_n_cycles**: Remove transient warmup
3. **drop_last_m_cycles**: Remove final unstable cycles
4. **compute_travel_distance**: Add derived fields
5. **Cycle Renumbering**: Reset cycle indices to 0, 1, 2, ... (automatic)

This order ensures:
- Invalid cycles removed first
- Slicing happens on valid cycles
- Derived computations use final filtered data
- Output has sequential cycle indices with no gaps

---

## Data Flow

```
EnvWorker.handle_done()
    ↓
FileWorker.write_rollout() → buffer
    ↓
FileWorker._flush() → FilterPipeline.apply()
    ↓
    Group by cycle (original indices)
    ↓
    Apply filters sequentially
    ↓
    Flatten back to arrays
    ↓
    Renumber cycles: 0, 1, 2, ... (sequential)
    ↓
save_to_hdf5() → HDF5 file
```

### Cycle Renumbering Example

**Scenario**: 30 cycles collected, with filters enabled

**Original cycles**: `[0, 1, 2, ..., 29]`

**After `drop_first_n_cycles: n=10`**:
Remaining: `[10, 11, 12, ..., 29]` (20 cycles)

**After `drop_last_m_cycles: m=5`**:
Remaining: `[10, 11, 12, ..., 24]` (15 cycles)

**After `drop_short_cycles: min_steps=10`** (assume cycle 15 was short):
Remaining: `[10, 11, 12, 13, 14, 16, 17, ..., 24]` (14 cycles)

**After automatic renumbering**:
Output cycles: `[0, 1, 2, 3, 4, 5, 6, ..., 13]` (14 cycles, sequential)

This ensures the HDF5 output always has:
- Sequential cycle indices starting from 0
- No gaps in cycle numbering
- Easy iteration and analysis

---

## Adding New Filters

To add a new filter:

### Step 1: Implement Filter Class
In `python/data_filters.py`:
```python
class MyNewFilter(DataFilter):
    def __init__(self, param: float):
        self.param = param

    def filter_cycles(self, cycles_dict, matrix_cycles_dict, fields):
        # Your filtering/processing logic
        filtered_cycles = {}
        for cycle_idx, cycle_data in cycles_dict.items():
            # Example: filter based on some condition
            if meets_condition(cycle_data):
                filtered_cycles[cycle_idx] = cycle_data

        return filtered_cycles, matrix_cycles_dict, fields
```

### Step 2: Add to FilterPipeline.from_config()
```python
# In FilterPipeline.from_config()
if config.get('my_new_filter', {}).get('enabled', False):
    param = config['my_new_filter'].get('param', 1.0)
    pipeline.add_filter(MyNewFilter(param))
```

### Step 3: Update Documentation
Document the new filter in this guide and update the example config YAML.

---

## Performance Considerations

**Memory**: Filters operate on buffered data (default 10 samples)
**CPU**: Cycle grouping/flattening is O(n) where n = num_steps
**I/O**: Filtered data reduces HDF5 file size and write time

**Optimization Tips**:
- Use `drop_short_cycles` early to reduce processing
- Compute derived fields last to avoid wasted computation
- Adjust `buffer_size` in FileWorker for memory/latency tradeoff

---

## Filter Statistics

After rollout completion, check:
- **Total samples**: Before filtering count
- **Success samples**: Successfully completed rollouts
- **Failed samples**: Early termination count

Note: Filter statistics (cycles dropped, etc.) not yet tracked but can be added to FileWorker.

---

## Troubleshooting

**Issue**: No data in output after filtering
**Solution**: Check filter thresholds (e.g., `min_steps` too high)

**Issue**: `travel_distance` not computed
**Solution**: Verify `root` position data exists in record config

**Issue**: Filters not applied
**Solution**: Check YAML syntax and `enabled: true` flags

---

## Future Extensions

Possible filter additions:
- **DropLowSpeedCycles**: Remove cycles below velocity threshold
- **ComputeEnergyMetrics**: Add metabolic cost calculations
- **ResampleCycles**: Normalize cycle lengths via interpolation
- **SmoothTrajectories**: Apply filtering to motion data
- **ComputeSymmetryMetrics**: Add left/right symmetry measures

Contributions welcome! Follow the "Adding New Filters" pattern above.

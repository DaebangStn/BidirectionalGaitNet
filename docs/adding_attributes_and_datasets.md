# Guide: Adding Attributes and Datasets to BidirectionalGaitNet

This manual explains how to add new data collection features to the rollout system.

## Table of Contents
1. [Data Categories](#data-categories)
2. [Adding Param-Level Attributes](#adding-param-level-attributes)
3. [Adding Cycle-Level Attributes](#adding-cycle-level-attributes)
4. [Computing Statistics of Cycle Attributes](#computing-statistics-of-cycle-attributes)
5. [Adding Time-Series Datasets](#adding-time-series-datasets)
6. [Adding Matrix Datasets](#adding-matrix-datasets)

---

## Data Categories

The rollout system organizes data into four categories:

### 1. **Param Attributes** (HDF5: `/param_{idx}/` attributes)
- **Purpose**: Properties that are constant for an entire parameter set
- **Examples**: `character_mass`, `success`, `skeleton_dof`
- **Storage**: HDF5 attributes on the param group
- **Scope**: One value per param_idx

### 2. **Cycle Attributes** (HDF5: `/param_{idx}/cycle_{idx}/` attributes)
- **Purpose**: Properties computed per gait cycle
- **Examples**: `num_steps`, `travel_distance`, `metabolic/cumulative/{TYPE}`, `metabolic/cot/{TYPE}`
- **Storage**: HDF5 attributes on each cycle group
- **Scope**: One value per cycle

### 3. **Time-Series Data** (HDF5: `/param_{idx}/cycle_{idx}/data` columns)
- **Purpose**: Per-timestep measurements during simulation
- **Examples**: `step`, `time`, `phase`, `grf/left`, `angle/HipR`
- **Storage**: Columns in the cycle data matrix
- **Scope**: One value per simulation timestep

### 4. **Matrix Data** (HDF5: `/param_{idx}/cycle_{idx}/motions`, etc.)
- **Purpose**: Multi-dimensional data arrays per cycle
- **Examples**: `motions` (full skeleton joint positions)
- **Storage**: Separate HDF5 datasets in cycle group
- **Scope**: N-dimensional array per cycle

---

## Adding Param-Level Attributes

Param attributes are **the easiest to add** - config-driven approach!

### Step 1: Add to YAML config

**File**: Your rollout config YAML (e.g., `data/rollout/metabolic.yaml`)

```yaml
# Environment attributes to save as param-level attributes
environment:
  character_mass: true    # Save character mass
  skeleton_dof: true      # ← ADD THIS
  simulation_hz: true     # ← ADD THIS
  control_hz: true        # ← ADD THIS
```

### Step 2: Add to EnvWorker (Python only)

**File**: `python/rollout_worker.py`

**Location**: `EnvWorker.__init__()` around line 380-392

```python
# Collect param-level attributes based on config
self.param_attributes = {}

# Add environment attributes if configured
env_config = (record_config or {}).get('environment', {})
if env_config.get('character_mass', False):
    self.param_attributes['character_mass'] = character_mass

# Add new attributes here - easy to extend based on config!
if env_config.get('skeleton_dof', False):              # ← ADD THIS
    self.param_attributes['skeleton_dof'] = self.rollout_env.get_skeleton_dof()
if env_config.get('simulation_hz', False):             # ← ADD THIS
    self.param_attributes['simulation_hz'] = self.rollout_env.get_simulation_hz()
if env_config.get('control_hz', False):                # ← ADD THIS
    self.param_attributes['control_hz'] = self.rollout_env.get_control_hz()
```

### Step 3: (Optional) Add C++ getter if needed

If the attribute requires a new C++ method:

**File**: `rollouts/RolloutEnvironment.h`
```cpp
class RolloutEnvironment {
public:
    // ... existing methods ...
    int GetSkeletonDOF() const;  // ← ADD DECLARATION
};
```

**File**: `rollouts/RolloutEnvironment.cpp`
```cpp
int RolloutEnvironment::GetSkeletonDOF() const {
    return const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getNumDofs();
}
```

**File**: `rollouts/PyRolloutEnvironment.cpp`
```cpp
void bind_RolloutEnvironment(py::module& m) {
    py::class_<RolloutEnvironment>(m, "RolloutEnvironment")
        // ... existing bindings ...
        .def("get_skeleton_dof", &RolloutEnvironment::GetSkeletonDOF);  // ← EXPOSE TO PYTHON
}
```

**That's it!** The generic HDF5 writer automatically handles new param attributes.

### Result in HDF5
```
/param_0/
  @character_mass = 75.5
  @skeleton_dof = 37       ← Automatically saved!
  @simulation_hz = 900     ← Automatically saved!
  @control_hz = 30         ← Automatically saved!
```

---

## Adding Cycle-Level Attributes

Cycle attributes are computed **per gait cycle** using the filter system.

### Step 1: Create a new DataFilter class

**File**: `python/data_filters.py`

```python
class YourNewFilter(DataFilter):
    """
    Compute your custom cycle-level attribute.

    Example: Max vertical position during cycle
    """

    def __init__(self, record_config: Optional[Dict] = None):
        """
        Args:
            record_config: Configuration dict for validation
        """
        self.record_config = record_config

        # Validate required fields exist
        if record_config:
            if not record_config.get('kinematics', {}).get('root', False):
                raise ValueError("YourNewFilter requires root position tracking")

    def filter_cycles(self, cycles_dict, matrix_cycles_dict, fields):
        """
        Compute attribute for each cycle.

        Args:
            cycles_dict: {cycle_idx: np.ndarray of shape (n_steps, n_fields)}
            matrix_cycles_dict: {key: {cycle_idx: value}} for storing results
            fields: List of field names

        Returns:
            (cycles_dict, matrix_cycles_dict, fields) - modified or unchanged
        """
        # Find required field indices
        root_z_idx = fields.index('root/z')

        # Storage key with _ prefix (becomes HDF5 attribute)
        storage_key = '_max_height'
        max_heights = {}

        # Process each cycle
        for cycle_idx, cycle_data in cycles_dict.items():
            root_z_values = cycle_data[:, root_z_idx]
            max_height = np.max(root_z_values)
            max_heights[cycle_idx] = max_height

        # Store in matrix_cycles_dict with _ prefix
        matrix_cycles_dict[storage_key] = max_heights

        return cycles_dict, matrix_cycles_dict, fields
```

### Step 2: Enable in FilterPipeline

**File**: `python/data_filters.py` in `FilterPipeline.from_config()`

```python
@staticmethod
def from_config(filter_config: Dict, record_config: Dict,
                metabolic_type: str = None, character_mass: float = 0.0):
    pipeline = FilterPipeline()

    # Add existing filters...

    # Add your new filter
    max_height_enabled = filter_config.get('max_height', {}).get('enabled', False)
    if max_height_enabled:
        pipeline.add_filter(YourNewFilter(record_config))

    return pipeline
```

### Step 3: Add config option

**File**: Your rollout YAML config (e.g., `configs/rollout_config.yaml`)

```yaml
filter:
  max_height:
    enabled: true
```

### Step 4: Map to HDF5 attribute

**File**: `python/rollout_worker.py` in `save_to_hdf5()`

Around line 212-226, add your mapping:

```python
# Store cycle-level attributes from filters
for key, value_dict in matrix_data.items():
    if key.startswith('_metabolic_cumulative_'):
        # ... existing code ...
    elif key.startswith('_metabolic_cot_'):
        # ... existing code ...
    elif key == '_max_height':  # ← ADD THIS
        if cycle_idx in value_dict:
            cycle_grp.attrs['max_height'] = float(value_dict[cycle_idx])
```

### Result in HDF5
```
/param_0/cycle_0/
  @num_steps = 90
  @travel_distance = 1.45
  @max_height = 1.02      ← Your new attribute!
```

---

## Computing Statistics of Cycle Attributes

The `StatisticsFilter` computes statistics (mean and standard deviation) of cycle-level attributes across all cycles and stores them as param-level attributes.

### Use Case
If you have cycle-level attributes (e.g., `metabolic/cot/MA`, `metabolic/cumulative/MA`) and want to compute their mean and standard deviation across all cycles as param-level attributes.

### Step 1: Enable in config

**File**: Your rollout YAML config (e.g., `data/rollout/metabolic.yaml`)

```yaml
filters:
  # Existing filters that create cycle attributes
  metabolic_cumulative:
    enabled: true
  metabolic_cot:
    enabled: true

  # Statistics filter - must be last in config!
  stat_filter:
    enabled: true
    keys:
      - 'metabolic/cot/MA'        # Compute mean/std of CoT across all cycles
      - 'metabolic/cumulative/MA'  # Compute mean/std of cumulative energy
```

**Important**: The `stat_filter` should be enabled **after** the filters that create the attributes you want to compute statistics for.

### Step 2: That's it!

The filter is **already implemented** - no code changes needed! It will:
1. Look for the specified keys in cycle-level attributes
2. Compute the mean and standard deviation across all cycles
3. Store the results as param-level attributes with `/mean` and `/std` suffixes

### Result in HDF5

**Before** (without StatisticsFilter):
```
/param_0/
  @character_mass = 72.9
  @success = True
  cycle_0/
    @metabolic/cot/MA = 1.52
    @metabolic/cumulative/MA = 320.5
  cycle_1/
    @metabolic/cot/MA = 1.48
    @metabolic/cumulative/MA = 315.2
  cycle_2/
    @metabolic/cot/MA = 1.50
    @metabolic/cumulative/MA = 318.0
```

**After** (with StatisticsFilter):
```
/param_0/
  @character_mass = 72.9
  @success = True
  @metabolic/cot/MA/mean = 1.50              ← Mean of [1.52, 1.48, 1.50]
  @metabolic/cot/MA/std = 0.0163             ← Std dev of [1.52, 1.48, 1.50]
  @metabolic/cumulative/MA/mean = 317.9      ← Mean of [320.5, 315.2, 318.0]
  @metabolic/cumulative/MA/std = 2.18        ← Std dev of [320.5, 315.2, 318.0]
  cycle_0/
    @metabolic/cot/MA = 1.52
    @metabolic/cumulative/MA = 320.5
  cycle_1/
    @metabolic/cot/MA = 1.48
    @metabolic/cumulative/MA = 315.2
  cycle_2/
    @metabolic/cot/MA = 1.50
    @metabolic/cumulative/MA = 318.0
```

**Key Points**:
- Cycle-level attributes are **preserved** - statistics computation doesn't remove them
- Param-level statistics have `/mean` and `/std` suffixes appended to the base attribute name
- Both mean and standard deviation are computed for each specified key
- You can specify any cycle-level attribute key (not just metabolic data)
- If a key doesn't exist, a warning is printed but execution continues

---

## Adding Time-Series Datasets

Time-series data is recorded **at every simulation timestep**.

### Step 1: Add to RecordConfig

**File**: `rollouts/RolloutRecord.h`

```cpp
struct RecordConfig {
    // ... existing configs ...

    struct YourNewConfig {
        bool enabled = false;
        bool your_field = false;  // ← Add your field flag
    } your_new;
};
```

### Step 2: Add to FieldsFromConfig

**File**: `rollouts/RolloutRecord.cpp`

```cpp
std::vector<std::string> RolloutRecord::FieldsFromConfig(
    const RecordConfig& config, int skeleton_dof) {
    std::vector<std::string> fields;

    // ... existing field additions ...

    // Your new fields
    if (config.your_new.enabled) {
        if (config.your_new.your_field) {
            fields.push_back("your_category/your_field");
        }
    }

    return fields;
}
```

### Step 3: Record data in RecordStep

**File**: `rollouts/RolloutEnvironment.cpp`

```cpp
void RolloutEnvironment::RecordStep(RolloutRecord* record) {
    std::unordered_map<std::string, double> data;

    // ... existing data recording ...

    // Your new data
    if (mRecordConfig.your_new.enabled) {
        if (mRecordConfig.your_new.your_field) {
            double value = /* compute or get from environment */;
            data["your_category/your_field"] = value;
        }
    }

    record->add(mEnv.getSimulationCount() - 1, data);
}
```

### Step 4: Update YAML loader

**File**: `rollouts/RolloutRecord.cpp` in `RecordConfig::LoadFromYAML()`

```cpp
RecordConfig RecordConfig::LoadFromYAML(const std::string& yaml_path) {
    RecordConfig cfg;
    YAML::Node root = YAML::LoadFile(yaml_path);

    // ... existing loading ...

    // Your new config
    if (root["your_new"]) {
        cfg.your_new.enabled = root["your_new"]["enabled"].as<bool>(false);
        cfg.your_new.your_field = root["your_new"]["your_field"].as<bool>(false);
    }

    return cfg;
}
```

### Step 5: Add to rollout config YAML

**File**: Your rollout YAML config

```yaml
record:
  your_new:
    enabled: true
    your_field: true
```

### Result in HDF5
```
/param_0/cycle_0/data
  [step, time, phase, ..., your_category/your_field]
  [0,    0.0,  0.0,   ..., 123.45]
  [1,    0.001,0.01,  ..., 124.56]
  ...
```

---

## Adding Matrix Datasets

Matrix data stores **multi-dimensional arrays** per cycle (e.g., full skeleton poses).

### Step 1: Add to VectorFieldsFromConfig

**File**: `rollouts/RolloutRecord.cpp`

```cpp
std::vector<std::string> RolloutRecord::VectorFieldsFromConfig(const RecordConfig& config) {
    std::vector<std::string> fields;

    // Existing: all joint positions
    if (config.kinematics.enabled && config.kinematics.all) {
        fields.push_back("motions");
    }

    // Add your matrix field
    if (config.your_new.enabled && config.your_new.your_matrix) {
        fields.push_back("your_matrix");  // ← ADD THIS
    }

    return fields;
}
```

### Step 2: Record matrix in RecordStep

**File**: `rollouts/RolloutEnvironment.cpp`

```cpp
void RolloutEnvironment::RecordStep(RolloutRecord* record) {
    // ... existing scalar data recording ...

    // Record matrix data
    if (mRecordConfig.kinematics.enabled && mRecordConfig.kinematics.all) {
        record->addVector("motions", mEnv.getSimulationCount() - 1,
                         skel->getPositions());
    }

    // Your new matrix
    if (mRecordConfig.your_new.enabled && mRecordConfig.your_new.your_matrix) {
        Eigen::VectorXd your_data = /* get your vector data */;
        record->addVector("your_matrix", mEnv.getSimulationCount() - 1, your_data);
    }
}
```

### Step 3: Handle in HDF5 writer

**File**: `python/rollout_worker.py` in `save_to_hdf5()`

```python
# Write matrix datasets if present
if matrix_data:
    for key in ['motions', 'your_matrix']:  # ← ADD YOUR KEY
        if key in matrix_data:
            cycle_matrix_data = matrix_data[key]
            if cycle_idx in cycle_matrix_data:
                matrix_array = cycle_matrix_data[cycle_idx]
                cycle_grp.create_dataset(key, data=matrix_array,
                                        compression='gzip', compression_opts=9)
```

### Result in HDF5
```
/param_0/cycle_0/
  motions [dataset: (90, 37)]      ← Full skeleton positions
  your_matrix [dataset: (90, N)]   ← Your custom matrix!
  data [dataset: (90, num_fields)] ← Time-series data
```

---

## Configuration Examples

### Example 1: Adding Center of Mass Velocity

**Param-level** (constant per param):
```python
# python/rollout_worker.py
self.param_attributes = {
    'character_mass': character_mass,
    'target_velocity': 1.2,  # ← Add this
}
```

**Cycle-level** (computed per cycle):
```python
# python/data_filters.py
class AverageVelocityFilter(DataFilter):
    def filter_cycles(self, cycles_dict, matrix_cycles_dict, fields):
        travel_key = '_travel_distance'
        avg_vel_key = '_average_velocity'
        avg_velocities = {}

        for cycle_idx, cycle_data in cycles_dict.items():
            time_idx = fields.index('time')
            times = cycle_data[:, time_idx]
            duration = times[-1] - times[0]

            distance = matrix_cycles_dict[travel_key][cycle_idx]
            avg_vel = distance / duration if duration > 0 else 0.0
            avg_velocities[cycle_idx] = avg_vel

        matrix_cycles_dict[avg_vel_key] = avg_velocities
        return cycles_dict, matrix_cycles_dict, fields
```

### Example 2: Adding Joint Torques

**Time-series** (per timestep):
```cpp
// rollouts/RolloutRecord.h
struct DynamicsConfig {
    bool enabled = false;
    bool joint_torques = false;
} dynamics;

// rollouts/RolloutEnvironment.cpp
if (mRecordConfig.dynamics.enabled && mRecordConfig.dynamics.joint_torques) {
    Eigen::VectorXd torques = skel->getForces();
    // Record as matrix since it's per-joint
    record->addVector("torques", mEnv.getSimulationCount() - 1, torques);
}
```

---

## Best Practices

### 1. **Use Appropriate Category**
- **Param attribute**: Constant for entire parameter set (character properties, config values)
- **Cycle attribute**: Summarizes a gait cycle (cumulative energy, distance traveled)
- **Time-series**: Per-timestep measurements (angles, velocities, forces)
- **Matrix**: Multi-dimensional per-cycle data (full poses, muscle activations)

### 2. **Naming Conventions**
- **Param attributes**: `snake_case` (e.g., `character_mass`, `skeleton_dof`)
- **Cycle attributes**: Use `_` prefix for filter keys (e.g., `_max_height`, `_metabolic_cot_A`)
- **Time-series fields**: Use `/` for categories (e.g., `angle/HipR`, `grf/left`)
- **Matrix datasets**: Simple names (e.g., `motions`, `torques`)

### 3. **Storage Key Patterns**
```python
# Filter internal keys (with _prefix)
'_metabolic_cumulative_A'    → Temporary storage in matrix_cycles_dict
'_travel_distance'           → Temporary storage

# HDF5 attribute keys (without _prefix)
'metabolic/cumulative/A'     → Final HDF5 attribute name
'travel_distance'            → Final HDF5 attribute name
```

The `_` prefix is used internally in filters, then stripped when writing to HDF5.

### 4. **Configuration Structure**
```yaml
# rollout_config.yaml structure
record:           # C++ RecordConfig - what to record
  kinematics:
    enabled: true
  metabolic:
    enabled: true
    type: "A"

filter:           # Python FilterPipeline - what to compute
  travel_distance:
    enabled: true
  metabolic_cot:
    enabled: true
```

### 5. **Dependency Management**
If your filter depends on other filters, enable them automatically:

```python
# In FilterPipeline.from_config()
cot_enabled = filter_config.get('metabolic_cot', {}).get('enabled', False)

# Auto-enable dependencies
if cot_enabled:
    metabolic_enabled = True      # ← Force dependency
    travel_distance_enabled = True # ← Force dependency
```

---

## Testing Your Changes

### 1. **Rebuild C++ if needed**
```bash
ninja -C build/release
```

### 2. **Test with a simple rollout**
```python
from rollout_worker import RolloutWorker

worker = RolloutWorker(
    metadata_xml="...",
    record_config_path="configs/your_config.yaml",
    filter_config={"your_filter": {"enabled": True}}
)

# Run rollout and check HDF5 output
# ...
```

### 3. **Verify HDF5 structure**
```python
import h5py

with h5py.File('output.h5', 'r') as f:
    # Check param attributes
    print(f['/param_0'].attrs.keys())

    # Check cycle attributes
    print(f['/param_0/cycle_0'].attrs.keys())

    # Check time-series fields
    print(f['/param_0/cycle_0/data'].dtype.names)

    # Check matrix datasets
    print(f['/param_0/cycle_0'].keys())
```

---

## Summary: Quick Reference

| What to Add | Where to Change | Difficulty |
|-------------|-----------------|------------|
| **Param Attribute** | 1) YAML config `environment:` section<br>2) `python/rollout_worker.py` line ~380 | ⭐ Easy (2 steps) |
| **Cycle Attribute** | `python/data_filters.py` (new filter class) | ⭐⭐ Medium |
| **Time-Series Field** | C++ `RolloutRecord.h/cpp`, `RolloutEnvironment.cpp` | ⭐⭐⭐ Advanced |
| **Matrix Dataset** | C++ `RolloutRecord.cpp`, `RolloutEnvironment.cpp`, Python HDF5 writer | ⭐⭐⭐ Advanced |

**Start with param attributes** - they're the easiest and config-driven!

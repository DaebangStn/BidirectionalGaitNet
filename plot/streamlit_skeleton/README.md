# Skeleton Variability Visualization

Visualizes skeleton calibration variability across multiple motion trials for patients. Shows how body scales, positions, orientations, and local translations vary between trials with `trimmed_unified` as the reference baseline.

## Quick Start

```bash
# Run the Streamlit app
cd /home/geon/BidirectionalGaitNet
streamlit run plot/streamlit_skeleton/app.py
```

## Pipeline Overview

```
C3D Files → Static Calibration → Dynamic Calibration → Skeleton YAML → Global Transform Export → Streamlit Visualization
```

### Step 1: Merge Trimmed C3D Files

Merge all `Trimmed_walk*.c3d` files into a single `trimmed_unified.c3d` for each PID/visit:

```bash
./build/release/viewer/c3d_inspect --batch-merge
```

This creates `/mnt/blue8T/CP/RM/{pid}/{visit}/gait/trimmed_unified.c3d` for all available PIDs.

### Step 2: Run Full Calibration

Run static + dynamic calibration to generate skeleton YAML files:

```bash
# All PIDs
./build/release/viewer/c3d_calibration_cli --mode full --motion trimmed -v

# Specific PID/visit
./build/release/viewer/c3d_calibration_cli --mode full --pid 29792292 --visit pre --motion trimmed -v
```

**Output**: Skeleton YAMLs at `@pid:{pid}/{visit}/skeleton/Trimmed_*.yaml` and `trimmed_unified.yaml`

### Step 3: Export Global Transforms

Export T-pose global transforms for visualization:

```bash
# Export all skeleton files
for skel in $(find /mnt/blue8T/CP/RM -path "*/skeleton/Trimmed*.yaml" -o -path "*/skeleton/trimmed*.yaml" | grep -v "/global/"); do
    ./build/release/viewer/skeleton_global_export -i "$skel"
done
```

**Output**: Global transform YAMLs at `@pid:{pid}/{visit}/skeleton/global/*.yaml`

## Output Format

### Global Transform YAML (`skeleton/global/{name}.yaml`)

```yaml
source: "Trimmed_walk01-Dynamic.yaml"
timestamp: "2026-01-27 09:45:00"
nodes:
  - name: Pelvis
    scale: [0.1757, 0.1415, 0.0955]
    global_position: [0.0, 0.9809, -0.0116]
    global_rotation:
      axis: [0.0, 0.0, 1.0]
      angle_deg: 0.0
    local_translation: [0.0, 0.9809, -0.0116]  # w.r.t. parent (root = global)
  - name: FemurR
    scale: [0.1067, 0.2319, 0.0873]
    global_position: [-0.0634, 0.8531, -0.0146]
    global_rotation:
      axis: [-0.0135, 0.0167, -0.9998]
      angle_deg: 11.86
    local_translation: [-0.0634, -0.1278, -0.0030]  # w.r.t. Pelvis frame
  # ... all 21 nodes
```

## Streamlit App Structure

```
plot/streamlit_skeleton/
├── app.py              # Main entry - PID/visit selection, view routing
├── core/
│   ├── __init__.py
│   └── skeleton.py     # Data loading from skeleton/global/ directory
└── views/
    ├── __init__.py
    ├── summary.py      # Summary table view - all nodes at once
    └── variability.py  # Per-node dot plot view
```

### Views

1. **Summary View**: Table showing variability metrics (std dev) for all body nodes
   - Scale (X, Y, Z)
   - Global Position (X, Y, Z)
   - Local Translation (X, Y, Z)
   - Highlights cells exceeding threshold (configurable in sidebar)

2. **Per Node View**: Dot plots for a single body node
   - Each dot = one motion trial
   - Dotted red line = `trimmed_unified` reference
   - Hover shows motion filename
   - Sections: Scale, Global Position, Global Orientation, Local Translation

## Data Availability

As of 2026-01-27:

| PID | pre | op1 | Notes |
|-----|-----|-----|-------|
| 12964246 | ✅ 10 | ✅ 9 | |
| 15998541 | ✅ 16 | ✅ 11 | |
| 18357745 | ✅ 16 | ✅ 20 | |
| 19161079 | ✅ 10 | ✅ 15 | |
| 20705431 | ✅ 15 | ✅ 16 | |
| 23953518 | ✅ 13 | ✅ 19 | |
| 24347325 | ✅ 23 | ✅ 12 | |
| 26763062 | ✅ 14 | ✅ 19 | |
| 26937274 | ✅ 19 | ✅ 17 | |
| 27216110 | ✅ 9 | ✅ 15 | |
| 28801441 | ✅ 17 | ✅ 11 | |
| 29792292 | ✅ 10 | ✅ 6 | |
| 29901382 | ✅ 25 | ✅ 14 | |
| 30157774 | ✅ 24 | ✅ 11 | |
| 32043213 | ✅ 16 | ✅ 20 | |
| 32389630 | ✅ 12 | ✅ 12 | |

**Total**: 476 global transform files across 32 PID/visit combinations

## Tools Reference

### c3d_inspect (batch merge)

```bash
./build/release/viewer/c3d_inspect --batch-merge
```

Merges all `Trimmed_*.c3d` files into `trimmed_unified.c3d` for each PID/visit.

### c3d_calibration_cli

```bash
./build/release/viewer/c3d_calibration_cli [options]

Options:
  --mode [static|dynamic|full]  Calibration mode
  --pid PID                     Specific patient ID (optional)
  --visit [pre|op1|op2]         Specific visit (optional)
  --motion FILTER               Motion name filter (e.g., "trimmed")
  -v                            Verbose output
```

### skeleton_global_export

```bash
./build/release/viewer/skeleton_global_export -i <skeleton.yaml>
```

Computes T-pose global transforms and exports to `skeleton/global/` directory.

**Output includes**:
- `scale`: Body node dimensions [x, y, z]
- `global_position`: World-space position
- `global_rotation`: Axis-angle representation
- `local_translation`: Position relative to parent body node's local frame

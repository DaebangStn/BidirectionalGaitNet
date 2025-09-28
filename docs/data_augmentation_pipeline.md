# Data Augmentation Pipeline - Raw Data Processing

## Overview

`augument_raw_data.py` transforms variable-length raw motion capture data into fixed-length, ML-ready training datasets for BidirectionalGaitNet. The pipeline performs phase-based temporal normalization, converting irregular simulation sequences into standardized 60-frame segments (2 gait cycles at 30 Hz).

## Pipeline Architecture

### Input Specifications

**Source Data Format** (`.npz` files):
```python
{
    'params': np.array,    # Shape: (N, param_dim) - Parameter sets for each sequence
    'motions': np.array,   # Shape: (total_frames, feature_dim) - Raw motion frames
    'lengths': np.array    # Shape: (N,) - Frame count per sequence
}
```

**Raw Motion Features**:
- Positions: Full skeletal pose data
- Phase: `motion[-1]` - Cyclic phase value (0.0 to 1.0, repeating)
- Variable framerate and sequence length

### Output Specifications

**Processed Data Format** (`.npz` files):
```python
{
    'motions': np.array,   # Shape: (batch_size, 60, feature_dim) - Fixed 60-frame sequences
    'params': np.array     # Shape: (batch_size, param_dim) - Corresponding parameters
}
```

**Refined Motion Features**:
- Fixed 60 frames per sequence (2 cycles × 30 frames/cycle)
- 6DOF position representation with velocity components
- Phase-normalized temporal alignment

## Core Processing Pipeline

### Stage 1: Phase Unwrapping (lines 62-68)

**Problem**: Raw phase values wrap cyclically (0.9 → 0.1 → 0.9 → 0.1...)

**Solution**: Convert to monotonically increasing phase coordinate

```python
# Detect phase wrapping
if prev_phi > loaded_motions[i+j][-1]:
    phi_offset += 1

# Unwrap phase
loaded_motions[i+j][-1] += phi_offset
```

**Result**: Phase becomes temporal coordinate (e.g., 0.9, 1.1, 1.9, 2.1...)

**Purpose**: Enables accurate interpolation across gait cycle boundaries

### Stage 2: Reference Phase Generation (lines 32-34)

**Target Phases**: Create uniform 60-frame sampling grid

```python
phis = []
for i in range(60):  # resolution * 2 (30 * 2)
    phis.append(i / 30)  # [0.0, 0.0333, 0.0667, ..., 1.967]
```

**Properties**:
- Uniform temporal spacing: Δt = 1/30 seconds
- Covers 2 complete gait cycles
- Phase range: [0, 2) for two cycles

### Stage 3: Phase-Based Interpolation (lines 73-91)

**Algorithm**: Weighted interpolation using phase as time coordinate

```python
while phi_idx < len(phis) and motion_idx < loaded_lengths[loaded_idx] - 1:
    if loaded_motions[i+motion_idx][-1] <= phis[phi_idx] < loaded_motions[i+motion_idx+1][-1]:
        # Calculate inverse distance weights
        w1 = loaded_motions[i+motion_idx+1][-1] - phis[phi_idx]
        w2 = phis[phi_idx] - loaded_motions[i+motion_idx][-1]

        # Interpolate positions (6DOF representation)
        p = (w1 * env.posToSixDof(loaded_motions[i+motion_idx][:-1]) +
             w2 * env.posToSixDof(loaded_motions[i+motion_idx+1][:-1])) / (w1 + w2)

        # Interpolate velocities
        v1 = loaded_motions[i+motion_idx][3:6] - loaded_motions[i+motion_idx-1][3:6]
        v2 = loaded_motions[i+motion_idx+1][3:6] - loaded_motions[i+motion_idx][3:6]
        v = (w1 * v1 + w2 * v2) / (w1 + w2)

        # Embed velocities into position vector
        p[6] = v[0]   # x-velocity
        p[8] = v[2]   # z-velocity

        refined_motion[0].append(p)
        phi_idx += 1
    else:
        motion_idx += 1
```

**Interpolation Mathematics**:

1. **Weight Calculation**: Inverse distance weighting
   - w1: Distance to next frame (larger when closer to current)
   - w2: Distance to current frame (larger when closer to next)
   - Ensures smooth transitions

2. **Position Interpolation**: 6DOF space
   - Converts raw positions to 6DOF representation via `env.posToSixDof()`
   - Linear interpolation: `p = (w1·p_curr + w2·p_next) / (w1 + w2)`

3. **Velocity Computation**: Finite differences
   - v1: Velocity entering current frame
   - v2: Velocity leaving current frame
   - Interpolated velocity embedded at positions 6 and 8

**Output**: 60-frame refined motion sequence with uniform temporal spacing

### Stage 4: Quality Filtering (lines 94-109)

**Acceptance Criteria**:
```python
if len(refined_motion[0]) == 60:
    result.append(refined_motion)
    stats["segments_generated"] += 1
else:
    stats["bad_refined_length"] += 1
    errors.append({
        "file": f,
        "index": int(loaded_idx),
        "reason": "refined_not_60",
        "detail": int(len(refined_motion[0]))
    })
```

**Quality Gates**:
- ✅ Exactly 60 frames: Accept for training
- ❌ Other lengths: Reject and log error
- ⚠️ Sequence too short (< min_length): Skip preprocessing

**Purpose**: Ensure dataset consistency for neural network training

### Stage 5: Batch Processing (lines 174-187)

**Batching Strategy**:
```python
batch_size = 4096  # Default from args

while len(results) >= batch_size:
    # Extract batch
    res = results[:batch_size]
    motions = np.array([r[0] for r in res])  # Shape: (4096, 60, feature_dim)
    params = np.array([r[1] for r in res])   # Shape: (4096, param_dim)

    # Save compressed batch
    np.savez_compressed(
        args.save + "/" + args.name + "_" + str(save_idx),
        motions=motions,
        params=params
    )

    results = results[batch_size:]
    save_idx += 1
```

**Batch Properties**:
- Fixed size: 4096 segments per file (configurable)
- Compressed format: NPZ with compression
- Sequential naming: `refined_data_0.npz`, `refined_data_1.npz`, ...
- Remainder handling: Final batch saves all remaining segments

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Variable-Length Raw Motion Data (.npz)                    │
│ - params: (N, param_dim)                                         │
│ - motions: (total_frames, feature_dim) with cyclic phase        │
│ - lengths: (N,) frame counts                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Phase Unwrapping                                        │
│ - Detect phase wraps (0.9 → 0.1)                                │
│ - Add offset to make monotonic (0.9 → 1.1)                      │
│ Output: Unwrapped phase as temporal coordinate                  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Reference Phase Generation                              │
│ - Create 60 uniform phases: [0, 1/30, 2/30, ..., 59/30]        │
│ - Covers 2 complete gait cycles at 30 Hz                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Phase-Based Interpolation                               │
│ - For each reference phase:                                      │
│   1. Find bracketing raw frames                                  │
│   2. Calculate inverse distance weights                          │
│   3. Interpolate 6DOF position                                   │
│   4. Interpolate velocity from position differences             │
│   5. Embed velocity into position vector [6] and [8]            │
│ Output: 60-frame sequence with uniform temporal spacing         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Quality Filtering                                       │
│ - Accept: Exactly 60 frames → Add to results                    │
│ - Reject: Other lengths → Log error                             │
│ - Skip: Sequences < min_length (default 140 frames)             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: Batch Processing & Output                               │
│ - Accumulate until batch_size (default 4096)                    │
│ - Split into motions and params arrays                          │
│ - Save compressed: refined_data_{idx}.npz                       │
│ Output: Fixed-length training batches                           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Output: ML-Ready Training Dataset (.npz)                         │
│ - motions: (batch_size, 60, feature_dim)                        │
│ - params: (batch_size, param_dim)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Parameters

### Command-Line Arguments

```python
parser.add_argument("--motion", type=str, default="...")
    # Input directory containing raw .npz files

parser.add_argument("--save", type=str, default="...")
    # Output directory for processed batches

parser.add_argument("--env", type=str, default="...")
    # Environment XML configuration file

parser.add_argument("--name", type=str, default="refined_data")
    # Prefix for output filenames

parser.add_argument("--min_length", type=int, default=140)
    # Minimum raw sequence length (frames) for processing
    # Shorter sequences are skipped

parser.add_argument("--resolution", type=int, default=30)
    # Temporal resolution (Hz): frames per gait cycle
    # Output frames = resolution × 2 (for 2 cycles)

parser.add_argument("--batch_size", type=int, default=4096)
    # Segments per output file

parser.add_argument("--progress_every", type=int, default=50)
    # Progress reporting frequency (files)

parser.add_argument("--log", type=str, default="")
    # Optional CSV error log path
```

### Default Configuration

```python
min_length = 140        # Minimum 140 raw frames (ensures 2+ cycles)
resolution = 30         # 30 Hz sampling rate
batch_size = 4096      # 4096 segments per batch file
output_frames = 60     # 2 cycles × 30 frames = 60 frames
```

## Statistics & Monitoring

### Tracked Metrics

```python
stats = {
    "files_total": len(train_filenames),           # Total input files
    "npz_files": 0,                                # Valid .npz files processed
    "file_load_errors": 0,                         # Failed file loads
    "sequences_total": 0,                          # Total raw sequences
    "sequences_too_short": 0,                      # Sequences < min_length
    "segments_generated": 0,                       # Valid 60-frame segments
    "bad_refined_length": 0,                       # Failed interpolation
}
```

### Error Logging

**Error Record Format** (CSV):
```python
{
    "file": str,           # Source .npz file path
    "index": int,          # Sequence index within file
    "reason": str,         # Error category
    "detail": int/str      # Additional context
}
```

**Error Categories**:
- `np_load_failed`: File load exception
- `sequence_too_short`: Length < min_length
- `refined_not_60`: Interpolation produced ≠ 60 frames

### Progress Reporting

```python
# Every 50 files (configurable)
print(f"[{file_idx}/{len(train_filenames)}] "
      f"npz:{stats['npz_files']} "
      f"segs:{stats['segments_generated']} "
      f"saved:{save_idx} "
      f"too_short:{stats['sequences_too_short']} "
      f"bad_len:{stats['bad_refined_length']} "
      f"errors:{len(errors)}")
```

## Integration with Training Pipeline

### Upstream Dependencies

**RayEnvManager** (`python/augument_raw_data.py:138`):
- `env.posToSixDof()`: Converts raw position to 6DOF representation
- Provides coordinate space transformation for interpolation

### Downstream Consumers

**Forward GaitNet Training** (`train_forward_gaitnet.py`):
- Expects: `(motions, params)` in NPZ format
- Motion shape: `(batch_size, 60, feature_dim)`
- Parameters shape: `(batch_size, param_dim)`
- Phase encoding added during training (not in augmented data)

**Data Flow**:
```
Raw Simulation → augument_raw_data.py → Refined Dataset → train_forward_gaitnet.py
```

## Key Design Decisions

### 1. Phase-Based Temporal Normalization

**Rationale**: Variable simulation framerates require temporal alignment
- Raw data: Irregular timing, variable frame counts
- Training requirement: Fixed-length sequences for batch processing
- Solution: Phase as universal time coordinate

**Benefits**:
- Gait-cycle alignment across different speeds
- Consistent temporal features for neural network
- Handles variable simulation step sizes

### 2. Two-Cycle Segments (60 frames)

**Rationale**: Capture complete gait patterns with temporal context
- Single cycle: Insufficient context for prediction
- Two cycles: Captures full left-right asymmetry and transition dynamics
- 30 Hz: Standard motion capture sampling rate

### 3. Velocity Embedding in Position Vector

**Rationale**: Preserve dynamic information during interpolation
- Position alone: Loses velocity information between frames
- Separate velocity: Increases data complexity
- Embedded velocity: Compact representation with dynamics

**Implementation**:
```python
p[6] = v[0]   # x-velocity (forward motion)
p[8] = v[2]   # z-velocity (lateral motion)
```

### 4. Inverse Distance Weighting

**Rationale**: Linear interpolation in phase space
- Simple and computationally efficient
- Preserves smoothness of motion
- Phase-proportional weighting ensures temporal accuracy

### 5. Strict Quality Filtering

**Rationale**: Dataset consistency critical for training stability
- Neural networks require uniform input shapes
- Malformed sequences cause training failures
- Error logging enables data quality assessment

## Performance Characteristics

### Computational Complexity

**Per Sequence**:
- Phase unwrapping: O(n) where n = raw frame count
- Interpolation: O(n × 60) frame comparisons
- Overall: O(n) linear in raw sequence length

**Batch Processing**:
- Memory efficient: Processes one file at a time
- Disk I/O: Sequential read/write pattern
- Compression: NPZ reduces storage by ~50-70%

### Typical Processing Rates

**Example**: 1000 raw files, 10 sequences per file
- Input: 10,000 raw sequences
- Output: ~8,000 refined segments (assuming 80% pass rate)
- Batching: 2 output files at batch_size=4096
- Processing time: ~10-30 seconds per file (depends on I/O)

### Memory Usage

**Peak Memory**:
```
≈ batch_size × 60 × feature_dim × 8 bytes (float64)
≈ 4096 × 60 × 101 × 8 bytes ≈ 200 MB per batch
```

**Optimization**: Results buffer cleared after each batch save

## Error Handling & Recovery

### Common Issues

**1. File Load Failures**
```python
try:
    loaded_file = np.load(f)
except Exception as e:
    errors.append({"file": f, "index": -1, "reason": "np_load_failed", "detail": str(e)})
    stats["file_load_errors"] += 1
    return result  # Skip file, continue processing
```

**2. Short Sequences**
- Cause: Insufficient raw data for 2-cycle interpolation
- Handling: Skip and log, no processing attempted
- Metric: `sequences_too_short`

**3. Interpolation Failures**
- Cause: Phase discontinuities, missing frames
- Symptom: Refined motion ≠ 60 frames
- Handling: Reject segment, log error with details
- Metric: `bad_refined_length`

### Debugging Workflow

**Enable Error Logging**:
```bash
python augument_raw_data.py \
    --motion /path/to/raw \
    --save /path/to/output \
    --log errors.csv
```

**Analyze Error Distribution**:
```python
import pandas as pd
errors_df = pd.read_csv("errors.csv")
print(errors_df['reason'].value_counts())
```

**Common Fixes**:
- `sequence_too_short`: Reduce `--min_length` parameter
- `refined_not_60`: Check phase data integrity in raw files
- `np_load_failed`: Verify NPZ file format and corruption

## Usage Examples

### Basic Processing

```bash
python python/augument_raw_data.py \
    --motion /data/raw_motions \
    --save /data/refined \
    --env /data/env.xml \
    --name training_data
```

### Custom Configuration

```bash
python python/augument_raw_data.py \
    --motion /data/raw_motions \
    --save /data/refined \
    --env /data/env.xml \
    --name training_data \
    --min_length 120 \
    --resolution 30 \
    --batch_size 8192 \
    --progress_every 100 \
    --log /logs/augmentation_errors.csv
```

### Output Structure

```
/data/refined/
├── training_data_0.npz     # First 4096 segments
├── training_data_1.npz     # Next 4096 segments
├── training_data_2.npz     # Remaining segments
└── ...
```

## Summary

`augument_raw_data.py` implements a sophisticated temporal normalization pipeline that:

1. **Unwraps cyclic phase data** to create continuous time coordinates
2. **Interpolates variable-length sequences** to fixed 60-frame representations
3. **Preserves motion dynamics** through velocity embedding
4. **Ensures data quality** through strict filtering and validation
5. **Produces ML-ready batches** for efficient neural network training

The pipeline is a critical preprocessing step that transforms irregular simulation data into the standardized format required by BidirectionalGaitNet's training system.
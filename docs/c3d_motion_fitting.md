# C3D Motion Fitting Guide

## Overview

This document describes how to fit motion from C3D motion capture files in BidirectionalGaitNet. The pipeline converts raw marker data into parameterized Motion structures suitable for gait synthesis.

## Motion Data Structure

**Location**: `sim/Environment.h:10-15`

```cpp
struct Motion
{
    std::string name;           // Motion identifier (e.g., "C3D")
    Eigen::VectorXd motion;     // Flattened motion data (60 frames × pose_dimensions)
    Eigen::VectorXd param;      // Motion parameters vector
};
```

### Motion Parameters

The `param` vector contains normalized motion characteristics:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | Stride | Absolute stride × 0.5 (half-cycle) |
| 1 | Cadence | Normalized cadence relative to reference |
| 2 | Global Ratio | Overall skeleton scale factor |
| 3 | Femur L Ratio | Left femur length / global ratio |
| 4 | Femur R Ratio | Right femur length / global ratio |
| 5 | Tibia L Ratio | Left tibia length / global ratio |
| 6 | Tibia R Ratio | Right tibia length / global ratio |
| 11 | Femur L Torsion | Left femur torsion angle |
| 12 | Femur R Torsion | Right femur torsion angle |

The `motion` field contains 60 frames of skeletal pose data, resampled from the original frame rate and flattened into a single vector.

## Pipeline Overview

```
C3D File → Python Loader → Skeleton Fitting → Frame Conversion → Resampling → Motion Struct
```

## Step-by-Step Process

### 1. Load C3D File

**Module**: `python/c3dTobvh.py`
**Function**: `load_c3d(c3d_path)`

```python
def load_c3d(c3d_path):
    # Returns: (labels, marker_positions, frame_rate)
    pass
```

**Process**:
- Opens C3D file using `c3d` library
- Reads marker labels and 3D positions per frame
- Converts units: millimeters → meters (× 0.001)
- Reorients coordinate system: `[X, Y, Z]_c3d → [Z, X, Y]_system`
- Returns marker labels, positions, and frame rate

**Output**:
- `labels`: List of marker names
- `marker_positions`: List of frames, each containing list of 3D positions
- `frame_rate`: Capture frame rate (Hz)

### 2. Initialize Skeleton

**Location**: `viewer/C3D_Reader.cpp:74-97`

```cpp
std::vector<Eigen::VectorXd> C3D_Reader::loadC3D(
    std::string path,
    double torsionL,      // Left femur torsion angle
    double torsionR,      // Right femur torsion angle
    double scale,         // Marker scale factor
    double height         // Height offset
)
```

**Process**:
1. Call Python `c3dTobvh.load_c3d(path)` (line 76)
2. Extract frame rate (line 78)
3. Initialize skeleton to T-pose (lines 82-97):
   - Set forearm angles to 90° (π/2)
   - Zero out tibia rotations for leg projection

### 3. Fit Skeleton to Markers

**Location**: `viewer/C3D_Reader.cpp:99-116`

**Process**:
1. Extract initial frame markers (lines 103-108):
   ```cpp
   init_markers = first_frame_markers * scale + Vector3d::UnitY() * height
   ```
2. Call `fitSkeletonToMarker(init_markers, torsionL, torsionR)` (line 110)
   - Adjusts skeleton dimensions to match marker distances
   - Applies femur torsion corrections
3. Store reference markers (line 112-113):
   ```cpp
   mRefMarkers = marker_set.getGlobalPos()
   ```
4. Store reference transformations (lines 115-116):
   ```cpp
   mRefBnTransformation = bodynode.getTransform()
   ```

### 4. Convert Each Frame

**Location**: `viewer/C3D_Reader.cpp:118-132`

**Process**:
For each frame in the C3D data:
1. Scale and offset markers (line 128):
   ```cpp
   markers = raw_markers * scale + Vector3d::UnitY() * height
   ```
2. Call `getPoseFromC3D(markers)` to compute joint angles (line 130)
3. Store resulting pose vector (line 130)
4. Store original markers (line 131)

#### Frame Conversion Details (`getPoseFromC3D`)

**Location**: `viewer/C3D_Reader.cpp:355-530`

Converts marker positions to skeletal pose using marker triads:

**Marker Groups**:
- **Pelvis** (markers 10, 11, 12): Root position and orientation
- **Right Leg**:
  - FemurR (markers 25, 13, 14): Hip rotation
  - TibiaR (markers 14, 15, 16): Knee rotation
  - TalusR (markers 16, 17, 18): Ankle rotation
- **Left Leg**:
  - FemurL (markers 26, 19, 20): Hip rotation
  - TibiaL (markers 20, 21, 22): Knee rotation
  - TalusL (markers 22, 23, 24): Ankle rotation
- **Torso** (markers 3, 4, 7): Spine/torso rotation
- **Head** (markers 0, 1, 2): Neck/head rotation
- **Right Arm**:
  - ArmR (markers 3, 5, 6): Shoulder rotation
  - ForeArmR: Elbow angle from marker distances
- **Left Arm**:
  - ArmL (markers 7, 8, 9): Shoulder rotation
  - ForeArmL: Elbow angle from marker distances

**Algorithm**:
For each body part:
1. Compute rotation matrix from marker triad (origin, reference, markers)
2. Calculate relative rotation: `T = current_markers × reference_markers^T`
3. Transform to parent frame: `pose = parent_transform^T × T`
4. Convert rotation matrix to joint positions (Euler/Ball joint)

### 5. Post-Process Motion

**Location**: `viewer/C3D_Reader.cpp:135-178`

**Process**:

1. **Recenter X/Z** (lines 138-145):
   ```cpp
   for frame in motion[1:]:
       frame[3] -= motion[0][3]  // X position
       frame[5] -= motion[0][5]  // Z position
   motion[0][3] = 0
   motion[0][5] = 0
   ```

2. **Create looping motion** (lines 147-161):
   - Calculate offset point: `3/8 × total_frames`
   - Take frames from offset to end
   - Append frames from start to offset (with position adjustment)
   - Creates seamless cyclic motion

3. **Final recentering** (lines 163-175):
   - Recenter all frames relative to new first frame
   - Zero out X/Z of first frame

**Output**: `std::vector<Eigen::VectorXd>` of processed poses

### 6. Convert to Motion Structure

**Location**: `viewer/C3D_Reader.cpp:182-325`
**Function**: `convertToMotion() → Motion`

**Process**:

#### A. Compute Global Ratio (lines 195-207)
```cpp
globalRatio = max(body_segment_scales)  // Exclude feet/talus
```
Finds the largest body segment scale to use as reference.

#### B. Compute Stride (lines 209-214)
```cpp
abs_stride = (last_frame[5] - first_frame[5]) / (globalRatio × refStride)
param[0] = abs_stride × 0.5  // Half-cycle stride
```

#### C. Compute Cadence (line 216)
```cpp
param[1] = refCadence × sqrt(globalRatio) / (duration × 0.5)
```

#### D. Store Global Ratio (line 218)
```cpp
param[2] = globalRatio
```

#### E. Compute Limb Ratios (lines 221-241)
```cpp
param[3] = femurL_scale / globalRatio  // Clipped [0, 1]
param[4] = femurR_scale / globalRatio
param[5] = tibiaL_scale / globalRatio
param[6] = tibiaR_scale / globalRatio
```

#### F. Store Torsion Angles (lines 255-256)
```cpp
param[11] = femurL_torsion
param[12] = femurR_torsion
```

#### G. Map to Target Skeleton (lines 258-277)
```cpp
for frame in mCurrentMotion:
    mBVHSkeleton.setPositions(frame)
    for joint in mBVHSkeleton.getJoints():
        target_joint = character.skeleton.getJoint(joint.name)
        target_joint.setPositions(joint.getPositions())
    converted_pos = character.posToSixDof(skeleton.getPositions())
    mConvertedPos.push_back(converted_pos)
```

#### H. Resample to 60 Frames (lines 280-322)

**Temporal Resampling**:
```cpp
// Create phase arrays
cur_phis = [2.0 × i / num_frames for i in range(num_frames)]
ref_phis = [2.0 × i / 60 for i in range(60)]  // Target 60 frames

// Interpolate
for ref_phi in ref_phis:
    find cur_idx where: cur_phis[cur_idx] ≤ ref_phi ≤ cur_phis[cur_idx+1]

    w0 = (ref_phi - cur_phis[cur_idx]) / delta_phi
    w1 = (cur_phis[cur_idx+1] - ref_phi) / delta_phi

    interpolated_pose = w0 × mConvertedPos[cur_idx+1] + w1 × mConvertedPos[cur_idx]
```

**Velocity Computation** (lines 307-312):
```cpp
v0 = (current_frame - previous_frame) × frameRate / 30.0
v1 = (next_frame - current_frame) × frameRate / 30.0
v = w0 × v0 + w1 × v1

motion_pos[6] = v[0]  // X velocity
motion_pos[8] = v[2]  // Z velocity
```

**Flatten to Motion Vector** (line 314):
```cpp
motion.motion.segment(phi_idx × pose_dims, pose_dims) = interpolated_pose
```

### 7. Output Motion Structure

**Final Motion struct contains**:
- `name = "C3D"`
- `motion`: 60-frame resampled motion (flattened VectorXd of size 60 × pose_dimensions)
- `param`: 13-element parameter vector with stride, cadence, ratios, and torsions

## Usage Example

```cpp
// Initialize C3D reader
C3D_Reader reader(skeleton_path, marker_set_path, environment);

// Load and fit C3D motion
std::vector<Eigen::VectorXd> raw_motion = reader.loadC3D(
    "path/to/motion.c3d",
    torsionL = 0.0,    // Left femur torsion
    torsionR = 0.0,    // Right femur torsion
    scale = 1.0,       // Marker scale
    height = 0.0       // Height offset
);

// Convert to Motion structure
Motion motion = reader.convertToMotion();

// Access motion data
std::cout << "Stride: " << motion.param[0] << std::endl;
std::cout << "Cadence: " << motion.param[1] << std::endl;
std::cout << "Global Ratio: " << motion.param[2] << std::endl;

// Motion vector contains 60 frames
int pose_dims = motion.motion.rows() / 60;
for (int frame = 0; frame < 60; frame++) {
    Eigen::VectorXd pose = motion.motion.segment(frame * pose_dims, pose_dims);
    // Use pose...
}
```

## Key Considerations

### Marker Set Requirements

The system expects 27 markers in specific anatomical locations:
- 3 head markers (0-2)
- 10 torso/arm markers (3-9, including shoulders)
- 3 pelvis markers (10-12)
- 12 leg markers (13-24): femur, tibia, talus L/R
- 2 virtual hip markers (25-26)

### Coordinate System

- **C3D Input**: X-forward, Y-up, Z-right (mm)
- **System**: X-right, Y-up, Z-forward (m)
- **Conversion**: `[Z, X, Y]_system = 0.001 × [X, Y, Z]_c3d`

### Motion Looping

The 3/8 offset creates a seamless loop by:
1. Taking last 5/8 of motion
2. Appending first 3/8 with position offset
3. Recentering to make first frame the origin

This ensures the motion cycles smoothly without discontinuities.

### Resampling Strategy

- Original frame rate can vary (typically 60-240 Hz)
- Always resamples to exactly 60 frames per cycle
- Uses linear interpolation for pose positions
- Computes velocities from interpolated positions
- Phase-based interpolation ensures temporal consistency

## Related Documentation

- [Skeleton from Motion Manual](skeleton_from_motion_manual.md)
- [Forward GaitNet Workflow](forward_gaitnet_workflow.md)
- [Data Augmentation Pipeline](data_augmentation_pipeline.md)
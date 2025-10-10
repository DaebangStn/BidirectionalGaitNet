# Surgery Script Automation System

## Overview

The Surgery Script Automation System allows you to record, save, load, and execute muscle surgery operations automatically using YAML scripts.

## Features

1. **Record GUI Operations**: All surgery operations performed through the GUI can be recorded
2. **YAML Script Format**: Operations are saved in human-readable YAML format
3. **Joint Angle Support**: Distribute and relax operations include joint angle configurations for reproducibility
4. **Preview Before Execute**: Review all operations before executing a script
5. **Full Operation Support**: All surgery operations are supported

## Usage

### Recording Surgery Operations

1. Open the Surgery Panel (press `G` key)
2. Click **"Start Recording"** button
3. Perform your surgery operations (distribute, relax, remove anchors, etc.)
4. Click **"Stop Recording"** button
5. Click **"Export Recording"** to save to `data/recorded_surgery.yaml`

### Loading and Executing Scripts

1. Open the Surgery Panel (press `G` key)
2. Click **"Load Script"** button (loads from `data/surgery_script.yaml`)
3. Review the operations in the preview popup
4. Click **"Execute Script"** to run all operations or **"Cancel"** to abort

### Editing Scripts

Scripts are stored in YAML format and can be edited manually. See `data/example_surgery.yaml` for examples.

## YAML Script Format

```yaml
version: "1.0"
description: "Optional description of surgery"
operations:
  # Reset all muscles
  - type: reset_muscles
  
  # Distribute passive force with joint angles
  - type: distribute_passive_force
    muscles: ["muscle1", "muscle2", "muscle3"]
    reference_muscle: "muscle1"
    joint_angles:
      JointName1: 0.5236  # radians
      JointName2: -0.1745
  
  # Relax passive force
  - type: relax_passive_force
    muscles: ["muscle4", "muscle5"]
    joint_angles:
      JointName: 1.5708
  
  # Remove anchor from muscle
  - type: remove_anchor
    muscle: "muscle_name"
    anchor_index: 2
  
  # Copy anchor between muscles
  - type: copy_anchor
    from_muscle: "muscle_src"
    from_anchor_index: 1
    to_muscle: "muscle_dst"
  
  # Edit anchor position
  - type: edit_anchor_position
    muscle: "muscle_name"
    anchor_index: 0
    position: [0.1, 0.2, 0.3]
  
  # Edit anchor weights (for LBS multi-body anchors)
  - type: edit_anchor_weights
    muscle: "muscle_name"
    anchor_index: 0
    weights: [0.5, 0.5]
  
  # Add body node to anchor
  - type: add_bodynode_to_anchor
    muscle: "muscle_name"
    anchor_index: 0
    bodynode: "BodyNodeName"
    weight: 1.0
  
  # Remove body node from anchor
  - type: remove_bodynode_from_anchor
    muscle: "muscle_name"
    anchor_index: 0
    bodynode_index: 1
  
  # Export muscles to file
  - type: export_muscles
    filepath: "data/muscle_modified.xml"
```

## Operation Types

### 1. reset_muscles
Resets all muscle modifications to original state.

**Parameters**: None

### 2. distribute_passive_force
Copies the passive force coefficient (lm_norm) from a reference muscle to selected muscles.

**Parameters**:
- `muscles`: List of muscle names to modify
- `reference_muscle`: Name of muscle to copy coefficient from
- `joint_angles` (optional): Map of joint names to angles (radians)

### 3. relax_passive_force
Applies passive force relaxation to selected muscles.

**Parameters**:
- `muscles`: List of muscle names to relax
- `joint_angles` (optional): Map of joint names to angles (radians)

### 4. remove_anchor
Removes an anchor from a muscle (maintains minimum of 2 anchors).

**Parameters**:
- `muscle`: Name of muscle
- `anchor_index`: Index of anchor to remove (0-based)

### 5. copy_anchor
Copies an anchor from one muscle to another.

**Parameters**:
- `from_muscle`: Source muscle name
- `from_anchor_index`: Source anchor index
- `to_muscle`: Destination muscle name

### 6. edit_anchor_position
Modifies the local position of an anchor.

**Parameters**:
- `muscle`: Name of muscle
- `anchor_index`: Index of anchor to modify
- `position`: [x, y, z] coordinates in local frame

### 7. edit_anchor_weights
Modifies the LBS weights for multi-body anchors.

**Parameters**:
- `muscle`: Name of muscle
- `anchor_index`: Index of anchor to modify
- `weights`: List of weight values (must match number of body nodes)

### 8. add_bodynode_to_anchor
Adds a body node to an anchor for LBS blending.

**Parameters**:
- `muscle`: Name of muscle
- `anchor_index`: Index of anchor to modify
- `bodynode`: Name of body node to add
- `weight`: LBS weight for the new body node

### 9. remove_bodynode_from_anchor
Removes a body node from an anchor (maintains minimum of 1 body node).

**Parameters**:
- `muscle`: Name of muscle
- `anchor_index`: Index of anchor to modify
- `bodynode_index`: Index of body node to remove (0-based)

### 10. export_muscles
Saves the current muscle configuration to an XML file.

**Parameters**:
- `filepath`: Output file path (e.g., "data/muscle_modified.xml")

## Joint Angles

When recording distribute_passive_force or relax_passive_force operations, the system automatically captures all current joint angles (excluding root joint). This ensures that:

1. Muscle passive properties are modified at a specific pose
2. Scripts can reproduce the exact same configuration
3. Joint angles are stored in radians

To specify joint angles manually in a script:
```yaml
joint_angles:
  FemurR: 0.5236   # 30 degrees
  TibiaR: -0.1745  # -10 degrees
  TalusR: 0.0
```

## Tips

1. **Test scripts on a copy**: Always test surgery scripts on a copy of your muscle configuration
2. **Incremental operations**: Break complex surgeries into multiple smaller scripts
3. **Comment your scripts**: Use YAML comments (#) to document your operations
4. **Version control**: Keep scripts in version control to track modifications
5. **Joint angles matter**: For distribute/relax operations, joint angles significantly affect results

## Example Workflows

### Workflow 1: Record and Replay
1. Start recording
2. Perform surgery operations manually through GUI
3. Stop recording and export
4. Edit the exported script if needed
5. Load and execute script on another configuration

### Workflow 2: Manual Script Creation
1. Create a new YAML file with desired operations
2. Test joint angles by manually posing the character
3. Record current pose joint angles (use "Capture Positions" button)
4. Add joint angles to script
5. Load and execute script

### Workflow 3: Batch Processing
1. Create a surgery script template
2. Apply to multiple muscle configurations
3. Compare results

## Troubleshooting

**Script fails to load:**
- Check YAML syntax (use a YAML validator)
- Verify file path is correct
- Check operation types are spelled correctly

**Operation fails during execution:**
- Verify muscle names exist in the character
- Check anchor indices are valid
- Ensure body node names are correct
- Check that joint names match skeleton

**Joint angles not applied:**
- Verify joint names match exactly (case-sensitive)
- Check angles are in radians, not degrees
- Ensure joints exist in the skeleton

## File Locations

- **Default script path**: `data/surgery_script.yaml`
- **Example script**: `data/example_surgery.yaml`
- **Recorded operations**: `data/recorded_surgery.yaml` (configurable)

## Keyboard Shortcuts

- `G`: Toggle Surgery Panel
- When recording is active, all surgery operations are automatically captured

## Console Output

The system provides detailed console output for:
- Recording start/stop
- Each recorded operation
- Script loading progress
- Operation execution status (success/failure)
- Detailed error messages

Monitor the console for debugging and verification.


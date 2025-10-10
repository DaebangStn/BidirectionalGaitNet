# Surgery Script Quick Reference

## Recording Operations

1. **Press `G`** to open Surgery Panel
2. Click **"Start Recording"** (button turns red)
3. Perform surgery operations through GUI
4. Click **"Stop Recording"**
5. Click **"Export Recording"** → saves to `data/recorded_surgery.yaml`

## Loading & Executing Scripts

1. **Press `G`** to open Surgery Panel
2. Edit `data/surgery_script.yaml` with your operations
3. Click **"Load Script"**
4. Review operations in preview popup
5. Click **"Execute Script"** or **"Cancel"**

## YAML Template

```yaml
version: "1.0"
description: "Your surgery description"
operations:
  - type: reset_muscles
  
  - type: distribute_passive_force
    muscles: ["muscle1", "muscle2"]
    reference_muscle: "muscle1"
    joint_angles:              # Optional
      JointName: 0.5236        # radians
  
  - type: relax_passive_force
    muscles: ["muscle3"]
    joint_angles:              # Optional
      JointName: 1.5708
  
  - type: remove_anchor
    muscle: "muscle_name"
    anchor_index: 2
  
  - type: copy_anchor
    from_muscle: "source"
    from_anchor_index: 1
    to_muscle: "destination"
  
  - type: edit_anchor_position
    muscle: "muscle_name"
    anchor_index: 0
    position: [0.1, 0.2, 0.3]
  
  - type: edit_anchor_weights
    muscle: "muscle_name"
    anchor_index: 0
    weights: [0.5, 0.5]
  
  - type: add_bodynode_to_anchor
    muscle: "muscle_name"
    anchor_index: 0
    bodynode: "BodyNodeName"
    weight: 1.0
  
  - type: remove_bodynode_from_anchor
    muscle: "muscle_name"
    anchor_index: 0
    bodynode_index: 1
  
  - type: export_muscles
    filepath: "data/output.xml"
```

## Quick Tips

- **Angles in radians**: 30° = 0.5236, 45° = 0.7854, 90° = 1.5708
- **Test first**: Always test on a copy
- **Joint angles**: Auto-captured when recording distribute/relax
- **Comments**: Use `#` for YAML comments
- **Validation**: Check console output for errors

## Common Operations

**Reset everything:**
```yaml
- type: reset_muscles
```

**Distribute at 90° knee flexion:**
```yaml
- type: distribute_passive_force
  muscles: ["GAS_R", "SOL_R"]
  reference_muscle: "GAS_R"
  joint_angles:
    TibiaR: 1.5708  # 90 degrees
```

**Save result:**
```yaml
- type: export_muscles
  filepath: "data/my_surgery.xml"
```

## Keyboard Shortcuts

- `G` - Toggle Surgery Panel
- Red "RECORDING" indicator when active

## File Locations

- Default script: `data/surgery_script.yaml`
- Example: `data/example_surgery.yaml`
- Recorded: `data/recorded_surgery.yaml`
- Full docs: `data/surgery_script_README.md`


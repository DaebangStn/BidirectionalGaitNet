# Skeleton Transform Investigation: Root Cause Analysis

## Summary

**Root Cause Identified**: PhysicalExam and surgery-cli use **different skeleton files** with **asymmetric Body transformations** on the right foot bones.

## The Issue

When exporting muscles:
- **PhysicalExam**: Exports correct local_positions for right foot bones
- **Surgery-cli**: Exports Y/Z sign-flipped local_positions for right foot bones (TalusR, FootThumbR, FootPinkyR)

## Root Cause

### Skeleton File Mismatch

1. **PhysicalExam** loads: `@data/skeleton/base.xml`
2. **Surgery-cli** loads: `@data/skeleton/gaitnet_narrow_model.xml`

### Critical Difference: Asymmetric Body Transformations

In `gaitnet_narrow_model.xml`, **ONLY the right-side foot bones** have 180° rotation transforms:

```xml
<!-- RIGHT SIDE - Has 180° rotation (1, -1, -1) -->
<Node name="TalusR" parent="TibiaR">
    <Body>
        <Transformation linear="1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 -1.0" .../>
    </Body>
</Node>

<Node name="FootPinkyR" parent="TalusR">
    <Body>
        <Transformation linear="1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 -1.0" .../>
    </Body>
</Node>

<Node name="FootThumbR" parent="TalusR">
    <Body>
        <Transformation linear="1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 -1.0" .../>
    </Body>
</Node>

<!-- LEFT SIDE - Has identity transform (1, 1, 1) -->
<Node name="TalusL" parent="TibiaL">
    <Body>
        <Transformation linear="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" .../>
    </Body>
</Node>

<Node name="FootPinkyL" parent="TalusL">
    <Body>
        <Transformation linear="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" .../>
    </Body>
</Node>

<Node name="FootThumbL" parent="TalusL">
    <Body>
        <Transformation linear="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" .../>
    </Body>
</Node>
```

In `base.xml`, **ALL bones** (both left and right) have identity transforms.

## Why This Causes the Problem

### During Muscle Loading (AddAnchor)

When `Muscle::AddAnchor()` converts global muscle waypoint positions to local coordinates:

```cpp
local_pos = bodynode->getTransform().inverse() * global_pos
```

The `getTransform()` includes the Body's transformation matrix from the skeleton XML.

**With gaitnet_narrow_model.xml (surgery-cli)**:
- TalusR world transform = 180° rotation matrix `diag(1, -1, -1)`
- Inverse = same matrix (self-inverse for diagonal)
- Result: Y and Z coordinates get **negated** during local_pos calculation

**With base.xml (PhysicalExam)**:
- TalusR world transform = identity matrix `diag(1, 1, 1)`
- Inverse = identity
- Result: Coordinates remain **unchanged**

### Debug Output Confirms

**PhysicalExam output (base.xml)**:
```
[DEBUG-MUSCLE-LOAD] R_Extensor_Digitorum_Longus -> TalusR local_pos: [-0.00180, 0.07250, -0.00130]
```

**Surgery-cli output (gaitnet_narrow_model.xml)** would show:
```
[DEBUG-MUSCLE-LOAD] R_Extensor_Digitorum_Longus -> TalusR local_pos: [-0.00180, -0.07250, 0.00130]
```
(Y and Z signs flipped)

## Why Only Right Side?

The asymmetry in `gaitnet_narrow_model.xml` is **suspicious and likely incorrect**:
- **Left foot bones**: Identity transforms (correct)
- **Right foot bones**: 180° rotation (problematic)

This suggests either:
1. An error was introduced during skeleton file creation/modification
2. An intentional but undocumented coordinate system change for right-side bones
3. A workaround for a different issue that introduced this inconsistency

## Impact

The muscle file `distribute_lower_only.xml` contains global waypoint positions that were likely authored for `base.xml` (identity transforms). When loaded with `gaitnet_narrow_model.xml`:

- **Left-side muscles**: Load correctly (TalusL has identity transform)
- **Right-side muscles**: Load with Y/Z sign flips (TalusR has 180° rotation)
- **Result**: Asymmetric muscle geometry that doesn't match the original muscle definition

## Solutions

### Option 1: Use base.xml Consistently (Recommended)
Change surgery-cli to use the same skeleton as PhysicalExam:
```cpp
// In surgery-cli configuration or default
skeleton_path = "@data/skeleton/base.xml"
```

**Pros**:
- Immediate fix, no coordinate conversion needed
- Consistent with PhysicalExam
- Muscle files work as originally authored

**Cons**:
- May not be compatible if gaitnet_narrow_model.xml has other important differences
- Need to verify if there's a reason surgery-cli uses gaitnet_narrow_model.xml

### Option 2: Fix gaitnet_narrow_model.xml
Remove the 180° rotations from TalusR, FootThumbR, FootPinkyR to match the left side:

```xml
<!-- Change from -->
<Transformation linear="1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 -1.0" .../>

<!-- To -->
<Transformation linear="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" .../>
```

**Pros**:
- Fixes the asymmetry
- Makes skeleton internally consistent

**Cons**:
- May break other code that depends on gaitnet_narrow_model.xml's current structure
- Need to verify this doesn't break visualizations or physics

### Option 3: Convert Muscle File for gaitnet_narrow_model.xml
Create a version of the muscle file specifically for gaitnet_narrow_model.xml by pre-transforming the right-side anchor positions.

**Pros**:
- Preserves both skeleton files

**Cons**:
- Requires maintaining two muscle file versions
- Error-prone, introduces duplication

## Recommendation

**Use Option 1**: Switch surgery-cli to use `base.xml` consistently with PhysicalExam.

This is the cleanest solution that:
1. Immediately fixes the coordinate issue
2. Maintains consistency between tools
3. Requires minimal changes
4. Doesn't introduce coordinate conversion complexity

## Next Steps

1. Investigate why gaitnet_narrow_model.xml has these asymmetric rotations
2. Check if there are other tools/scripts that depend on gaitnet_narrow_model.xml
3. Verify that base.xml has all the features needed for surgery operations
4. If gaitnet_narrow_model.xml is still needed, consider fixing it to remove the asymmetry

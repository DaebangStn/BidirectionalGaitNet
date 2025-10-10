# Surgery Tool - Standalone Surgery Script Executor

A command-line tool for executing surgery scripts on muscle/skeleton files without GUI dependencies.

## Installation

The tool is integrated with the project's UV package manager:

```bash
# The tool is automatically available after building the project
ninja -C build/release surgery_tool

# Or use via UV (recommended)
uv run surgery --help
```

## Usage

### Basic Usage

```bash
# Run with all defaults
uv run surgery

# Show help
uv run surgery --help

# Use custom script
uv run surgery --script data/my_surgery.yaml

# Specify all parameters
uv run surgery \
  --skeleton data/skeleton_gaitnet_narrow_model.xml \
  --muscle data/muscle_gaitnet.xml \
  --script data/example_surgery.yaml
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--skeleton PATH` | Path to skeleton XML file | `@data/skeleton_gaitnet_narrow_model.xml` |
| `--muscle PATH` | Path to muscle XML file | `@data/muscle_gaitnet.xml` |
| `--script PATH` | Path to surgery script YAML file | `@data/surgery_script.yaml` |
| `--help, -h` | Show help message | - |

### Path Resolution

The tool supports multiple path formats:

- **Absolute paths**: `/full/path/to/file.xml`
- **Relative paths**: `data/file.xml` (automatically converted to `@data/file.xml`)
- **URI paths**: `@data/file.xml` (resolved via URIResolver)

## Surgery Script Format

Surgery scripts are YAML files that define a sequence of operations. See the [Surgery Script README](data/surgery_script_README.md) for full documentation.

### Example Script

```yaml
version: "1.0"
description: "Example surgery operations"

operations:
  - type: reset_muscles
  
  - type: distribute_passive_force
    muscles: [SOL_L, SOL_R, GAS_L]
    reference_muscle: GAS_R
    joint_angles:
      KneeR: -1.57
      AnkleR: 0.52
  
  - type: relax_passive_force
    muscles: [ILIACUS_L, ILIACUS_R]
    
  - type: export_muscles
    filepath: data/muscle_modified.xml
```

## Supported Operations

1. **reset_muscles** - Reset all muscles to original state
2. **distribute_passive_force** - Distribute passive force from reference muscle
3. **relax_passive_force** - Relax passive force for muscles
4. **remove_anchor** - Remove anchor from muscle
5. **copy_anchor** - Copy anchor between muscles
6. **edit_anchor_position** - Edit anchor local position
7. **edit_anchor_weights** - Edit anchor body node weights
8. **add_bodynode_to_anchor** - Add body node to anchor
9. **remove_bodynode_from_anchor** - Remove body node from anchor
10. **export_muscles** - Save muscles to XML file

## Exit Codes

- `0` - All operations completed successfully
- `1` - One or more operations failed or invalid arguments

## Examples

### Run Default Surgery

```bash
# Uses default skeleton, muscles, and script
uv run surgery
```

### Apply Custom Surgery

```bash
# Use your own surgery script
uv run surgery --script my_custom_surgery.yaml
```

### Complete Workflow

```bash
# 1. Record operations in GUI (physical_exam)
uv run physical_exam data/config/physical_exam_example.yaml

# 2. Export recorded operations (done via GUI)
# Saves to: data/recorded_surgery.yaml

# 3. Execute the recorded script
uv run surgery --script data/recorded_surgery.yaml
```

## Integration with UV

The surgery tool is registered in `pyproject.toml`:

```toml
[project.scripts]
surgery = "python.scripts:surgery_tool"
```

This allows you to run it with:
```bash
uv run surgery [options]
```

## Architecture

- **SurgeryExecutor** (base class) - Core surgery logic, no GUI dependencies
- **PhysicalExam** (derived class) - GUI application with surgery recording
- **surgery_tool** (standalone executable) - CLI wrapper around SurgeryExecutor

## Troubleshooting

### Binary not found

```bash
Error: Binary not found at build/release/viewer/surgery_tool
```

**Solution:** Build the project first:
```bash
micromamba run -n bidir ninja -C build/release surgery_tool
```

### Script execution errors

The tool provides detailed error messages for each operation. Common issues:

- **Muscle not found**: Check muscle names in your script match the loaded muscle file
- **Invalid anchor index**: Verify anchor indices are within valid range
- **Joint angle mismatch**: Ensure joint DOFs match the specified angles

### No export operation warning

```
WARNING: No export operation found in script!
```

This means your script will modify muscles but not save them. Add an `export_muscles` operation:

```yaml
operations:
  - type: export_muscles
    filepath: data/output.xml
```

## See Also

- [Surgery Script README](data/surgery_script_README.md) - Full script format documentation
- [Quick Reference](data/QUICK_REFERENCE.md) - Quick reference guide
- [Implementation Details](SURGERY_SCRIPT_IMPLEMENTATION.md) - Technical implementation


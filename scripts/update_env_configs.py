#!/usr/bin/env python3
"""
Update data/env/*.yaml config files to use new RM path structure.

Path transformations:
    @pid:{pid}/gait/pre/skeleton/... -> @pid:{pid}/pre/skeleton/...
    @pid:{pid}/gait/post/skeleton/... -> @pid:{pid}/op1/skeleton/...
    @pid:{pid}/gait/pre/h5/... -> @pid:{pid}/pre/motion/...
    @pid:{pid}/gait/post/h5/... -> @pid:{pid}/op1/motion/...
    @pid:{pid}/gait/pre/muscle/... -> @pid:{pid}/pre/muscle/...
    @pid:{pid}/gait/post/muscle/... -> @pid:{pid}/op1/muscle/...
    @pid:{pid}/gait/metadata.yaml (with prepost: "pre") -> @pid:{pid}/pre/metadata.yaml
    @pid:{pid}/gait/metadata.yaml (with prepost: "post") -> @pid:{pid}/op1/metadata.yaml
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from typing import Tuple, List


def transform_pid_path(path: str, prepost_context: str = None) -> str:
    """Transform old-style @pid path to new-style path."""
    # Match @pid:{pid}/gait/{pre|post}/...
    match = re.match(r'@pid:(\d+)/gait/(pre|post)/(.*)', path)
    if match:
        pid = match.group(1)
        timepoint = match.group(2)
        rest = match.group(3)

        # Convert timepoint to visit
        visit = "pre" if timepoint == "pre" else "op1"

        # Map h5 -> motion
        if rest.startswith("h5/"):
            rest = "motion/" + rest[3:]
        elif rest.startswith("h5"):
            rest = "motion" + rest[2:]

        return f"@pid:{pid}/{visit}/{rest}"

    # Match @pid:{pid}/gait/metadata.yaml
    match = re.match(r'@pid:(\d+)/gait/metadata\.yaml', path)
    if match:
        pid = match.group(1)
        # Use prepost_context if available
        visit = "pre"
        if prepost_context == "post":
            visit = "op1"
        elif prepost_context == "pre":
            visit = "pre"
        return f"@pid:{pid}/{visit}/metadata.yaml"

    # No transformation needed
    return path


def process_yaml_value(value, prepost_context: str = None) -> Tuple[any, bool]:
    """Process a YAML value, transforming any @pid paths. Returns (new_value, changed)."""
    if isinstance(value, str):
        if value.startswith("@pid:"):
            new_value = transform_pid_path(value, prepost_context)
            return new_value, new_value != value
    return value, False


def process_yaml_dict(data: dict, prepost_context: str = None) -> bool:
    """Process a YAML dict recursively, transforming @pid paths. Returns True if changed."""
    changed = False

    # Check if this dict has a prepost key to use as context
    if "prepost" in data:
        prepost_context = data["prepost"]

    for key, value in data.items():
        if isinstance(value, dict):
            if process_yaml_dict(value, prepost_context):
                changed = True
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    if process_yaml_dict(item, prepost_context):
                        changed = True
                elif isinstance(item, str) and item.startswith("@pid:"):
                    new_value, item_changed = process_yaml_value(item, prepost_context)
                    if item_changed:
                        value[i] = new_value
                        changed = True
        elif isinstance(value, str) and value.startswith("@pid:"):
            new_value, value_changed = process_yaml_value(value, prepost_context)
            if value_changed:
                data[key] = new_value
                changed = True

    return changed


def update_config_file(filepath: Path, dry_run: bool = False) -> bool:
    """Update a single config file. Returns True if changed."""
    print(f"Processing {filepath.name}...")

    # Read file content
    with open(filepath) as f:
        content = f.read()

    # Check for @pid patterns
    if "@pid:" not in content:
        print(f"  No @pid paths found, skipping")
        return False

    # Parse YAML
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"  ERROR: Failed to parse YAML: {e}")
        return False

    # Process the data
    changed = process_yaml_dict(data)

    if not changed:
        print(f"  No changes needed")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would update file")
        # Show diffs
        new_content = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        for line in new_content.split('\n'):
            if '@pid:' in line:
                print(f"    {line}")
        return True

    # Write updated content
    # Preserve original format as much as possible by doing text replacement
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        if '@pid:' in line:
            # Find and transform all @pid paths in the line
            def replace_pid_path(match):
                path = match.group(0)
                # Determine prepost context from surrounding lines
                prepost_context = None
                if 'prepost: "pre"' in content or "prepost: 'pre'" in content:
                    prepost_context = "pre"
                elif 'prepost: "post"' in content or "prepost: 'post'" in content:
                    prepost_context = "post"
                return transform_pid_path(path, prepost_context)

            new_line = re.sub(r'@pid:\d+/gait/[^\s"\']+', replace_pid_path, line)
            if new_line != line:
                print(f"  - {line.strip()}")
                print(f"  + {new_line.strip()}")
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    new_content = '\n'.join(new_lines)

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  Updated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Update env config files for new RM structure')
    parser.add_argument('--env-dir', type=str,
                       default='/home/geon/BidirectionalGaitNet/data/env',
                       help='Directory containing env config files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be changed without making changes')
    parser.add_argument('--file', '-f', type=str,
                       help='Process a single file instead of all files')

    args = parser.parse_args()

    env_dir = Path(args.env_dir)

    if args.file:
        files = [env_dir / args.file]
    else:
        files = list(env_dir.glob("*.yaml"))

    print(f"Found {len(files)} config files")

    updated_count = 0
    for filepath in sorted(files):
        if update_config_file(filepath, args.dry_run):
            updated_count += 1

    print(f"\nSummary: {updated_count} files {'would be ' if args.dry_run else ''}updated")


if __name__ == '__main__':
    main()

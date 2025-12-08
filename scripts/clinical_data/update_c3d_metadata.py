#!/usr/bin/env python3
"""
Update c3d/metadata.yaml with personal.dat data from each session.
Extracts height, weight, and foot measurements.
"""

import yaml
import re
from pathlib import Path


def parse_personal_dat(filepath: Path) -> dict:
    """Parse personal.dat file and extract relevant measurements."""
    data = {}

    if not filepath.exists():
        return data

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Parse key-value pairs
    patterns = {
        'height': r'Height \(cm\):\s*([\d.]+)',
        'weight': r'Weight \(kg\):\s*([\d.]+)',
        'r_foot_length': r'R foot length \(cm\):\s*([\d.]+)',
        'r_foot_width': r'R foot width \(cm\):\s*([\d.]+)',
        'l_foot_length': r'L foot length \(cm\):\s*([\d.]+)',
        'l_foot_width': r'L foot width \(cm\):\s*([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            data[key] = float(match.group(1))

    return data


def build_session_data(timestamp: int, personal_data: dict) -> dict:
    """Build session data structure."""
    session = {
        'timestamp': timestamp,
        'height': personal_data.get('height'),
        'weight': personal_data.get('weight'),
        'foot': {
            'right': {
                'length': personal_data.get('r_foot_length'),
                'width': personal_data.get('r_foot_width'),
            },
            'left': {
                'length': personal_data.get('l_foot_length'),
                'width': personal_data.get('l_foot_width'),
            },
        },
    }
    return session


def main():
    rm_dir = Path('/mnt/blue8T/CP/RM')

    print("Updating c3d/metadata.yaml with personal.dat data...")
    print("=" * 50)

    for pid_dir in sorted(rm_dir.iterdir()):
        if not pid_dir.is_dir():
            continue

        pid = pid_dir.name
        c3d_dir = pid_dir / 'c3d'
        metadata_path = c3d_dir / 'metadata.yaml'

        if not metadata_path.exists():
            print(f"  [SKIP] {pid} - no c3d/metadata.yaml")
            continue

        # Read existing metadata
        with open(metadata_path, 'r') as f:
            old_metadata = yaml.safe_load(f) or {}

        new_metadata = {}

        # Process each session (pre, post, post2)
        for session_name in ['pre', 'post', 'post2']:
            if session_name not in old_metadata:
                continue

            # Handle both old format (int) and new format (dict with timestamp)
            old_value = old_metadata[session_name]
            if isinstance(old_value, dict):
                # Extract timestamp, handling nested dicts
                ts = old_value.get('timestamp')
                while isinstance(ts, dict):
                    ts = ts.get('timestamp')
                timestamp = ts
            else:
                timestamp = old_value

            session_dir = c3d_dir / session_name
            personal_dat = session_dir / 'personal.dat'

            personal_data = parse_personal_dat(personal_dat)
            new_metadata[session_name] = build_session_data(timestamp, personal_data)

        # Write updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_metadata, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        sessions = list(new_metadata.keys())
        print(f"  [UPDATED] {pid}: {', '.join(sessions)}")

    print("=" * 50)
    print("Done.")


if __name__ == '__main__':
    main()

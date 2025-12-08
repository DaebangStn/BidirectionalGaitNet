#!/usr/bin/env python3
"""
Export physical examination data to individual patient directories.
- metadata.yaml: patient info, gmfcs, surgery info
- rom.yaml: ROM measurements (pre_op and post_op)
"""

import yaml
from pathlib import Path


def load_physical_examination(path: str) -> dict:
    """Load the physical examination YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_metadata(pid: str, patient_data: dict) -> dict:
    """Extract metadata (non-ROM) information for a patient."""
    metadata = {
        'pid': pid,
        'name': patient_data.get('name'),
        'ser_no': patient_data.get('ser_no'),
        'gmfcs': patient_data.get('gmfcs'),
    }

    # Add surgery information
    if 'surgery1' in patient_data:
        s1 = patient_data['surgery1']
        metadata['surgery1'] = {
            'age': s1.get('age'),
            'name': s1.get('name'),
            'ct': s1.get('ct'),
            'xray': s1.get('xray'),
        }

    if 'surgery2' in patient_data:
        s2 = patient_data['surgery2']
        metadata['surgery2'] = {
            'age': s2.get('age'),
            'name': s2.get('name'),
            'ct': s2.get('ct'),
            'xray': s2.get('xray'),
        }

    return metadata


def extract_rom(patient_data: dict) -> dict:
    """Extract ROM measurements for a patient."""
    rom_data = {}

    if 'pre_op' in patient_data:
        pre_op = patient_data['pre_op']
        rom_data['pre_op'] = {
            'age': pre_op.get('age'),
            'rom': pre_op.get('rom'),
        }

    if 'post_op' in patient_data:
        post_op = patient_data['post_op']
        rom_data['post_op'] = {
            'age': post_op.get('age'),
            'rom': post_op.get('rom'),
        }

    return rom_data


def write_yaml(data: dict, path: Path):
    """Write data to a YAML file."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def main():
    # Paths
    source_path = Path('/home/geon/BidirectionalGaitNet/data/physical_examination.yaml')
    base_dir = Path('/mnt/blue8T/CP/RM')

    # Load data
    data = load_physical_examination(source_path)
    patients = data.get('patients', {})

    print(f"Found {len(patients)} patients in physical_examination.yaml")

    for pid, patient_data in patients.items():
        patient_dir = base_dir / pid

        if not patient_dir.exists():
            print(f"  [SKIP] {pid} - directory does not exist")
            continue

        # Extract and write metadata
        metadata = extract_metadata(pid, patient_data)
        metadata_path = patient_dir / 'metadata.yaml'
        write_yaml(metadata, metadata_path)

        # Extract and write ROM data
        rom_data = extract_rom(patient_data)
        rom_path = patient_dir / 'rom.yaml'
        write_yaml(rom_data, rom_path)

        print(f"  [EXPORTED] {pid} ({patient_data.get('name', 'Unknown')})")

    print("Done.")


if __name__ == '__main__':
    main()

"""Patient data loading for ROM Browser."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import os
import streamlit as st
import yaml
import h5py
import numpy as np
from rm import rm_mgr


# Mapping from HDF5 key suffix to (body_name, dof_index)
KINEMATICS_MAPPING = {
    'HipR': ('FemurR', 0),
    'HipAbR': ('FemurR', 1),
    'HipIRR': ('FemurR', 2),
    'HipL': ('FemurL', 0),
    'HipAbL': ('FemurL', 1),
    'HipIRL': ('FemurL', 2),
    'KneeR': ('TibiaR', 0),
    'KneeL': ('TibiaL', 0),
    'AnkleR': ('TalusR', 0),
    'AnkleL': ('TalusL', 0),
    'Tilt': ('Pelvis', 0),
    'Rotation': ('Pelvis', 1),
    'Obliquity': ('Pelvis', 2),
}


@st.cache_data
def list_pids() -> list[dict]:
    """List available PIDs with metadata.

    Returns:
        List of dicts with: pid, name, gmfcs
    """
    try:
        pids = rm_mgr.list("@pid")
        # Filter to numeric PIDs only
        pids = sorted([p for p in pids if p.isdigit()])

        result = []
        for pid in pids:
            info = {'pid': pid, 'name': '', 'gmfcs': ''}

            # Fetch patient name
            try:
                handle = rm_mgr.fetch(f"@pid:{pid}/name")
                info['name'] = handle.as_string().strip()
            except Exception:
                pass

            # Fetch GMFCS level
            try:
                handle = rm_mgr.fetch(f"@pid:{pid}/gmfcs")
                info['gmfcs'] = handle.as_string().strip()
            except Exception:
                pass

            result.append(info)

        return result
    except Exception as e:
        st.error(f"Error listing PIDs: {e}")
        return []


@st.cache_data
def list_visits(pid: str) -> list[str]:
    """List available visits for a patient.

    Args:
        pid: Patient ID

    Returns:
        List of visit names (e.g., ['pre', 'op1'])
    """
    visits = []
    for visit in ['pre', 'op1', 'op2']:
        try:
            # Check if rom.yaml exists for this visit
            handle = rm_mgr.fetch(f"@pid:{pid}/{visit}/rom.yaml")
            if handle.valid():
                visits.append(visit)
        except Exception:
            pass
    return visits


@st.cache_data
def load_rom_data(pid: str, visit: str) -> dict:
    """Load ROM data for a patient visit.

    Args:
        pid: Patient ID
        visit: Visit name (pre, op1, op2)

    Returns:
        Dict with 'age' and 'rom' keys, or None if not found
    """
    try:
        handle = rm_mgr.fetch(f"@pid:{pid}/{visit}/rom.yaml")
        if not handle.valid():
            return None

        data = yaml.safe_load(handle.as_string())
        return data
    except Exception as e:
        st.error(f"Error loading ROM data: {e}")
        return None


def get_rom_value(rom_data: dict, side: str, joint: str, field: str) -> float:
    """Extract a specific ROM value from loaded data.

    Args:
        rom_data: Loaded ROM data dict
        side: 'left' or 'right'
        joint: 'hip', 'knee', or 'ankle'
        field: ROM field name

    Returns:
        ROM value or None if not found
    """
    if not rom_data:
        return None

    try:
        return rom_data.get('rom', {}).get(side, {}).get(joint, {}).get(field)
    except Exception:
        return None


@st.cache_data
def load_surgery_info(pid: str, visit: str) -> list[str]:
    """Load surgery info from metadata.yaml for a patient visit.

    Args:
        pid: Patient ID
        visit: Visit name (pre, op1, op2)

    Returns:
        List of surgery names, or empty list if not found
    """
    try:
        handle = rm_mgr.fetch(f"@pid:{pid}/{visit}/metadata.yaml")
        if not handle.valid():
            return []

        data = yaml.safe_load(handle.as_string())
        return data.get('surgery', [])
    except Exception:
        return []


@st.cache_data
def list_all_surgeries() -> list[str]:
    """List all unique surgery names across all patients (without _Rt/_Lt suffix).

    Returns:
        Sorted list of unique surgery base names
    """
    pids = list_pids()
    surgeries = set()

    for p in pids:
        for visit in ['op1', 'op2']:
            surgery_list = load_surgery_info(p['pid'], visit)
            for surg in surgery_list:
                # Remove _Rt or _Lt suffix to get base name
                if surg.endswith('_Rt'):
                    base = surg[:-3]
                elif surg.endswith('_Lt'):
                    base = surg[:-3]
                else:
                    base = surg
                surgeries.add(base)

    return sorted(surgeries)


@st.cache_data
def load_all_patients_rom(visit: str, field: str) -> list[dict]:
    """Load ROM values for all patients for a specific field.

    Args:
        visit: Visit name (pre, op1, op2)
        field: ROM field name (e.g., 'abduction_flex90_r2')

    Returns:
        List of dicts with: pid, name, gmfcs, left, right
    """
    pids = list_pids()
    results = []

    for p in pids:
        rom_data = load_rom_data(p['pid'], visit)
        if not rom_data:
            continue

        # Determine joint from field name
        if 'dorsiflexion' in field or 'plantarflexion' in field:
            joint = 'ankle'
        elif 'popliteal' in field or 'flexion' in field.lower():
            joint = 'knee'
        else:
            joint = 'hip'

        left_val = get_rom_value(rom_data, 'left', joint, field)
        right_val = get_rom_value(rom_data, 'right', joint, field)

        if left_val is not None or right_val is not None:
            results.append({
                'pid': p['pid'],
                'name': p['name'],
                'gmfcs': p['gmfcs'],
                'left': left_val,
                'right': right_val
            })

    return results


def _get_motion_dir(pid: str, visit: str) -> str | None:
    """Get the motion directory path for a patient visit."""
    try:
        handle = rm_mgr.fetch(f"@pid:{pid}/{visit}/rom.yaml")
        if not handle.valid():
            return None
        rom_path = handle.local_path()
        base_path = os.path.dirname(rom_path)
        motion_dir = os.path.join(base_path, 'motion')
        if os.path.isdir(motion_dir):
            return motion_dir
        return None
    except Exception:
        return None


@st.cache_data
def list_motion_files(pid: str, visit: str) -> list[str]:
    """List .h5 files in motion directory, with trimmed_unified_edited.h5 first if exists.

    Args:
        pid: Patient ID
        visit: Visit name (pre, op1, op2)

    Returns:
        List of .h5 filenames sorted with trimmed_unified_edited.h5 first
    """
    motion_dir = _get_motion_dir(pid, visit)
    if not motion_dir:
        return []

    try:
        h5_files = [f for f in os.listdir(motion_dir) if f.endswith('.h5')]
        # Sort with trimmed_unified_edited.h5 first
        preferred = 'trimmed_unified_edited.h5'
        if preferred in h5_files:
            h5_files.remove(preferred)
            h5_files.insert(0, preferred)
        return h5_files
    except Exception:
        return []


@st.cache_data
def load_kinematics_data(pid: str, visit: str, motion_file: str) -> dict | None:
    """Load kinematics data with skeleton body names.

    Args:
        pid: Patient ID
        visit: Visit name (pre, op1, op2)
        motion_file: Name of the .h5 file

    Returns:
        Dict mapping joint names (e.g., 'FemurR[0]') to dict with:
            - mean: np.array (100,) gait cycle mean
            - std: np.array (100,) gait cycle std
            - min: np.array (100,) gait cycle min
            - max: np.array (100,) gait cycle max
            - range_min: float (min of mean curve)
            - range_max: float (max of mean curve)
            - range: float (range_max - range_min)
        Returns None if file not found or error
    """
    motion_dir = _get_motion_dir(pid, visit)
    if not motion_dir:
        return None

    h5_path = os.path.join(motion_dir, motion_file)
    if not os.path.exists(h5_path):
        return None

    try:
        result = {}
        with h5py.File(h5_path, 'r') as f:
            if 'kinematics' not in f:
                return None

            kin_group = f['kinematics']
            # Find all unique joint keys (without _mean/_std/_min/_max suffix)
            keys = set()
            for key in kin_group.keys():
                # Remove suffix to get base key
                for suffix in ['_mean', '_std', '_min', '_max']:
                    if key.endswith(suffix):
                        base_key = key[:-len(suffix)]
                        keys.add(base_key)
                        break

            for base_key in keys:
                # Extract joint suffix from angle_XXX format
                if not base_key.startswith('angle_'):
                    continue
                joint_suffix = base_key[6:]  # Remove 'angle_' prefix

                if joint_suffix not in KINEMATICS_MAPPING:
                    continue

                body_name, dof_idx = KINEMATICS_MAPPING[joint_suffix]
                joint_name = f"{body_name}[{dof_idx}]"

                # Load data
                mean_data = np.array(kin_group[f'{base_key}_mean'][:])
                std_data = np.array(kin_group[f'{base_key}_std'][:])
                min_data = np.array(kin_group[f'{base_key}_min'][:])
                max_data = np.array(kin_group[f'{base_key}_max'][:])

                result[joint_name] = {
                    'mean': mean_data,
                    'std': std_data,
                    'min': min_data,
                    'max': max_data,
                    'range_min': float(np.min(mean_data)),
                    'range_max': float(np.max(mean_data)),
                    'range': float(np.max(mean_data) - np.min(mean_data)),
                }

        return result if result else None
    except Exception as e:
        st.error(f"Error loading kinematics: {e}")
        return None

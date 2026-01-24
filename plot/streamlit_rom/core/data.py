"""Patient data loading for ROM Browser."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import streamlit as st
import yaml
from rm import rm_mgr


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

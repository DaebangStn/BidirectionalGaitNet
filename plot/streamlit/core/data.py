"""Shared HDF5 data loader with caching."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import streamlit as st
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from rm import rm_mgr

# Base path for sampled rollout data
SAMPLED_DIR = Path('/home/geon/BidirectionalGaitNet/sampled')
# Base path for angle sweep results
RESULTS_DIR = Path('/home/geon/BidirectionalGaitNet/results')


@st.cache_data
def list_pids() -> list[dict]:
    """List available PIDs with metadata.

    Returns list of dicts with:
        - pid: str
        - name: str (patient name, may be empty)
        - gmfcs: str (GMFCS level, may be empty)
    """
    try:
        pids = rm_mgr.list("@pid")
        pids = sorted(pids)

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


@st.cache_data(ttl=60)  # Cache for 60 seconds only
def list_hdf_files(pid: str, pre_post: str) -> list[str]:
    """List HDF5 files for a PID and session.

    Args:
        pid: Patient ID
        pre_post: "pre" or "post"

    Returns list of HDF5 filenames (without path).
    """
    try:
        uri = f"@pid:{pid}/gait/{pre_post}/h5"
        files = rm_mgr.list(uri)
        # Filter for .h5 files only
        h5_files = [f for f in files if f.lower().endswith('.h5')]
        return sorted(h5_files)
    except Exception as e:
        print(f"[list_hdf_files] exception: {e}")
        return []


@st.cache_data
def load_hdf5_data(uri: str) -> dict:
    """Load HDF5 data via ResourceManager.

    Returns dict with:
        - marker_error_free_data: (numFrames x numMarkers)
        - marker_error_free_mean: (numFrames,)
        - marker_error_motion_data: (numFrames x numMarkers)
        - marker_error_motion_mean: (numFrames,)
        - marker_names: list of marker names
        - num_markers: int
        - frame_rate: int
        - pid: str
        - pre_post: str
    """
    try:
        path = rm_mgr.resolve(uri)
        if not path:
            return None

        with h5py.File(path, 'r') as f:
            data = {}

            # Root attributes (decode bytes to str for h5py compatibility)
            marker_names_raw = f.attrs['marker_names']
            if isinstance(marker_names_raw, bytes):
                marker_names_raw = marker_names_raw.decode('utf-8')
            data['marker_names'] = marker_names_raw.split(',')
            data['num_markers'] = int(f.attrs['num_markers'])
            data['frame_rate'] = int(f.attrs['frame_rate'])
            pid_raw = f.attrs['pid']
            data['pid'] = pid_raw.decode('utf-8') if isinstance(pid_raw, bytes) else pid_raw
            pre_post_raw = f.attrs['pre_post']
            data['pre_post'] = pre_post_raw.decode('utf-8') if isinstance(pre_post_raw, bytes) else pre_post_raw
            data['num_frames'] = int(f.attrs['num_frames'])

            # Marker error data
            if 'marker_error' in f:
                if 'free' in f['marker_error']:
                    data['marker_error_free_data'] = f['marker_error/free/data'][:]
                    data['marker_error_free_mean'] = f['marker_error/free/mean'][:]
                if 'motion' in f['marker_error']:
                    data['marker_error_motion_data'] = f['marker_error/motion/data'][:]
                    data['marker_error_motion_mean'] = f['marker_error/motion/mean'][:]

            # Motion data
            if 'motions' in f:
                data['motions'] = f['motions'][:]
            if 'phase' in f:
                data['phase'] = f['phase'][:]
            if 'time' in f:
                data['time'] = f['time'][:]

            return data
    except Exception as e:
        st.error(f"Error loading HDF5: {e}")
        return None


@st.cache_data(ttl=60)
def list_sampled_dirs() -> list[str]:
    """List available sampled rollout directories.

    Returns list of directory names in the sampled/ folder.
    """
    try:
        if not SAMPLED_DIR.exists():
            return []
        dirs = [d.name for d in SAMPLED_DIR.iterdir() if d.is_dir()]
        return sorted(dirs)
    except Exception as e:
        print(f"[list_sampled_dirs] exception: {e}")
        return []


@st.cache_data
def load_sampled_hdf5(dir_name: str) -> dict:
    """Load rollout_data.h5 from a sampled directory.

    Args:
        dir_name: Name of the directory in sampled/

    Returns dict with:
        - source: 'sampled'
        - dir_name: str
        - cycles: {cycle_idx: {'angle': {'HipR': array, ...}, 'phase': array}}
        - averaged: {'angle': {...}, 'phase': array} or None if not available
    """
    try:
        h5_path = SAMPLED_DIR / dir_name / 'rollout_data.h5'
        if not h5_path.exists():
            st.error(f"File not found: {h5_path}")
            return None

        with h5py.File(h5_path, 'r') as f:
            data = {
                'source': 'sampled',
                'dir_name': dir_name,
                'cycles': {},
                'averaged': None,
                'muscle_names': None
            }

            # Load muscle names from root
            if 'muscle_names' in f:
                names = f['muscle_names'][:]
                data['muscle_names'] = [
                    n.decode() if isinstance(n, bytes) else n
                    for n in names
                ]

            # Find param groups (param_0, param_1, etc.)
            # For now, use param_0
            if 'param_0' not in f:
                st.error("No param_0 group found in HDF5")
                return None

            param_grp = f['param_0']

            # Load cycle data
            cycle_keys = [k for k in param_grp.keys() if k.startswith('cycle_')]
            for cycle_key in sorted(cycle_keys, key=lambda x: int(x.split('_')[1])):
                cycle_idx = int(cycle_key.split('_')[1])
                cycle_grp = param_grp[cycle_key]

                cycle_data = {'angle': {}, 'sway': {}, 'phase': None, 'muscle': {}}

                # Load all angles
                if 'angle' in cycle_grp:
                    for key in cycle_grp['angle'].keys():
                        cycle_data['angle'][key] = cycle_grp['angle'][key][:]

                # Load all sway metrics
                if 'sway' in cycle_grp:
                    for key in cycle_grp['sway'].keys():
                        cycle_data['sway'][key] = cycle_grp['sway'][key][:]

                # Load phase
                if 'phase' in cycle_grp:
                    cycle_data['phase'] = cycle_grp['phase'][:]

                # Load muscle data
                if 'muscle' in cycle_grp:
                    for metric in ['activation', 'force', 'passive', 'lm_norm']:
                        if metric in cycle_grp['muscle']:
                            cycle_data['muscle'][metric] = cycle_grp['muscle'][metric][:]

                data['cycles'][cycle_idx] = cycle_data

            # Check for pre-averaged data at param level
            if 'angle' in param_grp:
                averaged = {'angle': {}, 'sway': {}, 'phase': None}

                # Load all averaged angles
                for key in param_grp['angle'].keys():
                    averaged['angle'][key] = param_grp['angle'][key][:]

                # Load all averaged sway metrics
                if 'sway' in param_grp:
                    for key in param_grp['sway'].keys():
                        averaged['sway'][key] = param_grp['sway'][key][:]

                if 'phase' in param_grp:
                    averaged['phase'] = param_grp['phase'][:]

                # Only set averaged if we have data
                if averaged['angle'] or averaged['sway']:
                    data['averaged'] = averaged

            return data

    except Exception as e:
        st.error(f"Error loading sampled HDF5: {e}")
        return None


@st.cache_data(ttl=60)
def list_angle_sweep_files() -> list[str]:
    """List CSV files in results/ directory.

    Returns list of CSV filenames for angle sweep data.
    """
    try:
        if not RESULTS_DIR.exists():
            return []
        files = [f.name for f in RESULTS_DIR.iterdir()
                 if f.is_file() and f.suffix.lower() == '.csv']
        return sorted(files)
    except Exception as e:
        print(f"[list_angle_sweep_files] exception: {e}")
        return []


@st.cache_data
def load_angle_sweep_csv(filename: str) -> dict:
    """Load angle sweep CSV data.

    Args:
        filename: Name of the CSV file in results/

    Returns dict with:
        - source: 'angle_sweep'
        - filename: str
        - joint_angle_deg: np.array
        - passive_force_total: np.array
        - muscles: list[str]  (muscle names)
        - muscle_data: {
            'muscle_name': {
                'fp': np.array,
                'lm_norm': np.array,
                'jtp_mag': np.array
            }
        }
    """
    try:
        csv_path = RESULTS_DIR / filename
        if not csv_path.exists():
            st.error(f"File not found: {csv_path}")
            return None

        df = pd.read_csv(csv_path)

        data = {
            'source': 'angle_sweep',
            'filename': filename,
            'joint_angle_deg': df['joint_angle_deg'].values,
            'passive_force_total': df['passive_force_total'].values,
            'muscles': [],
            'muscle_data': {}
        }

        # Parse muscle columns: {muscle}_fp, {muscle}_lm_norm, {muscle}_jtp_mag
        columns = df.columns.tolist()
        muscle_names = set()
        for col in columns:
            if col.endswith('_fp'):
                muscle_names.add(col[:-3])
            elif col.endswith('_lm_norm'):
                muscle_names.add(col[:-8])
            elif col.endswith('_jtp_mag'):
                muscle_names.add(col[:-8])

        data['muscles'] = sorted(list(muscle_names))

        for muscle in data['muscles']:
            data['muscle_data'][muscle] = {}
            if f'{muscle}_fp' in df.columns:
                data['muscle_data'][muscle]['fp'] = df[f'{muscle}_fp'].values
            if f'{muscle}_lm_norm' in df.columns:
                data['muscle_data'][muscle]['lm_norm'] = df[f'{muscle}_lm_norm'].values
            if f'{muscle}_jtp_mag' in df.columns:
                data['muscle_data'][muscle]['jtp_mag'] = df[f'{muscle}_jtp_mag'].values

        return data

    except Exception as e:
        st.error(f"Error loading angle sweep CSV: {e}")
        return None

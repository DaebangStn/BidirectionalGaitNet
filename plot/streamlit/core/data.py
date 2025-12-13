"""Shared HDF5 data loader with caching."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import streamlit as st
import h5py
import numpy as np
from rm import rm_mgr


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

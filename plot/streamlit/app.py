"""Streamlit app for HDF5 marker error visualization."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import streamlit as st
import yaml
from pathlib import Path

from core.data import load_hdf5_data, list_pids, list_hdf_files
from core.registry import load_view


def format_pid_option(pid_info: dict) -> str:
    """Format PID info for display in selectbox."""
    pid = pid_info['pid']
    name = pid_info['name']
    gmfcs = pid_info['gmfcs']

    parts = [pid]
    if name:
        parts.append(name)
    if gmfcs:
        parts.append(gmfcs)  # Just the level (e.g., "II")

    if len(parts) == 1:
        return pid
    else:
        return f"{pid} ({', '.join(parts[1:])})"


def main():
    # Load config
    config_path = Path(__file__).parent / "config" / "app.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Page config
    st.set_page_config(page_title=config["app"]["title"], layout="wide")
    st.title(config["app"]["title"])

    # Sidebar: Data Source
    st.sidebar.header("Data Source")

    # Refresh button to clear cache
    if st.sidebar.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    # PID selection
    pids = list_pids()
    if not pids:
        st.sidebar.warning("No PIDs found")
        st.info("No patient data available")
        return

    pid_options = {format_pid_option(p): p['pid'] for p in pids}
    selected_pid_label = st.sidebar.selectbox(
        "Patient",
        list(pid_options.keys()),
        index=0
    )
    selected_pid = pid_options[selected_pid_label]

    # Session selection
    pre_post = st.sidebar.selectbox("Session", ["pre", "post"])

    # HDF file selection
    h5_files = list_hdf_files(selected_pid, pre_post)

    data = None
    if not h5_files:
        st.sidebar.info(f"No HDF5 files for {pre_post}")
    else:
        selected_h5 = st.sidebar.selectbox(
            "HDF5 File",
            h5_files,
            index=0
        )

        # Load data
        uri = f"@pid:{selected_pid}/gait/{pre_post}/h5/{selected_h5}"
        data = load_hdf5_data(uri)
        if data is None:
            st.error(f"Failed to load: {uri}")

    # View selector
    st.sidebar.header("Visualization")
    view_options = {v["label"]: v for v in config["views"]}
    selected_label = st.sidebar.selectbox(
        "View",
        list(view_options.keys()),
        index=0
    )

    # Render selected view
    if data is not None:
        view_cfg = view_options[selected_label]
        view_module = load_view(view_cfg["module"])
        view_module.render(data, view_cfg)
    else:
        st.info("Select a patient and HDF5 file to view data")


if __name__ == "__main__":
    main()

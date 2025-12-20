"""Streamlit app for HDF5 marker error visualization."""
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

import streamlit as st
import yaml
from pathlib import Path

from core.data import (
    load_hdf5_data, list_pids, list_hdf_files,
    list_sampled_dirs, load_sampled_hdf5,
    list_angle_sweep_h5_files, load_angle_sweep_h5
)
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
        parts.append(gmfcs)

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
    st.set_page_config(page_title="Rollout Browser", layout="wide")

    # Sidebar: Data Source
    st.sidebar.header("Data Source")

    # Refresh button to clear cache
    if st.sidebar.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    # Source type selection
    source_type = st.sidebar.radio(
        "Source Type",
        ["Sampled Rollout", "Patient (PID)", "Angle Sweep"],
        horizontal=True
    )

    data = None
    panel_data_list = []
    data_options = []  # List of selectable data items
    selected_item = None  # Default selected item

    if source_type == "Patient (PID)":
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

        if not h5_files:
            st.sidebar.info(f"No HDF5 files for {pre_post}")
        else:
            selected_h5 = st.sidebar.selectbox(
                "HDF5 File",
                h5_files,
                index=0
            )
            selected_item = selected_h5
            data_options = h5_files

    elif source_type == "Sampled Rollout":
        # Sampled directory selection
        sampled_dirs = list_sampled_dirs()
        if not sampled_dirs:
            st.sidebar.warning("No sampled directories found")
            st.info("No sampled rollout data available in sampled/")
            return

        # Default directory selector (used for single mode or as first panel)
        selected_dir = st.sidebar.selectbox(
            "Rollout Directory",
            sampled_dirs,
            index=0
        )
        selected_item = selected_dir
        data_options = sampled_dirs

    else:  # Angle Sweep
        # Angle sweep HDF5 file selection
        h5_files = list_angle_sweep_h5_files()
        if not h5_files:
            st.sidebar.warning("No angle sweep HDF5 files found")
            st.info("No angle sweep data available in results/")
            return

        selected_h5_file = st.sidebar.selectbox(
            "Trial HDF5",
            h5_files,
            index=0
        )
        selected_item = selected_h5_file
        data_options = h5_files

    # View selector - filter by source type
    st.sidebar.header("Visualization")
    source_key_map = {
        "Patient (PID)": "pid",
        "Sampled Rollout": "sampled",
        "Angle Sweep": "angle_sweep"
    }
    source_key = source_key_map.get(source_type, "sampled")
    available_views = [
        v for v in config["views"]
        if source_key in v.get("sources", [])
    ]

    if not available_views:
        st.warning(f"No views available for {source_type}")
        return

    view_options = {v["label"]: v for v in available_views}
    selected_label = st.sidebar.selectbox(
        "View",
        list(view_options.keys()),
        index=0
    )

    view_cfg = view_options[selected_label]
    view_module = load_view(view_cfg["module"])

    # Comparison mode toggle (under Visualization header)
    compare_mode = st.sidebar.checkbox("Comparison Mode")

    num_rows = 1
    num_cols = 1
    if compare_mode and data_options:
        # Grid configuration with select sliders
        col1, col2 = st.sidebar.columns(2)
        with col1:
            num_rows = st.select_slider("Rows", options=[1, 2, 3, 4], value=1)
        with col2:
            num_cols = st.select_slider("Cols", options=[1, 2, 3, 4], value=2)

        num_panels = int(num_rows * num_cols)

        # Panel selectors
        st.sidebar.markdown("**Panel Selection**")
        panel_items = []
        for i in range(num_panels):
            default_idx = min(i, len(data_options) - 1)
            panel_item = st.sidebar.selectbox(
                f"Panel {i + 1}",
                data_options,
                index=default_idx,
                key=f"panel_item_{i}"
            )
            panel_items.append(panel_item)

        # Load data for all panels based on source type
        for item in panel_items:
            if source_type == "Patient (PID)":
                uri = f"@pid:{selected_pid}/gait/{pre_post}/h5/{item}"
                pdata = load_hdf5_data(uri)
                if pdata is not None:
                    pdata['panel_title'] = item  # Use h5 filename as title
            elif source_type == "Sampled Rollout":
                pdata = load_sampled_hdf5(item)
                if pdata is not None:
                    pdata['panel_title'] = pdata.get('dir_name', item)
            else:  # Angle Sweep
                pdata = load_angle_sweep_h5(item)
                if pdata is not None:
                    pdata['panel_title'] = item

            if pdata is not None:
                panel_data_list.append(pdata)
            else:
                panel_data_list.append(None)

        # Use first panel's data as reference for controls
        data = panel_data_list[0] if panel_data_list else None
    else:
        # Single mode - load selected item
        if selected_item is not None:
            if source_type == "Patient (PID)":
                uri = f"@pid:{selected_pid}/gait/{pre_post}/h5/{selected_item}"
                data = load_hdf5_data(uri)
                if data is None:
                    st.error(f"Failed to load: {uri}")
            elif source_type == "Sampled Rollout":
                data = load_sampled_hdf5(selected_item)
                if data is None:
                    st.error(f"Failed to load sampled data: {selected_item}")
            else:  # Angle Sweep
                data = load_angle_sweep_h5(selected_item)
                if data is None:
                    st.error(f"Failed to load angle sweep data: {selected_item}")

    # Plot width preset
    width_options = {"30%": 0.3, "50%": 0.5, "100%": 1.0}
    width_label = st.sidebar.radio("Plot Width", list(width_options.keys()), horizontal=True, index=2)
    plot_width = width_options[width_label]

    # Render selected view
    if data is not None:
        if compare_mode and panel_data_list:
            # Check if view supports comparison mode
            if hasattr(view_module, 'render_controls') and hasattr(view_module, 'render_plot'):
                # Render header
                st.header(f"{selected_label} - Comparison")

                # Render shared controls at top of main page (pass all panel data for auto-range)
                controls = view_module.render_controls(data, view_cfg, all_data=panel_data_list)

                if controls is not None:
                    # Show control summary once at top (if enabled and view supports it)
                    if controls.get('show_info', True) and hasattr(view_module, 'get_summary'):
                        summary = view_module.get_summary(controls)
                        if summary:
                            st.caption(summary)

                    # Create container with specified width for entire grid
                    if plot_width < 1.0:
                        grid_container, _ = st.columns([plot_width, 1 - plot_width])
                    else:
                        grid_container = st.container()

                    with grid_container:
                        # Render plots in grid
                        for row_idx in range(int(num_rows)):
                            cols = st.columns(int(num_cols))
                            for col_idx, col in enumerate(cols):
                                panel_idx = row_idx * int(num_cols) + col_idx
                                if panel_idx < len(panel_data_list):
                                    pdata = panel_data_list[panel_idx]
                                    if pdata is not None:
                                        with col:
                                            view_module.render_plot(
                                                pdata,
                                                view_cfg,
                                                controls,
                                                title_prefix=pdata.get('panel_title', f'Panel {panel_idx + 1}'),
                                                plot_width=1.0  # Fill column, grid container handles overall width
                                            )
                                    else:
                                        with col:
                                            st.warning(f"Panel {panel_idx + 1}: Failed to load data")
            else:
                st.warning(f"View '{selected_label}' does not support comparison mode")
                st.info("Falling back to single view mode")
                view_module.render(data, view_cfg)
        else:
            # Single view mode - wrap in width container
            if plot_width < 1.0:
                view_container, _ = st.columns([plot_width, 1 - plot_width])
            else:
                view_container = st.container()
            with view_container:
                view_module.render(data, view_cfg)
    else:
        st.info("Select a data source to view")


if __name__ == "__main__":
    main()

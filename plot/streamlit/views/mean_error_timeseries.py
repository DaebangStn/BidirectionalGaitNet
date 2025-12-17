"""Mean error time series visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render_controls(data_sample: dict, cfg: dict, all_data: list = None) -> dict:
    """Render shared controls, return control values.

    Args:
        data_sample: Sample data to determine available options
        cfg: View configuration
        all_data: Optional list of all panel data (for comparison mode)

    Returns:
        Dict with control values, or None if no valid data
    """
    # Check what data is available
    has_free = 'marker_error_free_mean' in data_sample
    has_motion = 'marker_error_motion_mean' in data_sample

    if not has_free and not has_motion:
        st.warning("No mean error data available")
        return None

    # Character selection
    options = []
    if has_free and has_motion:
        options = ["both", "free", "motion"]
    elif has_free:
        options = ["free"]
    else:
        options = ["motion"]

    char_type = st.radio(
        "Character Type",
        options,
        format_func=lambda x: {
            "both": "Both Characters",
            "free": "Free Character Only",
            "motion": "Motion Character Only"
        }[x],
        horizontal=True
    )

    # Compute Y-axis range across all data
    if all_data:
        data_list = [d for d in all_data if d is not None]
    else:
        data_list = [data_sample]

    y_max = 0.0
    for d in data_list:
        if char_type == "both" or char_type == "free":
            if 'marker_error_free_mean' in d:
                y_max = max(y_max, float(np.nanmax(d['marker_error_free_mean'])))
        if char_type == "both" or char_type == "motion":
            if 'marker_error_motion_mean' in d:
                y_max = max(y_max, float(np.nanmax(d['marker_error_motion_mean'])))

    if y_max == 0.0:
        y_max = 10.0  # Fallback

    # Y-axis range controls
    st.caption("Y-axis Range (mm)")
    y_col1, y_col2, y_col3 = st.columns([1, 3, 1])

    # Initialize session state
    if "mets_ymin" not in st.session_state:
        st.session_state.mets_ymin = 0.0
    if "mets_ymax" not in st.session_state:
        st.session_state.mets_ymax = y_max
    if "mets_yslider" not in st.session_state:
        st.session_state.mets_yslider = (0.0, y_max)

    def sync_y_from_slider():
        slider_val = st.session_state.mets_yslider
        st.session_state.mets_ymin = slider_val[0]
        st.session_state.mets_ymax = slider_val[1]

    def sync_y_from_inputs():
        st.session_state.mets_yslider = (
            st.session_state.mets_ymin,
            st.session_state.mets_ymax
        )

    with y_col1:
        st.number_input("Min", min_value=0.0, max_value=y_max * 2,
                        step=1.0, format="%.1f", key="mets_ymin",
                        on_change=sync_y_from_inputs)
    with y_col2:
        st.slider(
            "Y Range",
            min_value=0.0,
            max_value=y_max * 1.2,
            step=y_max / 100 if y_max > 0 else 1.0,
            key="mets_yslider",
            label_visibility="collapsed",
            on_change=sync_y_from_slider
        )
    with y_col3:
        st.number_input("Max", min_value=0.0, max_value=y_max * 2,
                        step=1.0, format="%.1f", key="mets_ymax",
                        on_change=sync_y_from_inputs)

    ymin = st.session_state.mets_ymin
    ymax = st.session_state.mets_ymax

    # Display options
    show_info = st.checkbox("Show panel info", value=True)
    show_grid = st.checkbox("Show grid", value=True)

    return {
        'char_type': char_type,
        'ymin': ymin,
        'ymax': ymax,
        'show_info': show_info,
        'show_grid': show_grid,
    }


def get_summary(controls: dict) -> str:
    """Get a summary string of the current control values."""
    if controls is None:
        return ""
    char_labels = {
        "both": "Both",
        "free": "Free Only",
        "motion": "Motion Only"
    }
    parts = [
        f"**Type:** {char_labels[controls['char_type']]}",
        f"**Y Range:** [{controls['ymin']:.1f}, {controls['ymax']:.1f}] mm",
    ]
    return " | ".join(parts)


def render_plot(data: dict, cfg: dict, controls: dict, title_prefix: str = "", plot_width: float = 1.0) -> None:
    """Render plot using provided control values.

    Args:
        data: Data from load_hdf5_data()
        cfg: View configuration
        controls: Control values from render_controls()
        title_prefix: Optional prefix for plot title
        plot_width: Width of plot as fraction of container
    """
    if controls is None:
        st.warning("No valid controls")
        return

    char_type = controls['char_type']
    ymin = controls['ymin']
    ymax = controls['ymax']
    show_info = controls.get('show_info', True)
    show_grid = controls.get('show_grid', True)

    # Create container with specified width
    if plot_width < 1.0:
        plot_container, _ = st.columns([plot_width, 1 - plot_width])
    else:
        plot_container = st.container()

    with plot_container:
        # Show title/info
        if show_info:
            panel_title = title_prefix if title_prefix else f"PID: {data.get('pid', 'N/A')}"
            st.markdown(
                f'<code style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;font-size:0.8em">{panel_title}</code>',
                unsafe_allow_html=True
            )

        fig_width = 12 * plot_width
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        stats = []

        # Plot characters based on selection
        plot_configs = []
        if char_type == "both" or char_type == "free":
            plot_configs.append(('free', 'blue', 'Free Character'))
        if char_type == "both" or char_type == "motion":
            plot_configs.append(('motion', 'red', 'Motion Character'))

        for ctype, color, label in plot_configs:
            key = f'marker_error_{ctype}_mean'
            if key in data:
                mean_error = data[key]
                frames = np.arange(len(mean_error))
                ax.plot(frames, mean_error, color=color, label=label, linewidth=1.5)
                stats.append(f"{label}: {np.nanmean(mean_error):.2f}mm")

        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Mean Error (mm)')
        ax.set_ylim(ymin, ymax)
        ax.legend()
        if show_grid:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Statistics
        if show_info and stats:
            st.caption(" | ".join(stats))


def render(data: dict, cfg: dict) -> None:
    """Render mean error over time.

    Legacy single-view interface.
    """
    st.header("Mean Error Time Series")

    controls = render_controls(data, cfg)
    if controls is None:
        return

    render_plot(data, cfg, controls)

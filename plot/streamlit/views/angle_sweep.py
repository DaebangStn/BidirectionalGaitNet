"""Angle sweep visualization - joint angle vs passive force per muscle."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Metric labels and units
METRIC_INFO = {
    'fp': ('Passive Force', 'N'),
    'lm_norm': ('Normalized Length', ''),
    'jtp_mag': ('Joint Torque', 'Nm'),
}

# Color palette for muscles
MUSCLE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def compute_data_range(all_data_list: list, muscles: list, metric: str) -> tuple:
    """Compute min/max values across all data sources for a metric.

    Args:
        all_data_list: List of data dicts from multiple sources
        muscles: List of muscle names to include
        metric: 'fp', 'lm_norm', or 'jtp_mag'

    Returns:
        (min_val, max_val) or (None, None) if no data
    """
    all_values = []
    for data in all_data_list:
        if data is None:
            continue
        for muscle in muscles:
            muscle_data = data.get('muscle_data', {}).get(muscle, {})
            if metric in muscle_data:
                values = muscle_data[metric]
                if len(values) > 0:
                    all_values.extend(values)
    if all_values:
        return float(np.min(all_values)), float(np.max(all_values))
    return None, None


def render_controls(data_sample: dict, cfg: dict, all_data: list = None) -> dict:
    """Render shared controls, return control values.

    Args:
        data_sample: Sample data to determine available options
        cfg: View configuration
        all_data: Optional list of all panel data (for comparison mode auto-range)

    Returns:
        Dict with control values, or None if no valid data
    """
    muscles = data_sample.get('muscles', [])
    if not muscles:
        st.warning("No muscle data found")
        return None

    # Build list of all data for auto-range computation
    if all_data:
        all_data_list = [d for d in all_data if d and d.get('muscles')]
    else:
        all_data_list = [data_sample]

    # Metric selection
    metric_options = ['fp', 'lm_norm', 'jtp_mag']
    metric_labels = [f"{METRIC_INFO[m][0]} ({METRIC_INFO[m][1]})" if METRIC_INFO[m][1]
                     else METRIC_INFO[m][0] for m in metric_options]
    metric_idx = st.radio("Metric", range(len(metric_options)),
                          format_func=lambda i: metric_labels[i],
                          horizontal=True, key="metric_radio")
    selected_metric = metric_options[metric_idx]

    # Muscle selection with checkboxes
    label_col, sel_col, clr_col = st.columns([3, 1, 1])
    with label_col:
        st.markdown("**Select Muscles**")
    with sel_col:
        if st.button("Select All"):
            for muscle in muscles:
                st.session_state[f"chk_muscle_{muscle}"] = True
            st.rerun()
    with clr_col:
        if st.button("Clear All"):
            for muscle in muscles:
                st.session_state[f"chk_muscle_{muscle}"] = False
            st.rerun()

    selected_muscles = []
    # Display muscles in columns
    num_cols = min(4, len(muscles))
    cols = st.columns(num_cols)
    for i, muscle in enumerate(muscles):
        with cols[i % num_cols]:
            # Default: select first 3 muscles
            default_checked = i < 3
            if st.checkbox(muscle, value=default_checked, key=f"chk_muscle_{muscle}"):
                selected_muscles.append(muscle)

    if not selected_muscles:
        st.warning("Select at least one muscle")
        return None

    # Y-axis range control
    metric_label, metric_unit = METRIC_INFO[selected_metric]
    unit_str = f" ({metric_unit})" if metric_unit else ""

    st.markdown(f"**Y-Axis Range: {metric_label}{unit_str}**")

    # Compute data range for selected muscles
    data_min, data_max = compute_data_range(all_data_list, selected_muscles, selected_metric)

    # Set reasonable defaults based on metric
    if selected_metric == 'fp':
        abs_min, abs_max = 0.0, 500.0
        default_min, default_max = 0.0, 100.0
        step = 1.0
        fmt = "%.0f"
        padding = 10.0
    elif selected_metric == 'lm_norm':
        abs_min, abs_max = 0.0, 2.0
        default_min, default_max = 0.5, 1.5
        step = 0.01
        fmt = "%.2f"
        padding = 0.05
    else:  # jtp_mag
        abs_min, abs_max = 0.0, 100.0
        default_min, default_max = 0.0, 30.0
        step = 0.5
        fmt = "%.1f"
        padding = 2.0

    key_min = f"ymin_{selected_metric}"
    key_slider = f"yslider_{selected_metric}"
    key_max = f"ymax_{selected_metric}"

    label_col, auto_col = st.columns([4, 1])
    with auto_col:
        if st.button("Auto", key="auto_yrange"):
            if data_min is not None and data_max is not None:
                new_min = max(abs_min, data_min - padding)
                new_max = min(abs_max, data_max + padding)
                st.session_state[key_min] = new_min
                st.session_state[key_slider] = (new_min, new_max)
                st.session_state[key_max] = new_max
                st.rerun()

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        ymin = st.number_input("Min", min_value=abs_min, max_value=abs_max,
                               value=default_min, step=step, format=fmt, key=key_min)
    with c2:
        ymin_s, ymax_s = st.slider(
            "Range", min_value=abs_min, max_value=abs_max,
            value=(default_min, default_max), step=step,
            key=key_slider, label_visibility="collapsed"
        )
    with c3:
        ymax = st.number_input("Max", min_value=abs_min, max_value=abs_max,
                               value=default_max, step=step, format=fmt, key=key_max)

    # Use slider values if they differ from defaults (user dragged)
    if key_slider in st.session_state:
        ymin, ymax = st.session_state[key_slider]

    # Display options
    show_info = st.checkbox("Show panel info", value=True)
    show_legend = st.checkbox("Show legend", value=True)

    return {
        'selected_muscles': selected_muscles,
        'metric': selected_metric,
        'y_range': (ymin, ymax),
        'show_info': show_info,
        'show_legend': show_legend,
    }


def get_summary(controls: dict) -> str:
    """Get a summary string of the current control values."""
    if controls is None:
        return ""
    metric_label = METRIC_INFO[controls['metric']][0]
    muscle_count = len(controls.get('selected_muscles', []))
    return f"**Metric:** {metric_label} | **Muscles:** {muscle_count} selected"


def render_plot(data: dict, cfg: dict, controls: dict,
                title_prefix: str = "", plot_width: float = 1.0) -> None:
    """Render plot using provided control values.

    Args:
        data: Data from load_angle_sweep_csv()
        cfg: View configuration
        controls: Control values from render_controls()
        title_prefix: Optional prefix for plot title
        plot_width: Width of plot as fraction of container
    """
    if controls is None:
        st.warning("No valid controls")
        return

    selected_muscles = controls['selected_muscles']
    metric = controls['metric']
    y_range = controls['y_range']
    show_info = controls.get('show_info', True)
    show_legend = controls.get('show_legend', True)

    if not selected_muscles:
        st.warning("No muscles selected")
        return

    joint_angle_deg = data.get('joint_angle_deg')
    muscle_data = data.get('muscle_data', {})

    if joint_angle_deg is None:
        st.warning("No joint angle data")
        return

    # Create container with specified width
    if plot_width < 1.0:
        plot_container, _ = st.columns([plot_width, 1 - plot_width])
    else:
        plot_container = st.container()

    with plot_container:
        # Show filename label at top of panel
        if show_info:
            filename = title_prefix if title_prefix else data.get('filename', '')
            if filename:
                st.markdown(
                    f'<code style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;font-size:0.8em">{filename}</code>',
                    unsafe_allow_html=True
                )

        # Create figure
        fig_width = max(6, 4 * plot_width)
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        metric_label, metric_unit = METRIC_INFO[metric]
        unit_str = f" ({metric_unit})" if metric_unit else ""

        for i, muscle in enumerate(selected_muscles):
            color = MUSCLE_COLORS[i % len(MUSCLE_COLORS)]
            m_data = muscle_data.get(muscle, {})
            if metric in m_data:
                values = m_data[metric]
                ax.plot(joint_angle_deg, values, color=color,
                        linewidth=1.5, label=muscle)

        ax.set_xlabel('Joint Angle (deg)')
        ax.set_ylabel(f'{metric_label}{unit_str}')
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)

        if show_legend and len(selected_muscles) <= 10:
            ax.legend(loc='upper left', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Show summary stats
        if show_info:
            stats = []
            for muscle in selected_muscles[:3]:  # Show first 3
                m_data = muscle_data.get(muscle, {})
                if metric in m_data:
                    values = m_data[metric]
                    stats.append(f"{muscle}: {np.max(values):.1f} max")
            if stats:
                st.caption(" | ".join(stats))


def render(data: dict, cfg: dict) -> None:
    """Render angle sweep visualization.

    Legacy single-view interface.

    Args:
        data: Dict from load_angle_sweep_csv()
        cfg: View configuration dict
    """
    if data.get('source') != 'angle_sweep':
        st.warning("This view requires angle sweep data")
        return

    st.header("Angle Sweep - Passive Force")

    muscles = data.get('muscles', [])
    if not muscles:
        st.warning("No muscle data found")
        return

    controls = render_controls(data, cfg)
    if controls is None:
        return

    render_plot(data, cfg, controls)

    # Show info
    with st.expander("Data Info"):
        st.write(f"**File:** {data.get('filename', 'N/A')}")
        st.write(f"**Muscles:** {len(muscles)}")
        st.write(f"**Data points:** {len(data.get('joint_angle_deg', []))}")
        angle_range = data.get('joint_angle_deg', [])
        if len(angle_range) > 0:
            st.write(f"**Angle range:** {angle_range[0]:.1f} - {angle_range[-1]:.1f} deg")

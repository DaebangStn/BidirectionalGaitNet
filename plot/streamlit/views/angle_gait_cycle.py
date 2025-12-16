"""Joint angles and kinematics vs gait cycle visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

NUM_RESAMPLE_FRAMES = 500

# Define all known metrics with their properties
# key -> (prefix, label, color, unit)
METRIC_DEFINITIONS = {
    # Major angles
    "HipR": ("angle", "Hip", "#1f77b4", "deg"),
    "KneeR": ("angle", "Knee", "#ff7f0e", "deg"),
    "AnkleR": ("angle", "Ankle", "#2ca02c", "deg"),
    # Minor angles
    "HipIRR": ("angle", "Hip IR", "#d62728", "deg"),
    "HipAbR": ("angle", "Hip Ab", "#9467bd", "deg"),
    # Pelvis
    "Tilt": ("angle", "Pelvic Tilt", "#8c564b", "deg"),
    "Rotation": ("angle", "Pelvic Rot", "#e377c2", "deg"),
    "Obliquity": ("angle", "Pelvic Obl", "#7f7f7f", "deg"),
    # Sway position (meters)
    "FootR": ("sway", "Foot R", "#1f77b4", "m"),
    "FootL": ("sway", "Foot L", "#ff7f0e", "m"),
    "Torso": ("sway", "Torso", "#2ca02c", "m"),
    "ToeR": ("sway", "Toe R", "#d62728", "m"),
    "ToeL": ("sway", "Toe L", "#9467bd", "m"),
    # Sway angles (degrees)
    "FPA_R": ("sway", "FPA R", "#8c564b", "deg"),
    "FPA_L": ("sway", "FPA L", "#e377c2", "deg"),
    "AnteversionR": ("sway", "Antever R", "#17becf", "deg"),
    "AnteversionL": ("sway", "Antever L", "#bcbd22", "deg"),
}

# Default color cycle for unknown metrics
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def interpolate_cycle(angle_data: np.ndarray, num_frames: int = NUM_RESAMPLE_FRAMES) -> np.ndarray:
    """Interpolate angle data to fixed number of frames."""
    original_x = np.linspace(0, 1, len(angle_data))
    target_x = np.linspace(0, 1, num_frames)
    return np.interp(target_x, original_x, angle_data)


def compute_aggregate(cycles: dict, prefix: str, keys: list) -> dict:
    """Compute mean and std across all cycles after resampling."""
    result = {'mean': {}, 'std': {}}

    for key in keys:
        resampled = []
        for cycle_idx, cycle_data in cycles.items():
            if prefix in cycle_data and key in cycle_data[prefix]:
                values = cycle_data[prefix][key]
                if len(values) > 1:
                    resampled.append(interpolate_cycle(values))

        if resampled:
            stacked = np.stack(resampled, axis=0)
            result['mean'][key] = np.mean(stacked, axis=0)
            result['std'][key] = np.std(stacked, axis=0)

    return result


def get_available_metrics(cycles: dict) -> dict:
    """Get all available metrics from cycle data.

    Returns:
        Dict with 'angle' and 'sway' keys, each containing list of available metric keys
    """
    first_cycle = cycles[list(cycles.keys())[0]]
    available = {'angle': [], 'sway': []}

    for prefix in ['angle', 'sway']:
        if prefix in first_cycle:
            available[prefix] = list(first_cycle[prefix].keys())

    return available


def get_metric_info(key: str, prefix: str, color_idx: int = 0) -> tuple:
    """Get label, color, unit for a metric key.

    Returns:
        (label, color, unit)
    """
    if key in METRIC_DEFINITIONS:
        _, label, color, unit = METRIC_DEFINITIONS[key]
        return label, color, unit
    else:
        # Unknown metric - use key as label, default color
        color = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
        # Guess unit based on prefix
        unit = "deg" if prefix == "angle" else "m"
        return key, color, unit


def compute_data_range(all_cycles_list: list, prefix: str, key: str) -> tuple:
    """Compute min/max values across all cycles from multiple data sources.

    Args:
        all_cycles_list: List of cycles dicts from multiple data sources
        prefix: 'angle' or 'sway'
        key: metric key like 'HipR', 'KneeR', etc.
    """
    all_values = []
    for cycles in all_cycles_list:
        for cycle_data in cycles.values():
            if prefix in cycle_data and key in cycle_data[prefix]:
                values = cycle_data[prefix][key]
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
    cycles = data_sample.get('cycles', {})

    # Build list of all cycles for auto-range computation
    if all_data:
        all_cycles_list = [d.get('cycles', {}) for d in all_data if d and d.get('cycles')]
    else:
        all_cycles_list = [cycles]
    averaged = data_sample.get('averaged')

    if not cycles:
        return None

    # Get available metrics
    available = get_available_metrics(cycles)
    all_angle_keys = available.get('angle', [])
    all_sway_keys = available.get('sway', [])

    if not all_angle_keys and not all_sway_keys:
        return None

    # Metric selection with checkboxes
    label_col, btn_col = st.columns([4, 1])
    with label_col:
        st.markdown("**Select Metrics**")
    with btn_col:
        if st.button("Clear All"):
            # Clear all checkbox states
            for key in all_angle_keys:
                st.session_state[f"chk_angle_{key}"] = False
            for key in all_sway_keys:
                st.session_state[f"chk_sway_{key}"] = False
            st.rerun()

    selected_metrics = []  # List of (key, prefix)

    # Angle metrics
    if all_angle_keys:
        st.caption("Angles")
        angle_cols = st.columns(min(4, len(all_angle_keys)))
        for i, key in enumerate(all_angle_keys):
            label, _, _ = get_metric_info(key, 'angle', i)
            with angle_cols[i % len(angle_cols)]:
                # Default: select first 3 angle metrics
                default_checked = i < 3
                if st.checkbox(label, value=default_checked, key=f"chk_angle_{key}"):
                    selected_metrics.append((key, 'angle'))

    # Sway metrics
    if all_sway_keys:
        st.caption("Sway")
        sway_cols = st.columns(min(4, len(all_sway_keys)))
        for i, key in enumerate(all_sway_keys):
            label, _, _ = get_metric_info(key, 'sway', i)
            with sway_cols[i % len(sway_cols)]:
                if st.checkbox(label, value=False, key=f"chk_sway_{key}"):
                    selected_metrics.append((key, 'sway'))

    if not selected_metrics:
        st.warning("Select at least one metric")
        return None

    # Mode selection
    mode_options = ["Aggregate (Mean ± Std)", "Single Cycle"]
    if averaged is not None:
        mode_options.append("Pre-computed Average")

    mode = st.radio("Display Mode", mode_options, horizontal=True)

    # Cycle index (only for Single Cycle mode)
    cycle_idx = None
    if mode == "Single Cycle":
        sorted_indices = sorted(cycles.keys())
        cycle_idx = st.slider(
            "Cycle Index",
            min_value=min(sorted_indices),
            max_value=max(sorted_indices),
            value=min(sorted_indices),
            step=1
        )

    # Y-axis range controls for selected metrics
    y_ranges = {}
    if selected_metrics:
        st.markdown("**Y-Axis Ranges**")
        for idx, (key, prefix) in enumerate(selected_metrics):
            label, _, unit = get_metric_info(key, prefix, idx)

            # Ranges based on unit type
            is_deg = unit == "deg"
            if is_deg:
                abs_min, abs_max = -180.0, 180.0
                default_min, default_max = -30.0, 90.0
                step = 1.0
                fmt = "%.0f"
                padding = 5.0
            else:  # meters
                abs_min, abs_max = -2.0, 2.0
                default_min, default_max = -0.5, 0.5
                step = 0.01
                fmt = "%.2f"
                padding = 0.05

            # Widget keys
            key_min = f"ymin_{key}"
            key_slider = f"yslider_{key}"
            key_max = f"ymax_{key}"

            # Layout: label + auto button | min | slider | max
            label_col, auto_col = st.columns([4, 1])
            with label_col:
                st.caption(f"{label} ({unit})")
            with auto_col:
                if st.button("Auto", key=f"auto_{key}"):
                    data_min, data_max = compute_data_range(all_cycles_list, prefix, key)
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
            y_ranges[key] = (ymin, ymax)

    # Display options
    show_info = st.checkbox("Show panel info", value=True)

    return {
        'selected_metrics': selected_metrics,  # List of (key, prefix)
        'mode': mode,
        'cycle_idx': cycle_idx,
        'y_ranges': y_ranges,
        'show_info': show_info,
    }


def get_summary(controls: dict) -> str:
    """Get a summary string of the current control values."""
    if controls is None:
        return ""
    # Get labels for selected metrics
    metric_labels = []
    for key, prefix in controls.get('selected_metrics', []):
        label, _, _ = get_metric_info(key, prefix)
        metric_labels.append(label)

    parts = [
        f"**Metrics:** {', '.join(metric_labels)}",
        f"**Mode:** {controls['mode']}",
    ]
    if controls['mode'] == "Single Cycle" and controls['cycle_idx'] is not None:
        parts.append(f"**Cycle:** {controls['cycle_idx']}")
    return " | ".join(parts)


def render_plot(data: dict, cfg: dict, controls: dict, title_prefix: str = "") -> None:
    """Render plot using provided control values.

    Args:
        data: Data from load_sampled_hdf5()
        cfg: View configuration
        controls: Control values from render_controls()
        title_prefix: Optional prefix for plot title
    """
    if controls is None:
        st.warning("No valid controls")
        return

    cycles = data.get('cycles', {})
    averaged = data.get('averaged')

    if not cycles:
        st.warning("No cycle data found")
        return

    selected_metrics = controls['selected_metrics']
    mode = controls['mode']
    cycle_idx = controls['cycle_idx']
    y_ranges = controls.get('y_ranges', {})
    show_info = controls.get('show_info', True)

    if not selected_metrics:
        st.warning("No metrics selected")
        return

    # Show directory label at top of panel (single line, no wrap)
    if show_info:
        dir_name = title_prefix if title_prefix else data.get('dir_name', '')
        if dir_name:
            st.markdown(
                f'<code style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;font-size:0.8em">{dir_name}</code>',
                unsafe_allow_html=True
            )

    # Build plot info from selected metrics
    plot_info = []  # List of (key, prefix, label, color, unit)
    for idx, (key, prefix) in enumerate(selected_metrics):
        label, color, unit = get_metric_info(key, prefix, idx)
        plot_info.append((key, prefix, label, color, unit))

    # Create figure
    n_plots = len(plot_info)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    gait_pct = np.linspace(0, 100, NUM_RESAMPLE_FRAMES)

    if mode == "Single Cycle":
        cycle_data = cycles.get(cycle_idx)
        if cycle_data is None:
            st.error(f"Cycle {cycle_idx} not found")
            plt.close(fig)
            return

        for ax, (key, prefix, label, color, unit) in zip(axes, plot_info):
            if prefix not in cycle_data or key not in cycle_data[prefix]:
                ax.text(0.5, 0.5, f"No {label} data", ha='center', va='center', transform=ax.transAxes)
            else:
                values = cycle_data[prefix][key]
                interpolated = interpolate_cycle(values)
                ax.plot(gait_pct, interpolated, color=color, linewidth=1.5)

            ax.set_ylabel(f'{label} ({unit})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            if key in y_ranges:
                ax.set_ylim(y_ranges[key])

        title_suffix = f"Cycle {cycle_idx}"

    elif mode == "Aggregate (Mean ± Std)":
        # Compute aggregate for each metric separately (may have different prefixes)
        for ax, (key, prefix, label, color, unit) in zip(axes, plot_info):
            agg = compute_aggregate(cycles, prefix, [key])
            if key not in agg['mean']:
                ax.text(0.5, 0.5, f"No {label} data", ha='center', va='center', transform=ax.transAxes)
            else:
                mean = agg['mean'][key]
                std = agg['std'][key]
                ax.plot(gait_pct, mean, color=color, linewidth=1.5, label='Mean')
                ax.fill_between(gait_pct, mean - std, mean + std, color=color, alpha=0.3, label='±1 Std')

            ax.set_ylabel(f'{label} ({unit})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            if key in y_ranges:
                ax.set_ylim(y_ranges[key])

        title_suffix = f"Aggregate ({len(cycles)} cycles)"

    else:  # Pre-computed Average
        for ax, (key, prefix, label, color, unit) in zip(axes, plot_info):
            if prefix not in averaged or key not in averaged[prefix]:
                ax.text(0.5, 0.5, f"No {label} data", ha='center', va='center', transform=ax.transAxes)
            else:
                values = averaged[prefix][key]
                if averaged.get('phase') is not None and len(averaged['phase']) == len(values):
                    x = averaged['phase'] * 100
                else:
                    x = np.linspace(0, 100, len(values))
                ax.plot(x, values, color=color, linewidth=1.5)

            ax.set_ylabel(f'{label} ({unit})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            if key in y_ranges:
                ax.set_ylim(y_ranges[key])

        title_suffix = "Pre-computed Average"

    for ax in axes:
        ax.set_xlabel('Gait Cycle (%)')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Compute and display mean values across all gait cycles
    if show_info:
        mean_texts = []
        for key, prefix, label, color, unit in plot_info:
            all_values = []
            for cycle_data in cycles.values():
                if prefix in cycle_data and key in cycle_data[prefix]:
                    values = cycle_data[prefix][key]
                    if len(values) > 0:
                        all_values.extend(values)
            if all_values:
                mean_val = np.mean(all_values)
                mean_texts.append(f"{label}: {mean_val:.2f} {unit}")
        if mean_texts:
            st.caption(" | ".join(mean_texts))


def render(data: dict, cfg: dict) -> None:
    """Render kinematics vs normalized gait cycle (0-100%).

    Legacy single-view interface.

    Args:
        data: Dict from load_sampled_hdf5() with 'cycles' and optional 'averaged'
        cfg: View configuration dict
    """
    if data.get('source') != 'sampled':
        st.warning("This view requires sampled rollout data")
        return

    st.header("Kinematics vs Gait Cycle")

    cycles = data.get('cycles', {})
    if not cycles:
        st.warning("No cycle data found")
        return

    controls = render_controls(data, cfg)
    if controls is None:
        st.warning("No kinematics data found in the cycles")
        return

    render_plot(data, cfg, controls)

    # Show stats
    with st.expander("Data Info"):
        st.write(f"**Directory:** {data['dir_name']}")
        metric_labels = [get_metric_info(k, p)[0] for k, p in controls.get('selected_metrics', [])]
        st.write(f"**Metrics:** {', '.join(metric_labels)}")
        st.write(f"**Number of cycles:** {len(cycles)}")
        st.write(f"**Resample frames:** {NUM_RESAMPLE_FRAMES}")
        st.write(f"**Pre-computed average available:** {'Yes' if data.get('averaged') else 'No'}")

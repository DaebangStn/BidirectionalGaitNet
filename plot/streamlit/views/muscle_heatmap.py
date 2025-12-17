"""Muscle activity heatmap visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Rectangle, Patch

NUM_RESAMPLE_FRAMES = 500


def parse_muscle_name(name: str) -> tuple:
    """Parse muscle name into (side, base_name, full_name)."""
    side = 'L' if name.startswith('L_') else 'R'
    base = re.sub(r'^[LR]_', '', name)
    base_group = re.sub(r'\d+$', '', base)
    return side, base_group, name


def get_muscle_groups(muscle_names: list) -> dict:
    """Get unique muscle groups from muscle names."""
    groups = {}
    for idx, name in enumerate(muscle_names):
        _, base, _ = parse_muscle_name(name)
        if base not in groups:
            groups[base] = []
        groups[base].append(idx)
    return groups


def interpolate_to_gait_cycle(data: np.ndarray, num_frames: int = NUM_RESAMPLE_FRAMES) -> np.ndarray:
    """Interpolate data to fixed number of frames."""
    original_x = np.linspace(0, 1, data.shape[0])
    target_x = np.linspace(0, 1, num_frames)

    if data.ndim == 1:
        return np.interp(target_x, original_x, data)
    else:
        result = np.zeros((num_frames, data.shape[1]))
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_x, original_x, data[:, i])
        return result


def render_controls(data_sample: dict, cfg: dict, all_data: list = None) -> dict:
    """Render shared controls, return control values.

    Args:
        data_sample: Sample data to determine available options
        cfg: View configuration
        all_data: Optional list of all panel data (for comparison mode)

    Returns:
        Dict with control values, or None if no valid data
    """
    cycles = data_sample.get('cycles', {})
    muscle_names = data_sample.get('muscle_names')

    if not cycles or not muscle_names:
        return None

    first_cycle = next(iter(cycles.values()))
    if not first_cycle.get('muscle'):
        return None

    available_metrics = list(first_cycle['muscle'].keys())
    if not available_metrics:
        return None

    # --- Controls ---
    col1, col2 = st.columns(2)

    with col1:
        metric = st.radio(
            "Metric",
            available_metrics,
            horizontal=True,
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col2:
        side_filter = st.radio("Side", ["Right", "Both", "Left"], horizontal=True)

    mode = st.radio("Mode", ["Aggregate", "Single Cycle"], horizontal=True)

    # Muscle index range filter
    total_muscles = len(muscle_names)

    # Initialize session state for muscle index
    if "muscle_idx_from" not in st.session_state:
        st.session_state.muscle_idx_from = 0
    if "muscle_idx_to" not in st.session_state:
        st.session_state.muscle_idx_to = total_muscles - 1
    if "muscle_idx_slider" not in st.session_state:
        st.session_state.muscle_idx_slider = (0, total_muscles - 1)

    def sync_muscle_from_slider():
        slider_val = st.session_state.muscle_idx_slider
        st.session_state.muscle_idx_from = slider_val[0]
        st.session_state.muscle_idx_to = slider_val[1]

    def sync_muscle_from_inputs():
        st.session_state.muscle_idx_slider = (
            st.session_state.muscle_idx_from,
            st.session_state.muscle_idx_to
        )

    st.caption(f"Muscle Index Range (0-{total_muscles - 1})")
    idx_col1, idx_col2, idx_col3 = st.columns([1, 3, 1])
    with idx_col1:
        st.number_input("From", min_value=0, max_value=total_muscles - 1,
                        step=1, key="muscle_idx_from", on_change=sync_muscle_from_inputs)
    with idx_col2:
        st.slider(
            "Range", min_value=0, max_value=total_muscles - 1,
            step=1, key="muscle_idx_slider", label_visibility="collapsed",
            on_change=sync_muscle_from_slider
        )
    with idx_col3:
        st.number_input("To", min_value=0, max_value=total_muscles - 1,
                        step=1, key="muscle_idx_to", on_change=sync_muscle_from_inputs)

    # Read final values from session state
    idx_from = st.session_state.muscle_idx_from
    idx_to = st.session_state.muscle_idx_to

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

    # Build display indices based on index range and side filter
    display_indices = []
    display_names = []
    display_sides = []

    for idx in range(idx_from, idx_to + 1):
        side, base, full_name = parse_muscle_name(muscle_names[idx])
        if side_filter == "Both" or \
           (side_filter == "Left" and side == 'L') or \
           (side_filter == "Right" and side == 'R'):
            display_indices.append(idx)
            abbrev = re.sub(r'^[LR]_', '', full_name)
            display_names.append(abbrev)
            display_sides.append(side)

    if not display_indices:
        st.warning("No muscles match the current filter")
        return None

    # Color range - compute data max for reference
    # Use first cycle to estimate range
    if mode == "Single Cycle" and cycle_idx is not None:
        sample_data = cycles[cycle_idx]['muscle'].get(metric)
    else:
        sample_data = first_cycle['muscle'].get(metric)

    if sample_data is not None:
        interp_sample = interpolate_to_gait_cycle(sample_data)
        data_max = float(np.nanmax(interp_sample[:, display_indices]))
        if metric == 'activation':
            data_max = min(data_max, 1.0)
    else:
        data_max = 1.0

    # Color range controls - initialize session state
    if "cbar_vmin" not in st.session_state:
        st.session_state.cbar_vmin = 0.0
    if "cbar_vmax" not in st.session_state:
        st.session_state.cbar_vmax = data_max
    if "cbar_slider" not in st.session_state:
        st.session_state.cbar_slider = (0.0, data_max)

    def sync_cbar_from_slider():
        slider_val = st.session_state.cbar_slider
        st.session_state.cbar_vmin = slider_val[0]
        st.session_state.cbar_vmax = slider_val[1]

    def sync_cbar_from_inputs():
        st.session_state.cbar_slider = (
            st.session_state.cbar_vmin,
            st.session_state.cbar_vmax
        )

    cbar_col1, cbar_col2, cbar_col3 = st.columns([1, 3, 1])
    with cbar_col1:
        st.number_input("Min", min_value=0.0, max_value=data_max,
                        step=0.01, format="%.3f", key="cbar_vmin",
                        on_change=sync_cbar_from_inputs)
    with cbar_col2:
        st.slider(
            "Color Range",
            min_value=0.0,
            max_value=data_max,
            step=data_max / 100 if data_max > 0 else 0.01,
            key="cbar_slider",
            label_visibility="collapsed",
            on_change=sync_cbar_from_slider
        )
    with cbar_col3:
        st.number_input("Max", min_value=0.0, max_value=data_max,
                        step=0.01, format="%.3f", key="cbar_vmax",
                        on_change=sync_cbar_from_inputs)

    # Read final values from session state
    vmin = st.session_state.cbar_vmin
    vmax = st.session_state.cbar_vmax

    # Display options
    show_info = st.checkbox("Show panel info", value=True)

    return {
        'metric': metric,
        'side_filter': side_filter,
        'mode': mode,
        'idx_range': (idx_from, idx_to),
        'cycle_idx': cycle_idx,
        'display_indices': display_indices,
        'display_names': display_names,
        'display_sides': display_sides,
        'vmin': vmin,
        'vmax': vmax,
        'show_info': show_info,
    }


def get_summary(controls: dict) -> str:
    """Get a summary string of the current control values."""
    if controls is None:
        return ""
    metric_label = controls['metric'].replace('_', ' ').title()
    idx_from, idx_to = controls['idx_range']
    parts = [
        f"**Metric:** {metric_label}",
        f"**Side:** {controls['side_filter']}",
        f"**Mode:** {controls['mode']}",
        f"**Muscles:** [{idx_from}-{idx_to}]",
    ]
    if controls['mode'] == "Single Cycle" and controls['cycle_idx'] is not None:
        parts.append(f"**Cycle:** {controls['cycle_idx']}")
    parts.append(f"**Color:** [{controls['vmin']:.2f}, {controls['vmax']:.2f}]")
    return " | ".join(parts)


def render_plot(data: dict, cfg: dict, controls: dict, title_prefix: str = "", plot_width: float = 1.0) -> None:
    """Render plot using provided control values.

    Args:
        data: Data from load_sampled_hdf5()
        cfg: View configuration
        controls: Control values from render_controls()
        title_prefix: Optional prefix for plot title
        plot_width: Width of plot as fraction of container (0.3, 0.5, 1.0)
    """
    if controls is None:
        st.warning("No valid controls")
        return

    cycles = data.get('cycles', {})
    muscle_names = data.get('muscle_names')

    if not cycles or not muscle_names:
        st.warning("No cycle or muscle data found")
        return

    metric = controls['metric']
    mode = controls['mode']
    cycle_idx = controls['cycle_idx']
    display_indices = controls['display_indices']
    display_names = controls['display_names']
    display_sides = controls['display_sides']
    side_filter = controls['side_filter']
    vmin = controls['vmin']
    vmax = controls['vmax']
    show_info = controls.get('show_info', True)

    # Create container with specified width
    if plot_width < 1.0:
        plot_container, _ = st.columns([plot_width, 1 - plot_width])
    else:
        plot_container = st.container()

    with plot_container:
        # Show directory label at top of panel (single line, no wrap)
        if show_info:
            dir_name = title_prefix if title_prefix else data.get('dir_name', '')
            if dir_name:
                st.markdown(
                    f'<code style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;font-size:0.8em">{dir_name}</code>',
                    unsafe_allow_html=True
                )

        # --- Get data ---
        sorted_cycle_indices = sorted(cycles.keys())

        if mode == "Single Cycle":
            cycle_data = cycles[cycle_idx]
            if metric not in cycle_data['muscle']:
                st.error(f"Metric '{metric}' not found in cycle {cycle_idx}")
                return

            raw_data = cycle_data['muscle'][metric]
            interp_data = interpolate_to_gait_cycle(raw_data)
            heatmap_data = interp_data[:, display_indices].T

        else:  # Aggregate
            all_cycles = []
            for cidx in sorted_cycle_indices:
                if metric in cycles[cidx]['muscle']:
                    raw = cycles[cidx]['muscle'][metric]
                    interp = interpolate_to_gait_cycle(raw)
                    all_cycles.append(interp[:, display_indices])

            if not all_cycles:
                st.error("No valid cycles for aggregation")
                return

            stacked = np.stack(all_cycles, axis=0)
            mean_data = np.mean(stacked, axis=0)
            heatmap_data = mean_data.T

        # --- Plot ---
        fig_width = 14 * plot_width
        fig, ax = plt.subplots(figsize=(fig_width, max(6, len(display_indices) * 0.15)))

        im = ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
            extent=[0, 100, len(display_indices) - 0.5, -0.5]
        )

        ax.set_xlabel('Gait Cycle (%)')
        ax.set_xlim(0, 100)
        ax.set_ylabel('Muscle')
        ax.set_yticks(range(len(display_names)))
        ax.set_yticklabels(display_names, fontsize=7)

        # Add side indicator
        for i, side in enumerate(display_sides):
            color = '#1f77b4' if side == 'L' else '#ff7f0e'
            rect = Rectangle((-0.02, i - 0.5), 0.015, 1, facecolor=color, edgecolor='none',
                            clip_on=False, transform=ax.get_yaxis_transform())
            ax.add_patch(rect)

        # Add legend for sides
        if side_filter == "Both":
            legend_elements = [
                Patch(facecolor='#1f77b4', label='Left'),
                Patch(facecolor='#ff7f0e', label='Right')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        # Colorbar
        metric_label = metric.replace('_', ' ').title()
        fig.colorbar(im, ax=ax, label=metric_label, shrink=0.8)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Display mean and std across all muscles
        if show_info:
            mean_val = np.nanmean(heatmap_data)
            std_val = np.nanstd(heatmap_data)
            st.caption(f"Mean: {mean_val:.4f} | Std: {std_val:.4f}")


def render(data: dict, cfg: dict) -> None:
    """Render muscle activity heatmap.

    Legacy single-view interface.
    """
    if data.get('source') != 'sampled':
        st.warning("This view requires sampled rollout data")
        return

    st.header("Muscle Activity Heatmap")

    cycles = data.get('cycles', {})
    muscle_names = data.get('muscle_names')

    if not cycles:
        st.warning("No cycle data found")
        return

    if not muscle_names:
        st.warning("No muscle names found")
        return

    first_cycle = next(iter(cycles.values()))
    if not first_cycle.get('muscle'):
        st.warning("No muscle data in cycles")
        return

    controls = render_controls(data, cfg)
    if controls is None:
        return

    render_plot(data, cfg, controls)

    # Statistics
    st.subheader("Statistics")
    # Recompute heatmap_data for stats (simplified)
    metric = controls['metric']
    mode = controls['mode']
    cycle_idx = controls['cycle_idx']
    display_indices = controls['display_indices']

    if mode == "Single Cycle":
        raw_data = cycles[cycle_idx]['muscle'][metric]
        interp_data = interpolate_to_gait_cycle(raw_data)
        heatmap_data = interp_data[:, display_indices].T
    else:
        all_cycles = []
        for cidx in sorted(cycles.keys()):
            if metric in cycles[cidx]['muscle']:
                raw = cycles[cidx]['muscle'][metric]
                interp = interpolate_to_gait_cycle(raw)
                all_cycles.append(interp[:, display_indices])
        stacked = np.stack(all_cycles, axis=0)
        heatmap_data = np.mean(stacked, axis=0).T

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{np.nanmean(heatmap_data):.4f}")
    with col2:
        st.metric("Max", f"{np.nanmax(heatmap_data):.4f}")
    with col3:
        st.metric("Min", f"{np.nanmin(heatmap_data):.4f}")
    with col4:
        st.metric("Muscles Shown", f"{len(display_indices)}")

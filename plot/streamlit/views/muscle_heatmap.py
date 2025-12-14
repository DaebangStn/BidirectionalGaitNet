"""Muscle activity heatmap visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Rectangle

NUM_RESAMPLE_FRAMES = 500


def parse_muscle_name(name: str) -> tuple:
    """Parse muscle name into (side, base_name, full_name).

    Examples:
        'L_Gluteus_Maximus' -> ('L', 'Gluteus_Maximus', 'L_Gluteus_Maximus')
        'R_Bicep_Femoris1' -> ('R', 'Bicep_Femoris', 'R_Bicep_Femoris1')
    """
    side = 'L' if name.startswith('L_') else 'R'
    # Remove L_/R_ prefix
    base = re.sub(r'^[LR]_', '', name)
    # Remove trailing numbers for grouping
    base_group = re.sub(r'\d+$', '', base)
    return side, base_group, name


def get_muscle_groups(muscle_names: list) -> dict:
    """Get unique muscle groups from muscle names.

    Returns dict: {base_name: [indices]}
    """
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
        # 2D array (frames, muscles)
        result = np.zeros((num_frames, data.shape[1]))
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_x, original_x, data[:, i])
        return result


def render(data: dict, cfg: dict) -> None:
    """Render muscle activity heatmap."""
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

    # Check if muscle data exists
    first_cycle = next(iter(cycles.values()))
    if not first_cycle.get('muscle'):
        st.warning("No muscle data in cycles")
        return

    # Get available metrics
    available_metrics = list(first_cycle['muscle'].keys())
    if not available_metrics:
        st.warning("No muscle metrics available")
        return

    # Get muscle groups
    muscle_groups = get_muscle_groups(muscle_names)
    sorted_groups = sorted(muscle_groups.keys())

    # --- Controls ---
    col1, col2 = st.columns(2)

    with col1:
        # Metric selection (radio button)
        metric = st.radio(
            "Metric",
            available_metrics,
            horizontal=True,
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col2:
        # Side filter
        side_filter = st.radio("Side", ["Right", "Both", "Left"], horizontal=True)

    # Mode selection
    mode = st.radio("Mode", ["Single Cycle", "Aggregate"], horizontal=True)

    # Muscle group filter
    with st.expander("Filter Muscle Groups", expanded=False):
        select_all = st.checkbox("Select All", value=True)
        if select_all:
            selected_groups = sorted_groups
        else:
            selected_groups = st.multiselect(
                "Muscle Groups",
                sorted_groups,
                default=sorted_groups[:10]  # Default to first 10
            )

    if not selected_groups:
        st.warning("Please select at least one muscle group")
        return

    # Build muscle indices to display
    display_indices = []
    display_names = []
    display_sides = []

    for group in selected_groups:
        for idx in muscle_groups[group]:
            side, base, full_name = parse_muscle_name(muscle_names[idx])
            if side_filter == "Both" or \
               (side_filter == "Left" and side == 'L') or \
               (side_filter == "Right" and side == 'R'):
                display_indices.append(idx)
                # Create abbreviated name (without L_/R_ prefix)
                abbrev = re.sub(r'^[LR]_', '', full_name)
                display_names.append(abbrev)
                display_sides.append(side)

    if not display_indices:
        st.warning("No muscles match the current filter")
        return

    # --- Get data ---
    sorted_cycle_indices = sorted(cycles.keys())
    min_cycle = min(sorted_cycle_indices)
    max_cycle = max(sorted_cycle_indices)

    if mode == "Single Cycle":
        cycle_idx = st.slider(
            "Cycle Index",
            min_value=min_cycle,
            max_value=max_cycle,
            value=min_cycle,
            step=1
        )

        cycle_data = cycles[cycle_idx]
        if metric not in cycle_data['muscle']:
            st.error(f"Metric '{metric}' not found in cycle {cycle_idx}")
            return

        raw_data = cycle_data['muscle'][metric]  # (N, 164)
        # Interpolate and select muscles
        interp_data = interpolate_to_gait_cycle(raw_data)
        heatmap_data = interp_data[:, display_indices].T  # (num_muscles, 500)
        title_suffix = f"Cycle {cycle_idx}"

    else:  # Aggregate
        # Collect all cycles
        all_cycles = []
        for cidx in sorted_cycle_indices:
            if metric in cycles[cidx]['muscle']:
                raw = cycles[cidx]['muscle'][metric]
                interp = interpolate_to_gait_cycle(raw)
                all_cycles.append(interp[:, display_indices])

        if not all_cycles:
            st.error("No valid cycles for aggregation")
            return

        # Stack and compute mean: (num_cycles, 500, num_muscles) -> (500, num_muscles)
        stacked = np.stack(all_cycles, axis=0)
        mean_data = np.mean(stacked, axis=0)
        heatmap_data = mean_data.T  # (num_muscles, 500)
        title_suffix = f"Aggregate ({len(all_cycles)} cycles)"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, max(6, len(display_indices) * 0.15)))

    # Determine color scale
    data_max = float(np.nanmax(heatmap_data))
    if metric == 'activation':
        data_max = min(data_max, 1.0)  # Activation is typically 0-1

    # Color range controls
    cbar_col1, cbar_col2, cbar_col3 = st.columns([1, 3, 1])
    with cbar_col1:
        vmin = st.number_input("Min", min_value=0.0, max_value=data_max, value=0.0, step=0.01, format="%.3f")
    with cbar_col2:
        vmin_slider, vmax_slider = st.slider(
            "Color Range",
            min_value=0.0,
            max_value=data_max,
            value=(vmin, data_max),
            step=data_max / 100 if data_max > 0 else 0.01,
            label_visibility="collapsed"
        )
        # Use slider values if they differ from number inputs
        if vmin_slider != vmin:
            vmin = vmin_slider
        vmax = vmax_slider
    with cbar_col3:
        vmax = st.number_input("Max", min_value=0.0, max_value=data_max, value=vmax, step=0.01, format="%.3f")

    im = ax.imshow(
        heatmap_data,
        aspect='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        extent=[0, 100, len(display_indices) - 0.5, -0.5]
    )

    # X-axis
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_xlim(0, 100)

    # Y-axis: muscle names
    ax.set_ylabel('Muscle')
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=7)

    # Add side indicator (thin color bar on left)
    for i, side in enumerate(display_sides):
        color = '#1f77b4' if side == 'L' else '#ff7f0e'  # Blue for L, Orange for R
        rect = Rectangle((-0.02, i - 0.5), 0.015, 1, facecolor=color, edgecolor='none',
                        clip_on=False, transform=ax.get_yaxis_transform())
        ax.add_patch(rect)

    # Add legend for sides
    from matplotlib.patches import Patch
    if side_filter == "Both":
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Left'),
            Patch(facecolor='#ff7f0e', label='Right')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    # Title
    metric_label = metric.replace('_', ' ').title()
    ax.set_title(f'{metric_label} - {data["dir_name"]} ({title_suffix})')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label=metric_label, shrink=0.8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{np.nanmean(heatmap_data):.4f}")
    with col2:
        st.metric("Max", f"{np.nanmax(heatmap_data):.4f}")
    with col3:
        st.metric("Min", f"{np.nanmin(heatmap_data):.4f}")
    with col4:
        st.metric("Muscles Shown", f"{len(display_indices)}")

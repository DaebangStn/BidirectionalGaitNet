"""Joint angles and kinematics vs gait cycle visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

NUM_RESAMPLE_FRAMES = 500

# Define metric categories
METRIC_CATEGORIES = {
    "Major Angles": {
        "prefix": "angle",
        "keys": ["HipR", "KneeR", "AnkleR"],
        "labels": ["Hip", "Knee", "Ankle"],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        "ylabel": "deg"
    },
    "Minor Angles": {
        "prefix": "angle",
        "keys": ["HipIRR", "HipAbR"],
        "labels": ["Hip Internal Rotation", "Hip Abduction"],
        "colors": ["#d62728", "#9467bd"],
        "ylabel": "deg"
    },
    "Pelvis": {
        "prefix": "angle",
        "keys": ["Tilt", "Rotation", "Obliquity"],
        "labels": ["Pelvic Tilt", "Pelvic Rotation", "Pelvic Obliquity"],
        "colors": ["#8c564b", "#e377c2", "#7f7f7f"],
        "ylabel": "deg"
    },
    "Sway (Position)": {
        "prefix": "sway",
        "keys": ["FootR", "FootL", "Torso"],
        "labels": ["Foot R", "Foot L", "Torso"],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        "ylabel": "m"
    },
    "Sway (Toe)": {
        "prefix": "sway",
        "keys": ["ToeR", "ToeL"],
        "labels": ["Toe R (Y)", "Toe L (Y)"],
        "colors": ["#d62728", "#9467bd"],
        "ylabel": "m"
    },
    "Foot Progression": {
        "prefix": "sway",
        "keys": ["FPA_R", "FPA_L"],
        "labels": ["FPA Right", "FPA Left"],
        "colors": ["#8c564b", "#e377c2"],
        "ylabel": "deg"
    },
    "Anteversion": {
        "prefix": "sway",
        "keys": ["AnteversionR", "AnteversionL"],
        "labels": ["Anteversion R", "Anteversion L"],
        "colors": ["#17becf", "#bcbd22"],
        "ylabel": "deg"
    }
}


def interpolate_cycle(angle_data: np.ndarray, num_frames: int = NUM_RESAMPLE_FRAMES) -> np.ndarray:
    """Interpolate angle data to fixed number of frames."""
    original_x = np.linspace(0, 1, len(angle_data))
    target_x = np.linspace(0, 1, num_frames)
    return np.interp(target_x, original_x, angle_data)


def compute_aggregate(cycles: dict, prefix: str, keys: list) -> dict:
    """Compute mean and std across all cycles after resampling.

    Returns dict with:
        - mean: {key: array of shape (NUM_RESAMPLE_FRAMES,)}
        - std: {key: array of shape (NUM_RESAMPLE_FRAMES,)}
    """
    result = {'mean': {}, 'std': {}}

    for key in keys:
        # Collect all resampled cycles for this key
        resampled = []
        for cycle_idx, cycle_data in cycles.items():
            if prefix in cycle_data and key in cycle_data[prefix]:
                values = cycle_data[prefix][key]
                if len(values) > 1:
                    resampled.append(interpolate_cycle(values))

        if resampled:
            stacked = np.stack(resampled, axis=0)  # (num_cycles, NUM_RESAMPLE_FRAMES)
            result['mean'][key] = np.mean(stacked, axis=0)
            result['std'][key] = np.std(stacked, axis=0)

    return result


def render(data: dict, cfg: dict) -> None:
    """Render kinematics vs normalized gait cycle (0-100%).

    Args:
        data: Dict from load_sampled_hdf5() with 'cycles' and optional 'averaged'
        cfg: View configuration dict
    """
    # Check if this is sampled data
    if data.get('source') != 'sampled':
        st.warning("This view requires sampled rollout data")
        return

    st.header("Kinematics vs Gait Cycle")

    cycles = data.get('cycles', {})
    averaged = data.get('averaged')

    if not cycles:
        st.warning("No cycle data found")
        return

    # Category selection
    available_categories = []
    first_cycle = cycles[list(cycles.keys())[0]]
    for cat_name, cat_cfg in METRIC_CATEGORIES.items():
        prefix = cat_cfg["prefix"]
        if prefix in first_cycle:
            # Check if at least one key exists
            if any(k in first_cycle[prefix] for k in cat_cfg["keys"]):
                available_categories.append(cat_name)

    if not available_categories:
        st.warning("No kinematics data found in the cycles")
        return

    category = st.selectbox("Metric Category", available_categories)
    cat_cfg = METRIC_CATEGORIES[category]
    prefix = cat_cfg["prefix"]
    keys = cat_cfg["keys"]
    labels = cat_cfg["labels"]
    colors = cat_cfg["colors"]
    ylabel = cat_cfg["ylabel"]

    # Filter to only available keys
    available_keys = []
    available_labels = []
    available_colors = []
    for k, l, c in zip(keys, labels, colors):
        if prefix in first_cycle and k in first_cycle[prefix]:
            available_keys.append(k)
            available_labels.append(l)
            available_colors.append(c)

    if not available_keys:
        st.warning(f"No {category} data found")
        return

    # Mode selection
    mode_options = ["Single Cycle", "Aggregate (Mean ± Std)"]
    if averaged is not None:
        mode_options.append("Pre-computed Average")

    mode = st.radio("Display Mode", mode_options, horizontal=True)

    # Cycle selection
    sorted_cycle_indices = sorted(cycles.keys())
    min_cycle = min(sorted_cycle_indices)
    max_cycle = max(sorted_cycle_indices)

    # Create figure with N subplots in a single row
    n_plots = len(available_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    # Normalized gait cycle (0-100%)
    gait_pct = np.linspace(0, 100, NUM_RESAMPLE_FRAMES)

    if mode == "Single Cycle":
        cycle_idx = st.slider(
            "Cycle Index",
            min_value=min_cycle,
            max_value=max_cycle,
            value=min_cycle,
            step=1
        )
        cycle_data = cycles.get(cycle_idx)
        if cycle_data is None:
            st.error(f"Cycle {cycle_idx} not found")
            return

        for ax, key, label, color in zip(axes, available_keys, available_labels, available_colors):
            if prefix not in cycle_data or key not in cycle_data[prefix]:
                ax.text(0.5, 0.5, f"No {label} data", ha='center', va='center',
                        transform=ax.transAxes)
            else:
                values = cycle_data[prefix][key]
                interpolated = interpolate_cycle(values)
                ax.plot(gait_pct, interpolated, color=color, linewidth=1.5)

            ax.set_ylabel(f'{label} ({ylabel})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

        title_suffix = f"Cycle {cycle_idx}"

    elif mode == "Aggregate (Mean ± Std)":
        agg = compute_aggregate(cycles, prefix, available_keys)

        for ax, key, label, color in zip(axes, available_keys, available_labels, available_colors):
            if key not in agg['mean']:
                ax.text(0.5, 0.5, f"No {label} data", ha='center', va='center',
                        transform=ax.transAxes)
            else:
                mean = agg['mean'][key]
                std = agg['std'][key]
                ax.plot(gait_pct, mean, color=color, linewidth=1.5, label='Mean')
                ax.fill_between(gait_pct, mean - std, mean + std,
                               color=color, alpha=0.3, label='±1 Std')

            ax.set_ylabel(f'{label} ({ylabel})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

        title_suffix = f"Aggregate ({len(cycles)} cycles)"

    else:  # Pre-computed Average
        for ax, key, label, color in zip(axes, available_keys, available_labels, available_colors):
            if prefix not in averaged or key not in averaged[prefix]:
                ax.text(0.5, 0.5, f"No {label} data", ha='center', va='center',
                        transform=ax.transAxes)
            else:
                values = averaged[prefix][key]
                if averaged.get('phase') is not None and len(averaged['phase']) == len(values):
                    x = averaged['phase'] * 100
                else:
                    x = np.linspace(0, 100, len(values))
                ax.plot(x, values, color=color, linewidth=1.5)

            ax.set_ylabel(f'{label} ({ylabel})')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

        title_suffix = "Pre-computed Average"

    for ax in axes:
        ax.set_xlabel('Gait Cycle (%)')

    # Title
    title = f"{category} - {data['dir_name']} ({title_suffix})"
    fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Show stats
    with st.expander("Data Info"):
        st.write(f"**Directory:** {data['dir_name']}")
        st.write(f"**Category:** {category}")
        st.write(f"**Metrics:** {', '.join(available_labels)}")
        st.write(f"**Number of cycles:** {len(cycles)}")
        st.write(f"**Resample frames:** {NUM_RESAMPLE_FRAMES}")
        st.write(f"**Pre-computed average available:** {'Yes' if averaged else 'No'}")

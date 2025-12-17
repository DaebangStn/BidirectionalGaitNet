"""Marker error heatmap visualization."""
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
    # Character selection - order: both, free, motion
    char_type = st.radio(
        "Character Type",
        ["both", "free", "motion"],
        format_func=lambda x: {
            "both": "Both (Side by Side)",
            "free": "Free Character",
            "motion": "Motion Character"
        }[x],
        horizontal=True
    )

    # Compute shared max across all data for colorbar range
    if all_data:
        data_list = [d for d in all_data if d is not None]
    else:
        data_list = [data_sample]

    shared_max = 0.0
    for d in data_list:
        if char_type == "both":
            free_key = 'marker_error_free_data'
            motion_key = 'marker_error_motion_data'
            if free_key in d:
                shared_max = max(shared_max, float(np.nanmax(d[free_key])))
            if motion_key in d:
                shared_max = max(shared_max, float(np.nanmax(d[motion_key])))
        else:
            key = f'marker_error_{char_type}_data'
            if key in d:
                shared_max = max(shared_max, float(np.nanmax(d[key])))

    if shared_max == 0.0:
        shared_max = 1.0  # Fallback

    # Initialize session state for color bar
    if "me_cbar_vmin" not in st.session_state:
        st.session_state.me_cbar_vmin = 0.0
    if "me_cbar_vmax" not in st.session_state:
        st.session_state.me_cbar_vmax = shared_max
    if "me_cbar_slider" not in st.session_state:
        st.session_state.me_cbar_slider = (0.0, shared_max)

    def sync_cbar_from_slider():
        slider_val = st.session_state.me_cbar_slider
        st.session_state.me_cbar_vmin = slider_val[0]
        st.session_state.me_cbar_vmax = slider_val[1]

    def sync_cbar_from_inputs():
        st.session_state.me_cbar_slider = (
            st.session_state.me_cbar_vmin,
            st.session_state.me_cbar_vmax
        )

    # Colormap range control with slider and float inputs
    cbar_col1, cbar_col2, cbar_col3 = st.columns([1, 3, 1])
    with cbar_col1:
        st.number_input("Min (mm)", min_value=0.0, max_value=shared_max,
                        step=1.0, format="%.1f", key="me_cbar_vmin",
                        on_change=sync_cbar_from_inputs)
    with cbar_col2:
        st.slider(
            "Color Range",
            min_value=0.0,
            max_value=shared_max,
            step=shared_max / 100 if shared_max > 0 else 1.0,
            key="me_cbar_slider",
            label_visibility="collapsed",
            on_change=sync_cbar_from_slider
        )
    with cbar_col3:
        st.number_input("Max (mm)", min_value=0.0, max_value=shared_max,
                        step=1.0, format="%.1f", key="me_cbar_vmax",
                        on_change=sync_cbar_from_inputs)

    vmin = st.session_state.me_cbar_vmin
    vmax = st.session_state.me_cbar_vmax

    # Display options
    show_info = st.checkbox("Show panel info", value=True)

    return {
        'char_type': char_type,
        'vmin': vmin,
        'vmax': vmax,
        'shared_max': shared_max,
        'show_info': show_info,
    }


def get_summary(controls: dict) -> str:
    """Get a summary string of the current control values."""
    if controls is None:
        return ""
    char_labels = {
        "both": "Both",
        "free": "Free Character",
        "motion": "Motion Character"
    }
    parts = [
        f"**Type:** {char_labels[controls['char_type']]}",
        f"**Color:** [{controls['vmin']:.1f}, {controls['vmax']:.1f}] mm",
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
    vmin = controls['vmin']
    vmax = controls['vmax']
    show_info = controls.get('show_info', True)

    marker_names = data.get('marker_names', [])
    if not marker_names:
        st.warning("No marker names found")
        return

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

        if char_type == "both":
            free_key = 'marker_error_free_data'
            motion_key = 'marker_error_motion_data'
            if free_key not in data or motion_key not in data:
                st.warning("Both free and motion marker error data required")
                return
            error_data_free = data[free_key].T
            error_data_motion = data[motion_key].T

            # Create side-by-side plots with space for colorbar
            fig_width = 22 * plot_width
            fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(fig_width, 8),
                                                 gridspec_kw={'width_ratios': [1, 1, 0.05]})

            # Free character plot
            im1 = ax1.imshow(
                error_data_free,
                aspect='auto',
                cmap='hot',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest'
            )
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Marker')
            ax1.set_title('Free Character')
            ax1.set_yticks(range(len(marker_names)))
            ax1.set_yticklabels(marker_names, fontsize=8)

            # Motion character plot
            im2 = ax2.imshow(
                error_data_motion,
                aspect='auto',
                cmap='hot',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest'
            )
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Marker')
            ax2.set_title('Motion Character')
            ax2.set_yticks(range(len(marker_names)))
            ax2.set_yticklabels(marker_names, fontsize=8)

            # Shared colorbar
            fig.colorbar(im2, cax=cax, label='Error (mm)')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Statistics
            if show_info:
                mean_free = np.nanmean(error_data_free)
                mean_motion = np.nanmean(error_data_motion)
                st.caption(f"Mean: Free={mean_free:.2f}mm | Motion={mean_motion:.2f}mm")

        else:
            key = f'marker_error_{char_type}_data'
            if key not in data:
                st.warning(f"No {char_type} marker error data available")
                return
            error_matrix = data[key].T

            fig_width = 14 * plot_width
            fig, ax = plt.subplots(figsize=(fig_width, 8))

            im = ax.imshow(
                error_matrix,
                aspect='auto',
                cmap='hot',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest'
            )

            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Marker')
            ax.set_yticks(range(len(marker_names)))
            ax.set_yticklabels(marker_names, fontsize=8)

            fig.colorbar(im, ax=ax, label='Error (mm)')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Statistics
            if show_info:
                mean_val = np.nanmean(error_matrix)
                std_val = np.nanstd(error_matrix)
                st.caption(f"Mean: {mean_val:.2f}mm | Std: {std_val:.2f}mm")


def render(data: dict, cfg: dict) -> None:
    """Render marker error heatmap (y: markers, x: frame).

    Legacy single-view interface.
    """
    st.header("Marker Error Heatmap")

    controls = render_controls(data, cfg)
    if controls is None:
        return

    render_plot(data, cfg, controls)

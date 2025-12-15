"""Marker error heatmap visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render(data: dict, cfg: dict) -> None:
    """Render marker error heatmap (y: markers, x: frame)."""
    st.header("Marker Error Heatmap")

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

    marker_names = data['marker_names']

    # Determine which data to use
    if char_type == "both":
        free_key = 'marker_error_free_data'
        motion_key = 'marker_error_motion_data'
        if free_key not in data or motion_key not in data:
            st.warning("Both free and motion marker error data required for 'Both' view")
            return
        error_data_free = data[free_key].T  # (numMarkers x numFrames)
        error_data_motion = data[motion_key].T

        # Compute shared max for colorbar range
        shared_max = max(float(np.nanmax(error_data_free)), float(np.nanmax(error_data_motion)))
    else:
        key = f'marker_error_{char_type}_data'
        if key not in data:
            st.warning(f"No {char_type} marker error data available")
            return
        error_matrix = data[key].T  # (numMarkers x numFrames)
        shared_max = float(np.nanmax(error_matrix))

    # Colormap range control with slider and float inputs
    cbar_col1, cbar_col2, cbar_col3 = st.columns([1, 3, 1])
    with cbar_col1:
        vmin = st.number_input("Min (mm)", min_value=0.0, max_value=shared_max, value=0.0, step=1.0, format="%.1f")
    with cbar_col2:
        vmin_slider, vmax_slider = st.slider(
            "Color Range",
            min_value=0.0,
            max_value=shared_max,
            value=(vmin, shared_max),
            step=shared_max / 100 if shared_max > 0 else 1.0,
            label_visibility="collapsed"
        )
        if vmin_slider != vmin:
            vmin = vmin_slider
        vmax = vmax_slider
    with cbar_col3:
        vmax = st.number_input("Max (mm)", min_value=0.0, max_value=shared_max, value=vmax, step=1.0, format="%.1f")

    if char_type == "both":
        # Create side-by-side plots with space for colorbar
        fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(22, 8),
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
        ax1.set_title(f'Free Character - PID: {data["pid"]} ({data["pre_post"]})')
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
        ax2.set_title(f'Motion Character - PID: {data["pid"]} ({data["pre_post"]})')
        ax2.set_yticks(range(len(marker_names)))
        ax2.set_yticklabels(marker_names, fontsize=8)

        # Shared colorbar in dedicated axis
        fig.colorbar(im2, cax=cax, label='Error (mm)')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary statistics for both
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Free Character**")
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Mean Error", f"{np.nanmean(error_data_free):.2f} mm")
            with subcol2:
                st.metric("Max Error", f"{np.nanmax(error_data_free):.2f} mm")
            with subcol3:
                st.metric("Std Error", f"{np.nanstd(error_data_free):.2f} mm")
        with col2:
            st.markdown("**Motion Character**")
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Mean Error", f"{np.nanmean(error_data_motion):.2f} mm")
            with subcol2:
                st.metric("Max Error", f"{np.nanmax(error_data_motion):.2f} mm")
            with subcol3:
                st.metric("Std Error", f"{np.nanstd(error_data_motion):.2f} mm")

    else:
        # Single plot (free or motion)
        fig, ax = plt.subplots(figsize=(14, 8))

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
        ax.set_title(f'Marker Tracking Error ({char_type.title()} Character) - PID: {data["pid"]} ({data["pre_post"]})')
        ax.set_yticks(range(len(marker_names)))
        ax.set_yticklabels(marker_names, fontsize=8)

        fig.colorbar(im, ax=ax, label='Error (mm)')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Error", f"{np.nanmean(error_matrix):.2f} mm")
        with col2:
            st.metric("Max Error", f"{np.nanmax(error_matrix):.2f} mm")
        with col3:
            st.metric("Std Error", f"{np.nanstd(error_matrix):.2f} mm")

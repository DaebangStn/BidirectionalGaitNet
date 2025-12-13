"""Marker error heatmap visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render(data: dict, cfg: dict) -> None:
    """Render marker error heatmap (y: markers, x: frame)."""
    st.header("Marker Error Heatmap")

    # Character selection
    char_type = st.selectbox(
        "Character Type",
        ["free", "motion"],
        format_func=lambda x: "Free Character" if x == "free" else "Motion Character"
    )

    key = f'marker_error_{char_type}_data'
    if key not in data:
        st.warning(f"No {char_type} marker error data available")
        return

    error_data = data[key]  # (numFrames x numMarkers)
    marker_names = data['marker_names']

    # Transpose for heatmap (markers on y-axis, frames on x-axis)
    error_matrix = error_data.T  # (numMarkers x numFrames)

    # Colormap range control
    col1, col2 = st.columns(2)
    with col1:
        vmin = st.number_input("Min Error (mm)", value=0.0, step=1.0)
    with col2:
        vmax = st.number_input("Max Error (mm)", value=float(np.nanmax(error_matrix)), step=1.0)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        error_matrix,
        aspect='auto',
        cmap='hot',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    # Labels
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Marker')
    ax.set_title(f'Marker Tracking Error ({char_type.title()} Character) - PID: {data["pid"]} ({data["pre_post"]})')

    # Y-axis: marker names
    ax.set_yticks(range(len(marker_names)))
    ax.set_yticklabels(marker_names, fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Error (mm)')

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

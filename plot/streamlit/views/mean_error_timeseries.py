"""Mean error time series visualization."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render(data: dict, cfg: dict) -> None:
    """Render mean error over time."""
    st.header("Mean Error Time Series")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot both characters if available
    for char_type, color, label in [
        ('free', 'blue', 'Free Character'),
        ('motion', 'red', 'Motion Character')
    ]:
        key = f'marker_error_{char_type}_mean'
        if key in data:
            mean_error = data[key]
            frames = np.arange(len(mean_error))
            ax.plot(frames, mean_error, color=color, label=label, linewidth=1.5)

    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Mean Error (mm)')
    ax.set_title(f'Mean Marker Error Over Time - PID: {data["pid"]} ({data["pre_post"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

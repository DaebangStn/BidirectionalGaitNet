#!/usr/bin/env python3
"""Pre/Post patient comparison plot for all PIDs.

Shows pre vs post measurements with each patient in a unique color.
Multiple subplots for different metrics.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/home/geon/BidirectionalGaitNet')

from rm import rm_mgr
from utils import set_plot, style_axis

# =============================================================================
# CONFIGURATION - Edit these to select which metrics to plot
# =============================================================================
GMFCS = 3  # Set to 1, 2, or 3 to filter by GMFCS level; None = no filter

METRICS = [
    ('height', 'Height (cm)', lambda d: d.get('height')),
    ('weight', 'Weight (kg)', lambda d: d.get('weight')),
    # ('foot_left_length', 'Foot Left Length (cm)', lambda d: d.get('foot', {}).get('left', {}).get('length')),
    # ('foot_left_width', 'Foot Left Width (cm)', lambda d: d.get('foot', {}).get('left', {}).get('width')),
    # ('foot_right_length', 'Foot Right Length (cm)', lambda d: d.get('foot', {}).get('right', {}).get('length')),
    # ('foot_right_width', 'Foot Right Width (cm)', lambda d: d.get('foot', {}).get('right', {}).get('width')),
]

# Color map for patients
COLORMAP = 'tab20'  # Good for distinguishing many categories

# Layout configuration
N_ROWS = 1


def get_metric_value(data, extractor):
    """Safely extract metric value from data dict."""
    try:
        val = extractor(data)
        return float(val) if val is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def main():
    # 1. List all PIDs
    pids = rm_mgr.list("@pid")
    print(f"Found {len(pids)} patients: {pids}")

    # 2. Collect pre/post data and metadata for each PID
    pre_data_all = {}
    post_data_all = {}
    gmfcs_all = {}
    for pid in pids:
        try:
            pre_data_all[pid] = rm_mgr(f"@pid:{pid}/gait/pre")
        except Exception as e:
            print(f"Warning: Could not load pre data for {pid}: {e}")
            pre_data_all[pid] = {}
        try:
            post_data_all[pid] = rm_mgr(f"@pid:{pid}/gait/post")
        except Exception as e:
            print(f"Warning: Could not load post data for {pid}: {e}")
            post_data_all[pid] = {}
        try:
            gmfcs_all[pid] = rm_mgr(f"@pid:{pid}/gmfcs")
        except Exception as e:
            print(f"Warning: Could not load gmfcs for {pid}: {e}")
            gmfcs_all[pid] = None

    # 3. Filter by GMFCS if specified
    if GMFCS is not None:
        pids = [pid for pid in pids if gmfcs_all.get(pid) == GMFCS]
        print(f"Filtered to {len(pids)} patients with GMFCS={GMFCS}: {pids}")

    # 4. Setup colors for each PID
    n_pids = len(pids)
    cmap = plt.colormaps[COLORMAP]
    colors = {pid: cmap(i / max(n_pids, 1)) for i, pid in enumerate(pids)}

    # 4. Create subplots
    n_metrics = len(METRICS)
    n_rows = N_ROWS
    n_cols = (n_metrics + n_rows - 1) // n_rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs = np.atleast_2d(axs).flatten()

    # X positions for pre and post
    x_pre, x_post = 0, 1

    # 5. Plot each metric in its own subplot
    for metric_idx, (metric_key, metric_label, extractor) in enumerate(METRICS):
        ax = axs[metric_idx]

        pre_values = []
        post_values = []

        ratios = []  # Store ratios for mean calculation
        for pid in pids:
            pre_val = get_metric_value(pre_data_all[pid], extractor)
            post_val = get_metric_value(post_data_all[pid], extractor)

            if pre_val is not None and post_val is not None and pre_val != 0:
                pre_values.append(pre_val)
                post_values.append(post_val)
                color = colors[pid]

                # Calculate increase ratio (percentage)
                ratio = (post_val - pre_val) / pre_val * 100
                ratios.append(ratio)

                # Plot line connecting pre to post
                ax.plot([x_pre, x_post], [pre_val, post_val],
                        '-', color=color, linewidth=1.5, alpha=0.7)
                # Plot pre point (circle)
                ax.plot(x_pre, pre_val, 'o', color=color, markersize=8, label=pid)
                # Plot post point (circle)
                ax.plot(x_post, post_val, 'o', color=color, markersize=8)

                # Add ratio text next to the post value
                sign = '+' if ratio >= 0 else ''
                ax.text(x_post + 0.05, post_val, f'{sign}{ratio:.1f}%',
                        fontsize=8, color=color, va='center', ha='left')

        # Plot mean lines and mean ratio
        if pre_values:
            mean_pre = np.mean(pre_values)
            mean_post = np.mean(post_values)
            mean_ratio = np.mean(ratios)
            ax.axhline(mean_pre, color='blue', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Mean Pre: {mean_pre:.1f}')
            ax.axhline(mean_post, color='red', linestyle='--', linewidth=2,
                       alpha=0.8, label=f'Mean Post: {mean_post:.1f}')

            # Add mean ratio text
            sign = '+' if mean_ratio >= 0 else ''
            ax.text(0.5, mean_post + (mean_post - mean_pre) * 0.1,
                    f'Mean: {sign}{mean_ratio:.1f}%',
                    fontsize=10, fontweight='bold', color='black',
                    va='bottom', ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Style axis
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([x_pre, x_post])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_ylabel(metric_label)
        ax.set_title(metric_key)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axs)):
        axs[idx].set_visible(False)

    # Add legend (PIDs with GMFCS, in separate legend box)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[pid], markersize=8,
                          label=f"{pid} ({gmfcs_all[pid]})" if gmfcs_all[pid] else pid)
               for pid in pids]
    # Add mean legend entries
    handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Mean Pre'))
    handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean Post'))

    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.15, 0.5),
               title='Patient ID')

    plt.tight_layout()
    set_plot(plt, screen_idx=1, fullscreen=False, legend=False)
    plt.show()


if __name__ == '__main__':
    main()

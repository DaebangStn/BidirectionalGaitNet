#!/usr/bin/env python3
"""
Plotting utilities for BidirectionalGaitNet visualizations.

Provides HDF5 data extraction and interactive plotting setup.
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False


KEY_CMD_MAP = {
    '1': 'xy',
    '2': 'yz',
    '3': 'zx'
}


def extract_hdf5_data(hdf5_path: str, in_lbls: List[str], out_lbls: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract parameter states and output attributes from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 rollout data file
        in_lbls: List of input parameter names (e.g., ['gait_cadence', 'gait_stride'])
        out_lbls: List of output attribute names (e.g., ['metabolic/cot/MA/mean'])

    Returns:
        (inputs, outputs) tuple:
            - inputs: np.ndarray of shape (N, len(in_lbls)) with parameter values
            - outputs: np.ndarray of shape (N, len(out_lbls)) with attribute values
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Get parameter names
        if 'parameter_names' in f:
            param_names_data = f['parameter_names'][()]
        elif 'metadata/parameter_names' in f:
            param_names_data = f['metadata/parameter_names'][()]
        else:
            raise KeyError(f"Could not find parameter_names in {hdf5_path}")

        # Handle both string and bytes
        if isinstance(param_names_data, np.ndarray):
            param_names = [s.decode('utf-8') if isinstance(s, bytes) else s for s in param_names_data]
        elif isinstance(param_names_data, bytes):
            param_names = param_names_data.decode('utf-8').split(',')
        else:
            param_names = param_names_data.split(',')

        # Get indices for input labels
        in_indices = []
        for lbl in in_lbls:
            if lbl not in param_names:
                raise ValueError(f"Input label '{lbl}' not found in parameter_names: {param_names}")
            in_indices.append(param_names.index(lbl))

        # Get all param groups
        param_keys = sorted([k for k in f.keys() if k.startswith('param_')],
                          key=lambda x: int(float(x.split('_')[1])))

        # Collect data
        inputs_list = []
        outputs_list = []

        for param_key in param_keys:
            param_group = f[param_key]
            param_state = param_group['param_state'][:]

            # Extract input parameters
            input_vals = [param_state[i] for i in in_indices]
            inputs_list.append(input_vals)

            # Extract output attributes
            output_vals = []
            for out_lbl in out_lbls:
                if out_lbl in param_group.attrs:
                    output_vals.append(param_group.attrs[out_lbl])
                else:
                    raise KeyError(f"Output attribute '{out_lbl}' not found in {param_key} attrs")
            outputs_list.append(output_vals)

        inputs = np.array(inputs_list)
        outputs = np.array(outputs_list)

    return inputs, outputs


def get_parameter_ranges(hdf5_path: str, param_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Get min/max ranges for specified parameters from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 rollout data file
        param_names: List of parameter names to get ranges for

    Returns:
        Dictionary mapping parameter name to (min, max) tuple
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Get all parameter names
        if 'parameter_names' in f:
            all_param_names_data = f['parameter_names'][()]
        elif 'metadata/parameter_names' in f:
            all_param_names_data = f['metadata/parameter_names'][()]
        else:
            raise KeyError(f"Could not find parameter_names in {hdf5_path}")

        # Handle both string and bytes
        if isinstance(all_param_names_data, np.ndarray):
            all_param_names = [s.decode('utf-8') if isinstance(s, bytes) else s for s in all_param_names_data]
        elif isinstance(all_param_names_data, bytes):
            all_param_names = all_param_names_data.decode('utf-8').split(',')
        else:
            all_param_names = all_param_names_data.split(',')

        # Get indices for requested parameters
        param_indices = []
        for name in param_names:
            if name not in all_param_names:
                raise ValueError(f"Parameter '{name}' not found in parameter_names: {all_param_names}")
            param_indices.append(all_param_names.index(name))

        # Get all param groups
        param_keys = sorted([k for k in f.keys() if k.startswith('param_')],
                          key=lambda x: int(float(x.split('_')[1])))

        # Collect values for each parameter
        all_values = {name: [] for name in param_names}

        for param_key in param_keys:
            param_state = f[param_key]['param_state'][:]
            for i, name in zip(param_indices, param_names):
                all_values[name].append(param_state[i])

        # Compute ranges
        ranges = {}
        for name in param_names:
            values = np.array(all_values[name])
            ranges[name] = (float(values.min()), float(values.max()))

    return ranges


def on_key(event):
    """Handle keyboard events for plot interaction."""
    fig = plt.gcf()
    axs = fig.get_axes()

    if event.key == 'escape':
        plt.close()
        sys.exit()
    elif event.key == ' ':
        plt_to_clipboard()
    elif event.key == 'm':
        for ax in axs:
            if isinstance(ax, Axes3D):
                ax.view_init(elev=ax.elev, azim=ax.azim + 180)

    cmd = KEY_CMD_MAP.get(event.key)
    if cmd in ['xy', 'yz', 'zx']:
        for ax in axs:
            if isinstance(ax, Axes3D):
                change_ax_view(ax, cmd)
        fig.canvas.draw()


def on_click(event):
    """Handle mouse click events for plot interaction."""
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        print("Clicked outside any Axes")
        return

    ax = event.inaxes
    x, y = event.xdata, event.ydata

    try:
        fig = plt.gcf()
        ax_idx = fig.axes.index(ax)
    except ValueError:
        ax_idx = "?"
    ax_label = ax.get_label() or f"ax[{ax_idx}]"

    print(f"Clicked {ax_label}: x={x:.3f}, y={y:.3f}")


def change_ax_view(ax: Axes, view_init: str):
    """Change 3D axes view to predefined orientations."""
    if view_init == 'xy':
        ax.view_init(elev=90, azim=270)
    elif view_init == 'yz':
        ax.view_init(elev=0, azim=90)
    elif view_init == 'zx':
        ax.view_init(elev=0, azim=0)


def plt_to_clipboard():
    """Copy current plot to clipboard as image."""
    if not PYPERCLIP_AVAILABLE:
        print("pyperclip not available - cannot copy to clipboard")
        return

    try:
        import io
        from PIL import Image

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)

        # Copy to clipboard
        output = io.BytesIO()
        img.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:]  # Remove BMP header
        output.close()

        pyperclip.copy(data)
        print("Plot copied to clipboard")
    except ImportError:
        print("PIL not available - cannot copy to clipboard")
    except Exception as e:
        print(f"Failed to copy to clipboard: {e}")


def set_plot(plt, fullscreen: bool = False, legend: bool = False):
    """
    Setup plot with event handlers and display options.

    Args:
        plt: matplotlib.pyplot module
        fullscreen: Whether to display in fullscreen mode
        legend: Whether to add legends to axes with labeled artists

    Keyboard shortcuts:
        - escape: Close plot and exit
        - space: Copy plot to clipboard
        - m: Mirror view (rotate 180°)
        - 1: XY view (top-down)
        - 2: YZ view (side)
        - 3: ZX view (front)
    """
    fig = plt.gcf()

    # Add legends if requested
    if legend:
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0 and len(labels) > 0:
                ax.legend()

    # Connect event handlers
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Set fullscreen if requested
    if fullscreen:
        mng = plt.get_current_fig_manager()
        try:
            mng.full_screen_toggle()
        except AttributeError:
            print("Fullscreen not supported on this backend")

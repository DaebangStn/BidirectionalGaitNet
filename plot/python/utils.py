"""
Matplotlib plotting utilities for BidirectionalGaitNet visualization.

Usage:
    import matplotlib
    matplotlib.use('TkAgg')  # MUST be before importing pyplot
    import matplotlib.pyplot as plt
    from utils import set_plot, plt_to_clipboard

    # Create your plot
    plt.plot(x, y, label='data')

    # Set up interactive features
    set_plot(plt, screen_idx=0, fullscreen=False, legend=True)
    plt.show()

Key bindings:
    ESC   - Close plot and exit
    SPACE - Copy plot to clipboard
"""

import sys
import subprocess
from io import BytesIO
from screeninfo import get_monitors


def plt_to_clipboard(plt):
    """Copy current matplotlib figure to system clipboard as PNG.

    Requires xclip to be installed: sudo apt install xclip
    """
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    buffer.seek(0)
    subprocess.run(
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-i"],
        input=buffer.read(),
        check=True
    )
    print("Plot copied to clipboard")


def on_key(event, plt):
    """Handle keyboard events for interactive plot control.

    ESC   - Close plot and exit
    SPACE - Copy plot to clipboard
    """
    if event.key == 'escape':
        plt.close()
        sys.exit()
    elif event.key == ' ':
        plt_to_clipboard(plt)


def set_plot(plt, screen_idx=0, fullscreen=False, legend=False):
    """Configure matplotlib figure with interactive features and window positioning.

    Args:
        plt: matplotlib.pyplot module
        screen_idx: Monitor index for window placement (default: 0)
        fullscreen: Toggle fullscreen mode (default: False)
        legend: Auto-add legends to axes with labeled artists (default: False)

    Example:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from utils import set_plot

        plt.plot([1,2,3], [1,4,9], label='quadratic')
        set_plot(plt, legend=True)
        plt.show()
    """
    fig = plt.gcf()

    # Add legends to axes that have labeled artists
    if legend:
        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0 and len(labels) > 0:
                ax.legend()

    # Connect key event handler with closure to capture plt
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, plt))

    # Set window position or fullscreen
    if fullscreen:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
    else:
        monitors = get_monitors()
        if screen_idx < len(monitors):
            monitor = monitors[screen_idx]
            mng = plt.get_current_fig_manager()
            mng.window.wm_geometry(f"+{monitor.x}+{monitor.y}")


def create_figure(plt, figsize=(10, 6), dpi=100):
    """Create a new figure with common defaults.

    Args:
        plt: matplotlib.pyplot module
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch

    Returns:
        tuple: (fig, ax) or (fig, axs) for subplots
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def style_axis(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """Apply common styling to an axis.

    Args:
        ax: matplotlib axis object
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Show grid (default: True)
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)

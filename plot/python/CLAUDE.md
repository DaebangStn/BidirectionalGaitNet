# Plot Python Scripts

Matplotlib plotting utilities for BidirectionalGaitNet visualization.

## Required Setup

**CRITICAL**: All plot scripts MUST use TkAgg backend before importing pyplot:

```python
import matplotlib
matplotlib.use('TkAgg')  # MUST be first, before pyplot import
import matplotlib.pyplot as plt
from utils import set_plot
```

## Template for New Plot Scripts

```python
#!/usr/bin/env python3
"""Description of what this plot visualizes."""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from utils import set_plot, style_axis, create_figure

def main():
    # Create figure
    fig, ax = create_figure(plt, figsize=(10, 6))

    # Your plotting code here
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)')

    # Style and configure
    style_axis(ax, title='My Plot', xlabel='X', ylabel='Y', grid=True)

    # Enable interactive features
    set_plot(plt, screen_idx=0, fullscreen=False, legend=True)
    plt.show()

if __name__ == '__main__':
    main()
```

## Keyboard Shortcuts

| Key   | Action                    |
|-------|---------------------------|
| ESC   | Close plot and exit       |
| SPACE | Copy plot to clipboard    |

## utils.py Functions

### `set_plot(plt, screen_idx=0, fullscreen=False, legend=False)`
Configure figure with interactive features and window positioning.
- `screen_idx`: Monitor index (0 = primary)
- `fullscreen`: Toggle fullscreen mode
- `legend`: Auto-add legends to labeled axes

### `plt_to_clipboard(plt)`
Copy current figure to system clipboard as PNG.
Requires: `sudo apt install xclip`

### `create_figure(plt, figsize=(10, 6), dpi=100)`
Create new figure with defaults. Returns `(fig, ax)`.

### `style_axis(ax, title=None, xlabel=None, ylabel=None, grid=True)`
Apply common styling to an axis.

## Dependencies

```bash
# Required packages
pip install matplotlib screeninfo

# For clipboard support
sudo apt install xclip
```

## Running Scripts

```bash
# From project root
python plot/python/your_script.py

# Or with micromamba environment
/opt/miniconda3/bin/micromamba run -n bidir python plot/python/your_script.py
```

## pre_post_comparison.py

Pre/Post patient comparison plot showing measurements before and after for all PIDs.

### Configuration (at top of file)

```python
# Select metrics to plot - comment/uncomment as needed
METRICS = [
    ('height', 'Height (cm)', lambda d: d.get('height')),
    ('weight', 'Weight (kg)', lambda d: d.get('weight')),
    ('foot_left_length', 'Foot Left Length (cm)', lambda d: d.get('foot', {}).get('left', {}).get('length')),
    ('foot_left_width', 'Foot Left Width (cm)', lambda d: d.get('foot', {}).get('left', {}).get('width')),
    ('foot_right_length', 'Foot Right Length (cm)', lambda d: d.get('foot', {}).get('right', {}).get('length')),
    ('foot_right_width', 'Foot Right Width (cm)', lambda d: d.get('foot', {}).get('right', {}).get('width')),
]

# Color map for patients
COLORMAP = 'tab20'

# Layout configuration
N_ROWS = 2  # Number of rows in subplot grid
```

### Features
- Each patient shown in unique color
- Pre/Post on X-axis, connected by line
- Percentage change shown next to each post point
- Mean lines (blue=pre, red=post) with mean ratio displayed

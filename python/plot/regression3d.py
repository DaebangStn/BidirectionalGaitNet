#!/usr/bin/env python3
"""
3D visualization of regression model predictions vs actual data.

Plots input parameters (X, Y) against output predictions (Z) with both
actual data points (scatter) and model predictions (surface).
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.learn.network import RegressionNet
from python.plot.util import extract_hdf5_data, get_parameter_ranges, set_plot


def find_latest_checkpoint(data_path: str) -> str:
    """
    Find the latest checkpoint in the data directory.

    Args:
        data_path: Path to sample directory or HDF5 file

    Returns:
        Path to the latest checkpoint file

    Raises:
        FileNotFoundError: If no checkpoints found
    """
    # Resolve to directory
    if data_path.endswith('.h5'):
        data_dir = Path(data_path).parent
    else:
        data_dir = Path(data_path)

    # Find all version directories (v01, v02, etc.)
    version_dirs = sorted([d for d in data_dir.glob('v*') if d.is_dir()])

    if not version_dirs:
        raise FileNotFoundError(f"No version directories (v*) found in {data_dir}")

    # Use the latest version directory
    latest_version = version_dirs[-1]

    # Find all checkpoints in the latest version
    checkpoint_dir = latest_version / 'checkpoints'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {latest_version}")

    # Find all .ckpt files, sort by epoch number
    checkpoints = sorted(checkpoint_dir.glob('ep_*.ckpt'),
                        key=lambda p: int(p.stem.split('_')[1]))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Return the latest checkpoint (highest epoch number)
    latest_ckpt = checkpoints[-1]
    print(f"[Auto-detected] Using latest checkpoint: {latest_ckpt}")

    return str(latest_ckpt)


def load_checkpoint(ckpt_path: str):
    """
    Load model and configuration from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file

    Returns:
        (model, config) tuple where config includes data section
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract configuration
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'config' key: {ckpt_path}")
    config = checkpoint['config']

    # Reconstruct model
    model = RegressionNet(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        layers=config['layers'],
        input_mean=torch.tensor(config['input_mean'], device=device),
        input_std=torch.tensor(config['input_std'], device=device),
        target_mean=torch.tensor(config['target_mean'], device=device),
        target_std=torch.tensor(config['target_std'], device=device),
        residual=config['residual']
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load full config from config.yaml
    ckpt_dir = Path(ckpt_path).parent.parent
    config_path = ckpt_dir / 'config.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
    else:
        # Fallback: construct minimal config
        raise FileNotFoundError(f"Config file not found at {config_path}")

    return model, full_config, device


def resolve_data_path(path: str) -> str:
    """
    Resolve data path to rollout_data.h5 file.

    If path is a directory (sample directory), append '/rollout_data.h5'.
    If path is already a .h5 file, use it directly.
    """
    path_obj = Path(path)

    if path.endswith('.h5'):
        resolved_path = path
    else:
        resolved_path = str(path_obj / 'rollout_data.h5')

    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"HDF5 file not found at {resolved_path}")

    return resolved_path


def plot_regression3d(checkpoint_path: Optional[str], data_path: str,
                     grid_size: int = 50, fullscreen: bool = False,
                     legend: bool = False, use_optimizer: bool = False,
                     optimize_mode: str = 'min', optimizer_points: int = 30,
                     trial_size: int = 256, max_iter: int = 500,
                     constraint_weight: float = 1000.0):
    """
    Create 3D visualization of regression model predictions vs actual data.

    Args:
        checkpoint_path: Path to trained model checkpoint (None to auto-detect latest)
        data_path: Path to sample directory or HDF5 file
        grid_size: Resolution of prediction meshgrid (default: 50)
        fullscreen: Display in fullscreen mode
        legend: Show legend
    """
    # Auto-detect latest checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(data_path)

    print(f"[Regression3D] Loading checkpoint: {checkpoint_path}")
    model, config, device = load_checkpoint(checkpoint_path)

    # Get data configuration
    in_lbls = config['data']['in_lbl']
    out_lbls = config['data']['out_lbl']

    # Resolve data path
    hdf5_path = resolve_data_path(data_path)
    print(f"[Regression3D] Loading data: {hdf5_path}")

    # Extract actual data points
    inputs, outputs = extract_hdf5_data(hdf5_path, in_lbls, out_lbls)
    print(f"[Regression3D] Loaded {len(inputs)} data points")

    # Get parameter ranges
    param_ranges = get_parameter_ranges(hdf5_path, in_lbls)
    print(f"[Regression3D] Parameter ranges:")
    for lbl, (min_val, max_val) in param_ranges.items():
        print(f"  {lbl}: [{min_val:.4f}, {max_val:.4f}]")

    # Create meshgrid for predictions
    if len(in_lbls) != 2:
        raise ValueError(f"Expected 2 input parameters for 3D plot, got {len(in_lbls)}")

    x_lbl, y_lbl = in_lbls[0], in_lbls[1]
    x_min, x_max = param_ranges[x_lbl]
    y_min, y_max = param_ranges[y_lbl]

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Generate predictions on meshgrid
    print(f"[Regression3D] Generating predictions on {grid_size}x{grid_size} grid...")
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    with torch.no_grad():
        predictions = model(grid_tensor, normalize=True, denormalize=True)
        Z = predictions[:, 0].cpu().numpy().reshape(X.shape)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual data points (scatter)
    if len(out_lbls) != 1:
        raise ValueError(f"Expected 1 output parameter for 3D plot, got {len(out_lbls)}")

    scatter = ax.scatter(inputs[:, 0], inputs[:, 1], outputs[:, 0],
                        c='red', marker='o', s=30, alpha=0.6,
                        label='Actual Data')

    # Plot prediction surface
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7,
                             linewidth=0, antialiased=True,
                             label='MLP Predictions')

    # Set labels
    ax.set_xlabel(x_lbl, fontsize=12)
    ax.set_ylabel(y_lbl, fontsize=12)
    ax.set_zlabel(out_lbls[0], fontsize=12)
    ax.set_title(f'Regression Model: {out_lbls[0]} vs {x_lbl}, {y_lbl}', fontsize=14)

    # Run batched optimizer if requested
    if use_optimizer:
        print(f"\n[Optimizer] Running batched optimization...")
        from python.learn.batched_optimizer import BatchedOptimizer

        # Create batched optimizer
        opt = BatchedOptimizer(
            checkpoint_path, data_path,
            maximize=(optimize_mode == 'max')
        )

        # Compute product range from actual data
        products = inputs[:, 0] * inputs[:, 1]  # cadence × stride
        product_min = products.min()
        product_max = products.max()

        print(f"[Optimizer] Product range: [{product_min:.4f}, {product_max:.4f}]")

        # Generate linspace of product constraints
        product_constraints = np.linspace(product_min, product_max, optimizer_points)

        print(f"[Optimizer] Optimizing {optimizer_points} product constraints...")

        # Run batched optimization
        results = opt.run_with_product_constraint(
            opt_field=out_lbls[0],
            product_values=product_constraints,
            trial_size=trial_size,
            max_iter=max_iter,
            constraint_weight=constraint_weight,
            verbose=True
        )

        # Extract optimal points for plotting
        opt_cadences = np.array([r[x_lbl] for r in results])
        opt_strides = np.array([r[y_lbl] for r in results])
        opt_values = np.array([r[out_lbls[0]] for r in results])

        print(f"[Optimizer] Optimal value range: [{opt_values.min():.4f}, {opt_values.max():.4f}]")

        # Plot optimal curve on 3D surface
        color = 'gold' if optimize_mode == 'min' else 'dodgerblue'
        ax.plot(opt_cadences, opt_strides, opt_values,
               color=color, linewidth=3, marker='o',
               markersize=8, alpha=0.9,
               label=f'Optimizer ({optimize_mode})')

        # Also plot as scatter for emphasis
        ax.scatter(opt_cadences, opt_strides, opt_values,
                  c=color, s=100, marker='*', alpha=0.8,
                  edgecolors='black', linewidths=0.5)

    # Add legend manually (surface plot needs special handling)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Actual Data'),
        Patch(facecolor='green', alpha=0.7, label='MLP Predictions')
    ]
    if use_optimizer:
        opt_color = 'gold' if optimize_mode == 'min' else 'dodgerblue'
        legend_elements.append(
            Patch(facecolor=opt_color, alpha=0.8, label=f'Optimizer ({optimize_mode})')
        )
    if legend or use_optimizer:
        ax.legend(handles=legend_elements, loc='upper right')

    # Setup interactive controls
    set_plot(plt, fullscreen=fullscreen, legend=False)

    # Print statistics
    print(f"\n[Regression3D] Statistics:")
    print(f"  Actual data mean: {outputs[:, 0].mean():.4f} ± {outputs[:, 0].std():.4f}")
    print(f"  Prediction mean: {Z.mean():.4f} ± {Z.std():.4f}")
    print(f"  Min/Max actual: [{outputs[:, 0].min():.4f}, {outputs[:, 0].max():.4f}]")
    print(f"  Min/Max predicted: [{Z.min():.4f}, {Z.max():.4f}]")

    print("\n[Regression3D] Interactive controls:")
    print("  escape: Close plot and exit")
    print("  space: Copy plot to clipboard")
    print("  m: Mirror view (rotate 180°)")
    print("  1: XY view (top-down)")
    print("  2: YZ view (side)")
    print("  3: ZX view (front)")

    plt.show()


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='3D visualization of regression model predictions vs actual data'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to sample directory or HDF5 file')
    parser.add_argument('--grid-size', type=int, default=50,
                       help='Resolution of prediction meshgrid (default: 50)')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Display in fullscreen mode')
    parser.add_argument('--legend', action='store_true',
                       help='Show legend')

    args = parser.parse_args()

    plot_regression3d(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        grid_size=args.grid_size,
        fullscreen=args.fullscreen,
        legend=args.legend
    )


if __name__ == '__main__':
    main()

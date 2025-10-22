#!/usr/bin/env python3
"""
2D heatmap visualization of sampled parameters with optimizer results.

Shows input parameters (X, Y) with output values as heatmap colors,
overlaid with optimizer-selected optimal points.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.plot.util import extract_hdf5_data, set_plot
from python.plot.regression3d import find_latest_checkpoint, resolve_data_path


def plot_heatmap2d(
    checkpoint_path: Optional[str],
    data_path: str,
    fullscreen: bool = False,
    legend: bool = True,
    use_optimizer: bool = False,
    optimize_mode: str = 'min',
    optimizer_points: int = 30,
    trial_size: int = 256,
    max_iter: int = 500,
    constraint_weight: float = 1000.0
):
    """
    Create 2D heatmap visualization of sampled parameters with output values.

    Args:
        checkpoint_path: Path to trained model checkpoint (None to auto-detect)
        data_path: Path to sample directory or HDF5 file
        fullscreen: Display in fullscreen mode
        legend: Show colorbar legend
        use_optimizer: Run optimizer and overlay optimal points
        optimize_mode: 'min' or 'max' optimization mode
        optimizer_points: Number of product constraints for optimizer
        trial_size: Parallel trials per constraint
        max_iter: Adam iterations
        constraint_weight: Product constraint penalty weight
    """
    # Auto-detect latest checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(data_path)

    print(f"[Heatmap2D] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint to get config
    import torch
    import yaml
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'config' key: {checkpoint_path}")

    # Load full config
    ckpt_dir = Path(checkpoint_path).parent.parent
    config_path = ckpt_dir / 'config.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Get data configuration
    in_lbls = config['data']['in_lbl']
    out_lbls = config['data']['out_lbl']

    if len(in_lbls) != 2:
        raise ValueError(f"Heatmap requires exactly 2 input parameters, got {len(in_lbls)}")
    if len(out_lbls) != 1:
        raise ValueError(f"Heatmap requires exactly 1 output parameter, got {len(out_lbls)}")

    # Resolve data path
    hdf5_path = resolve_data_path(data_path)
    print(f"[Heatmap2D] Loading data: {hdf5_path}")

    # Extract actual data points
    inputs, outputs = extract_hdf5_data(hdf5_path, in_lbls, out_lbls)
    print(f"[Heatmap2D] Loaded {len(inputs)} data points")

    x_lbl, y_lbl = in_lbls[0], in_lbls[1]
    out_lbl = out_lbls[0]

    # Get parameter ranges from HDF5
    from python.plot.util import get_parameter_ranges
    param_ranges = get_parameter_ranges(hdf5_path, in_lbls)
    x_min, x_max = param_ranges[x_lbl]
    y_min, y_max = param_ranges[y_lbl]

    # Create meshgrid for continuous heatmap
    grid_size = 100
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Generate predictions on meshgrid
    print(f"[Heatmap2D] Generating predictions on {grid_size}x{grid_size} grid...")
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # Load model
    from python.learn.network import RegressionNet
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    model = RegressionNet(
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        layers=model_config['layers'],
        input_mean=torch.tensor(model_config['input_mean'], device=device),
        input_std=torch.tensor(model_config['input_std'], device=device),
        target_mean=torch.tensor(model_config['target_mean'], device=device),
        target_std=torch.tensor(model_config['target_std'], device=device),
        residual=model_config['residual']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(grid_tensor, normalize=True, denormalize=True)
        out_idx = out_lbls.index(out_lbl)
        Z = predictions[:, out_idx].cpu().numpy().reshape(X.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot continuous heatmap using contourf
    levels = 50
    heatmap = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)

    # Add colorbar
    if legend:
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label(out_lbl, fontsize=12)

    # Plot sampled data points as small dots
    ax.scatter(
        inputs[:, 0], inputs[:, 1],
        c='white',
        s=20,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
        label='Sampled',
        zorder=5
    )

    # Set labels and title
    ax.set_xlabel(x_lbl, fontsize=14)
    ax.set_ylabel(y_lbl, fontsize=14)
    ax.set_title(f'Parameter Heatmap: {out_lbl} vs {x_lbl}, {y_lbl}', fontsize=16)
    ax.grid(True, alpha=0.3)

    # Run optimizer if requested
    if use_optimizer:
        print(f"\n[Optimizer] Running batched optimization...")
        from python.learn.batched_optimizer import BatchedOptimizer

        # Create batched optimizer
        opt = BatchedOptimizer(
            checkpoint_path, data_path,
            maximize=(optimize_mode == 'max')
        )

        # Compute product range from actual data
        products = inputs[:, 0] * inputs[:, 1]  # param1 × param2
        product_min = products.min()
        product_max = products.max()

        print(f"[Optimizer] Product range: [{product_min:.4f}, {product_max:.4f}]")

        # Generate linspace of product constraints
        product_constraints = np.linspace(product_min, product_max, optimizer_points)

        print(f"[Optimizer] Optimizing {optimizer_points} product constraints...")

        # Run batched optimization
        results = opt.run_with_product_constraint(
            opt_field=out_lbl,
            product_values=product_constraints,
            trial_size=trial_size,
            max_iter=max_iter,
            constraint_weight=constraint_weight,
            verbose=True
        )

        # Extract optimal points
        opt_x = np.array([r[x_lbl] for r in results])
        opt_y = np.array([r[y_lbl] for r in results])
        opt_values = np.array([r[out_lbl] for r in results])

        print(f"[Optimizer] Optimal value range: [{opt_values.min():.4f}, {opt_values.max():.4f}]")

        # Plot optimizer selected points as small black dots
        ax.scatter(
            opt_x, opt_y,
            c='black',
            s=25,
            marker='o',
            alpha=0.9,
            edgecolors='white',
            linewidths=1,
            label=f'Selected ({optimize_mode})',
            zorder=10
        )

        # Connect with line to show trajectory
        color = 'gold' if optimize_mode == 'min' else 'dodgerblue'
        ax.plot(
            opt_x, opt_y,
            color=color,
            linewidth=1.5,
            alpha=0.6,
            zorder=9
        )

    # Add legend
    if use_optimizer:
        ax.legend(loc='best', fontsize=12)

    # Setup interactive controls
    set_plot(plt, fullscreen=fullscreen, legend=False)

    # Print statistics
    print(f"\n[Heatmap2D] Statistics:")
    print(f"  Output mean: {outputs[:, 0].mean():.4f} ± {outputs[:, 0].std():.4f}")
    print(f"  Output range: [{outputs[:, 0].min():.4f}, {outputs[:, 0].max():.4f}]")
    print(f"  {x_lbl} range: [{inputs[:, 0].min():.4f}, {inputs[:, 0].max():.4f}]")
    print(f"  {y_lbl} range: [{inputs[:, 1].min():.4f}, {inputs[:, 1].max():.4f}]")

    print("\n[Heatmap2D] Interactive controls:")
    print("  escape: Close plot and exit")
    print("  space: Copy plot to clipboard")

    plt.tight_layout()
    plt.show()


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='2D heatmap visualization of sampled parameters with optimizer results'
    )
    parser.add_argument(
        '-d', '--data', type=str, required=True,
        help='Path to sample directory or HDF5 file'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to trained model checkpoint (default: auto-detect latest)'
    )
    parser.add_argument(
        '--fullscreen', action='store_true',
        help='Display in fullscreen mode'
    )
    parser.add_argument(
        '--legend', action='store_true', default=True,
        help='Show colorbar legend'
    )
    parser.add_argument(
        '--optimizer', action='store_true',
        help='Enable batched optimizer visualization'
    )
    parser.add_argument(
        '--optimize-mode', type=str, choices=['min', 'max'], default='min',
        help='Optimization mode: minimize or maximize (default: min)'
    )
    parser.add_argument(
        '--optimizer-points', type=int, default=30,
        help='Number of product constraint points (default: 30)'
    )
    parser.add_argument(
        '--trial-size', type=int, default=256,
        help='Parallel trials per constraint point (default: 256)'
    )
    parser.add_argument(
        '--max-iter', type=int, default=500,
        help='Maximum Adam iterations (default: 500)'
    )
    parser.add_argument(
        '--constraint-weight', type=float, default=1000.0,
        help='Product constraint penalty weight (default: 1000.0)'
    )

    args = parser.parse_args()

    plot_heatmap2d(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        fullscreen=args.fullscreen,
        legend=args.legend,
        use_optimizer=args.optimizer,
        optimize_mode=args.optimize_mode,
        optimizer_points=args.optimizer_points,
        trial_size=args.trial_size,
        max_iter=args.max_iter,
        constraint_weight=args.constraint_weight
    )


if __name__ == '__main__':
    main()

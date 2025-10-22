#!/usr/bin/env python3
"""
CLI interface for plotting commands.

Provides subcommands for different plot types, similar to the rollout CLI pattern.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description='Plotting tools for BidirectionalGaitNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available plot types:
  regression3d    3D visualization of regression model predictions vs data
  heatmap2d       2D heatmap of sampled parameters with optimizer overlay

Examples:
  # 3D regression plot (auto-detect latest checkpoint)
  ./scripts/plot regression3d -d sampled/base_anchor_all-026000-1014_092619+metabolic+on_20251023_131811

  # 3D with optimizer
  ./scripts/plot regression3d -d sampled/... --optimizer --optimizer-points 30

  # 2D heatmap
  ./scripts/plot heatmap2d -d sampled/...

  # 2D heatmap with optimizer overlay
  ./scripts/plot heatmap2d -d sampled/... --optimizer --optimizer-points 20
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Plot type to generate')

    # Subcommand: regression3d
    parser_regression3d = subparsers.add_parser(
        'regression3d',
        help='3D visualization of regression model predictions vs actual data'
    )
    parser_regression3d.add_argument(
        '-d', '--data', type=str, required=True,
        help='Path to sample directory or HDF5 file'
    )
    parser_regression3d.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to trained model checkpoint (default: auto-detect latest from data directory)'
    )
    parser_regression3d.add_argument(
        '--grid-size', type=int, default=50,
        help='Resolution of prediction meshgrid (default: 50)'
    )
    parser_regression3d.add_argument(
        '--fullscreen', action='store_true',
        help='Display in fullscreen mode'
    )
    parser_regression3d.add_argument(
        '--legend', action='store_true',
        help='Show legend'
    )
    parser_regression3d.add_argument(
        '--optimizer', action='store_true',
        help='Enable batched optimizer visualization'
    )
    parser_regression3d.add_argument(
        '--optimize-mode', type=str, choices=['min', 'max'], default='min',
        help='Optimization mode: minimize or maximize (default: min)'
    )
    parser_regression3d.add_argument(
        '--optimizer-points', type=int, default=30,
        help='Number of product constraint points (default: 30)'
    )
    parser_regression3d.add_argument(
        '--trial-size', type=int, default=256,
        help='Parallel trials per constraint point (default: 256)'
    )
    parser_regression3d.add_argument(
        '--max-iter', type=int, default=500,
        help='Maximum Adam iterations (default: 500)'
    )
    parser_regression3d.add_argument(
        '--constraint-weight', type=float, default=1000.0,
        help='Product constraint penalty weight (default: 1000.0)'
    )

    # Subcommand: heatmap2d
    parser_heatmap2d = subparsers.add_parser(
        'heatmap2d',
        help='2D heatmap of sampled parameters with optimizer overlay'
    )
    parser_heatmap2d.add_argument(
        '-d', '--data', type=str, required=True,
        help='Path to sample directory or HDF5 file'
    )
    parser_heatmap2d.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to trained model checkpoint (default: auto-detect latest from data directory)'
    )
    parser_heatmap2d.add_argument(
        '--fullscreen', action='store_true',
        help='Display in fullscreen mode'
    )
    parser_heatmap2d.add_argument(
        '--legend', action='store_true',
        help='Show colorbar legend'
    )
    parser_heatmap2d.add_argument(
        '--optimizer', action='store_true',
        help='Enable batched optimizer visualization'
    )
    parser_heatmap2d.add_argument(
        '--optimize-mode', type=str, choices=['min', 'max'], default='min',
        help='Optimization mode: minimize or maximize (default: min)'
    )
    parser_heatmap2d.add_argument(
        '--optimizer-points', type=int, default=30,
        help='Number of product constraint points (default: 30)'
    )
    parser_heatmap2d.add_argument(
        '--trial-size', type=int, default=256,
        help='Parallel trials per constraint point (default: 256)'
    )
    parser_heatmap2d.add_argument(
        '--max-iter', type=int, default=500,
        help='Maximum Adam iterations (default: 500)'
    )
    parser_heatmap2d.add_argument(
        '--constraint-weight', type=float, default=1000.0,
        help='Product constraint penalty weight (default: 1000.0)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'regression3d':
        from python.plot.regression3d import plot_regression3d
        plot_regression3d(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            grid_size=args.grid_size,
            fullscreen=args.fullscreen,
            legend=args.legend,
            use_optimizer=args.optimizer,
            optimize_mode=args.optimize_mode,
            optimizer_points=args.optimizer_points,
            trial_size=args.trial_size,
            max_iter=args.max_iter,
            constraint_weight=args.constraint_weight
        )
    elif args.command == 'heatmap2d':
        from python.plot.heatmap2d import plot_heatmap2d
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

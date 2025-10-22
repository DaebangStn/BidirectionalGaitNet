#!/usr/bin/env python3
"""
Main training script for regression MLP.

Usage:
    python train.py --config config.yaml --data path/to/rollout_data.h5
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from python.learn.hdf5_dataset import HDF5DataModule
from python.learn.network import RegressionNet
from python.learn.trainer import Trainer
from python.learn.logger import SimpleLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_data_path(path: str) -> str:
    """
    Resolve data path to rollout_data.h5 file.

    If path is a directory (sample directory), append '/rollout_data.h5'.
    If path is already a .h5 file, use it directly.

    Args:
        path: Either a sample directory or direct path to .h5 file

    Returns:
        Full path to rollout_data.h5 file

    Raises:
        SystemExit: If the resolved .h5 file doesn't exist
    """
    path_obj = Path(path)

    # If it's already a .h5 file, use it directly
    if path.endswith('.h5'):
        resolved_path = path
    else:
        # Otherwise, assume it's a sample directory and append rollout_data.h5
        resolved_path = str(path_obj / 'rollout_data.h5')

    # Verify the file exists
    if not Path(resolved_path).exists():
        print(f"Error: HDF5 file not found at {resolved_path}")
        print(f"       Expected rollout_data.h5 in directory: {path}")
        sys.exit(1)

    return resolved_path


def get_next_version_dir(data_path: str) -> Path:
    """Get next version directory (v01, v02, etc.) in the sampled data directory."""
    data_dir = Path(data_path).parent

    # Find existing version directories
    existing_versions = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('v') and item.name[1:].isdigit():
            try:
                version_num = int(item.name[1:])
                existing_versions.append(version_num)
            except ValueError:
                continue

    # Get next version number
    next_version = max(existing_versions, default=0) + 1
    version_dir = data_dir / f"v{next_version:02d}"

    return version_dir


def main():
    parser = argparse.ArgumentParser(description='Train regression MLP on HDF5 rollout dataset')
    parser.add_argument('--config', type=str, default='data/regress/config.yaml',
                       help='Path to YAML configuration file (default: data/regress/config.yaml)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to sample directory or HDF5 rollout data file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: auto-version in sampled data directory)')
    args = parser.parse_args()

    # Verify config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found at {args.config}")
        print(f"       Please provide a valid config file with --config argument")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Resolve data path to actual HDF5 file
    resolved_data_path = resolve_data_path(args.data)
    print(f"Using HDF5 file: {resolved_data_path}")

    # Override data path if provided
    if 'data' not in config:
        config['data'] = {}
    config['data']['hdf5_path'] = resolved_data_path

    # Extract configuration sections
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    trainer_config = config.get('trainer', {})

    # Create output directory with auto-versioning
    if args.output is None:
        output_dir = get_next_version_dir(resolved_data_path)
        print(f"Auto-versioned output directory: {output_dir}")
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Save configuration to output directory
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_save_path}")

    # Initialize data module
    print("\n" + "="*80)
    print("Initializing HDF5 data module...")
    print("="*80)
    data_module = HDF5DataModule(
        hdf5_path=data_config['hdf5_path'],
        in_lbl=data_config.get('in_lbl', ['gait_cadence', 'gait_stride']),
        out_lbl=data_config.get('out_lbl', ['metabolic/cot/MA/mean']),
        batch_size=data_config.get('batch_size', 65536),
        train_split=data_config.get('train_split', 0.8),
        random_seed=data_config.get('random_seed', 42)
    )

    # Initialize network
    print("\n" + "="*80)
    print("Initializing network...")
    print("="*80)
    model = RegressionNet(
        input_dim=data_module.input_dim,
        output_dim=data_module.output_dim,
        layers=model_config.get('layers', [64, 32]),
        input_mean=data_module.input_mean,
        input_std=data_module.input_std,
        target_mean=data_module.target_mean,
        target_std=data_module.target_std,
        residual=model_config.get('residual', True),
        activation=model_config.get('activation', 'gelu')
    )

    print(f"Model architecture:")
    print(f"  Input dim: {data_module.input_dim}")
    print(f"  Output dim: {data_module.output_dim}")
    print(f"  Hidden layers: {model_config.get('layers', [64, 32])}")
    print(f"  Residual connections: {model_config.get('residual', True)}")
    print(f"  Activation: {model_config.get('activation', 'gelu')}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize logger
    logger = SimpleLogger(log_dir=log_dir, experiment_name='regression_hdf5')

    # Log hyperparameters
    hparams = {
        'data': data_config,
        'model': model_config,
        'trainer': trainer_config,
    }
    logger.log_hyperparameters(hparams)

    # Initialize trainer
    print("\n" + "="*80)
    print("Initializing trainer...")
    print("="*80)
    trainer = Trainer(
        model=model,
        data=data_module,
        logger=logger,
        ckpt_dir=ckpt_dir,
        lr=trainer_config.get('lr', 0.1),
        min_lr=trainer_config.get('min_lr', 0.0005),
        l1_weight=trainer_config.get('l1_weight', 0.001),
        l2_weight=trainer_config.get('l2_weight', 0.001),
        grad_weight=trainer_config.get('grad_weight', 0.01),
        recon_weight=trainer_config.get('recon_weight', 10),
        recon_delta=trainer_config.get('recon_delta', 1.0),
        recon_type=trainer_config.get('recon_type', 'huber'),
        recon_start_epoch=trainer_config.get('recon_start_epoch', 0),
        log_period=trainer_config.get('log_period', 50),
        max_epochs=trainer_config.get('max_epochs', 10000),
        save_period=trainer_config.get('save_period', 1000)
    )

    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    trainer.train()

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Logs saved to: {log_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

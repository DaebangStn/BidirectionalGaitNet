#!/usr/bin/env python3
"""
Batched optimizer using Adam for parallel optimization with product constraints.

Optimizes multiple points simultaneously using gradient-based optimization
with cadence × stride product constraints.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict
from tqdm import trange

from python.learn.network import RegressionNet
from python.plot.util import get_parameter_ranges


class BatchedOptimizer:
    """Adam-based batched optimizer for regression models with product constraints."""

    def __init__(self, checkpoint_path: str, data_path: str, maximize: bool = False):
        """
        Initialize batched optimizer with trained model checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint file
            data_path: Path to sample directory or HDF5 file
            maximize: If True, maximize output; if False, minimize
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.maximize = maximize

        # Load model and configuration
        self.model, self.config = self._load_checkpoint(checkpoint_path)
        self.device = next(self.model.parameters()).device
        self.model.eval()

        # Extract data configuration
        self.in_lbl = self.config['data']['in_lbl']
        self.out_lbl = self.config['data']['out_lbl']

        # Resolve data path
        if str(data_path).endswith('.h5'):
            hdf5_path = data_path
        else:
            hdf5_path = Path(data_path) / 'rollout_data.h5'

        # Compute parameter bounds from training data
        self.param_ranges = get_parameter_ranges(str(hdf5_path), self.in_lbl)

        # Convert bounds to tensors (normalized space)
        self.low = torch.tensor(
            [self._normalize_value(lbl, self.param_ranges[lbl][0]) for lbl in self.in_lbl],
            device=self.device, dtype=torch.float32
        )
        self.high = torch.tensor(
            [self._normalize_value(lbl, self.param_ranges[lbl][1]) for lbl in self.in_lbl],
            device=self.device, dtype=torch.float32
        )

        print(f"[BatchedOptimizer] Loaded model from {checkpoint_path}")
        print(f"[BatchedOptimizer] Device: {self.device}")
        print(f"[BatchedOptimizer] Input parameters: {self.in_lbl}")
        print(f"[BatchedOptimizer] Output parameters: {self.out_lbl}")
        print(f"[BatchedOptimizer] Optimization mode: {'maximize' if maximize else 'minimize'}")

    def _load_checkpoint(self, ckpt_path: str):
        """Load model and configuration from checkpoint."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ckpt_path, map_location=device)

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

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Load full config
        ckpt_dir = Path(ckpt_path).parent.parent
        config_path = ckpt_dir / 'config.yaml'

        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        return model, full_config

    def _normalize_value(self, param_name: str, value: float) -> float:
        """Normalize a single parameter value."""
        idx = self.in_lbl.index(param_name)
        mean = self.model.input_mean[idx].item()
        std = self.model.input_std[idx].item()
        return (value - mean) / std

    def _denormalize_value(self, param_name: str, value: float) -> float:
        """Denormalize a single parameter value."""
        idx = self.in_lbl.index(param_name)
        mean = self.model.input_mean[idx].item()
        std = self.model.input_std[idx].item()
        return value * std + mean

    def run_with_product_constraint(
        self,
        opt_field: str,
        product_values: np.ndarray,
        trial_size: int = 256,
        max_iter: int = 500,
        lr: float = 5e-2,
        constraint_weight: float = 1000.0,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Optimize with product constraint: cadence × stride = product_value.

        Args:
            opt_field: Output field to optimize
            product_values: (N,) array of product constraint values
            trial_size: Number of parallel trials per constraint
            max_iter: Number of Adam iterations
            lr: Learning rate
            constraint_weight: Penalty weight for constraint violation
            verbose: Show progress bar

        Returns:
            List of N dicts with optimal inputs and output value
        """
        if len(self.in_lbl) != 2:
            raise ValueError(f"Product constraint requires exactly 2 inputs, got {len(self.in_lbl)}")

        if opt_field not in self.out_lbl:
            raise ValueError(f"Output field '{opt_field}' not in model outputs: {self.out_lbl}")

        out_idx = self.out_lbl.index(opt_field)
        N = len(product_values)
        B = trial_size

        # Normalize product constraints
        product_tensor = torch.tensor(product_values, device=self.device, dtype=torch.float32)

        # Initialize random parameters (N, B, 2)
        raw = torch.randn(N, B, 2, device=self.device, requires_grad=True)
        optim = torch.optim.Adam([raw], lr=lr)

        # Optimization loop
        pbar = trange(max_iter, desc="Optimizing", disable=not verbose, ncols=100)
        for iteration in pbar:
            optim.zero_grad()

            # Sigmoid reparameterization to enforce bounds
            x = self.low + (self.high - self.low) * torch.sigmoid(raw)  # (N, B, 2)

            # Model forward pass
            x_flat = x.reshape(N * B, 2)
            y = self.model(x_flat, normalize=False, denormalize=False)  # (N*B, output_dim)
            y = y[:, out_idx].reshape(N, B)  # (N, B)

            # Objective loss
            if self.maximize:
                obj_loss = -y.sum()
            else:
                obj_loss = y.sum()

            # Denormalize for product constraint
            x_denorm = torch.stack([
                x[:, :, i] * self.model.input_std[i] + self.model.input_mean[i]
                for i in range(2)
            ], dim=2)  # (N, B, 2)

            # Product constraint: cadence × stride = product_value
            products = x_denorm[:, :, 0] * x_denorm[:, :, 1]  # (N, B)
            constraint_loss = ((products - product_tensor.unsqueeze(1)) ** 2).sum()

            # Total loss
            loss = obj_loss + constraint_weight * constraint_loss

            # Backward pass
            loss.backward()
            optim.step()

            if verbose and iteration % 50 == 0:
                pbar.set_postfix(
                    obj=f"{obj_loss.item():.2f}",
                    const=f"{constraint_loss.item():.4f}",
                    lr=f"{optim.param_groups[0]['lr']:.6f}"
                )

        # Select best result for each constraint
        with torch.no_grad():
            x = self.low + (self.high - self.low) * torch.sigmoid(raw)
            x_flat = x.reshape(N * B, 2)
            y = self.model(x_flat, normalize=False, denormalize=False)
            y = y[:, out_idx].reshape(N, B)

            # Denormalize
            x_denorm = torch.stack([
                x[:, :, i] * self.model.input_std[i] + self.model.input_mean[i]
                for i in range(2)
            ], dim=2)

            # Add constraint penalty for selection
            products = x_denorm[:, :, 0] * x_denorm[:, :, 1]
            constraint_penalty = ((products - product_tensor.unsqueeze(1)) ** 2)

            if self.maximize:
                objective = -y + constraint_weight * constraint_penalty
                best_idx = torch.argmin(objective, dim=1)  # min penalty
            else:
                objective = y + constraint_weight * constraint_penalty
                best_idx = torch.argmin(objective, dim=1)

            # Extract best solutions
            batch_indices = torch.arange(N, device=self.device)
            best_x = x_denorm[batch_indices, best_idx]  # (N, 2)
            best_y_norm = y[batch_indices, best_idx]  # (N,)

            # Denormalize output
            out_mean = self.model.target_mean[out_idx].item()
            out_std = self.model.target_std[out_idx].item()
            best_y = best_y_norm * out_std + out_mean

        # Convert to list of dicts
        results = []
        for i in range(N):
            result = {
                self.in_lbl[0]: best_x[i, 0].item(),
                self.in_lbl[1]: best_x[i, 1].item(),
                opt_field: best_y[i].item()
            }
            results.append(result)

        return results

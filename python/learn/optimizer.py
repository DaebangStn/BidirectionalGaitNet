#!/usr/bin/env python3
"""
Optimizer for finding input parameters that minimize/maximize model outputs.

Uses gradient-based optimization with scipy.minimize to find optimal inputs
given a trained regression model.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize, OptimizeResult
from functools import partial

from python.learn.network import RegressionNet


class Optimizer:
    """Gradient-based optimizer for regression models."""

    def __init__(self, ckpt_path: str, maximize: bool = False, trials: int = 20):
        """
        Initialize optimizer with trained model checkpoint.

        Args:
            ckpt_path: Path to model checkpoint file
            maximize: If True, maximize output; if False, minimize
            trials: Number of random initializations to try
        """
        self.ckpt_path = Path(ckpt_path)
        self.maximize = maximize
        self.trials = trials

        # Load model and configuration
        self.model, self.config = self._load_checkpoint(ckpt_path)
        self.device = next(self.model.parameters()).device
        self.model.eval()

        # Extract data configuration for bounds computation
        # Support both data_path (old) and hdf5_path (new) naming
        self.data_path = self.config['data'].get('hdf5_path') or self.config['data'].get('data_path')
        self.in_lbl = self.config['data']['in_lbl']
        self.out_lbl = self.config['data']['out_lbl']

        # Compute parameter bounds from training data
        self.param_bounds = self._compute_bounds(self.data_path, self.in_lbl)

        print(f"[Optimizer] Loaded model from {ckpt_path}")
        print(f"[Optimizer] Device: {self.device}")
        print(f"[Optimizer] Input parameters: {self.in_lbl}")
        print(f"[Optimizer] Output parameters: {self.out_lbl}")
        print(f"[Optimizer] Optimization mode: {'maximize' if maximize else 'minimize'}")
        print(f"[Optimizer] Trials: {trials}")

    def _load_checkpoint(self, ckpt_path: str) -> Tuple[RegressionNet, dict]:
        """
        Load model and configuration from checkpoint.

        Args:
            ckpt_path: Path to checkpoint file

        Returns:
            (model, config) tuple
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

        # Load full config including data/trainer sections
        # The checkpoint should have saved the full config from train.py
        ckpt_dir = Path(ckpt_path).parent.parent
        config_path = ckpt_dir / 'config.yaml'

        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
        else:
            # Fallback: construct minimal config
            full_config = {
                'data': {
                    'data_path': str(Path(ckpt_path).parent.parent.parent / 'rollout_data.h5'),
                    'in_lbl': ['gait_cadence', 'gait_stride'],
                    'out_lbl': ['metabolic/cot/MA/mean']
                },
                'model': {
                    'layers': config['layers'],
                    'residual': config['residual']
                }
            }
            print(f"[Warning] Config file not found, using fallback configuration")

        return model, full_config

    def _compute_bounds(self, data_path: str, in_lbl: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Compute parameter bounds from training data.

        Args:
            data_path: Path to HDF5 data file
            in_lbl: List of input parameter names

        Returns:
            Dictionary mapping parameter name to (min, max) tuple
        """
        # Convert to absolute path
        data_path_obj = Path(data_path)
        if not data_path_obj.is_absolute():
            # Data path in config is relative to project root
            # Checkpoint: sampled/.../vXX/checkpoints/ep_NNNNNN.ckpt or /abs/path/.../vXX/checkpoints/ep_NNNNNN.ckpt
            # Config:     sampled/.../vXX/config.yaml
            # Data:       sampled/.../rollout_data.h5

            ckpt_abs = Path(self.ckpt_path).resolve()
            version_dir = ckpt_abs.parent.parent  # Up to vXX

            # Get current working directory (where test is run from)
            cwd = Path.cwd()

            # Try data_path relative to cwd
            candidate1 = cwd / data_path

            # Try data_path relative to version_dir parent (sampled/...)
            candidate2 = version_dir.parent / data_path.split('/')[-1]  # Just the filename

            # Try up one more level (for when checkpoint path is relative)
            candidate3 = version_dir.parent / 'rollout_data.h5'

            if candidate1.exists():
                data_path = candidate1
            elif candidate2.exists():
                data_path = candidate2
            elif candidate3.exists():
                data_path = candidate3
            else:
                raise FileNotFoundError(f"Could not find data file. Tried:\n  {candidate1}\n  {candidate2}\n  {candidate3}")
        else:
            data_path = data_path_obj

        bounds = {}

        with h5py.File(data_path, 'r') as f:
            # Get parameter names from metadata
            if 'metadata/parameter_names' in f:
                param_names_data = f['metadata/parameter_names'][()]
            elif '/parameter_names' in f:
                param_names_data = f['/parameter_names'][()]
            else:
                # Try to get from first param group attributes
                param_keys = sorted([k for k in f.keys() if k.startswith('param_')])
                if not param_keys:
                    raise ValueError(f"No param groups found in {data_path}")
                first_param = f[param_keys[0]]
                param_names_data = first_param.attrs.get('parameter_names', None)
                if param_names_data is None:
                    raise KeyError(f"Could not find parameter_names in {data_path}")

            # Handle both string and bytes
            if isinstance(param_names_data, bytes):
                param_names = param_names_data.decode('utf-8').split(',')
            elif isinstance(param_names_data, np.ndarray):
                param_names = [s.decode('utf-8') if isinstance(s, bytes) else s for s in param_names_data]
            else:
                param_names = param_names_data.split(',')

            # Get param keys after we have parameter_names
            param_keys = sorted([k for k in f.keys() if k.startswith('param_')])

            # Get indices for input labels
            in_indices = []
            for lbl in in_lbl:
                if lbl not in param_names:
                    raise ValueError(f"Input label '{lbl}' not found in parameter_names: {param_names}")
                in_indices.append(param_names.index(lbl))

            # Collect all parameter values to find global min/max
            all_values = {lbl: [] for lbl in in_lbl}

            for param_key in param_keys:
                param_group = f[param_key]
                param_state = param_group['param_state'][:]

                for i, lbl in zip(in_indices, in_lbl):
                    all_values[lbl].append(param_state[i])

            # Compute bounds (denormalized space)
            for lbl in in_lbl:
                values = np.array(all_values[lbl])
                bounds[lbl] = (float(values.min()), float(values.max()))

        print(f"[Optimizer] Parameter bounds (denormalized):")
        for lbl, (min_val, max_val) in bounds.items():
            print(f"  {lbl}: [{min_val:.6f}, {max_val:.6f}]")

        return bounds

    def normalize(self, values_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize parameter values using model statistics.

        Args:
            values_dict: Dictionary of denormalized parameter values

        Returns:
            Dictionary of normalized values
        """
        normalized = {}
        for i, lbl in enumerate(self.in_lbl):
            if lbl in values_dict:
                val = values_dict[lbl]
                mean = self.model.input_mean[i].item()
                std = self.model.input_std[i].item()
                normalized[lbl] = (val - mean) / std
        return normalized

    def denormalize(self, values_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Denormalize parameter values using model statistics.

        Args:
            values_dict: Dictionary of normalized parameter values

        Returns:
            Dictionary of denormalized values
        """
        denormalized = {}

        # Denormalize inputs
        for i, lbl in enumerate(self.in_lbl):
            if lbl in values_dict:
                val_norm = values_dict[lbl]
                mean = self.model.input_mean[i].item()
                std = self.model.input_std[i].item()
                denormalized[lbl] = val_norm * std + mean

        # Denormalize outputs
        for i, lbl in enumerate(self.out_lbl):
            if lbl in values_dict:
                val_norm = values_dict[lbl]
                mean = self.model.target_mean[i].item()
                std = self.model.target_std[i].item()
                denormalized[lbl] = val_norm * std + mean

        return denormalized

    def _objective(self, x_opt: np.ndarray, out_idx: int,
                   fixed_fields: Dict[str, float],
                   optimize_fields: List[str]) -> Tuple[float, np.ndarray]:
        """
        Compute objective value and gradient for scipy.minimize.

        Args:
            x_opt: Normalized optimization parameters (1D array)
            out_idx: Index of output to optimize
            fixed_fields: Normalized fixed parameter values
            optimize_fields: Names of parameters being optimized

        Returns:
            (objective_value, gradient) both as float64 for scipy
        """
        # Reconstruct full input vector
        x_dict = {field: fixed_fields.get(field, None) for field in self.in_lbl}
        for i, field in enumerate(optimize_fields):
            x_dict[field] = float(x_opt[i])

        x_full = torch.tensor(
            [x_dict[field] for field in self.in_lbl],
            device=self.device,
            dtype=torch.float32,
            requires_grad=True
        )

        # Forward pass (input already normalized)
        y = self.model(x_full.unsqueeze(0), normalize=False, denormalize=False).squeeze(0)[out_idx]

        # Compute gradient
        y.backward()

        grads = x_full.grad.cpu().detach().numpy()
        grad_opt = np.array([grads[self.in_lbl.index(field)] for field in optimize_fields])

        value = y.cpu().detach().numpy()

        # Apply maximize flag
        if self.maximize:
            value = -value
            grad_opt = -grad_opt

        # Ensure float64 for scipy compatibility
        return np.float64(value), np.asarray(grad_opt, dtype=np.float64)

    def _check_optimized(self, best: Optional[OptimizeResult], found: float) -> bool:
        """Check if found value is better than best value."""
        return (best is None) or (self.maximize and found > best.fun) or (not self.maximize and found < best.fun)

    def run(self, out_field: str, fixed_fields: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], float]:
        """
        Run optimization to find inputs that minimize/maximize output.

        Args:
            out_field: Name of output parameter to optimize
            fixed_fields: Optional dict of parameter_name -> denormalized_value for fixed inputs

        Returns:
            (optimized_params_denorm, optimal_value_denorm) tuple where:
                - optimized_params_denorm: Dict of all input parameters (denormalized)
                - optimal_value_denorm: Optimal output value (denormalized)
        """
        if fixed_fields is None:
            fixed_fields = {}

        # Normalize fixed fields
        fixed_fields_norm = self.normalize(fixed_fields)

        # Identify optimization fields
        optimize_fields = [field for field in self.in_lbl if field not in fixed_fields]

        if not optimize_fields:
            raise ValueError("No fields to optimize - all inputs are fixed")

        # Get output index
        if out_field not in self.out_lbl:
            raise ValueError(f"Output field '{out_field}' not in model outputs: {self.out_lbl}")
        out_idx = self.out_lbl.index(out_field)

        # Create objective function
        obj = partial(self._objective, out_idx=out_idx,
                     fixed_fields=fixed_fields_norm,
                     optimize_fields=optimize_fields)

        # Normalize bounds for optimization fields
        bounds_norm = []
        for field in optimize_fields:
            min_denorm, max_denorm = self.param_bounds[field]
            min_norm = self.normalize({field: min_denorm})[field]
            max_norm = self.normalize({field: max_denorm})[field]
            bounds_norm.append((min_norm, max_norm))

        # Run multiple trials
        best = None
        print(f"\n[Optimizer] Starting optimization for '{out_field}'")
        print(f"[Optimizer] Optimizing fields: {optimize_fields}")
        print(f"[Optimizer] Fixed fields: {list(fixed_fields.keys())}")

        for trial in range(self.trials):
            # Random initialization in normalized space
            initial_guess = np.array([
                np.random.uniform(bounds_norm[i][0], bounds_norm[i][1])
                for i in range(len(optimize_fields))
            ])

            # Optimize
            result = minimize(
                obj,
                initial_guess,
                jac=True,
                bounds=bounds_norm,
                method='SLSQP',
                options={'ftol': 1e-6, 'maxiter': 200}
            )

            if result.success and self._check_optimized(best, result.fun):
                best = result
                print(f"  Trial {trial+1}/{self.trials}: success, fun={result.fun:.6f}")
            elif not result.success:
                print(f"  Trial {trial+1}/{self.trials}: failed - {result.message}")

        if best is None:
            raise RuntimeError("Optimization failed in all trials")

        # Reconstruct full optimized input (normalized)
        best_x_norm = {field: fixed_fields_norm.get(field, None) for field in self.in_lbl}
        for i, field in enumerate(optimize_fields):
            best_x_norm[field] = best.x[i] if best.x.ndim > 0 else float(best.x)

        # Denormalize results
        best_x_denorm = self.denormalize(best_x_norm)

        # Compute denormalized output value
        best_y_norm = -best.fun if self.maximize else best.fun
        best_y_denorm = self.denormalize({out_field: best_y_norm})[out_field]

        print(f"\n[Optimizer] Optimization complete!")
        print(f"[Optimizer] Optimal value: {best_y_denorm:.6f}")
        print(f"[Optimizer] Optimal inputs:")
        for field in self.in_lbl:
            print(f"  {field}: {best_x_denorm[field]:.6f}")

        return best_x_denorm, best_y_denorm

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
from pathlib import Path


class RolloutDataset(Dataset):
    """PyTorch Dataset for loading data from rollout files.

    Reads param_N groups from HDF5 file, extracts inputs from param_state dataset
    and outputs from attributes.
    """

    def __init__(self, data_path: str, in_lbl: List[str], out_lbl: List[str]):
        """
        Args:
            data_path: Path to HDF5 file
            in_lbl: Input feature labels (e.g., ["gait_cadence", "gait_stride"])
            out_lbl: Output feature labels (e.g., ["metabolic/cot/MA/mean"])
        """
        self.data_path = data_path
        self.in_lbl = in_lbl
        self.out_lbl = out_lbl

        # Load data from HDF5
        inputs_list = []
        outputs_list = []

        with h5py.File(data_path, 'r') as f:
            # Get parameter names mapping
            parameter_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                             for name in f['parameter_names'][:]]

            # Find indices for input labels in parameter_names
            in_indices = [parameter_names.index(lbl) for lbl in in_lbl]

            # Iterate through all param_* groups
            param_keys = sorted([key for key in f.keys() if key.startswith('param_')])

            for param_key in param_keys:
                param_group = f[param_key]

                # Check if rollout was successful
                success = param_group.attrs.get('success', True)
                if isinstance(success, np.ndarray):
                    success = success.item()

                if not success:
                    continue

                # Extract inputs from param_state dataset
                if 'param_state' in param_group:
                    param_state = param_group['param_state'][:]
                    inputs = param_state[in_indices].astype(np.float32)
                else:
                    # Fallback: try to get from attributes
                    inputs = np.array([param_group.attrs.get(lbl, 0.0) for lbl in in_lbl],
                                    dtype=np.float32)

                # Extract outputs from attributes
                # Use label directly as attribute path (e.g., 'metabolic/cot/MA/mean')
                outputs = []
                for lbl in out_lbl:
                    val = param_group.attrs.get(lbl, 0.0)
                    outputs.append(val)

                outputs = np.array(outputs, dtype=np.float32)

                inputs_list.append(inputs)
                outputs_list.append(outputs)

        # Convert to tensors
        self.inputs = torch.from_numpy(np.stack(inputs_list))
        self.targets = torch.from_numpy(np.stack(outputs_list))

        print(f"[RolloutDataset] Loaded {len(self.inputs)} samples from {Path(data_path).name}")
        print(f"[RolloutDataset] Input shape: {self.inputs.shape}, Target shape: {self.targets.shape}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std for normalization.

        Returns:
            (mean, std) tensors of shape (input_dim + output_dim,)
        """
        combined = torch.cat([self.inputs, self.targets], dim=1)
        mean = combined.mean(dim=0)
        std = combined.std(dim=0, unbiased=False)  # Use biased estimator to avoid NaN with 1 sample
        # Prevent division by zero - replace NaN or very small values with 1.0
        std = torch.where(torch.isnan(std) | (std < 1e-6), torch.ones_like(std), std)
        return mean, std


class DataModule:
    """Data module for managing dataset loading and preprocessing."""

    def __init__(self, data_path: str, in_lbl: List[str], out_lbl: List[str],
                 batch_size: int = 65536, train_split: float = 0.8,
                 random_seed: int = 42):
        """
        Args:
            data_path: Path to data file
            in_lbl: Input feature labels
            out_lbl: Output feature labels
            batch_size: Batch size for dataloaders
            train_split: Fraction of data to use for training (rest for validation)
            random_seed: Random seed for train/val split
        """
        self.data_path = data_path
        self.in_lbl = in_lbl
        self.out_lbl = out_lbl
        self.batch_size = batch_size
        self.train_split = train_split
        self.random_seed = random_seed

        self._train_ds = None
        self._val_ds = None
        self._input_mean = None
        self._input_std = None
        self._target_mean = None
        self._target_std = None

        self.setup()

    def setup(self):
        """Load data (no train/val split - uses all data)."""
        # Create datasets - load all param_* groups
        self._train_ds = RolloutDataset(self.data_path, self.in_lbl, self.out_lbl)
        self._val_ds = self._train_ds  # Same dataset for validation

        print(f"[DataModule] Loaded {len(self._train_ds)} samples (using all for train and val)")

        # Compute normalization statistics from training data
        mean, std = self._train_ds.get_statistics()
        input_dim = len(self.in_lbl)
        self._input_mean = mean[:input_dim]
        self._input_std = std[:input_dim]
        self._target_mean = mean[input_dim:]
        self._target_std = std[input_dim:]

        print(f"[DataModule] Input mean: {self._input_mean}, std: {self._input_std}")
        print(f"[DataModule] Target mean: {self._target_mean}, std: {self._target_std}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    @property
    def input_mean(self) -> torch.Tensor:
        return self._input_mean

    @property
    def input_std(self) -> torch.Tensor:
        return self._input_std

    @property
    def target_mean(self) -> torch.Tensor:
        return self._target_mean

    @property
    def target_std(self) -> torch.Tensor:
        return self._target_std

    @property
    def input_dim(self) -> int:
        return len(self.in_lbl)

    @property
    def output_dim(self) -> int:
        return len(self.out_lbl)

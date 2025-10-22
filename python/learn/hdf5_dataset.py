import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
from pathlib import Path


class HDF5Dataset(Dataset):
    """PyTorch Dataset for loading data from HDF5 rollout files.

    Reads param_N groups from HDF5 file, extracts inputs from param_state dataset
    and outputs from attributes.
    """

    def __init__(self, hdf5_path: str, in_lbl: List[str], out_lbl: List[str],
                 indices: Optional[List[int]] = None, device: str = 'cpu'):
        """
        Args:
            hdf5_path: Path to HDF5 file
            in_lbl: Input feature labels (e.g., ["Phase", "Stride"])
            out_lbl: Output feature labels (e.g., ["velocity", "cot_ma15"])
            indices: Optional subset of param indices to use (for train/val split)
            device: Device to load tensors on ('cpu' or 'cuda')
        """
        self.hdf5_path = hdf5_path
        self.in_lbl = in_lbl
        self.out_lbl = out_lbl
        self.device = torch.device(device)

        # Load data from HDF5
        inputs_list = []
        outputs_list = []

        with h5py.File(hdf5_path, 'r') as f:
            # Get parameter names mapping
            parameter_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                             for name in f['parameter_names'][:]]

            # Find indices for input labels in parameter_names
            in_indices = [parameter_names.index(lbl) for lbl in in_lbl]

            # Iterate through all param_N groups
            param_indices = []
            param_keys = {}  # Map index to actual key name
            for key in f.keys():
                if key.startswith('param_'):
                    # Handle both integer format (param_0) and float format (param_0.0)
                    param_idx = int(float(key.split('_')[1]))
                    param_indices.append(param_idx)
                    param_keys[param_idx] = key

            param_indices.sort()

            # If indices provided, filter to those
            if indices is not None:
                param_indices = [idx for idx in param_indices if idx in indices]

            for param_idx in param_indices:
                param_group = f[param_keys[param_idx]]

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
                outputs = []
                for lbl in out_lbl:
                    # Try different attribute paths
                    if lbl == 'velocity':
                        val = param_group.attrs.get('velocity', 0.0)
                    elif lbl in ['cot_ma', 'cot_ma15', 'cot_ma3']:
                        # Extract from metabolic/cot/MA/mean or similar
                        val = param_group.attrs.get(f'metabolic/cot/MA/mean', 0.0)
                    else:
                        val = param_group.attrs.get(lbl, 0.0)

                    outputs.append(val)

                outputs = np.array(outputs, dtype=np.float32)

                inputs_list.append(inputs)
                outputs_list.append(outputs)

        # Convert to tensors and move to device
        self.inputs = torch.from_numpy(np.stack(inputs_list)).to(self.device)
        self.targets = torch.from_numpy(np.stack(outputs_list)).to(self.device)

        print(f"[HDF5Dataset] Loaded {len(self.inputs)} samples from {Path(hdf5_path).name}")
        print(f"[HDF5Dataset] Input shape: {self.inputs.shape}, Target shape: {self.targets.shape}")
        print(f"[HDF5Dataset] Device: {self.device}")

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


class HDF5DataModule:
    """Data module for managing HDF5 dataset loading and preprocessing."""

    def __init__(self, hdf5_path: str, in_lbl: List[str], out_lbl: List[str],
                 batch_size: int = 65536, train_split: float = 0.8,
                 random_seed: int = 42, device: str = 'auto'):
        """
        Args:
            hdf5_path: Path to HDF5 file
            in_lbl: Input feature labels
            out_lbl: Output feature labels
            batch_size: Batch size for dataloaders
            train_split: Fraction of data to use for training (rest for validation)
            random_seed: Random seed for train/val split
            device: Device to load data on ('cpu', 'cuda', or 'auto' for automatic detection)
        """
        self.hdf5_path = hdf5_path
        self.in_lbl = in_lbl
        self.out_lbl = out_lbl
        self.batch_size = batch_size
        self.train_split = train_split
        self.random_seed = random_seed

        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._train_ds = None
        self._val_ds = None
        self._input_mean = None
        self._input_std = None
        self._target_mean = None
        self._target_std = None

        self.setup()

    def setup(self):
        """Load data and create train/val splits."""
        # First, load all data to get total number of samples
        with h5py.File(self.hdf5_path, 'r') as f:
            param_indices = []
            for key in f.keys():
                if key.startswith('param_'):
                    # Handle both integer format (param_0) and float format (param_0.0)
                    param_idx = int(float(key.split('_')[1]))
                    param_indices.append(param_idx)
            param_indices.sort()

        # Use all data for training (no split)
        train_indices = param_indices
        val_indices = param_indices

        print(f"[HDF5DataModule] Total samples: {len(param_indices)} (using all for train and val)")
        print(f"[HDF5DataModule] Loading data on device: {self.device}")

        # Create datasets with device placement
        self._train_ds = HDF5Dataset(self.hdf5_path, self.in_lbl, self.out_lbl, train_indices, device=self.device)
        self._val_ds = HDF5Dataset(self.hdf5_path, self.in_lbl, self.out_lbl, val_indices, device=self.device)

        # Compute normalization statistics from training data
        mean, std = self._train_ds.get_statistics()
        input_dim = len(self.in_lbl)
        self._input_mean = mean[:input_dim]
        self._input_std = std[:input_dim]
        self._target_mean = mean[input_dim:]
        self._target_std = std[input_dim:]

        print(f"[HDF5DataModule] Input mean: {self._input_mean}, std: {self._input_std}")
        print(f"[HDF5DataModule] Target mean: {self._target_mean}, std: {self._target_std}")

    def train_dataloader(self) -> DataLoader:
        # When data is on CUDA, use num_workers=0 and pin_memory=False for better performance
        use_cuda = self.device == 'cuda'
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 if use_cuda else 4,
            pin_memory=False if use_cuda else True
        )

    def val_dataloader(self) -> DataLoader:
        # When data is on CUDA, use num_workers=0 and pin_memory=False for better performance
        use_cuda = self.device == 'cuda'
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0 if use_cuda else 4,
            pin_memory=False if use_cuda else True
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

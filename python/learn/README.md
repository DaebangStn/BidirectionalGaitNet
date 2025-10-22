# HDF5-based Regression MLP Implementation

Pure PyTorch implementation of regression MLP for learning from HDF5 rollout data.

## Overview

This implementation provides a complete training pipeline for regression models using HDF5 datasets, without PyTorch Lightning dependency.

## Components

### 1. **hdf5_dataset.py**
- `HDF5Dataset`: PyTorch Dataset for loading param_N groups from HDF5 files
- `HDF5DataModule`: Data management with automatic normalization statistics

**Features:**
- Extracts inputs from `param_state` dataset using `parameter_names` mapping
- Extracts outputs from attributes (e.g., `metabolic/cot/MA/mean`)
- Filters by `success` attribute
- Computes normalization statistics (mean/std) from training data
- Handles single-sample edge cases

### 2. **network.py**
- `ResidualMLP`: Multi-layer perceptron with skip connections
- `SequentialMLP`: Standard sequential architecture (no residuals)
- `RegressionNet`: Main network with normalization and configuration management

**Features:**
- Configurable architecture (layer sizes, activation functions)
- Optional residual connections
- Built-in input normalization and output denormalization
- Xavier initialization
- Checkpoint save/load support

### 3. **trainer.py**
- `Trainer`: Training loop with multi-component loss

**Loss Components:**
- Reconstruction loss: MSE, Huber, Cauchy, Welsch, Geman-McClure, Tukey biweight
- L1 regularization: Weight sparsity
- L2 regularization: Weight decay
- Gradient penalty: Smooth predictions

**Features:**
- Adam optimizer with ReduceLROnPlateau scheduler
- Epoch-based training with validation
- Automatic checkpointing
- Progress tracking with tqdm

### 4. **logger.py**
- `SimpleLogger`: JSON-based metrics logging

**Features:**
- Per-epoch metrics storage
- Hyperparameter logging
- Timestamped log files
- Easy metric retrieval

### 5. **train_hdf5.py**
- Main training script with YAML configuration

## Usage

### Basic Training

```bash
/home/geon/micromamba/envs/bidir/bin/python python/learn/train_hdf5.py \
    --config python/learn/config_hdf5.yaml \
    --hdf5 path/to/rollout_data.h5 \
    --output output/experiment_name
```

### Configuration

Edit `config_hdf5.yaml` to customize:

```yaml
data:
  in_lbl: ["gait_cadence", "gait_stride"]  # Input features from param_state
  out_lbl: ["cot_ma15"]                     # Output from attributes
  batch_size: 65536
  train_split: 1.0  # Use all data (no validation split)

model:
  residual: true      # Use residual connections
  layers: [64, 32]    # Hidden layer sizes
  activation: 'gelu'  # Activation function

trainer:
  max_epochs: 10002
  lr: 0.1
  min_lr: 0.0005
  recon_type: 'huber'  # Loss type
  recon_weight: 10
  grad_weight: 0.01
  l1_weight: 0.001
  l2_weight: 0.001
  log_period: 50       # Log every N epochs
  save_period: 1000    # Save checkpoint every N epochs
```

### Testing Data Loading

```bash
/home/geon/micromamba/envs/bidir/bin/python python/learn/test_hdf5_loading.py \
    --hdf5 path/to/rollout_data.h5
```

## HDF5 Structure

Expected HDF5 file structure:

```
rollout_data.h5
├── metadata (Group)
│   └── Attributes: checkpoint_path, commit_hash, etc.
├── parameter_names (Dataset)
│   └── Array of parameter names [gait_stride, gait_cadence, ...]
└── param_N (Groups, N=0,1,2,...)
    ├── Attributes:
    │   ├── success: bool
    │   ├── metabolic/cot/MA/mean: float (output)
    │   ├── metabolic/cot/MA/std: float
    │   └── ...
    └── param_state (Dataset)
        └── Array of parameter values [shape: (13,)]
```

## Feature Mapping

**Inputs** (from `param_state` using `parameter_names` indices):
- `gait_cadence` → parameter_names[1]
- `gait_stride` → parameter_names[0]

**Outputs** (from attributes):
- `cot_ma15` → `metabolic/cot/MA/mean` attribute

## Output Structure

```
output/
├── checkpoints/
│   ├── ep_000000.ckpt
│   ├── ep_001000.ckpt
│   └── ...
├── logs/
│   ├── regression_hdf5_TIMESTAMP.json
│   └── regression_hdf5_TIMESTAMP_hparams.json
└── config.yaml (saved configuration)
```

## Checkpoint Format

Each checkpoint contains:
- `epoch`: Training epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `config`: Model configuration (architecture, normalization stats)

## Loading Checkpoints

```python
from network import RegressionNet
import torch

checkpoint = torch.load('output/checkpoints/ep_001000.ckpt')
config = checkpoint['config']

# Reconstruct model
model = RegressionNet(
    input_dim=config['input_dim'],
    output_dim=config['output_dim'],
    layers=config['layers'],
    input_mean=torch.tensor(config['input_mean']),
    input_std=torch.tensor(config['input_std']),
    target_mean=torch.tensor(config['target_mean']),
    target_std=torch.tensor(config['target_std']),
    residual=config['residual']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Dependencies

- PyTorch
- h5py
- numpy
- scikit-learn
- PyYAML
- tqdm

## Implementation Notes

### No Train/Val Split
- By default, uses all data for both training and validation
- Suitable for small datasets or when external validation is performed
- Modify `HDF5DataModule.setup()` to enable splitting if needed

### Single Sample Handling
- Gracefully handles HDF5 files with only one param_N group
- Uses biased std estimator to avoid NaN with single samples
- Replaces NaN/zero std with 1.0 to prevent normalization errors

### Loss Function Options
- **MSE**: Standard mean squared error
- **Huber**: Robust loss, less sensitive to outliers (default)
- **Cauchy, Welsch, Geman-McClure, Tukey**: Advanced robust losses with different outlier handling characteristics

### Gradient Penalty
- Encourages smooth predictions by penalizing large input gradients
- Helps prevent overfitting and improves generalization
- Configurable via `grad_weight` parameter

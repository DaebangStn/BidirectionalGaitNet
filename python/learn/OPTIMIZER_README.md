# Optimizer for Regression MLP

Gradient-based optimizer for finding input parameters that minimize or maximize model outputs.

## Overview

The `Optimizer` class uses scipy's SLSQP algorithm with gradients computed via PyTorch autograd to find optimal input parameters for a trained regression model.

## Features

- **Gradient-based optimization**: Uses automatic differentiation for accurate gradients
- **Multiple trials**: Runs multiple random initializations to find global optimum
- **Fixed parameters**: Support for optimizing subset of parameters while fixing others
- **Minimize or maximize**: Can find both minimum and maximum outputs
- **Denormalized results**: Returns results in original (denormalized) units
- **Bounds from training data**: Automatically computes valid parameter ranges

## Installation

Required dependencies:
```bash
pip install torch numpy scipy h5py pyyaml
```

## Usage

### Basic Minimization

```python
from optimizer import Optimizer

# Load trained model checkpoint
opt = Optimizer('path/to/checkpoint.ckpt', maximize=False, trials=20)

# Find inputs that minimize output
best_params, best_value = opt.run('metabolic/cot/MA/mean')

print(f"Minimum value: {best_value}")
print(f"Optimal parameters: {best_params}")
```

### Optimization with Fixed Parameters

```python
# Fix cadence, optimize stride
best_params, best_value = opt.run(
    'metabolic/cot/MA/mean',
    fixed_fields={'gait_cadence': 1.0}
)
```

### Maximization

```python
# Find maximum (worst case)
opt = Optimizer('path/to/checkpoint.ckpt', maximize=True, trials=20)
best_params, best_value = opt.run('metabolic/cot/MA/mean')
```

## API Reference

### Optimizer Class

```python
Optimizer(ckpt_path, maximize=False, trials=20)
```

**Parameters:**
- `ckpt_path` (str): Path to trained model checkpoint
- `maximize` (bool): If True, maximize output; if False, minimize
- `trials` (int): Number of random initializations to try

**Attributes:**
- `model`: Loaded RegressionNet model
- `config`: Training configuration
- `in_lbl`: List of input parameter names
- `out_lbl`: List of output parameter names
- `param_bounds`: Dict mapping parameter names to (min, max) tuples
- `device`: PyTorch device (cuda/cpu)

### run Method

```python
run(out_field, fixed_fields=None)
```

Find input parameters that optimize the specified output.

**Parameters:**
- `out_field` (str): Name of output parameter to optimize
- `fixed_fields` (dict, optional): Dict of parameter_name -> value for fixed inputs

**Returns:**
- `best_params` (dict): Optimized parameters (all inputs, denormalized)
- `best_value` (float): Optimal output value (denormalized)

## Implementation Details

### Optimization Algorithm

- **Method**: SLSQP (Sequential Least Squares Programming)
- **Gradients**: Computed via PyTorch autograd
- **Bounds**: From training data min/max values
- **Initialization**: Random uniform within bounds
- **Stopping**: ftol=1e-6, maxiter=200

### Normalization

The optimizer handles normalization internally:
1. Input bounds are computed in denormalized space
2. Fixed fields are provided in denormalized space
3. Optimization runs in normalized space
4. Results are denormalized before returning

### Multiple Trials

The optimizer runs multiple trials with different random initializations:
- Helps find global optimum (not local minimum)
- Default: 20 trials
- Returns best result across all successful trials

## Examples

See the example scripts:
- `example_optimizer.py`: Basic usage examples
- `test_optimizer.py`: Comprehensive test suite

### Example 1: Minimize Metabolic Cost

```bash
python example_optimizer.py
```

Output:
```
Minimum metabolic cost: 67.6363
Optimal gait parameters:
  Cadence: 0.8661 Hz
  Stride:  0.8552 m
```

### Example 2: Fixed Cadence Optimization

```python
opt.run('metabolic/cot/MA/mean', fixed_fields={'gait_cadence': 1.0})
```

### Example 3: Find Worst Case

```python
opt = Optimizer(ckpt_path, maximize=True, trials=20)
best_params, best_value = opt.run('metabolic/cot/MA/mean')
```

## Testing

Run the test suite:
```bash
python test_optimizer.py --ckpt path/to/checkpoint.ckpt
```

Tests include:
1. Gradient computation verification
2. Basic optimization (minimize)
3. Fixed field constraints
4. Maximization mode
5. Numerical gradient checking

## Comparison with Reference Implementation

### Similarities
- Multiple trials with random initialization
- Gradient-based optimization
- Support for fixed fields
- Careful normalization/denormalization

### Differences
- Uses scipy.minimize instead of Adam optimizer
- Single optimization (no batching in this version)
- No regularization terms
- No constraint functions (yet)
- Simpler initialization strategy

## Future Extensions

Potential additions for batched version:
- Batched optimization (optimize multiple scenarios in parallel)
- Regularization terms (L1/L2, gradient penalty)
- Constraint functions (e.g., velocity = cadence × stride)
- Custom initializers
- Progress visualization

## Troubleshooting

### FileNotFoundError for data file
- Ensure checkpoint was trained with correct config
- Check that HDF5 data file exists in expected location
- Verify config.yaml is in same directory as checkpoint

### Optimization fails in all trials
- Check parameter bounds are reasonable
- Verify model is properly trained
- Try increasing number of trials
- Check if output is well-behaved in parameter space

### Gradients are zero or very small
- Model may not be sensitive to inputs
- Check model is in eval mode (not training mode)
- Verify input normalization is correct

## File Structure

```
python/learn/
├── optimizer.py             # Main optimizer implementation
├── test_optimizer.py        # Comprehensive test suite
├── example_optimizer.py     # Usage examples
├── OPTIMIZER_README.md      # This file
├── network.py               # RegressionNet model
├── dataset.py               # Data loading
└── trainer.py               # Training code
```

## References

- Reference implementations: `python/learn/prev/optimizer.py`, `batched_optimizer.py`
- scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- PyTorch autograd: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

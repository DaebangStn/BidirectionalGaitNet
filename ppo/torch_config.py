"""
PyTorch Threading Configuration

CRITICAL: This module MUST be imported before any other torch imports.

Purpose: Prevent thread oversubscription in multi-environment parallel training.

Problem:
    BatchRolloutEnv uses ThreadPool for environment-level parallelism (hardware_concurrency threads).
    Without this configuration, each libtorch operation spawns OMP_NUM_THREADS worker threads,
    leading to thread explosion: num_envs × OMP_NUM_THREADS (e.g., 16 × 64 = 1024 threads!).
    This causes severe performance degradation and resource contention.

Solution:
    - Set OMP/MKL environment variables to '1' before torch import
    - Configure PyTorch internal threading to single-threaded mode
    - Environment-level parallelism (ThreadPool) + single-threaded operations = optimal performance

Usage:
    import ppo.torch_config  # Must be first import in training/benchmark scripts
    import torch  # Now safe to import
    # ... rest of code
"""

import os

# MUST be set before torch import
os.environ.setdefault("OMP_NUM_THREADS", '1')
os.environ.setdefault("MKL_NUM_THREADS", '1')

import torch

# Configure PyTorch's internal threading limits
# Use try-except to handle cases where threading is already configured (e.g., when imported as module)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Threading already configured, ignore
    pass

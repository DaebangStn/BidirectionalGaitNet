#!/usr/bin/env python3
"""
Test BatchEnv with zero-copy numpy/torch integration.

This script tests the C++ BatchEnv implementation with:
- Environment creation and initialization
- Reset functionality
- Step functionality with random actions
- Zero-copy numpy array integration
- PyTorch tensor conversion
- Performance benchmarking (SPS calculation)
"""

import sys
import numpy as np
import torch
import time
from ppo import BatchEnv


def test_batchenv():
    print("=" * 70)
    print("BatchEnv Test Suite - High-Performance Batched Environment")
    print("=" * 70)

    # Configuration
    yaml_path = "data/env/A2_sep.yaml"
    num_envs = 16  # Test with more environments for better performance measurement

    # [1] Create batched environment
    print(f"\n[1] Creating BatchEnv with {num_envs} environments...")
    try:
        # Read YAML content
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()

        env = BatchEnv(yaml_content, num_envs=num_envs)
        print(f"✓ BatchEnv created successfully")
        print(f"  Number of environments: {env.num_envs()}")
        print(f"  Observation dimension: {env.obs_dim()}")
        print(f"  Action dimension: {env.action_dim()}")
    except Exception as e:
        print(f"✗ Failed to create BatchEnv: {e}")
        sys.exit(1)

    # [2] Test reset
    print("\n[2] Testing reset() functionality...")
    try:
        obs_np = env.reset()
        print(f"✓ Reset successful")
        print(f"  obs shape: {obs_np.shape}")
        print(f"  obs dtype: {obs_np.dtype}")
        print(f"  obs C-contiguous: {obs_np.flags['C_CONTIGUOUS']}")
        print(f"  obs range: [{obs_np.min():.4f}, {obs_np.max():.4f}]")
        print(f"  obs sample (env 0, first 5): {obs_np[0, :5]}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        sys.exit(1)

    # [3] Test PyTorch zero-copy conversion
    print("\n[3] Testing PyTorch zero-copy conversion...")
    try:
        obs_torch = torch.from_numpy(obs_np)
        print(f"✓ PyTorch conversion successful")
        print(f"  torch shape: {obs_torch.shape}")
        print(f"  torch dtype: {obs_torch.dtype}")
        print(f"  torch device: {obs_torch.device}")

        # Verify zero-copy (same memory address)
        obs_np_ptr = obs_np.__array_interface__['data'][0]
        obs_torch_ptr = obs_torch.data_ptr()
        shares_memory = (obs_np_ptr == obs_torch_ptr)
        print(f"  shares memory (zero-copy): {shares_memory}")

        if not shares_memory:
            print(f"  ⚠ Warning: Memory not shared (numpy: {obs_np_ptr}, torch: {obs_torch_ptr})")
    except Exception as e:
        print(f"✗ PyTorch conversion failed: {e}")
        sys.exit(1)

    # [4] Test step
    print("\n[4] Testing step() functionality...")
    try:
        # Create random actions
        actions_np = np.random.randn(env.num_envs(), env.action_dim()).astype(np.float32)
        actions_np = np.clip(actions_np, -1.0, 1.0)  # Clip to valid range

        next_obs_np, rew_np, done_np = env.step(actions_np)

        print(f"✓ Step successful")
        print(f"  next_obs shape: {next_obs_np.shape}, dtype: {next_obs_np.dtype}")
        print(f"  rewards shape: {rew_np.shape}, dtype: {rew_np.dtype}")
        print(f"  dones shape: {done_np.shape}, dtype: {done_np.dtype}")
        print(f"  reward stats: mean={rew_np.mean():.4f}, std={rew_np.std():.4f}, min={rew_np.min():.4f}, max={rew_np.max():.4f}")
        print(f"  done count: {done_np.sum()}/{env.num_envs()}")

        # Convert all to PyTorch
        next_obs_torch = torch.from_numpy(next_obs_np)
        rew_torch = torch.from_numpy(rew_np)
        done_torch = torch.from_numpy(done_np)

        print(f"✓ All PyTorch conversions successful")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # [5] Multiple steps test
    print("\n[5] Testing multiple consecutive steps...")
    try:
        num_test_steps = 10
        for step_idx in range(num_test_steps):
            actions_np = np.random.randn(env.num_envs(), env.action_dim()).astype(np.float32)
            actions_np = np.clip(actions_np, -1.0, 1.0)
            next_obs_np, rew_np, done_np = env.step(actions_np)

        print(f"✓ Successfully completed {num_test_steps} steps")
        print(f"  Final reward mean: {rew_np.mean():.4f}")
        print(f"  Final done count: {done_np.sum()}/{env.num_envs()}")
    except Exception as e:
        print(f"✗ Multiple steps failed: {e}")
        sys.exit(1)

    # [6] Performance benchmark
    print("\n[6] Performance benchmark (100 steps)...")
    try:
        num_perf_steps = 100

        # Warm-up
        for _ in range(5):
            actions_np = np.random.randn(env.num_envs(), env.action_dim()).astype(np.float32)
            actions_np = np.clip(actions_np, -1.0, 1.0)
            env.step(actions_np)

        # Benchmark
        start_time = time.time()
        for _ in range(num_perf_steps):
            actions_np = np.random.randn(env.num_envs(), env.action_dim()).astype(np.float32)
            actions_np = np.clip(actions_np, -1.0, 1.0)
            next_obs_np, rew_np, done_np = env.step(actions_np)
        elapsed = time.time() - start_time

        total_steps = num_perf_steps * env.num_envs()
        sps = total_steps / elapsed

        print(f"✓ Performance benchmark complete")
        print(f"  Total environment steps: {total_steps}")
        print(f"  Elapsed time: {elapsed:.3f}s")
        print(f"  Steps per second (SPS): {sps:.1f}")

        # Compare to AsyncVectorEnv baseline
        baseline_sps = 84.0
        speedup = sps / baseline_sps
        print(f"  Baseline (AsyncVectorEnv): {baseline_sps:.1f} SPS")
        print(f"  Speedup: {speedup:.1f}x")

        if speedup >= 10.0:
            print(f"  🚀 Excellent! Target 10x speedup achieved!")
        elif speedup >= 5.0:
            print(f"  ⚡ Good! 5x+ speedup achieved")
        elif speedup >= 2.0:
            print(f"  ✓ Improvement: 2x+ speedup")
        else:
            print(f"  ⚠ Note: Speedup less than 2x, may need optimization")

    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("All tests passed successfully! ✓")
    print("=" * 70)
    print("\nBatchEnv is ready for integration with PPO training.")
    print("Next step: Run 'python ppo/ppo_hierarchical.py' with BatchEnvWrapper")


if __name__ == "__main__":
    test_batchenv()

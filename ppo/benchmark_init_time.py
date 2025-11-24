#!/usr/bin/env python3
"""
Benchmark environment initialization time with parallel vs sequential creation.

Tests BatchRolloutEnv and BatchEnv initialization times for different num_envs.
"""

import sys
import time
from pathlib import Path

# Configure threading before torch import (prevents thread oversubscription)
import ppo.torch_config
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def benchmark_init_time(num_envs_list, env_name="A2_sep"):
    """Benchmark initialization time for different num_envs."""

    yaml_path = Path(__file__).parent.parent / "data" / "env" / f"{env_name}.yaml"
    with open(yaml_path) as f:
        yaml_content = f.read()

    print("="*80)
    print("ENVIRONMENT INITIALIZATION BENCHMARK")
    print("="*80)
    print(f"\nEnvironment: {env_name}.yaml")
    print(f"Testing num_envs: {num_envs_list}\n")

    results = []

    for num_envs in num_envs_list:
        print(f"Testing num_envs={num_envs}...", end=" ", flush=True)

        # Import here to avoid reusing cached environments
        import sys
        sys.path.insert(0, 'ppo')
        from batchrolloutenv import BatchRolloutEnv

        # Benchmark BatchRolloutEnv initialization
        start_time = time.perf_counter()
        env = BatchRolloutEnv(yaml_content, num_envs, 64)  # 64 steps
        init_time = (time.perf_counter() - start_time) * 1000  # ms

        print(f"{init_time:.0f}ms")

        results.append({
            'num_envs': num_envs,
            'init_time_ms': init_time
        })

        # Clean up
        del env

    # Print results table
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n{'Num Envs':<10} {'Init Time (ms)':<20} {'Time per Env (ms)':<20} {'Speedup':<10}")
    print("-"*60)

    baseline_time_per_env = None
    for r in results:
        time_per_env = r['init_time_ms'] / r['num_envs']

        if baseline_time_per_env is None:
            baseline_time_per_env = time_per_env
            speedup = 1.0
        else:
            # Speedup in initialization efficiency (lower time per env = better parallelization)
            speedup = baseline_time_per_env / time_per_env

        print(f"{r['num_envs']:<10} {r['init_time_ms']:<20.0f} {time_per_env:<20.1f} {speedup:<10.2f}x")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if len(results) >= 2:
        # Compare smallest vs largest
        small = results[0]
        large = results[-1]

        sequential_estimate = (small['init_time_ms'] / small['num_envs']) * large['num_envs']
        actual_time = large['init_time_ms']
        parallel_speedup = sequential_estimate / actual_time

        print(f"\nIf environments were initialized SEQUENTIALLY:")
        print(f"  {large['num_envs']} envs would take: ~{sequential_estimate:.0f}ms")
        print(f"\nWith PARALLEL initialization:")
        print(f"  {large['num_envs']} envs actually took: {actual_time:.0f}ms")
        print(f"\nParallel initialization speedup: {parallel_speedup:.2f}x faster!")

    print("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs-list", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16, 32],
                       help="List of num_envs to test")
    parser.add_argument("--env", type=str, default="A2_sep",
                       help="Environment name")
    args = parser.parse_args()

    benchmark_init_time(args.num_envs_list, args.env)

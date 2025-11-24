#!/usr/bin/env python3
"""
Benchmark weight synchronization time for muscle networks.

Measures the performance improvement from parallel weight updates in
update_muscle_weights() for both BatchEnv and BatchRolloutEnv.
"""

import sys
import time
from pathlib import Path

# Configure threading before torch import (prevents thread oversubscription)
import ppo.torch_config
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def benchmark_weight_sync(num_envs_list, env_name="A2_sep", num_iterations=10):
    """Benchmark muscle weight synchronization time."""

    yaml_path = Path(__file__).parent.parent / "data" / "env" / f"{env_name}.yaml"
    with open(yaml_path) as f:
        yaml_content = f.read()

    print("="*80)
    print("MUSCLE WEIGHT SYNCHRONIZATION BENCHMARK")
    print("="*80)
    print(f"\nEnvironment: {env_name}.yaml")
    print(f"Iterations per test: {num_iterations}")
    print(f"Testing num_envs: {num_envs_list}\n")

    results = []

    for num_envs in num_envs_list:
        print(f"Testing num_envs={num_envs}...", end=" ", flush=True)

        # Import BatchRolloutEnv
        sys.path.insert(0, 'ppo')
        from batchrolloutenv import BatchRolloutEnv
        from ppo_rollout_learner import Agent

        # Create environment and policy
        env = BatchRolloutEnv(yaml_content, num_envs, 64)

        # Check if hierarchical (has muscle networks)
        if not env.is_hierarchical():
            print("SKIPPED (not hierarchical)")
            del env
            continue

        # Create dummy policy and muscle network weights
        from muscle_learner import MuscleLearner

        muscle_learner = MuscleLearner(
            num_actuator_action=env.getNumActuatorAction(),
            num_muscles=env.getNumMuscles(),
            num_muscle_dofs=env.getNumMuscleDof(),
            learning_rate=1e-4,
            num_epochs=1,
            batch_size=128,
            is_cascaded=env.use_cascading(),
            device='cpu'
        )

        # Get muscle network state dict
        muscle_state_dict = muscle_learner.model.state_dict()

        # Warmup: do a few sync operations
        for _ in range(3):
            env.update_muscle_weights(muscle_state_dict)

        # Benchmark: measure sync time
        sync_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            env.update_muscle_weights(muscle_state_dict)
            sync_time = (time.perf_counter() - start_time) * 1000  # ms
            sync_times.append(sync_time)

        avg_sync_time = np.mean(sync_times)
        std_sync_time = np.std(sync_times)
        min_sync_time = np.min(sync_times)
        max_sync_time = np.max(sync_times)

        print(f"{avg_sync_time:.2f}ms (±{std_sync_time:.2f}ms)")

        results.append({
            'num_envs': num_envs,
            'avg_sync_time_ms': avg_sync_time,
            'std_sync_time_ms': std_sync_time,
            'min_sync_time_ms': min_sync_time,
            'max_sync_time_ms': max_sync_time
        })

        # Clean up
        del env
        del muscle_learner

    # Print results table
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n{'Num Envs':<10} {'Avg Sync (ms)':<15} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Time per Env (ms)':<20}")
    print("-"*90)

    for r in results:
        time_per_env = r['avg_sync_time_ms'] / r['num_envs']
        print(f"{r['num_envs']:<10} {r['avg_sync_time_ms']:<15.2f} {r['std_sync_time_ms']:<12.2f} "
              f"{r['min_sync_time_ms']:<12.2f} {r['max_sync_time_ms']:<12.2f} {time_per_env:<20.3f}")

    # Speedup analysis
    if len(results) >= 2:
        print("\n" + "="*80)
        print("PARALLEL SPEEDUP ANALYSIS")
        print("="*80)

        baseline = results[0]
        baseline_time_per_env = baseline['avg_sync_time_ms'] / baseline['num_envs']

        print(f"\n{'Num Envs':<10} {'Sync Time (ms)':<15} {'Sequential Est (ms)':<20} {'Speedup':<10}")
        print("-"*55)

        for r in results:
            sequential_estimate = baseline_time_per_env * r['num_envs']
            actual_time = r['avg_sync_time_ms']
            speedup = sequential_estimate / actual_time if actual_time > 0 else 0

            print(f"{r['num_envs']:<10} {actual_time:<15.2f} {sequential_estimate:<20.0f} {speedup:<10.2f}x")

        # Compare largest vs smallest
        large = results[-1]
        sequential_est_large = baseline_time_per_env * large['num_envs']
        parallel_speedup = sequential_est_large / large['avg_sync_time_ms']

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\nFor {large['num_envs']} environments:")
        print(f"  Sequential estimate: {sequential_est_large:.0f}ms")
        print(f"  Parallel actual:     {large['avg_sync_time_ms']:.0f}ms")
        print(f"  **Parallel speedup:  {parallel_speedup:.2f}x faster!**")

        # Time saved per sync
        time_saved = sequential_est_large - large['avg_sync_time_ms']
        print(f"\n  Time saved per weight sync: {time_saved:.0f}ms")
        print(f"  Percentage reduction: {(time_saved / sequential_est_large * 100):.1f}%")

    print("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs-list", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16],
                       help="List of num_envs to test")
    parser.add_argument("--env", type=str, default="A2_sep",
                       help="Environment name")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations per test")
    args = parser.parse_args()

    benchmark_weight_sync(args.num_envs_list, args.env, args.iterations)

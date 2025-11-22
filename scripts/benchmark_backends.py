#!/usr/bin/env python3
"""
Benchmark script to compare AsyncVectorEnv vs BatchEnv performance.

Usage:
    python scripts/benchmark_backends.py --env-file data/env/A2_sep.yaml --num-envs 32 --num-steps 1000
"""

# CRITICAL: Set threading BEFORE any imports that might load torch/libtorch
# This prevents nested parallelism: BatchEnv uses ThreadPool for environment-level parallelism
# Setting OMP/MKL to 1 ensures each libtorch operation runs single-threaded within its thread
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import argparse
import numpy as np

# Add ppo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ppo'))

import gymnasium as gym
from ppo.env_wrapper import make_env
from ppo.batch_env_wrapper import make_batch_env


def benchmark_async_vector_env(env_file: str, num_envs: int, num_steps: int, warmup_steps: int = 10):
    """
    Benchmark AsyncVectorEnv performance.

    Args:
        env_file: Path to environment configuration file
        num_envs: Number of parallel environments
        num_steps: Number of steps to benchmark
        warmup_steps: Number of warmup steps before timing

    Returns:
        Tuple of (steps_per_second, total_time, avg_step_time)
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking AsyncVectorEnv: {num_envs} environments, {num_steps} steps")
    print(f"{'='*80}")

    # Create AsyncVectorEnv
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_file, i) for i in range(num_envs)],
        shared_memory=True,
        context='spawn',
        autoreset_mode=gym.vector.AutoresetMode.DISABLED
    )

    # Get action dimension
    action_dim = envs.single_action_space.shape[0]

    # Reset environments
    print("Resetting environments...")
    obs, _ = envs.reset()
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Warmup
    print(f"Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        actions = np.random.randn(num_envs, action_dim).astype(np.float32)
        obs, rewards, terminations, truncations, infos = envs.step(actions)

    # Benchmark
    print(f"Running benchmark ({num_steps} steps)...")
    step_times = []
    start_time = time.time()

    for i in range(num_steps):
        actions = np.random.randn(num_envs, action_dim).astype(np.float32)

        step_start = time.time()
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        step_time = time.time() - step_start
        step_times.append(step_time)

        if (i + 1) % 100 == 0:
            current_sps = num_envs * (i + 1) / (time.time() - start_time)
            print(f"  Step {i+1}/{num_steps}: {current_sps:.1f} SPS")

    total_time = time.time() - start_time
    total_steps = num_envs * num_steps
    sps = total_steps / total_time
    avg_step_time = np.mean(step_times)

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total env-steps: {total_steps}")
    print(f"  Steps per second (SPS): {sps:.1f}")
    print(f"  Average step time: {avg_step_time*1000:.2f}ms")
    print(f"  Min step time: {np.min(step_times)*1000:.2f}ms")
    print(f"  Max step time: {np.max(step_times)*1000:.2f}ms")

    envs.close()

    return sps, total_time, avg_step_time


def benchmark_batch_env(env_file: str, num_envs: int, num_steps: int, warmup_steps: int = 10):
    """
    Benchmark BatchEnv (C++ parallel) performance.

    Args:
        env_file: Path to environment configuration file
        num_envs: Number of parallel environments
        num_steps: Number of steps to benchmark
        warmup_steps: Number of warmup steps before timing

    Returns:
        Tuple of (steps_per_second, total_time, avg_step_time)
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking BatchEnv (C++ ThreadPool): {num_envs} environments, {num_steps} steps")
    print(f"{'='*80}")

    # Create BatchEnv
    try:
        envs = make_batch_env(env_file, num_envs)
    except ImportError as e:
        print(f"ERROR: Failed to import BatchEnv: {e}")
        print("Make sure batchenv.so is built with: ninja -C build/release batchenv.so")
        sys.exit(1)

    # Get action dimension
    action_dim = envs.single_action_space.shape[0]

    # Reset environments
    print("Resetting environments...")
    obs, _ = envs.reset()
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Warmup
    print(f"Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        actions = np.random.randn(num_envs, action_dim).astype(np.float32)
        obs, rewards, terminations, truncations, infos = envs.step(actions)

    # Benchmark
    print(f"Running benchmark ({num_steps} steps)...")
    step_times = []
    start_time = time.time()

    for i in range(num_steps):
        actions = np.random.randn(num_envs, action_dim).astype(np.float32)

        step_start = time.time()
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        step_time = time.time() - step_start
        step_times.append(step_time)

        if (i + 1) % 100 == 0:
            current_sps = num_envs * (i + 1) / (time.time() - start_time)
            print(f"  Step {i+1}/{num_steps}: {current_sps:.1f} SPS")

    total_time = time.time() - start_time
    total_steps = num_envs * num_steps
    sps = total_steps / total_time
    avg_step_time = np.mean(step_times)

    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total env-steps: {total_steps}")
    print(f"  Steps per second (SPS): {sps:.1f}")
    print(f"  Average step time: {avg_step_time*1000:.2f}ms")
    print(f"  Min step time: {np.min(step_times)*1000:.2f}ms")
    print(f"  Max step time: {np.max(step_times)*1000:.2f}ms")

    envs.close()

    return sps, total_time, avg_step_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark AsyncVectorEnv vs BatchEnv performance")
    parser.add_argument("--env-file", type=str, default="data/env/A2_sep.yaml",
                        help="Path to environment configuration file")
    parser.add_argument("--num-envs", type=int, nargs='+', default=[32],
                        help="Number of parallel environments (can specify multiple: --num-envs 4 8 16 32)")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Number of steps to benchmark")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Number of warmup steps")
    parser.add_argument("--backend", type=str, choices=["async", "batch", "both"], default="both",
                        help="Which backend to benchmark (default: both)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Backend Performance Benchmark")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Environment file: {args.env_file}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Number of steps: {args.num_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Backend: {args.backend}")

    # Store results for all environment counts
    all_results = {}

    # Run benchmarks for each environment count
    for num_envs in args.num_envs:
        print(f"\n{'='*80}")
        print(f"Testing with {num_envs} environments")
        print(f"{'='*80}")

        results = {}

        # Benchmark AsyncVectorEnv
        if args.backend in ["async", "both"]:
            try:
                async_sps, async_time, async_step_time = benchmark_async_vector_env(
                    args.env_file, num_envs, args.num_steps, args.warmup_steps
                )
                results["AsyncVectorEnv"] = {
                    "sps": async_sps,
                    "total_time": async_time,
                    "avg_step_time": async_step_time
                }
            except Exception as e:
                print(f"ERROR in AsyncVectorEnv benchmark: {e}")
                import traceback
                traceback.print_exc()

        # Benchmark BatchEnv
        if args.backend in ["batch", "both"]:
            try:
                batch_sps, batch_time, batch_step_time = benchmark_batch_env(
                    args.env_file, num_envs, args.num_steps, args.warmup_steps
                )
                results["BatchEnv"] = {
                    "sps": batch_sps,
                    "total_time": batch_time,
                    "avg_step_time": batch_step_time
                }
            except Exception as e:
                print(f"ERROR in BatchEnv benchmark: {e}")
                import traceback
                traceback.print_exc()

        # Store results
        all_results[num_envs] = results

        # Print comparison for this environment count
        if len(results) == 2:
            print(f"\n{'='*80}")
            print(f"Performance Comparison ({num_envs} environments)")
            print(f"{'='*80}")

            async_sps = results["AsyncVectorEnv"]["sps"]
            batch_sps = results["BatchEnv"]["sps"]
            speedup = batch_sps / async_sps

            print(f"\nSteps per Second (SPS):")
            print(f"  AsyncVectorEnv: {async_sps:.1f} SPS")
            print(f"  BatchEnv:       {batch_sps:.1f} SPS")
            print(f"  Speedup:        {speedup:.2f}x")

            async_step = results["AsyncVectorEnv"]["avg_step_time"]
            batch_step = results["BatchEnv"]["avg_step_time"]

            print(f"\nAverage Step Time:")
            print(f"  AsyncVectorEnv: {async_step*1000:.2f}ms")
            print(f"  BatchEnv:       {batch_step*1000:.2f}ms")
            print(f"  Improvement:    {(1 - batch_step/async_step)*100:.1f}%")

            print(f"\n{'='*80}")
            if speedup > 1.5:
                print(f"✅ BatchEnv is significantly faster ({speedup:.2f}x speedup)!")
            elif speedup > 1.1:
                print(f"✅ BatchEnv is faster ({speedup:.2f}x speedup)")
            elif speedup > 0.95:
                print(f"⚖️  Performance is comparable ({speedup:.2f}x)")
            else:
                print(f"⚠️  AsyncVectorEnv is faster (BatchEnv is {1/speedup:.2f}x slower)")
            print(f"{'='*80}\n")

    # Print summary table if multiple environment counts tested
    if len(args.num_envs) > 1 and args.backend == "both":
        print(f"\n{'='*80}")
        print(f"Summary Table")
        print(f"{'='*80}\n")

        # Table header
        print(f"{'Envs':>6} | {'Async SPS':>10} | {'Batch SPS':>10} | {'Speedup':>8} | {'Winner':>15}")
        print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*15}")

        # Table rows
        for num_envs in sorted(args.num_envs):
            if num_envs in all_results and len(all_results[num_envs]) == 2:
                async_sps = all_results[num_envs]["AsyncVectorEnv"]["sps"]
                batch_sps = all_results[num_envs]["BatchEnv"]["sps"]
                speedup = batch_sps / async_sps

                if speedup > 1.05:
                    winner = "BatchEnv ✅"
                elif speedup < 0.95:
                    winner = "AsyncVectorEnv ⚠️"
                else:
                    winner = "Comparable ⚖️"

                print(f"{num_envs:6d} | {async_sps:10.1f} | {batch_sps:10.1f} | {speedup:7.2f}x | {winner:>15}")

        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

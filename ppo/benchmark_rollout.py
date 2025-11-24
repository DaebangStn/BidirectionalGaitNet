#!/usr/bin/env python3
"""
Benchmark: BatchEnv vs BatchRolloutEnv Performance Comparison

Compares learning performance between:
- BatchEnv: Python policy inference (ppo_hierarchical.py architecture)
- BatchRolloutEnv: C++ policy inference with autonomous rollout

Metrics:
- Samples Per Second (SPS)
- Rollout time
- Learning time
- Total iteration time
- Memory usage
"""

import argparse
import time
import psutil
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Configure threading before torch import (prevents thread oversubscription)
import ppo.torch_config
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    num_envs: int
    num_steps: int
    total_timesteps: int

    # Timing metrics (milliseconds)
    rollout_time_ms: float
    learning_time_ms: float
    total_time_ms: float

    # Performance metrics
    samples_per_second: float
    iterations: int

    # Memory metrics (MB)
    memory_used_mb: float
    peak_memory_mb: float


class PerformanceMonitor:
    """Monitor performance metrics during benchmark."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, mem_mb)
        return mem_mb

    def reset_peak(self):
        """Reset peak memory counter."""
        self.peak_memory_mb = self.get_memory_mb()


def benchmark_batchenv(args) -> BenchmarkResult:
    """Benchmark BatchEnv with Python policy inference."""
    print("\n" + "="*80)
    print("BENCHMARK: BatchEnv (Python Policy Inference)")
    print("="*80)

    import batchenv
    from muscle_learner import MuscleLearner

    monitor = PerformanceMonitor()
    monitor.reset_peak()

    # Create environment
    print(f"Creating BatchEnv: {args.num_envs} envs")
    yaml_path = Path(__file__).parent.parent / "data" / "env" / f"{args.env}.yaml"
    with open(yaml_path) as f:
        yaml_content = f.read()

    envs = batchenv.BatchEnv(yaml_content, args.num_envs)
    obs_dim = envs.obs_dim()
    action_dim = envs.action_dim()

    print(f"Environment: {args.env}.yaml")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Create policy agent (simple PPO without full training)
    from ppo_hierarchical import Agent
    # Force CUDA for all learning (policy + muscle), CPU for rollout only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Policy device: {device}")

    # ppo_hierarchical.Agent expects env with .single_observation_space/.single_action_space
    # Create wrapper to provide gym-style attributes
    class GymStyleEnvWrapper:
        def __init__(self, obs_dim, action_dim):
            from gymnasium.spaces import Box
            self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.single_action_space = Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)

    wrapped_env = GymStyleEnvWrapper(obs_dim, action_dim)
    agent = Agent(wrapped_env).to(device)

    # Setup muscle learner if hierarchical
    # Use same device as policy (CUDA for all learning)
    muscle_learner = None
    if envs.is_hierarchical():
        print(f"Hierarchical control: True")
        muscle_learner = MuscleLearner(
            num_actuator_action=envs.getNumActuatorAction(),
            num_muscles=envs.getNumMuscles(),
            num_muscle_dofs=envs.getNumMuscleDof(),
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=128,
            is_cascaded=envs.use_cascading(),
            device=str(device)  # Convert torch.device to string for MuscleLearner
        )

    # Benchmark loop
    num_iterations = args.total_timesteps // (args.num_envs * args.num_steps)
    batch_size = args.num_envs * args.num_steps

    print(f"\nStarting benchmark: {num_iterations} iterations")
    print(f"Batch size: {args.num_envs} envs × {args.num_steps} steps = {batch_size} samples")

    total_rollout_time = 0.0
    total_learning_time = 0.0

    obs = envs.reset()

    for iteration in range(num_iterations):
        # Rollout phase
        rollout_start = time.perf_counter()

        # Storage for rollout data
        obs_batch = np.zeros((args.num_steps, args.num_envs, obs_dim), dtype=np.float32)
        actions_batch = np.zeros((args.num_steps, args.num_envs, action_dim), dtype=np.float32)
        rewards_batch = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        values_batch = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        logprobs_batch = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        dones_batch = np.zeros((args.num_steps, args.num_envs), dtype=np.uint8)

        # Collect rollout
        for step in range(args.num_steps):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device)
                action_t, logprob_t, _, value_t = agent.get_action_and_value(obs_t)

                action_np = action_t.cpu().numpy()
                value_np = value_t.cpu().numpy().flatten()
                logprob_np = logprob_t.cpu().numpy()

            # Step environments
            next_obs, reward, done = envs.step(action_np)

            # Store data
            obs_batch[step] = obs
            actions_batch[step] = action_np
            rewards_batch[step] = reward
            values_batch[step] = value_np
            logprobs_batch[step] = logprob_np
            dones_batch[step] = done

            obs = next_obs

        rollout_time = (time.perf_counter() - rollout_start) * 1000
        total_rollout_time += rollout_time

        # Learning phase (simplified - just forward passes)
        learning_start = time.perf_counter()

        # Flatten batches
        b_obs = torch.from_numpy(obs_batch.reshape(-1, obs_dim)).to(device)
        b_actions = torch.from_numpy(actions_batch.reshape(-1, action_dim)).to(device)

        # Policy loss (just compute, don't backprop for benchmark)
        with torch.no_grad():
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)

        # Muscle learning if hierarchical
        if muscle_learner is not None:
            muscle_tuples = envs.get_muscle_tuples()
            if muscle_tuples:
                # MuscleLearner.learn() expects numpy arrays and handles device placement internally
                _ = muscle_learner.learn(muscle_tuples)

        learning_time = (time.perf_counter() - learning_start) * 1000
        total_learning_time += learning_time

        if (iteration + 1) % max(1, num_iterations // 10) == 0:
            progress = (iteration + 1) / num_iterations * 100
            print(f"Progress: {progress:.0f}% | Rollout: {rollout_time:.1f}ms | Learning: {learning_time:.1f}ms")

    total_time_ms = total_rollout_time + total_learning_time
    sps = (args.total_timesteps / total_time_ms) * 1000

    result = BenchmarkResult(
        name="BatchEnv (Python Policy)",
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        rollout_time_ms=total_rollout_time,
        learning_time_ms=total_learning_time,
        total_time_ms=total_time_ms,
        samples_per_second=sps,
        iterations=num_iterations,
        memory_used_mb=monitor.get_memory_mb(),
        peak_memory_mb=monitor.peak_memory_mb
    )

    return result


def benchmark_batchrolloutenv(args) -> BenchmarkResult:
    """Benchmark BatchRolloutEnv with C++ policy inference."""
    print("\n" + "="*80)
    print("BENCHMARK: BatchRolloutEnv (C++ Policy Inference)")
    print("="*80)

    import batchrolloutenv
    from muscle_learner import MuscleLearner

    monitor = PerformanceMonitor()
    monitor.reset_peak()

    # Create environment
    print(f"Creating BatchRolloutEnv: {args.num_envs} envs, {args.num_steps} steps")
    yaml_path = Path(__file__).parent.parent / "data" / "env" / f"{args.env}.yaml"
    with open(yaml_path) as f:
        yaml_content = f.read()

    envs = batchrolloutenv.BatchRolloutEnv(yaml_content, args.num_envs, args.num_steps)
    obs_dim = envs.obs_size()
    action_dim = envs.action_size()

    print(f"Environment: {args.env}.yaml")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Create policy agent
    from ppo_rollout_learner import Agent
    # Force CUDA for all learning (policy + muscle), CPU for rollout only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Policy device: {device}")

    # ppo_rollout_learner.Agent expects (num_states, num_actions) directly
    agent = Agent(obs_dim, action_dim).to(device)

    # Setup muscle learner if hierarchical
    # Use same device as policy (CUDA for all learning)
    muscle_learner = None
    if envs.is_hierarchical():
        print(f"Hierarchical control: True")
        muscle_learner = MuscleLearner(
            num_actuator_action=envs.getNumActuatorAction(),
            num_muscles=envs.getNumMuscles(),
            num_muscle_dofs=envs.getNumMuscleDof(),
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=128,
            is_cascaded=envs.use_cascading(),
            device=str(device)  # Convert torch.device to string for MuscleLearner
        )

    # Synchronize initial weights to C++
    agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
    envs.update_policy_weights(agent_state_cpu)

    # Benchmark loop
    num_iterations = args.total_timesteps // (args.num_envs * args.num_steps)
    batch_size = args.num_envs * args.num_steps

    print(f"\nStarting benchmark: {num_iterations} iterations")
    print(f"Batch size: {args.num_envs} envs × {args.num_steps} steps = {batch_size} samples")

    total_rollout_time = 0.0
    total_learning_time = 0.0

    for iteration in range(num_iterations):
        # Rollout phase (autonomous C++ rollout)
        rollout_start = time.perf_counter()
        trajectory = envs.collect_rollout()
        rollout_time = (time.perf_counter() - rollout_start) * 1000
        total_rollout_time += rollout_time

        # Learning phase (simplified - just forward passes)
        learning_start = time.perf_counter()

        # Get trajectory data
        b_obs = torch.from_numpy(trajectory["obs"]).to(device)
        b_actions = torch.from_numpy(trajectory["actions"]).to(device)

        # Policy loss (just compute, don't backprop for benchmark)
        with torch.no_grad():
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)

        # Muscle learning if hierarchical
        if muscle_learner is not None:
            muscle_tuples = envs.get_muscle_tuples()
            if muscle_tuples:
                # MuscleLearner.learn() expects numpy arrays and handles device placement internally
                _ = muscle_learner.learn(muscle_tuples)

        learning_time = (time.perf_counter() - learning_start) * 1000
        total_learning_time += learning_time

        # Synchronize weights back to C++ (for next rollout)
        agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
        envs.update_policy_weights(agent_state_cpu)

        if (iteration + 1) % max(1, num_iterations // 10) == 0:
            progress = (iteration + 1) / num_iterations * 100
            print(f"Progress: {progress:.0f}% | Rollout: {rollout_time:.1f}ms | Learning: {learning_time:.1f}ms")

    total_time_ms = total_rollout_time + total_learning_time
    sps = (args.total_timesteps / total_time_ms) * 1000

    result = BenchmarkResult(
        name="BatchRolloutEnv (C++ Policy)",
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        rollout_time_ms=total_rollout_time,
        learning_time_ms=total_learning_time,
        total_time_ms=total_time_ms,
        samples_per_second=sps,
        iterations=num_iterations,
        memory_used_mb=monitor.get_memory_mb(),
        peak_memory_mb=monitor.peak_memory_mb
    )

    return result


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    # Configuration
    if results:
        r = results[0]
        print(f"\nConfiguration:")
        print(f"  Environments: {r.num_envs}")
        print(f"  Steps per rollout: {r.num_steps}")
        print(f"  Total timesteps: {r.total_timesteps}")
        print(f"  Iterations: {r.iterations}")

    # Results table
    print(f"\n{'Method':<30} {'Rollout (ms)':<15} {'Learning (ms)':<15} {'Total (ms)':<15} {'SPS':<15}")
    print("-" * 90)

    for r in results:
        print(f"{r.name:<30} {r.rollout_time_ms:>14.1f} {r.learning_time_ms:>14.1f} "
              f"{r.total_time_ms:>14.1f} {r.samples_per_second:>14.1f}")

    # Speedup comparison
    if len(results) == 2:
        print("\n" + "="*80)
        print("SPEEDUP ANALYSIS")
        print("="*80)

        baseline = results[0]
        optimized = results[1]

        rollout_speedup = baseline.rollout_time_ms / optimized.rollout_time_ms
        total_speedup = baseline.total_time_ms / optimized.total_time_ms
        sps_speedup = optimized.samples_per_second / baseline.samples_per_second

        print(f"\nRollout speedup: {rollout_speedup:.2f}x")
        print(f"Total speedup: {total_speedup:.2f}x")
        print(f"SPS improvement: {sps_speedup:.2f}x")

        rollout_reduction = (1 - optimized.rollout_time_ms / baseline.rollout_time_ms) * 100
        total_reduction = (1 - optimized.total_time_ms / baseline.total_time_ms) * 100

        print(f"\nRollout time reduction: {rollout_reduction:.1f}%")
        print(f"Total time reduction: {total_reduction:.1f}%")

    # Memory comparison
    print("\n" + "="*80)
    print("MEMORY USAGE")
    print("="*80)

    print(f"\n{'Method':<30} {'Current (MB)':<15} {'Peak (MB)':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r.name:<30} {r.memory_used_mb:>14.1f} {r.peak_memory_mb:>14.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BatchEnv vs BatchRolloutEnv")
    parser.add_argument("--env", type=str, default="A2_sep", help="Environment config name")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=64, help="Steps per rollout")
    parser.add_argument("--total-timesteps", type=int, default=1200, help="Total timesteps to benchmark")
    parser.add_argument("--only-python", action="store_true", help="Only benchmark Python policy")
    parser.add_argument("--only-cpp", action="store_true", help="Only benchmark C++ policy")
    parser.add_argument("--warmup", action="store_true", help="Run warmup iterations first")

    args = parser.parse_args()

    print("="*80)
    print("PPO ROLLOUT BENCHMARK")
    print("="*80)
    print(f"\nEnvironment: {args.env}.yaml")
    print(f"Configuration: {args.num_envs} envs × {args.num_steps} steps")
    print(f"Total timesteps: {args.total_timesteps:,}")

    results = []

    # Run benchmarks
    if not args.only_cpp:
        try:
            result = benchmark_batchenv(args)
            results.append(result)
        except Exception as e:
            print(f"\nError in BatchEnv benchmark: {e}")
            import traceback
            traceback.print_exc()

    if not args.only_python:
        try:
            result = benchmark_batchrolloutenv(args)
            results.append(result)
        except Exception as e:
            print(f"\nError in BatchRolloutEnv benchmark: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if results:
        print_comparison(results)

        # Save results to file
        output_dir = Path(__file__).parent.parent / "benchmark_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"rollout_benchmark_{timestamp}.txt"

        with open(output_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("PPO ROLLOUT BENCHMARK RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Environment: {args.env}.yaml\n")
            f.write(f"Configuration: {args.num_envs} envs × {args.num_steps} steps\n")
            f.write(f"Total timesteps: {args.total_timesteps:,}\n")

            for r in results:
                f.write(f"\n{r.name}:\n")
                f.write(f"  Rollout time: {r.rollout_time_ms:.1f} ms\n")
                f.write(f"  Learning time: {r.learning_time_ms:.1f} ms\n")
                f.write(f"  Total time: {r.total_time_ms:.1f} ms\n")
                f.write(f"  Samples/sec: {r.samples_per_second:.1f}\n")
                f.write(f"  Memory: {r.memory_used_mb:.1f} MB (peak: {r.peak_memory_mb:.1f} MB)\n")

            if len(results) == 2:
                baseline = results[0]
                optimized = results[1]
                speedup = baseline.total_time_ms / optimized.total_time_ms
                f.write(f"\nSpeedup: {speedup:.2f}x\n")

        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo benchmark results available.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark: Number of Environments Scaling for ppo_rollout_learner.py

Tests how BatchRolloutEnv performance scales with different numbers of parallel environments.
Measures:
- Samples Per Second (SPS)
- Rollout time
- Learning time
- Total iteration time
- Memory usage
- Speedup relative to baseline (num_envs=2)

Expected behavior:
- Rollout time should scale sub-linearly (parallel execution)
- Learning time should scale linearly with batch size
- SPS should increase with num_envs until bottleneck (CPU/GPU)
"""

import argparse
import time
import psutil
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Configure threading before torch import (prevents thread oversubscription)
import ppo.torch_config
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    num_envs: int
    num_steps: int
    total_timesteps: int

    # Timing metrics (milliseconds)
    rollout_time_ms: float
    learning_time_ms: float
    weight_sync_time_ms: float
    total_time_ms: float

    # Performance metrics
    samples_per_second: float
    iterations: int
    batch_size: int

    # Memory metrics (MB)
    memory_used_mb: float
    memory_peak_mb: float

    # Scaling metrics (relative to baseline)
    speedup: float = 1.0
    efficiency: float = 1.0  # speedup / num_envs


class PerformanceMonitor:
    """Monitor CPU and memory usage."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0.0

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_memory_mb()
        if current > self.peak_memory_mb:
            self.peak_memory_mb = current

    def reset_peak(self):
        """Reset peak memory tracking."""
        self.peak_memory_mb = self.get_memory_mb()


def benchmark_num_envs(args: argparse.Namespace, num_envs: int) -> BenchmarkResult:
    """Benchmark ppo_rollout_learner with specific number of environments."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK: num_envs={num_envs}, num_steps={args.num_steps}")
    print(f"{'='*80}")

    from batchrolloutenv import BatchRolloutEnv
    from ppo.muscle_learner import MuscleLearner
    from ppo.ppo_rollout_learner import Agent

    monitor = PerformanceMonitor()
    monitor.reset_peak()

    # Create environment
    yaml_path = Path(__file__).parent.parent / "data" / "env" / f"{args.env}.yaml"
    with open(yaml_path) as f:
        yaml_content = f.read()

    envs = BatchRolloutEnv(yaml_content, num_envs, args.num_steps)
    obs_dim = envs.obs_size()
    action_dim = envs.action_size()

    batch_size = num_envs * args.num_steps
    num_iterations = args.total_timesteps // batch_size

    print(f"Batch size: {num_envs} envs × {args.num_steps} steps = {batch_size} samples")
    print(f"Iterations: {num_iterations}")

    # Create policy agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)

    # Create muscle learner if hierarchical
    is_hierarchical = envs.is_hierarchical()
    muscle_learner = None
    if is_hierarchical:
        muscle_learner = MuscleLearner(
            num_actuator_action=envs.getNumActuatorAction(),
            num_muscles=envs.getNumMuscles(),
            num_muscle_dofs=envs.getNumMuscleDof(),
            learning_rate=1e-4,
            num_epochs=4,
            batch_size=64,
            is_cascaded=envs.use_cascading(),
            device=str(device)
        )
        envs.update_muscle_weights(muscle_learner.get_state_dict())

    # Sync initial policy weights
    agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
    envs.update_policy_weights(agent_state_cpu)

    # Warmup iteration (exclude from timing)
    if args.warmup:
        print("Warmup iteration...")
        _ = envs.collect_rollout()

    # Benchmark iterations
    print(f"Starting benchmark: {num_iterations} iterations")

    total_rollout_time = 0.0
    total_learning_time = 0.0
    total_sync_time = 0.0

    start_time = time.perf_counter()

    for iteration in range(1, num_iterations + 1):
        # Rollout phase
        rollout_start = time.perf_counter()
        trajectory = envs.collect_rollout()
        rollout_time = (time.perf_counter() - rollout_start) * 1000
        total_rollout_time += rollout_time

        # Convert to tensors
        obs = torch.from_numpy(trajectory['obs']).to(device)
        actions = torch.from_numpy(trajectory['actions']).to(device)
        logprobs = torch.from_numpy(trajectory['logprobs']).to(device)
        rewards = torch.from_numpy(trajectory['rewards']).to(device)
        terminations = torch.from_numpy(trajectory['terminations']).to(device).float()
        truncations = torch.from_numpy(trajectory['truncations']).to(device).float()
        values = torch.from_numpy(trajectory['values']).to(device)

        # Reshape for GAE
        obs = obs.reshape(args.num_steps, num_envs, -1)
        actions = actions.reshape(args.num_steps, num_envs, -1)
        logprobs = logprobs.reshape(args.num_steps, num_envs)
        rewards = rewards.reshape(args.num_steps, num_envs)
        terminations = terminations.reshape(args.num_steps, num_envs)
        truncations = truncations.reshape(args.num_steps, num_envs)
        values = values.reshape(args.num_steps, num_envs)

        dones = torch.logical_or(terminations, truncations).float()

        # Terminal bootstrapping
        if 'truncated_final_obs' in trajectory and len(trajectory['truncated_final_obs']) > 0:
            with torch.no_grad():
                for step, env_idx, final_obs_np in trajectory['truncated_final_obs']:
                    final_obs_tensor = torch.from_numpy(final_obs_np).unsqueeze(0).to(device)
                    terminal_value = agent.get_value(final_obs_tensor).item()
                    rewards[step, env_idx] += 0.99 * terminal_value

        # Learning phase
        learning_start = time.perf_counter()

        # GAE computation
        with torch.no_grad():
            next_obs = obs[-1]
            next_done = dones[-1]
            next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.99 * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update (4 epochs, 4 minibatches)
        minibatch_size = batch_size // 4
        b_inds = np.arange(batch_size)

        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -0.2, 0.2
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss + v_loss * 1.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        learning_time = (time.perf_counter() - learning_start) * 1000
        total_learning_time += learning_time

        # Weight sync
        sync_start = time.perf_counter()
        agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
        envs.update_policy_weights(agent_state_cpu)
        sync_time = (time.perf_counter() - sync_start) * 1000
        total_sync_time += sync_time

        # Muscle learning (if hierarchical)
        if muscle_learner is not None:
            muscle_tuples = envs.get_muscle_tuples()
            _ = muscle_learner.learn(muscle_tuples)
            envs.update_muscle_weights(muscle_learner.get_state_dict())

        monitor.update_peak()

        # Progress logging
        if iteration % max(1, num_iterations // 5) == 0:
            progress = (iteration / num_iterations) * 100
            print(f"Progress: {progress:.0f}% | "
                  f"Rollout: {rollout_time:.1f}ms | "
                  f"Learning: {learning_time:.1f}ms | "
                  f"Sync: {sync_time:.1f}ms")

    total_time = (time.perf_counter() - start_time) * 1000

    # Calculate metrics
    avg_rollout = total_rollout_time / num_iterations
    avg_learning = total_learning_time / num_iterations
    avg_sync = total_sync_time / num_iterations
    sps = (args.total_timesteps / total_time) * 1000

    return BenchmarkResult(
        num_envs=num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        rollout_time_ms=avg_rollout,
        learning_time_ms=avg_learning,
        weight_sync_time_ms=avg_sync,
        total_time_ms=total_time / num_iterations,
        samples_per_second=sps,
        iterations=num_iterations,
        batch_size=batch_size,
        memory_used_mb=monitor.get_memory_mb(),
        memory_peak_mb=monitor.peak_memory_mb
    )


def print_results(results: List[BenchmarkResult], baseline_sps: float):
    """Print benchmark results in formatted table."""

    print("\n" + "="*100)
    print("BENCHMARK RESULTS: Number of Environments Scaling")
    print("="*100)

    print(f"\nConfiguration:")
    print(f"  Steps per rollout: {results[0].num_steps}")
    print(f"  Total timesteps: {results[0].total_timesteps}")

    # Calculate speedups relative to baseline
    for result in results:
        result.speedup = result.samples_per_second / baseline_sps
        result.efficiency = result.speedup / (result.num_envs / results[0].num_envs)

    # Performance table
    print(f"\n{'Num Envs':<10} {'Batch':<8} {'Rollout (ms)':<15} {'Learning (ms)':<15} "
          f"{'Sync (ms)':<12} {'Total (ms)':<12} {'SPS':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 100)

    for r in results:
        print(f"{r.num_envs:<10} {r.batch_size:<8} {r.rollout_time_ms:<15.1f} "
              f"{r.learning_time_ms:<15.1f} {r.weight_sync_time_ms:<12.1f} "
              f"{r.total_time_ms:<12.1f} {r.samples_per_second:<10.1f} "
              f"{r.speedup:<10.2f}x {r.efficiency:<12.1%}")

    # Memory table
    print(f"\n{'='*60}")
    print("MEMORY USAGE")
    print(f"{'='*60}")
    print(f"\n{'Num Envs':<15} {'Current (MB)':<20} {'Peak (MB)':<20}")
    print("-" * 60)

    for r in results:
        print(f"{r.num_envs:<15} {r.memory_used_mb:<20.1f} {r.memory_peak_mb:<20.1f}")

    # Scaling analysis
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS")
    print(f"{'='*60}")

    print(f"\nRollout time scaling (ideal: sub-linear due to parallelism)")
    for i, r in enumerate(results):
        if i == 0:
            continue
        expected = results[0].rollout_time_ms
        actual = r.rollout_time_ms
        ratio = actual / expected
        print(f"  {results[0].num_envs} → {r.num_envs} envs: "
              f"{results[0].rollout_time_ms:.1f}ms → {actual:.1f}ms "
              f"(ratio: {ratio:.2f}x, ideal: 1.0x)")

    print(f"\nLearning time scaling (expected: linear with batch size)")
    for i, r in enumerate(results):
        if i == 0:
            continue
        expected_factor = r.batch_size / results[0].batch_size
        actual_factor = r.learning_time_ms / results[0].learning_time_ms
        print(f"  {results[0].num_envs} → {r.num_envs} envs: "
              f"{results[0].learning_time_ms:.1f}ms → {r.learning_time_ms:.1f}ms "
              f"(factor: {actual_factor:.2f}x, expected: {expected_factor:.2f}x)")

    print(f"\nThroughput (SPS) scaling (target: maximize)")
    for i, r in enumerate(results):
        print(f"  {r.num_envs} envs: {r.samples_per_second:.1f} SPS "
              f"(speedup: {r.speedup:.2f}x, efficiency: {r.efficiency:.1%})")

    # Optimal configuration recommendation
    best_sps = max(results, key=lambda r: r.samples_per_second)
    best_efficiency = max(results, key=lambda r: r.efficiency)

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print(f"\nBest throughput: {best_sps.num_envs} envs ({best_sps.samples_per_second:.1f} SPS)")
    print(f"Best efficiency: {best_efficiency.num_envs} envs ({best_efficiency.efficiency:.1%})")

    if best_sps.num_envs == results[-1].num_envs:
        print(f"\nNote: Peak performance at maximum tested envs ({best_sps.num_envs})")
        print(f"      Consider testing higher values to find true optimum")


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save results to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"num_envs_benchmark_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("="*100 + "\n")
        f.write("BatchRolloutEnv Number of Environments Scaling Benchmark\n")
        f.write("="*100 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps per rollout: {results[0].num_steps}\n")
        f.write(f"Total timesteps: {results[0].total_timesteps}\n\n")

        f.write(f"{'Num Envs':<10} {'Batch':<8} {'Rollout (ms)':<15} {'Learning (ms)':<15} "
                f"{'Sync (ms)':<12} {'Total (ms)':<12} {'SPS':<10} {'Speedup':<10} {'Efficiency':<12}\n")
        f.write("-" * 100 + "\n")

        for r in results:
            f.write(f"{r.num_envs:<10} {r.batch_size:<8} {r.rollout_time_ms:<15.1f} "
                    f"{r.learning_time_ms:<15.1f} {r.weight_sync_time_ms:<12.1f} "
                    f"{r.total_time_ms:<12.1f} {r.samples_per_second:<10.1f} "
                    f"{r.speedup:<10.2f}x {r.efficiency:<12.1%}\n")

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ppo_rollout_learner scaling with number of environments"
    )
    parser.add_argument("--env", type=str, default="A2_sep", help="Environment name")
    parser.add_argument("--num-steps", type=int, default=64, help="Steps per rollout")
    parser.add_argument("--total-timesteps", type=int, default=2048,
                        help="Total timesteps for benchmark")
    parser.add_argument("--num-envs-list", type=int, nargs='+',
                        default=[2, 4, 8, 16, 32],
                        help="List of num_envs values to test")
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup iteration before benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory for results")

    args = parser.parse_args()

    print("="*100)
    print("PPO ROLLOUT NUM_ENVS SCALING BENCHMARK")
    print("="*100)
    print(f"\nEnvironment: {args.env}.yaml")
    print(f"Steps per rollout: {args.num_steps}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Testing num_envs: {args.num_envs_list}")

    results = []
    baseline_sps = None

    try:
        for num_envs in args.num_envs_list:
            result = benchmark_num_envs(args, num_envs)
            results.append(result)

            if baseline_sps is None:
                baseline_sps = result.samples_per_second

        # Print and save results
        print_results(results, baseline_sps)
        save_results(results, Path(args.output_dir))

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        if results:
            print_results(results, results[0].samples_per_second if results else 1.0)
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()

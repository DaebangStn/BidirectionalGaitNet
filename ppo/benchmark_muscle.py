#!/usr/bin/env python3
"""
Benchmark muscle learning pipeline to identify bottlenecks.

Measures each stage of the muscle time:
  1. get_muscle_tuples()   — C++ Eigen → Python numpy (GIL-held)
  2. np.stack()            — list of arrays → single array
  3. torch.tensor()        — CPU numpy → GPU tensor
  4. muscle_learner.learn()— actual GPU gradient steps
  5. get_state_dict()      — GPU → CPU numpy
  6. update_muscle_weights()— C++ weight sync

Usage:
    pixi run python -m ppo.benchmark_muscle --env_file data/env/base_imit.yaml
    pixi run python -m ppo.benchmark_muscle --env_file data/env/base_imit.yaml --num_steps 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running as module from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
# batchrolloutenv.so lives in ppo/
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark muscle learning pipeline")
    p.add_argument("--env_file", type=str, required=True,
                   help="Path to env YAML config (e.g. data/env/base_imit.yaml)")
    p.add_argument("--num_envs", type=int, default=32,
                   help="Number of parallel environments (default: 32)")
    p.add_argument("--num_steps", type=int, default=5,
                   help="Number of rollout+learn iterations to benchmark (default: 5)")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup iterations excluded from stats (default: 2)")
    p.add_argument("--muscle_epochs", type=int, default=3,
                   help="Muscle training epochs per iteration (default: 3)")
    p.add_argument("--muscle_batch_size", type=int, default=128,
                   help="Muscle minibatch size (default: 128)")
    return p.parse_args()


def timed(label, fn):
    """Run fn(), return (result, elapsed_ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


def stack_and_upload(all_tuples, device):
    """np.stack then torch.tensor → GPU for each component. Returns (tensors, stack_ms, upload_ms)."""
    tau_list, JtA_red_list, JtA_list = all_tuples[0], all_tuples[1], all_tuples[2]

    t0 = time.perf_counter()
    tau_np     = np.stack(tau_list)
    JtA_red_np = np.stack(JtA_red_list)
    JtA_np     = np.stack(JtA_list)
    stack_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tau_t     = torch.tensor(tau_np,     device=device, dtype=torch.float32)
    JtA_red_t = torch.tensor(JtA_red_np, device=device, dtype=torch.float32)
    JtA_t     = torch.tensor(JtA_np,     device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    upload_ms = (time.perf_counter() - t0) * 1000

    return (tau_t, JtA_red_t, JtA_t), stack_ms, upload_ms


def main():
    args = parse_args()

    print("=" * 60)
    print("Muscle Pipeline Benchmark")
    print("=" * 60)
    print(f"  env_file:     {args.env_file}")
    print(f"  num_envs:     {args.num_envs}")
    print(f"  warmup:       {args.warmup}")
    print(f"  steps:        {args.num_steps}")
    print(f"  muscle_epochs:{args.muscle_epochs}")
    print(f"  batch_size:   {args.muscle_batch_size}")
    print()

    # --- Import C++ module ---
    from batchrolloutenv import BatchRolloutEnv
    from ppo.muscle_learner import MuscleLearner

    # Determine horizon from env config
    import yaml
    with open(args.env_file) as f:
        env_cfg = yaml.safe_load(f)
    horizon = env_cfg.get("args", {}).get("num_steps", 128)
    print(f"  horizon:      {horizon}")
    print()

    print("Creating BatchRolloutEnv...")
    envs = BatchRolloutEnv(args.env_file, args.num_envs, horizon)

    if not envs.is_hierarchical():
        print("ERROR: This environment is not hierarchical (no muscle learning).")
        sys.exit(1)

    num_actuator_action = envs.getNumActuatorAction()
    num_muscles         = envs.getNumMuscles()
    num_muscle_dofs     = envs.getNumMuscleDof()

    print(f"  Muscle dims:  {num_muscles} muscles, {num_muscle_dofs} DOFs, {num_actuator_action} actuators")
    expected_tuples = args.num_envs * horizon
    expected_JtA_floats = expected_tuples * num_muscles * num_actuator_action
    print(f"  Tuples/iter:  {expected_tuples:,}")
    print(f"  JtA size:     ({num_muscles}, {num_actuator_action}) = {num_muscles * num_actuator_action:,} floats/tuple")
    print(f"  Total JtA:    {expected_JtA_floats * 4 / 1024**2:.1f} MB")
    print()

    print("Creating MuscleLearner...")
    learner = MuscleLearner(
        num_actuator_action=num_actuator_action,
        num_muscles=num_muscles,
        num_muscle_dofs=num_muscle_dofs,
        learning_rate=1e-4,
        num_epochs=args.muscle_epochs,
        batch_size=args.muscle_batch_size,
        is_cascaded=envs.use_cascading(),
    )
    device = learner.device

    # Initialize weights & reset
    envs.update_muscle_weights(learner.get_state_dict())
    agent_obs_size = envs.obs_size()
    agent_act_size = envs.action_size()

    # Simple random policy for rollout
    class RandomPolicy:
        def sample_action(self, obs):
            return np.random.randn(agent_act_size).astype(np.float32), 0.0, False

    print("Resetting envs...")
    envs.reset()
    print()

    # Timing accumulators
    keys = ["rollout_ms", "get_tuples_ms", "flatten_ms",
            "stack_ms", "upload_ms", "learn_ms", "get_sd_ms", "update_w_ms", "total_ms"]
    stats = {k: [] for k in keys}

    total_iters = args.warmup + args.num_steps

    for it in range(total_iters):
        is_warmup = it < args.warmup
        label = "WARMUP" if is_warmup else f"ITER  {it - args.warmup:2d}"

        t_iter_start = time.perf_counter()

        # 1. Collect rollout (random actions to populate muscle buffers)
        t0 = time.perf_counter()
        envs.collect_rollout()
        rollout_ms = (time.perf_counter() - t0) * 1000

        # 2. get_muscle_tuples — C++ Eigen → Python numpy
        t0 = time.perf_counter()
        muscle_tuples = envs.get_muscle_tuples()
        get_tuples_ms = (time.perf_counter() - t0) * 1000

        # 3. Flatten the list-of-envs structure
        t0 = time.perf_counter()
        num_components = 3
        all_tuples = [[] for _ in range(num_components)]
        for env_tuples in muscle_tuples:
            for c in range(num_components):
                all_tuples[c].extend(env_tuples[c])
        flatten_ms = (time.perf_counter() - t0) * 1000
        num_tuples = len(all_tuples[0])

        # 4. np.stack
        t0 = time.perf_counter()
        tau_np     = np.stack(all_tuples[0])
        JtA_red_np = np.stack(all_tuples[1])
        JtA_np     = np.stack(all_tuples[2])
        stack_ms = (time.perf_counter() - t0) * 1000

        # 5. CPU → GPU upload
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tau_t     = torch.tensor(tau_np,     device=device, dtype=torch.float32)
        JtA_red_t = torch.tensor(JtA_red_np, device=device, dtype=torch.float32)
        JtA_t     = torch.tensor(JtA_np,     device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        upload_ms = (time.perf_counter() - t0) * 1000

        # 6. Actual muscle.learn() (includes its own stack+upload internally — see below)
        #    We call the real learn() to match production exactly
        t0 = time.perf_counter()
        muscle_loss = learner.learn(muscle_tuples)
        torch.cuda.synchronize()
        learn_ms = (time.perf_counter() - t0) * 1000
        # Sub-breakdown from inside learn():
        conv_ms   = muscle_loss['time']['converting_time_ms']
        grad_ms   = muscle_loss['time']['learning_time_ms']

        # 7. get_state_dict — GPU → CPU numpy
        t0 = time.perf_counter()
        sd = learner.get_state_dict()
        get_sd_ms = (time.perf_counter() - t0) * 1000

        # 8. update_muscle_weights — C++ weight sync
        t0 = time.perf_counter()
        envs.update_muscle_weights(sd)
        update_w_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_iter_start) * 1000

        print(f"[{label}] total={total_ms:.0f}ms | "
              f"rollout={rollout_ms:.0f} "
              f"get_tuples={get_tuples_ms:.0f} "
              f"flatten={flatten_ms:.0f} "
              f"stack={stack_ms:.0f} "
              f"upload={upload_ms:.0f} "
              f"learn={learn_ms:.0f}(conv={conv_ms:.0f}+grad={grad_ms:.0f}) "
              f"get_sd={get_sd_ms:.0f} "
              f"sync_w={update_w_ms:.0f}  "
              f"n={num_tuples}")

        if not is_warmup:
            stats["rollout_ms"].append(rollout_ms)
            stats["get_tuples_ms"].append(get_tuples_ms)
            stats["flatten_ms"].append(flatten_ms)
            stats["stack_ms"].append(stack_ms)
            stats["upload_ms"].append(upload_ms)
            stats["learn_ms"].append(learn_ms)
            stats["get_sd_ms"].append(get_sd_ms)
            stats["update_w_ms"].append(update_w_ms)
            stats["total_ms"].append(total_ms)

    if not stats["total_ms"]:
        print("No timed iterations (increase --num_steps).")
        return

    print()
    print("=" * 60)
    print("Summary (mean over timed iterations)")
    print("=" * 60)

    def mean(lst): return sum(lst) / len(lst)

    total_mean = mean(stats["total_ms"])
    rows = [
        ("rollout (C++ sim)",         mean(stats["rollout_ms"])),
        ("get_muscle_tuples (C++→py)", mean(stats["get_tuples_ms"])),
        ("flatten list",               mean(stats["flatten_ms"])),
        ("np.stack arrays",            mean(stats["stack_ms"])),
        ("upload CPU→GPU",             mean(stats["upload_ms"])),
        ("muscle learn() total",       mean(stats["learn_ms"])),
        ("get_state_dict GPU→CPU",     mean(stats["get_sd_ms"])),
        ("update_muscle_weights",      mean(stats["update_w_ms"])),
        ("TOTAL",                      total_mean),
    ]

    for name, ms in rows:
        bar = "#" * int(ms / total_mean * 40)
        print(f"  {name:<35s} {ms:7.1f} ms  {ms/total_mean*100:5.1f}%  {bar}")

    print()
    print(f"  Tuples per iteration: {num_tuples:,}")
    print(f"  JtA matrix shape:     ({num_muscles}, {num_actuator_action})")
    print()
    print("Note: learn() internally re-runs stack+upload, so those costs are")
    print("      counted twice above. The learn() total is authoritative.")


if __name__ == "__main__":
    main()

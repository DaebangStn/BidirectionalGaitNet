"""
PPO with Autonomous C++ Rollout

New architecture where C++ handles entire rollout autonomously:
- C++ PolicyNet (libtorch) runs inference
- C++ collects complete trajectory
- Python does GAE computation and PPO learning
- Synchronize weights per iteration

Expected performance: 2.0-2.3x speedup over BatchEnv (target: 900-1000 SPS)
"""

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# CRITICAL: Disable nested parallelism to prevent thread oversubscription
# BatchRolloutEnv uses ThreadPool for environment-level parallelism (hardware_concurrency threads)
# Setting OMP/MKL to 1 ensures each libtorch operation runs single-threaded within its thread
# This prevents thread explosion: num_envs × OMP_NUM_THREADS (e.g., 16 × 64 = 1024 threads!)
import os
os.environ.setdefault("OMP_NUM_THREADS", '1')
os.environ.setdefault("MKL_NUM_THREADS", '1')

import torch
# Also set PyTorch's internal threading limits
# Use try-except to handle cases where threading is already configured (e.g., when imported as module)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Threading already configured, ignore
    pass
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo.muscle_learner import MuscleLearner


@dataclass
class Args:
    exp_name: str = "ppo_rollout_learner"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    checkpoint_interval: Optional[int] = 1000
    """save checkpoint every K iterations (None = no checkpoints, only final save)"""

    # Algorithm specific arguments
    env_file: str = "data/env/A2_sep.yaml"
    """path to environment configuration file"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.99
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: Optional[float] = None
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = 0.01
    """the target KL divergence threshold"""

    # Muscle learning specific arguments
    muscle_lr: float = 1e-4
    """muscle network learning rate"""
    muscle_num_epochs: int = 4
    """muscle network training epochs"""
    muscle_batch_size: int = 64
    """muscle network batch size"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO agent with actor-critic architecture

    IMPORTANT: This must exactly match PolicyNet.cpp architecture for weight compatibility!
    """

    def __init__(self, num_states, num_actions):
        super().__init__()

        # Value network: 3 hidden layers of 512 units with ReLU
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_states, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        # Policy network: 3 hidden layers of 512 units with ReLU
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(num_states, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )

        # Initialize log_std
        init_log_std = torch.ones(num_actions)
        if num_actions > 18:
            init_log_std[18:] *= 0.5  # For upper body
        init_log_std[-1] = 1.0  # For cascading

        self.actor_logstd = nn.Parameter(init_log_std.unsqueeze(0))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Hyperparameter validation
    if args.minibatch_size < 16:
        print(f"ERROR: minibatch_size ({args.minibatch_size}) is too small. Must be >= 16.")
        print(f"Current configuration: num_envs={args.num_envs}, num_steps={args.num_steps}, num_minibatches={args.num_minibatches}")
        print(f"Computed: batch_size={args.batch_size}, minibatch_size={args.minibatch_size}")
        print("Suggestion: Increase num_envs, num_steps, or decrease num_minibatches")
        sys.exit(1)

    if args.muscle_batch_size > 4 * args.batch_size:
        print(f"ERROR: muscle_batch_size ({args.muscle_batch_size}) is too large relative to batch_size ({args.batch_size}).")
        print(f"Constraint: muscle_batch_size must be <= 4 * batch_size (currently {4 * args.batch_size})")
        print(f"Current configuration: num_envs={args.num_envs}, num_steps={args.num_steps}")
        print(f"Computed: batch_size={args.batch_size}")
        print("Suggestion: Decrease muscle_batch_size or increase num_envs/num_steps")
        sys.exit(1)

    run_name = f"{Path(args.env_file).stem}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # BatchRolloutEnv setup
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from batchrolloutenv import BatchRolloutEnv

        # Read YAML content
        with open(args.env_file, 'r') as f:
            yaml_content = f.read()

        print(f"Creating BatchRolloutEnv: {args.num_envs} envs, {args.num_steps} steps")
        envs = BatchRolloutEnv(yaml_content, args.num_envs, args.num_steps)

        # Get dimensions
        num_states = envs.obs_size()
        num_actions = envs.action_size()

        print(f"Environment: {Path(args.env_file).name}")
        print(f"Observation dim: {num_states}, Action dim: {num_actions}")

    except ImportError as e:
        print(f"ERROR: Failed to import BatchRolloutEnv: {e}")
        print("Make sure batchrolloutenv.so is built with: ninja -C build/release")
        sys.exit(1)

    # Get environment configuration
    is_hierarchical = envs.is_hierarchical()
    use_cascading = envs.use_cascading() if is_hierarchical else False

    print(f"Hierarchical control: {is_hierarchical}")
    if is_hierarchical:
        print(f"Cascading mode: {use_cascading}")

    # Create main PPO agent
    agent = Agent(num_states, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Create muscle learner if hierarchical
    muscle_learner = None
    if is_hierarchical:
        num_actuator_action = envs.getNumActuatorAction()
        num_muscles = envs.getNumMuscles()
        num_muscle_dofs = envs.getNumMuscleDof()

        print(f"Muscle configuration: {num_muscles} muscles, {num_muscle_dofs} DOFs, {num_actuator_action} actuators")

        muscle_learner = MuscleLearner(
            num_actuator_action=num_actuator_action,
            num_muscles=num_muscles,
            num_muscle_dofs=num_muscle_dofs,
            learning_rate=args.muscle_lr,
            num_epochs=args.muscle_num_epochs,
            batch_size=args.muscle_batch_size,
            is_cascaded=use_cascading,
            device="cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        # Initialize muscle weights in all environments
        state_dict = muscle_learner.get_state_dict()
        envs.update_muscle_weights(state_dict)

    # Initialize C++ policy weights
    print("Synchronizing initial policy weights to C++...")
    # Move all tensors to CPU before passing to C++
    agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
    envs.update_policy_weights(agent_state_cpu)

    # Training loop
    global_step = 0
    start_time = time.time()

    use_tqdm = sys.stdout.isatty()
    
    print(f"Training loop started with {args.num_iterations} iterations")

    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Iterations", ncols=100, disable=not use_tqdm):
        # Log progress periodically when tqdm is disabled (SLURM batch jobs)
        if not use_tqdm and iteration % 10 == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            print(f"[Iteration {iteration}/{args.num_iterations}] Steps: {global_step}, SPS: {sps}, Elapsed: {elapsed:.1f}s")

        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===== AUTONOMOUS C++ ROLLOUT =====
        rollout_start = time.perf_counter()

        # C++ runs entire rollout autonomously with libtorch inference
        # Returns: dict with numpy arrays (zero-copy)
        trajectory = envs.collect_rollout()

        rollout_time = (time.perf_counter() - rollout_start) * 1000
        writer.add_scalar("perf/rollout_time_ms", rollout_time, global_step)

        global_step += args.batch_size

        # Convert trajectory to torch tensors
        # Trajectory shape: (steps*envs, dim)
        obs = torch.from_numpy(trajectory['obs']).to(device)  # (batch, obs_dim)
        actions = torch.from_numpy(trajectory['actions']).to(device)  # (batch, action_dim)
        logprobs = torch.from_numpy(trajectory['logprobs']).to(device)  # (batch,)
        rewards = torch.from_numpy(trajectory['rewards']).to(device)  # (batch,)
        terminations = torch.from_numpy(trajectory['terminations']).to(device).float()  # (batch,)
        truncations = torch.from_numpy(trajectory['truncations']).to(device).float()  # (batch,)
        values = torch.from_numpy(trajectory['values']).to(device)  # (batch,)

        # Reshape to (num_steps, num_envs) for GAE computation
        obs = obs.reshape(args.num_steps, args.num_envs, -1)
        actions = actions.reshape(args.num_steps, args.num_envs, -1)
        logprobs = logprobs.reshape(args.num_steps, args.num_envs)
        rewards = rewards.reshape(args.num_steps, args.num_envs)
        terminations = terminations.reshape(args.num_steps, args.num_envs)
        truncations = truncations.reshape(args.num_steps, args.num_envs)
        values = values.reshape(args.num_steps, args.num_envs)

        # Compute dones (terminations OR truncations) for GAE masking
        dones = torch.logical_or(terminations, truncations).float()

        # Terminal value bootstrapping for truncated episodes
        # CRITICAL: Must happen BEFORE GAE computation
        if 'truncated_final_obs' in trajectory and len(trajectory['truncated_final_obs']) > 0:
            with torch.no_grad():
                for step, env_idx, final_obs_np in trajectory['truncated_final_obs']:
                    # Get terminal value for truncated episode
                    final_obs_tensor = torch.from_numpy(final_obs_np).unsqueeze(0).to(device)
                    terminal_value = agent.get_value(final_obs_tensor).item()
                    # Bootstrap: add discounted terminal value to reward
                    rewards[step, env_idx] += args.gamma * terminal_value

        ppo_learn_start = time.perf_counter()

        # ===== GAE COMPUTATION (Python) =====
        with torch.no_grad():
            # Get next value for bootstrap
            next_obs = obs[-1]  # Last step observation
            next_done = dones[-1]  # Last step done flags
            next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    # Use dones for advantage masking (both termination types end episodes)
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch for minibatch training
        b_obs = obs.reshape((-1, num_states))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, num_actions))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ===== PPO LEARNING =====
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        ppo_learn_time = (time.perf_counter() - ppo_learn_start) * 1000
        writer.add_scalar("perf/ppo_learn_time_ms", ppo_learn_time, global_step)

        # ===== SYNCHRONIZE WEIGHTS TO C++ =====
        sync_start = time.perf_counter()
        # Move all tensors to CPU before passing to C++
        agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
        envs.update_policy_weights(agent_state_cpu)
        sync_time = (time.perf_counter() - sync_start) * 1000
        writer.add_scalar("perf/weight_sync_time_ms", sync_time, global_step)

        # ===== MUSCLE LEARNING (if hierarchical) =====
        if muscle_learner is not None:
            muscle_learn_start = time.perf_counter()

            # Get muscle tuples from C++
            muscle_tuples = envs.get_muscle_tuples()

            # Train muscle network
            muscle_loss = muscle_learner.learn(muscle_tuples)

            # Update muscle weights in C++
            muscle_state_dict = muscle_learner.get_state_dict()
            envs.update_muscle_weights(muscle_state_dict)

            muscle_learn_time = (time.perf_counter() - muscle_learn_start) * 1000

            writer.add_scalar("perf/muscle_learn_time_ms", muscle_learn_time, global_step)
            # muscle_loss is a dict, extract the total loss
            if isinstance(muscle_loss, dict):
                writer.add_scalar("losses/muscle_loss", muscle_loss.get("loss", 0.0), global_step)
            else:
                writer.add_scalar("losses/muscle_loss", muscle_loss, global_step)

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # Performance metrics
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed) if elapsed > 0 else 0
        writer.add_scalar("charts/SPS", sps, global_step)

        # Log averaged info metrics from C++ accumulation
        if 'info' in trajectory:
            for key, avg_value in trajectory['info'].items():
                writer.add_scalar(f"rollout/{key}", avg_value, global_step)

        # Log episode statistics from C++ accumulation
        if 'avg_episode_return' in trajectory:
            writer.add_scalar("rollout/avg_episode_return", trajectory['avg_episode_return'], global_step)
            writer.add_scalar("rollout/avg_episode_length", trajectory['avg_episode_length'], global_step)
            writer.add_scalar("rollout/episode_count", trajectory['episode_count'], global_step)

        # Checkpoint saving
        if args.checkpoint_interval is not None and iteration % args.checkpoint_interval == 0:
            checkpoint_path = f"runs/{run_name}/checkpoint_{iteration}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'iteration': iteration,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'muscle_state_dict': muscle_learner.get_state_dict() if muscle_learner else None,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final model save
    if args.save_model:
        model_path = f"runs/{run_name}/final_model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'muscle_state_dict': muscle_learner.get_state_dict() if muscle_learner else None,
        }, model_path)
        print(f"Final model saved: {model_path}")

    writer.close()

"""
CleanRL PPO with Hierarchical Muscle Control

Based on CleanRL's ppo_continuous_action.py with hierarchical muscle learning.
Implements Option C design where muscle learning happens after PPO learning.
"""

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo.env_wrapper import HierarchicalEnv, make_env
from ppo.muscle_learner import MuscleLearner


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_file: str = "data/env/A2_sep.yaml"
    """path to environment configuration file"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer (Ray default)"""
    # num_envs: int = 4
    num_envs: int = 16
    """the number of parallel game environments (ppo_small_pc default)"""
    num_steps: int = 64
    # num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.99
    """the lambda for the general advantage estimation (Ray default)"""
    num_minibatches: int = 8
    # num_minibatches: int = 64
    """the number of mini-batches (computed from batch_size and sgd_minibatch_size)"""
    update_epochs: int = 4
    """the K epochs to update the policy (Ray num_sgd_iter)"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function (Ray vf_loss_coeff)"""
    max_grad_norm: float = None
    """the maximum norm for the gradient clipping (Ray grad_clip=None)"""
    target_kl: float = 0.01
    """the target KL divergence threshold (Ray kl_target)"""

    # Muscle learning specific arguments (Ray defaults)
    muscle_lr: float = 1e-4
    """muscle network learning rate"""
    muscle_num_epochs: int = 10
    """muscle network training epochs (Ray default)"""
    muscle_batch_size: int = 64
    # muscle_batch_size: int = 512
    """muscle network batch size (ppo_small sgd_minibatch_size)"""

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
    """PPO agent with actor-critic architecture (matching Ray's SimulationNN)"""

    def __init__(self, envs):
        super().__init__()
        num_states = np.array(envs.single_observation_space.shape).prod()
        num_actions = np.prod(envs.single_action_space.shape)

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

        # Initialize log_std to match Ray's initialization
        init_log_std = torch.ones(num_actions)
        if num_actions > 18:
            init_log_std[18:] *= 0.5  # For upper body
        init_log_std[-1] = 1.0  # For cascading

        # todo: learn log_std
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
    run_name = f"{Path(args.env_file).stem}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # Use 'spawn' context for CUDA compatibility with multiprocessing
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_file, i) for i in range(args.num_envs)],
        shared_memory=True,
        context='spawn'  # Required for CUDA compatibility
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Get environment configuration from first env using call()
    is_hierarchical = envs.call('is_hierarchical')[0]
    use_cascading = envs.call('use_cascading')[0] if is_hierarchical else False

    print(f"Environment: {Path(args.env_file).name}")
    print(f"Hierarchical control: {is_hierarchical}")
    if is_hierarchical:
        print(f"Cascading mode: {use_cascading}")

    # Create main PPO agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Create muscle learner if hierarchical
    muscle_learner = None
    if is_hierarchical:
        # Get muscle configuration from C++ environment using call()
        num_actuator_action = envs.call('getNumActuatorAction')[0]
        num_muscles = envs.call('getNumMuscles')[0]
        num_muscle_dofs = envs.call('getNumMuscleDof')[0]

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
        envs.call('update_muscle_weights', state_dict)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Iterations", ncols=100):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===== ROLLOUT PHASE (Standard CleanRL) =====
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data
            # Muscle substeps happen internally during env.step()!
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # https://github.com/DLR-RM/stable-baselines3/pull/658
            for idx, trunc in enumerate(truncations):
                if trunc and not terminations[idx]:
                    real_next_obs = infos["final_observation"][idx]
                    with torch.no_grad():
                        terminal_value = agent.get_value(torch.Tensor(real_next_obs).to(device)).reshape(1, -1)[0][0]
                    rewards[step][idx] += args.gamma * terminal_value


            # Handle truncation (from CleanRL)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ===== PPO LEARNING PHASE (Standard CleanRL) =====
        # bootstrap value if not done
        with torch.no_grad():
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
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ===== MUSCLE LEARNING PHASE (Hierarchical Control Extension) =====
        if muscle_learner is not None:
            muscle_start = time.perf_counter()

            # Collect muscle tuples from all environments
            all_tuples = envs.call('get_muscle_tuples')

            # Train muscle network
            muscle_stats = muscle_learner.learn(all_tuples)

            # Distribute updated weights to all environments
            state_dict = muscle_learner.get_state_dict()
            envs.call('update_muscle_weights', state_dict)

            muscle_time = (time.perf_counter() - muscle_start) * 1000

            # Log muscle training stats
            writer.add_scalar("muscle/loss", muscle_stats['loss_muscle'], global_step)
            writer.add_scalar("muscle/loss_target", muscle_stats['loss_target'], global_step)
            writer.add_scalar("muscle/loss_reg", muscle_stats['loss_reg'], global_step)
            writer.add_scalar("muscle/loss_act", muscle_stats['loss_act'], global_step)
            writer.add_scalar("muscle/num_tuples", muscle_stats['num_tuples'], global_step)
            writer.add_scalar("muscle/time_ms", muscle_time, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"Iteration {iteration}/{args.num_iterations}, SPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Save models
    if args.save_model:
        model_path = f"runs/{run_name}"
        os.makedirs(model_path, exist_ok=True)

        # Save policy agent
        torch.save(agent.state_dict(), f"{model_path}/agent.pt")
        print(f"Policy agent saved to {model_path}/agent.pt")

        # Save muscle learner if hierarchical
        if muscle_learner is not None:
            muscle_learner.save(f"{model_path}/muscle.pt")
            print(f"Muscle network saved to {model_path}/muscle.pt")

    envs.close()
    writer.close()

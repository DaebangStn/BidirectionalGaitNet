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
import yaml
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import psutil

# Configure threading before torch import (prevents thread oversubscription)
import ppo.torch_config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo.muscle_learner import MuscleLearner
from ppo.discriminator import DiscriminatorLearner
from ppo.utils import parse_args_with_presets


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    checkpoint_interval: int = 500
    """save checkpoint every K iterations"""

    # Algorithm specific arguments
    env_file: str = "data/env/A.yaml"
    """path to environment configuration file"""
    total_timesteps: int = 200_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the initial learning rate of the optimizer"""
    lr_final: float = 2e-5
    """the final learning rate after annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.99
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
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

    # Discriminator learning specific arguments (ADD-style energy efficiency)
    disc_lr: float = 1e-4
    """discriminator network learning rate"""
    disc_num_epochs: int = 3
    """discriminator training epochs per update"""
    disc_batch_size: int = 0
    """discriminator minibatch size (0 = use muscle_batch_size)"""
    disc_buffer_size: int = 100000
    """discriminator replay buffer size"""
    disc_grad_penalty: float = 10.0
    """discriminator gradient penalty coefficient"""
    disc_logit_reg: float = 0.01
    """discriminator logit regularization coefficient"""
    disc_weight_decay: float = 0.0001
    """discriminator weight decay"""
    disc_reward_weight: float = 0.5
    """weight for discriminator reward in total reward (additive)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    log_interval: int = 10
    """log progress every K iterations (for non-tty batch jobs)"""
    learn_std: bool = False
    """if True, log_std is learned (nn.Parameter); if False, constant"""
    init_log_std: float = 1.0
    """initial value for log_std (action noise)"""

    # Checkpointing and resume
    save_optimizer: bool = False
    """if True, save full training state (optimizer, iteration) for resume"""
    resume_from: Optional[str] = None
    """path to checkpoint directory to resume training from"""
    pd_from: Optional[str] = None
    """path to checkpoint dir - loads ONLY agent.pt (fresh training state)"""
    ckpt_from: Optional[str] = None
    """path to checkpoint dir - loads FULL checkpoint (resume training)"""
    no_clear_cache: bool = False
    """if toggled, skip clearing rm_cache at startup (for prefetched data)"""

    # --- Preset classmethods matching shell scripts ---

    @classmethod
    def gait(cls) -> "Args":
        """Preset matching scripts/train/gait.sh (CPU cluster, 128 cores)."""
        return cls(
            num_envs=128,
            num_steps=128,
            muscle_batch_size=512,
            num_minibatches=16,
        )

    @classmethod
    def a6000(cls) -> "Args":
        """Preset matching scripts/train/a6000.sh (GPU node, 96 cores)."""
        return cls(
            num_envs=96,
            num_steps=128,
            muscle_batch_size=512,
            num_minibatches=16,
        )

    @classmethod
    def debug(cls) -> "Args":
        return cls(
            num_envs=1,
            num_steps=32,
            muscle_batch_size=32,
            num_minibatches=1,
        )


# Dynamically build preset map from classmethods (None = default constructor)
PRESET_FNS = {None: Args}
for name, method in vars(Args).items():
    if isinstance(method, classmethod):
        PRESET_FNS[name] = getattr(Args, name)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def save_checkpoint(
    checkpoint_path: str,
    agent: nn.Module,
    muscle_learner,
    env_file: str,
    save_full_state: bool = False,
    optimizer: optim.Optimizer = None,
    iteration: int = None,
    global_step: int = None,
    args = None,
    resumed_from: str = None,
):
    """Save a training checkpoint with agent, muscle network, and metadata.

    Args:
        checkpoint_path: Directory path to save checkpoint files
        agent: PPO agent module
        muscle_learner: Muscle learner (or None if not hierarchical)
        env_file: Path to environment config file for metadata
        save_full_state: If True, save optimizer and training state for resume
        optimizer: PPO optimizer (required if save_full_state=True)
        iteration: Current iteration number (required if save_full_state=True)
        global_step: Total environment steps (required if save_full_state=True)
        args: Training args for validation on resume
        resumed_from: Original checkpoint path if this is a resumed run
    """
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save policy agent checkpoint
    torch.save(agent.state_dict(), f"{checkpoint_path}/agent.pt")

    # Save muscle learner checkpoint if hierarchical
    if muscle_learner is not None:
        muscle_learner.save(f"{checkpoint_path}/muscle.pt", save_optimizer=save_full_state)

    # Save full training state for resume
    if save_full_state:
        # PPO optimizer state (includes Adam momentum buffers)
        torch.save(optimizer.state_dict(), f"{checkpoint_path}/optimizer.pt")

        # Training state
        torch.save({
            'iteration': iteration,
            'global_step': global_step,
            'args': vars(args),
        }, f"{checkpoint_path}/training_state.pt")

    # Save metadata with lineage info
    with open(env_file, 'r') as f:
        metadata = yaml.safe_load(f)
    if resumed_from:
        metadata['resumed_from'] = resumed_from
    with open(f"{checkpoint_path}/metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)

    print(f"Checkpoint saved to {checkpoint_path}" +
          (" (full state)" if save_full_state else ""))


class Agent(nn.Module):
    """PPO agent with actor-critic architecture

    IMPORTANT: This must exactly match PolicyNet.cpp architecture for weight compatibility!
    """

    def __init__(self, num_states, num_actions, learn_std=True, init_log_std=1.0):
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
        log_std_init = torch.ones(num_actions)
        if num_actions > 18:
            log_std_init[18:] *= 0.5 # For upper body
        log_std_init[-1] = 1.0 # For cascading
        log_std_init = log_std_init * init_log_std

        self.learn_std = learn_std
        if learn_std:
            self.actor_logstd = nn.Parameter(log_std_init.unsqueeze(0))
        else:
            self.register_buffer('actor_logstd', log_std_init.unsqueeze(0))

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
    args = parse_args_with_presets(Args, PRESET_FNS)

    # Clear rm_cache at startup (unless using prefetch workflow)
    if not args.no_clear_cache:
        rm_cache_path = Path(__file__).parent.parent / ".tmp" / "rm_cache"
        lock_path = Path(__file__).parent.parent / ".tmp" / "rm_cache.lock"

        # Check lock before clearing (another job may be using cache)
        if lock_path.exists():
            print(f"WARNING: Cache is locked by another job. Skipping cache clear.")
            print(f"To force clear: rm {lock_path}")
        elif rm_cache_path.exists():
            shutil.rmtree(rm_cache_path)
            print(f"Cleared cache: {rm_cache_path}")

    # Override args from YAML config if 'args' section exists
    with open(args.env_file, 'r') as f:
        env_config = yaml.safe_load(f)
    if 'args' in env_config:
        for key, value in env_config['args'].items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown arg '{key}' in YAML config, ignoring")

    # Check for pd_from or ckpt_from in train section
    train_cfg = env_config.get('train')
    if train_cfg:

        # Get default pid for expanding @pid:/ URIs (same logic as prefetch.py)
        default_pid = env_config.get('pid')
        if not default_pid and 'environment' in env_config:
            default_pid = env_config['environment'].get('pid')

        # Helper to expand @pid: URIs (keeps as URI for later fetch)
        def expand_ckpt_uri(uri):
            if not uri:
                return None
            if uri.startswith('@pid:'):
                # Expand @pid:/path to @pid:{default_pid}/path
                if default_pid and uri.startswith('@pid:/'):
                    return f"@pid:{default_pid}/" + uri[6:]
                return uri
            return uri  # Already absolute path

        # pd_from: load only agent.pt (fresh training)
        if 'pd_from' in train_cfg and train_cfg['pd_from']:
            args.pd_from = expand_ckpt_uri(train_cfg['pd_from'])
            print(f"pd_from: {args.pd_from}")

        # ckpt_from: full checkpoint resume
        if 'ckpt_from' in train_cfg and train_cfg['ckpt_from']:
            args.ckpt_from = expand_ckpt_uri(train_cfg['ckpt_from'])
            print(f"ckpt_from: {args.ckpt_from}")

    # Parse curriculum config for imitation masking
    imit_curriculum = []
    if train_cfg and 'curriculum' in train_cfg:
        curriculum = train_cfg['curriculum']
        if 'imit_mask' in curriculum:
            imit_curriculum = sorted(curriculum['imit_mask'], key=lambda x: x['from_iteration'])
            print(f"Imitation mask curriculum: {len(imit_curriculum)} stages")
            for stage in imit_curriculum:
                op = stage.get('op', 'mask')  # default to mask for backward compatibility
                print(f"  - iteration {stage['from_iteration']}: {op} {stage['joints']}")

    imit_curriculum_applied_stages = set()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.disc_batch_size == 0:
        args.disc_batch_size = args.muscle_batch_size

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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = 'cuda'

    # BatchRolloutEnv setup
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from batchrolloutenv import BatchRolloutEnv

        print(f"Creating BatchRolloutEnv: {args.num_envs} envs, {args.num_steps} steps")
        envs = BatchRolloutEnv(args.env_file, args.num_envs, args.num_steps)

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
    agent = Agent(num_states, num_actions, learn_std=args.learn_std, init_log_std=args.init_log_std).to(device)
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
        )

        # Initialize muscle weights in all environments
        state_dict = muscle_learner.get_state_dict()
        envs.update_muscle_weights(state_dict)

    # Create discriminator learner if enabled
    disc_learner = None
    use_discriminator = envs.use_discriminator()
    if use_discriminator:
        disc_obs_dim = envs.getDiscObsDim()
        disc_reward_scale = envs.getDiscRewardScale()

        print(f"Discriminator enabled: disc_obs_dim={disc_obs_dim}, reward_scale={disc_reward_scale}")

        disc_learner = DiscriminatorLearner(
            disc_obs_dim=disc_obs_dim,
            learning_rate=args.disc_lr,
            num_epochs=args.disc_num_epochs,
            batch_size=args.disc_batch_size,
            buffer_size=args.disc_buffer_size,
            grad_penalty=args.disc_grad_penalty,
            logit_reg=args.disc_logit_reg,
            weight_decay=args.disc_weight_decay,
        )

        # Initialize discriminator weights in all environments
        disc_state_dict = disc_learner.get_state_dict()
        envs.update_discriminator_weights(disc_state_dict)

    # Resume from checkpoint if specified
    # Priority: resume_from > ckpt_from > pd_from
    start_iteration = 1
    global_step = 0
    ckpt_path = args.resume_from or args.ckpt_from  # Full checkpoint loading
    pd_only_path = args.pd_from if not ckpt_path else None  # Agent.pt only
    resumed_from = ckpt_path or pd_only_path  # Track lineage for metadata

    # Helper to fetch checkpoint file (supports @pid: URIs and local paths)
    def fetch_ckpt_file(base_path: str, filename: str) -> str:
        """Fetch checkpoint file, returns local path. Supports @pid: URIs."""
        if base_path.startswith('@pid:'):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "rm/python"))
            import pyrm
            rm_config = str(Path(__file__).parent.parent / "data/rm_config.yaml")
            rm = pyrm.ResourceManager(rm_config)
            uri = f"{base_path}/{filename}"
            handle = rm.fetch(uri)
            return handle.local_path()
        else:
            return str(Path(base_path) / filename)

    def ckpt_file_exists(base_path: str, filename: str) -> bool:
        """Check if checkpoint file exists. Supports @pid: URIs."""
        if base_path.startswith('@pid:'):
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "rm/python"))
            import pyrm
            rm_config = str(Path(__file__).parent.parent / "data/rm_config.yaml")
            rm = pyrm.ResourceManager(rm_config)
            uri = f"{base_path}/{filename}"
            return rm.exists(uri)
        else:
            return (Path(base_path) / filename).exists()

    if ckpt_path:
        # Full checkpoint resume (loads agent, muscle, optimizer, training_state)
        print(f"Resuming from checkpoint: {ckpt_path}")

        # Check if agent.pt exists before loading
        if ckpt_file_exists(ckpt_path, "agent.pt"):
            agent_file = fetch_ckpt_file(ckpt_path, "agent.pt")
            agent.load_state_dict(torch.load(agent_file, map_location=device))
            print(f"  Loaded agent.pt from {agent_file}")

            # Load muscle learner if exists
            if muscle_learner and ckpt_file_exists(ckpt_path, "muscle.pt"):
                has_full_state = ckpt_file_exists(ckpt_path, "training_state.pt")
                muscle_file = fetch_ckpt_file(ckpt_path, "muscle.pt")
                muscle_learner.load(muscle_file, load_optimizer=has_full_state)
                print(f"  Loaded muscle.pt from {muscle_file}")

            # Load full training state if available
            if ckpt_file_exists(ckpt_path, "training_state.pt"):
                state_file = fetch_ckpt_file(ckpt_path, "training_state.pt")
                state = torch.load(state_file, map_location=device)

                # Validate args compatibility
                saved_args = state.get('args', {})
                critical_args = ['num_envs', 'num_steps', 'learning_rate', 'lr_final']
                for arg in critical_args:
                    if arg in saved_args and getattr(args, arg) != saved_args[arg]:
                        print(f"Warning: {arg} differs: saved={saved_args[arg]}, current={getattr(args, arg)}")

                # Load PPO optimizer
                optimizer_file = fetch_ckpt_file(ckpt_path, "optimizer.pt")
                optimizer.load_state_dict(torch.load(optimizer_file, map_location=device))

                # Restore training progress
                start_iteration = state['iteration'] + 1
                global_step = state['global_step']

                print(f"Resumed: iteration={start_iteration}, global_step={global_step}")
            else:
                print("Warning: No training_state.pt found, starting from iteration 1 with loaded weights")
        else:
            print(f"Warning: agent.pt not found at {ckpt_path}, starting fresh training")
            resumed_from = None  # Clear lineage since checkpoint was not loaded

    elif pd_only_path:
        # pd_from: Load ONLY agent.pt (fresh training state)
        print(f"Loading pre-trained policy from: {pd_only_path}")

        if ckpt_file_exists(pd_only_path, "agent.pt"):
            agent_file = fetch_ckpt_file(pd_only_path, "agent.pt")
            agent.load_state_dict(torch.load(agent_file, map_location=device))
            print(f"Loaded agent.pt only - starting fresh training from iteration 1")
        else:
            print(f"Warning: agent.pt not found at {pd_only_path}, starting fresh training")
            resumed_from = None  # Clear lineage since checkpoint was not loaded
        # Do NOT load optimizer, training_state, or muscle

    # Initialize C++ policy weights
    agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
    envs.update_policy_weights(agent_state_cpu)
    if muscle_learner:
        envs.update_muscle_weights(muscle_learner.get_state_dict())

    # Reset all environments to initial state before first rollout
    envs.reset()

    # Release cache lock (if prefetch workflow was used)
    if args.no_clear_cache:
        lock_path = Path(__file__).parent.parent / ".tmp" / "rm_cache.lock"
        if lock_path.exists():
            lock_path.unlink()
            print(f"Released cache lock: {lock_path}")

    # Always create fresh run_name (new TensorBoard directory)
    env_name = Path(args.env_file).stem
    start_localtime = time.localtime()
    run_name = f"{env_name}/{time.strftime('%y%m%d_%H%M%S', start_localtime)}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Training loop
    start_time = time.time()

    use_tqdm = sys.stdout.isatty()

    print(f"Training loop started with {args.num_iterations} iterations (starting from {start_iteration})")

    for iteration in tqdm(range(start_iteration, args.num_iterations + 1), desc="Iterations", ncols=100, disable=not use_tqdm):
        # Apply curriculum masking/demasking for imitation reward
        for idx, stage in enumerate(imit_curriculum):
            if idx not in imit_curriculum_applied_stages and iteration >= stage['from_iteration']:
                op = stage.get('op', 'mask')
                for joint_name in stage['joints']:
                    if op == 'demask':
                        envs.demask_imit_joint(joint_name)
                    else:
                        envs.mask_imit_joint(joint_name)
                print(f"[Iteration {iteration}] Curriculum: {op} {stage['joints']}")
                imit_curriculum_applied_stages.add(idx)

        # Log progress periodically when tqdm is disabled (SLURM batch jobs)
        if not use_tqdm and iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            # Format elapsed time
            elapsed_hours = int(elapsed // 3600)
            elapsed_minutes = int((elapsed % 3600) // 60)
            elapsed_str = f"{elapsed_hours}h {elapsed_minutes}m"
            # Calculate ETA
            remaining_iterations = args.num_iterations - iteration
            if sps > 0:
                remaining_steps = remaining_iterations * args.batch_size
                eta_seconds = remaining_steps / sps
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_str = f"{eta_hours}h {eta_minutes}m"
            else:
                eta_str = "N/A"
            print(f"[Iteration {iteration}/{args.num_iterations}] Steps: {global_step}, SPS: {sps}, Elapsed: {elapsed_str}, ETA: {eta_str}")

        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = args.lr_final + frac * (args.learning_rate - args.lr_final)
            optimizer.param_groups[0]["lr"] = lrnow

        # ===== AUTONOMOUS C++ ROLLOUT =====
        epoch_start = time.perf_counter()

        # C++ runs entire rollout autonomously with libtorch inference
        # Returns: dict with numpy arrays (zero-copy)
        trajectory = envs.collect_rollout()

        rollout_time = (time.perf_counter() - epoch_start) * 1000

        global_step += args.batch_size

        # Convert trajectory to torch tensors (direct GPU allocation)
        # Trajectory shape: (steps*envs, dim)
        obs = torch.as_tensor(trajectory['obs'], device=device, dtype=torch.float32)  # (batch, obs_dim)
        actions = torch.as_tensor(trajectory['actions'], device=device, dtype=torch.float32)  # (batch, action_dim)
        logprobs = torch.as_tensor(trajectory['logprobs'], device=device, dtype=torch.float32)  # (batch,)
        rewards = torch.as_tensor(trajectory['rewards'], device=device, dtype=torch.float32)  # (batch,)
        terminations = torch.as_tensor(trajectory['terminations'], device=device, dtype=torch.float32)  # (batch,)
        truncations = torch.as_tensor(trajectory['truncations'], device=device, dtype=torch.float32)  # (batch,)
        values = torch.as_tensor(trajectory['values'], device=device, dtype=torch.float32)  # (batch,)
        next_obs = torch.as_tensor(trajectory['next_obs'], device=device, dtype=torch.float32)  # (num_envs, obs_dim)
        dones = torch.as_tensor(trajectory['dones'], device=device, dtype=torch.float32)
        next_done = torch.as_tensor(trajectory['next_done'], device=device, dtype=torch.float32)  # (num_envs,)

        # Reshape to (num_steps, num_envs) for GAE computation
        obs = obs.reshape(args.num_steps, args.num_envs, -1)
        actions = actions.reshape(args.num_steps, args.num_envs, -1)
        logprobs = logprobs.reshape(args.num_steps, args.num_envs)
        rewards = rewards.reshape(args.num_steps, args.num_envs)
        dones = dones.reshape(args.num_steps, args.num_envs)
        terminations = terminations.reshape(args.num_steps, args.num_envs)
        truncations = truncations.reshape(args.num_steps, args.num_envs)
        values = values.reshape(args.num_steps, args.num_envs)

        # Terminal value bootstrapping for truncated episodes
        with torch.no_grad():
            for step, env_idx, final_obs_np in trajectory['truncated_final_obs']:
                # Get terminal value for truncated episode (direct GPU allocation)
                final_obs_tensor = torch.as_tensor(final_obs_np, device=device, dtype=torch.float32).unsqueeze(0)
                terminal_value = agent.get_value(final_obs_tensor).item()
                # Bootstrap: add discounted terminal value to reward
                rewards[step, env_idx] += args.gamma * terminal_value

        ppo_learn_start = time.perf_counter()

        # ===== GAE COMPUTATION (Python) =====
        with torch.no_grad():
            # Use next_obs and next_done from C++ (observations AFTER rollout completes)
            next_value = agent.get_value(next_obs).reshape(1, -1)
            bootstrap_done = next_done

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - bootstrap_done
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

        # ===== SYNCHRONIZE WEIGHTS TO C++ =====
        sync_start = time.perf_counter()
        # Move all tensors to CPU before passing to C++
        agent_state_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
        envs.update_policy_weights(agent_state_cpu)
        sync_time = (time.perf_counter() - sync_start) * 1000

        # ===== MUSCLE LEARNING (if hierarchical) =====
        muscle_loss = None
        muscle_learn_time = 0.0
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

        # ===== DISCRIMINATOR LEARNING (if enabled) =====
        disc_loss = None
        disc_learn_time = 0.0
        if disc_learner is not None:
            disc_learn_start = time.perf_counter()

            # Get disc_obs (muscle activations) from C++
            disc_obs = envs.get_disc_obs()

            if disc_obs.shape[0] > 0:
                # Train discriminator
                disc_loss = disc_learner.learn(disc_obs)

                # Update discriminator weights in C++
                disc_state_dict = disc_learner.get_state_dict()
                envs.update_discriminator_weights(disc_state_dict)

            disc_learn_time = (time.perf_counter() - disc_learn_start) * 1000

        # Checkpoint saving - either on interval or triggered by save_ckpt file
        save_ckpt_trigger = Path(f"runs/{run_name}/save_ckpt")
        if save_ckpt_trigger.exists():
            save_ckpt_trigger.unlink()  # Remove trigger file
            should_save_checkpoint = True
        else:
            should_save_checkpoint = iteration % args.checkpoint_interval == 0

        if should_save_checkpoint:
            run_title = run_name.split('/')[0]
            checkpoint_name = f"{run_title}-{iteration:05d}-{time.strftime('%m%d_%H%M%S', start_localtime)}"
            checkpoint_path = f"runs/{run_name}/{checkpoint_name}"
            save_checkpoint(
                checkpoint_path, agent, muscle_learner, args.env_file,
                save_full_state=args.save_optimizer,
                optimizer=optimizer,
                iteration=iteration,
                global_step=global_step,
                args=args,
                resumed_from=resumed_from,
            )

        # ===== TENSORBOARD LOGGING (at end of iteration to prevent blocking) =====
        if iteration % args.log_interval == 0:
            writer.add_scalar("perf/rollout_time_ms", rollout_time, global_step)
            writer.add_scalar("perf/ppo_learn_time_ms", ppo_learn_time, global_step)
            writer.add_scalar("perf/weight_sync_time_ms", sync_time, global_step)

            if muscle_loss is not None:
                writer.add_scalar("muscle/loss", muscle_loss['loss_muscle'], global_step)
                writer.add_scalar("muscle/loss_target", muscle_loss['loss_target'], global_step)
                writer.add_scalar("muscle/loss_reg", muscle_loss['loss_reg'], global_step)
                writer.add_scalar("muscle/loss_act", muscle_loss['loss_act'], global_step)
                writer.add_scalar("perf/muscle_time_ms", muscle_learn_time, global_step)

            if disc_loss is not None:
                writer.add_scalar("disc/loss", disc_loss['loss_disc'], global_step)
                writer.add_scalar("disc/loss_pos", disc_loss['loss_pos'], global_step)
                writer.add_scalar("disc/loss_neg", disc_loss['loss_neg'], global_step)
                writer.add_scalar("disc/loss_gp_raw", disc_loss['loss_gp_raw'], global_step)
                writer.add_scalar("disc/loss_gp_scaled", disc_loss['loss_gp_scaled'], global_step)
                writer.add_scalar("disc/accuracy", disc_loss['accuracy'], global_step)
                writer.add_scalar("disc/replay_buffer_size", disc_loss['replay_buffer_size'], global_step)
                writer.add_scalar("perf/disc_time_ms", disc_learn_time, global_step)
                # Diagnostic metrics: D_fake distribution
                writer.add_scalar("disc/D_fake_mean", disc_loss['D_fake_mean'], global_step)
                writer.add_scalar("disc/D_fake_std", disc_loss['D_fake_std'], global_step)
                writer.add_scalar("disc/D_fake_p10", disc_loss['D_fake_p10'], global_step)
                writer.add_scalar("disc/D_fake_p90", disc_loss['D_fake_p90'], global_step)
                # Logit margin
                writer.add_scalar("disc/logit_margin", disc_loss['logit_margin'], global_step)
                # r_disc reward distribution
                writer.add_scalar("info/r_disc_mean", disc_loss['r_disc_mean'], global_step)
                writer.add_scalar("info/r_disc_std", disc_loss['r_disc_std'], global_step)
                writer.add_scalar("info/r_disc_p10", disc_loss['r_disc_p10'], global_step)
                writer.add_scalar("info/r_disc_p90", disc_loss['r_disc_p90'], global_step)
                # GP gradient norm stats
                writer.add_scalar("disc/grad_norm_mean", disc_loss['grad_norm_mean'], global_step)
                writer.add_scalar("disc/grad_norm_std", disc_loss['grad_norm_std'], global_step)

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
            writer.add_scalar("losses/log_std_mean", agent.actor_logstd.mean().item(), global_step)

            # Performance metrics
            iteration_time = (time.perf_counter() - epoch_start) * 1000
            writer.add_scalar("perf/iteration_time_ms", iteration_time, global_step)
            writer.add_scalar("perf/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Log averaged info metrics from C++ accumulation
            if 'info' in trajectory:
                for key, avg_value in trajectory['info'].items():
                    writer.add_scalar(f"info/{key}", avg_value, global_step)

            # Log episode statistics from C++ accumulation
            if 'avg_episode_return' in trajectory:
                writer.add_scalar("charts/episodic_return", trajectory['avg_episode_return'], global_step)
                writer.add_scalar("charts/episodic_length", trajectory['avg_episode_length'], global_step)

            # Log system resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            writer.add_scalar("system/cpu_percent", cpu_percent, global_step)
            writer.add_scalar("system/memory_used_gb", memory.used / (1024**3), global_step)
            writer.add_scalar("system/memory_percent", memory.percent, global_step)
            writer.add_scalar("system/memory_available_gb", memory.available / (1024**3), global_step)

    # Save final models to 'last' subdirectory
    model_path = f"runs/{run_name}/last"
    save_checkpoint(
        model_path, agent, muscle_learner, args.env_file,
        save_full_state=args.save_optimizer,
        optimizer=optimizer,
        iteration=args.num_iterations,
        global_step=global_step,
        args=args,
        resumed_from=resumed_from,
    )

    writer.close()

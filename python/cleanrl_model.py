"""
CleanRL Checkpoint Loader for Viewer

Loads checkpoints saved by ppo/ppo_hierarchical.py and provides compatible
interface with Ray checkpoint loader (PolicyNN/MuscleNN).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.distributions.normal import Normal


class NoFilter:
    """
    Simple pass-through observation filter (no normalization).

    Replaces Ray's NoFilter to avoid importing Ray dependencies.
    """
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, obs, update=False):
        """Pass through observations unchanged."""
        return obs

    def copy(self):
        """Return a copy of this filter."""
        return NoFilter(self.shape)


def weights_init(m):
    """Initialize weights for neural network layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class MuscleNN(nn.Module):
    """
    Muscle control network (copied from ray_model.py to avoid Ray import).

    Compatible with both Ray and CleanRL checkpoints.
    """
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles, is_cpu=False, is_cascaded=False):
        super(MuscleNN, self).__init__()

        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles
        self.isCuda = False
        self.isCascaded = is_cascaded

        num_h1 = 256
        num_h2 = 256
        num_h3 = 256

        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs + num_dofs +
                      (num_muscles + 1 if self.isCascaded else 0), num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
        )

        # Normalization
        self.std_muscle_tau = torch.ones(self.num_total_muscle_related_dofs) * 200
        self.std_tau = torch.ones(self.num_dofs) * 200

        if torch.cuda.is_available() and not is_cpu:
            self.isCuda = True
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.fc.apply(weights_init)

    def forward(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
        return torch.relu(torch.tanh(out))

    def unnormalized_no_grad_forward(self, muscle_tau, tau, prev_out=None, out_np=False, weight=None):
        with torch.no_grad():
            if isinstance(self.std_muscle_tau, torch.Tensor) and not isinstance(muscle_tau, torch.Tensor):
                muscle_tau = torch.FloatTensor(muscle_tau).to(self.device)

            if isinstance(self.std_tau, torch.Tensor) and not isinstance(tau, torch.Tensor):
                tau = torch.FloatTensor(tau).to(self.device)

            muscle_tau = muscle_tau / self.std_muscle_tau
            tau = tau / self.std_tau

            out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))

            if out_np:
                out = out.cpu().numpy()

            return out

    def forward_filter(self, unnormalized_activation):
        """Apply activation function to unnormalized muscle activations."""
        return torch.relu(torch.tanh(torch.FloatTensor(unnormalized_activation))).cpu().numpy()

    def get_activation(self, muscle_tau, tau):
        act = self.forward(torch.FloatTensor(muscle_tau.reshape(1, -1)).to(self.device),
                           torch.FloatTensor(tau.reshape(1, -1)).to(self.device))
        return act.cpu().detach().numpy()[0]

    def to(self, *args, **kwargs):
        """Override to() to update self.device"""
        self = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        else:
            self.device = next(self.parameters()).device
        return self


def _detect_cleanrl_checkpoint(path):
    """
    Detect if path is a CleanRL checkpoint directory.

    Args:
        path: Path to potential checkpoint directory

    Returns:
        bool: True if CleanRL checkpoint detected
    """
    if not os.path.isdir(path):
        return False

    agent_pt = os.path.join(path, "agent.pt")
    return os.path.exists(agent_pt)


def _load_metadata_yaml(checkpoint_dir):
    """
    Load metadata.yaml from CleanRL checkpoint directory.
    Falls back to default A2_sep.yaml if not found.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        str: YAML metadata string
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.yaml")

    if not os.path.exists(metadata_path):
        # Use default environment config
        default_config = "data/env/A2_sep.yaml"
        if os.path.exists(default_config):
            print(f"Warning: metadata.yaml not found in {checkpoint_dir}")
            print(f"Using default environment config: {default_config}")
            with open(default_config, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(
                f"metadata.yaml not found in {checkpoint_dir} and default config not found at {default_config}\n"
                f"Please copy environment config: cp data/env/A2_sep.yaml {checkpoint_dir}/metadata.yaml"
            )

    with open(metadata_path, 'r') as f:
        return f.read()


class CleanRLPolicyWrapper:
    """
    Wrapper to make CleanRL Agent compatible with viewer's PolicyNN interface.

    The viewer expects PolicyNN with methods:
    - get_action(obs, is_random=False) -> action array
    - get_value(obs) -> value scalar
    - get_filter() -> observation filter
    """

    def __init__(self, agent_state_dict, num_states, num_actions, device):
        """
        Initialize from CleanRL Agent state_dict.

        Args:
            agent_state_dict: torch.load(agent.pt) result
            num_states: Observation space dimension
            num_actions: Action space dimension
            device: torch device (cpu/cuda)
        """
        self.device = device
        self.num_states = num_states
        self.num_actions = num_actions

        # Reconstruct actor_mean network (3 hidden layers of 512 units)
        self.actor_mean = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        ).to(device)

        # Load actor_mean weights
        actor_mean_state = {
            k.replace('actor_mean.', ''): v
            for k, v in agent_state_dict.items()
            if k.startswith('actor_mean.')
        }
        self.actor_mean.load_state_dict(actor_mean_state)
        self.actor_mean.eval()

        # Load actor_logstd parameter
        self.actor_logstd = agent_state_dict['actor_logstd'].to(device)

        # Reconstruct critic network (3 hidden layers of 512 units)
        self.critic = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        ).to(device)

        # Load critic weights
        critic_state = {
            k.replace('critic.', ''): v
            for k, v in agent_state_dict.items()
            if k.startswith('critic.')
        }
        self.critic.load_state_dict(critic_state)
        self.critic.eval()

        # Create dummy NoFilter (CleanRL doesn't use observation filters)
        self.filter = NoFilter((num_states,))

    def get_action(self, obs, is_random=False):
        """
        Get action from observation (matches PolicyNN interface).

        Args:
            obs: Observation array
            is_random: If True, sample from distribution; if False, use mean

        Returns:
            numpy array: Action
        """
        obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            action_mean = self.actor_mean(obs_tensor)

            if is_random:
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                dist = Normal(action_mean, action_std)
                action = dist.sample()
            else:
                action = action_mean

        return action.cpu().numpy()

    def get_value(self, obs):
        """
        Get value estimate from observation (matches PolicyNN interface).

        Args:
            obs: Observation array

        Returns:
            float: Value estimate
        """
        obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            value = self.critic(obs_tensor)

        return float(value.cpu().numpy())

    def get_filter(self):
        """Get observation filter (matches PolicyNN interface)."""
        return self.filter


def _load_cleanrl_checkpoint(checkpoint_dir, num_states, num_actions, use_mcn, device):
    """
    Load CleanRL checkpoint and return (policy, muscle_state_dict) tuple.

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_states: Observation space dimension
        num_actions: Action space dimension
        use_mcn: Whether to load muscle network state_dict
        device: torch device (cpu/cuda)

    Returns:
        tuple: (policy_wrapper, muscle_state_dict or None)
            - policy_wrapper: CleanRLPolicyWrapper for joint control
            - muscle_state_dict: Dict of muscle network weights (to be loaded into C++ MuscleNN)
    """
    # Load agent weights
    agent_path = os.path.join(checkpoint_dir, "agent.pt")
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"agent.pt not found in {checkpoint_dir}")

    agent_state_dict = torch.load(agent_path, map_location=device)

    # Create policy wrapper
    policy = CleanRLPolicyWrapper(agent_state_dict, num_states, num_actions, device)

    # Load muscle network state_dict if requested (to be loaded into C++ MuscleNN)
    muscle_state_dict = None
    if use_mcn:
        muscle_path = os.path.join(checkpoint_dir, "muscle.pt")
        if os.path.exists(muscle_path):
            # Load muscle state_dict (will be passed to C++ via setMuscleNetworkWeight)
            muscle_state_dict = torch.load(muscle_path, map_location=device)

    return policy, muscle_state_dict


def loading_network(checkpoint_dir, num_states, num_actions, use_mcn, device="cpu",
                    num_muscles=None, num_muscle_dofs=None, num_actuator_action=None):
    """
    Main entry point for loading CleanRL checkpoint.

    Compatible interface with ray_model.loading_network().

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_states: Observation space dimension
        num_actions: Action space dimension
        use_mcn: Whether to load muscle network state_dict
        device: torch device (cpu/cuda)
        num_muscles: (unused, kept for compatibility)
        num_muscle_dofs: (unused, kept for compatibility)
        num_actuator_action: (unused, kept for compatibility)

    Returns:
        tuple: (policy, muscle_state_dict or None)
            - policy: CleanRLPolicyWrapper for joint control
            - muscle_state_dict: Dict of weights to load into C++ MuscleNN via setMuscleNetworkWeight
    """
    return _load_cleanrl_checkpoint(checkpoint_dir, num_states, num_actions, use_mcn, device)


def loading_metadata(checkpoint_dir):
    """
    Load metadata from CleanRL checkpoint.

    Compatible interface with ray_model.loading_metadata().

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        str: XML metadata content from metadata.yaml
    """
    return _load_metadata_yaml(checkpoint_dir)

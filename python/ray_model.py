import torch
import torch.nn as nn
import numpy as np
import dill
import os
import yaml
from dill import Unpickler
from io import BytesIO
from python.dummy import Dummy
from python.log_config import log_verbose
from python.uri_resolver import resolve_path

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.filter import get_filter

# Configure device-aware tensor loading for Ray workers
if not torch.cuda.is_available():
    # Fallback to CPU if no GPU available - avoid duplicate map_location
    original_torch_load = torch.load
    def safe_cpu_load(*args, **kwargs):
        # Remove any existing map_location to avoid conflicts
        kwargs.pop('map_location', None)
        return original_torch_load(*args, **kwargs, map_location='cpu')
    torch.load = safe_cpu_load

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(
    self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class MuscleNN(nn.Module):
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles, is_cpu=False, is_cascaded=False):
        super(MuscleNN, self).__init__()

        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs  # Exclude Joint Root dof
        self.num_muscles = num_muscles
        self.isCuda = False
        self.isCascaded = is_cascaded

        num_h1 = 256
        num_h2 = 256
        num_h3 = 256

        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs+num_dofs +
                      (num_muscles + 1 if self.isCascaded else 0), num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
        )

        # Normalization
        self.std_muscle_tau = torch.ones(
            self.num_total_muscle_related_dofs) * 200
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

    def forward_with_prev_out_wo_relu(self, muscle_tau, tau, prev_out, weight=1.0):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        if type(prev_out) == np.ndarray:
            with torch.no_grad():
                prev_out = torch.FloatTensor(prev_out)
                out = prev_out + weight * \
                    self.fc.forward(
                        torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))
                return out
        else:
            out = prev_out + weight * \
                self.fc.forward(
                    torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))
            return out

    def forward_wo_relu(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
        return out

    def forward(self, muscle_tau, tau):
        return torch.relu(torch.tanh(self.forward_wo_relu(muscle_tau, tau)))

    def forward_with_prev_out(self, muscle_tau, tau, prev_out, weight=1.0):
        return torch.relu(torch.tanh(self.forward_with_prev_out_wo_relu(muscle_tau, tau, prev_out, weight)))

    def unnormalized_no_grad_forward(self, muscle_tau, tau, prev_out=None, out_np=False, weight=None):
        with torch.no_grad():
            if type(self.std_muscle_tau) == torch.Tensor and type(muscle_tau) != torch.Tensor:
                if self.isCuda:
                    muscle_tau = torch.FloatTensor(muscle_tau).cuda()
                else:
                    muscle_tau = torch.FloatTensor(muscle_tau)

            if type(self.std_tau) == torch.Tensor and type(tau) != torch.Tensor:
                if self.isCuda:
                    tau = torch.FloatTensor(tau).cuda()
                else:
                    tau = torch.FloatTensor(tau)

            if type(weight) != type(None):
                if self.isCuda:
                    weight = torch.FloatTensor([weight]).cuda()
                else:
                    weight = torch.FloatTensor([weight])

            muscle_tau = muscle_tau / self.std_muscle_tau
            tau = tau / self.std_tau

            if type(prev_out) == type(None) and type(weight) == type(None):
                out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
            else:
                if self.isCuda:
                    prev_out = torch.FloatTensor(prev_out).cuda()
                else:
                    prev_out = torch.FloatTensor(prev_out)

                if type(weight) == type(None):
                    print('Weight Error')
                    exit(-1)
                out = self.fc.forward(
                    torch.cat([0.5 * prev_out, weight, muscle_tau, tau], dim=-1))

            if out_np:
                out = out.cpu().numpy()

            return out

    def forward_filter(self, unnormalized_activation):
        return torch.relu(torch.tanh(torch.FloatTensor(unnormalized_activation))).cpu().numpy()

    def load(self, path):
        print('load muscle nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(torch.Tensorpath))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        act = self.forward(torch.FloatTensor(muscle_tau.reshape(1, -1)).to(self.device),
                           torch.FloatTensor(tau.reshape(1, -1)).to(self.device))
        return act.cpu().detach().numpy()[0]

    def to(self, *args, **kwargs):
        """Override to() to update self.device"""
        self = super().to(*args, **kwargs)
        # Extract device from args or kwargs
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        else:
            # Infer device from parameters
            self.device = next(self.parameters()).device
        return self


class SimulationNN(nn.Module):
    def __init__(self, num_states, num_actions, learningStd=False):
        nn.Module.__init__(self)
        self.num_states = num_states
        self.num_actions = num_actions

        self.num_h1 = 512
        self.num_h2 = 512
        self.num_h3 = 512

        self.log_std = None
        init_log_std = 1.0 * torch.ones(num_actions)
        init_log_std[18:] *= 0.5 ## For Upper Body

        ## For Cascading 
        init_log_std[-1] = 1.0

        if learningStd:
            self.log_std = nn.Parameter(init_log_std)
        else:
            self.log_std = init_log_std

        self.p_fc = nn.Sequential(
            nn.Linear(self.num_states, self.num_h1),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h1, self.num_h2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h2, self.num_h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h3, self.num_actions),
        )

        self.v_fc = nn.Sequential(
            nn.Linear(self.num_states, self.num_h1),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h1, self.num_h2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h2, self.num_h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h3, 1),
        )

        self.reset()

        if torch.cuda.is_available():
            if not learningStd:
                self.log_std = self.log_std.cuda()
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def reset(self):
        self.p_fc.apply(weights_init)
        self.v_fc.apply(weights_init)

    def forward(self, x):
        p_out = MultiVariateNormal(self.p_fc.forward(x), self.log_std.exp())
        v_out = self.v_fc.forward(x)
        return p_out, v_out

    def load(self, path):
        print('load simulation nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        ts = torch.tensor(s, device=self.device)
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy()

    def get_value(self, s):
        ts = torch.tensor(s, device=self.device)
        _, v = self.forward(ts)
        return v.cpu().detach().numpy()

    def get_random_action(self, s):
        # print(self.log_std)
        ts = torch.tensor(s, device=self.device)
        p, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy()

    def get_noise(self):
        return self.log_std.exp().mean().item()

    def to(self, *args, **kwargs):
        """Override to() to update self.device"""
        self = super().to(*args, **kwargs)
        # Extract device from args or kwargs
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif 'device' in kwargs:
            self.device = torch.device(kwargs['device'])
        else:
            # Infer device from parameters
            self.device = next(self.parameters()).device
        return self


class RolloutNNRay(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, {}, "RolloutNNRay")
        nn.Module.__init__(self)

        self.num_states = np.prod(obs_space.shape)
        self.num_actions = np.prod(action_space.shape)

        self.action_dist_loc = torch.zeros(self.num_actions)
        self.action_dist_scale = torch.zeros(self.num_actions)
        self._value = None
        self.dummy_param = nn.Parameter(torch.ones(self.num_actions))

    # @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)

        action_tensor = None
        if torch.cuda.is_available():
            action_tensor = torch.zeros(
                obs.shape[0], 2 * self.num_actions).cuda()
            self._value = torch.zeros(obs.shape[0], 1).cuda()
        else:
            action_tensor = torch.zeros(obs.shape[0], 2 * self.num_actions)
            self._value = torch.zeros(obs.shape[0], 1)

        return action_tensor, state

    # @override(TorchModelV2)
    def value_function(self):
        return self._value.squeeze(1)  # self._value.squeeze(1)


class SimulationNN_Ray(TorchModelV2, SimulationNN):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(
            self, num_states, num_actions, kwargs['learningStd'])
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, {}, "SimulationNN_Ray")
        num_outputs = 2 * np.prod(action_space.shape)
        self._value = None

    def get_value(self, obs):
        with torch.no_grad():
            _, v = SimulationNN.forward(self, obs)
            return v

    # @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._value = SimulationNN.forward(self, x)
        action_tensor = torch.cat(
            [action_dist.loc, action_dist.scale.log()], dim=1)
        return action_tensor, state

    # @override(TorchModelV2)
    def value_function(self):
        return self._value.squeeze(1)

    def reset(self):
        SimulationNN.reset(self)

    def vf_reset(self):
        SimulationNN.vf_reset(self)

    def pi_reset(self):
        SimulationNN.pi_reset(self)


class PolicyNN:
    def __init__(self, num_states, num_actions, policy_state, filter_state, device, learningStd=False):

        self.policy = SimulationNN(
            num_states, num_actions, learningStd).to(device)

        self.policy.log_std = self.policy.log_std.to(device)
        self.policy.load_state_dict(convert_to_torch_tensor(policy_state))
        self.policy.eval()
        self.filter = filter_state
        # self.cascading_type = cascading_type

    def get_filter(self):
        return self.filter.copy()

    def get_value(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        v = self.policy.get_value(obs)
        return v

    def get_value_function_weight(self):
        return self.policy.value_function_state_dict()

    def get_action(self, obs, is_random=False):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return self.policy.get_action(obs) if not is_random else self.policy.get_random_action(obs)

    def get_filtered_obs(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return obs

    def weight_filter(self, unnormalized, beta):
        scale_factor = 1000.0
        return torch.sigmoid(torch.tensor([scale_factor * (unnormalized - beta)])).numpy()[0]

    def state_dict(self):
        state = {}
        state["weight"] = (self.policy.state_dict())
        state["filter"] = self.filter
        return state

    def soft_load_state_dict(self, _state_dict):
        self.policy.soft_load_state_dict(_state_dict)


def generating_muscle_nn(num_total_muscle_related_dofs, num_dof, num_muscles, is_cpu=True, is_cascaded=False):
    muscle = MuscleNN(num_total_muscle_related_dofs, num_dof,
                      num_muscles, is_cpu, is_cascaded)
    return muscle


def loading_metadata(path):
    """
    Load metadata from checkpoint supporting both Ray 2.0.1 and 2.12.0 formats.

    Args:
        path: File path (Ray 2.0.1) or directory path (Ray 2.12.0)

    Returns:
        dict or None: Metadata dictionary if present, None otherwise

    Examples:
        # Load from Ray 2.0.1 checkpoint (single file)
        metadata = loading_metadata("ray_results/A_knee_mult-016000-1025_112219")

        # Load from Ray 2.12.0 checkpoint (directory)
        metadata = loading_metadata("ray_results/checkpoint_000010")
    """
    # Resolve URI before loading
    resolved_path = resolve_path(path)

    # Detect checkpoint format
    checkpoint_format = _detect_checkpoint_format(resolved_path)

    metadata = None
    if checkpoint_format == "ray_2.0.1":
        # Ray 2.0.1: metadata is at top level of single file
        state = dill.load(open(resolved_path, "rb"))
        metadata = state.get("metadata", None)

    elif checkpoint_format == "ray_2.12.0":
        # Ray 2.12.0: metadata is in algorithm_state.pkl
        algo_state_path = os.path.join(resolved_path, "algorithm_state.pkl")
        if os.path.exists(algo_state_path):
            state = dill.load(open(algo_state_path, "rb"))
            metadata = state.get("metadata", None)
        else:
            log_verbose(f"[Warning] algorithm_state.pkl not found at {algo_state_path}")
            metadata = None

    else:
        raise ValueError(
            f"Unknown checkpoint format at {resolved_path}. "
            f"Expected either a single file (Ray 2.0.1) or a directory with "
            f"algorithm_state.pkl (Ray 2.12.0)"
        )

    # Check if metadata is empty, dict type, or should use fallback
    if not metadata or isinstance(metadata, dict):
        log_verbose(f"[Warning] Metadata is empty or dict type in checkpoint at {resolved_path}, loading fallback from data/A_knee.yaml")
        fallback_path = resolve_path("@data/A_knee.yaml")
        # Return the YAML file content as string (not dict, not path)
        with open(fallback_path, 'r') as f:
            yaml_content = f.read()
        log_verbose(f"[Python] Loaded fallback metadata content from {fallback_path}")
        return yaml_content

    return metadata


class SelectiveUnpickler(Unpickler):
    def __init__(self, file):
        super().__init__(file)
        self._allowed_classes = {
            ('builtins', 'dict'),
            ('builtins', 'list'),
            ('builtins', 'str'),
            ('builtins', 'int'),
            ('builtins', 'float'),
            ('builtins', 'bool'),
            ('torch', 'Tensor'),
            ('numpy', 'ndarray'),
            ('numpy.core.numeric', '_frombuffer'),
            ('numpy', 'dtype'),
            ('ray.rllib.utils.filter', 'NoFilter'),
        }

    def find_class(self, module, name):
        if (module, name) in self._allowed_classes:
            return super().find_class(module, name)
        else:
            # print(f"Class {module}.{name} is not allowed to be unpickled.")
            return Dummy


def _detect_checkpoint_format(path):
    """
    Detect checkpoint format from path.

    Args:
        path: Resolved file or directory path

    Returns:
        str: "ray_2.0.1" for single file, "ray_2.12.0" for directory, "unknown" otherwise
    """
    if os.path.isfile(path):
        # Single file = Ray 2.0.1 format
        return "ray_2.0.1"

    elif os.path.isdir(path):
        # Check for Ray 2.12.0 directory structure
        algo_state = os.path.join(path, "algorithm_state.pkl")
        policies_dir = os.path.join(path, "policies")

        # Valid if either algorithm_state or policies directory exists
        if os.path.exists(algo_state) or os.path.isdir(policies_dir):
            return "ray_2.12.0"

    return "unknown"


def _find_policy_state_file(checkpoint_dir):
    """
    Find policy_state.pkl in Ray 2.12.0 checkpoint directory.

    Searches in order:
        1. checkpoint_dir/policy_state.pkl (if directly provided)
        2. checkpoint_dir/policies/default_policy/policy_state.pkl
        3. checkpoint_dir/policies/*/policy_state.pkl (first match)

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        str: Path to policy_state.pkl

    Raises:
        FileNotFoundError: If policy_state.pkl cannot be found
    """
    # Check direct path
    direct_path = os.path.join(checkpoint_dir, "policy_state.pkl")
    if os.path.exists(direct_path):
        return direct_path

    # Check default policy path
    default_path = os.path.join(checkpoint_dir, "policies", "default_policy", "policy_state.pkl")
    if os.path.exists(default_path):
        return default_path

    # Search in policies subdirectory
    policies_dir = os.path.join(checkpoint_dir, "policies")
    if os.path.isdir(policies_dir):
        for policy_name in os.listdir(policies_dir):
            policy_path = os.path.join(policies_dir, policy_name, "policy_state.pkl")
            if os.path.exists(policy_path):
                log_verbose(f"[Python] Found policy state at {policy_path}")
                return policy_path

    raise FileNotFoundError(
        f"Cannot find policy_state.pkl in {checkpoint_dir}. "
        f"Expected at policies/default_policy/policy_state.pkl or policies/*/policy_state.pkl"
    )


def _create_muscle_network_from_weights(muscle_weights, env_config):
    """
    Create MuscleNN instance from weights dictionary and environment config.

    Args:
        muscle_weights: Dictionary of muscle network weights
        env_config: Environment configuration dictionary

    Returns:
        MuscleNN: Loaded muscle network instance
    """
    # Handle both typo variants of actuator action key
    if 'num_actuactor_action' in env_config:
        num_actuator_action = env_config['num_actuactor_action']
    else:
        num_actuator_action = env_config['num_actuator_action']

    num_muscles = env_config['num_muscles']
    num_muscle_dofs = env_config['num_muscle_dofs']
    is_cascaded = env_config.get('cascading', False)

    muscle = MuscleNN(
        num_muscle_dofs,
        num_actuator_action,
        num_muscles,
        is_cpu=True,
        is_cascaded=is_cascaded
    )
    muscle.load_state_dict(convert_to_torch_tensor(muscle_weights))

    return muscle


def _load_checkpoint_ray_2_0_1(checkpoint_path, num_states, num_actions, use_mcn, device):
    """
    Load checkpoint in Ray 2.0.1 format (single file).

    Structure:
        state['worker'] → worker_state (pickled bytes)
        worker_state['state']['default_policy']['weights'] → policy weights
        worker_state['filters']['default_policy'] → observation filter
        state['muscle'] → muscle network weights
        state['metadata'] → custom metadata

    Args:
        checkpoint_path: Path to checkpoint file
        num_states: Number of observation dimensions
        num_actions: Number of action dimensions
        use_mcn: Whether to load muscle network
        device: Target device

    Returns:
        tuple: (policy, muscle) - PolicyNN and MuscleNN instances
    """
    log_verbose(f"[Python] Loading Ray 2.0.1 checkpoint from {checkpoint_path}")

    # Load checkpoint file
    state = dill.load(open(checkpoint_path, "rb"))

    # Unpickle worker state
    worker_state = SelectiveUnpickler(BytesIO(state['worker'])).load()
    policy_state = worker_state["state"]['default_policy']['weights']
    filter_state = worker_state["filters"]['default_policy']

    # Create policy network
    device = torch.device(device)
    learningStd = ('log_std' in policy_state.keys())
    policy = PolicyNN(num_states, num_actions, policy_state, filter_state, device, learningStd)

    # Load muscle network if requested
    muscle = None
    if use_mcn and 'muscle' in state:
        # Try to get env_config from worker state
        env_config = None
        if 'policy_config' in worker_state and 'env_config' in worker_state['policy_config']:
            env_config = worker_state['policy_config']['env_config']
        elif 'config' in worker_state and 'env_config' in worker_state['config']:
            env_config = worker_state['config']['env_config']

        if env_config is None:
            log_verbose("[Warning] No env_config found in checkpoint, muscle network disabled")
        else:
            # Handle cascading flag from top-level state
            muscle_weights = state['muscle']
            env_config_with_cascading = env_config.copy()
            if 'cascading' not in env_config_with_cascading and 'cascading' in state:
                env_config_with_cascading['cascading'] = state['cascading']

            muscle = _create_muscle_network_from_weights(muscle_weights, env_config_with_cascading)
            log_verbose("[Python] Muscle network loaded successfully")

    return policy, muscle


def _load_checkpoint_ray_2_12_0(checkpoint_dir, num_states, num_actions, use_mcn, device):
    """
    Load checkpoint in Ray 2.12.0 format (directory with multiple files).

    Structure:
        checkpoint_dir/
        ├─ algorithm_state.pkl
        │  ├─ muscle: muscle network weights
        │  ├─ muscle_optimizer: optimizer state
        │  ├─ config.env_config: environment configuration
        │  └─ metadata: custom metadata
        └─ policies/default_policy/
           └─ policy_state.pkl
              ├─ weights: policy network weights
              ├─ policy_spec.config.env_config: environment configuration
              └─ policy_spec.config.observation_filter: filter type

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_states: Number of observation dimensions
        num_actions: Number of action dimensions
        use_mcn: Whether to load muscle network
        device: Target device

    Returns:
        tuple: (policy, muscle) - PolicyNN and MuscleNN instances
    """
    log_verbose(f"[Python] Loading Ray 2.12.0 checkpoint from {checkpoint_dir}")

    # Find and load policy state
    policy_state_path = _find_policy_state_file(checkpoint_dir)
    policy_state = dill.load(open(policy_state_path, "rb"))

    # Extract policy weights
    policy_weights = policy_state['weights']
    learningStd = ('log_std' in policy_weights.keys())

    # Extract filter from policy spec
    filter_config = policy_state['policy_spec']['config']['observation_filter']
    filter_state = get_filter(filter_config, (num_states,))

    # Create policy network
    device = torch.device(device)
    policy = PolicyNN(num_states, num_actions, policy_weights, filter_state, device, learningStd)
    log_verbose("[Python] Policy network loaded successfully")

    # Load muscle network if requested
    muscle = None
    if use_mcn:
        algo_state_path = os.path.join(checkpoint_dir, "algorithm_state.pkl")
        if os.path.exists(algo_state_path):
            algo_state = dill.load(open(algo_state_path, "rb"))

            if 'muscle' in algo_state:
                # Get env_config from policy_state (preferred) or algo_state
                env_config = policy_state['policy_spec']['config'].get('env_config')
                if env_config is None and hasattr(algo_state.get('config'), 'env_config'):
                    env_config = algo_state['config'].env_config

                if env_config is not None:
                    muscle = _create_muscle_network_from_weights(algo_state['muscle'], env_config)
                    log_verbose("[Python] Muscle network loaded successfully")
                else:
                    log_verbose("[Warning] No env_config found for muscle network")
            else:
                log_verbose("[Warning] No muscle weights found in algorithm_state.pkl")
        else:
            log_verbose(f"[Warning] algorithm_state.pkl not found at {algo_state_path}, muscle network disabled")

    return policy, muscle


def loading_network(path, num_states, num_actions, use_mcn, device="cpu"):
    """
    Unified checkpoint loader supporting Ray 2.0.1 and 2.12.0 formats.

    Automatically detects checkpoint format and applies appropriate loading logic:
    - Single file → Ray 2.0.1 format (legacy)
    - Directory → Ray 2.12.0 format (current)

    Args:
        path: File path (Ray 2.0.1) or directory path (Ray 2.12.0)
        num_states: Number of observation dimensions
        num_actions: Number of action dimensions
        use_mcn: Whether to load muscle control network
        device: Target device ("cpu" or "cuda")

    Returns:
        tuple: (policy, muscle) where:
            - policy: PolicyNN instance with loaded weights
            - muscle: MuscleNN instance (if use_mcn=True) or None

    Examples:
        # Load Ray 2.0.1 checkpoint (single file)
        policy, muscle = loading_network(
            "ray_results/A_knee_mult-016000-1025_112219",
            num_states=506, num_actions=51, use_mcn=True
        )

        # Load Ray 2.12.0 checkpoint (directory)
        policy, muscle = loading_network(
            "ray_results/checkpoint_000010",
            num_states=506, num_actions=51, use_mcn=True
        )
    """
    # Resolve URI before loading
    resolved_path = resolve_path(path)

    # Detect checkpoint format
    checkpoint_format = _detect_checkpoint_format(resolved_path)

    if checkpoint_format == "ray_2.0.1":
        return _load_checkpoint_ray_2_0_1(resolved_path, num_states, num_actions, use_mcn, device)

    elif checkpoint_format == "ray_2.12.0":
        return _load_checkpoint_ray_2_12_0(resolved_path, num_states, num_actions, use_mcn, device)

    else:
        raise ValueError(
            f"Unknown checkpoint format at {resolved_path}. "
            f"Expected either a single file (Ray 2.0.1) or a directory with "
            f"algorithm_state.pkl or policies/ subdirectory (Ray 2.12.0)"
        )

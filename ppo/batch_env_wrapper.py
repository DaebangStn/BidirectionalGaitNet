"""
Gymnasium VectorEnv-compatible wrapper for BatchEnv (C++ parallel environment).

This module wraps the BatchEnv C++ binding to provide a standard Gymnasium
VectorEnv interface compatible with ppo_hierarchical.py.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, List
import sys
import os

# Add ppo directory to path for batchenv import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import batchenv
except ImportError as e:
    raise ImportError(
        f"Failed to import batchenv module: {e}\n"
        "Make sure batchenv.so is built with: ninja -C build/release batchenv.so"
    )


class BatchEnvWrapper(gym.vector.VectorEnv):
    """
    Gymnasium VectorEnv wrapper for BatchEnv C++ parallel environment.

    Provides standard VectorEnv interface:
    - reset() returns (obs, infos)
    - step() returns (obs, rewards, terminations, truncations, infos)
    - Muscle tuple collection via get_muscle_tuples()
    - Weight updates via update_muscle_weights()
    """

    def __init__(self, env_file: str, num_envs: int):
        """
        Initialize the batched parallel environment.

        Args:
            env_file: Path to environment configuration YAML file
            num_envs: Number of parallel environments
        """
        # Load YAML content
        with open(env_file, 'r') as f:
            yaml_content = f.read()

        # Create C++ BatchEnv
        self.batch_env = batchenv.BatchEnv(yaml_content, num_envs)

        # Get dimensions
        self.num_envs = num_envs
        self.obs_dim = self.batch_env.obs_dim()
        self.action_dim = self.batch_env.action_dim()

        # Create observation and action spaces
        self.single_observation_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.single_action_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # Set VectorEnv attributes (no __init__ call needed for VectorEnv)
        # VectorEnv is just a protocol/interface, not a concrete class with __init__

        # Hierarchical control configuration
        self.is_hierarchical = self.batch_env.is_hierarchical()
        self.use_cascading = self.batch_env.use_cascading() if self.is_hierarchical else False

        # Episode tracking for infos
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments in parallel.

        Args:
            seed: Random seed (not used, environment handles seeding)
            options: Additional options (not used)

        Returns:
            observations: Array of shape (num_envs, obs_dim)
            infos: Dictionary with empty lists (no episodes finished on reset)
        """
        # Reset episode tracking
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0

        # Call C++ parallel reset
        observations = self.batch_env.reset()

        # Return with empty infos (Gymnasium VectorEnv format)
        infos = {}

        return observations, infos

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step all environments in parallel with given actions.

        Args:
            actions: Action array of shape (num_envs, action_dim)

        Returns:
            observations: Array of shape (num_envs, obs_dim)
            rewards: Array of shape (num_envs,)
            terminations: Array of shape (num_envs,) - environment ended
            truncations: Array of shape (num_envs,) - episode truncated
            infos: Dictionary with episode statistics
        """
        # Validate action shape
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Invalid action shape: expected ({self.num_envs}, {self.action_dim}), "
                f"got {actions.shape}"
            )

        # Ensure float32
        if actions.dtype != np.float32:
            actions = actions.astype(np.float32)

        # Call C++ parallel step
        observations, rewards, dones = self.batch_env.step(actions)

        # Update episode tracking
        self.episode_returns += rewards
        self.episode_lengths += 1

        # Convert dones to terminations/truncations
        # BatchEnv returns combined done flag, we treat all as terminations
        terminations = dones.astype(bool)
        truncations = np.zeros(self.num_envs, dtype=bool)

        # Build infos with episode statistics
        infos = {}

        # Add final episode info for environments that finished
        if np.any(terminations):
            infos["final_info"] = []
            for i in range(self.num_envs):
                if terminations[i]:
                    episode_info = {
                        "episode": {
                            "r": float(self.episode_returns[i]),
                            "l": int(self.episode_lengths[i])
                        }
                    }
                    infos["final_info"].append(episode_info)

                    # Reset tracking for this environment
                    self.episode_returns[i] = 0.0
                    self.episode_lengths[i] = 0
                else:
                    infos["final_info"].append(None)

        return observations, rewards, terminations, truncations, infos

    def close(self):
        """
        Clean up environment resources.
        """
        # C++ destructor will handle cleanup
        pass

    def get_muscle_tuples(self) -> List:
        """
        Get collected muscle training tuples from all environments.

        Returns:
            List of muscle tuple buffers, one per environment.
            Each buffer contains [tau_des, JtA_reduced, JtA] lists
            (+ [prev_out, weight] for cascading mode)
        """
        if not self.is_hierarchical:
            return []

        return self.batch_env.get_muscle_tuples()

    def update_muscle_weights(self, state_dict: Dict) -> None:
        """
        Update muscle network weights in all environments.

        Args:
            state_dict: PyTorch state_dict with muscle network parameters
        """
        if not self.is_hierarchical:
            return

        self.batch_env.update_muscle_weights(state_dict)

    def call(self, method_name: str, *args, **kwargs):
        """
        Call a method on all environments (VectorEnv compatibility).

        Args:
            method_name: Name of the method to call
            *args, **kwargs: Arguments to pass to the method

        Returns:
            List of results from each environment
        """
        # Query methods - return list with same value for all envs (queried from first env)
        if method_name == "is_hierarchical":
            result = self.is_hierarchical
            return [result] * self.num_envs
        elif method_name == "use_cascading":
            result = self.use_cascading
            return [result] * self.num_envs
        elif method_name == "getNumActuatorAction":
            result = self.batch_env.getNumActuatorAction()
            return [result] * self.num_envs
        elif method_name == "getNumMuscles":
            result = self.batch_env.getNumMuscles()
            return [result] * self.num_envs
        elif method_name == "getNumMuscleDof":
            result = self.batch_env.getNumMuscleDof()
            return [result] * self.num_envs

        # Muscle tuple collection - return list of buffers per environment
        elif method_name == "get_muscle_tuples":
            return self.get_muscle_tuples()

        # Muscle weight update - broadcast to all environments
        elif method_name == "update_muscle_weights":
            self.update_muscle_weights(*args, **kwargs)
            return [None] * self.num_envs

        else:
            raise AttributeError(f"BatchEnvWrapper has no method '{method_name}'")

    @property
    def unwrapped(self):
        """Return the underlying BatchEnv."""
        return self.batch_env


def make_batch_env(env_file: str, num_envs: int) -> BatchEnvWrapper:
    """
    Factory function to create BatchEnvWrapper.

    Args:
        env_file: Path to environment configuration YAML file
        num_envs: Number of parallel environments

    Returns:
        BatchEnvWrapper instance
    """
    return BatchEnvWrapper(env_file, num_envs)

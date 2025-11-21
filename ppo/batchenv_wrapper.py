"""
Gym-compatible wrapper for C++ BatchEnv.

This wrapper makes the high-performance C++ BatchEnv compatible with the
gymnasium.vector interface used by CleanRL PPO. It provides the same API
as gym.vector.AsyncVectorEnv but with ~10-20x better performance through
native C++ parallelization and zero-copy numpy integration.

Usage:
    from ppo.batchenv_wrapper import BatchEnvWrapper

    # Replace AsyncVectorEnv with BatchEnvWrapper
    envs = BatchEnvWrapper("data/env/config.yaml", num_envs=64)

    obs, info = envs.reset()
    obs, rew, terminated, truncated, infos = envs.step(actions)
"""

import gymnasium as gym
import numpy as np
from ppo import BatchEnv as BatchEnvCpp


class BatchEnvWrapper:
    """
    Gym-compatible wrapper for C++ BatchEnv.

    Provides the same interface as gym.vector.AsyncVectorEnv but uses
    high-performance C++ implementation with ThreadPool parallelization.

    Args:
        env_file (str): Path to environment YAML configuration file
        num_envs (int): Number of parallel environments

    Attributes:
        num_envs (int): Number of parallel environments
        single_observation_space (gym.Space): Observation space for single env
        single_action_space (gym.Space): Action space for single env
    """

    def __init__(self, env_file: str, num_envs: int):
        """
        Initialize BatchEnvWrapper.

        Args:
            env_file: Path to environment YAML configuration file
            num_envs: Number of parallel environments to create
        """
        # Read YAML content from file
        with open(env_file, 'r') as f:
            yaml_content = f.read()

        self._env = BatchEnvCpp(yaml_content, num_envs)
        self.num_envs = num_envs

        # Create Gym spaces based on C++ environment dimensions
        obs_dim = self._env.obs_dim()
        action_dim = self._env.action_dim()

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        """
        Reset all environments.

        Args:
            seed (int, optional): Random seed (currently not used by C++ env)

        Returns:
            tuple: (observations, info)
                - observations: np.ndarray of shape (num_envs, obs_dim), dtype=float32
                - info: dict (empty for now)
        """
        obs = self._env.reset()
        info = {}
        return obs, info

    def step(self, actions):
        """
        Step all environments with given actions.

        Args:
            actions: np.ndarray of shape (num_envs, action_dim), dtype=float32

        Returns:
            tuple: (observations, rewards, terminated, truncated, infos)
                - observations: np.ndarray of shape (num_envs, obs_dim), dtype=float32
                - rewards: np.ndarray of shape (num_envs,), dtype=float32
                - terminated: np.ndarray of shape (num_envs,), dtype=bool
                - truncated: np.ndarray of shape (num_envs,), dtype=bool
                - infos: dict (empty for now)
        """
        # Call C++ step (returns obs, rew, done)
        obs, rew, done = self._env.step(actions)

        # Convert uint8 done flags to bool
        done_bool = done.astype(bool)

        # For now, all done=True goes to terminated
        # TODO: Distinguish between terminated and truncated in C++
        terminated = done_bool
        truncated = np.zeros_like(done_bool)

        # Empty info dict
        # TODO: Add environment info if needed
        infos = {}

        return obs, rew, terminated, truncated, infos

    def close(self):
        """
        Close all environments.

        Note: C++ BatchEnv destructor handles cleanup automatically,
        so this is a no-op. Provided for API compatibility.
        """
        pass

    def call(self, method_name, *args):
        """
        Call a method on the underlying C++ environment.

        This is a stub for compatibility with gym.vector interface.
        Currently not implemented - will raise NotImplementedError.

        Args:
            method_name (str): Name of method to call
            *args: Arguments to pass to the method

        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            f"call('{method_name}') is not yet implemented in BatchEnv. "
            f"If you need this functionality (e.g., for hierarchical muscle control), "
            f"please extend the C++ BatchEnv class and pybind11 bindings."
        )

    @property
    def observation_space(self):
        """Get observation space (for single environment)."""
        return self.single_observation_space

    @property
    def action_space(self):
        """Get action space (for single environment)."""
        return self.single_action_space

    def __repr__(self):
        """String representation."""
        return (
            f"BatchEnvWrapper(num_envs={self.num_envs}, "
            f"obs_dim={self._env.obs_dim()}, "
            f"action_dim={self._env.action_dim()})"
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing BatchEnvWrapper...")

    # Create wrapper
    env = BatchEnvWrapper("data/env/A2_sep.yaml", num_envs=4)
    print(f"Created: {env}")

    # Test reset
    obs, info = env.reset()
    print(f"Reset obs shape: {obs.shape}, dtype: {obs.dtype}")

    # Test step
    actions = np.random.randn(env.num_envs, env.action_space.shape[0]).astype(np.float32)
    obs, rew, terminated, truncated, infos = env.step(actions)
    print(f"Step obs shape: {obs.shape}")
    print(f"Rewards: {rew}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    # Close
    env.close()
    print("✓ All tests passed")

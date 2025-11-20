"""
Gymnasium-compatible environment wrapper for CleanRL PPO.

This module wraps the GymEnvManager C++ binding to provide a standard
Gymnasium interface with hierarchical muscle control support.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ppo import GymEnvManager


class HierarchicalEnv(gym.Env):
    """
    Gymnasium environment wrapper for hierarchical muscle control.

    Implements Option C design:
    - Standard Gymnasium step() interface
    - Muscle substeps hidden inside C++ environment
    - Muscle tuple collection via get_muscle_tuples()
    - Weight updates via update_muscle_weights()
    """

    metadata = {"render_modes": []}

    def __init__(self, env_file: str):
        """
        Initialize the hierarchical environment.

        Args:
            env_file: Path to environment configuration file
        """
        super().__init__()

        # Load environment content
        with open(env_file, 'r') as f:
            env_content = f.read()

        # Create C++ environment
        self.env = GymEnvManager(env_content)

        # Get environment configuration
        self.num_state = None  # Will be set after first reset
        self.num_action = self.env.getNumAction()

        # Define observation and action spaces
        # Observation space will be set after first reset when we know the state size
        self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(1,),  # Placeholder, will be updated
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(self.num_action,),
            dtype=np.float32
        )

        # Hierarchical control configuration
        self.is_two_level = self.env.isTwoLevelController()
        self.use_cascading = self.env.getUseCascading() if self.is_two_level else False

        # Environment metadata
        self.env_metadata = self.env.getMetadata()

        # Create spec for Gymnasium
        self.spec = gym.envs.registration.EnvSpec(
            id="HierarchicalEnv-v0",
            max_episode_steps=10000  # Will be controlled by training loop
        )

        # Parameter update counter (from ray_env.py)
        self.param_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Update parameters periodically (from ray_env.py pattern)
        if self.param_count > 300:
            self.env.updateParamState()
            self.param_count = 0

        # Reset environment
        obs, info = self.env.reset()

        # Update observation space on first reset
        if self.num_state is None:
            self.num_state = len(obs)
            self.observation_space = gym.spaces.Box(
                low=np.float32(-np.inf),
                high=np.float32(np.inf),
                shape=(self.num_state,),
                dtype=np.float32
            )

        return obs.astype(np.float32), info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Muscle substeps happen internally in C++ during this call.
        Muscle tuples are collected automatically for later retrieval.

        Args:
            action: Action to execute (high-level policy output)

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (goal reached or failure)
            truncated: Whether episode was truncated (time limit)
            info: Additional information including reward components
        """
        # Ensure action is float32
        action = action.astype(np.float32)

        # Execute step (muscle substeps happen internally)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Increment parameter counter
        self.param_count += 1

        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def get_muscle_tuples(self) -> list:
        """
        Get collected muscle tuples and clear buffer.

        Returns list of lists of numpy arrays:
        - Non-cascading: [tau_des, JtA_reduced, JtA]
        - Cascading: [tau_des, JtA_reduced, JtA, prev_out, weight]

        Each component is a list of numpy arrays from steps since last call.

        Returns:
            List of muscle tuple components (each is a list of arrays)
        """
        if not self.is_two_level:
            return []

        return self.env.get_muscle_tuples()

    def update_muscle_weights(self, state_dict: dict) -> None:
        """
        Update muscle network weights in the C++ environment.

        Args:
            state_dict: State dict from MuscleNN model (torch tensors, not numpy)
        """
        if not self.is_two_level:
            return

        self.env.update_muscle_weights(state_dict)

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    @property
    def is_hierarchical(self) -> bool:
        """Whether this environment uses hierarchical control."""
        return self.is_two_level

    def getNumActuatorAction(self) -> int:
        """Get number of actuator actions (for AsyncVectorEnv compatibility)."""
        return self.env.getNumActuatorAction()

    def getNumMuscles(self) -> int:
        """Get number of muscles (for AsyncVectorEnv compatibility)."""
        return self.env.getNumMuscles()

    def getNumMuscleDof(self) -> int:
        """Get number of muscle DOFs (for AsyncVectorEnv compatibility)."""
        return self.env.getNumMuscleDof()


def make_env(env_file: str, idx: int = 0):
    """
    Create environment factory for vectorized environments.

    Args:
        env_file: Path to environment configuration
        idx: Environment index (for seeding)

    Returns:
        Callable that creates a new environment instance
    """
    def thunk():
        env = HierarchicalEnv(env_file)
        env.reset(seed=idx)
        return env

    return thunk

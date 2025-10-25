from python.pysim import RayEnvManager
import numpy as np
import gymnasium
import ray
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from python.uri_resolver import resolve_path, ensure_directory_exists


class MyEnv(gymnasium.Env):
    # Gymnasium requires metadata to be a class-level dict
    metadata = {}

    def __init__(self, env_metadata, is_xml=False):
        if is_xml:
            self.env = RayEnvManager(env_metadata, True)
        else:
            self.env = RayEnvManager(env_metadata)

        self.env.updateParamState()
        self.env.reset()
        self.obs = self.env.getState()

        self.num_state = len(self.obs)
        self.num_action = len(self.env.getAction())
        # Store the actual metadata in a different attribute
        self.env_metadata = self.env.getMetadata()

        self.observation_space = gymnasium.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_state,))
        self.action_space = gymnasium.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_action,))

        self.use_cascading = self.env.getUseCascading()

        # For Mass Actuator (2-Level)
        self.isTwoLevelActuator = self.env.isTwoLevelController()

        if self.isTwoLevelActuator:
            if not self.use_cascading:
                self.muscle_tuples = [[], [], []]
            else:
                self.muscle_tuples = [[], [], [], [], []]
        self.param_count = 0

        # For reward map logging
        self.reward_map_buffer = []

    def reset(self, seed=None, options=None):
        # Gymnasium API: reset() returns (observation, info)
        super().reset(seed=seed)

        if self.param_count > 300:
            self.env.updateParamState()
            self.param_count = 0
        self.env.reset()
        self.obs = self.env.getState()

        info = {
            'param_state': self.env.getParamState(),
            'normalized_phase': self.env.getNormalizedPhase(),
            'world_phase': self.env.getWorldPhase()
        }

        return self.obs, info

    def step(self, action):
        # Gymnasium API: step() returns (obs, reward, terminated, truncated, info)
        self.env.setAction(action)
        self.env.step()
        self.param_count += 1

        self.obs = self.env.getState()
        reward = self.env.getReward()

        # Accumulate reward map for logging
        reward_map = dict(self.env.getRewardMap())
        self.reward_map_buffer.append(reward_map)

        # Use direct Gymnasium-aligned methods for episode termination
        terminated = self.env.isTerminated()  # Episode ended due to failure (fall, out of bounds)
        truncated = self.env.isTruncated()    # Episode ended due to time/step limit

        # Build info dictionary
        info = {
            'reward_map': reward_map,
            'param_state': self.env.getParamState(),
            'normalized_phase': self.env.getNormalizedPhase(),
            'world_phase': self.env.getWorldPhase()
        }

        # Collect muscle tuples if two-level actuator
        if self.isTwoLevelActuator:
            mt = self.env.getRandomMuscleTuple()
            for i in range(len(mt)):
                self.muscle_tuples[i].append(mt[i])
            # Store muscle tuples in info for access by trainer
            info['muscle_tuples'] = [np.array(self.muscle_tuples[i], dtype=np.float32) for i in range(len(self.muscle_tuples))]

        return self.obs, reward, terminated, truncated, info

    def get_muscle_tuple(self, idx):
        assert (self.isTwoLevelActuator)
        res = np.array(self.muscle_tuples[idx], dtype=np.float32)
        if self.isTwoLevelActuator:
            self.muscle_tuples[idx] = []  # = [[],[],[]]
        return res

    def load_muscle_model_weights(self, w):
        self.env.setMuscleNetworkWeight(convert_to_torch_tensor(ray.get(w)))

    def get_reward_map_average(self):
        """Return average of accumulated reward maps and clear buffer"""
        if not self.reward_map_buffer:
            return {}

        # Collect all keys
        all_keys = set()
        for reward_map in self.reward_map_buffer:
            all_keys.update(reward_map.keys())

        # Average each key across all accumulated maps
        result = {}
        for key in all_keys:
            values = [rm[key] for rm in self.reward_map_buffer if key in rm]
            if values:
                result[key] = np.mean(values)

        # Clear buffer after averaging
        self.reward_map_buffer = []
        return result


def createEnv():
    return MyEnv()


if __name__ == "__main__":
    print("MAIN")
    e = MyEnv("data/env.xml")
    e.reset()

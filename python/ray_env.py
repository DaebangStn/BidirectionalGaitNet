from python.pysim import RayEnvManager
import numpy as np
import gym
import ray
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from python.uri_resolver import resolve_path, ensure_directory_exists


class EnvSpec:
    """Minimal spec class to satisfy Gymnasium's max_episode_steps check"""
    def __init__(self, id="MyEnv", max_episode_steps=None):
        self.id = id
        self.max_episode_steps = max_episode_steps


class MyEnv(gym.Env):
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

        self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_state,))
        self.action_space = gym.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_action,))

        # Add spec to prevent Gym warning about max_episode_steps
        # The actual horizon is controlled by RLlib config, this just silences the warning
        self.spec = EnvSpec(max_episode_steps=10000)

        self.use_cascading = self.env.getUseCascading()

        # For Mass Actuator (2-Level)
        self.isTwoLevelActuator = self.env.isTwoLevelController()

        if self.isTwoLevelActuator:
            if not self.use_cascading:
                self.muscle_tuples = [[], [], []]
            else:
                self.muscle_tuples = [[], [], [], [], []]
        self.param_count = 0

        # Buffer for averaging info maps across episode
        self.info_map_buffer = []

    def reset(self, seed=None, options=None):
        # Gym API: reset() returns (observation, info)
        super().reset(seed=seed)

        if self.param_count > 300:
            self.env.updateParamState()
            self.param_count = 0
        self.env.reset()
        self.obs = self.env.getState()

        info = {}
        # return self.obs, info
        return self.obs

    def step(self, action):
        # Gym API: step() returns (obs, reward, terminated, truncated, info)
        self.env.setAction(action)
        self.env.step()
        self.param_count += 1

        self.obs = self.env.getState()
        reward = self.env.getReward()
        terminated = self.env.isTerminated()
        truncated = self.env.isTruncated()

        # Build info dictionary with reward components and termination/truncation status from mInfoMap
        info_map = dict(self.env.getInfoMap())
        self.info_map_buffer.append(info_map)

        info = info_map.copy()

        # Collect muscle tuples if two-level actuator
        if self.isTwoLevelActuator:
            mt = self.env.getRandomMuscleTuple()
            for i in range(len(mt)):
                self.muscle_tuples[i].append(mt[i])

        return self.obs, reward, terminated or truncated, info

    def get_muscle_tuple(self, idx):
        assert (self.isTwoLevelActuator)
        res = np.array(self.muscle_tuples[idx], dtype=np.float32)
        if self.isTwoLevelActuator:
            self.muscle_tuples[idx] = []  # = [[],[],[]]
        return res

    def load_muscle_model_weights(self, w):
        self.env.setMuscleNetworkWeight(convert_to_torch_tensor(ray.get(w)))

    def get_info_map_average(self):
        """Return average of accumulated info maps and clear buffer"""
        if not self.info_map_buffer:
            return {}

        # Collect all keys
        all_keys = set()
        for info_map in self.info_map_buffer:
            all_keys.update(info_map.keys())

        # Average each key across all accumulated maps
        result = {}
        for key in all_keys:
            values = [im[key] for im in self.info_map_buffer if key in im]
            if values:
                result[key] = np.mean(values)

        # Clear buffer after averaging
        self.info_map_buffer = []
        return result


def createEnv():
    return MyEnv()


if __name__ == "__main__":
    print("MAIN")
    e = MyEnv("data/env.xml")
    e.reset()

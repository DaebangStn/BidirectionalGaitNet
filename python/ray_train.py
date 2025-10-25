import torch.optim as optim
import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict
import torch
import pickle
import xml.etree.ElementTree as ET
from python.ray_model import SimulationNN_Ray, MuscleNN, RolloutNNRay
from python.ray_env import MyEnv
from python.util import timestamp

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from python.ray_config import CONFIG


def mean_dict_list(dict_list):
    """Average values across list of dictionaries"""
    if not dict_list:
        return {}

    # Collect all keys
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())

    # Average each key
    result = {}
    for key in all_keys:
        values = [d[key] for d in dict_list if key in d]
        if values:
            result[key] = np.mean(values)

    return result


class MuscleLearner:
    def __init__(self, device, num_actuator_action, num_muscles, num_muscle_dofs,
                 learning_rate=1e-4, num_epochs=3, batch_size=128, model=None, is_cascaded=False):
        self.device = device
        self.num_actuator_action = num_actuator_action
        self.num_muscles = num_muscles
        self.num_epochs_muscle = num_epochs
        self.muscle_batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate
        # self.use_timewarp = use_timewarp
        self.is_cascaded = is_cascaded

        if model:
            self.model = model
        else:
            self.model = MuscleNN(num_muscle_dofs, self.num_actuator_action,
                                  self.num_muscles, is_cascaded=self.is_cascaded).to(self.device)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def save(self, name):
        path = Path(name)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(),
                   path.with_suffix(".opt" + path.suffix))

    def load(self, name):
        path = Path(name)
        self.model.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(
            path.with_suffix(".opt" + path.suffix)))

    def learn(self, muscle_transitions: list) -> Dict:
        converting_time = 0.0
        learning_time = 0.0

        start_time = time.perf_counter()
        l = len(muscle_transitions[0])

        idx_all = np.asarray(range(len(muscle_transitions[0])))

        tau_des_net_all = torch.tensor(np.asarray(
            muscle_transitions[0]), device="cuda")
        JtA_reduced_all = torch.tensor(np.asarray(
            muscle_transitions[1]), device="cuda")
        JtA_all = torch.tensor(np.asarray(
            muscle_transitions[2]), device="cuda")

        prev_out_all = None
        w_all = None
        if self.is_cascaded:
            prev_out_all = torch.tensor(np.asarray(
                muscle_transitions[3]), device="cuda")
            w_all = torch.tensor(np.asarray(
                muscle_transitions[4]), device="cuda")

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_avg = 0.
        loss_reg_avg = 0.
        loss_target_avg = 0.
        loss_act_reg = 0.
        for _ in range(self.num_epochs_muscle):
            np.random.shuffle(idx_all)
            loss_avg = 0.
            loss_reg_avg = 0.
            loss_target_avg = 0.
            loss_act_reg = 0.
            for i in range(l // self.muscle_batch_size):
                mini_batch_idx = torch.from_numpy(
                    idx_all[i*self.muscle_batch_size: (i+1)*self.muscle_batch_size]).cuda()
                tau_des = torch.index_select(
                    tau_des_net_all, 0, mini_batch_idx)
                JtA_reduced = torch.index_select(
                    JtA_reduced_all, 0, mini_batch_idx)
                JtA = torch.index_select(JtA_all, 0, mini_batch_idx)

                activation = None
                activation_wo_relu = None
                if self.is_cascaded:
                    w = torch.index_select(w_all, 0, mini_batch_idx)
                    prev_out = torch.index_select(
                        prev_out_all, 0, mini_batch_idx)
                    activation_wo_relu = self.model.forward_with_prev_out_wo_relu(
                        JtA_reduced, tau_des, prev_out, w).unsqueeze(2)
                else:
                    activation_wo_relu = self.model.forward_wo_relu(
                        JtA_reduced, tau_des).unsqueeze(2)

                activation = torch.relu(torch.tanh(activation_wo_relu))
                activation_wo_relu = activation_wo_relu

                tau = torch.bmm(JtA, activation).squeeze(-1)
                loss_reg_wo_relu = activation_wo_relu.pow(2).mean()
                loss_target = (((tau - tau_des) / 100.0).pow(2)).mean()
                loss_reg_act = activation.pow(2).mean()

                loss = 0.01 * loss_reg_act + loss_target + 0.01 * loss_reg_wo_relu

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.grad != None:

                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
                loss_act_reg += activation.pow(
                    2).mean().cpu().detach().numpy().tolist()
                loss_avg += loss.cpu().detach().numpy().tolist()
                loss_reg_avg += loss_reg_wo_relu.cpu().detach().numpy().tolist()
                loss_target_avg += loss_target.cpu().detach().numpy().tolist()

        loss_muscle = loss_avg / (l // self.muscle_batch_size)

        loss_muscle_reg = loss_reg_avg / (l // self.muscle_batch_size)
        loss_muscle_target = loss_target_avg / (l // self.muscle_batch_size)
        loss_act_reg = loss_act_reg / (l // self.muscle_batch_size)

        learning_time = (time.perf_counter() - start_time) * 1000
        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}

        return {
            'num_tuples': l,
            'loss_muscle': loss_muscle,
            'loss_reg': loss_muscle_reg,
            'loss_target': loss_muscle_target,
            'loss_act': loss_act_reg,
            'time': time_stat
        }


class MyTrainer(PPO):
    def get_default_policy_class(self, config):
        return PPOTorchPolicy

    def setup(self, config):

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rollout = config.pop("rollout")
        self.metadata = config.pop("metadata")
        self.isTwoLevelActuator = config.pop("isTwoLevelActuator")

        self.trainer_config = config.pop("trainer_config")
        PPO.setup(self, config=config)
        self.max_reward = 0
        self.remote_workers = self.workers.remote_workers()

        self.env_config = config.pop("env_config")

        if self.isTwoLevelActuator and not self.rollout:
            self.muscle_learner = MuscleLearner(self.device, self.env_config['num_actuator_action'], self.env_config['num_muscles'], self.env_config["num_muscle_dofs"], learning_rate=self.trainer_config[
                                                "muscle_lr"], num_epochs=self.trainer_config["muscle_num_epochs"], batch_size=self.trainer_config["muscle_sgd_minibatch_size"], is_cascaded=self.env_config["cascading"])

            model_weights = ray.put(
                self.muscle_learner.get_model_weights(device=torch.device("cpu")))
            for worker in self.remote_workers:
                worker.foreach_env.remote(
                    lambda env: env.load_muscle_model_weights(model_weights))

        if self.rollout:
            for worker in self.remote_workers:
                worker.foreach_env.remote(lambda env: env.set_is_rollout())

            def apply_seed_to_worker(worker, callable, _n):
                envs = worker.async_env.vector_env.get_unwrapped()
                for env in envs:
                    callable(env, _n)
                    _n += 1

            seed_idx = 0
            for worker in self.remote_workers:
                worker.apply.remote(lambda worker: apply_seed_to_worker(
                    worker, lambda env, _n: env.set_idx(_n), seed_idx))
                seed_idx += config['num_envs_per_worker']

    def step(self):
        # Simulation NN Learning
        result = PPO.step(self)
        result["num_tuples"] = {}
        result["loss"] = {}

        # Collect reward component averages with optimized IPC
        if not self.rollout:
            # Each environment returns its own average (not list of maps)
            buffer = [worker.foreach_env.remote(
                lambda env: env.get_reward_map_average())
                for worker in self.remote_workers]

            # Collect all worker averages
            worker_rewards = []
            for worker_maps in ray.get(buffer):
                # Each env already returned averaged dict
                for avg_map in worker_maps:
                    if avg_map:  # Skip empty dicts
                        worker_rewards.append(avg_map)

            # Average across all workers/envs and add to metrics
            if worker_rewards:
                avg_reward_components = mean_dict_list(worker_rewards)
                result['sampler_results']['custom_metrics']['reward'] = avg_reward_components

        # For Two Level Controller
        if self.isTwoLevelActuator and not self.rollout:
            start = time.perf_counter()
            mts = []
            muscle_transitions = []

            for idx in range(5 if self.env_config["cascading"] else 3):
                mts.append(ray.get([worker.foreach_env.remote(
                    lambda env: env.get_muscle_tuple(idx)) for worker in self.remote_workers]))
                muscle_transitions.append([])

            idx = 0
            for mts_i in range(len(mts)):
                for worker_i in range(len(mts[mts_i])):
                    for env_i in range(len(mts[mts_i][worker_i])):
                        for i in range(len(mts[mts_i][worker_i][env_i])):
                            muscle_transitions[idx].append(
                                mts[mts_i][worker_i][env_i][i])
                idx += 1

            loading_time = (time.perf_counter() - start) * 1000
            stats = self.muscle_learner.learn(muscle_transitions)

            distribute_time = time.perf_counter()
            model_weights = ray.put(self.muscle_learner.get_model_weights(device=torch.device("cpu")))
            for worker in self.remote_workers:
                worker.foreach_env.remote(lambda env: env.load_muscle_model_weights(model_weights))

            distribute_time = (time.perf_counter() - distribute_time) * 1000
            total_time = (time.perf_counter() - start) * 1000

            result['timers']['muscle_learning'] = stats.pop('time')
            result['num_tuples']['muscle_learning'] = stats.pop('num_tuples')
            result['timers']['muscle_learning']['distribute_time_ms'] = distribute_time
            result['timers']['muscle_learning']['loading_time_ms'] = loading_time
            result['timers']['muscle_learning']['total_ms'] = total_time
            result["loss"].update(stats)

        current_reward = result['episode_reward_mean']

        if self.max_reward < current_reward:
            self.max_reward = current_reward
            self.save_max_checkpoint(self._logdir)

        return result

    def __getstate__(self):
        state = PPO.__getstate__(self)
        state["metadata"] = self.metadata
        if self.isTwoLevelActuator and not self.rollout:
            state["muscle"] = self.muscle_learner.get_weights()
            state["muscle_optimizer"] = self.muscle_learner.get_optimizer_weights()
        if self.env_config["cascading"]:
            state["cascading"] = True
        return state

    def __setstate__(self, state):
        PPO.__setstate__(self, state)
        if self.isTwoLevelActuator and not self.rollout:
            self.muscle_learner.set_weights(state["muscle"])
            self.muscle_learner.set_optimizer_weights(
                state["muscle_optimizer"])

    def save_max_checkpoint(self, checkpoint_path) -> str:
        max_checkpoint_dir = Path(checkpoint_path) / "max_checkpoint"
        max_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return PPO.save_checkpoint(self, str(max_checkpoint_dir))

    def load_checkpoint(self, checkpoint_path):
        print(f'Loading checkpoint at path {checkpoint_path}')
        checkpoint_file = list(Path(checkpoint_path).glob("checkpoint-*"))
        if len(checkpoint_file) == 0:
            raise RuntimeError("Missing checkpoint file!")
        PPO.load_checkpoint(self, checkpoint_file[0])


def get_config_from_file(filename: str, config: str):
    exec(open(filename).read(), globals())
    config = CONFIG[config]
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="ppo_mini")
parser.add_argument("--config-file", type=str, default="python/ray_config.py")
parser.add_argument("--env", type=str, default="data/base_lonly.xml")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--epoch", type=int, default=None, help="Maximum training epochs (overrides XML config)")
parser.add_argument("--rollout", action='store_true')

if __name__ == "__main__":

    env_content = None
    is_xml = False
    checkpoint_path = None
    args = parser.parse_args()
    print('Argument : ', args)
    checkpoint_path = args.checkpoint

    # Detect file type and read metadata content
    env_path = args.env
    is_xml = env_path.endswith(".xml")
    with open(env_path) as f:
        env_content = f.read()
    print(f"Loading {'XML' if is_xml else 'YAML'} environment done......")

    # Parse training configuration from file
    training_config = {}
    if is_xml:
        # Parse training config from XML
        try:
            # Wrap XML fragment in root element for parsing
            wrapped_xml = f"<root>{env_content}</root>"
            root = ET.fromstring(wrapped_xml)
            training_elem = root.find('training')
            if training_elem is not None:
                max_epoch_elem = training_elem.find('max_epoch')
                checkpoint_freq_elem = training_elem.find('checkpoint_freq')
                if max_epoch_elem is not None:
                    training_config['max_epoch'] = int(max_epoch_elem.text)
                if checkpoint_freq_elem is not None:
                    training_config['checkpoint_freq'] = int(checkpoint_freq_elem.text)
        except Exception as e:
            print(f"Warning: Failed to parse training config from XML: {e}")
    else:
        # Parse training config from YAML
        try:
            import yaml
            config_data = yaml.safe_load(env_content)
            if 'training' in config_data:
                training = config_data['training']
                if 'max_epoch' in training:
                    training_config['max_epoch'] = training['max_epoch']
                if 'checkpoint_freq' in training:
                    training_config['checkpoint_freq'] = training['checkpoint_freq']
        except Exception as e:
            print(f"Warning: Failed to parse training config from YAML: {e}")

    # Use command line argument if provided, otherwise use XML config, otherwise default
    max_epoch = args.epoch if args.epoch is not None else training_config.get('max_epoch', 100002)
    checkpoint_freq = training_config.get('checkpoint_freq', 1000)
    print(f"Training configuration: max_epoch={max_epoch}, checkpoint_freq={checkpoint_freq}")

    ray.init(address="auto")

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    config = get_config_from_file(args.config_file, args.config)
    if args.rollout:
        config["rollout"] = args.rollout
        config["num_sgd_iter"] = 0
        config["kl_coeff"] = 0
        ModelCatalog.register_custom_model("MyModel", RolloutNNRay)
        with open('rollout/env.xml', 'w') as f:
            f.write(env_content)
    else:
        config["rollout"] = False
        ModelCatalog.register_custom_model("MyModel", SimulationNN_Ray)

    # Register environment with appropriate file type
    if is_xml:
        register_env("MyEnv", lambda config: MyEnv(env_content, is_xml=True))
    else:
        register_env("MyEnv", lambda config: MyEnv(env_content))

    print(f'Loading config {args.config} from config file {args.config_file}.')

    # Always use auto for rollout_fragment_length to avoid batch size division errors
    config["rollout_fragment_length"] = "auto"

    if args.rollout:
        config["batch_mode"] = "complete_episodes"

    # Initialize environment with appropriate file type
    with MyEnv(env_content, is_xml=is_xml) if is_xml else MyEnv(env_content) as env:
        config["metadata"] = env.metadata
        config["isTwoLevelActuator"] = env.isTwoLevelActuator
        config["model"]["custom_model_config"]["learningStd"] = env.env.getLearningStd()
        config["env_config"]["cascading"] = env.env.getUseCascading()
        config["env_config"]["num_action"] = env.env.getNumAction()
        config["env_config"]["num_actuator_action"] = env.env.getNumActuatorAction()

        config["env_config"]["num_muscles"] = env.env.getNumMuscles()
        config["env_config"]["num_muscle_dofs"] = env.env.getNumMuscleDof()
    
    tune.run(MyTrainer,
             name=Path(args.env).stem,
             config=config,
             trial_dirname_creator=(lambda trial: timestamp()),
             storage_path=str(Path.cwd() / "ray_results"),
             restore=checkpoint_path,
             stop={"training_iteration": max_epoch},
             progress_reporter=CLIReporter(max_report_frequency=500),
             checkpoint_freq=checkpoint_freq)
    ray.shutdown()

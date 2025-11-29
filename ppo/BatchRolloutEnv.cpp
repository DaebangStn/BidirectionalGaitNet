#include "BatchRolloutEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// Helper function: Convert Eigen::VectorXd to numpy array (float32)
static py::array_t<float> toNumPyArray(const Eigen::VectorXd& vec) {
    py::array_t<float> arr(vec.size());
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    for (int i = 0; i < vec.size(); ++i) {
        ptr[i] = static_cast<float>(vec[i]);
    }
    return arr;
}

// Helper function: Convert Eigen::VectorXf to numpy array (float32)
static py::array_t<float> toNumPyArray(const Eigen::VectorXf& vec) {
    py::array_t<float> arr(vec.size());
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::memcpy(ptr, vec.data(), vec.size() * sizeof(float));
    return arr;
}

// Helper function: Convert Eigen::MatrixXd to numpy array (float32, row-major)
static py::array_t<float> toNumPyArray(const Eigen::MatrixXd& mat) {
    py::array_t<float> arr({mat.rows(), mat.cols()});
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    int idx = 0;
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            ptr[idx++] = static_cast<float>(mat(i, j));
        }
    }
    return arr;
}

BatchRolloutEnv::BatchRolloutEnv(const std::string& yaml_content, int num_envs, int num_steps)
    : num_envs_(num_envs), num_steps_(num_steps)
{
    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    if (num_steps <= 0) {
        throw std::invalid_argument("num_steps must be positive");
    }

    // Create thread pool
    pool_ = std::make_unique<ThreadPool>(num_envs);

    // IMPORTANT: Initialize URIResolver eagerly in main thread BEFORE parallel environment creation
    // This eliminates the need for thread synchronization during Environment::initialize()
    PMuscle::URIResolver::getInstance().initialize();

    // Create environments in parallel
    envs_.resize(num_envs);

    for (int i = 0; i < num_envs; ++i) {
        pool_->enqueue([this, i, &yaml_content]() {
            auto env = std::make_unique<Environment>();
            env->initialize(yaml_content);
            env->reset();
            envs_[i] = std::move(env);
        });
    }

    // Wait for environment creation
    pool_->wait();

    // Query dimensions
    obs_dim_ = envs_[0]->getState().size();
    action_dim_ = envs_[0]->getAction().size();

    // Create trajectory buffer and policy network
    trajectory_ = std::make_unique<TrajectoryBuffer>(num_steps, num_envs, obs_dim_, action_dim_);
    policy_ = std::make_shared<PolicyNetImpl>(obs_dim_, action_dim_);

    // Initialize muscle tuple buffers if hierarchical control enabled
    if (envs_[0]->isTwoLevelController()) {
        muscle_tuple_buffers_.resize(num_envs_);
    }

    // Initialize discriminator observation buffers if discriminator enabled
    if (envs_[0]->getUseDiscriminator()) {
        num_muscles_ = envs_[0]->getCharacter()->getNumMuscles();
        disc_obs_buffers_.resize(num_envs_);
    } else {
        num_muscles_ = 0;
    }

    // Initialize episode tracking
    episode_returns_.resize(num_envs_, 0.0);
    episode_lengths_.resize(num_envs_, 0);
    next_done_.resize(num_envs_, 0);  // Initialize to False (like Python's torch.zeros)
}

// Reset all environments to initial state
void BatchRolloutEnv::reset() {
    for (int i = 0; i < num_envs_; ++i) {
        pool_->enqueue([this, i]() {
            envs_[i]->reset();
            episode_returns_[i] = 0.0;
            episode_lengths_[i] = 0;
            next_done_[i] = 0;  // Reset to False
        });
    }
    pool_->wait();
}

// Internal method: Execute rollout without GIL (no Python object creation)
void BatchRolloutEnv::collect_rollout_nogil() {
    trajectory_->reset();

    for (int i = 0; i < num_envs_; ++i) {
        pool_->enqueue([this, i]() {
            for (int step = 0; step < num_steps_; ++step) {
                Eigen::VectorXf obs = envs_[i]->getState().cast<float>();
                auto [action, value, logprob] = policy_->sample_action(obs);
                envs_[i]->setAction(action.cast<double>().eval());
                envs_[i]->step();

                float reward = static_cast<float>(envs_[i]->getReward());
                uint8_t terminated = envs_[i]->isTerminated() ? 1 : 0;
                uint8_t truncated = envs_[i]->isTruncated() ? 1 : 0;

                // Append to trajectory (done = cached done status from BEFORE action)
                // Matches Python: dones[step] = next_done (from previous iteration)
                trajectory_->append(step, i, obs, action, reward, value, logprob,
                                    next_done_[i], terminated, truncated);
                trajectory_->accumulate_info(envs_[i]->getInfoMap());

                episode_returns_[i] += reward;
                episode_lengths_[i] += 1;

                // Update next_done with result from THIS step (matches Python semantics)
                // Python: next_done = np.logical_or(terminations, truncations)
                // This cached value will be used in the NEXT iteration's append() call
                next_done_[i] = (terminated || truncated) ? 1 : 0;

                if (terminated || truncated) {
                    trajectory_->accumulate_episode(episode_returns_[i], episode_lengths_[i]);

                    if (truncated && !terminated) {
                        // Capture truncated final obs (state after step, before reset)
                        Eigen::VectorXf truncated_final_obs = envs_[i]->getState().cast<float>();
                        trajectory_->store_truncated_final_obs(step, i, truncated_final_obs);
                    }

                    envs_[i]->reset();
                    episode_returns_[i] = 0.0;
                    episode_lengths_[i] = 0;
                    // DO NOT reset next_done_[i] here! It should remain 1 for the next iteration's append()
                    // The next step will naturally set next_done_[i] = 0 after the reset environment returns terminated=0, truncated=0
                }

                if (!muscle_tuple_buffers_.empty()) {
                    MuscleTuple mt = envs_[i]->getRandomMuscleTuple();
                    Eigen::VectorXd dt = envs_[i]->getRandomDesiredTorque();

                    muscle_tuple_buffers_[i].tau_des.push_back(dt);
                    muscle_tuple_buffers_[i].JtA_reduced.push_back(mt.JtA_reduced);
                    muscle_tuple_buffers_[i].JtA.push_back(mt.JtA);

                    if (envs_[i]->getUseCascading()) {
                        muscle_tuple_buffers_[i].prev_out.push_back(envs_[i]->getRandomPrevOut());
                        muscle_tuple_buffers_[i].weight.push_back(envs_[i]->getRandomWeight());
                    }
                }

                // Collect disc_obs (muscle activations) for discriminator training
                if (!disc_obs_buffers_.empty()) {
                    disc_obs_buffers_[i].push_back(envs_[i]->getRandomDiscObs());
                }
            }

            // Capture final next_obs for GAE bootstrap (AFTER all steps complete)
            // Only needed once at the end, not every step
            Eigen::VectorXf final_next_obs = envs_[i]->getState().cast<float>();
            trajectory_->set_next_obs(i, final_next_obs, next_done_[i]);
        });
    }

    pool_->wait();
}

// Public method: Convert trajectory to numpy (requires GIL)
py::dict BatchRolloutEnv::collect_rollout() {
    return trajectory_->to_numpy();
}

void BatchRolloutEnv::update_policy_weights(py::dict state_dict) {
    // Load weights into policy network
    policy_->load_state_dict(state_dict);
}

void BatchRolloutEnv::update_muscle_weights(py::dict state_dict) {
    // OPTIMIZED: Pre-convert Python dict to C++ format, then parallel load without GIL
    //
    // PHASE 1: Convert Python dict to C++ format (with GIL, sequential)
    // This phase MUST be sequential because Python dict iteration requires GIL
    std::unordered_map<std::string, torch::Tensor> cpp_state_dict;

    for (auto item : state_dict) {
        std::string key = item.first.cast<std::string>();
        py::array_t<float> np_array = item.second.cast<py::array_t<float>>();

        // Convert numpy array to torch::Tensor
        auto buf = np_array.request();
        std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());

        torch::Tensor tensor = torch::from_blob(
            buf.ptr,
            shape,
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();  // Clone to own the memory

        cpp_state_dict[key] = tensor;
    }

    // PHASE 2: Parallel broadcast to all environments (WITHOUT GIL, parallel!)
    // Each environment has its own MuscleNN instance, so updates are independent
    {
        py::gil_scoped_release release;  // Release GIL for parallel execution

        for (int i = 0; i < num_envs_; ++i) {
            pool_->enqueue([this, i, &cpp_state_dict]() {
                // No GIL needed - pure libtorch operations
                (*envs_[i]->getMuscleNN())->load_state_dict(cpp_state_dict);
            });
        }

        pool_->wait();
    }
    // GIL automatically reacquired here
}

void BatchRolloutEnv::update_discriminator_weights(py::dict state_dict) {
    // Similar to update_muscle_weights: Pre-convert Python dict to C++ format
    // PHASE 1: Convert Python dict to C++ format (with GIL, sequential)
    std::unordered_map<std::string, torch::Tensor> cpp_state_dict;

    for (auto item : state_dict) {
        std::string key = item.first.cast<std::string>();
        py::array_t<float> np_array = item.second.cast<py::array_t<float>>();

        auto buf = np_array.request();
        std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());

        torch::Tensor tensor = torch::from_blob(
            buf.ptr,
            shape,
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        cpp_state_dict[key] = tensor;
    }

    // PHASE 2: Parallel broadcast to all environments (WITHOUT GIL)
    {
        py::gil_scoped_release release;

        for (int i = 0; i < num_envs_; ++i) {
            pool_->enqueue([this, i, &cpp_state_dict]() {
                if (envs_[i]->getUseDiscriminator()) {
                    (*envs_[i]->getDiscriminatorNN())->load_state_dict(cpp_state_dict);
                }
            });
        }

        pool_->wait();
    }
}

py::array_t<float> BatchRolloutEnv::get_disc_obs() {
    // Return disc_obs buffer as numpy array, then clear buffers
    if (disc_obs_buffers_.empty() || num_muscles_ == 0) {
        // Return empty array if discriminator not enabled
        return py::array_t<float>(std::vector<ssize_t>{0, 0});
    }

    // Calculate total samples
    int total_samples = 0;
    for (const auto& buf : disc_obs_buffers_) {
        total_samples += buf.size();
    }

    // Allocate numpy array: shape (total_samples, num_muscles)
    py::array_t<float> result(std::vector<ssize_t>{total_samples, num_muscles_});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);

    // Copy data: flatten across all environments
    int idx = 0;
    for (int i = 0; i < num_envs_; ++i) {
        for (const auto& obs : disc_obs_buffers_[i]) {
            std::memcpy(ptr + idx * num_muscles_, obs.data(), num_muscles_ * sizeof(float));
            idx++;
        }
        // Clear buffer after copying
        disc_obs_buffers_[i].clear();
    }

    return result;
}

py::list BatchRolloutEnv::get_muscle_tuples() {
    // Convert raw Eigen data to numpy arrays (with GIL held)
    py::list all_tuples;

    if (muscle_tuple_buffers_.empty()) {
        return all_tuples;  // Not hierarchical, return empty list
    }

    bool use_cascading = envs_[0]->getUseCascading();

    for (int i = 0; i < num_envs_; ++i) {
        // Create list of numpy arrays for this environment
        py::list env_tuple;

        // tau_des list
        py::list tau_des_list;
        for (const auto& data : muscle_tuple_buffers_[i].tau_des) {
            tau_des_list.append(toNumPyArray(data));
        }
        env_tuple.append(tau_des_list);

        // JtA_reduced list
        py::list JtA_reduced_list;
        for (const auto& data : muscle_tuple_buffers_[i].JtA_reduced) {
            JtA_reduced_list.append(toNumPyArray(data));
        }
        env_tuple.append(JtA_reduced_list);

        // JtA list
        py::list JtA_list;
        for (const auto& data : muscle_tuple_buffers_[i].JtA) {
            JtA_list.append(toNumPyArray(data));
        }
        env_tuple.append(JtA_list);

        if (use_cascading) {
            // prev_out list
            py::list prev_out_list;
            for (const auto& data : muscle_tuple_buffers_[i].prev_out) {
                prev_out_list.append(toNumPyArray(data));
            }
            env_tuple.append(prev_out_list);

            // weight list
            py::list weight_list;
            for (const auto& data : muscle_tuple_buffers_[i].weight) {
                weight_list.append(toNumPyArray(data));
            }
            env_tuple.append(weight_list);
        }

        all_tuples.append(env_tuple);

        // Clear buffers for next collection
        muscle_tuple_buffers_[i].tau_des.clear();
        muscle_tuple_buffers_[i].JtA_reduced.clear();
        muscle_tuple_buffers_[i].JtA.clear();
        muscle_tuple_buffers_[i].prev_out.clear();
        muscle_tuple_buffers_[i].weight.clear();
    }

    return all_tuples;
}

// ===== HIERARCHICAL CONTROL QUERY METHODS =====

bool BatchRolloutEnv::is_hierarchical() const {
    return envs_[0]->isTwoLevelController();
}

bool BatchRolloutEnv::use_cascading() const {
    return envs_[0]->getUseCascading();
}

int BatchRolloutEnv::getNumActuatorAction() const {
    return envs_[0]->getNumActuatorAction();
}

int BatchRolloutEnv::getNumMuscles() const {
    return envs_[0]->getCharacter()->getNumMuscles();
}

int BatchRolloutEnv::getNumMuscleDof() const {
    return envs_[0]->getCharacter()->getNumMuscleRelatedDof();
}

// ===== DISCRIMINATOR QUERY METHODS =====

bool BatchRolloutEnv::use_discriminator() const {
    return envs_[0]->getUseDiscriminator();
}

float BatchRolloutEnv::getDiscRewardScale() const {
    return static_cast<float>(envs_[0]->getDiscConfig().reward_scale);
}

// ===== PYBIND11 MODULE =====

PYBIND11_MODULE(batchrolloutenv, m) {
    // Configure libtorch threading BEFORE any torch operations
    // Use try-catch to handle case where it's already configured
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
    } catch (...) {
        // Already configured, ignore
    }

    m.doc() = "Autonomous rollout environment with C++ policy inference and zero-copy trajectory return";

    py::class_<BatchRolloutEnv>(m, "BatchRolloutEnv")
        .def(py::init<std::string, int, int>(),
             py::arg("yaml_content"),
             py::arg("num_envs"),
             py::arg("num_steps"),
             "Create autonomous rollout environment\n\n"
             "Args:\n"
             "    yaml_content (str): YAML configuration content (NOT file path!)\n"
             "    num_envs (int): Number of parallel environments\n"
             "    num_steps (int): Rollout length (steps per trajectory)\n\n"
             "Example:\n"
             "    with open('data/env/config.yaml') as f:\n"
             "        yaml_content = f.read()\n"
             "    env = BatchRolloutEnv(yaml_content, num_envs=32, num_steps=64)")

        .def("reset", &BatchRolloutEnv::reset,
             "Reset all environments to initial state\n\n"
             "Should be called before the first rollout to ensure environments\n"
             "start from a valid initial state.")

        .def("collect_rollout", [](BatchRolloutEnv& self) {
            // Phase 1: Release GIL during C++ rollout (no Python objects)
            {
                py::gil_scoped_release release;
                self.collect_rollout_nogil();
            }
            // Phase 2: Convert to numpy with GIL held (Python object creation)
            return self.collect_rollout();
        }, "Collect trajectory with autonomous C++ rollout\n\n"
           "Runs num_steps of parallel environment stepping with C++ policy inference.\n"
           "Returns complete trajectory for Python learning.\n\n"
           "Returns:\n"
           "    dict: Trajectory data with zero-copy numpy arrays\n"
           "        - obs: (steps*envs, obs_dim) float32\n"
           "        - actions: (steps*envs, action_dim) float32\n"
           "        - rewards: (steps*envs,) float32\n"
           "        - values: (steps*envs,) float32\n"
           "        - logprobs: (steps*envs,) float32\n"
           "        - dones: (steps*envs,) uint8")

        .def("update_policy_weights", &BatchRolloutEnv::update_policy_weights,
             py::arg("state_dict"),
             "Update policy network weights from PyTorch state_dict\n\n"
             "Args:\n"
             "    state_dict (dict): PyTorch state_dict with policy network weights")

        .def("update_muscle_weights", &BatchRolloutEnv::update_muscle_weights,
             py::arg("state_dict"),
             "Update muscle network weights (hierarchical control)\n\n"
             "Args:\n"
             "    state_dict (dict): PyTorch state_dict with muscle network weights")

        .def("get_muscle_tuples", &BatchRolloutEnv::get_muscle_tuples,
             "Get muscle training tuples (hierarchical control)\n\n"
             "Returns:\n"
             "    list: List of muscle tuple buffers, one per environment")

        .def("update_discriminator_weights", &BatchRolloutEnv::update_discriminator_weights,
             py::arg("state_dict"),
             "Update discriminator network weights from PyTorch state_dict\n\n"
             "Args:\n"
             "    state_dict (dict): PyTorch state_dict with discriminator network weights")

        .def("get_disc_obs", &BatchRolloutEnv::get_disc_obs,
             "Get disc_obs (muscle activations) collected during rollout\n\n"
             "Returns:\n"
             "    numpy.ndarray: shape (num_steps*num_envs, num_muscles)")

        .def("num_envs", &BatchRolloutEnv::numEnvs, "Get number of parallel environments")
        .def("num_steps", &BatchRolloutEnv::numSteps, "Get rollout length")
        .def("obs_size", &BatchRolloutEnv::obsSize, "Get observation dimension")
        .def("action_size", &BatchRolloutEnv::actionSize, "Get action dimension")

        // Hierarchical control query methods
        .def("is_hierarchical", &BatchRolloutEnv::is_hierarchical,
             "Check if environment uses hierarchical (two-level) control")
        .def("use_cascading", &BatchRolloutEnv::use_cascading,
             "Check if cascading control mode is enabled")
        .def("getNumActuatorAction", &BatchRolloutEnv::getNumActuatorAction,
             "Get number of actuator actions (joint torque dimension)")
        .def("getNumMuscles", &BatchRolloutEnv::getNumMuscles,
             "Get number of muscles in the character")
        .def("getNumMuscleDof", &BatchRolloutEnv::getNumMuscleDof,
             "Get number of muscle-related degrees of freedom")

        // Discriminator query methods
        .def("use_discriminator", &BatchRolloutEnv::use_discriminator,
             "Check if discriminator is enabled for this environment")
        .def("getDiscRewardScale", &BatchRolloutEnv::getDiscRewardScale,
             "Get discriminator reward scale factor");
}

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

BatchRolloutEnv::BatchRolloutEnv(const std::string& yaml_content, int num_envs, int num_steps, bool enable_numa)
    : num_envs_(num_envs), num_steps_(num_steps)
{
    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    if (num_steps <= 0) {
        throw std::invalid_argument("num_steps must be positive");
    }

    // Create only the thread pool we'll use
    if (enable_numa) {
        pool_numa_ = std::make_unique<NUMAThreadPool>(num_envs, true);
        numa_enabled_ = pool_numa_->numa_enabled();

        if (numa_enabled_) {
            num_nodes_ = pool_numa_->num_numa_nodes();
            envs_per_node_ = num_envs / num_nodes_;
            std::cout << "BatchRolloutEnv: NUMA enabled with " << num_nodes_ << " nodes, "
                      << envs_per_node_ << " envs per node" << std::endl;
        }
    } else {
        pool_ = std::make_unique<ThreadPool>(num_envs);
        numa_enabled_ = false;
    }

    // IMPORTANT: Initialize URIResolver eagerly in main thread BEFORE parallel environment creation
    // This eliminates the need for thread synchronization during Environment::initialize()
    PMuscle::URIResolver::getInstance().initialize();

    // Create environments in parallel
    envs_.resize(num_envs);

    for (int i = 0; i < num_envs; ++i) {
        auto task = [this, i, &yaml_content]() {
            auto env = std::make_unique<Environment>();
            env->initialize(yaml_content);
            env->reset();
            envs_[i] = std::move(env);
        };

        if (numa_enabled_) {
            pool_numa_->enqueue(task);
        } else {
            pool_->enqueue(task);
        }
    }

    // Wait for environment creation
    if (numa_enabled_) {
        pool_numa_->wait();
    } else {
        pool_->wait();
    }

    // Query dimensions
    obs_dim_ = envs_[0]->getState().size();
    action_dim_ = envs_[0]->getAction().size();

    // Create trajectory buffers and policy networks based on NUMA configuration
    if (numa_enabled_) {
        // Per-NUMA resources
        for (int node = 0; node < num_nodes_; ++node) {
            trajectory_numa_.push_back(
                std::make_unique<TrajectoryBuffer>(num_steps, envs_per_node_, obs_dim_, action_dim_)
            );
            policy_numa_.push_back(std::make_shared<PolicyNetImpl>(obs_dim_, action_dim_));
        }
        // Master trajectory for aggregation
        master_trajectory_ = std::make_unique<TrajectoryBuffer>(num_steps, num_envs, obs_dim_, action_dim_);
    } else {
        // Single trajectory and policy
        trajectory_ = std::make_unique<TrajectoryBuffer>(num_steps, num_envs, obs_dim_, action_dim_);
        policy_ = std::make_shared<PolicyNetImpl>(obs_dim_, action_dim_);
    }

    // Initialize muscle tuple buffers if hierarchical control enabled
    if (envs_[0]->isTwoLevelController()) {
        muscle_tuple_buffers_.resize(num_envs_);
    }

    // Initialize episode tracking
    episode_returns_.resize(num_envs_, 0.0);
    episode_lengths_.resize(num_envs_, 0);
}

// Internal method: Execute rollout without GIL (no Python object creation)
void BatchRolloutEnv::collect_rollout_nogil() {
    if (numa_enabled_) {
        // NUMA-aware rollout with per-NUMA buffers

        // Reset NUMA-local trajectory buffers
        for (auto& traj : trajectory_numa_) {
            traj->reset();
        }

        // Rollout loop with NUMA-local writes
        for (int i = 0; i < num_envs_; ++i) {
            pool_numa_->enqueue([this, i]() {
                // Determine NUMA node for this environment
                int my_node = pool_numa_->get_numa_node(i);
                int local_env_idx = i % envs_per_node_;

                for (int step = 0; step < num_steps_; ++step) {
                    // 1. Get observation
                    Eigen::VectorXf obs = envs_[i]->getState().cast<float>();

                    // 2. Policy inference from NUMA-local policy
                    auto [action, value, logprob] = policy_numa_[my_node]->sample_action(obs);

                    // 3. Step environment
                    envs_[i]->setAction(action.cast<double>().eval());
                    envs_[i]->step();

                    // 4. Get reward and flags
                    float reward = static_cast<float>(envs_[i]->getReward());
                    uint8_t terminated = envs_[i]->isTerminated() ? 1 : 0;
                    uint8_t truncated = envs_[i]->isTruncated() ? 1 : 0;

                    // 5. Store in NUMA-local trajectory buffer
                    trajectory_numa_[my_node]->append(step, local_env_idx, obs, action, reward, value, logprob, terminated, truncated);

                    // 6. Accumulate info metrics
                    trajectory_numa_[my_node]->accumulate_info(envs_[i]->getInfoMap());

                    // 7. Track episode progress
                    episode_returns_[i] += reward;
                    episode_lengths_[i] += 1;

                    // 8. On episode end
                    if (terminated || truncated) {
                        trajectory_numa_[my_node]->accumulate_episode(episode_returns_[i], episode_lengths_[i]);

                        if (truncated && !terminated) {
                            Eigen::VectorXf final_obs = envs_[i]->getState().cast<float>();
                            trajectory_numa_[my_node]->store_truncated_final_obs(step, local_env_idx, final_obs);
                        }

                        envs_[i]->reset();
                        episode_returns_[i] = 0.0;
                        episode_lengths_[i] = 0;
                    }

                    // 9. Collect muscle tuples
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
                }

                // After rollout completes, capture next observation for GAE bootstrap
                // Environment has already stepped to the next state after last action
                Eigen::VectorXf next_obs = envs_[i]->getState().cast<float>();
                uint8_t next_done = (envs_[i]->isTerminated() || envs_[i]->isTruncated()) ? 1 : 0;
                trajectory_numa_[my_node]->set_next_obs(local_env_idx, next_obs, next_done);
            });
        }

        pool_numa_->wait();

        // Aggregate NUMA-local trajectories → master
        aggregate_trajectories();

    } else {
        // Standard rollout (non-NUMA)
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

                    trajectory_->append(step, i, obs, action, reward, value, logprob, terminated, truncated);
                    trajectory_->accumulate_info(envs_[i]->getInfoMap());

                    episode_returns_[i] += reward;
                    episode_lengths_[i] += 1;

                    if (terminated || truncated) {
                        trajectory_->accumulate_episode(episode_returns_[i], episode_lengths_[i]);

                        if (truncated && !terminated) {
                            Eigen::VectorXf final_obs = envs_[i]->getState().cast<float>();
                            trajectory_->store_truncated_final_obs(step, i, final_obs);
                        }

                        envs_[i]->reset();
                        episode_returns_[i] = 0.0;
                        episode_lengths_[i] = 0;
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
                }

                // After rollout completes, capture next observation for GAE bootstrap
                // Environment has already stepped to the next state after last action
                Eigen::VectorXf next_obs = envs_[i]->getState().cast<float>();
                uint8_t next_done = (envs_[i]->isTerminated() || envs_[i]->isTruncated()) ? 1 : 0;
                trajectory_->set_next_obs(i, next_obs, next_done);
            });
        }

        pool_->wait();
    }
}

// Aggregate NUMA-local trajectories into master trajectory
void BatchRolloutEnv::aggregate_trajectories() {
    if (!numa_enabled_) {
        std::cout << "BatchRolloutEnv: NUMA not enabled, skipping aggregation" << std::endl;
        return;
    }

    // CRITICAL: This is called WITHOUT GIL from collect_rollout_nogil()
    // Must NOT create Python objects (no to_numpy(), py::dict, py::array, etc.)
    // Use direct C++ memory copy instead

    master_trajectory_->reset();

    // Direct C++ aggregation without Python objects
    int env_offset = 0;
    for (int node = 0; node < num_nodes_; ++node) {
        auto& traj = trajectory_numa_[node];

        // Copy trajectory data directly from NUMA-local buffer to master
        for (int step = 0; step < num_steps_; ++step) {
            for (int local_env = 0; local_env < envs_per_node_; ++local_env) {
                int global_env = env_offset + local_env;
                int src_idx = step * envs_per_node_ + local_env;

                // Get data from NUMA-local trajectory
                Eigen::VectorXf obs = traj->get_obs_row(src_idx);
                Eigen::VectorXf action = traj->get_action_row(src_idx);
                float reward = traj->get_reward(src_idx);
                float value = traj->get_value(src_idx);
                float logprob = traj->get_logprob(src_idx);
                uint8_t termination = traj->get_termination(src_idx);
                uint8_t truncation = traj->get_truncation(src_idx);

                // Write to master trajectory
                master_trajectory_->append(step, global_env, obs, action, reward, value, logprob, termination, truncation);
            }
        }

        env_offset += envs_per_node_;
    }

    // Also copy next_obs for GAE bootstrap
    env_offset = 0;
    for (int node = 0; node < num_nodes_; ++node) {
        auto& traj = trajectory_numa_[node];
        for (int local_env = 0; local_env < envs_per_node_; ++local_env) {
            int global_env = env_offset + local_env;
            Eigen::VectorXf next_obs = traj->get_next_obs_row(local_env);
            uint8_t next_done = traj->get_next_done(local_env);
            master_trajectory_->set_next_obs(global_env, next_obs, next_done);
        }
        env_offset += envs_per_node_;
    }

    // Aggregate episode statistics and info metrics (GIL-free, pure C++)
    env_offset = 0;
    for (int node = 0; node < num_nodes_; ++node) {
        auto& traj = trajectory_numa_[node];

        // Merge info metrics
        master_trajectory_->merge_info(traj->get_info_sums(), traj->get_info_counts());

        // Merge episode statistics
        master_trajectory_->merge_episodes(
            traj->get_episode_count(),
            traj->get_episode_return_sum(),
            traj->get_episode_length_sum()
        );

        // Merge truncated final observations (adjust env indices to global)
        master_trajectory_->merge_truncated_obs(traj->get_truncated_final_obs(), env_offset);

        env_offset += envs_per_node_;
    }
}

// Public method: Convert trajectory to numpy (requires GIL)
py::dict BatchRolloutEnv::collect_rollout() {
    if (numa_enabled_) {
        return master_trajectory_->to_numpy();
    } else {
        return trajectory_->to_numpy();
    }
}

void BatchRolloutEnv::update_policy_weights(py::dict state_dict) {
    if (numa_enabled_) {
        // CRITICAL: Cannot pass Python objects to worker threads (not thread-safe without GIL)
        // Solution: Each policy network loads independently from the SAME Python dict sequentially
        // This is safe because we're in the main thread with GIL held
        for (int node = 0; node < num_nodes_; ++node) {
            policy_numa_[node]->load_state_dict(state_dict);
        }
    } else {
        // Load weights into single policy network
        policy_->load_state_dict(state_dict);
    }
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
            auto task = [this, i, &cpp_state_dict]() {
                // No GIL needed - pure libtorch operations
                (*envs_[i]->getMuscleNN())->load_state_dict(cpp_state_dict);
            };

            if (numa_enabled_) {
                pool_numa_->enqueue(task);
            } else {
                pool_->enqueue(task);
            }
        }

        if (numa_enabled_) {
            pool_numa_->wait();
        } else {
            pool_->wait();
        }
    }
    // GIL automatically reacquired here
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
        .def(py::init<std::string, int, int, bool>(),
             py::arg("yaml_content"),
             py::arg("num_envs"),
             py::arg("num_steps"),
             py::arg("enable_numa") = false,
             "Create autonomous rollout environment\n\n"
             "Args:\n"
             "    yaml_content (str): YAML configuration content (NOT file path!)\n"
             "    num_envs (int): Number of parallel environments\n"
             "    num_steps (int): Rollout length (steps per trajectory)\n"
             "    enable_numa (bool): Enable NUMA-aware thread affinity (default: False)\n\n"
             "Example:\n"
             "    with open('data/env/config.yaml') as f:\n"
             "        yaml_content = f.read()\n"
             "    env = BatchRolloutEnv(yaml_content, num_envs=32, num_steps=64, enable_numa=True)")

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

        .def("num_envs", &BatchRolloutEnv::numEnvs, "Get number of parallel environments")
        .def("num_steps", &BatchRolloutEnv::numSteps, "Get rollout length")
        .def("obs_size", &BatchRolloutEnv::obsSize, "Get observation dimension")
        .def("action_size", &BatchRolloutEnv::actionSize, "Get action dimension")

        // NUMA query methods
        .def("numa_enabled", &BatchRolloutEnv::numaEnabled,
             "Check if NUMA-aware threading is enabled")
        .def("num_numa_nodes", &BatchRolloutEnv::numNumaNodes,
             "Get number of NUMA nodes (1 if NUMA disabled)")

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
             "Get number of muscle-related degrees of freedom");
}

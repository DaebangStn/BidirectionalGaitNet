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
    : num_envs_(num_envs), num_steps_(num_steps), pool_(num_envs)  // Match thread count to num_envs
{
    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }
    if (num_steps <= 0) {
        throw std::invalid_argument("num_steps must be positive");
    }

    // Create environments in parallel (each in its dedicated thread)
    // This significantly reduces initialization time for large num_envs
    envs_.resize(num_envs);

    for (int i = 0; i < num_envs; ++i) {
        pool_.enqueue([this, i, &yaml_content]() {
            // Each environment is created and initialized in its own thread
            auto env = std::make_unique<Environment>();
            env->initialize(yaml_content);
            env->reset();  // CRITICAL: Must call reset() after initialize()

            // Thread-safe assignment (each thread writes to different index)
            envs_[i] = std::move(env);
        });
    }

    // Wait for all environments to finish initialization
    pool_.wait();

    // Query dimensions from first environment
    obs_dim_ = envs_[0]->getState().size();
    action_dim_ = envs_[0]->getAction().size();

    // Create properly-sized trajectory buffer
    trajectory_ = std::make_unique<TrajectoryBuffer>(num_steps, num_envs, obs_dim_, action_dim_);

    // Create policy network
    policy_ = std::make_shared<PolicyNetImpl>(obs_dim_, action_dim_);

    // Initialize muscle tuple buffers if hierarchical control enabled
    if (envs_[0]->isTwoLevelController()) {
        muscle_tuple_buffers_.resize(num_envs_);
        // Vectors will be populated during rollout
    }

    // Initialize episode tracking
    episode_returns_.resize(num_envs_, 0.0);
    episode_lengths_.resize(num_envs_, 0);
}

// Internal method: Execute rollout without GIL (no Python object creation)
void BatchRolloutEnv::collect_rollout_nogil() {
    // Reset trajectory buffer
    trajectory_->reset();

    // Rollout loop: Enqueue all work asynchronously without waiting
    // Each environment will independently run through all its steps
    for (int i = 0; i < num_envs_; ++i) {
        pool_.enqueue([this, i]() {
            // Each environment runs its own rollout loop independently
            for (int step = 0; step < num_steps_; ++step) {
                // 1. Get observation (double → float)
                Eigen::VectorXf obs = envs_[i]->getState().cast<float>();

                // 2. Policy inference (sequential, per-env, no GIL needed)
                auto [action, value, logprob] = policy_->sample_action(obs);

                // 3. Step environment (float → double for action)
                envs_[i]->setAction(action.cast<double>().eval());
                envs_[i]->step();

                // 4. Get reward and separate termination/truncation flags
                float reward = static_cast<float>(envs_[i]->getReward());
                uint8_t terminated = envs_[i]->isTerminated() ? 1 : 0;
                uint8_t truncated = envs_[i]->isTruncated() ? 1 : 0;

                // 5. Store in trajectory buffer with separate flags
                trajectory_->append(step, i, obs, action, reward, value, logprob, terminated, truncated);

                // 6. Accumulate info metrics (every step)
                trajectory_->accumulate_info(envs_[i]->getInfoMap());

                // 7. Track episode progress
                episode_returns_[i] += reward;
                episode_lengths_[i] += 1;

                // 8. On episode end: accumulate stats and store truncated final obs
                if (terminated || truncated) {
                    // Accumulate episode statistics
                    trajectory_->accumulate_episode(episode_returns_[i], episode_lengths_[i]);

                    // Store final observation for PURE truncated episodes (for terminal bootstrapping)
                    // Only bootstrap if truncated but NOT terminated (matches ppo_hierarchical.py:343)
                    if (truncated && !terminated) {
                        Eigen::VectorXf final_obs = envs_[i]->getState().cast<float>();
                        trajectory_->store_truncated_final_obs(step, i, final_obs);
                    }

                    // AUTO-RESET: Reset environment to start new episode
                    // This matches GymEnvManager behavior (lines 137-149 in GymEnvManager.cpp)
                    envs_[i]->reset();

                    // Reset episode tracking for this environment
                    episode_returns_[i] = 0.0;
                    episode_lengths_[i] = 0;
                }

                // 9. Collect muscle tuples (if hierarchical)
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
        });
    }

    // Wait only once at the end for all environments to finish their complete rollouts
    pool_.wait();
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
            pool_.enqueue([this, i, &cpp_state_dict]() {
                // No GIL needed - pure libtorch operations
                (*envs_[i]->getMuscleNN())->load_state_dict(cpp_state_dict);
            });
        }
        pool_.wait();
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

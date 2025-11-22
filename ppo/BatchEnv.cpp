#include "BatchEnv.h"
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

BatchEnv::BatchEnv(const std::string& yaml_content, int num_envs)
    : num_envs_(num_envs), pool_(num_envs) {  // Match thread count to num_envs

    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }

    // Create environments in parallel (each in its dedicated thread)
    // This significantly reduces initialization time for large num_envs
    envs_.resize(num_envs);

    for (int i = 0; i < num_envs; ++i) {
        pool_.enqueue([this, i, &yaml_content]() {
            // Each environment is created and initialized in its own thread
            auto env = std::make_unique<Environment>();
            env->initialize(yaml_content);  // Pass YAML content, not file path
            env->reset();  // CRITICAL: Must call reset() after initialize() to set up DART collision detector

            // Thread-safe assignment (each thread writes to different index)
            envs_[i] = std::move(env);
        });
    }

    // Wait for all environments to finish initialization
    pool_.wait();

    // Query dimensions from first environment
    obs_dim_ = envs_[0]->getState().size();
    action_dim_ = envs_[0]->getAction().size();

    // Allocate Eigen buffers (row-major for numpy C-contiguous)
    obs_buffer_ = RowMajorMatrixXf::Zero(num_envs_, obs_dim_);
    rew_buffer_ = Eigen::VectorXf::Zero(num_envs_);
    done_buffer_ = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>::Zero(num_envs_);

    // Initialize muscle tuple buffers if hierarchical control enabled
    if (envs_[0]->isTwoLevelController()) {
        muscle_tuple_buffers_.resize(num_envs_);
        // Vectors will be populated during step()
    }
}

void BatchEnv::reset() {
    // Parallel execution: Now that MuscleNN is in C++, no GIL constraints!
    for (int i = 0; i < num_envs_; ++i) {
        pool_.enqueue([this, i]() {
            envs_[i]->reset();

            // Copy observation: VectorXd (double) → MatrixXf row (float32)
            Eigen::VectorXd obs_d = envs_[i]->getState();
            obs_buffer_.row(i) = obs_d.cast<float>();

            // Initialize reward and done
            rew_buffer_[i] = 0.0f;
            done_buffer_[i] = 0;
        });
    }

    // Wait for all reset tasks to complete
    pool_.wait();
}

void BatchEnv::step(const RowMajorMatrixXf& actions) {
    // Validate input shape
    if (actions.rows() != num_envs_ || actions.cols() != action_dim_) {
        throw std::runtime_error(
            "Invalid action shape: expected (" + std::to_string(num_envs_) +
            ", " + std::to_string(action_dim_) + "), got (" +
            std::to_string(actions.rows()) + ", " + std::to_string(actions.cols()) + ")"
        );
    }

    // Parallel execution: Now that MuscleNN is in C++, no GIL constraints!
    for (int i = 0; i < num_envs_; ++i) {
        pool_.enqueue([this, i, &actions]() {
            // Auto-reset if environment is done
            if (done_buffer_[i]) {
                envs_[i]->reset();
            }

            // Set action: MatrixXf row (float32) → VectorXd (double)
            // Important: Use eval() to force evaluation of the expression template
            Eigen::VectorXd action_d = actions.row(i).cast<double>().eval();
            envs_[i]->setAction(action_d);

            // Step environment
            envs_[i]->step();

            // Write observation: VectorXd (double) → MatrixXf row (float32)
            Eigen::VectorXd obs_d = envs_[i]->getState();
            obs_buffer_.row(i) = obs_d.cast<float>();

            // Write reward (float32 cast)
            rew_buffer_[i] = static_cast<float>(envs_[i]->getReward());

            // Write done flag (terminated OR truncated)
            done_buffer_[i] = (envs_[i]->isTerminated() || envs_[i]->isTruncated()) ? 1 : 0;

            // Collect muscle tuples if hierarchical control enabled
            // Note: Muscle tuple collection must happen with GIL held (after pool_.wait())
            // because py::list operations require GIL
        });
    }

    // Wait for all step tasks to complete
    pool_.wait();

    // Collect muscle tuples after step (store raw Eigen data, no GIL needed)
    if (!muscle_tuple_buffers_.empty()) {
        for (int i = 0; i < num_envs_; ++i) {
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
}

// ===== HIERARCHICAL CONTROL METHODS =====

bool BatchEnv::is_hierarchical() const {
    // Query first environment
    return envs_[0]->isTwoLevelController();
}

bool BatchEnv::use_cascading() const {
    // Query first environment
    return envs_[0]->getUseCascading();
}

int BatchEnv::getNumActuatorAction() const {
    // Query first environment
    return envs_[0]->getNumActuatorAction();
}

int BatchEnv::getNumMuscles() const {
    // Query first environment
    return envs_[0]->getCharacter()->getNumMuscles();
}

int BatchEnv::getNumMuscleDof() const {
    // Query first environment
    return envs_[0]->getCharacter()->getNumMuscleRelatedDof();
}

py::list BatchEnv::get_muscle_tuples() {
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

void BatchEnv::update_muscle_weights(py::dict state_dict) {
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

// ===== PYBIND11 MODULE =====

PYBIND11_MODULE(batchenv, m) {
    // Configure libtorch threading BEFORE any torch operations
    // Use try-catch to handle case where it's already configured
    try {
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
    } catch (...) {
        // Already configured, ignore
    }

    m.doc() = "High-performance batched environment with zero-copy numpy bindings and ThreadPool parallelization";

    py::class_<BatchEnv>(m, "BatchEnv")
        .def(py::init<std::string, int>(),
             py::arg("yaml_content"),
             py::arg("num_envs"),
             "Create a batched environment\n\n"
             "Args:\n"
             "    yaml_content (str): YAML configuration content (NOT file path!)\n"
             "    num_envs (int): Number of parallel environments\n\n"
             "Example:\n"
             "    with open('data/env/config.yaml') as f:\n"
             "        yaml_content = f.read()\n"
             "    env = BatchEnv(yaml_content, num_envs=64)")

        .def("reset", [](BatchEnv& self) {
            // Release GIL during C++ work
            py::gil_scoped_release release;
            self.reset();
            py::gil_scoped_acquire acquire;

            // Zero-copy: return numpy array viewing Eigen buffer
            // Row-major Eigen matrix → C-contiguous numpy array
            return py::array_t<float>(
                {self.numEnvs(), self.obsDim()},                    // shape: (num_envs, obs_dim)
                {self.obsDim() * sizeof(float), sizeof(float)},     // strides: row-major
                self.obsData(),                                     // data pointer
                py::cast(self, py::return_value_policy::reference)  // keep BatchEnv alive
            );
        }, "Reset all environments and return observations\n\n"
           "Returns:\n"
           "    np.ndarray: Observations array of shape (num_envs, obs_dim), dtype=float32")

        .def("step", [](BatchEnv& self, py::array_t<float> actions_np) {
            auto buf = actions_np.request();

            // Validate input dimensions
            if (buf.ndim != 2) {
                throw std::runtime_error("actions must be 2D array (num_envs, action_dim)");
            }
            if (buf.shape[0] != self.numEnvs() || buf.shape[1] != self.actionDim()) {
                throw std::runtime_error(
                    "actions shape mismatch: expected (" + std::to_string(self.numEnvs()) +
                    ", " + std::to_string(self.actionDim()) + "), got (" +
                    std::to_string(buf.shape[0]) + ", " + std::to_string(buf.shape[1]) + ")"
                );
            }

            // Copy numpy array to Eigen matrix
            BatchEnv::RowMajorMatrixXf actions_copy(buf.shape[0], buf.shape[1]);
            std::memcpy(actions_copy.data(), buf.ptr, buf.shape[0] * buf.shape[1] * sizeof(float));

            // Release GIL during parallel C++ execution (MuscleNN is now in C++!)
            {
                py::gil_scoped_release release;
                self.step(actions_copy);
            }

            // Return zero-copy numpy views: (obs, rew, done)
            auto obs = py::array_t<float>(
                {self.numEnvs(), self.obsDim()},
                {self.obsDim() * sizeof(float), sizeof(float)},
                self.obsData(),
                py::cast(self, py::return_value_policy::reference)
            );

            auto rew = py::array_t<float>(
                {self.numEnvs()},
                {sizeof(float)},
                self.rewData(),
                py::cast(self, py::return_value_policy::reference)
            );

            auto done = py::array_t<uint8_t>(
                {self.numEnvs()},
                {sizeof(uint8_t)},
                self.doneData(),
                py::cast(self, py::return_value_policy::reference)
            );

            return py::make_tuple(obs, rew, done);
        }, py::arg("actions"),
           "Step all environments with given actions\n\n"
           "Args:\n"
           "    actions (np.ndarray): Action array of shape (num_envs, action_dim), dtype=float32\n\n"
           "Returns:\n"
           "    tuple: (observations, rewards, dones)\n"
           "        - observations: np.ndarray of shape (num_envs, obs_dim), dtype=float32\n"
           "        - rewards: np.ndarray of shape (num_envs,), dtype=float32\n"
           "        - dones: np.ndarray of shape (num_envs,), dtype=uint8")

        .def("num_envs", &BatchEnv::numEnvs, "Get number of parallel environments")
        .def("obs_dim", &BatchEnv::obsDim, "Get observation dimension")
        .def("action_dim", &BatchEnv::actionDim, "Get action dimension")

        // Hierarchical control methods
        .def("is_hierarchical", &BatchEnv::is_hierarchical,
             "Check if environment uses hierarchical (two-level) control\n\n"
             "Returns:\n"
             "    bool: True if hierarchical control is enabled")
        .def("use_cascading", &BatchEnv::use_cascading,
             "Check if cascading control mode is enabled\n\n"
             "Returns:\n"
             "    bool: True if cascading mode is enabled")
        .def("getNumActuatorAction", &BatchEnv::getNumActuatorAction,
             "Get number of actuator actions (joint torque dimension)\n\n"
             "Returns:\n"
             "    int: Number of actuator actions")
        .def("getNumMuscles", &BatchEnv::getNumMuscles,
             "Get number of muscles in the character\n\n"
             "Returns:\n"
             "    int: Number of muscles")
        .def("getNumMuscleDof", &BatchEnv::getNumMuscleDof,
             "Get number of muscle-related degrees of freedom\n\n"
             "Returns:\n"
             "    int: Number of muscle DOFs")
        .def("get_muscle_tuples", &BatchEnv::get_muscle_tuples,
             "Collect muscle training tuples from all environments\n\n"
             "Returns:\n"
             "    list: List of tuple lists, one per environment")
        .def("update_muscle_weights", &BatchEnv::update_muscle_weights,
             py::arg("state_dict"),
             "Update muscle network weights in all environments\n\n"
             "Args:\n"
             "    state_dict (dict): PyTorch state_dict with muscle network weights");
}

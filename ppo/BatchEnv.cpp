#include "BatchEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

BatchEnv::BatchEnv(const std::string& yaml_content, int num_envs)
    : num_envs_(num_envs), pool_(std::thread::hardware_concurrency()) {

    if (num_envs <= 0) {
        throw std::invalid_argument("num_envs must be positive");
    }

    // Create environments
    envs_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i) {
        auto env = std::make_unique<Environment>();
        env->initialize(yaml_content);  // Pass YAML content, not file path
        env->reset();  // CRITICAL: Must call reset() after initialize() to set up DART collision detector
        envs_.push_back(std::move(env));
    }

    // Query dimensions from first environment
    obs_dim_ = envs_[0]->getState().size();
    action_dim_ = envs_[0]->getAction().size();

    // Allocate Eigen buffers (row-major for numpy C-contiguous)
    obs_buffer_ = RowMajorMatrixXf::Zero(num_envs_, obs_dim_);
    rew_buffer_ = Eigen::VectorXf::Zero(num_envs_);
    done_buffer_ = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>::Zero(num_envs_);
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
        });
    }

    // Wait for all step tasks to complete
    pool_.wait();
}

// ===== PYBIND11 MODULE =====

PYBIND11_MODULE(batchenv, m) {
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
        .def("action_dim", &BatchEnv::actionDim, "Get action dimension");
}

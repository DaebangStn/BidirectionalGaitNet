#include "TrajectoryBuffer.h"
#include <stdexcept>
#include <cstring>

TrajectoryBuffer::TrajectoryBuffer(int num_steps, int num_envs, int obs_dim, int action_dim)
    : num_steps_(num_steps), num_envs_(num_envs), obs_dim_(obs_dim), action_dim_(action_dim),
      episode_count_(0), episode_return_sum_(0.0), episode_length_sum_(0) {

    if (num_steps <= 0 || num_envs <= 0 || obs_dim <= 0 || action_dim <= 0) {
        throw std::invalid_argument("All dimensions must be positive");
    }

    int total_size = num_steps * num_envs;

    // Preallocate all matrices (row-major for numpy C-contiguous)
    obs_ = RowMajorMatrixXf::Zero(total_size, obs_dim);
    actions_ = RowMajorMatrixXf::Zero(total_size, action_dim);
    rewards_ = Eigen::VectorXf::Zero(total_size);
    values_ = Eigen::VectorXf::Zero(total_size);
    logprobs_ = Eigen::VectorXf::Zero(total_size);
    terminations_ = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>::Zero(total_size);
    truncations_ = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>::Zero(total_size);

    // Preallocate next observation storage for GAE bootstrap
    next_obs_ = RowMajorMatrixXf::Zero(num_envs, obs_dim);
    next_done_ = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>::Zero(num_envs);
}

void TrajectoryBuffer::append(int step, int env_idx,
                               const Eigen::VectorXf& obs,
                               const Eigen::VectorXf& action,
                               float reward,
                               float value,
                               float logprob,
                               uint8_t terminated,
                               uint8_t truncated) {
    // Validate indices
    if (step < 0 || step >= num_steps_) {
        throw std::out_of_range("Step index out of range: " + std::to_string(step));
    }
    if (env_idx < 0 || env_idx >= num_envs_) {
        throw std::out_of_range("Environment index out of range: " + std::to_string(env_idx));
    }

    // Validate dimensions
    if (obs.size() != obs_dim_) {
        throw std::invalid_argument(
            "Observation dimension mismatch: expected " + std::to_string(obs_dim_) +
            ", got " + std::to_string(obs.size())
        );
    }
    if (action.size() != action_dim_) {
        throw std::invalid_argument(
            "Action dimension mismatch: expected " + std::to_string(action_dim_) +
            ", got " + std::to_string(action.size())
        );
    }

    // Linear index: step * num_envs + env_idx
    int idx = step * num_envs_ + env_idx;

    // Write to matrices
    obs_.row(idx) = obs;
    actions_.row(idx) = action;
    rewards_[idx] = reward;
    values_[idx] = value;
    logprobs_[idx] = logprob;
    terminations_[idx] = terminated;
    truncations_[idx] = truncated;
}

void TrajectoryBuffer::accumulate_info(const std::map<std::string, double>& info_map) {
    std::lock_guard<std::mutex> lock(info_mutex_);
    for (const auto& [key, value] : info_map) {
        info_sums_[key] += value;
        info_counts_[key] += 1;
    }
}

void TrajectoryBuffer::store_truncated_final_obs(int step, int env_idx, const Eigen::VectorXf& obs) {
    std::lock_guard<std::mutex> lock(truncated_mutex_);
    truncated_final_obs_.push_back({step, env_idx, obs});
}

void TrajectoryBuffer::accumulate_episode(double episode_return, int episode_length) {
    std::lock_guard<std::mutex> lock(episode_mutex_);
    episode_count_ += 1;
    episode_return_sum_ += episode_return;
    episode_length_sum_ += episode_length;
}

void TrajectoryBuffer::set_next_obs(int env_idx, const Eigen::VectorXf& obs, uint8_t done) {
    // Validate index
    if (env_idx < 0 || env_idx >= num_envs_) {
        throw std::out_of_range("Environment index out of range: " + std::to_string(env_idx));
    }

    // Validate dimension
    if (obs.size() != obs_dim_) {
        throw std::invalid_argument(
            "Next observation dimension mismatch: expected " + std::to_string(obs_dim_) +
            ", got " + std::to_string(obs.size())
        );
    }

    // Store next observation and done flag
    next_obs_.row(env_idx) = obs;
    next_done_[env_idx] = done;
}

// Accessor methods for C++ aggregation (GIL-free)
Eigen::VectorXf TrajectoryBuffer::get_obs_row(int idx) const {
    return obs_.row(idx);
}

Eigen::VectorXf TrajectoryBuffer::get_action_row(int idx) const {
    return actions_.row(idx);
}

Eigen::VectorXf TrajectoryBuffer::get_next_obs_row(int env_idx) const {
    return next_obs_.row(env_idx);
}

void TrajectoryBuffer::reset() {
    // Zero out all data for next rollout
    obs_.setZero();
    actions_.setZero();
    rewards_.setZero();
    values_.setZero();
    logprobs_.setZero();
    terminations_.setZero();
    truncations_.setZero();
    next_obs_.setZero();
    next_done_.setZero();

    // Clear accumulation data
    info_sums_.clear();
    info_counts_.clear();
    episode_count_ = 0;
    episode_return_sum_ = 0.0;
    episode_length_sum_ = 0;
    truncated_final_obs_.clear();
}

py::dict TrajectoryBuffer::to_numpy() {
    py::dict result;

    int total_size = num_steps_ * num_envs_;

    // Zero-copy conversion: numpy arrays view Eigen memory directly
    // TrajectoryBuffer is owned by BatchRolloutEnv, so we don't need explicit lifetime management

    // Observations: (steps*envs, obs_dim)
    result["obs"] = py::array_t<float>(
        {total_size, obs_dim_},                    // shape
        {obs_dim_ * sizeof(float), sizeof(float)}, // row-major strides
        obs_.data()                                // data pointer
        // No parent - TrajectoryBuffer is a member of BatchRolloutEnv (stays alive)
    );

    // Actions: (steps*envs, action_dim)
    result["actions"] = py::array_t<float>(
        {total_size, action_dim_},
        {action_dim_ * sizeof(float), sizeof(float)},
        actions_.data()
    );

    // Rewards: (steps*envs,)
    result["rewards"] = py::array_t<float>(
        {total_size},
        {sizeof(float)},
        rewards_.data()
    );

    // Values: (steps*envs,)
    result["values"] = py::array_t<float>(
        {total_size},
        {sizeof(float)},
        values_.data()
    );

    // Log probabilities: (steps*envs,)
    result["logprobs"] = py::array_t<float>(
        {total_size},
        {sizeof(float)},
        logprobs_.data()
    );

    // Terminations: (steps*envs,) - episode ended naturally
    result["terminations"] = py::array_t<uint8_t>(
        {total_size},
        {sizeof(uint8_t)},
        terminations_.data()
    );

    // Truncations: (steps*envs,) - time limit reached
    result["truncations"] = py::array_t<uint8_t>(
        {total_size},
        {sizeof(uint8_t)},
        truncations_.data()
    );

    // Next observations for GAE bootstrap: (num_envs, obs_dim)
    result["next_obs"] = py::array_t<float>(
        {num_envs_, obs_dim_},
        {obs_dim_ * sizeof(float), sizeof(float)},  // row-major strides
        next_obs_.data()
    );

    // Next done flags for GAE bootstrap: (num_envs,)
    result["next_done"] = py::array_t<uint8_t>(
        {num_envs_},
        {sizeof(uint8_t)},
        next_done_.data()
    );

    // Averaged info metrics (scalars)
    py::dict info_averages;
    for (const auto& [key, sum] : info_sums_) {
        double avg = sum / info_counts_[key];
        info_averages[key.c_str()] = avg;
    }
    result["info"] = info_averages;

    // Episode statistics (scalars)
    if (episode_count_ > 0) {
        result["avg_episode_return"] = episode_return_sum_ / episode_count_;
        result["avg_episode_length"] = static_cast<double>(episode_length_sum_) / episode_count_;
    } else {
        result["avg_episode_return"] = 0.0;
        result["avg_episode_length"] = 0.0;
    }
    result["episode_count"] = episode_count_;

    // Sparse truncated final observations (list of tuples)
    py::list truncated_obs_list;
    for (const auto& trunc_obs : truncated_final_obs_) {
        // Convert Eigen vector to numpy array
        py::array_t<float> obs_array(trunc_obs.obs.size());
        auto buf = obs_array.request();
        float* ptr = static_cast<float*>(buf.ptr);
        std::memcpy(ptr, trunc_obs.obs.data(), trunc_obs.obs.size() * sizeof(float));

        // Create tuple (step, env_idx, obs_array)
        py::tuple obs_tuple = py::make_tuple(trunc_obs.step, trunc_obs.env_idx, obs_array);
        truncated_obs_list.append(obs_tuple);
    }
    result["truncated_final_obs"] = truncated_obs_list;

    return result;
}

#pragma once

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <map>
#include <vector>
#include <string>

namespace py = pybind11;

/**
 * TrajectoryBuffer: Efficient 2D matrix storage for rollout trajectories
 *
 * Uses row-major Eigen matrices for zero-copy numpy conversion.
 * Memory layout: (num_steps * num_envs, dimension)
 *
 * Linear indexing: idx = step * num_envs + env_idx
 *
 * Advantages:
 * - Zero-copy numpy conversion (direct memory view)
 * - Contiguous memory (better cache locality)
 * - Preallocated (no dynamic resizing)
 * - Direct Eigen operations
 *
 * Usage:
 *   TrajectoryBuffer buffer(64, 32, 506, 51);  // 64 steps, 32 envs
 *   buffer.append(step, env_idx, obs, action, reward, value, logprob, done);
 *   py::dict traj = buffer.to_numpy();  // Zero-copy conversion
 */

class TrajectoryBuffer {
public:
    // Row-major matrix for numpy C-contiguous compatibility
    using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * Construct trajectory buffer with preallocated matrices.
     *
     * @param num_steps Rollout length (steps per trajectory)
     * @param num_envs Number of parallel environments
     * @param obs_dim Observation dimension
     * @param action_dim Action dimension
     */
    TrajectoryBuffer(int num_steps, int num_envs, int obs_dim, int action_dim);

    /**
     * Append data for one timestep in one environment.
     *
     * @param step Step index [0, num_steps)
     * @param env_idx Environment index [0, num_envs)
     * @param obs Observation vector
     * @param action Action vector
     * @param reward Reward scalar
     * @param value Value estimate
     * @param logprob Log probability of action
     * @param terminated Episode ended naturally (failure/success)
     * @param truncated Episode cut off by time limit
     */
    void append(int step, int env_idx,
                const Eigen::VectorXf& obs,
                const Eigen::VectorXf& action,
                float reward,
                float value,
                float logprob,
                uint8_t terminated,
                uint8_t truncated);

    /**
     * Reset buffer for next rollout.
     * Clears all data (sets to zero).
     */
    void reset();

    /**
     * Accumulate info metrics (lightweight, called every step).
     *
     * @param info_map Map of metric name → value from Environment::getInfoMap()
     */
    void accumulate_info(const std::map<std::string, double>& info_map);

    /**
     * Store final observation for truncated episode (sparse).
     *
     * @param step Step index where truncation occurred
     * @param env_idx Environment index
     * @param obs Final observation vector
     */
    void store_truncated_final_obs(int step, int env_idx, const Eigen::VectorXf& obs);

    /**
     * Accumulate episode statistics (called on episode completion).
     *
     * @param episode_return Total return for completed episode
     * @param episode_length Length of completed episode
     */
    void accumulate_episode(double episode_return, int episode_length);

    /**
     * Convert to numpy arrays (zero-copy).
     *
     * Returns dict with keys:
     * - 'obs': (steps*envs, obs_dim) float32
     * - 'actions': (steps*envs, action_dim) float32
     * - 'rewards': (steps*envs,) float32
     * - 'values': (steps*envs,) float32
     * - 'logprobs': (steps*envs,) float32
     * - 'terminations': (steps*envs,) uint8 (episode ended naturally)
     * - 'truncations': (steps*envs,) uint8 (time limit reached)
     * - 'info': dict of averaged scalar metrics
     * - 'avg_episode_return': scalar
     * - 'avg_episode_length': scalar
     * - 'episode_count': scalar
     * - 'truncated_final_obs': list of (step, env_idx, obs_array) tuples
     *
     * @return Python dict with numpy array views and accumulated metrics
     */
    py::dict to_numpy();

    // Metadata accessors
    int numSteps() const { return num_steps_; }
    int numEnvs() const { return num_envs_; }
    int obsDim() const { return obs_dim_; }
    int actionDim() const { return action_dim_; }
    int totalSize() const { return num_steps_ * num_envs_; }

    // Direct memory access (for testing/validation)
    float* obsData() { return obs_.data(); }
    float* actionsData() { return actions_.data(); }
    float* rewardsData() { return rewards_.data(); }
    float* valuesData() { return values_.data(); }
    float* logprobsData() { return logprobs_.data(); }
    uint8_t* terminationsData() { return terminations_.data(); }
    uint8_t* truncationsData() { return truncations_.data(); }

private:
    // Trajectory data (row-major for numpy compatibility)
    RowMajorMatrixXf obs_;       // (steps*envs, obs_dim)
    RowMajorMatrixXf actions_;   // (steps*envs, action_dim)
    Eigen::VectorXf rewards_;    // (steps*envs)
    Eigen::VectorXf values_;     // (steps*envs)
    Eigen::VectorXf logprobs_;   // (steps*envs)
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> terminations_;  // (steps*envs) episode ended naturally
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> truncations_;   // (steps*envs) time limit reached

    // Info metric accumulation (running sums for averaging)
    std::map<std::string, double> info_sums_;     // Sum of each metric
    std::map<std::string, int> info_counts_;      // Count for each metric

    // Episode statistics accumulation
    int episode_count_;           // Total episodes completed
    double episode_return_sum_;   // Sum of episode returns
    int episode_length_sum_;      // Sum of episode lengths

    // Sparse truncated final observations (~5% of steps)
    struct TruncatedObs {
        int step;
        int env_idx;
        Eigen::VectorXf obs;
    };
    std::vector<TruncatedObs> truncated_final_obs_;

    // Metadata
    int num_steps_;
    int num_envs_;
    int obs_dim_;
    int action_dim_;
};

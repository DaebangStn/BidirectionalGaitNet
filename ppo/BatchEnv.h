#pragma once
#include <memory>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "Environment.h"
#include "ThreadPool.h"

/**
 * High-performance batched environment with ThreadPool parallelization.
 *
 * Features:
 * - Zero-copy numpy integration via pybind11
 * - Parallel reset() and step() using ThreadPool
 * - Automatic reset on done
 * - float32 (Eigen::MatrixXf) for PyTorch compatibility
 *
 * Usage:
 *   BatchEnv env("data/env/config.yaml", 64);
 *   env.reset();
 *   env.step(actions);  // actions: (num_envs, action_dim) float32
 */
class BatchEnv {
public:
    // Row-major matrix for numpy C-contiguous compatibility
    using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * Construct BatchEnv with N environment instances.
     *
     * @param yaml_content YAML configuration content (not file path!)
     * @param num_envs Number of parallel environments
     */
    BatchEnv(const std::string& yaml_content, int num_envs);

    /**
     * Reset all environments in parallel.
     * Fills obs_buffer_ with initial observations.
     */
    void reset();

    /**
     * Step all environments in parallel with given actions.
     * Automatically resets environments that are done.
     *
     * @param actions Action matrix (num_envs, action_dim) in row-major order
     */
    void step(const RowMajorMatrixXf& actions);

    // Zero-copy accessors for numpy bindings
    float* obsData() { return obs_buffer_.data(); }
    float* rewData() { return rew_buffer_.data(); }
    uint8_t* doneData() { return done_buffer_.data(); }

    const float* obsData() const { return obs_buffer_.data(); }
    const float* rewData() const { return rew_buffer_.data(); }
    const uint8_t* doneData() const { return done_buffer_.data(); }

    // Metadata accessors
    int numEnvs() const { return num_envs_; }
    int obsDim() const { return obs_dim_; }
    int actionDim() const { return action_dim_; }

private:
    // Environment instances
    std::vector<std::unique_ptr<Environment>> envs_;

    // Eigen buffers (contiguous, aligned memory)
    RowMajorMatrixXf obs_buffer_;  // (num_envs, obs_dim)
    Eigen::VectorXf rew_buffer_;   // (num_envs)
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> done_buffer_;  // (num_envs)

    // Thread pool for parallel execution
    ThreadPool pool_;

    // Environment metadata
    int num_envs_;
    int obs_dim_;
    int action_dim_;
};

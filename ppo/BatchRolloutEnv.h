#pragma once

#include <memory>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include "Environment.h"
#include "ThreadPool.h"
#include "PolicyNet.h"
#include "TrajectoryBuffer.h"
#include "DiscriminatorNN.h"

namespace py = pybind11;

/**
 * BatchRolloutEnv: Autonomous C++ rollout worker with libtorch policy inference
 *
 * Architecture:
 * - Triggered mode: Python calls collect_rollout(), C++ runs autonomous rollout
 * - Sequential policy inference (one forward pass per environment)
 * - Parallel environment stepping with ThreadPool
 * - Zero-copy trajectory return via numpy arrays
 *
 * Workflow:
 *   1. Python: env.collect_rollout() → blocks
 *   2. C++: for each step:
 *           - Get obs from all envs (parallel)
 *           - Policy inference (sequential per env)
 *           - Step envs (parallel)
 *           - Collect trajectory data
 *   3. C++: Return trajectory dict → Python unblocks
 *   4. Python: GAE + PPO learning
 *   5. Python: env.update_policy_weights()
 *
 * Usage:
 *   auto env = BatchRolloutEnv(yaml_content, num_envs=32, num_steps=64);
 *   env.update_policy_weights(agent_state_dict);  // Initialize weights
 *
 *   py::dict traj = env.collect_rollout();  // Autonomous rollout
 *   // ... Python PPO learning ...
 *   env.update_policy_weights(updated_state_dict);
 */

class BatchRolloutEnv {
public:
    /**
     * Construct BatchRolloutEnv with N environment instances.
     *
     * @param filepath Path to YAML configuration file
     * @param num_envs Number of parallel environments
     * @param num_steps Rollout length (steps per trajectory)
     */
    BatchRolloutEnv(const std::string& filepath, int num_envs, int num_steps);

    /**
     * Reset all environments to initial state.
     * Should be called before the first rollout.
     */
    void reset();

    /**
     * Internal: Execute rollout without GIL (no Python object creation).
     * Must be called without GIL held.
     */
    void collect_rollout_nogil();

    /**
     * Convert trajectory to numpy (requires GIL).
     * Must be called with GIL held.
     *
     * @return dict with zero-copy numpy arrays:
     *   - 'obs': (steps*envs, obs_dim) float32
     *   - 'actions': (steps*envs, action_dim) float32
     *   - 'rewards': (steps*envs,) float32
     *   - 'values': (steps*envs,) float32
     *   - 'logprobs': (steps*envs,) float32
     *   - 'dones': (steps*envs,) uint8
     */
    py::dict collect_rollout();

    /**
     * Update policy network weights from PyTorch state_dict.
     *
     * @param state_dict Python dict with network weights
     */
    void update_policy_weights(py::dict state_dict);

    /**
     * Update muscle network weights (hierarchical control).
     *
     * @param state_dict Python dict with muscle network weights
     */
    void update_muscle_weights(py::dict state_dict);

    /**
     * Update discriminator network weights from Python training.
     *
     * @param state_dict Python dict with discriminator network weights
     */
    void update_discriminator_weights(py::dict state_dict);

    /**
     * Get muscle training tuples (hierarchical control).
     *
     * @return list of muscle tuple buffers, one per environment
     */
    py::list get_muscle_tuples();

    /**
     * Get disc_obs (muscle activations) collected during rollout.
     * Used for training the discriminator in Python.
     *
     * @return numpy array of shape (num_envs * num_steps, num_muscles)
     */
    py::array_t<float> get_disc_obs();

    // Metadata accessors
    int numEnvs() const { return num_envs_; }
    int numSteps() const { return num_steps_; }
    int obsSize() const { return obs_dim_; }
    int actionSize() const { return action_dim_; }
    int getHorizon() const { return envs_[0]->getHorizon(); }

    // Hierarchical control query methods
    bool is_hierarchical() const;
    bool use_cascading() const;
    int getNumActuatorAction() const;
    int getNumMuscles() const;
    int getNumMuscleDof() const;

    // Discriminator query methods
    bool use_discriminator() const;
    float getDiscRewardScale() const;
    int getDiscObsDim() const;

    // Curriculum learning: mask/demask joints from imitation reward
    void mask_imit_joint(const std::string& jointName);
    void demask_imit_joint(const std::string& jointName);

    // Virtual root force curriculum
    void set_virtual_force_kp(double kp_start, double discount_rate);

private:
    // Environment instances
    std::vector<std::unique_ptr<Environment>> envs_;

    // Thread pool
    std::unique_ptr<ThreadPool> pool_;

    // Policy network
    PolicyNet policy_;

    // Trajectory buffer
    std::unique_ptr<TrajectoryBuffer> trajectory_;

    // Environment metadata
    int num_envs_;
    int num_steps_;
    int obs_dim_;
    int action_dim_;

    // Muscle tuple buffers (for hierarchical control)
    // Structure: Same as BatchEnv
    struct MuscleTupleData {
        std::vector<Eigen::VectorXd> tau_des;
        std::vector<Eigen::VectorXd> JtA_reduced;
        std::vector<Eigen::MatrixXd> JtA;
        std::vector<Eigen::VectorXd> prev_out;  // cascading only
        std::vector<Eigen::VectorXf> weight;    // cascading only
    };
    std::vector<MuscleTupleData> muscle_tuple_buffers_;

    // Episode tracking (per environment)
    std::vector<double> episode_returns_;  // Current episode return for each env
    std::vector<uint8_t> next_done_;       // Cached done status (matches Python's next_done semantics)
    std::vector<int> episode_lengths_;     // Current episode length for each env

    // Discriminator observation buffer
    // Stores muscle activations (disc_obs) for each step, flattened across all envs
    // Shape: (num_steps * num_envs, num_muscles) when converted to numpy
    std::vector<std::vector<Eigen::VectorXf>> disc_obs_buffers_;  // Per-env buffers
    int num_muscles_;  // Number of muscles (discriminator input dimension)
};

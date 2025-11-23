#pragma once

#include <memory>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include "Environment.h"
#include "ThreadPool.h"
#include "NUMAThreadPool.h"
#include "PolicyNet.h"
#include "TrajectoryBuffer.h"

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
     * @param yaml_content YAML configuration content (not file path!)
     * @param num_envs Number of parallel environments
     * @param num_steps Rollout length (steps per trajectory)
     * @param enable_numa Enable NUMA-aware thread affinity (default: false)
     */
    BatchRolloutEnv(const std::string& yaml_content, int num_envs, int num_steps, bool enable_numa = false);

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
     * Get muscle training tuples (hierarchical control).
     *
     * @return list of muscle tuple buffers, one per environment
     */
    py::list get_muscle_tuples();

    // Metadata accessors
    int numEnvs() const { return num_envs_; }
    int numSteps() const { return num_steps_; }
    int obsSize() const { return obs_dim_; }
    int actionSize() const { return action_dim_; }

    // NUMA accessors
    bool numaEnabled() const { return numa_enabled_; }
    int numNumaNodes() const { return numa_enabled_ ? pool_numa_.num_numa_nodes() : 1; }

    // Hierarchical control query methods
    bool is_hierarchical() const;
    bool use_cascading() const;
    int getNumActuatorAction() const;
    int getNumMuscles() const;
    int getNumMuscleDof() const;

private:
    void aggregate_trajectories();  // Aggregate NUMA-local trajectories → master

    // Environment instances
    std::vector<std::unique_ptr<Environment>> envs_;

    // NUMA configuration
    bool numa_enabled_{false};
    int num_nodes_{1};
    int envs_per_node_{0};

    // Thread pools (only one is active based on numa_enabled_)
    ThreadPool pool_;              // Used when NUMA disabled
    NUMAThreadPool pool_numa_;     // Used when NUMA enabled

    // Policy networks
    PolicyNet policy_;                                      // Single policy (NUMA disabled)
    std::vector<PolicyNet> policy_numa_;                    // Per-NUMA policies (NUMA enabled)

    // Trajectory buffers
    std::unique_ptr<TrajectoryBuffer> trajectory_;          // Single trajectory (NUMA disabled)
    std::vector<std::unique_ptr<TrajectoryBuffer>> trajectory_numa_;  // Per-NUMA trajectories (NUMA enabled)
    std::unique_ptr<TrajectoryBuffer> master_trajectory_;   // Master for aggregation (NUMA enabled)

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
    std::vector<int> episode_lengths_;     // Current episode length for each env
};

#ifndef ROLLOUT_SAMPLE_ENV_H
#define ROLLOUT_SAMPLE_ENV_H

#include "Environment.h"
#include "RolloutRecord.h"
#include "PolicyNet.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <map>
#include <memory>
#include <string>

namespace py = pybind11;

/**
 * RolloutSampleEnv: Single-core rollout environment for checkpoint evaluation
 *
 * Key differences from RolloutEnvironment (ray/):
 * - Loads libtorch PolicyNet weights directly (no Python inference)
 * - Single-threaded, sequential rollout (no ThreadPool)
 * - Complete rollout loop in C++ (collect_rollout returns full trajectory)
 *
 * Usage:
 *   RolloutSampleEnv env(metadata_xml);
 *   env.load_config(config_yaml);
 *   env.load_policy_weights(state_dict);
 *   py::dict result = env.collect_rollout(param_dict);
 */
class RolloutSampleEnv {
public:
    explicit RolloutSampleEnv(const std::string& metadata_xml);
    ~RolloutSampleEnv();

    // Configuration
    void LoadRecordConfig(const std::string& yaml_path);
    void SetTargetCycles(int cycles) { mTargetCycles = cycles; }

    // Weight loading (from PolicyNet pattern)
    void LoadPolicyWeights(py::dict state_dict);
    void LoadMuscleWeights(py::object weights);

    // Single-shot rollout
    // Returns dict with:
    //   'data': np.ndarray (steps x fields) - scalar data
    //   'matrix_data': dict of np.ndarray - per-step vector data (e.g., motions)
    //   'fields': list of field names
    //   'param_state': np.ndarray - parameter values used
    //   'cycle_attributes': dict - per-cycle metadata
    //   'success': bool - whether target cycles reached
    //   'metrics': dict - step count, cycle count, etc.
    py::dict CollectRollout(py::object param_dict = py::none());

    // Queries
    int GetStateDim() const;
    int GetActionDim() const;
    int GetSkeletonDOF() const;
    double GetMass() const;
    std::vector<std::string> GetParameterNames();
    std::vector<std::string> GetMuscleNames();
    std::vector<std::string> GetRecordFields() const;
    bool IsHierarchical() const { return mUseMuscle; }

    // Skeleton-aware pose interpolation
    Eigen::VectorXd InterpolatePose(const Eigen::VectorXd& pose1,
                                    const Eigen::VectorXd& pose2,
                                    double t,
                                    bool extrapolate_root = false);

    // Direct access (for debugging)
    int GetTargetCycles() const { return mTargetCycles; }
    RecordConfig GetRecordConfig() const { return mRecordConfig; }

private:
    // Internal helpers
    void SetParameters(const std::map<std::string, double>& params);
    void RecordStep(class PyRolloutRecord* record);

    // Environment
    Environment mEnv;
    RecordConfig mRecordConfig;
    int mTargetCycles = 5;

    // PolicyNet (libtorch)
    std::unique_ptr<PolicyNetImpl> mPolicy;
    bool mPolicyLoaded = false;

    // Muscle network (hierarchical control)
    bool mUseMuscle = false;
};

#endif // ROLLOUT_SAMPLE_ENV_H

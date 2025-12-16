#ifndef ROLLOUT_ENVIRONMENT_H
#define ROLLOUT_ENVIRONMENT_H

#include "Environment.h"
#include "RolloutRecord.h"
#include "PyRolloutRecord.h"
#include <pybind11/pybind11.h>
#include <map>
#include <memory>
#include <string>

namespace py = pybind11;

class RolloutEnvironment {
public:
    explicit RolloutEnvironment(const std::string& metadata_path);
    ~RolloutEnvironment();
    
    // Configuration
    void LoadRecordConfig(const std::string& yaml_path);
    
    // Environment delegation
    void Reset(double phase = -1.0);
    Eigen::VectorXd GetState();
    void SetAction(const Eigen::VectorXd& action);
    void Step(RolloutRecord* record = nullptr);
    void SetMuscleNetworkWeight(py::object weights);

    // Rollout status
    int GetCycleCount();
    int getGaitCycleCount();
    bool isTerminated() { return mEnv.isTerminated(); }
    
    // Record configuration
    std::vector<std::string> GetRecordFields() const;
    RecordConfig GetRecordConfig() const { return mRecordConfig; }
    std::string GetMetabolicType() const { return mRecordConfig.metabolic.type; }
    int GetSkeletonDOF() const;
    double GetMass() const;
    
    // Parameter control
    void SetParameters(const std::map<std::string, double>& params);
    std::vector<std::string> GetParameterNames();
    Eigen::VectorXd GetParamState(bool isMirror = false);
    Eigen::VectorXd GetParamDefault();

    // Environment getters (delegate to mEnv)
    int GetSimulationHz();
    int GetControlHz();
    double GetWorldTime();

    // Motion interpolation
    Eigen::VectorXd InterpolatePose(const Eigen::VectorXd& pose1,
                                    const Eigen::VectorXd& pose2,
                                    double t,
                                    bool extrapolate_root = false);

private:
    void RecordStep(RolloutRecord* record);

    Environment mEnv;
    RecordConfig mRecordConfig;
    int mTargetCycles = 5;
};

#endif // ROLLOUT_ENVIRONMENT_H


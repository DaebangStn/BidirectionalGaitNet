#ifndef ROLLOUT_ENVIRONMENT_H
#define ROLLOUT_ENVIRONMENT_H

#include "Environment.h"
#include "RolloutRecord.h"
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
    int IsEndOfEpisode();
    
    // Record configuration
    std::vector<std::string> GetRecordFields() const;
    RecordConfig GetRecordConfig() const { return mRecordConfig; }
    int GetSkeletonDOF() const;
    
    // Parameter control
    void SetParameters(const std::map<std::string, double>& params);
    std::vector<std::string> GetParameterNames();
    Eigen::VectorXd GetParamState(bool isMirror = false);
    Eigen::VectorXd GetParamDefault();

    // Environment getters (delegate to mEnv)
    int GetSimulationHz();
    int GetControlHz();
    double GetWorldTime();
    double GetNormalizedPhase();
    int GetWorldPhaseCount();
    
private:
    void RecordStep(RolloutRecord* record);

    Environment mEnv;
    RecordConfig mRecordConfig;
    int mTargetCycles = 5;

    // Metabolic energy tracking
    double mCumulativeMetabolicEnergy = 0.0;
    int mLastCycleCount = 0;
};

#endif // ROLLOUT_ENVIRONMENT_H


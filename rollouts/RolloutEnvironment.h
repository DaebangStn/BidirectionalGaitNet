#ifndef ROLLOUT_ENVIRONMENT_H
#define ROLLOUT_ENVIRONMENT_H

#include "Environment.h"
#include "RolloutRecord.h"
#include <memory>
#include <string>

class RolloutEnvironment {
public:
    explicit RolloutEnvironment(const std::string& metadata_path);
    ~RolloutEnvironment();
    
    // Configuration
    void LoadRecordConfig(const std::string& yaml_path);
    
    // Environment delegation
    void Reset();
    Eigen::VectorXd GetState();
    void SetAction(const Eigen::VectorXd& action);
    void Step(RolloutRecord* record = nullptr);
    
    // Rollout status
    int GetCycleCount();
    int IsEndOfEpisode();
    
    // Record configuration
    std::vector<std::string> GetRecordFields() const;
    RecordConfig GetRecordConfig() const { return mRecordConfig; }
    
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
    int mSimulationStepCount = 0;
};

#endif // ROLLOUT_ENVIRONMENT_H


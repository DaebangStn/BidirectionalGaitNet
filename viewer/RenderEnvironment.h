#ifndef RENDER_ENVIRONMENT_H
#define RENDER_ENVIRONMENT_H

#include "Environment.h"
#include "CBufferData.h"
#include <map>
#include <string>

class RenderEnvironment {
public:
    RenderEnvironment(std::string metadata, CBufferData<double>* graph_data);
    ~RenderEnvironment();

    // Step with automatic graph data recording
    void step();

    // Direct access to underlying environment (for legacy/compatibility)
    Environment* GetEnvironment() { return mEnv; }

    // ===== Environment Delegation Methods =====
    // Character access
    Character* getCharacter() { return mEnv->getCharacter(); }

    // State and action
    Eigen::VectorXd getState() { return mEnv->getState(); }
    Eigen::VectorXd getAction() { return mEnv->getAction(); }
    void setAction(const Eigen::VectorXd& action) { mEnv->setAction(action); }
    double getActionScale() { return mEnv->getActionScale(); }
    int getNumActuatorAction() { return mEnv->getNumActuatorAction(); }

    // Reset and episode control
    void reset(double phase = -1.0) { mEnv->reset(phase); }
    bool isTerminated() { return mEnv->isTerminated(); }
    bool isTruncated() { return mEnv->isTruncated(); }
    bool isGaitCycleComplete() { return mEnv->isGaitCycleComplete(); }

    // Phase and timing
    double getNormalizedPhase() { return mEnv->getNormalizedPhase(); }
    double getLocalPhase(bool mod_one = false, int character_idx = 0) {
        return mEnv->getLocalPhase(mod_one, character_idx);
    }
    double getGlobalTime() { return mEnv->getGlobalTime(); }

    // World access
    dart::simulation::WorldPtr getWorld() { return mEnv->getWorld(); }

    // Control parameters
    int getControlHz() { return mEnv->getControlHz(); }
    double getCadence() { return mEnv->getCadence(); }

    // Parameter state management
    int getNumParamState() { return mEnv->getNumParamState(); }
    int getNumKnownParam() { return mEnv->getNumKnownParam(); }
    Eigen::VectorXd getParamState(bool isMirror = false) { return mEnv->getParamState(isMirror); }
    void setParamState(const Eigen::VectorXd& param, bool onlyMuscle = false, bool doOptimization = false) {
        mEnv->setParamState(param, onlyMuscle, doOptimization);
    }
    void setParamDefault() { mEnv->setParamDefault(); }
    void updateParamState() { mEnv->updateParamState(); }
    Eigen::VectorXd getParamStateFromNormalized(const Eigen::VectorXd& normalized_param) {
        return mEnv->getParamStateFromNormalized(normalized_param);
    }
    Eigen::VectorXd getNormalizedParamStateFromParam(const Eigen::VectorXd& param) {
        return mEnv->getNormalizedParamStateFromParam(param);
    }
    const std::vector<std::string>& getParamName() { return mEnv->getParamName(); }
    Eigen::VectorXd getParamMin() { return mEnv->getParamMin(); }
    Eigen::VectorXd getParamMax() { return mEnv->getParamMax(); }
    Eigen::VectorXd getParamDefault() { return mEnv->getParamDefault(); }
    const std::vector<param_group>& getGroupParam() { return mEnv->getGroupParam(); }
    void setGroupParam(const Eigen::VectorXd& param) { mEnv->setGroupParam(param); }

    // Motion reference
    Motion* getMotion() { return mEnv->getMotion(); }
    void setMotion(Motion* motion) { mEnv->setMotion(motion); }
    Eigen::VectorXd getTargetPositions() { return mEnv->getTargetPositions(); }

    // Velocity and COM
    double getTargetCOMVelocity() { return mEnv->getTargetCOMVelocity(); }
    Eigen::Vector3d getAvgVelocity() { return mEnv->getAvgVelocity(); }

    // Footstep information
    Eigen::Vector3d getCurrentFootStep() { return mEnv->getCurrentFootStep(); }
    Eigen::Vector3d getCurrentTargetFootStep() { return mEnv->getCurrentTargetFootStep(); }
    Eigen::Vector3d getNextTargetFootStep() { return mEnv->getNextTargetFootStep(); }
    bool getIsLeftLegStance() { return mEnv->getIsLeftLegStance(); }

    // Muscle network
    bool getUseMuscle() { return mEnv->getUseMuscle(); }
    void setMuscleNetwork(py::object muscle_nn) { mEnv->setMuscleNetwork(muscle_nn); }

    // Weights and learning parameters
    std::vector<double> getWeights() { return mEnv->getWeights(); }
    std::vector<bool> getUseWeights() { return mEnv->getUseWeights(); }
    void setUseWeights(const std::vector<bool>& use_weights) { mEnv->setUseWeights(use_weights); }
    std::vector<double> getDmins() { return mEnv->getDmins(); }
    std::vector<double> getBetas() { return mEnv->getBetas(); }
    double getLimitY() { return mEnv->getLimitY(); }
    double getMetabolicWeight() { return mEnv->getMetabolicWeight(); }
    void setMetabolicWeight(double weight) { mEnv->setMetabolicWeight(weight); }
    double getKneePainWeight() { return mEnv->getKneePainWeight(); }
    void setKneePainWeight(double weight) { mEnv->setKneePainWeight(weight); }
    double getScaleKneePain() { return mEnv->getScaleKneePain(); }
    void setScaleKneePain(double scale) { mEnv->setScaleKneePain(scale); }
    bool getUseMultiplicativeKneePain() { return mEnv->getUseMultiplicativeKneePain(); }
    void setUseMultiplicativeKneePain(bool use) { mEnv->setUseMultiplicativeKneePain(use); }

    // Logging and debugging
    const std::vector<Eigen::VectorXd>& getDesiredTorqueLogs() { return mEnv->getDesiredTorqueLogs(); }
    const std::map<std::string, double>& getInfoMap() { return mEnv->getInfoMap(); }
    RewardType getRewardType() { return mEnv->getRewardType(); }

    // Metadata
    std::string getMetadata() { return mEnv->getMetadata(); }

private:
    void RecordGraphData();
    void RecordInfoData();

    Environment* mEnv;
    CBufferData<double>* mGraphData;
};

#endif // RENDER_ENVIRONMENT_H


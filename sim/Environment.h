#ifndef __MS_ENVIRONMENT_H__
#define __MS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "BVH_Parser.h"
#include "Character.h"
#include "dart/collision/bullet/bullet.hpp"
#include "export.h"
#include <map>
#include <string>

// Forward declaration
template <typename T> class CBufferData;

// MotionData struct (for storing motion vector with parameters and name)
struct MotionData
{
    std::string name;
    Eigen::VectorXd motion;
    Eigen::VectorXd param;
};

struct param_group
{
    std::vector<std::string> param_names;
    std::vector<int> param_idxs;
    double v;
    std::string name;
    bool is_uniform;
};

struct DLL_PUBLIC Network
{
    std::string name; // Actually Path
    py::object joint;
    py::object muscle;

    // Only for cascading learning
    Eigen::VectorXd minV;
    Eigen::VectorXd maxV;
};

enum RewardType
{
    deepmimic,
    gaitnet,
    scadiver
};

// Bitflags for reward components
enum RewardFlags
{
    REWARD_NONE = 0,
    REWARD_METABOLIC = 1 << 0,  // 0x01 (only for multiplicative flag)
    REWARD_KNEE_PAIN = 1 << 1,  // 0x02
};

// Centralized reward configuration
struct RewardConfig
{
    int active = REWARD_NONE;
    int multiplicative = REWARD_NONE;

    // Metabolic reward parameters
    double metabolic_weight = 0.05;
    double metabolic_scale = 1.0;
    bool separate_torque_energy = false;

    // Knee pain reward parameters
    double knee_pain_weight = 1.0;
    double knee_pain_scale = 1.0;
    bool use_knee_pain_termination = false;

    // Knee pain max (per gait cycle) reward parameters
    bool use_knee_pain_max = false;
    double knee_pain_max_weight = 1.0;

    // Locomotion reward weights (always active for gaitnet)
    double head_linear_acc_weight = 4.0;
    double head_rot_weight = 4.0;
    double step_weight = 2.0;
    double avg_vel_weight = 6.0;
};

enum EOEType
{
    abstime,
    tuple
};

class DLL_PUBLIC Environment
{
public:
    Environment();
    ~Environment();

    double getRefStride() { return mRefStride; }
    double getRefCadence() { return mMotion->getMaxTime(); }
    void initialize(std::string yaml_content);  // Default: YAML content
    void initialize_xml(std::string xml_content);  // Backward compatibility: XML content

    // Simulation environment configuration
    void addCharacter(std::string path, double kp, double kv, double damping);
    void addObject(std::string path = nullptr);

    Character *getCharacter() { return mCharacter; }
    Motion *getMotion() { return mMotion; }
    void setMotion(Motion* motion) { mMotion = motion; }

    void setAction(Eigen::VectorXd _action);

    void step();
    void reset(double phase = -1.0);  // phase: 0.0-1.0 for specific phase, -1.0 for randomized
    double getReward() { return mReward; }
    const std::map<std::string, double>& getInfoMap() const { return mInfoMap; }

    bool isTerminated() const {
        auto it = mInfoMap.find("terminated");
        return it != mInfoMap.end() && it->second > 0.5;
    }
    bool isTruncated() const {
        auto it = mInfoMap.find("truncated");
        return it != mInfoMap.end() && it->second > 0.5;
    }
    // void setRefMotion(BVH *_bvh, Character *_character);

    void updateTargetPosAndVel(bool isInit = false);

    Eigen::VectorXd getTargetPositions() { return mTargetPositions; }
    Eigen::VectorXd getTargetVelocities() { return mTargetVelocities; }

    double getLocalPhase(bool mod_one = false, int character_idx = 0) { return (mCharacter->getLocalTime() / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))) - (mod_one ? (int)(mCharacter->getLocalTime() / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))) : 0.0); }

    Eigen::VectorXd getState();
    std::pair<Eigen::VectorXd, Eigen::VectorXd> getProjState(const Eigen::VectorXd minV, const Eigen::VectorXd maxV);

    Eigen::VectorXd getJointState(bool isMirror);

    double calcReward();
    Eigen::VectorXd getAction() { return mAction; }

    int getSimulationHz() { return mSimulationHz; }
    int getControlHz() { return mControlHz; }
    int getSimulationStep() const { return mSimulationStep; }
    std::string getMetadata() { return mMetadata; }

    bool isMirror(int character_idx = 0) { return mEnforceSymmetry && ((mHardPhaseClipping) ? (getNormalizedPhase() > 0.5) : (getLocalPhase(true) > 0.5)); }

    bool isFall();
    dart::simulation::WorldPtr getWorld() { return mWorld; }

    double getActionScale() { return mActionScale; }

    // Metabolic Reward
    void setIncludeMetabolicReward(bool _includeMetabolicReward) { mIncludeMetabolicReward = _includeMetabolicReward; }
    bool getIncludeMetabolicReward() { return mIncludeMetabolicReward; }
    void setMuscleNetwork(py::object nn)
    {
        if (!mLoadedMuscleNN)
        {
            std::vector<int> child_elem;

            for (int i = 0; i < mPrevNetworks.size(); i++)
            {
                mEdges.push_back(Eigen::Vector2i(i, mPrevNetworks.size()));
                child_elem.push_back(i);
            }
            mChildNetworks.push_back(child_elem);
        }

        mMuscleNN = nn;
        mLoadedMuscleNN = true;
    }
    void setMuscleNetworkWeight(py::object w)
    {
        if (!mLoadedMuscleNN)
        {
            std::vector<int> child_elem;

            for (int i = 0; i < mPrevNetworks.size(); i++)
            {
                mEdges.push_back(Eigen::Vector2i(i, mPrevNetworks.size()));
                child_elem.push_back(i);
            }
            mChildNetworks.push_back(child_elem);
        }
        mMuscleNN.attr("load_state_dict")(w);
        mLoadedMuscleNN = true;
    }

    int getNumAction() { return mAction.rows(); }
    int getNumActuatorAction() { return mNumActuatorAction; }

    MuscleTuple getRandomMuscleTuple() { return mRandomMuscleTuple; }
    Eigen::VectorXd getRandomDesiredTorque() { return mRandomDesiredTorque; }

    Eigen::VectorXd getRandomPrevOut() { return mRandomPrevOut; }
    Eigen::VectorXf getRandomWeight()
    {
        Eigen::VectorXf res = Eigen::VectorXf(1);
        res[0] = (float)mRandomWeight;
        return res;
    }

    bool getUseCascading() { return mUseCascading; }
    bool getUseMuscle() { return mUseMuscle; }
    bool isTwoLevelController() { return mCharacter->getActuatorType() == mass || mCharacter->getActuatorType() == mass_lower; }

    // get Reward Term
    void updateFootStep(bool isInit = false);
    Eigen::Vector3d getCurrentFootStep() { return mCurrentFoot; }
    Eigen::Vector3d getCurrentTargetFootStep() { return mCurrentTargetFoot; }
    Eigen::Vector3d getNextTargetFootStep() { return mNextTargetFoot; }
    bool getIsLeftLegStance() { return mIsLeftLegStance; }

    RewardType getRewardType() { return mRewardType; }

    double getEnergyReward();
    double getKneePainReward();
    double getKneePainMaxReward();
    double getKneeLoadingMaxCycle() const { return mKneeLoadingMaxCycle; }
    double getStepReward();
    double getAvgVelReward();
    double getLocoReward();
    Eigen::Vector3d getAvgVelocity();
    double getTargetCOMVelocity() { return (mRefStride * mStride * mCharacter->getGlobalRatio()) / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))); }
    double getNormalizedPhase() { return mGlobalTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))) - (int)(mGlobalTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))); }
    double getWorldPhase() { return mWorldTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))) - (int)(mWorldTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))); }
    
    // Time and cycle getters for rollout
    double getWorldTime() const { return mWorldTime; }
    int getWorldPhaseCount() const { return mWorldPhaseCount; }
    int getSimulationCount() const { return mSimulationCount; }

    // For Parameterization

    // Parameters
    double getCadence() { return mCadence; }
    double getStride() { return mStride; }

    void setParamState(Eigen::VectorXd _param_state, bool onlyMuscle = false, bool doOptimization = false);
    void setNormalizedParamState(Eigen::VectorXd _param_state, bool onlyMuscle = false, bool doOptimization = false);
    Eigen::VectorXd getParamState(bool isMirror = false);
    Eigen::VectorXd getNormalizedParamState(Eigen::VectorXd minV, Eigen::VectorXd maxV, bool isMirror = false)
    {
        Eigen::VectorXd norm_p = getParamState(isMirror);
        for (int i = 0; i < norm_p.rows(); i++)
        {
            double range = maxV[i] - minV[i];
            norm_p[i] = (std::abs(range) < 1e-9) ? 0.0 : (norm_p[i] - minV[i]) / range;
        }
        return norm_p;
    }
    const std::vector<std::string> &getParamName() { return mParamName; };
    Eigen::VectorXd getParamMin() { return mParamMin; }
    Eigen::VectorXd getParamMax() { return mParamMax; }
    Eigen::VectorXd getParamDefault() { return mParamDefault; }
    void setParamDefault() { setParamState(getParamDefault(), false, true); }
    Eigen::VectorXd getParamSample();
    const std::vector<param_group> &getGroupParam() { return mParamGroups; }
    void setGroupParam(Eigen::VectorXd v)
    {
        Eigen::VectorXd sampled_param = mParamMin;
        sampled_param.setOnes();
        int i = 0;
        for (auto &p : mParamGroups)
        {
            p.v = v[i];
            for (auto idx : p.param_idxs)
            {
                double param_w = mParamMax[idx] - mParamMin[idx];
                sampled_param[idx] = mParamMin[idx] + param_w * p.v;
            }
            i++;
        }
        setParamState(sampled_param, false, true);
    }
    int getNumParamState() { return mNumParamState; }
    void updateParamState() { setParamState(getParamSample(), false, true); }
    double getLimitY() { return mLimitY; }

    bool getLearningStd() { return mLearningStd; }
    void setLearningStd(bool learningStd) { mLearningStd = learningStd; }
    void poseOptimization(int iter = 100);

    Eigen::Vector2i getIsContact();
    Eigen::Vector2d getFootGRF(); // Get normalized GRF for left and right foot

    // For Cascading
    Network loadPrevNetworks(std::string path, bool isFirst); // Neot
    std::pair<Eigen::VectorXd, Eigen::VectorXd> getSpace(std::string metadata);
    std::vector<double> getWeights() { return mWeights; }
    std::vector<double> getDmins() { return mDmins; }
    std::vector<double> getBetas() { return mBetas; }
    // Reward configuration accessors
    RewardConfig& getRewardConfig() { return mRewardConfig; }
    const RewardConfig& getRewardConfig() const { return mRewardConfig; }

    // Legacy accessors for compatibility
    double getMetabolicWeight() { return mRewardConfig.metabolic_weight; }
    void setMetabolicWeight(double weight) { mRewardConfig.metabolic_weight = weight; }
    double getScaleMetabolic() { return mRewardConfig.metabolic_scale; }
    void setScaleMetabolic(double scale) { mRewardConfig.metabolic_scale = scale; }
    bool getSeparateTorqueEnergy() { return mRewardConfig.separate_torque_energy; }
    void setSeparateTorqueEnergy(bool separate) { mRewardConfig.separate_torque_energy = separate; }
    double getKneePainWeight() { return mRewardConfig.knee_pain_weight; }
    void setKneePainWeight(double weight) { mRewardConfig.knee_pain_weight = weight; }
    double getScaleKneePain() { return mRewardConfig.knee_pain_scale; }
    void setScaleKneePain(double scale) { mRewardConfig.knee_pain_scale = scale; }
    bool getUseMultiplicativeKneePain() { return mRewardConfig.multiplicative & REWARD_KNEE_PAIN; }
    void setUseMultiplicativeKneePain(bool use) {
        if (use) mRewardConfig.multiplicative |= REWARD_KNEE_PAIN;
        else mRewardConfig.multiplicative &= ~REWARD_KNEE_PAIN;
    }

    void setUseWeights(std::vector<bool> _useWeights)
    {
        for (int i = 0; i < _useWeights.size(); i++)
            mUseWeights[i] = _useWeights[i];
    }
    std::vector<bool> getUseWeights() { return mUseWeights; }

    const std::vector<Eigen::VectorXd> &getDesiredTorqueLogs() { return mDesiredTorqueLogs; }

    Eigen::VectorXd getParamStateFromNormalized(Eigen::VectorXd normalizedParamState)
    {
        Eigen::VectorXd paramState = Eigen::VectorXd::Zero(mNumParamState);
        
        for(int i = 0; i < mNumParamState; i++)
            paramState[i] = normalizedParamState[i] * (mParamMax[i] - mParamMin[i]) + mParamMin[i];
        return paramState;
    }

    Eigen::VectorXd getNormalizedParamStateFromParam(Eigen::VectorXd paramState)
    {
        Eigen::VectorXd norm_p = paramState;
        for (int i = 0; i < norm_p.rows(); i++)
            norm_p[i] = (norm_p[i] - mParamMin[i]) / (mParamMax[i] - mParamMin[i]);
        return norm_p;
    }
    int getNumKnownParam() {return mNumKnownParam;}
    double getGlobalTime() { return mGlobalTime; }

    // Gait cycle completion detection
    bool isGaitCycleComplete();
    int getGaitCycleCount() const { return mWorldPhaseCount; }

    void muscleStep();
    int getNumSubSteps() { return mNumSubSteps; }
    void postStep();
private:
    // Config parsing methods
    void parseEnvConfigXml(const std::string& metadata);
    void parseEnvConfigYaml(const std::string& filepath);

    // Step method components
    void calcActivation();
    void postMuscleStep();

    // Termination/truncation check methods (called in postStep)
    void checkTerminated();
    void checkTruncated();

    Eigen::VectorXd mTargetPositions;
    Eigen::VectorXd mTargetVelocities;
    double mActionScale;

    // Parameter (General)
    int mSimulationHz, mControlHz, mNumSubSteps;

    // Parameter (Muscle)
    bool mUseMuscle;

    int mInferencePerSim;

    // Simulation
    Eigen::VectorXd mAction;

    dart::simulation::WorldPtr mWorld;
    Character *mCharacter;
    std::vector<dart::dynamics::SkeletonPtr> mObjects;
    Motion *mMotion;

    // Metadata
    std::string mMetadata;

    // Residual Control
    bool mIsResidual;

    // [Advanced Option]
    bool mIncludeMetabolicReward;

    // Cyclic or Not
    bool mCyclic;

    int mSimulationCount, mSimulationStep;
    int mHeightCalibration; // 0 : No, 1 : Only avoid collision, 2: Strict
    bool mEnforceSymmetry;

    // Muscle Learning Tuple
    bool mTupleFilled;
    Eigen::VectorXd mRandomDesiredTorque;
    MuscleTuple mRandomMuscleTuple;
    Eigen::VectorXd mRandomPrevOut;
    double mRandomWeight;

    // Network
    py::object mMuscleNN;

    // Reward Type (Deep Mimic or GaitNet)
    RewardType mRewardType;
    double mReward;
    std::map<std::string, double> mInfoMap;

    // GaitNet
    double mRefStride;
    double mStride, mCadence;  // Ratio of Foot stride & time displacement [default = 1]
    bool mIsLeftLegStance;
    Eigen::Vector3d mNextTargetFoot, mCurrentTargetFoot, mCurrentFoot;
    double mPhaseDisplacement, mPhaseDisplacementScale;
    int mNumActuatorAction;

    double mLimitY; // For EOE

    // Offset for Stance phase at current bvh;
    double mStanceOffset;

    bool mLoadedMuscleNN, mUseJointState, mLearningStd;

    // Parameter
    std::vector<param_group> mParamGroups;
    Eigen::VectorXd mParamMin, mParamMax, mParamDefault;
    std::vector<std::string> mParamName;
    int mNumParamState;

    RewardConfig mRewardConfig;

    // Simulation Setting
    bool mSoftPhaseClipping, mHardPhaseClipping;
    int mPhaseCount, mWorldPhaseCount, mPrevWorldPhaseCount;
    EOEType mEOEType;
    double mGlobalTime, mWorldTime;

    // Pose Optimization
    bool mMusclePoseOptimization;
    int mPoseOptimizationMode;

    // Gait Analysis
    Eigen::Vector2i mPrevContact; // Previous contact state for heel strike detection
    double mKneeLoadingMaxCycle;  // Maximum knee loading for current gait cycle

    // nFor Cascading
    bool mUseCascading;
    std::vector<Network> mPrevNetworks;

    std::vector<Eigen::Vector2i> mEdges;
    std::vector<std::vector<int>> mChildNetworks;

    std::vector<Eigen::VectorXd> mProjStates, mProjJointStates;

    std::vector<double> mDmins, mWeights, mBetas;

    Eigen::VectorXd mState, mJointState;

    py::object loading_network;

    std::vector<bool> mUseWeights; // Only For Rendering
    int mHorizon;

    std::vector<Eigen::VectorXd> mDesiredTorqueLogs;

    bool mUseNormalizedParamState;
    int mNumKnownParam;
};
#endif
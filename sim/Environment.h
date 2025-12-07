#ifndef __MS_ENVIRONMENT_H__
#define __MS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "BVH_Parser.h"
#include "Character.h"
#include "GaitPhase.h"
#include "NoiseInjector.h"
#include "MuscleNN.h"
#include "DiscriminatorNN.h"
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
    py::object joint;        // Python joint network (for compatibility)
    MuscleNN muscle;         // C++ muscle network (libtorch)

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
    REWARD_METABOLIC = 1 << 0,  // 0x01
    REWARD_KNEE_PAIN = 1 << 1,  // 0x02
    REWARD_KNEE_PAIN_MAX = 1 << 2,  // 0x04
    REWARD_SEP_TORQUE_ENERGY = 1 << 3,  // 0x08
    TERM_KNEE_PAIN = 1 << 4,  // 0x10
    REWARD_AVG_VEL_CONSIDER_X = 1 << 5,  // 0x20
    REWARD_DRAG_X = 1 << 6,  // 0x40
    REWARD_PHASE = 1 << 7,  // 0x80
};

// Centralized reward configuration
struct RewardConfig
{
    int flags = REWARD_NONE;

    // Metabolic reward parameters
    double metabolic_weight = 0.05;
    double metabolic_scale = 1.0;

    // Knee pain reward parameters
    double knee_pain_weight = 1.0;
    double knee_pain_scale = 1.0;

    // Knee pain max (per gait cycle) reward parameters
    double knee_pain_max_weight = 1.0;

    // Locomotion reward weights (always active for gaitnet)
    double head_linear_acc_weight = 4.0;
    double head_rot_weight = 4.0;
    double step_weight = 2.0;
    double step_clip = 0.075;  // Z-axis step clipping value
    double avg_vel_weight = 6.0;
    double avg_vel_window_mult = 1.0;
    double avg_vel_clip = -1.0;  // -1 means no clipping
    double drag_weight = 1.0;
    double drag_x_threshold = 0.0;
    double phase_weight = 1.0;

    // Reward clipping for initial simulation steps
    int clip_step = 0;
    double clip_value = 0.0;
};

// ADD-style discriminator configuration (energy efficiency via muscle activation minimization)
struct DiscriminatorConfig
{
    bool enabled = false;           // Whether discriminator is enabled
    bool normalize = false;         // Whether to normalize disc_obs (optional)
    double reward_scale = 1.0;      // Scale factor for discriminator reward
    bool multiplicative = false;    // If true, multiplies with main reward; if false, additive
    bool upper_body = false;        // If true, include upper body torques in disc_obs
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

    Character *getCharacter() { return mCharacter; }
    Motion *getMotion() { return mMotion; }
    void setMotion(Motion* motion) { mMotion = motion; }
    NoiseInjector* getNoiseInjector() { return mNoiseInjector.get(); }
    void createNoiseInjector(const std::string& config_path);

    void setAction(Eigen::VectorXd _action);

    virtual void preStep();
    virtual void step();
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

    Eigen::VectorXd getRefPose() { return mRefPose; }
    Eigen::VectorXd getTargetVelocities() { return mTargetVelocities; }

    double getLocalPhase(bool mod_one = false) { return (mGaitPhase->getLocalTime() / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))) - (mod_one ? (int)(mGaitPhase->getLocalTime() / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))) : 0.0); }

    Eigen::VectorXd getState();
    std::pair<Eigen::VectorXd, Eigen::VectorXd> getProjState(const Eigen::VectorXd minV, const Eigen::VectorXd maxV);

    Eigen::VectorXd getJointState(bool isMirror);

    double calcReward();
    Eigen::VectorXd getAction() { return mAction; }

    int getSimulationHz() { return mSimulationHz; }
    int getControlHz() { return mControlHz; }
    int getSimulationStep() const { return mSimulationStep; }
    std::string getMetadata() { return mMetadata; }

    bool isMirror() { return getNormalizedPhase() > 0.5; }

    bool isFall();
    dart::simulation::WorldPtr getWorld() { return mWorld; }

    double getActionScale() { return mActionScale; }

    // Metabolic Reward
    void setIncludeMetabolicReward(bool _includeMetabolicReward) { mIncludeMetabolicReward = _includeMetabolicReward; }
    bool getIncludeMetabolicReward() { return mIncludeMetabolicReward; }
    // DEPRECATED: MuscleNN is now created in C++ during initialize()
    // This method is kept for backward compatibility but does nothing
    void setMuscleNetwork(py::object nn)
    {
        // Network is already created in initialize() with C++ libtorch
        // This method is now a no-op for backward compatibility
        std::cout << "Warning: setMuscleNetwork is deprecated. MuscleNN is created automatically." << std::endl;
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

        // Convert Python state_dict to C++ format
        std::unordered_map<std::string, torch::Tensor> state_dict;
        py::dict py_dict = w.cast<py::dict>();

        for (auto item : py_dict) {
            std::string key = item.first.cast<std::string>();
            py::array_t<float> np_array = item.second.cast<py::array_t<float>>();

            // Convert numpy array to torch::Tensor
            auto buf = np_array.request();
            std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());

            torch::Tensor tensor = torch::from_blob(
                buf.ptr,
                shape,
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone();  // Clone to own the memory

            state_dict[key] = tensor;
        }

        mMuscleNN->load_state_dict(state_dict);
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
    MuscleNN* getMuscleNN() { return &mMuscleNN; }

    // Discriminator accessors
    bool getUseDiscriminator() const { return mDiscConfig.enabled; }
    DiscriminatorConfig& getDiscConfig() { return mDiscConfig; }
    const DiscriminatorConfig& getDiscConfig() const { return mDiscConfig; }
    DiscriminatorNN* getDiscriminatorNN() { return &mDiscriminatorNN; }

    /**
     * Get randomly sampled disc_obs (muscle activations) for this control step.
     * Sampled per control step (like muscle tuples), not per substep.
     *
     * @return Muscle activations as VectorXf (num_muscles dimensions)
     */
    Eigen::VectorXf getRandomDiscObs() const { return mRandomDiscObs; }

    /**
     * Get discriminator observation dimension.
     * Dimension depends on config: num_muscles + (upper_body ? upper_body_dim : 0)
     *
     * @return Dimension of disc_obs vector
     */
    int getDiscObsDim() const;

    /**
     * Get current discriminator observation (disc_obs).
     * Composition depends on config:
     * - Base: muscle activations
     * - upper_body=true: [activations, upperBodyTorque]
     *
     * @return disc_obs as VectorXf
     */
    Eigen::VectorXf getDiscObs() const;

    /**
     * Get upper body DOF dimension (cached).
     * @return Number of upper body DOFs
     */
    int getUpperBodyDim() const { return mUpperBodyDim; }

    /**
     * Get mean activation for tensorboard logging.
     * @return Mean of absolute muscle activations
     */
    double getMeanActivation() const { return mMeanActivation; }

    // get Reward Term
    Eigen::Vector3d getCurrentFootStep() { return mGaitPhase->getCurrentFoot(); }
    Eigen::Vector3d getCurrentTargetFootStep() { return mGaitPhase->getCurrentTargetFoot(); }
    Eigen::Vector3d getNextTargetFootStep() { return mGaitPhase->getNextTargetFoot(); }
    bool getIsLeftLegStance() { return mGaitPhase->isLeftLegStance(); }

    // GaitPhase accessor
    GaitPhase* getGaitPhase() { return mGaitPhase.get(); }

    RewardType getRewardType() { return mRewardType; }

    double getEnergyReward();
    double getKneePainReward();
    double getKneePainMaxReward();
    double getKneeLoadingMaxCycle() const { return mKneeLoadingMaxCycle; }
    double getStepReward();
    double getAvgVelReward();
    double getHeadLinearAccReward();
    double getHeadRotReward();
    double getDragXReward();
    double getPhaseReward();
    Eigen::Vector3d getAvgVelocity();
    int getAvgVelocityHorizonSteps() const;
    double getTargetCOMVelocity() { return (mRefStride * mStride * mCharacter->getGlobalRatio()) / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))); }
    double getNormalizedPhase() { return mGlobalTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))) - (int)(mGlobalTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))); }
    double getWorldPhase() { return mWorldTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))) - (int)(mWorldTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))); }
    
    // Time and cycle getters for rollout
    double getWorldTime() const { return mWorldTime; }
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

    void poseOptimization(int iter = 100);

    // Contact detection (delegated to GaitPhase if available)
    Eigen::Vector2i getIsContact();
    Eigen::Vector2d getFootGRF();

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
    double getAvgVelWindowMult() const { return mRewardConfig.avg_vel_window_mult; }
    void setAvgVelWindowMult(double mult) { mRewardConfig.avg_vel_window_mult = (mult < 0.0 ? 0.0 : mult); }
    bool getAvgVelConsiderX() const { return mRewardConfig.flags & REWARD_AVG_VEL_CONSIDER_X; }
    void setAvgVelConsiderX(bool consider) { if (consider) mRewardConfig.flags |= REWARD_AVG_VEL_CONSIDER_X; else mRewardConfig.flags &= ~REWARD_AVG_VEL_CONSIDER_X; }
    bool getDragX() const { return mRewardConfig.flags & REWARD_DRAG_X; }
    void setDragX(bool drag) { if (drag) mRewardConfig.flags |= REWARD_DRAG_X; else mRewardConfig.flags &= ~REWARD_DRAG_X; }
    double getDragWeight() const { return mRewardConfig.drag_weight; }
    void setDragWeight(double weight) { mRewardConfig.drag_weight = (weight < 0.0 ? 0.0 : weight); }
    double getDragXThreshold() const { return mRewardConfig.drag_x_threshold; }
    void setDragXThreshold(double threshold) { mRewardConfig.drag_x_threshold = (threshold < 0.0 ? 0.0 : threshold); }

    double getStepWeight() const { return mRewardConfig.step_weight; }
    void setStepWeight(double weight) { mRewardConfig.step_weight = (weight < 0.0 ? 0.0 : weight); }

    double getStepClip() const { return mRewardConfig.step_clip; }
    void setStepClip(double clip) { mRewardConfig.step_clip = (clip < 0.0 ? 0.0 : clip); }
    double getAvgVelWeight() const { return mRewardConfig.avg_vel_weight; }
    void setAvgVelWeight(double weight) { mRewardConfig.avg_vel_weight = (weight < 0.0 ? 0.0 : weight); }
    double getAvgVelClip() const { return mRewardConfig.avg_vel_clip; }
    void setAvgVelClip(double clip) { mRewardConfig.avg_vel_clip = clip; }  // -1 means no clipping

    bool getPhaseRewardEnabled() const { return mRewardConfig.flags & REWARD_PHASE; }
    void setPhaseRewardEnabled(bool enabled) { if (enabled) mRewardConfig.flags |= REWARD_PHASE; else mRewardConfig.flags &= ~REWARD_PHASE; }
    double getPhaseWeight() const { return mRewardConfig.phase_weight; }
    void setPhaseWeight(double weight) { mRewardConfig.phase_weight = (weight < 0.0 ? 0.0 : weight); }
    bool getSeparateTorqueEnergy() { return mRewardConfig.flags & REWARD_SEP_TORQUE_ENERGY; }
    void setSeparateTorqueEnergy(bool separate) { mRewardConfig.flags |= REWARD_SEP_TORQUE_ENERGY; }
    double getKneePainWeight() { return mRewardConfig.knee_pain_weight; }
    void setKneePainWeight(double weight) { mRewardConfig.flags |= REWARD_KNEE_PAIN; mRewardConfig.knee_pain_weight = weight; }
    double getScaleKneePain() { return mRewardConfig.flags & REWARD_KNEE_PAIN ? mRewardConfig.knee_pain_scale : -1.0; }
    void setScaleKneePain(double scale) { if (mRewardConfig.flags & REWARD_KNEE_PAIN) mRewardConfig.knee_pain_scale = scale; else mRewardConfig.knee_pain_scale = -1.0; }
    bool getUseMultiplicativeKneePain() { return mRewardConfig.flags & REWARD_KNEE_PAIN; }
    void setUseMultiplicativeKneePain(bool use) {
        if (use) mRewardConfig.flags |= REWARD_KNEE_PAIN;
        else mRewardConfig.flags &= ~REWARD_KNEE_PAIN;
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
    void clearGaitCycleComplete();  // Clear the PD-level completion flag
    int getGaitCycleCount() const { return mWorldPhaseCount; }

    // Step completion detection
    bool isStepComplete();
    void clearStepComplete();  // Clear the PD-level step completion flag

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

    Eigen::VectorXd mRefPose;
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
    dart::dynamics::SkeletonPtr mGround;
    Motion *mMotion;
    std::unique_ptr<NoiseInjector> mNoiseInjector;

    // Metadata
    std::string mMetadata;

    // Residual Control
    bool mIsResidual;

    // [Advanced Option]
    bool mIncludeMetabolicReward;

    int mSimulationCount, mSimulationStep;
    bool mEnforceSymmetry;

    // Muscle Learning Tuple
    bool mTupleFilled;
    Eigen::VectorXd mRandomDesiredTorque;
    MuscleTuple mRandomMuscleTuple;
    Eigen::VectorXd mRandomPrevOut;
    double mRandomWeight;

    // Network (C++ libtorch for thread-safe inference)
    MuscleNN mMuscleNN;

    // Discriminator for energy-efficient muscle activation (ADD-style)
    DiscriminatorConfig mDiscConfig;
    DiscriminatorNN mDiscriminatorNN;
    bool mLoadedDiscriminatorNN = false;
    Eigen::VectorXf mRandomDiscObs;     // Sampled disc_obs per control step (like muscle tuples)
    bool mDiscObsFilled = false;        // Whether disc_obs has been sampled this step
    double mMeanActivation = 0.0;       // Mean activation for tensorboard logging
    double mDiscRewardAccum = 0.0;      // Accumulated discriminator reward across substeps
    int mUpperBodyDim = 0;              // Cached upper body DOF dimension for disc_obs

    // Reward Type (Deep Mimic or GaitNet)
    RewardType mRewardType;
    double mReward;
    std::map<std::string, double> mInfoMap;

    // GaitNet
    double mRefStride;
    double mStride, mCadence;  // Ratio of Foot stride & time displacement [default = 1]
    std::unique_ptr<GaitPhase> mGaitPhase;  // Gait phase tracking (stores update mode internally)
    double mPhaseDisplacementScale;
    int mNumActuatorAction;

    double mLimitY; // For EOE

    // Offset for Stance phase at current bvh;
    double mStanceOffset;

    bool mLoadedMuscleNN, mUseJointState;

    // Parameter
    std::vector<param_group> mParamGroups;
    Eigen::VectorXd mParamMin, mParamMax, mParamDefault;
    std::vector<std::string> mParamName;
    int mNumParamState;

    RewardConfig mRewardConfig;

    // Simulation Setting
    bool mSoftPhaseClipping, mHardPhaseClipping;
    int mPhaseCount, mWorldPhaseCount;
    double mGlobalTime, mWorldTime;

    // Pose Optimization
    bool mMusclePoseOptimization;
    int mPoseOptimizationMode;

    // Gait Analysis
    double mKneeLoadingMaxCycle;  // Maximum knee loading for current gait cycle
    double mDragStartX;

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

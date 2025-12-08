#include "Environment.h"
#include "UriResolver.h"
#include "CBufferData.h"
#include "NPZ.h"
#include "HDF.h"
#include "Log.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <random>

// Thread-safe random number generation
// Each thread gets its own random number generator to avoid data races
namespace {
    thread_local std::mt19937 thread_rng(std::random_device{}());

    // Thread-safe replacement for dart::math::Random::uniform(min, max)
    inline double thread_safe_uniform(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(thread_rng);
    }

    // Thread-safe replacement for dart::math::Random::normal(mean, stddev)
    inline double thread_safe_normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(thread_rng);
    }
}

Environment::Environment()
    : mSimulationHz(600), mControlHz(30), mUseMuscle(false), mInferencePerSim(1),
    mEnforceSymmetry(false), mLimitY(0.6)
{
    // Initialize URI resolver for path resolution
    PMuscle::URIResolver::getInstance().initialize();

    mWorld = std::make_shared<dart::simulation::World>();
    mIsResidual = true;
    mSimulationCount = 0;
    mActionScale = 0.04;
    mIncludeMetabolicReward = true;
    mRewardType = deepmimic;
    mStanceOffset = 0.07;

    // GaitNet
    mRefStride = 1.34;
    mStride = 1.0;
    mCadence = 1.0;
    mPhaseDisplacementScale = -1.0;
    mNumActuatorAction = 0;

    mLoadedMuscleNN = false;
    mUseJointState = false;
    // Parameter
    mNumParamState = 0;

    // Simulation Setting
    mSimulationStep = 0;

    mSoftPhaseClipping = false;
    mHardPhaseClipping = false;
    mPhaseCount = 0;
    mWorldPhaseCount = 0;
    mKneeLoadingMaxCycle = 0.0;
    mGlobalTime = 0.0;
    mWorldTime = 0.0;
    mDragStartX = 0.0;

    mMusclePoseOptimization = false;

    mUseCascading = false;
    mUseNormalizedParamState = true;
    // 0 : one foot , 1 : mid feet
    mPoseOptimizationMode = 0;
    mHorizon = 600;

    // Initialize reward config with defaults (already set in struct definition)
}

Environment::~Environment()
{
}

void Environment::initialize(std::string content)
{
    mMetadata = content;

    // Auto-detect format by examining first non-whitespace character
    size_t start = content.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        // Empty content - use YAML as default
        parseEnvConfigYaml(content);
        return;
    }

    // Detect format: XML starts with '<', YAML doesn't
    if (content[start] == '<') parseEnvConfigXml(content);
    else parseEnvConfigYaml(content);
}

void Environment::initialize_xml(std::string xml_content)
{
    // Backward compatibility: XML content
    mMetadata = xml_content;
    parseEnvConfigXml(xml_content);
}

void Environment::parseEnvConfigXml(const std::string& metadata)
{
    TiXmlDocument doc;
    doc.Parse(metadata.c_str());

    // Cascading Setting
    if (doc.FirstChildElement("cascading") != NULL)
        mUseCascading = true;

    // Skeleton Loading
    if (TiXmlElement* skeletonElem = doc.FirstChildElement("skeleton"))
    {
        std::string skeletonPath = Trim(std::string(skeletonElem->GetText()));
        std::string resolvedSkeletonPath = PMuscle::URIResolver::getInstance().resolve(skeletonPath);
        mCharacter = new Character(resolvedSkeletonPath, SKEL_DEFAULT);

        std::string _actTypeString;
        if (skeletonElem->Attribute("actuator") != NULL) _actTypeString = Trim(skeletonElem->Attribute("actuator"));
        else if (skeletonElem->Attribute("actuactor") != NULL) _actTypeString = Trim(skeletonElem->Attribute("actuactor"));
        ActuatorType _actType = getActuatorType(_actTypeString);
        mCharacter->setActuatorType(_actType);

        mRefPose = mCharacter->getSkeleton()->getPositions();
        mTargetVelocities = mCharacter->getSkeleton()->getVelocities();
    }

    // Muscle Loading
    if (doc.FirstChildElement("muscle") != NULL)
    {
        // Check LBS Weight Setting
        bool meshLbsWeight = false;
        bool useVelocityForce = false;

        if (doc.FirstChildElement("meshLbsWeight") != NULL)
            meshLbsWeight = doc.FirstChildElement("meshLbsWeight")->BoolText();

        if (doc.FirstChildElement("useVelocityForce") != NULL)
            useVelocityForce = doc.FirstChildElement("useVelocityForce")->BoolText();

        if (doc.FirstChildElement("useJointState") != NULL)
            mUseJointState = doc.FirstChildElement("useJointState")->BoolText();

        std::string muscle_path = Trim(std::string(doc.FirstChildElement("muscle")->GetText()));
        std::string resolvedMusclePath = PMuscle::URIResolver::getInstance().resolve(muscle_path);
        mCharacter->setMuscles(resolvedMusclePath, useVelocityForce, meshLbsWeight);
        mUseMuscle = true;
    }
    
    // Phase Displacement Reward
    if (doc.FirstChildElement("timeWarping") != NULL)
        mPhaseDisplacementScale = doc.FirstChildElement("timeWarping")->DoubleText();

    // mAction Setting
    ActuatorType _actType = mCharacter->getActuatorType();
    if (_actType == tor || _actType == pd || _actType == mass || _actType == mass_lower)
    {
        mAction = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs() + (mPhaseDisplacementScale > 0 ? 1 : 0) + (mUseCascading ? 1 : 0));
        mNumActuatorAction = mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs();
    }
    else if (_actType == mus)
    {
        mAction = Eigen::VectorXd::Zero(mCharacter->getMuscles().size() + (mPhaseDisplacementScale > 0 ? 1 : 0) + (mUseCascading ? 1 : 0));
        mNumActuatorAction = mCharacter->getMuscles().size();
    }
    // Ground Loading (hardcoded)
    mGround = BuildFromFile(PMuscle::URIResolver::getInstance().resolve("@data/ground.xml"), SKEL_DEFAULT);

    // Controller Setting (hardcoded)
    mIsResidual = true;

    // Simulation Setting
    if (doc.FirstChildElement("simHz") != NULL)
        mSimulationHz = doc.FirstChildElement("simHz")->IntText();
    if (doc.FirstChildElement("controlHz") != NULL)
        mControlHz = doc.FirstChildElement("controlHz")->IntText();

    if (mSimulationHz % mControlHz != 0) {
        std::cout << "[ERROR] Simulation Hz must be divisible by control Hz. Got " << mSimulationHz << " / " << mControlHz << " != 0" << std::endl;
        exit(-1);
    }
    mNumSubSteps = mSimulationHz / mControlHz;

    // Action Scale
    if (doc.FirstChildElement("actionScale") != NULL)
        mActionScale = doc.FirstChildElement("actionScale")->DoubleText();

    // Inference Per Sim (hardcoded)
    mInferencePerSim = 1;

    // Phase Clipping (hardcoded)
    mSoftPhaseClipping = false;
    mHardPhaseClipping = true;

    if (doc.FirstChildElement("musclePoseOptimization") != NULL)
    {
        if (doc.FirstChildElement("musclePoseOptimization")->Attribute("rot") != NULL)
        {
            if (std::string(doc.FirstChildElement("musclePoseOptimization")->Attribute("rot")) == "one_foot")
                mPoseOptimizationMode = 0;
            else if (std::string(doc.FirstChildElement("musclePoseOptimization")->Attribute("rot")) == "mid_feet")
                mPoseOptimizationMode = 1;
        }
        mMusclePoseOptimization = doc.FirstChildElement("musclePoseOptimization")->BoolText();
    }

    // Advanced settings (hardcoded)
    mCharacter->setTorqueClipping(false);
    mCharacter->setIncludeJtPinSPD(false);

    if (doc.FirstChildElement("rewardType") != NULL)
    {
        std::string str_rewardType = doc.FirstChildElement("rewardType")->GetText();
        if (str_rewardType == "deepmimic")
            mRewardType = deepmimic;
        if (str_rewardType == "gaitnet")
            mRewardType = gaitnet;
        if (str_rewardType == "scadiver")
            mRewardType = scadiver;
    }

    // EOEType is always tuple (hardcoded)

    // Simulation World Wetting
    mWorld->setTimeStep(1.0 / mSimulationHz);
    // mWorld->getConstraintSolver()->setLCPSolver(dart::common::make_unique<dart::constraint::PGSLCPSolver>(mWorld->getTimeStep));
    // mWorld->setConstraintSolver(std::make_unique<dart::constraint::BoxedLcpConstraintSolver>(std::make_shared<dart::constraint::PgsBoxedLcpSolver>()));
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
    // Add Character
    mWorld->addSkeleton(mCharacter->getSkeleton());
    // Add Ground
    mWorld->addSkeleton(mGround);

    // Motion Loading (BVH or NPZ)
    // World Setting 후에 함. 왜냐하면 Height Calibration 을 위해서는 충돌 감지를 필요로 하기 때문.
    if (doc.FirstChildElement("bvh") != NULL)
    {
        std::string bvh_path = Trim(std::string(doc.FirstChildElement("bvh")->GetText()));
        std::string resolvedBvhPath = PMuscle::URIResolver::getInstance().resolve(bvh_path);
        LOG_VERBOSE("[Environment] BVH Path resolved: " << bvh_path << " -> " << resolvedBvhPath);
        BVH *new_bvh = new BVH(resolvedBvhPath);
        new_bvh->setMode(std::string(doc.FirstChildElement("bvh")->Attribute("symmetry")) == "true");

        new_bvh->setRefMotion(mCharacter, mWorld);
        mMotion = new_bvh;
    }
    else if (doc.FirstChildElement("npz") != NULL)
    {
        std::string npz_path = Trim(std::string(doc.FirstChildElement("npz")->GetText()));
        std::string resolvedNpzPath = PMuscle::URIResolver::getInstance().resolve(npz_path);
        LOG_VERBOSE("[Environment] NPZ Path resolved: " << npz_path << " -> " << resolvedNpzPath);
        NPZ *new_npz = new NPZ(resolvedNpzPath);

        new_npz->setRefMotion(mCharacter, mWorld);
        mMotion = new_npz;
    }
    else if (doc.FirstChildElement("hdf") != NULL || doc.FirstChildElement("h5") != NULL)
    {
        TiXmlElement* hdfElement = doc.FirstChildElement("hdf");
        if (hdfElement == NULL)
            hdfElement = doc.FirstChildElement("h5");

        std::string hdf_path = Trim(std::string(hdfElement->GetText()));
        std::string resolvedHdfPath = PMuscle::URIResolver::getInstance().resolve(hdf_path);
        LOG_VERBOSE("[Environment] HDF Path resolved: " << hdf_path << " -> " << resolvedHdfPath);

        HDF *new_hdf = new HDF(resolvedHdfPath);
        new_hdf->setRefMotion(mCharacter, mWorld);
        mMotion = new_hdf;
    }


    // Advanced Option (hardcoded)
    mEnforceSymmetry = true;

    if (isTwoLevelController())
    {
        Character *character = mCharacter;
        // Create C++ MuscleNN (libtorch) for thread-safe inference
        // Force CPU to avoid CUDA context allocation issues in multi-process scenarios
        mMuscleNN = make_muscle_nn(character->getNumMuscleRelatedDof(), getNumActuatorAction(), character->getNumMuscles(), mUseCascading, true);
        mLoadedMuscleNN = true;
    }

    if (doc.FirstChildElement("Horizon") != NULL)
        mHorizon = doc.FirstChildElement("Horizon")->IntText();

    // =================== Reward ======================
    // =================================================

    // Use normalized param state (hardcoded)
    mUseNormalizedParamState = false;

    if (doc.FirstChildElement("HeadLinearAccWeight") != NULL)
        mRewardConfig.head_linear_acc_weight = doc.FirstChildElement("HeadLinearAccWeight")->DoubleText();

    if (doc.FirstChildElement("HeadRotWeight") != NULL)
        mRewardConfig.head_rot_weight = doc.FirstChildElement("HeadRotWeight")->DoubleText();

    if (doc.FirstChildElement("StepWeight") != NULL)
        mRewardConfig.step_weight = doc.FirstChildElement("StepWeight")->DoubleText();

    if (doc.FirstChildElement("MetabolicWeight") != NULL)
        mRewardConfig.metabolic_weight = doc.FirstChildElement("MetabolicWeight")->DoubleText();

    if (doc.FirstChildElement("AvgVelWeight") != NULL)
        mRewardConfig.avg_vel_weight = doc.FirstChildElement("AvgVelWeight")->DoubleText();

    if (doc.FirstChildElement("AvgVelWindowMult") != NULL)
        mRewardConfig.avg_vel_window_mult = doc.FirstChildElement("AvgVelWindowMult")->DoubleText();

    if (doc.FirstChildElement("AvgVelConsiderX") != NULL) {
        if (doc.FirstChildElement("AvgVelConsiderX")->BoolText())
            mRewardConfig.flags |= REWARD_AVG_VEL_CONSIDER_X;
        else
            mRewardConfig.flags &= ~REWARD_AVG_VEL_CONSIDER_X;
    }

    if (doc.FirstChildElement("DragX") != NULL) {
        if (doc.FirstChildElement("DragX")->BoolText())
            mRewardConfig.flags |= REWARD_DRAG_X;
        else
            mRewardConfig.flags &= ~REWARD_DRAG_X;
    }

    if (doc.FirstChildElement("DragWeight") != NULL)
        mRewardConfig.drag_weight = doc.FirstChildElement("DragWeight")->DoubleText();

    if (doc.FirstChildElement("DragXThreshold") != NULL)
        mRewardConfig.drag_x_threshold = doc.FirstChildElement("DragXThreshold")->DoubleText();

    if (doc.FirstChildElement("ScaleMetabolic") != NULL)
        mRewardConfig.metabolic_scale = doc.FirstChildElement("ScaleMetabolic")->DoubleText();

    if (doc.FirstChildElement("KneePainWeight") != NULL) {
        mRewardConfig.flags |= REWARD_KNEE_PAIN;
        mRewardConfig.knee_pain_weight = doc.FirstChildElement("KneePainWeight")->DoubleText();
    }

    if (doc.FirstChildElement("ScaleKneePain") != NULL)
        mRewardConfig.knee_pain_scale = doc.FirstChildElement("ScaleKneePain")->DoubleText();

    if (doc.FirstChildElement("UseMultiplicativeKneePain") != NULL)
    {
        std::string useMultStr = Trim(std::string(doc.FirstChildElement("UseMultiplicativeKneePain")->GetText()));
        if (useMultStr == "true" || useMultStr == "True" || useMultStr == "TRUE")
            mRewardConfig.flags |= REWARD_KNEE_PAIN;
    }

    if (doc.FirstChildElement("UseKneePainTermination") != NULL)
    {
        std::string useTermStr = Trim(std::string(doc.FirstChildElement("UseKneePainTermination")->GetText()));
        mRewardConfig.flags |= TERM_KNEE_PAIN;
    }

    if (doc.FirstChildElement("UseMultiplicativeMetabolic") != NULL)
    {
        std::string useMultStr = Trim(std::string(doc.FirstChildElement("UseMultiplicativeMetabolic")->GetText()));
        if (useMultStr == "true" || useMultStr == "True" || useMultStr == "TRUE")
            mRewardConfig.flags |= REWARD_METABOLIC;
    }

    // Parse MetabolicType configuration
    if (doc.FirstChildElement("MetabolicType") != NULL)
    {
        std::string metabolicTypeStr = Trim(std::string(doc.FirstChildElement("MetabolicType")->GetText()));
        if (metabolicTypeStr == "LEGACY") mCharacter->setMetabolicType(LEGACY);
        else if (metabolicTypeStr == "A") mCharacter->setMetabolicType(A);
        else if (metabolicTypeStr == "A2") mCharacter->setMetabolicType(A2);
        else if (metabolicTypeStr == "MA") mCharacter->setMetabolicType(MA);
        else if (metabolicTypeStr == "MA2") mCharacter->setMetabolicType(MA2);
    }

    // Parse TorqueCoeff configuration
    if (doc.FirstChildElement("TorqueCoeff") != NULL)
    {
        double torqueCoeff = doc.FirstChildElement("TorqueCoeff")->DoubleText();
        mCharacter->setTorqueEnergyCoeff(torqueCoeff);
    }

    // ============= For parameterization ==============
    // =================================================

    std::vector<double> minV;
    std::vector<double> maxV;
    std::vector<double> defaultV;
    if (doc.FirstChildElement("parameter") != NULL)
    {
        auto parameter = doc.FirstChildElement("parameter");
        for (TiXmlElement *group = parameter->FirstChildElement(); group != NULL; group = group->NextSiblingElement())
        {
            for (TiXmlElement *elem = group->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
            {
                minV.push_back(std::stod(elem->Attribute("min")));
                maxV.push_back(std::stod(elem->Attribute("max")));
                if (elem->Attribute("default") == NULL)
                    defaultV.push_back(1.0);
                else
                    defaultV.push_back(std::stod(elem->Attribute("default")));

                mParamName.push_back(std::string(group->Name()) + "_" + std::string(elem->Name()));

                // Determine sampling strategy for this parameter
                bool is_uniform = (elem->Attribute("sampling") != NULL) && (std::string(elem->Attribute("sampling")) == "uniform");

                bool isExist = false;

                if (elem->Attribute("group") != NULL)
                {
                    std::string group_name = std::string(group->Name()) + "_" + elem->Attribute("group");
                    for (auto &p : mParamGroups)
                    {
                        if (p.name == group_name)
                        {
                            p.param_names.push_back(mParamName.back());
                            p.param_idxs.push_back(mParamName.size() - 1);
                            isExist = true;
                        }
                    }
                    if (!isExist)
                    {
                        param_group p;
                        p.name = group_name;
                        p.param_idxs.push_back(mParamName.size() - 1);
                        p.param_names.push_back(mParamName.back());
                        double range = maxV.back() - minV.back();
                        p.v = (std::abs(range) < 1e-9) ? 0.0 : (defaultV.back() - minV.back()) / range;
                        p.is_uniform = is_uniform;
                        mParamGroups.push_back(p);
                    }
                }
                else
                {
                    param_group p;
                    p.name = mParamName.back();
                    p.param_idxs.push_back(mParamName.size() - 1);
                    p.param_names.push_back(mParamName.back());
                    double range = maxV.back() - minV.back();
                    p.v = (std::abs(range) < 1e-9) ? 0.0 : (defaultV.back() - minV.back()) / range;
                    p.is_uniform = is_uniform;
                    mParamGroups.push_back(p);
                }
            }
        }
    }

    mParamMin = Eigen::VectorXd::Zero(minV.size());
    mParamMax = Eigen::VectorXd::Zero(minV.size());
    mParamDefault = Eigen::VectorXd::Zero(minV.size());

    for (int i = 0; i < minV.size(); i++)
    {
        mParamMin[i] = minV[i];
        mParamMax[i] = maxV[i];
        mParamDefault[i] = defaultV[i];
    }

    mNumParamState = minV.size();

    // ================== Cascading ====================

    if (doc.FirstChildElement("cascading") != NULL)
    {
        mPrevNetworks.clear();
        mEdges.clear();
        mChildNetworks.clear();
        if (mUseCascading)
        {
            loading_network = py::module::import("python.ray_model").attr("loading_network");
            auto networks = doc.FirstChildElement("cascading")->FirstChildElement();
            auto edges = doc.FirstChildElement("cascading")->LastChildElement();
            int idx = 0;
            for (TiXmlElement *network = networks->FirstChildElement(); network != NULL; network = network->NextSiblingElement()) {
                std::string networkPath = network->GetText();
                std::string resolvedNetworkPath = PMuscle::URIResolver::getInstance().resolve(networkPath);
                mPrevNetworks.push_back(loadPrevNetworks(resolvedNetworkPath, (idx++ == 0)));
            }

            for (TiXmlElement *edge_ = edges->FirstChildElement(); edge_ != NULL; edge_ = edge_->NextSiblingElement())
            {
                Eigen::Vector2i edge = Eigen::Vector2i(std::stoi(edge_->Attribute("start")), std::stoi(edge_->Attribute("end")));
                mEdges.push_back(edge);
            }

            for (int i = 0; i < mPrevNetworks.size(); i++)
            {
                std::vector<int> child_elem;
                mChildNetworks.push_back(child_elem);
            }
            for (auto e : mEdges)
                mChildNetworks[e[1]].push_back(e[0]);
        }
    }

    // =================================================
    // =================================================
    mUseWeights.clear();
    for (int i = 0; i < mPrevNetworks.size() + 1; i++)
    {
        mUseWeights.push_back(true);
        if (mUseMuscle)
            mUseWeights.push_back(true);
    }

    // set num known param which is the dof of gait parameters and skeleton parameters
    // find paramname which include "skeleton" or "stride" or "cadence"
    mNumKnownParam = 0;
    for(int i = 0; i < mParamName.size(); i++)
    {
        if (mParamName[i].find("skeleton") != std::string::npos || mParamName[i].find("stride") != std::string::npos || mParamName[i].find("cadence") != std::string::npos || mParamName[i].find("torsion") != std::string::npos)
            mNumKnownParam++;
    }
    // std::cout << "Num Known Param : " << mNumKnownParam << std::endl;

    // Initialize GaitPhase after all configuration is loaded (default to PHASE mode)
    mGaitPhase = std::make_unique<GaitPhase>(mCharacter, mWorld, mMotion->getMaxTime(), mRefStride, GaitPhase::PHASE, mControlHz, mSimulationHz);
}

void Environment::parseEnvConfigYaml(const std::string& yaml_content)
{
    YAML::Node config = YAML::Load(yaml_content);
    if (!config["environment"]) {
        throw std::runtime_error("Missing 'environment' key in YAML config");
    }
    YAML::Node env = config["environment"];

    // Local variable to track gait phase mode during parsing
    std::string gaitUpdateMode = "phase";  // Default to phase-based mode

    // === Cascading ===
    if (config["cascading"])
        mUseCascading = true;

    // === Skeleton ===
    if (env["skeleton"]) {
        auto skel = env["skeleton"];
        std::string skelPath = skel["file"].as<std::string>();
        std::string resolved = PMuscle::URIResolver::getInstance().resolve(skelPath);
        bool selfCollide = skel["self_collide"].as<bool>(false);
        int skelFlags = SKEL_DEFAULT;
        if (selfCollide) skelFlags |= SKEL_COLLIDE_ALL;
        mCharacter = new Character(resolved, skelFlags);

        std::string actType = skel["actuator"].as<std::string>();
        mCharacter->setActuatorType(getActuatorType(actType));

        mRefPose = mCharacter->getSkeleton()->getPositions();
        mTargetVelocities = mCharacter->getSkeleton()->getVelocities();
    }

    // === Muscle ===
    if (env["muscle"]) {
        auto muscle = env["muscle"];

        bool meshLbs = muscle["mesh_lbs_weight"].as<bool>(false);
        bool useVelForce = muscle["use_velocity_force"].as<bool>(false);
        mUseJointState = muscle["use_joint_state"].as<bool>(false);

        std::string musclePath = muscle["file"].as<std::string>();
        std::string resolved = PMuscle::URIResolver::getInstance().resolve(musclePath);
        mCharacter->setMuscles(resolved, useVelForce, meshLbs);
        mUseMuscle = true;

        if (muscle["pose_optimization"]) {
            auto poseOpt = muscle["pose_optimization"];
            mMusclePoseOptimization = poseOpt["enabled"].as<bool>(false);
            std::string rot = poseOpt["rot"].as<std::string>("one_foot");
            mPoseOptimizationMode = (rot == "one_foot") ? 0 : 1;
        }
    }

    // === Noise Injection ===
    if (env["noise_injection"]) {
        auto ni = env["noise_injection"];
        if (ni["file"]) {
            std::string niPath = ni["file"].as<std::string>();
            std::string resolved = PMuscle::URIResolver::getInstance().resolve(niPath);
            mNoiseInjector = std::make_unique<NoiseInjector>(resolved, mWorld->getTimeStep());
            LOG_INFO("[Environment] Loaded noise injection config: " << resolved);
        }
    } else {
        // Create default NoiseInjector (disabled by default)
        mNoiseInjector = std::make_unique<NoiseInjector>("", mWorld->getTimeStep());
        mNoiseInjector->setEnabled(false);
        LOG_VERBOSE("[Environment] Created default NoiseInjector (disabled)");
    }

    // === Action ===
    double contactDebounceAlpha = 0.25;  // Default value
    double stepMinRatio = 0.3;  // Default value
    if (env["action"]) {
        auto action = env["action"];
        if (action["time_warping"])
            mPhaseDisplacementScale = action["time_warping"].as<double>(-1.0);
        if (action["gait_phase_mode"])
            gaitUpdateMode = action["gait_phase_mode"].as<std::string>("phase");
        if (action["contact_debounce_alpha"])
            contactDebounceAlpha = action["contact_debounce_alpha"].as<double>(0.25);
        if (action["step_min_ratio"])
            stepMinRatio = action["step_min_ratio"].as<double>(0.3);
    }

    // === mAction sizing ===
    ActuatorType _actType = mCharacter->getActuatorType();
    if (_actType == tor || _actType == pd || _actType == mass || _actType == mass_lower) {
        mAction = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs() + (mPhaseDisplacementScale > 0 ? 1 : 0) + (mUseCascading ? 1 : 0));
        mNumActuatorAction = mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs();
    }
    else if (_actType == mus) {
        mAction = Eigen::VectorXd::Zero(mCharacter->getMuscles().size() + (mPhaseDisplacementScale > 0 ? 1 : 0) + (mUseCascading ? 1 : 0));
        mNumActuatorAction = mCharacter->getMuscles().size();
    }

    // === Ground === (hardcoded)
    mGround = BuildFromFile(PMuscle::URIResolver::getInstance().resolve("@data/ground.xml"), SKEL_DEFAULT);

    // === Motion === (hardcoded)
    // Height calibration is always applied in strict mode (no config needed)

    // === Action (residual) === (hardcoded)
    mIsResidual = true;

    // === Simulation ===
    if (env["simulation"]) {
        auto sim = env["simulation"];
        mSimulationHz = sim["sim_hz"].as<int>(600);
        mControlHz = sim["control_hz"].as<int>(30);

        if (mSimulationHz % mControlHz != 0) {
            std::cout << "[ERROR] sim_hz must be divisible by control_hz. Got " << mSimulationHz << " / " << mControlHz << " != 0" << std::endl;
            exit(-1);
        }
        mNumSubSteps = mSimulationHz / mControlHz;
    }

    // === Action scale ===
    if (env["action"] && env["action"]["scale"])
        mActionScale = env["action"]["scale"].as<double>(0.04);

    // === Inference per sim === (hardcoded)
    mInferencePerSim = 1;

    // === Advanced === (hardcoded)
    {
        mSoftPhaseClipping = false;
        mHardPhaseClipping = true;
        mCharacter->setTorqueClipping(false);
        mCharacter->setIncludeJtPinSPD(false);
    }

    // === Reward Type ===
    if (env["reward"] && env["reward"]["type"]) {
        std::string rewardType = env["reward"]["type"].as<std::string>("deepmimic");
        if (rewardType == "gaitnet") mRewardType = gaitnet;
        else if (rewardType == "deepmimic") mRewardType = deepmimic;
        else if (rewardType == "scadiver") mRewardType = scadiver;
    }

    // === World Setup ===
    mWorld->setTimeStep(1.0 / mSimulationHz);
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
    mWorld->addSkeleton(mCharacter->getSkeleton());
    mWorld->addSkeleton(mGround);

    // === Motion Loading ===
    if (env["motion"]) {
        auto motion = env["motion"];
        std::string motionPath = motion["file"].as<std::string>();
        std::string resolved = PMuscle::URIResolver::getInstance().resolve(motionPath);

        std::string motionType = motion["type"].as<std::string>();
        if (motionType == "h5" || motionType == "hdf") {
            HDF *new_hdf = new HDF(resolved);
            new_hdf->setRefMotion(mCharacter, mWorld);
            mMotion = new_hdf;
        }
        else if (motionType == "bvh") {
            BVH *new_bvh = new BVH(resolved);
            // Handle BVH attributes from YAML if needed
            new_bvh->setRefMotion(mCharacter, mWorld);
            mMotion = new_bvh;
        }
        else if (motionType == "npz") {
            NPZ *new_npz = new NPZ(resolved);
            new_npz->setRefMotion(mCharacter, mWorld);
            mMotion = new_npz;
        }
    }

    // === Advanced: height calibration already handled above ===

    // === Advanced: enforce symmetry === (hardcoded)
    mEnforceSymmetry = true;

    // === Two-level controller ===
    if (isTwoLevelController()) {
        Character *character = mCharacter;
        // Create C++ MuscleNN (libtorch) for thread-safe inference
        // Force CPU to avoid CUDA context allocation issues in multi-process scenarios
        mMuscleNN = make_muscle_nn(character->getNumMuscleRelatedDof(), getNumActuatorAction(), character->getNumMuscles(), mUseCascading, true);
        mLoadedMuscleNN = true;
    }

    // === Horizon ===
    if (config["Horizon"])
        mHorizon = config["Horizon"].as<int>();

    // === Reward parameters === (hardcoded)
    mUseNormalizedParamState = false;

    // === Reward clipping ===
    if (env["reward"] && env["reward"]["clip"]) {
        auto clip = env["reward"]["clip"];
        if (clip["step"])
            mRewardConfig.clip_step = clip["step"].as<int>(mRewardConfig.clip_step);
        if (clip["value"])
            mRewardConfig.clip_value = std::abs(clip["value"].as<double>(mRewardConfig.clip_value));
    }

    // === Locomotion rewards ===
    if (env["reward"] && env["reward"]["locomotion"]) {
        auto loco = env["reward"]["locomotion"];
        if (loco["head_linear_acc_weight"])
            mRewardConfig.head_linear_acc_weight = loco["head_linear_acc_weight"].as<double>(4.0);
        if (loco["head_rot_weight"])
            mRewardConfig.head_rot_weight = loco["head_rot_weight"].as<double>(4.0);

        // Parse step configuration (hierarchical or flat for backward compatibility)
        if (loco["step"]) {
            // New hierarchical structure
            auto step = loco["step"];
            if (step["weight"])
                mRewardConfig.step_weight = step["weight"].as<double>(2.0);
            if (step["clip"])
                mRewardConfig.step_clip = step["clip"].as<double>(0.075);
        } else if (loco["step_weight"]) {
            // Backward compatibility: flat structure (deprecated)
            mRewardConfig.step_weight = loco["step_weight"].as<double>(2.0);
        }

        // Parse avg_vel configuration (hierarchical or flat for backward compatibility)
        if (loco["avg_vel"]) {
            // New hierarchical structure
            auto avg_vel = loco["avg_vel"];
            if (avg_vel["weight"])
                mRewardConfig.avg_vel_weight = avg_vel["weight"].as<double>(6.0);
            if (avg_vel["window_mult"])
                mRewardConfig.avg_vel_window_mult = avg_vel["window_mult"].as<double>(1.0);
            if (avg_vel["clip"])
                mRewardConfig.avg_vel_clip = avg_vel["clip"].as<double>(-1.0);
            if (avg_vel["consider_x"]) {
                if (avg_vel["consider_x"].as<bool>(true))
                    mRewardConfig.flags |= REWARD_AVG_VEL_CONSIDER_X;
                else
                    mRewardConfig.flags &= ~REWARD_AVG_VEL_CONSIDER_X;
            }
        } else {
            // Backward compatibility: flat structure (deprecated)
            bool found_flat = false;
            if (loco["avg_vel_weight"]) {
                mRewardConfig.avg_vel_weight = loco["avg_vel_weight"].as<double>(6.0);
                found_flat = true;
            }
            if (loco["avg_vel_window_mult"]) {
                mRewardConfig.avg_vel_window_mult = loco["avg_vel_window_mult"].as<double>(1.0);
                found_flat = true;
            }
            if (loco["avg_vel_consider_x"]) {
                if (loco["avg_vel_consider_x"].as<bool>(true))
                    mRewardConfig.flags |= REWARD_AVG_VEL_CONSIDER_X;
                else
                    mRewardConfig.flags &= ~REWARD_AVG_VEL_CONSIDER_X;
                found_flat = true;
            }
            if (found_flat) {
                std::cout << "[WARNING] Flat avg_vel_* configuration is deprecated. "
                          << "Please use hierarchical avg_vel: { weight, window_mult, clip, consider_x } structure."
                          << std::endl;
            }
        }

        // Parse dragX configuration (hierarchical or flat for backward compatibility)
        if (loco["dragX"]) {
            // New hierarchical structure
            auto dragX = loco["dragX"];
            if (dragX["use"]) {
                if (dragX["use"].as<bool>(false))
                    mRewardConfig.flags |= REWARD_DRAG_X;
                else
                    mRewardConfig.flags &= ~REWARD_DRAG_X;
            }
            if (dragX["weight"])
                mRewardConfig.drag_weight = dragX["weight"].as<double>(1.0);
            if (dragX["threshold"])
                mRewardConfig.drag_x_threshold = dragX["threshold"].as<double>(0.0);
        } else {
            // Backward compatibility: flat structure (deprecated)
            bool found_flat = false;
            if (loco["drag_x"]) {
                if (loco["drag_x"].as<bool>(false))
                    mRewardConfig.flags |= REWARD_DRAG_X;
                else
                    mRewardConfig.flags &= ~REWARD_DRAG_X;
                found_flat = true;
            }
            if (loco["drag_weight"]) {
                mRewardConfig.drag_weight = loco["drag_weight"].as<double>(1.0);
                found_flat = true;
            }
            if (loco["drag_x_threshold"]) {
                mRewardConfig.drag_x_threshold = loco["drag_x_threshold"].as<double>(0.0);
                found_flat = true;
            }
            if (found_flat) {
                std::cout << "[WARNING] Flat drag_* configuration is deprecated. "
                          << "Please use hierarchical dragX: { use, weight, threshold } structure."
                          << std::endl;
            }
        }

        // Parse phase configuration
        if (loco["phase"]) {
            auto phase = loco["phase"];
            if (phase["use"]) {
                if (phase["use"].as<bool>(false))
                    mRewardConfig.flags |= REWARD_PHASE;
                else
                    mRewardConfig.flags &= ~REWARD_PHASE;
            }
            if (phase["weight"])
                mRewardConfig.phase_weight = phase["weight"].as<double>(1.0);
        }
    }

    // === Metabolic (always active) ===
    if (env["reward"] && env["reward"]["metabolic"]) {
        auto metabolic_config = env["reward"]["metabolic"];

        if (metabolic_config["weight"])
            mRewardConfig.metabolic_weight = metabolic_config["weight"].as<double>(0.05);
        if (metabolic_config["scale"])
            mRewardConfig.metabolic_scale = metabolic_config["scale"].as<double>(1.0);
        if (metabolic_config["multiplicative"] && metabolic_config["multiplicative"].as<bool>(false))
            mRewardConfig.flags |= REWARD_METABOLIC;

        if (metabolic_config["type"]) {
            std::string metaType = metabolic_config["type"].as<std::string>("A");
            if (metaType == "A") mCharacter->setMetabolicType(A);
            else if (metaType == "A2") mCharacter->setMetabolicType(A2);
            else if (metaType == "MA") mCharacter->setMetabolicType(MA);
            else if (metaType == "MA2") mCharacter->setMetabolicType(MA2);
        }

        // Torque energy coefficient (nested under metabolic)
        // Support both flat and nested structure for backward compatibility
        if (metabolic_config["torque_coeff"]) {
            double coeff = metabolic_config["torque_coeff"].as<double>(1.0);
            mCharacter->setTorqueEnergyCoeff(coeff);
        }
        if (metabolic_config["torque"]) {
            auto torque_config = metabolic_config["torque"];
            if (torque_config["coeff"]) {
                double coeff = torque_config["coeff"].as<double>(1.0);
                mCharacter->setTorqueEnergyCoeff(coeff);
            }
            if (torque_config["separate"] && torque_config["separate"].as<bool>(false)) mRewardConfig.flags |= REWARD_SEP_TORQUE_ENERGY;
        }
    }

    // === Knee pain ===
    if (env["reward"] && env["reward"]["knee_pain"]) {
        auto knee = env["reward"]["knee_pain"];
        if (knee["use"] && knee["use"].as<bool>(false)) mRewardConfig.flags |= REWARD_KNEE_PAIN;
        if (knee["termination"] && knee["termination"].as<bool>(false)) mRewardConfig.flags |= TERM_KNEE_PAIN;
        if (knee["use_max"] && knee["use_max"].as<bool>(false)) mRewardConfig.flags |= REWARD_KNEE_PAIN_MAX;
        if (knee["weight"]) mRewardConfig.knee_pain_weight = knee["weight"].as<double>(1.0);
        if (knee["scale"]) mRewardConfig.knee_pain_scale = knee["scale"].as<double>(1.0);
        if (knee["max_weight"]) mRewardConfig.knee_pain_max_weight = knee["max_weight"].as<double>(1.0);
    }

    // === Parameters (gait, skeleton, torsion) ===
    std::vector<double> minV;
    std::vector<double> maxV;
    std::vector<double> defaultV;

    if (env["parameters"])
    {
        auto params = env["parameters"];

        // === Parse gait parameters ===
        if (params["gait"])
        {
            auto gait = params["gait"];
            for (YAML::const_iterator it = gait.begin(); it != gait.end(); ++it)
            {
                std::string param_name = it->first.as<std::string>();
                auto param = it->second;

                if (!param["min"] || !param["max"])
                    continue;

                minV.push_back(param["min"].as<double>());
                maxV.push_back(param["max"].as<double>());
                defaultV.push_back(param["default"] ? param["default"].as<double>() : 1.0);

                std::string full_name = "gait_" + param_name;
                mParamName.push_back(full_name);

                // Create param_group
                param_group p;
                p.name = full_name;
                p.param_idxs.push_back(mParamName.size() - 1);
                p.param_names.push_back(full_name);
                double range = maxV.back() - minV.back();
                p.v = (std::abs(range) < 1e-9) ? 0.0 : (defaultV.back() - minV.back()) / range;
                p.is_uniform = (param["sampling"] && param["sampling"].as<std::string>() == "uniform");
                mParamGroups.push_back(p);
            }
        }

        // === Parse skeleton parameters ===
        if (params["skeleton"])
        {
            auto skeleton = params["skeleton"];
            for (YAML::const_iterator it = skeleton.begin(); it != skeleton.end(); ++it)
            {
                std::string bone_type = it->first.as<std::string>();
                auto bone = it->second;

                // Handle 2-level parameters (e.g., global)
                if (bone["min"] && bone["max"])
                {
                    minV.push_back(bone["min"].as<double>());
                    maxV.push_back(bone["max"].as<double>());
                    defaultV.push_back(bone["default"] ? bone["default"].as<double>() : 1.0);

                    std::string full_name = "skeleton_" + bone_type;
                    mParamName.push_back(full_name);

                    param_group p;
                    p.name = full_name;
                    p.param_idxs.push_back(mParamName.size() - 1);
                    p.param_names.push_back(full_name);
                    double range = maxV.back() - minV.back();
                    p.v = (std::abs(range) < 1e-9) ? 0.0 : (defaultV.back() - minV.back()) / range;
                    p.is_uniform = (bone["sampling"] && bone["sampling"].as<std::string>() == "uniform");
                    mParamGroups.push_back(p);
                }
                // Handle 3-level parameters (e.g., femur.left, femur.right)
                else
                {
                    for (YAML::const_iterator side_it = bone.begin(); side_it != bone.end(); ++side_it)
                    {
                        std::string side = side_it->first.as<std::string>();
                        auto side_param = side_it->second;

                        if (!side_param["min"] || !side_param["max"])
                            continue;

                        minV.push_back(side_param["min"].as<double>());
                        maxV.push_back(side_param["max"].as<double>());
                        defaultV.push_back(side_param["default"] ? side_param["default"].as<double>() : 1.0);

                        std::string full_name = "skeleton_" + bone_type + "_" + side;
                        mParamName.push_back(full_name);

                        param_group p;
                        p.name = full_name;
                        p.param_idxs.push_back(mParamName.size() - 1);
                        p.param_names.push_back(full_name);
                        double range = maxV.back() - minV.back();
                        p.v = (std::abs(range) < 1e-9) ? 0.0 : (defaultV.back() - minV.back()) / range;
                        p.is_uniform = (side_param["sampling"] && side_param["sampling"].as<std::string>() == "uniform");
                        mParamGroups.push_back(p);
                    }
                }
            }
        }

        // === Parse torsion parameters ===
        if (params["torsion"])
        {
            auto torsion = params["torsion"];
            for (YAML::const_iterator it = torsion.begin(); it != torsion.end(); ++it)
            {
                std::string param_name = it->first.as<std::string>();
                auto param = it->second;

                if (!param["min"] || !param["max"])
                    continue;

                minV.push_back(param["min"].as<double>());
                maxV.push_back(param["max"].as<double>());
                defaultV.push_back(param["default"] ? param["default"].as<double>() : 1.0);

                std::string full_name = "torsion_" + param_name;
                mParamName.push_back(full_name);

                param_group p;
                p.name = full_name;
                p.param_idxs.push_back(mParamName.size() - 1);
                p.param_names.push_back(full_name);
                double range = maxV.back() - minV.back();
                p.v = (std::abs(range) < 1e-9) ? 0.0 : (defaultV.back() - minV.back()) / range;
                p.is_uniform = (param["sampling"] && param["sampling"].as<std::string>() == "uniform");
                mParamGroups.push_back(p);
            }
        }

        // Convert vectors to Eigen
        mParamMin = Eigen::VectorXd::Zero(minV.size());
        mParamMax = Eigen::VectorXd::Zero(minV.size());
        mParamDefault = Eigen::VectorXd::Zero(defaultV.size());

        for (int i = 0; i < minV.size(); i++)
        {
            mParamMin[i] = minV[i];
            mParamMax[i] = maxV[i];
            mParamDefault[i] = defaultV[i];
        }

        mNumParamState = minV.size();
    }

    // === Initialize mUseWeights ===
    mUseWeights.clear();
    for (int i = 0; i < mPrevNetworks.size() + 1; i++) {
        mUseWeights.push_back(true);
        if (mUseMuscle)
            mUseWeights.push_back(true);
    }

    // === Set num known param ===
    mNumKnownParam = 0;
    for(int i = 0; i < mParamName.size(); i++) {
        if (mParamName[i].find("skeleton") != std::string::npos ||
            mParamName[i].find("stride") != std::string::npos ||
            mParamName[i].find("cadence") != std::string::npos ||
            mParamName[i].find("torsion") != std::string::npos)
            mNumKnownParam++;
    }

    // === Discriminator configuration ===
    if (env["discriminator"]) {
        auto disc = env["discriminator"];
        mDiscConfig.enabled = disc["enabled"].as<bool>(false);
        mDiscConfig.normalize = disc["normalize"].as<bool>(false);
        mDiscConfig.reward_scale = disc["reward_scale"].as<double>(1.0);
        mDiscConfig.multiplicative = disc["multiplicative"].as<bool>(false);

        // Parse upper_body config (supports both bool and nested map)
        if (disc["upper_body"]) {
            auto ub = disc["upper_body"];
            if (ub.IsMap()) {
                // Nested config: upper_body: { enabled: true, scale: 0.01 }
                mDiscConfig.upper_body = ub["enabled"].as<bool>(false);
                mDiscConfig.upper_body_scale = ub["scale"].as<double>(1.0);
            } else {
                // Simple bool: upper_body: true
                mDiscConfig.upper_body = ub.as<bool>(false);
            }
        }

        if (mDiscConfig.enabled && mUseMuscle) {
            // Cache upper body dimension if needed
            if (mDiscConfig.upper_body) {
                int rootDof = mCharacter->getSkeleton()->getRootJoint()->getNumDofs();
                int lowerBodyDof = 18;  // First 18 DOFs after root are lower body
                mUpperBodyDim = mCharacter->getSkeleton()->getNumDofs() - rootDof - lowerBodyDof;
            }

            // Create C++ DiscriminatorNN (libtorch) for thread-safe inference
            // Force CPU to avoid CUDA context allocation issues in multi-process scenarios
            int disc_dim = getDiscObsDim();
            mDiscriminatorNN = make_discriminator_nn(disc_dim, true);
            mRandomDiscObs = Eigen::VectorXf::Zero(disc_dim);
        }
    }

    // Initialize GaitPhase after all configuration is loaded
    GaitPhase::UpdateMode mode = (gaitUpdateMode == "contact") ? GaitPhase::CONTACT : GaitPhase::PHASE;
    mGaitPhase = std::make_unique<GaitPhase>(mCharacter, mWorld, mMotion->getMaxTime(), mRefStride, mode, mControlHz, mSimulationHz);
    mGaitPhase->setContactDebounceAlpha(contactDebounceAlpha);
    mGaitPhase->setStepMinRatio(stepMinRatio);
}

void Environment::setAction(Eigen::VectorXd _action)
{
    double phaseAction = 0.0;
    mAction.setZero();
    if (mAction.rows() != _action.rows())
    {
        std::cout << "[ERROR] Environment SetAction" << std::endl;
        exit(-1);
    }
    // Cascading
    if (mUseCascading)
    {
        mProjStates.clear();
        mProjJointStates.clear();
        for (Network nn : mPrevNetworks)
        {
            std::pair<Eigen::VectorXd, Eigen::VectorXd> prev_states = getProjState(nn.minV, nn.maxV);
            mProjStates.push_back(prev_states.first);
            mProjJointStates.push_back(prev_states.second);
        }
        mProjStates.push_back(mState);
        mProjJointStates.push_back(mJointState);

        mDmins.clear();
        mWeights.clear();
        mBetas.clear();

        for (int i = 0; i < mPrevNetworks.size() + 1; i++)
        {
            mDmins.push_back(99999999);
            mWeights.push_back(0.0);
            mBetas.push_back(0.0);
        }

        if (mPrevNetworks.size() > 0)
        {
            mDmins[0] = 0.0;
            mWeights[0] = 1.0;
            mBetas[0] = 0.0;
        }

        for (Eigen::Vector2i edge : mEdges)
        {
            double d = (mProjJointStates[edge[1]] - mProjJointStates[edge[0]]).norm() * 0.008;
            if (mDmins[edge[1]] > d)
                mDmins[edge[1]] = d;
        }

        for (int i = 0; i < mPrevNetworks.size(); i++)
        {
            Eigen::VectorXd prev_action = mPrevNetworks[i].joint.attr("get_action")(mProjStates[i]).cast<Eigen::VectorXd>();
            if (i == 0)
            {
                mAction.head(mNumActuatorAction) = mActionScale * (mUseWeights[i * (mUseMuscle ? 2 : 1)] ? 1 : 0) * prev_action.head(mNumActuatorAction);
                mAction.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction) += (mUseWeights[i * (mUseMuscle ? 2 : 1)] ? 1 : 0) * prev_action.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction);
                phaseAction += mPhaseDisplacementScale * prev_action[mNumActuatorAction];
                continue;
            }
            double beta = 0.2 + 0.1 * prev_action[prev_action.rows() - 1];
            mBetas[i] = beta;
            mWeights[i] = mPrevNetworks.front().joint.attr("weight_filter")(mDmins[i], beta).cast<double>();

            // Joint Anlge 부분은 add position 을 통해서
            mAction.head(mNumActuatorAction) = mCharacter->addPositions(mAction.head(mNumActuatorAction), (mUseWeights[i * (mUseMuscle ? 2 : 1)] ? 1 : 0) * mWeights[i] * mActionScale * prev_action.head(mNumActuatorAction), false); // mAction.head(mNumActuatorAction)
            mAction.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction) += (mUseWeights[i * (mUseMuscle ? 2 : 1)] ? 1 : 0) * mWeights[i] * prev_action.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction);
            phaseAction += mWeights[i] * mPhaseDisplacementScale * prev_action[mNumActuatorAction];
        }
        // Current Networks
        if (mLoadedMuscleNN)
        {
            double beta = 0.2 + 0.1 * _action[_action.rows() - 1];
            mBetas[mBetas.size() - 1] = beta;
            mWeights[mWeights.size() - 1] = mPrevNetworks.front().joint.attr("weight_filter")(mDmins.back(), beta).cast<double>();
            // mAction.head(mAction.rows() - 1) += (mUseWeights[mWeights.size() - 1] ? 1 : 0) * mWeights[mWeights.size() - 1] * _action.head(mAction.rows() - 1);
            mAction.head(mNumActuatorAction) = mCharacter->addPositions(mAction.head(mNumActuatorAction), (mUseWeights[mUseWeights.size() - (mUseMuscle ? 2 : 1)] ? 1 : 0) * mWeights.back() * mActionScale * _action.head(mNumActuatorAction), false); // mAction.head(mNumActuatorAction)
            mAction.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction) += (mUseWeights[mUseWeights.size() - (mUseMuscle ? 2 : 1)] ? 1 : 0) * mWeights.back() * _action.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction);
        }
    }
    else
    {
        mAction = _action;
        mAction.head(mNumActuatorAction) *= mActionScale;
    }
    
    if (mPhaseDisplacementScale > 0.0) phaseAction += (mWeights.size() > 0 ? mWeights.back() : 1.0) * mPhaseDisplacementScale * mAction[mNumActuatorAction];
    else phaseAction = 0.0;

    mGaitPhase->setPhaseAction(phaseAction);

    Eigen::VectorXd actuatorAction = mAction.head(mNumActuatorAction);

    updateTargetPosAndVel();

    if (mCharacter->getActuatorType() == pd || 
        mCharacter->getActuatorType() == mass || 
        mCharacter->getActuatorType() == mass_lower)
    {
        Eigen::VectorXd action = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
        action.tail(actuatorAction.rows()) = actuatorAction;
        if (isMirror()) action = mCharacter->getMirrorPosition(action);
        action = mCharacter->addPositions(mRefPose, action);
        mCharacter->setPDTarget(action);
    }
    else if (mCharacter->getActuatorType() == tor)
    {
        Eigen::VectorXd torque = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
        torque.tail(actuatorAction.rows()) = actuatorAction;
        if (isMirror()) torque = mCharacter->getMirrorPosition(torque);
        mCharacter->setTorque(torque);
    }
    else if (mCharacter->getActuatorType() == mus)
    {
        Eigen::VectorXd activation = (!isMirror() ? actuatorAction : mCharacter->getMirrorActivation(actuatorAction));
        // Clipping Function
        mCharacter->setActivations(activation);
    }

    mSimulationStep++;
}

void Environment::updateTargetPosAndVel(bool isInit)
{
    double dTime = 1.0 / mControlHz;
    double dPhase = dTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
    double ofsPhase = (isInit ? 0.0 : dPhase) + getLocalPhase();
    
    mRefPose = mMotion->getTargetPose(ofsPhase);
    const auto nextPose = mMotion->getTargetPose(ofsPhase + dPhase);
    mTargetVelocities = mCharacter->getSkeleton()->getPositionDifferences(nextPose, mRefPose) / dTime;
}

void Environment::checkTerminated()
{
    // Episode ends due to failure: fall or character below height limit
    double root_y = mCharacter->getSkeleton()->getCOM()[1];
    bool is_fall = root_y < mLimitY * mCharacter->getGlobalRatio();
    bool knee_pain = (mRewardConfig.flags & TERM_KNEE_PAIN) && (mSimulationStep > 100) && (mCharacter->getKneeLoadingMax() > 8.0);
    bool terminated = is_fall || knee_pain;

    // Log to mInfoMap for TensorBoard
    mInfoMap["terminated"] = terminated ? 1.0 : 0.0;
    mInfoMap["termination_fall"] = is_fall ? 1.0 : 0.0;
    mInfoMap["termination_knee_pain"] = knee_pain ? 1.0 : 0.0;
}

void Environment::checkTruncated()
{
    // Episode ends due to step count (EOEType is always tuple)
    bool step_limit = mSimulationStep >= mHorizon;
    bool truncated = step_limit;

    // Log to mInfoMap for TensorBoard
    mInfoMap["truncated"] = truncated ? 1.0 : 0.0;
    mInfoMap["truncation_time"] = 0.0;  // Always 0 since time_limit is never true
    mInfoMap["truncation_steps"] = step_limit ? 1.0 : 0.0;
}

double Environment::calcReward()
{
    double r = 0.0;
    if (mRewardType == deepmimic || mRewardType == scadiver)
    {
        // Deep Mimic Reward Setting
        double w_p = 0.65;
        double w_v = 0.1;
        double w_ee = 0.45;
        double w_com = 0.1;
        double w_metabolic = 0.2;

        auto skel = mCharacter->getSkeleton();
        Eigen::VectorXd pos = skel->getPositions();
        Eigen::VectorXd vel = skel->getVelocities();

        Eigen::VectorXd pos_diff = skel->getPositionDifferences(mRefPose, pos);
        Eigen::VectorXd vel_diff = skel->getVelocityDifferences(mTargetVelocities, vel);

        auto ees = mCharacter->getEndEffectors();
        Eigen::VectorXd ee_diff(ees.size() * 3);
        Eigen::Vector3d com_diff;
        for (int i = 0; i < ees.size(); i++)
        {
            auto ee = ees[i];
            ee_diff.segment(i * 3, 3) = -ee->getCOM(skel->getRootBodyNode());
        }
        com_diff = -skel->getCOM();
        skel->setPositions(mRefPose);
        for (int i = 0; i < ees.size(); i++)
        {
            auto ee = ees[i];
            ee_diff.segment(i * 3, 3) += ee->getCOM(skel->getRootBodyNode());
        }
        com_diff += skel->getCOM();
        skel->setPositions(pos);

        double r_p, r_v, r_ee, r_com, r_metabolic;
        r_ee = exp(-40 * ee_diff.squaredNorm() / ee_diff.rows());
        r_p = exp(-20 * pos_diff.squaredNorm() / pos_diff.rows());
        r_v = exp(-10 * vel_diff.squaredNorm() / vel_diff.rows());
        r_com = exp(-10 * com_diff.squaredNorm() / com_diff.rows());
        r_metabolic = 0.0;

        r_metabolic = getEnergyReward();

        if (mRewardType == deepmimic) r = w_p * r_p + w_v * r_v + w_com * r_com + w_ee * r_ee + w_metabolic * r_metabolic;
        else if (mRewardType == scadiver) r = (0.1 + 0.9 * r_p) * (0.1 + 0.9 * r_v) * (0.1 + 0.9 * r_com) * (0.1 + 0.9 * r_ee) * (0.1 + 0.9 * r_metabolic);
    }
    else if (mRewardType == gaitnet)
    {
        double r_head_linear_acc = getHeadLinearAccReward();
        double r_head_rot_diff = getHeadRotReward();
        double r_loco = r_head_linear_acc * r_head_rot_diff;
        double r_avg = getAvgVelReward();
        double r_step = getStepReward();
        double r_drag_x = getDragXReward();
        double r_phase = getPhaseReward();

        // Build multiplicative and additive components separately using bitflags
        double multiplicative_part = r_loco * r_avg * r_step * r_drag_x * r_phase;
        double additive_part = 0.0;

        // Apply energy reward (always active, multiplicative or additive)
        double r_energy = getEnergyReward();
        
        // When torque energy is separated, calculate and apply separate torque reward
        if (mRewardConfig.flags & REWARD_SEP_TORQUE_ENERGY) {
            mInfoMap.insert(std::make_pair("r_metabolic", r_energy));

            double r_torque = exp(-mRewardConfig.metabolic_weight * mCharacter->getTorqueEnergy());
            if (mRewardConfig.flags & REWARD_METABOLIC) r_energy *= r_torque;
            else r_energy +=  r_torque;
            mInfoMap.insert(std::make_pair("r_torque", r_torque));
        }

        if ((mRewardConfig.flags & REWARD_METABOLIC) && mRewardConfig.metabolic_scale > 0.001) multiplicative_part *= r_energy;
        else additive_part += mRewardConfig.metabolic_scale * r_energy;
        mInfoMap.insert(std::make_pair("r_energy", r_energy));

        // Apply knee pain reward (multiplicative or additive based on bitflag)
        if (mRewardConfig.flags & REWARD_KNEE_PAIN)
        {
            double r_knee_pain = getKneePainReward();
            multiplicative_part *= r_knee_pain;
            mInfoMap.insert(std::make_pair("r_knee_pain", r_knee_pain));

        }
        // Apply knee pain max reward (per gait cycle) if enabled
        if (mRewardConfig.flags & REWARD_KNEE_PAIN_MAX)
        {
            double r_knee_pain_max = getKneePainMaxReward();
            multiplicative_part *= r_knee_pain_max;
            mInfoMap.insert(std::make_pair("r_knee_pain_max", r_knee_pain_max));
        }

        // Apply discriminator reward (ADD-style, multiplicative or additive based on config)
        // Uses accumulated reward across all substeps, averaged
        if (mDiscConfig.enabled && mDiscriminatorNN)
        {
            double r_disc = mDiscRewardAccum / static_cast<double>(mNumSubSteps);
            if (mDiscConfig.multiplicative) {
                // Multiplicative: no scaling, direct multiplication with main reward
                multiplicative_part *= r_disc;
            } else {
                // Additive: apply reward_scale
                additive_part += mDiscConfig.reward_scale * r_disc;
            }
            mInfoMap.insert(std::make_pair("r_disc", r_disc));
        }

        r = multiplicative_part + additive_part;

        // Populate reward map for gaitnet
        mInfoMap.insert(std::make_pair("r_head_linear_acc", r_head_linear_acc));
        mInfoMap.insert(std::make_pair("r_head_rot_diff", r_head_rot_diff));
        mInfoMap.insert(std::make_pair("r_loco", r_loco));
        mInfoMap.insert(std::make_pair("r_avg", r_avg));
        mInfoMap.insert(std::make_pair("r_step", r_step));
        mInfoMap.insert(std::make_pair("r_drag_x", r_drag_x));
        mInfoMap.insert(std::make_pair("r_phase", r_phase));
    }
    if (mCharacter->getActuatorType() == mus) r = 1.0;

    if (mRewardConfig.clip_step > 0 && mSimulationStep < mRewardConfig.clip_step) {
        double clip_bound = mRewardConfig.clip_value;
        if (clip_bound == 0.0) {
            r = 0.0;
        } else {
            if (r > clip_bound) r = clip_bound;
            else if (r < -clip_bound) r = -clip_bound;
        }
    }

    // Always store total reward
    mInfoMap.insert(std::make_pair("r", r));

    return r;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Environment::getProjState(const Eigen::VectorXd minV, const Eigen::VectorXd maxV)
{
    if (minV.rows() != maxV.rows())
        exit(-1);

    Eigen::VectorXd curParamState = getParamState();
    Eigen::VectorXd projState = Eigen::VectorXd::Zero(mNumParamState);

    for (int i = 0; i < projState.rows(); i++)
        projState[i] = dart::math::clip(curParamState[i], minV[i], maxV[i]);

    std::vector<int> projectedParamIdx;
    for (int i = 0; i < minV.rows(); i++)
        if (abs(minV[i] - maxV[i]) > 1E-3)
            projectedParamIdx.push_back(i);

    Eigen::VectorXd p, v;
    auto skel = mCharacter->getSkeleton();
    Eigen::Vector3d com = skel->getCOM();


    if (mRewardType == gaitnet)
    {
        com[0] = 0;
        com[2] = 0;
    }
    int num_body_nodes = skel->getNumBodyNodes();

    p.resize(num_body_nodes * 3 + num_body_nodes * 6);
    v.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);

    p.setZero();
    v.setZero();

    if (!isMirror())
    {
        for (int i = 0; i < num_body_nodes; i++)
        {
            p.segment<3>(i * 3) = skel->getBodyNode(i)->getCOM() - skel->getCOM();
            Eigen::Isometry3d transform = skel->getBodyNode(i)->getTransform();
            p.segment<6>(num_body_nodes * 3 + 6 * i) << transform.linear()(0, 0), transform.linear()(0, 1), transform.linear()(0, 2),
                transform.linear()(1, 0), transform.linear()(1, 1), transform.linear()(1, 2);
            v.segment<3>(i * 3) = skel->getBodyNode(i)->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
            v.segment<3>((num_body_nodes + 1) * 3 + i * 3) = 0.1 * skel->getBodyNode(i)->getAngularVelocity();
        }
        v.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();
    }
    else
    {
        int idx = 0;
        std::vector<Eigen::Matrix3d> body_node_transforms = mCharacter->getBodyNodeTransform();
        for (auto j_pair : mCharacter->getPairs())
        {
            int first_idx = j_pair.first->getChildBodyNode()->getIndexInSkeleton();
            int second_idx = j_pair.second->getChildBodyNode()->getIndexInSkeleton();

            Eigen::Vector3d first_pos = j_pair.second->getChildBodyNode()->getCOM() - skel->getCOM();
            first_pos[0] *= -1;
            Eigen::Vector3d second_pos = j_pair.first->getChildBodyNode()->getCOM() - skel->getCOM();
            second_pos[0] *= -1;

            Eigen::AngleAxisd first_rot = Eigen::AngleAxisd(j_pair.second->getChildBodyNode()->getTransform().linear());
            first_rot.axis() = Eigen::Vector3d(first_rot.axis()[0], -first_rot.axis()[1], -first_rot.axis()[2]);

            Eigen::AngleAxisd second_rot = Eigen::AngleAxisd(j_pair.first->getChildBodyNode()->getTransform().linear());
            second_rot.axis() = Eigen::Vector3d(second_rot.axis()[0], -second_rot.axis()[1], -second_rot.axis()[2]);

            Eigen::Matrix3d first_rot_mat = first_rot.toRotationMatrix() * body_node_transforms[idx].transpose();
            Eigen::Matrix3d second_rot_mat = second_rot.toRotationMatrix() * body_node_transforms[idx];

            p.segment<3>(first_idx * 3) = first_pos;
            p.segment<3>(second_idx * 3) = second_pos;

            p.segment<6>(num_body_nodes * 3 + first_idx * 6) << first_rot_mat(0, 0), first_rot_mat(0, 1), first_rot_mat(0, 2), first_rot_mat(1, 0), first_rot_mat(1, 1), first_rot_mat(1, 2);
            p.segment<6>(num_body_nodes * 3 + second_idx * 6) << second_rot_mat(0, 0), second_rot_mat(0, 1), second_rot_mat(0, 2), second_rot_mat(1, 0), second_rot_mat(1, 1), second_rot_mat(1, 2);

            Eigen::Vector3d first_vel = j_pair.second->getChildBodyNode()->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
            first_vel[0] *= -1;

            Eigen::Vector3d second_vel = j_pair.first->getChildBodyNode()->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
            second_vel[0] *= -1;

            v.segment<3>(first_idx * 3) = first_vel;
            v.segment<3>(second_idx * 3) = second_vel;

            Eigen::Vector3d first_ang = 0.1 * j_pair.second->getChildBodyNode()->getAngularVelocity();
            first_ang[1] *= -1;
            first_ang[2] *= -1;
            v.segment<3>((num_body_nodes + 1) * 3 + first_idx * 3) = first_ang;

            Eigen::Vector3d second_ang = 0.1 * j_pair.first->getChildBodyNode()->getAngularVelocity();
            second_ang[1] *= -1;
            second_ang[2] *= -1;
            v.segment<3>((num_body_nodes + 1) * 3 + second_idx * 3) = second_ang;
            idx++;
        }
        v.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();
        v.segment<3>(num_body_nodes * 3)[0] *= -1;
    }

    // Motion information (phase)

    Eigen::VectorXd phase = Eigen::VectorXd::Zero(1 + (mPhaseDisplacementScale > 0.0 ? 1 : 0));
    phase[0] = getNormalizedPhase();
   

    if (mPhaseDisplacementScale > 0.0)
        phase[1] = getLocalPhase(true);

    if (isMirror())
        for (int i = 0; i < phase.rows(); i++)
            phase[i] = (phase[i] + 0.5) - (int)(phase[i] + 0.5);

    // Gait Information (Step)
    Eigen::VectorXd step_state = Eigen::VectorXd::Zero(0);

    if (mRewardType == gaitnet)
    {
        step_state.resize(1);
        step_state[0] = getNextTargetFootStep()[2] - mCharacter->getSkeleton()->getCOM()[2];
    }

    // Muscle State
    setParamState(projState, true);

    Eigen::VectorXd joint_state = Eigen::VectorXd::Zero(0);

    if (mUseJointState)
        joint_state = getJointState(isMirror());

    // Parameter State
    Eigen::VectorXd param_state = (mUseNormalizedParamState ? getNormalizedParamState(minV, maxV, isMirror()) : getParamState(isMirror()));
    Eigen::VectorXd proj_param_state = Eigen::VectorXd::Zero(projectedParamIdx.size());
    for (int i = 0; i < projectedParamIdx.size(); i++)
        proj_param_state[i] = param_state[projectedParamIdx[i]];

    setParamState(curParamState, true);

    // Integration of all states

    Eigen::VectorXd state = Eigen::VectorXd::Zero(com.rows() + p.rows() + v.rows() + phase.rows() + step_state.rows() + joint_state.rows() + proj_param_state.rows());
    state << com, p, v, phase, step_state, 0.008 * joint_state, proj_param_state;

    // ============================
    // Integration with Foot Step
    // Eigen::VectorXd state;
    // if (mRewardType == deepmimic)
    // {
    //     state = Eigen::VectorXd::Zero(com.rows() + p.rows() + v.rows() + phase.rows());
    //     state << com, p, v, phase;
    // }
    // else if (mRewardType == gaitnet)
    // {
    //     Eigen::VectorXd d = Eigen::VectorXd::Zero(1);
    //     d[0] = mNextTargetFoot[2] - mCharacter->getSkeleton()->getCOM()[2];
    //     state = Eigen::VectorXd::Zero(com.rows() + p.rows() + v.rows() + phase.rows() + 1);
    //     state << com, p, v, phase, d;
    // }
    return std::make_pair(state, joint_state);
}

Eigen::VectorXd Environment::getState()
{
    std::pair<Eigen::VectorXd, Eigen::VectorXd> res = getProjState(mParamMin, mParamMax);
    mState = res.first;
    mJointState = res.second;
    return mState;
}

int Environment::getDiscObsDim() const
{
    int dim = mCharacter->getNumMuscles();
    if (mDiscConfig.upper_body) {
        dim += mUpperBodyDim;
    }
    return dim;
}

Eigen::VectorXf Environment::getDiscObs() const
{
    Eigen::VectorXf disc_obs(getDiscObsDim());
    Eigen::VectorXf activations = mCharacter->getActivations().cast<float>();

    if (mDiscConfig.upper_body) {
        // Concatenate: [activations, scaled_upperBodyTorque]
        Eigen::VectorXf upperTorque = mCharacter->getUpperBodyTorque()
            .tail(mUpperBodyDim).cast<float>();
        // Apply scale to upper body torques (they can be much larger than activations)
        upperTorque *= static_cast<float>(mDiscConfig.upper_body_scale);
        disc_obs.head(activations.size()) = activations;
        disc_obs.tail(mUpperBodyDim) = upperTorque;
    } else {
        disc_obs = activations;
    }

    return disc_obs;
}

void Environment::calcActivation()
{
    MuscleTuple mt = mCharacter->getMuscleTuple(isMirror());

    Eigen::VectorXd fullJtp = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
    if (mCharacter->getIncludeJtPinSPD()) fullJtp.tail(fullJtp.rows() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs()) = mt.JtP;
    if (isMirror()) fullJtp = mCharacter->getMirrorPosition(fullJtp);

    Eigen::VectorXd fulldt = mCharacter->getSPDForces(mCharacter->getPDTarget(), fullJtp);
    mDesiredTorqueLogs.push_back(fulldt);

    if (isMirror()) fulldt = mCharacter->getMirrorPosition(fulldt);
    Eigen::VectorXd dt = fulldt.tail(mt.JtP.rows());
    if (!mCharacter->getIncludeJtPinSPD()) dt -= mt.JtP;

    std::vector<Eigen::VectorXf> prev_activations;

    for (int j = 0; j < mPrevNetworks.size() + 1; j++) // Include Current Network
        prev_activations.push_back(Eigen::VectorXf::Zero(mCharacter->getMuscles().size()));

    // For base network
    if (mPrevNetworks.size() > 0) {
        prev_activations[0] = mPrevNetworks[0].muscle->unnormalized_no_grad_forward(mt.JtA_reduced, dt, nullptr, 1.0);
    }

    for (int j = 1; j < mPrevNetworks.size(); j++)
    {
        Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());
        for (int k : mChildNetworks[j]) prev_activation += prev_activations[k];
        prev_activations[j] = (mUseWeights[j * 2 + 1] ? 1 : 0) * mWeights[j] * mPrevNetworks[j].muscle->unnormalized_no_grad_forward(mt.JtA_reduced, dt, &prev_activation, mWeights[j]);
    }
    // Current Network
    if (mLoadedMuscleNN)
    {
        Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());

        if (!mChildNetworks.empty()) {
            for (int k : mChildNetworks.back()) prev_activation += prev_activations[k];
        }

        if (mPrevNetworks.size() > 0) {
            prev_activations[prev_activations.size() - 1] = (mUseWeights.back() ? 1 : 0) * mWeights.back() * mMuscleNN->unnormalized_no_grad_forward(mt.JtA_reduced, dt, &prev_activation, mWeights.back());
        } else {
            prev_activations[prev_activations.size() - 1] = mMuscleNN->unnormalized_no_grad_forward(mt.JtA_reduced, dt, nullptr, 1.0);
        }
    }

    Eigen::VectorXf activations = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());
    for (Eigen::VectorXf a : prev_activations) activations += a;

    activations = mMuscleNN->forward_filter(activations);

    if (isMirror()) activations = mCharacter->getMirrorActivation(activations.cast<double>()).cast<float>();

    mCharacter->setActivations(activations.cast<double>());

    if (thread_safe_uniform(0.0, 1.0) < 1.0 / static_cast<double>(mNumSubSteps) || !mTupleFilled)
    {
        mRandomMuscleTuple = mt;
        mRandomDesiredTorque = dt;
        if (mUseCascading)
        {
            Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());
            for (int k : mChildNetworks.back())
                prev_activation += prev_activations[k];
            mRandomPrevOut = prev_activation.cast<double>();
            mRandomWeight = mWeights.back();
        }
        mTupleFilled = true;
    }

    // Accumulate mean activation every substep (always, for tensorboard logging)
    mMeanActivation += activations.cwiseAbs().mean();

    // Note: Discriminator reward computation moved to postMuscleStep()
    // where full disc_obs (including upper body torque) is available
}

void Environment::postMuscleStep()
{
    mSimulationCount++;
    mGlobalTime += 1.0 / mSimulationHz;
    mWorldTime += 1.0 / mSimulationHz;

    // Update gait phase tracking (also updates local time using phase displacement set in setAction)
    mGaitPhase->step();

    // Check for gait cycle completion (muscle-step level check)
    if (mGaitPhase->isGaitCycleComplete()) {
        mWorldPhaseCount++;
        mWorldTime = mGaitPhase->getLocalTime();

        // Store the maximum knee loading from the completed gait cycle
        mKneeLoadingMaxCycle = mCharacter->getKneeLoadingMax();

        // Reset the character's knee loading max for the new cycle
        mCharacter->resetKneeLoadingMax();

        // Note: Flag will persist for PD-level check, don't clear here
    }

    // Hard phase clipping (always enabled)
    int currentGlobalCount = mGlobalTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
    int currentLocalCount = mGaitPhase->getLocalTime() / ((mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))));

    if (currentGlobalCount > currentLocalCount) mGaitPhase->setLocalTime(mGlobalTime);
    else if (currentGlobalCount < currentLocalCount) mGaitPhase->setLocalTime(currentLocalCount * ((mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))));

    // Discriminator processing after character step (when upperBodyTorque is computed)
    if (mDiscConfig.enabled) {
        // Get full disc_obs (activations + optional upper body torque)
        Eigen::VectorXf disc_obs = getDiscObs();

        // Accumulate discriminator reward every substep
        if (mDiscriminatorNN) {
            mDiscRewardAccum += mDiscriminatorNN->compute_reward(disc_obs);
        }

        // Sample disc_obs once per control step (for training data)
        if (thread_safe_uniform(0.0, 1.0) < 1.0 / static_cast<double>(mNumSubSteps) || !mDiscObsFilled) {
            mRandomDiscObs = disc_obs;
            mDiscObsFilled = true;
        }
    }
}

void Environment::muscleStep()
{
    if (mCharacter->getActuatorType() == mass || mCharacter->getActuatorType() == mass_lower) {
        calcActivation();
    }

    if (mNoiseInjector) mNoiseInjector->step(mCharacter);
    mCharacter->step();
    mWorld->step();
    postMuscleStep();
}

void Environment::preStep()
{
    // Clear PD-level step completion flag at the beginning of each PD step
    mGaitPhase->clearStepComplete();

    // Reset mean activation accumulator (always, for tensorboard logging)
    mMeanActivation = 0.0;

    // Reset discriminator accumulators
    if (mDiscConfig.enabled) {
        mDiscRewardAccum = 0.0;
    }
}

void Environment::step()
{
    preStep();
    for (int i = 0; i < mNumSubSteps; i++) muscleStep();
    postStep();
}

void Environment::postStep()
{
    mInfoMap.clear();
    mCharacter->evalStep();
    mKneeLoadingMaxCycle = std::max(mKneeLoadingMaxCycle, mCharacter->getKneeLoadingMax());
    mReward = calcReward();

    // Add mean_activation to info map for tensorboard logging (always enabled)
    // Average over all substeps
    mInfoMap["mean_activation"] = mMeanActivation / static_cast<double>(mNumSubSteps);

    // Reset disc_obs filled flag for next control step
    mDiscObsFilled = false;

    // Check and cache termination/truncation status
    checkTerminated();
    checkTruncated();
}

void Environment::poseOptimization(int iter)
{
    if (!mUseMuscle) return;
    auto skel = mCharacter->getSkeleton();

    double step_size = 1E-4;
    double threshold = 100.0;
    int i = 0;
    for (i = 0; i < iter; i++)
    {
        MuscleTuple mt = mCharacter->getMuscleTuple(false);
        Eigen::VectorXd dp = Eigen::VectorXd::Zero(skel->getNumDofs());
        dp.tail(mt.JtP.rows()) = mt.JtP;
        bool isDone = true;
        for (int j = 0; j < dp.rows(); j++)
            if (std::abs(dp[j]) > threshold)
            {
                // std::cout << dp.transpose() << std::endl;
                isDone = false;
                break;
            }

        if (isDone)
            break;
        // Right Leg
        dp[8] *= 0.1;
        dp[11] *= 0.25;
        dp[12] *= 0.25;

        // Left Leg
        dp[17] *= 0.1;
        dp[20] *= 0.25;
        dp[21] *= 0.25;

        dp *= step_size;
        skel->setPositions(skel->getPositions() + dp);
    }

    double phase = getLocalPhase(true);
    // Note: mIsLeftLegStance removed - use GaitPhase instead if needed
    // For pose optimization, we just need to set the phase state
    bool isLeftLegStance = !((0.33 < phase) && (phase <= 0.83));

    // Stance Leg Hip anlge Change
    double angle_threshold = 1;
    auto femur_joint = skel->getJoint((isLeftLegStance ? "FemurL" : "FemurR"));
    auto foot_bn = skel->getBodyNode((isLeftLegStance ? "TalusL" : "TalusR"));
    Eigen::VectorXd prev_angle = femur_joint->getPositions();
    Eigen::VectorXd cur_angle = femur_joint->getPositions();
    
    Eigen::VectorXd initial_JtP = mCharacter->getMuscleTuple(false).JtP;

    while (true)
    {
        prev_angle = cur_angle;
        Eigen::Vector3d root_com = skel->getRootBodyNode()->getCOM();
        Eigen::Vector3d foot_com = foot_bn->getCOM() - root_com;
        Eigen::Vector3d target_com = skel->getBodyNode("Head")->getCOM() - root_com;
        target_com[1] *= -1;
        target_com[2] *= -1;

        double angle_diff = atan2(target_com[1], target_com[2]) - atan2(foot_com[1], foot_com[2]);
        // std::cout << "Angle Diff " << angle_diff << std::endl;
        if (abs(angle_diff) < M_PI * 10 / 180.0) break;

        double step = (angle_diff > 0 ? -1.0 : 1.0) * M_PI / 180.0;
        cur_angle[0] += step;
        femur_joint->setPositions(cur_angle);
        Eigen::VectorXd current_Jtp = mCharacter->getMuscleTuple(false).JtP;
        bool isDone = false;
        for (int i = 0; i < current_Jtp.rows(); i++)
        {
            if (abs(current_Jtp[i]) > abs(initial_JtP[i]) + 1)
            {
                // std::cout << i << "-th Joint " << abs(current_Jtp[i]) - abs(initial_JtP[i]) << std::endl;
                femur_joint->setPositions(prev_angle);
                isDone = true;
                break;
            }
        }
        if (isDone) break;
    }

    // Rotation Change
    Eigen::Vector3d com = skel->getCOM(skel->getRootBodyNode());
    Eigen::Vector3d foot;
    if (mPoseOptimizationMode == 0)
        foot = skel->getBodyNode(isLeftLegStance ? "TalusL" : "TalusR")->getCOM(skel->getRootBodyNode());
    else if (mPoseOptimizationMode == 1)
        foot = (skel->getBodyNode("TalusL")->getCOM(skel->getRootBodyNode()) + skel->getBodyNode("TalusR")->getCOM(skel->getRootBodyNode())) * 0.5;
    // is it stance boundary?
    double global_diff = (skel->getCOM() - skel->getBodyNode(isLeftLegStance ? "TalusL" : "TalusR")->getCOM())[2];
    if (-0.07 < global_diff && global_diff < 0.1)
        return;

    // Remove X Components;
    com[0] = 0.0;
    foot[0] = 0.0;

    Eigen::Vector3d character_y = (com - foot).normalized();
    Eigen::Vector3d unit_y = Eigen::Vector3d::UnitY();

    double sin = character_y.cross(unit_y).norm();
    double cos = character_y.dot(unit_y);

    Eigen::VectorXd axis = character_y.cross(unit_y).normalized();
    double angle = atan2(sin, cos);

    Eigen::Matrix3d rot = Eigen::AngleAxisd(angle, axis).toRotationMatrix();

    Eigen::Isometry3d rootTransform = FreeJoint::convertToTransform(skel->getPositions().head(6));
    rootTransform.linear() = rot * rootTransform.linear();
    skel->getRootJoint()->setPositions(FreeJoint::convertToPositions(rootTransform));
}

void Environment::reset(double phase)
{
    // Clear info map (includes termination/truncation status)
    mInfoMap.clear();

    mTupleFilled = false;
    mDiscObsFilled = false;
    mMeanActivation = 0.0;
    mSimulationStep = 0;
    mPhaseCount = 0;
    mWorldPhaseCount = 0;
    mSimulationCount = 0;
    mKneeLoadingMaxCycle = 0.0;

    // Reset Initial Time
    double time = 0.0;
    if (phase >= 0.0 && phase <= 1.0) {
        // Use specified phase (0.0 to 1.0)
        time = phase * (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
    }
    else if (mRewardType == deepmimic) {
        time = thread_safe_uniform(1E-2, mMotion->getMaxTime() - 1E-2);
    }
    else if (mRewardType == gaitnet)
    {
        time = (thread_safe_uniform(0.0, 1.0) > 0.5 ? 0.5 : 0.0) + mStanceOffset + thread_safe_uniform(-0.05, 0.05);
        time *= (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
    }    
    
    // Collision Detector Reset
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->getConstraintSolver()->clearLastCollisionResult();

    mGlobalTime = time;
    mWorldTime = time;
    mWorld->setTime(time);

    // Reset Skeletons
    mCharacter->getSkeleton()->setPositions(mCharacter->getSkeleton()->getPositions().setZero());
    mCharacter->getSkeleton()->setVelocities(mCharacter->getSkeleton()->getVelocities().setZero());

    mCharacter->getSkeleton()->clearConstraintImpulses();
    mCharacter->getSkeleton()->clearInternalForces();
    mCharacter->getSkeleton()->clearExternalForces();

    mGaitPhase->setLocalTime(time);
    mGaitPhase->setPhaseAction(0.0);

    // Initial Pose Setting
    updateTargetPosAndVel(true);
    
    if(mRewardType == gaitnet)
    {
        // mTargetPositions.segment(6, 18) *= (mStride * (mCharacter->getGlobalRatio()));
        mTargetVelocities.head(24) *= (mStride * (mCharacter->getGlobalRatio()));
    }
    
    mCharacter->getSkeleton()->setPositions(mRefPose);
    mCharacter->getSkeleton()->setVelocities(mTargetVelocities);

    updateTargetPosAndVel();

    // if (mMusclePoseOptimization) poseOptimization();
    if (mRewardType == gaitnet)
    {
        Eigen::Vector3d ref_initial_vel = mTargetVelocities.segment(3, 3);
        ref_initial_vel = 
            FreeJoint::convertToTransform(mCharacter->getSkeleton()->getRootJoint()->getPositions()).linear().transpose() * 
            (FreeJoint::convertToTransform(mRefPose.head(6)).linear() * ref_initial_vel);
        Eigen::Vector6d vel = mCharacter->getSkeleton()->getRootJoint()->getVelocities();
        vel.segment(3, 3) = ref_initial_vel;
        mCharacter->getSkeleton()->getRootJoint()->setVelocities(vel);
    }
    
    // Height / Pose Optimization (always strict)
    mCharacter->heightCalibration(mWorld);

    // Pose In ROM
    Eigen::VectorXd cur_pos = mCharacter->getSkeleton()->getPositions();
    Eigen::VectorXd rom_min = mCharacter->getSkeleton()->getPositionLowerLimits();
    Eigen::VectorXd rom_max = mCharacter->getSkeleton()->getPositionUpperLimits();
    cur_pos = cur_pos.cwiseMax(rom_min).cwiseMin(rom_max);
    mCharacter->getSkeleton()->setPositions(cur_pos);

    mCharacter->setPDTarget(mRefPose);
    mCharacter->setTorque(mCharacter->getTorque().setZero());
    if (mUseMuscle) {
        mCharacter->setActivations(mCharacter->getActivations().setZero());
        mCharacter->resetStep();
    }

    mCharacter->clearLogs();
    mDesiredTorqueLogs.clear();

    mDragStartX = mCharacter->getSkeleton()->getCOM()[0];

    // Reset GaitPhase (already initialized in parseEnvConfig)
    mGaitPhase->reset();

    mCharacter->getSkeleton()->clearInternalForces();
    mCharacter->getSkeleton()->clearExternalForces();
    mCharacter->getSkeleton()->clearConstraintImpulses();
}

// Check whether the character falls or not
bool Environment::isFall()
{
    const auto results = mWorld->getConstraintSolver()->getLastCollisionResult();
    bool is_fall = false;
    for (int i = 0; i < results.getNumContacts(); i++)
    {

        const auto &c = results.getContact(i);

        if (c.collisionObject1->getShapeFrame()->getName().find("ground") != std::string::npos ||
            c.collisionObject2->getShapeFrame()->getName().find("ground") != std::string::npos)
        {
            if (c.collisionObject1->getShapeFrame()->getName().find("Foot") == std::string::npos &&
                c.collisionObject1->getShapeFrame()->getName().find("Talus") == std::string::npos &&

                c.collisionObject2->getShapeFrame()->getName().find("Foot") == std::string::npos &&
                c.collisionObject2->getShapeFrame()->getName().find("Talus") == std::string::npos

            )
                is_fall = true;
        }
    }

    return is_fall;
}

double Environment::getEnergyReward()
{
    double energy;
    if (mRewardConfig.flags & REWARD_SEP_TORQUE_ENERGY) energy = mCharacter->getMetabolicEnergy();
    else energy = mCharacter->getEnergy();
    double r_energy = exp(-mRewardConfig.metabolic_weight * energy);
    return r_energy;
}

double Environment::getKneePainReward()
{
    // Use accumulated averaged knee loading instead of instantaneous
    double avg_knee_loading = mCharacter->getKneeLoading();
    double r_knee = exp(-mRewardConfig.knee_pain_weight * avg_knee_loading);
    // std::cout << "knee_loading: " << avg_knee_loading << " r_knee: " << r_knee << " weight: " << mRewardConfig.knee_pain_weight << std::endl;
    return r_knee;
}

double Environment::getKneePainMaxReward()
{
    // Use maximum knee loading across the gait cycle
    double max_knee_loading = mKneeLoadingMaxCycle;
    double r_knee_max = exp(-mRewardConfig.knee_pain_max_weight * max_knee_loading);
    return r_knee_max;
}

double Environment::getHeadLinearAccReward()
{
    const std::vector<Eigen::Vector3d>& headVels = mCharacter->getHeadVelLogs();
    if (mNumSubSteps <= 0 || headVels.size() < static_cast<size_t>(mNumSubSteps)) return 1.0;

    const Eigen::Vector3d& currentVel = headVels.back();
    const Eigen::Vector3d& previousVel = headVels[headVels.size() - static_cast<size_t>(mNumSubSteps)];
    Eigen::Vector3d headLinearAcc = currentVel - previousVel;

    return exp(-mRewardConfig.head_linear_acc_weight * headLinearAcc.squaredNorm() / headLinearAcc.rows());
}

double Environment::getHeadRotReward()
{
    auto headNode = mCharacter->getSkeleton()->getBodyNode("Head");
    if (!headNode) return 1.0;

    double headRotDiff = Eigen::AngleAxisd(headNode->getTransform().linear()).angle();
    return exp(-mRewardConfig.head_rot_weight * headRotDiff * headRotDiff);
}

double Environment::getDragXReward()
{
    if (!(mRewardConfig.flags & REWARD_DRAG_X) || mSimulationStep < getAvgVelocityHorizonSteps()) return 1.0;

    double currentX = mCharacter->getSkeleton()->getCOM()[0];
    double diff = std::max(0.0, std::abs(currentX - mDragStartX) - mRewardConfig.drag_x_threshold);
    if (diff < mRewardConfig.drag_x_threshold) return 1.0;
    double r = exp(-mRewardConfig.drag_weight * diff);
    return r;
}

double Environment::getPhaseReward()
{
    if (!(mRewardConfig.flags & REWARD_PHASE)) return 1.0;

    // Get phase total times for both feet from GaitPhase
    double phaseTotalL = mGaitPhase->getPhaseTotalL();
    double phaseTotalR = mGaitPhase->getPhaseTotalR();

    // Get reference cycle time from motion
    double motionMaxTime = mGaitPhase->getMotionCycleTime();

    // Calculate phase errors for both feet
    double phaseErrorL = std::abs(phaseTotalL - motionMaxTime);
    double phaseErrorR = std::abs(phaseTotalR - motionMaxTime);

    // Calculate exponential reward based on average phase error
    double reward = std::exp(-mRewardConfig.phase_weight * (phaseErrorL + phaseErrorR));

    return reward;
}

double Environment::getStepReward()
{
    // Only Z position (forward progression) matters for step reward
    double foot_diff_z = std::abs(getCurrentFootStep()[2] - getCurrentTargetFootStep()[2]);
    if (foot_diff_z > mRewardConfig.step_clip) return exp(-mRewardConfig.step_weight * foot_diff_z);
    else return 1.0;
}

int Environment::getAvgVelocityHorizonSteps() const
{
    double stride_duration = mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()));
    double window_seconds = stride_duration * mRewardConfig.avg_vel_window_mult;
    double min_window = 1.0 / static_cast<double>(mSimulationHz);
    window_seconds = std::max(window_seconds, min_window);
    return std::max(1, static_cast<int>(window_seconds * mSimulationHz));
}

Eigen::Vector3d Environment::getAvgVelocity()
{
    Eigen::Vector3d avg_vel = Eigen::Vector3d::Zero();
    const std::vector<Eigen::Vector3d> &coms = mCharacter->getCOMLogs();
    int horizon = getAvgVelocityHorizonSteps();
    double window_seconds = static_cast<double>(horizon) / static_cast<double>(mSimulationHz);
    if (coms.size() > static_cast<size_t>(horizon))
    {
        Eigen::Vector3d cur_com = coms.back();
        Eigen::Vector3d prev_com = coms[coms.size() - horizon];
        avg_vel = (cur_com - prev_com) / window_seconds;
    }
    else avg_vel[2] = getTargetCOMVelocity();
    return avg_vel;
}

double Environment::getAvgVelReward()
{
    Eigen::Vector3d curAvgVel = getAvgVelocity();
    double targetCOMVel = getTargetCOMVelocity();

    Eigen::Vector3d vel_diff = curAvgVel - Eigen::Vector3d(0, 0, targetCOMVel);
    if (!(mRewardConfig.flags & REWARD_AVG_VEL_CONSIDER_X)) vel_diff[0] = 0;
    const double vel_diff_norm = vel_diff.norm();
    if (vel_diff_norm > mRewardConfig.avg_vel_clip) return exp(-mRewardConfig.avg_vel_weight * vel_diff_norm);
    else return 1.0;
}

Eigen::VectorXd Environment::getJointState(bool isMirror)
{
    Eigen::VectorXd joint_state = Eigen::VectorXd::Zero(3 * (mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs()));
    Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs());
    Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs());

    auto mt = mCharacter->getMuscleTuple(isMirror);

    for (int i = 0; i < mt.JtA.rows(); i++)
    {
        for (int j = 0; j < mt.JtA.cols(); j++)
        {
            if (mt.JtA(i, j) < 0)
                min_tau[i] += mt.JtA(i, j);
            else
                max_tau[i] += mt.JtA(i, j);
        }
    }
    joint_state << 0.5 * min_tau, 0.5 * max_tau, 1.0 * mt.JtP;
    return joint_state;
}

void Environment::setParamState(Eigen::VectorXd _param_state, bool onlyMuscle, bool doOptimization)
{
    int idx = 0;
    // skeleton parameter
    if (!onlyMuscle)
    {
        std::vector<std::pair<std::string, double>> skel_info;
        for (auto name : mParamName)
        {
            // gait parameter
            if (name.find("stride") != std::string::npos)
                mStride = _param_state[idx];

            if (name.find("cadence") != std::string::npos)
                mCadence = _param_state[idx];

            if (name.find("skeleton") != std::string::npos)
                skel_info.push_back(std::make_pair((name.substr(9)), _param_state[idx]));

            if (name.find("torsion") != std::string::npos)
                skel_info.push_back(std::make_pair(name, _param_state[idx]));

            idx++;
        }
        mCharacter->setSkelParam(skel_info, doOptimization);

        // Sync stride and cadence to GaitPhase (only if initialized)
        if (mGaitPhase) {
            mGaitPhase->setStride(mStride);
            mGaitPhase->setCadence(mCadence);
        }
    }

    idx = 0;
    for (auto name : mParamName)
    {
        if (name.find("muscle_length") != std::string::npos)
        {
            mCharacter->setMuscleParam(name.substr(14), "length", _param_state[idx]);
        }
        else if (name.find("muscle_force") != std::string::npos)
        {
            mCharacter->setMuscleParam(name.substr(13), "force", _param_state[idx]);
        }
        idx++;
    }
    mCharacter->cacheMuscleMass();
}

void Environment::setNormalizedParamState(Eigen::VectorXd _param_state, bool onlyMuscle, bool doOptimization)
{
    int idx = 0;
    // skeleton parameter
    if (!onlyMuscle)
    {
        std::vector<std::pair<std::string, double>> skel_info;
        for (auto name : mParamName)
        {
            // gait parameter
            if (name.find("stride") != std::string::npos) mStride = mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]);
            if (name.find("cadence") != std::string::npos) mCadence = mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]);
            if (name.find("skeleton") != std::string::npos) skel_info.push_back(std::make_pair((name.substr(9)), mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx])));
            if (name.find("torsion") != std::string::npos) skel_info.push_back(std::make_pair(name, mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx])));
            idx++;
        }
        mCharacter->setSkelParam(skel_info, doOptimization);

        // Sync stride and cadence to GaitPhase (only if initialized)
        if (mGaitPhase) {
            mGaitPhase->setStride(mStride);
            mGaitPhase->setCadence(mCadence);
        }
    }

    idx = 0;
    for (auto name : mParamName)
    {
        if (name.find("muscle_length") != std::string::npos)
        {
            mCharacter->setMuscleParam(name.substr(14), "length",
                mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]));
        }
        else if (name.find("muscle_force") != std::string::npos)
        {
            mCharacter->setMuscleParam(name.substr(13), "force",
                mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]));
        }
        idx++;
    }
    mCharacter->cacheMuscleMass();
}

Eigen::VectorXd Environment::getParamState(bool isMirror)
{
    Eigen::VectorXd ParamState = Eigen::VectorXd::Zero(mNumParamState);
    int idx = 0;
    for (auto name : mParamName)
    {
        if (name.find("stride") != std::string::npos)
            ParamState[idx] = mStride;
        if (name.find("cadence") != std::string::npos)
            ParamState[idx] = mCadence;
        if (name.find("skeleton") != std::string::npos)
            ParamState[idx] = mCharacter->getSkelParamValue(name.substr(9));

        if (name.find("torsion") != std::string::npos)
            ParamState[idx] = mCharacter->getTorsionValue(name.substr(8));

        if (name.find("muscle_length") != std::string::npos)
            for (auto m : mCharacter->getMuscles())
                if (name.substr(14) == m->GetName())
                {
                    ParamState[idx] = m->ratio_l();
                    break;
                }

        if (name.find("muscle_force") != std::string::npos)
            for (auto m : mCharacter->getMuscles())
                if (name.substr(13) == m->GetName())
                {
                    ParamState[idx] = m->ratio_f();
                    break;
                }
        idx++;
    }

    if (isMirror)
    {
        int offset = 0;
        for (int i = 0; i < (int)mParamName.size() - 1; i++)
        {
            if (mParamName[i].find("skeleton") != std::string::npos)
                offset = 9;
            else if (mParamName[i].find("torsion") != std::string::npos)
                offset = 8;
            else if (mParamName[i].find("muscle_length") != std::string::npos)
                offset = 14;
            else if (mParamName[i].find("muscle_force") != std::string::npos)
                offset = 13;
            else
                continue;

            if ((mParamName[i].substr(1 + offset) == mParamName[i + 1].substr(1 + offset)) || (mParamName[i].substr(offset, mParamName[i].size() - 1 - offset) == mParamName[i + 1].substr(offset, mParamName[i + 1].size() - 1 - offset)))
            {
                double tmp = 0;
                tmp = ParamState[i];
                ParamState[i] = ParamState[i + 1];
                ParamState[i + 1] = tmp;
                i += 1;
                continue;
            }
        }
    }

    return ParamState;
}

Eigen::VectorXd Environment::getParamSample()
{
    Eigen::VectorXd sampled_param = mParamMin;
    for (auto p : mParamGroups)
    {
        double w = 1;
        std::vector<double> locs;
        locs.push_back(0);
        locs.push_back(1);

        if (p.is_uniform)
        {
            w *= 0.25;
            for (int i = 1; i < 4; i++)
                locs.push_back(i * w);
            if (p.name.find("torsion") != std::string::npos)
                locs.push_back(0.5);
        }

        int sampled_c = (int)thread_safe_uniform(0.0, locs.size() - 0.01);
        double scale = locs[sampled_c]; // + thread_safe_normal(0.0, (mParamMin[p.param_idxs[0]] < 0.1? 0.1 : 0.5) * w);

        scale = dart::math::clip(scale, 0.0, 1.0);

        bool isAllSample = true; //(thread_safe_uniform(0, 1) < (1.0 / 10)?true:false);

        p.v = scale;

        double std_dev = thread_safe_normal(0.0, 0.025);
        for (auto idx : p.param_idxs)
        {
            double param_w = mParamMax[idx] - mParamMin[idx];
            if (isAllSample)
            {
                sampled_c = (int)thread_safe_uniform(0.0, locs.size() - 0.01);
                scale = locs[sampled_c];
                std_dev = thread_safe_normal(0.0, 0.025);
            }
            // std::cout << p.name << " param w " << param_w << " scale " << scale << "loc size " << locs.size() << " is uniform " << p.is_uniform << std::endl;
            sampled_param[idx] = mParamMin[idx] + param_w * scale + std_dev;
            sampled_param[idx] = dart::math::clip(sampled_param[idx], mParamMin[idx], mParamMax[idx]);
        }
    }
    
    return sampled_param;
}

// Contact detection helpers (kept for backward compatibility with Rollout/Viewer)
Eigen::Vector2i Environment::getIsContact()
{
    // Delegate to GaitPhase - returns cached contact state
    return mGaitPhase->getContactState();
}

Eigen::Vector2d Environment::getFootGRF()
{
    // Delegate to GaitPhase - returns cached normalized GRF
    return mGaitPhase->getNormalizedGRF();
}

bool Environment::isGaitCycleComplete()
{
    // Delegate to GaitPhase - returns PD-level persistent flag
    return mGaitPhase->isGaitCycleComplete();
}

void Environment::clearGaitCycleComplete()
{
    // Clear the PD-level completion flag after consumption
    mGaitPhase->clearGaitCycleComplete();
}

bool Environment::isStepComplete()
{
    // Delegate to GaitPhase - returns PD-level persistent flag
    return mGaitPhase->isStepComplete();
}

void Environment::clearStepComplete()
{
    // Clear the PD-level step completion flag after consumption
    mGaitPhase->clearStepComplete();
}

Network Environment::loadPrevNetworks(std::string path, bool isFirst)
{
    Network nn;
    
    // Fix hardcoded network paths from checkpoint metadata
    if (path == "../data/trained_nn/skel_no_mesh_lbs") {
        path = "data/trained_nn/skel_no_mesh_lbs";
    } else if (path == "../data/trained_nn/hip_no_mesh_lbs") {
        path = "data/trained_nn/hip_no_mesh_lbs";
    } else if (path == "../data/trained_nn/ankle_no_mesh_lbs") {
        path = "data/trained_nn/ankle_no_mesh_lbs";
    } else if (path == "../data/trained_nn/merge_no_mesh_lbs") {
        path = "data/trained_nn/merge_no_mesh_lbs";
    }
    // path, state size, action size, acuator type
    py::object py_metadata = py::module::import("python.ray_model").attr("loading_metadata")(path);
    std::string metadata = "";
    if (!py_metadata.is_none())
        metadata = py_metadata.cast<std::string>();
    std::pair<Eigen::VectorXd, Eigen::VectorXd> space = getSpace(metadata);

    Eigen::VectorXd projState = getProjState(space.first, space.second).first;

    py::tuple res = loading_network(path, projState.rows(), mAction.rows() - (isFirst ? 1 : 0), true);

    nn.joint = res[0];

    // Convert Python muscle state_dict to C++ MuscleNN
    if (!res[1].is_none()) {
        Character *character = mCharacter;
        int num_muscles = character->getNumMuscles();
        int num_muscle_dofs = character->getNumMuscleRelatedDof();
        int num_actuator_action = getNumActuatorAction();
        bool is_cascaded = false;  // Prev networks don't use cascading

        // Create C++ MuscleNN
        // Force CPU to avoid CUDA context allocation issues in multi-process scenarios
        nn.muscle = make_muscle_nn(num_muscle_dofs, num_actuator_action, num_muscles, is_cascaded, true);

        // res[1] is now a state_dict (Python dict), not a network object
        py::dict state_dict = res[1].cast<py::dict>();

        // Convert Python state_dict to C++ format
        std::unordered_map<std::string, torch::Tensor> cpp_state_dict;
        for (auto item : state_dict) {
            std::string key = item.first.cast<std::string>();
            py::array_t<float> np_array = item.second.cast<py::array_t<float>>();

            auto buf = np_array.request();
            std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());

            torch::Tensor tensor = torch::from_blob(
                buf.ptr,
                shape,
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone();

            cpp_state_dict[key] = tensor;
        }

        nn.muscle->load_state_dict(cpp_state_dict);
    }

    nn.minV = space.first;
    nn.maxV = space.second;
    nn.name = path;

    return nn;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Environment::getSpace(std::string metadata)
{
    TiXmlDocument doc;
    Eigen::VectorXd minV = Eigen::VectorXd::Ones(mNumParamState);
    Eigen::VectorXd maxV = Eigen::VectorXd::Ones(mNumParamState);

    doc.Parse(metadata.c_str());
    if (doc.FirstChildElement("parameter") != NULL)
    {
        auto parameter = doc.FirstChildElement("parameter");
        for (TiXmlElement *group = parameter->FirstChildElement(); group != NULL; group = group->NextSiblingElement())
        {
            for (TiXmlElement *elem = group->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
            {
                std::string name = std::string(group->Name()) + "_" + std::string(elem->Name());
                for (int i = 0; i < mParamName.size(); i++)
                {
                    if (mParamName[i] == name)
                    {
                        minV[i] = std::stod(elem->Attribute("min"));
                        maxV[i] = std::stod(elem->Attribute("max"));
                    }
                }
            }
        }
    }
    // std::cout <<"[MIN V] : " << minV.transpose() << std::endl;
    // std::cout <<"[MAX V] : " << maxV.transpose() << std::endl;

    return std::make_pair(minV, maxV);
}

void Environment::createNoiseInjector(const std::string& config_path)
{
    if (mNoiseInjector) {
        LOG_WARN("[Environment] NoiseInjector already exists, recreating with new config");
    }

    mNoiseInjector = std::make_unique<NoiseInjector>(config_path, mWorld->getTimeStep());
    LOG_INFO("[Environment] NoiseInjector created" << (config_path.empty() ? " with default parameters" : " from config: " + config_path));
}

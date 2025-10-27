#include "Environment.h"
#include "UriResolver.h"
#include "CBufferData.h"
#include "NPZ.h"
#include "HDF.h"
#include "../viewer/Log.h"
#include <yaml-cpp/yaml.h>


Environment::Environment()
    : mSimulationHz(600), mControlHz(30), mUseMuscle(false), mInferencePerSim(1), 
    mHeightCalibration(0), mEnforceSymmetry(false), mLimitY(0.6), mLearningStd(false)
{
    // Initialize URI resolver for path resolution
    PMuscle::URIResolver::getInstance().initialize();
    
    mWorld = std::make_shared<dart::simulation::World>();
    mCyclic = true;
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
    mPhaseDisplacement = 0.0;
    mNumActuatorAction = 0;

    mLoadedMuscleNN = false;
    mUseJointState = false;
    // Parameter
    mNumParamState = 0;
    mLearningStd = false;

    // Simulation Setting
    mSimulationStep = 0;
    mEOEType = EOEType::abstime;

    mSoftPhaseClipping = false;
    mHardPhaseClipping = false;
    mPhaseCount = 0;
    mWorldPhaseCount = 0;
    mPrevWorldPhaseCount = 0;
    mPrevContact = Eigen::Vector2i(0, 0);
    mGlobalTime = 0.0;
    mWorldTime = 0.0;

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
    if (doc.FirstChildElement("skeleton") != NULL)
    {
        double defaultKp = std::stod(doc.FirstChildElement("skeleton")->Attribute("defaultKp"));
        double defaultKv = std::stod(doc.FirstChildElement("skeleton")->Attribute("defaultKv"));
        double defaultDamping = 0.4;
        if (doc.FirstChildElement("skeleton")->Attribute("damping") != NULL)
            defaultDamping = std::stod(doc.FirstChildElement("skeleton")->Attribute("damping"));

        std::string skeletonPath = Trim(std::string(doc.FirstChildElement("skeleton")->GetText()));
        std::string resolvedSkeletonPath = PMuscle::URIResolver::getInstance().resolve(skeletonPath);
        addCharacter(resolvedSkeletonPath, defaultKp, defaultKv, defaultDamping);

        std::string _actTypeString;
        if (doc.FirstChildElement("skeleton")->Attribute("actuator") != NULL) _actTypeString = Trim(doc.FirstChildElement("skeleton")->Attribute("actuator"));
        else if (doc.FirstChildElement("skeleton")->Attribute("actuactor") != NULL) _actTypeString = Trim(doc.FirstChildElement("skeleton")->Attribute("actuactor"));
        ActuatorType _actType = getActuatorType(_actTypeString);
        mCharacter->setActuatorType(_actType);

        mTargetPositions = mCharacter->getSkeleton()->getPositions();
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
    
    // Learning Std
    if (doc.FirstChildElement("learningStd") != NULL)
        mLearningStd = doc.FirstChildElement("learningStd")->BoolText();

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
    // Ground Loading
    if (doc.FirstChildElement("ground") != NULL) {
        std::string groundPath = Trim(std::string(doc.FirstChildElement("ground")->GetText()));
        std::string resolvedGroundPath = PMuscle::URIResolver::getInstance().resolve(groundPath);
        addObject(resolvedGroundPath);
    }

    // Cyclic Mode
    if (doc.FirstChildElement("cyclicbvh") != NULL)
        mCyclic = doc.FirstChildElement("cyclicbvh")->BoolText();

    // Controller Setting
    if (doc.FirstChildElement("residual") != NULL)
        mIsResidual = doc.FirstChildElement("residual")->BoolText();

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

    // Inference Per Sim
    if (doc.FirstChildElement("inferencePerSim") != NULL)
        mInferencePerSim = doc.FirstChildElement("inferencePerSim")->IntText();

    // soft Phase Clipping
    if (doc.FirstChildElement("softPhaseClipping") != NULL)
        mSoftPhaseClipping = doc.FirstChildElement("softPhaseClipping")->BoolText();

    // hard Phase Clipping
    if (doc.FirstChildElement("hardPhaseClipping") != NULL)
        mHardPhaseClipping = doc.FirstChildElement("hardPhaseClipping")->BoolText();

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

    // Torque Clipping
    if (doc.FirstChildElement("torqueClipping") != NULL)
        mCharacter->setTorqueClipping(doc.FirstChildElement("torqueClipping")->BoolText());

    // Include JtP in SPD
    if (doc.FirstChildElement("includeJtPinSPD") != NULL)
        mCharacter->setIncludeJtPinSPD(doc.FirstChildElement("includeJtPinSPD")->BoolText());

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

    if (doc.FirstChildElement("eoeType") != NULL)
    {
        std::string str_eoeType = doc.FirstChildElement("eoeType")->GetText();
        if (str_eoeType == "time") mEOEType = EOEType::abstime;
        else if (str_eoeType == "tuple") mEOEType = EOEType::tuple;
    }

    // Simulation World Wetting
    mWorld->setTimeStep(1.0 / mSimulationHz);
    // mWorld->getConstraintSolver()->setLCPSolver(dart::common::make_unique<dart::constraint::PGSLCPSolver>(mWorld->getTimeStep));
    // mWorld->setConstraintSolver(std::make_unique<dart::constraint::BoxedLcpConstraintSolver>(std::make_shared<dart::constraint::PgsBoxedLcpSolver>()));
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
    // Add Character
    mWorld->addSkeleton(mCharacter->getSkeleton());
    // Add Objects
    for (auto o : mObjects)
        mWorld->addSkeleton(o);

    // Motion Loading (BVH or NPZ)
    // World Setting 후에 함. 왜냐하면 Height Calibration 을 위해서는 충돌 감지를 필요로 하기 때문.
    if (doc.FirstChildElement("bvh") != NULL)
    {
        std::string bvh_path = Trim(std::string(doc.FirstChildElement("bvh")->GetText()));
        std::string resolvedBvhPath = PMuscle::URIResolver::getInstance().resolve(bvh_path);
        LOG_VERBOSE("[Environment] BVH Path resolved: " << bvh_path << " -> " << resolvedBvhPath);
        BVH *new_bvh = new BVH(resolvedBvhPath);
        new_bvh->setMode(std::string(doc.FirstChildElement("bvh")->Attribute("symmetry")) == "true");
        new_bvh->setHeightCalibration(std::string(doc.FirstChildElement("bvh")->Attribute("heightCalibration")) == "true");

        new_bvh->setRefMotion(mCharacter, mWorld);
        mMotion = new_bvh;
    }
    else if (doc.FirstChildElement("npz") != NULL)
    {
        std::string npz_path = Trim(std::string(doc.FirstChildElement("npz")->GetText()));
        std::string resolvedNpzPath = PMuscle::URIResolver::getInstance().resolve(npz_path);
        LOG_VERBOSE("[Environment] NPZ Path resolved: " << npz_path << " -> " << resolvedNpzPath);
        NPZ *new_npz = new NPZ(resolvedNpzPath);
        new_npz->setHeightCalibration(std::string(doc.FirstChildElement("npz")->Attribute("heightCalibration")) == "true");

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

    // Advanced Option
    if (doc.FirstChildElement("heightCalibration") != NULL)
    {
        if (doc.FirstChildElement("heightCalibration")->BoolText())
        {
            mHeightCalibration++;
            if (std::string(doc.FirstChildElement("heightCalibration")->Attribute("strict")) == "true")
                mHeightCalibration++;
        }
    }

    if (doc.FirstChildElement("enforceSymmetry") != NULL)
        mEnforceSymmetry = doc.FirstChildElement("enforceSymmetry")->BoolText();

    if (isTwoLevelController())
    {
        Character *character = mCharacter;
        mMuscleNN = py::module::import("python.ray_model").attr("generating_muscle_nn")(character->getNumMuscleRelatedDof(), getNumActuatorAction(), character->getNumMuscles(), true, mUseCascading);
    }

    if (doc.FirstChildElement("Horizon") != NULL)
        mHorizon = doc.FirstChildElement("Horizon")->IntText();

    // =================== Reward ======================
    // =================================================

    if (doc.FirstChildElement("useNormalizedParamState") != NULL)
        mUseNormalizedParamState = doc.FirstChildElement("useNormalizedParamState")->BoolText();

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

    if (doc.FirstChildElement("ScaleMetabolic") != NULL)
        mRewardConfig.metabolic_scale = doc.FirstChildElement("ScaleMetabolic")->DoubleText();

    if (doc.FirstChildElement("KneePainWeight") != NULL) {
        mRewardConfig.active |= REWARD_KNEE_PAIN;
        mRewardConfig.knee_pain_weight = doc.FirstChildElement("KneePainWeight")->DoubleText();
    }

    if (doc.FirstChildElement("ScaleKneePain") != NULL)
        mRewardConfig.knee_pain_scale = doc.FirstChildElement("ScaleKneePain")->DoubleText();

    if (doc.FirstChildElement("UseMultiplicativeKneePain") != NULL)
    {
        std::string useMultStr = Trim(std::string(doc.FirstChildElement("UseMultiplicativeKneePain")->GetText()));
        if (useMultStr == "true" || useMultStr == "True" || useMultStr == "TRUE")
            mRewardConfig.multiplicative |= REWARD_KNEE_PAIN;
    }

    if (doc.FirstChildElement("UseKneePainTermination") != NULL)
    {
        std::string useTermStr = Trim(std::string(doc.FirstChildElement("UseKneePainTermination")->GetText()));
        mRewardConfig.use_knee_pain_termination = (useTermStr == "true" || useTermStr == "True" || useTermStr == "TRUE");
    }

    if (doc.FirstChildElement("UseMultiplicativeMetabolic") != NULL)
    {
        std::string useMultStr = Trim(std::string(doc.FirstChildElement("UseMultiplicativeMetabolic")->GetText()));
        if (useMultStr == "true" || useMultStr == "True" || useMultStr == "TRUE")
            mRewardConfig.multiplicative |= REWARD_METABOLIC;
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
}

void Environment::parseEnvConfigYaml(const std::string& yaml_content)
{
    YAML::Node config = YAML::Load(yaml_content);
    YAML::Node env = config["environment"];

    // === Cascading ===
    if (config["cascading"])
        mUseCascading = true;

    // === Skeleton ===
    if (env["skeleton"]) {
        auto skel = env["skeleton"];
        std::string skelPath = skel["file"].as<std::string>();
        std::string resolved = PMuscle::URIResolver::getInstance().resolve(skelPath);
        addCharacter(
            resolved,
            skel["kp"].as<double>(),
            skel["kv"].as<double>(),
            skel["damping"].as<double>(0.4)
        );

        std::string actType = skel["actuator"].as<std::string>();
        mCharacter->setActuatorType(getActuatorType(actType));

        mTargetPositions = mCharacter->getSkeleton()->getPositions();
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

    // === Learning Std ===
    if (env["advanced"] && env["advanced"]["learning_std"])
        mLearningStd = env["advanced"]["learning_std"].as<bool>(false);

    // === Action ===
    if (env["action"]) {
        auto action = env["action"];
        if (action["time_warping"])
            mPhaseDisplacementScale = action["time_warping"].as<double>(-1.0);
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

    // === Ground ===
    if (env["ground"]) {
        std::string groundPath = env["ground"]["file"].as<std::string>();
        std::string resolved = PMuscle::URIResolver::getInstance().resolve(groundPath);
        addObject(resolved);
    }

    // === Motion ===
    if (env["motion"]) {
        auto motion = env["motion"];
        mCyclic = motion["cyclic"].as<bool>(true);

        if (motion["height_calibration"]) {
            auto hcal = motion["height_calibration"];
            if (hcal["enabled"].as<bool>()) {
                mHeightCalibration++;
                if (hcal["strict"].as<bool>(false))
                    mHeightCalibration++;
            }
        }
    }

    // === Action (residual) ===
    if (env["action"] && env["action"]["residual"])
        mIsResidual = env["action"]["residual"].as<bool>(true);

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

    // === Inference per sim ===
    if (env["simulation"] && env["simulation"]["inference_per_sim"])
        mInferencePerSim = env["simulation"]["inference_per_sim"].as<int>(1);

    // === Advanced ===
    if (env["advanced"]) {
        auto adv = env["advanced"];

        if (adv["soft_phase_clipping"])
            mSoftPhaseClipping = adv["soft_phase_clipping"].as<bool>(false);

        if (adv["hard_phase_clipping"])
            mHardPhaseClipping = adv["hard_phase_clipping"].as<bool>(false);

        if (adv["torque_clipping"])
            mCharacter->setTorqueClipping(adv["torque_clipping"].as<bool>());

        if (adv["include_jtp_in_spd"])
            mCharacter->setIncludeJtPinSPD(adv["include_jtp_in_spd"].as<bool>());
    }

    // === Reward Type ===
    if (env["reward"] && env["reward"]["type"]) {
        std::string rewardType = env["reward"]["type"].as<std::string>("deepmimic");
        if (rewardType == "gaitnet") mRewardType = gaitnet;
        else if (rewardType == "deepmimic") mRewardType = deepmimic;
        else if (rewardType == "scadiver") mRewardType = scadiver;
    }

    // === EOE Type ===
    if (env["reward"] && env["reward"]["eoe_type"]) {
        std::string eoeType = env["reward"]["eoe_type"].as<std::string>("time");
        mEOEType = (eoeType == "tuple") ? EOEType::tuple : EOEType::abstime;
    }

    // === World Setup ===
    mWorld->setTimeStep(1.0 / mSimulationHz);
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
    mWorld->addSkeleton(mCharacter->getSkeleton());
    for (auto o : mObjects)
        mWorld->addSkeleton(o);

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

    // === Advanced: enforce symmetry ===
    if (env["advanced"] && env["advanced"]["enforce_symmetry"])
        mEnforceSymmetry = env["advanced"]["enforce_symmetry"].as<bool>(false);

    // === Two-level controller ===
    if (isTwoLevelController()) {
        Character *character = mCharacter;
        mMuscleNN = py::module::import("python.ray_model").attr("generating_muscle_nn")(character->getNumMuscleRelatedDof(), getNumActuatorAction(), character->getNumMuscles(), true, mUseCascading);
    }

    // === Horizon ===
    if (config["Horizon"])
        mHorizon = config["Horizon"].as<int>();

    // === Reward parameters ===
    if (env["advanced"] && env["advanced"]["use_normalized_param_state"])
        mUseNormalizedParamState = env["advanced"]["use_normalized_param_state"].as<bool>(false);

    // === Locomotion rewards ===
    if (env["reward"] && env["reward"]["locomotion"]) {
        auto loco = env["reward"]["locomotion"];
        if (loco["head_linear_acc_weight"])
            mRewardConfig.head_linear_acc_weight = loco["head_linear_acc_weight"].as<double>(4.0);
        if (loco["head_rot_weight"])
            mRewardConfig.head_rot_weight = loco["head_rot_weight"].as<double>(4.0);
        if (loco["step_weight"])
            mRewardConfig.step_weight = loco["step_weight"].as<double>(2.0);
        if (loco["avg_vel_weight"])
            mRewardConfig.avg_vel_weight = loco["avg_vel_weight"].as<double>(6.0);
    }

    // === Metabolic (always active) ===
    if (env["reward"] && env["reward"]["metabolic"]) {
        auto metabolic_config = env["reward"]["metabolic"];

        if (metabolic_config["weight"])
            mRewardConfig.metabolic_weight = metabolic_config["weight"].as<double>(0.05);
        if (metabolic_config["scale"])
            mRewardConfig.metabolic_scale = metabolic_config["scale"].as<double>(1.0);
        if (metabolic_config["multiplicative"] && metabolic_config["multiplicative"].as<bool>(false))
            mRewardConfig.multiplicative |= REWARD_METABOLIC;

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
            if (torque_config["separate"]) {
                mRewardConfig.separate_torque_energy = torque_config["separate"].as<bool>(false);
            }
        }
    }

    // === Knee pain ===
    if (env["reward"] && env["reward"]["knee_pain"]) {
        auto knee = env["reward"]["knee_pain"];
        mRewardConfig.active |= REWARD_KNEE_PAIN;

        if (knee["weight"])
            mRewardConfig.knee_pain_weight = knee["weight"].as<double>(1.0);
        if (knee["scale"])
            mRewardConfig.knee_pain_scale = knee["scale"].as<double>(1.0);
        if (knee["multiplicative"] && knee["multiplicative"].as<bool>(false))
            mRewardConfig.multiplicative |= REWARD_KNEE_PAIN;
        if (knee["use_termination"])
            mRewardConfig.use_knee_pain_termination = knee["use_termination"].as<bool>(false);
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
}

void Environment::addCharacter(std::string path, double kp, double kv, double damping)
{
    mCharacter = new Character(path, kp, kv, damping);
    // std::cout << "Skeleton Added " << mCharacter->getSkeleton()->getName() << " Degree Of Freedom : " << mCharacter->getSkeleton()->getNumDofs() << std::endl;
}

void Environment::addObject(std::string path)
{
    mObjects.push_back(BuildFromFile(path));
}

void Environment::setAction(Eigen::VectorXd _action)
{
    mPhaseDisplacement = 0.0;
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
                mPhaseDisplacement += mPhaseDisplacementScale * prev_action[mNumActuatorAction];
                continue;
            }
            double beta = 0.2 + 0.1 * prev_action[prev_action.rows() - 1];
            mBetas[i] = beta;
            mWeights[i] = mPrevNetworks.front().joint.attr("weight_filter")(mDmins[i], beta).cast<double>();

            // Joint Anlge 부분은 add position 을 통해서
            mAction.head(mNumActuatorAction) = mCharacter->addPositions(mAction.head(mNumActuatorAction), (mUseWeights[i * (mUseMuscle ? 2 : 1)] ? 1 : 0) * mWeights[i] * mActionScale * prev_action.head(mNumActuatorAction), false); // mAction.head(mNumActuatorAction)
            mAction.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction) += (mUseWeights[i * (mUseMuscle ? 2 : 1)] ? 1 : 0) * mWeights[i] * prev_action.segment(mNumActuatorAction, (mAction.rows() - 1) - mNumActuatorAction);
            mPhaseDisplacement += mWeights[i] * mPhaseDisplacementScale * prev_action[mNumActuatorAction];
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
    
    if (mPhaseDisplacementScale > 0.0)
        mPhaseDisplacement += (mWeights.size() > 0 ? mWeights.back() : 1.0) * mPhaseDisplacementScale * mAction[mNumActuatorAction];
    else
        mPhaseDisplacement = 0.0;

    if (mPhaseDisplacement < (-1.0 / mControlHz))
        mPhaseDisplacement = -1.0 / mControlHz;

    Eigen::VectorXd actuatorAction = mAction.head(mNumActuatorAction);

    updateTargetPosAndVel();

    if (mCharacter->getActuatorType() == pd || 
        mCharacter->getActuatorType() == mass || 
        mCharacter->getActuatorType() == mass_lower)
    {
        Eigen::VectorXd action = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
        action.tail(actuatorAction.rows()) = actuatorAction;
        if (isMirror()) action = mCharacter->getMirrorPosition(action);
        if (mIsResidual) action = mCharacter->addPositions(mTargetPositions, action);
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
    
    mTargetPositions = mMotion->getTargetPose(ofsPhase);
    const auto nextPose = mMotion->getTargetPose(ofsPhase + dPhase);
    mTargetVelocities = mCharacter->getSkeleton()->getPositionDifferences(nextPose, mTargetPositions) / dTime;
}

void Environment::checkTerminated()
{
    // Episode ends due to failure: fall or character below height limit
    double root_y = mCharacter->getSkeleton()->getCOM()[1];
    bool is_fall = root_y < mLimitY * mCharacter->getGlobalRatio();
    bool knee_pain = mRewardConfig.use_knee_pain_termination && (mSimulationStep > 100) && (mCharacter->getKneeLoadingMax() > 5.0);
    bool terminated = is_fall || knee_pain;

    // Log to mInfoMap for TensorBoard
    mInfoMap["terminated"] = terminated ? 1.0 : 0.0;
    mInfoMap["termination_fall"] = is_fall ? 1.0 : 0.0;
    mInfoMap["termination_knee_pain"] = knee_pain ? 1.0 : 0.0;
}

void Environment::checkTruncated()
{
    // Episode ends due to external limits: time or step count
    bool time_limit = false;
    bool step_limit = false;

    if (mEOEType == EOEType::tuple)
        step_limit = mSimulationStep >= mHorizon;
    else if (mEOEType == EOEType::abstime)
        time_limit = mWorld->getTime() > 10.0;

    bool truncated = time_limit || step_limit;

    // Log to mInfoMap for TensorBoard
    mInfoMap["truncated"] = truncated ? 1.0 : 0.0;
    mInfoMap["truncation_time"] = time_limit ? 1.0 : 0.0;
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

        Eigen::VectorXd pos_diff = skel->getPositionDifferences(mTargetPositions, pos);
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
        skel->setPositions(mTargetPositions);
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
        double w_gait = 2.0;
        double r_loco = getLocoReward();
        double r_avg = getAvgVelReward();
        double r_step = getStepReward();
        double r_energy = getEnergyReward();
        double r_knee_pain = getKneePainReward();

        // Build multiplicative and additive components separately using bitflags
        double multiplicative_part = w_gait * r_loco * r_avg * r_step;
        double additive_part = 0.0;

        // Apply energy reward (always active, multiplicative or additive)
        if (mRewardConfig.multiplicative & REWARD_METABOLIC)
            multiplicative_part *= r_energy;
        else
            additive_part += mRewardConfig.metabolic_scale * r_energy;

        // When torque energy is separated, calculate and apply separate torque reward
        if (mRewardConfig.separate_torque_energy) {
            double r_torque = exp(-mRewardConfig.metabolic_weight * mCharacter->getTorqueEnergy());
            if (mRewardConfig.multiplicative & REWARD_METABOLIC)
                multiplicative_part *= r_torque;
            else
                additive_part += mRewardConfig.metabolic_scale * r_torque;
            mInfoMap.insert(std::make_pair("r_torque", r_torque));
        }

        // Apply knee pain reward (multiplicative or additive based on bitflag)
        if (mRewardConfig.active & REWARD_KNEE_PAIN)
        {
            if (mRewardConfig.multiplicative & REWARD_KNEE_PAIN)
                multiplicative_part *= r_knee_pain;
            else
                additive_part += mRewardConfig.knee_pain_scale * r_knee_pain;
        }

        r = multiplicative_part + additive_part;

        // Populate reward map for gaitnet
        mInfoMap.insert(std::make_pair("r_loco", r_loco));
        mInfoMap.insert(std::make_pair("r_avg", r_avg));
        mInfoMap.insert(std::make_pair("r_step", r_step));
        mInfoMap.insert(std::make_pair("r_energy", r_energy));

        // Log separate metabolic when separation is enabled
        if (mRewardConfig.separate_torque_energy) {
            double r_metabolic = exp(-mRewardConfig.metabolic_weight * mCharacter->getMetabolicEnergy());
            mInfoMap.insert(std::make_pair("r_metabolic", r_metabolic));
        }

        mInfoMap.insert(std::make_pair("r_knee_pain", r_knee_pain));
    }

    if (mCharacter->getActuatorType() == mus) r = 1.0;

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
        step_state[0] = mNextTargetFoot[2] - mCharacter->getSkeleton()->getCOM()[2];
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
    if (mPrevNetworks.size() > 0)
        prev_activations[0] = mPrevNetworks[0].muscle.attr("unnormalized_no_grad_forward")(mt.JtA_reduced, dt, py::cast<py::none>(Py_None), true, py::cast<py::none>(Py_None)).cast<Eigen::VectorXf>();

    for (int j = 1; j < mPrevNetworks.size(); j++)
    {
        Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());
        for (int k : mChildNetworks[j]) prev_activation += prev_activations[k];
        prev_activations[j] = (mUseWeights[j * 2 + 1] ? 1 : 0) * mWeights[j] * mPrevNetworks[j].muscle.attr("unnormalized_no_grad_forward")(mt.JtA_reduced, dt, prev_activation, true, mWeights[j]).cast<Eigen::VectorXf>();
    }
    // Current Network
    if (mLoadedMuscleNN)
    {
        Eigen::VectorXf prev_activation = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());
        for (int k : mChildNetworks.back()) prev_activation += prev_activations[k];

        if (mPrevNetworks.size() > 0) prev_activations[prev_activations.size() - 1] = (mUseWeights.back() ? 1 : 0) * mWeights.back() * mMuscleNN.attr("unnormalized_no_grad_forward")(mt.JtA_reduced, dt, prev_activation, true, mWeights.back()).cast<Eigen::VectorXf>();
        else prev_activations[prev_activations.size() - 1] = mMuscleNN.attr("unnormalized_no_grad_forward")(mt.JtA_reduced, dt, py::cast<py::none>(Py_None), true, py::cast<py::none>(Py_None)).cast<Eigen::VectorXf>();
    }

    Eigen::VectorXf activations = Eigen::VectorXf::Zero(mCharacter->getMuscles().size());
    for (Eigen::VectorXf a : prev_activations) activations += a;

    activations = mMuscleNN.attr("forward_filter")(activations).cast<Eigen::VectorXf>();

    if (isMirror()) activations = mCharacter->getMirrorActivation(activations.cast<double>()).cast<float>();

    mCharacter->setActivations(activations.cast<double>());

    if (dart::math::Random::uniform(0.0, 1.0) < 1.0 / static_cast<double>(mNumSubSteps) || !mTupleFilled)
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
}

void Environment::postMuscleStep()
{
    mSimulationCount++;
    mGlobalTime += 1.0 / mSimulationHz;
    mWorldTime += 1.0 / mSimulationHz;
    mCharacter->updateLocalTime((1.0 + mPhaseDisplacement * mControlHz) / mSimulationHz);

    // Contact-based gait cycle detection: detect right foot heel strike (swing→stance transition)
    // This replaces time-based mWorldPhaseCount update for more accurate cycle tracking
    // GRF threshold (0.2 * body weight) ensures the foot is actually bearing weight
    Eigen::Vector2i contact = getIsContact();
    Eigen::Vector2d grf = getFootGRF();

    const double grf_threshold = 0.2;  // 20% of body weight
    if (mPrevContact[1] == 0 && contact[1] == 1 && grf[1] > grf_threshold)  // Right foot: swing→stance with weight
    {
        mWorldPhaseCount++;
        mWorldTime = mCharacter->getLocalTime();
    }
    mPrevContact = contact;

    if (mHardPhaseClipping)
    {
        int currentGlobalCount = mGlobalTime / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
        int currentLocalCount = mCharacter->getLocalTime() / ((mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))));

        if (currentGlobalCount > currentLocalCount) mCharacter->setLocalTime(mGlobalTime);
        else if (currentGlobalCount < currentLocalCount) mCharacter->setLocalTime(currentLocalCount * ((mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())))));
    }
    else if (mSoftPhaseClipping)
    {
        // FIXED LOCAL PHASE TIME
        int currentCount = mCharacter->getLocalTime() / (0.5 * (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))));
        // int currentCount = mCharacter->getLocalTime() / ((mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))));
        if (mPhaseCount != currentCount)
        {
            mGlobalTime = mCharacter->getLocalTime();
            mPhaseCount = currentCount;
        }
    }
}

void Environment::muscleStep()
{
    if (mCharacter->getActuatorType() == mass || mCharacter->getActuatorType() == mass_lower)  calcActivation();
    mCharacter->step();
    mWorld->step();
    postMuscleStep();
}

void Environment::step()
{
    for (int i = 0; i < mNumSubSteps; i++) muscleStep();
    postStep();
}

void Environment::postStep()
{
    mInfoMap.clear();
    mCharacter->evalStep();
    if (mRewardType == gaitnet) updateFootStep();
    mReward = calcReward();

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
    mIsLeftLegStance = !((0.33 < phase) && (phase <= 0.83));

    // Stance Leg Hip anlge Change
    if (true)
    {
        double angle_threshold = 1;
        auto femur_joint = skel->getJoint((mIsLeftLegStance ? "FemurL" : "FemurR"));
        auto foot_bn = skel->getBodyNode((mIsLeftLegStance ? "TalusL" : "TalusR"));
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
            if (abs(angle_diff) < M_PI * 10 / 180.0)
                break;

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
            if (isDone)
                break;
        }
    }

    // Rotation Change
    Eigen::Vector3d com = skel->getCOM(skel->getRootBodyNode());
    Eigen::Vector3d foot;
    if (mPoseOptimizationMode == 0)
        foot = skel->getBodyNode(mIsLeftLegStance ? "TalusL" : "TalusR")->getCOM(skel->getRootBodyNode());
    else if (mPoseOptimizationMode == 1)
        foot = (skel->getBodyNode("TalusL")->getCOM(skel->getRootBodyNode()) + skel->getBodyNode("TalusR")->getCOM(skel->getRootBodyNode())) * 0.5;
    // is it stance boundary?
    double global_diff = (skel->getCOM() - skel->getBodyNode(mIsLeftLegStance ? "TalusL" : "TalusR")->getCOM())[2];
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
    mSimulationStep = 0;
    mPhaseCount = 0;
    mWorldPhaseCount = 0;
    mPrevWorldPhaseCount = 0;
    mSimulationCount = 0;
    mPrevContact = Eigen::Vector2i(0, 0); // Initialize contact state

    // Reset Initial Time
    double time = 0.0;
    if (phase >= 0.0 && phase <= 1.0) {
        // Use specified phase (0.0 to 1.0)
        time = phase * (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
    }
    else if (mRewardType == deepmimic) {
        time = dart::math::Random::uniform(1E-2, mMotion->getMaxTime() - 1E-2);
    }
    else if (mRewardType == gaitnet)
    {
        time = (dart::math::Random::uniform(0.0, 1.0) > 0.5 ? 0.5 : 0.0) + mStanceOffset + dart::math::Random::uniform(-0.05, 0.05);
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

    mCharacter->setLocalTime(time);

    // Initial Pose Setting
    updateTargetPosAndVel(true);
    
    if(mRewardType == gaitnet)
    {
        // mTargetPositions.segment(6, 18) *= (mStride * (mCharacter->getGlobalRatio()));
        mTargetVelocities.head(24) *= (mStride * (mCharacter->getGlobalRatio()));
    }
    
    mCharacter->getSkeleton()->setPositions(mTargetPositions);
    mCharacter->getSkeleton()->setVelocities(mTargetVelocities);

    updateTargetPosAndVel();
    
    // auto pose = mCharacter->getSkeleton()->getPositions().head(6);
    // std::cout << "skel orientation: " << pose.transpose() * 180.0 / M_PI << std::endl;
    // auto r_before = FreeJoint::convertToTransform(pose).linear();
    // std::cout << "r_before: " << r_before.transpose() << std::endl;

    // if (mMusclePoseOptimization) poseOptimization();
    if (mRewardType == gaitnet)
    {
        Eigen::Vector3d ref_initial_vel = mTargetVelocities.segment(3, 3);
        ref_initial_vel = 
            FreeJoint::convertToTransform(mCharacter->getSkeleton()->getRootJoint()->getPositions()).linear().transpose() * 
            (FreeJoint::convertToTransform(mTargetPositions.head(6)).linear() * ref_initial_vel);
        Eigen::Vector6d vel = mCharacter->getSkeleton()->getRootJoint()->getVelocities();
        vel.segment(3, 3) = ref_initial_vel;
        mCharacter->getSkeleton()->getRootJoint()->setVelocities(vel);
    }
    
    // pose = mCharacter->getSkeleton()->getPositions().head(6);
    // std::cout << "skel2 orientation: " << pose.transpose() * 180.0 / M_PI << std::endl;
    // auto r_after = FreeJoint::convertToTransform(pose).linear();
    // std::cout << "r_after: " << r_after.transpose() << std::endl;
    
    // Height / Pose Optimization
    if (mHeightCalibration != 0) mCharacter->heightCalibration(mWorld, mHeightCalibration == 2);

    // Pose In ROM
    Eigen::VectorXd cur_pos = mCharacter->getSkeleton()->getPositions();
    Eigen::VectorXd rom_min = mCharacter->getSkeleton()->getPositionLowerLimits();
    Eigen::VectorXd rom_max = mCharacter->getSkeleton()->getPositionUpperLimits();
    cur_pos = cur_pos.cwiseMax(rom_min).cwiseMin(rom_max);
    mCharacter->getSkeleton()->setPositions(cur_pos);

    mCharacter->setPDTarget(mTargetPositions);
    mCharacter->setTorque(mCharacter->getTorque().setZero());
    if (mUseMuscle) {
        mCharacter->setActivations(mCharacter->getActivations().setZero());
        mCharacter->resetStep();
    }

    mCharacter->clearLogs();
    mDesiredTorqueLogs.clear();

    if (mRewardType == gaitnet) updateFootStep(true);

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
    if (mRewardConfig.separate_torque_energy) {
        // When separated, use only metabolic energy
        energy = mCharacter->getMetabolicEnergy();
    } else {
        // Default behavior: combined metabolic + torque energy
        energy = mCharacter->getEnergy();
    }
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

double Environment::getLocoReward()
{
    const std::vector<Eigen::Vector3d> &headVels = mCharacter->getHeadVelLogs();
    if (headVels.size() == 0) return 1.0;

    Eigen::Vector3d headLinearAcc = headVels.back() - headVels[headVels.size() - mNumSubSteps];

    double headRotDiff = Eigen::AngleAxisd(mCharacter->getSkeleton()->getBodyNode("Head")->getTransform().linear()).angle();
    double r_head_linear_acc = exp(-mRewardConfig.head_linear_acc_weight * headLinearAcc.squaredNorm() / headLinearAcc.rows());
    double r_head_rot_diff = exp(-mRewardConfig.head_rot_weight * headRotDiff * headRotDiff);
    double r_loco = r_head_linear_acc * r_head_rot_diff;

    return r_loco;
}

double Environment::getStepReward()
{
    Eigen::Vector3d foot_diff = mCurrentFoot - mCurrentTargetFoot;
    foot_diff[0] = 0; // Ignore X axis difference

    Eigen::Vector3d clipped_foot_diff = foot_diff.cwiseMax(-0.075).cwiseMin(0.075);
    foot_diff -= clipped_foot_diff;
    Eigen::Vector2i is_contact = getIsContact();
    if ((mIsLeftLegStance && is_contact[0] == 1) || (!mIsLeftLegStance && is_contact[1] == 1)) foot_diff[1] = 0;
    foot_diff *= 8;
    double r = exp(-mRewardConfig.step_weight * foot_diff.squaredNorm() / foot_diff.rows());
    return r;
}

Eigen::Vector3d Environment::getAvgVelocity()
{
    Eigen::Vector3d avg_vel = Eigen::Vector3d::Zero();
    const std::vector<Eigen::Vector3d> &coms = mCharacter->getCOMLogs();
    int horizon = (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio()))) * mSimulationHz;
    if (coms.size() > horizon)
    {
        Eigen::Vector3d cur_com = coms.back();
        Eigen::Vector3d prev_com = coms[coms.size() - horizon];
        avg_vel = (cur_com - prev_com) / (mMotion->getMaxTime() / (mCadence / sqrt(mCharacter->getGlobalRatio())));
    }
    else
        avg_vel[2] = getTargetCOMVelocity();

    return avg_vel;
}

double Environment::getAvgVelReward()
{
    Eigen::Vector3d curAvgVel = getAvgVelocity();
    double targetCOMVel = getTargetCOMVelocity();

    Eigen::Vector3d vel_diff = curAvgVel - Eigen::Vector3d(0, 0, targetCOMVel);
    double vel_reward = exp(-mRewardConfig.avg_vel_weight * vel_diff.squaredNorm());
    return vel_reward;
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

void Environment::updateFootStep(bool isInit)
{

    double phase = getLocalPhase(true);
    if (0.33 < phase && phase <= 0.83)
    {
        // Transition Timing
        if (!isInit)
            if (mIsLeftLegStance)
            {
                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentFoot + Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacter->getGlobalRatio();
            }

        mIsLeftLegStance = false;
        mCurrentFoot = mCharacter->getSkeleton()->getBodyNode("TalusR")->getCOM();

        if (isInit)
        {
            mCurrentTargetFoot = mCurrentFoot;
            mNextTargetFoot = mCurrentFoot + 0.5 * Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacter->getGlobalRatio();
        }
    }
    else
    {
        // Transition Timing
        if (!isInit)
            if (!mIsLeftLegStance)
            {
                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentFoot + Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacter->getGlobalRatio();
            }

        mIsLeftLegStance = true;

        mCurrentFoot = mCharacter->getSkeleton()->getBodyNode("TalusL")->getCOM();

        if (isInit)
        {
            mCurrentTargetFoot = mCurrentFoot;
            mNextTargetFoot = mCurrentFoot + 0.5 * Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacter->getGlobalRatio();
        }
    }
    mCurrentTargetFoot[1] = 0.0;
    mNextTargetFoot[1] = 0.0;
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

        int sampled_c = (int)dart::math::Random::uniform(0.0, locs.size() - 0.01);
        double scale = locs[sampled_c]; // + dart::math::Random::normal(0.0, (mParamMin[p.param_idxs[0]] < 0.1? 0.1 : 0.5) * w);

        scale = dart::math::clip(scale, 0.0, 1.0);

        bool isAllSample = true; //(dart::math::Random::uniform(0, 1) < (1.0 / 10)?true:false);

        p.v = scale;

        double std_dev = dart::math::Random::normal(0.0, 0.025);
        for (auto idx : p.param_idxs)
        {
            double param_w = mParamMax[idx] - mParamMin[idx];
            if (isAllSample)
            {
                sampled_c = (int)dart::math::Random::uniform(0.0, locs.size() - 0.01);
                scale = locs[sampled_c];
                std_dev = dart::math::Random::normal(0.0, 0.025);
            }
            // std::cout << p.name << " param w " << param_w << " scale " << scale << "loc size " << locs.size() << " is uniform " << p.is_uniform << std::endl;
            sampled_param[idx] = mParamMin[idx] + param_w * scale + std_dev;
            sampled_param[idx] = dart::math::clip(sampled_param[idx], mParamMin[idx], mParamMax[idx]);
        }
    }
    
    return sampled_param;
}

Eigen::Vector2i Environment::getIsContact()
{
    Eigen::Vector2i result = Eigen::Vector2i(0, 0);
    const auto results = mWorld->getConstraintSolver()->getLastCollisionResult();
    for (auto bn : results.getCollidingBodyNodes())
    {
        if (bn->getName() == "TalusL" || ((bn->getName() == "FootPinkyL" || bn->getName() == "FootThumbL")))
            result[0] = 1;

        if (bn->getName() == "TalusR" || ((bn->getName() == "FootPinkyR" || bn->getName() == "FootThumbR")))
            result[1] = 1;
    }
    return result;
}

Eigen::Vector2d Environment::getFootGRF()
{
    Eigen::Vector2d grf = Eigen::Vector2d::Zero();
    const auto results = mWorld->getConstraintSolver()->getLastCollisionResult();
    const double mass = mCharacter->getSkeleton()->getMass();
    const double g = 9.81;

    for (std::size_t i = 0; i < results.getNumContacts(); ++i)
    {
        const auto& contact = results.getContact(i);
        const std::string name1 = contact.collisionObject1->getShapeFrame()->getName();
        const std::string name2 = contact.collisionObject2->getShapeFrame()->getName();

        // Check left foot contact
        if (name1.find("TalusL") != std::string::npos || name1.find("FootPinkyL") != std::string::npos || name1.find("FootThumbL") != std::string::npos ||
            name2.find("TalusL") != std::string::npos || name2.find("FootPinkyL") != std::string::npos || name2.find("FootThumbL") != std::string::npos)
        {
            grf[0] += contact.force.norm();
        }

        // Check right foot contact
        if (name1.find("TalusR") != std::string::npos || name1.find("FootPinkyR") != std::string::npos || name1.find("FootThumbR") != std::string::npos ||
            name2.find("TalusR") != std::string::npos || name2.find("FootPinkyR") != std::string::npos || name2.find("FootThumbR") != std::string::npos)
        {
            grf[1] += contact.force.norm();
        }
    }

    // Normalize by body weight (mass * g)
    grf /= (mass * g);

    return grf;
}

bool Environment::isGaitCycleComplete()
{
    if (mWorldPhaseCount > mPrevWorldPhaseCount){
        mPrevWorldPhaseCount = mWorldPhaseCount;
        return true;
    }
    return false;
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
    nn.muscle = res[1];
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
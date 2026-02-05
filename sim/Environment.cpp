#include "Environment.h"
#include "rm/rm.hpp"
#include "CBufferData.h"
#include "NPZ.h"
#include "HDF.h"
#include "Log.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

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

Environment::Environment(const std::string& filepath)
    : mSimulationHz(600), mControlHz(30), mUseMuscle(false), mInferencePerSim(1),
      mUseMirror(true), mLocalState(false), mZeroAnkle0OnReset(false), mLimitY(0.6), mIsResidual(true),
      mSimulationCount(0), mActionScale(0.04), mIncludeMetabolicReward(true),
      mRewardType(deepmimic), mRefStride(1.34), mStride(1.0), mCadence(1.0),
      mPhaseDisplacementScale(-1.0), mNumActuatorAction(0),
      mUseJointState(false), mNumParamState(0), mSimulationStep(0),
      mKneeLoadingMaxCycle(0.0), mDragStartX(0.0), mUseCascading(false),
      mUseNormalizedParamState(true),
      mHorizon(600)
{
    mWorld = std::make_shared<dart::simulation::World>();

    // Read config file
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open config: " + filepath);
    std::stringstream ss;
    ss << ifs.rdbuf();
    mMetadata = ss.str();

    // Parse YAML config (inlined from former parseEnvConfigYaml)
    YAML::Node config = YAML::Load(mMetadata);
    if (!config["environment"]) {
        throw std::runtime_error("Missing 'environment' key in YAML config");
    }
    YAML::Node env = config["environment"];

    // === Global PID for @pid:/ URI expansion ===
    // Supports both: pid: 20705431 or pid: "20705431/pre"
    if (env["pid"]) {
        mGlobalPid = env["pid"].as<std::string>();
    }

    // === Cascading ===
    if (config["cascading"]) mUseCascading = true;

    // === Skeleton ===
    ActuatorType actuatorType = mass;  // Default
    bool scaleTauOnWeight = false;
    if (env["skeleton"]) {
        auto skel = env["skeleton"];
        std::string skelPath = skel["file"].as<std::string>();
        std::string resolved = rm::resolve(rm::expand_pid(skelPath, mGlobalPid));
        bool selfCollide = skel["self_collide"].as<bool>(false);
        int skelFlags = SKEL_DEFAULT;
        if (selfCollide) skelFlags |= SKEL_COLLIDE_ALL;
        mCharacter = new Character(resolved, skelFlags);

        std::string actType = skel["actuator"].as<std::string>();
        actuatorType = getActuatorType(actType);

        // Enable upper body torque scaling based on body mass (stabilizes light bodies)
        if (skel["scale_tau_on_weight"]) {
            scaleTauOnWeight = skel["scale_tau_on_weight"].as<bool>();
            mCharacter->setScaleTauOnWeight(scaleTauOnWeight);
        }

        // Enable critical damping: Kv = 2 * sqrt(Kp * M_diag)
        if (skel["use_critical_damping"]) {
            bool useCriticalDamping = skel["use_critical_damping"].as<bool>();
            mCharacter->setUseCriticalDamping(useCriticalDamping);
        }

        mRefPose = mCharacter->getSkeleton()->getPositions();
        mTargetVelocities = mCharacter->getSkeleton()->getVelocities();

        // Initialize imitation mask (1.0 = active, 0.0 = masked via curriculum)
        mImitMask = Eigen::ArrayXd::Ones(mCharacter->getSkeleton()->getNumDofs());

        // Create Controller with configuration
        int numDofs = mCharacter->getSkeleton()->getNumDofs();
        int rootDof = mCharacter->getSkeleton()->getRootJoint()->getNumDofs();
        int lowerBodyDof = 18;  // First 18 DOFs after root are lower body

        // Muscle DOF count depends on actuator type
        int numMuscleDof = (actuatorType == mass) ? (numDofs - rootDof) : lowerBodyDof;

        ControllerConfig ctrlConfig;
        ctrlConfig.skeleton = mCharacter->getSkeleton();
        ctrlConfig.kp = mCharacter->getKpVector();
        ctrlConfig.kv = mCharacter->getKvVector();
        ctrlConfig.actuatorType = actuatorType;
        ctrlConfig.inferencePerSim = mInferencePerSim;
        ctrlConfig.numMuscleDof = numMuscleDof;
        ctrlConfig.scaleTauOnWeight = scaleTauOnWeight;
        ctrlConfig.torqueMassRatio = scaleTauOnWeight ? mCharacter->getTorqueMassRatio() : 1.0;

        mController = std::make_unique<Controller>(ctrlConfig);

        // Set Character pointer for Controller to handle mirroring internally
        mController->setMirrorCharacter(mCharacter);
    }

    // === Muscle ===
    if (env["muscle"]) {
        auto muscle = env["muscle"];

        bool meshLbs = muscle["mesh_lbs_weight"].as<bool>(false);

        std::string musclePath = muscle["file"].as<std::string>();
        std::string resolved = rm::resolve(rm::expand_pid(musclePath, mGlobalPid));
        mCharacter->setMuscles(resolved, meshLbs);
        mUseMuscle = true;

        // Clip lm_norm for passive force calculation
        if (muscle["clip_lm_norm"]) {
            double clip = muscle["clip_lm_norm"].as<double>();
            mCharacter->setClipLmNorm(clip);
        }

        // === Weight from metadata or direct value ===
        // Supports two formats:
        //   weight_from: 17.7                      (direct numeric value)
        //   weight_from: "@pid:/metadata.yaml"    (load from metadata["weight"])
        if (env["skeleton"]["weight_from"]) {
            auto weightNode = env["skeleton"]["weight_from"];

            if (weightNode.IsScalar()) {
                std::string val = weightNode.as<std::string>();

                // Check if it's a URI (starts with @)
                if (!val.empty() && val[0] == '@') {
                    try {
                        std::string resolved = rm::resolve(rm::expand_pid(val, mGlobalPid));
                        YAML::Node metadata = YAML::LoadFile(resolved);

                        if (metadata["weight"]) {
                            double weight = metadata["weight"].as<double>();
                            mCharacter->setBodyMass(weight);
                            LOG_VERBOSE("[Environment] Set body mass from " << val << ": " << weight << " kg");
                        } else {
                            LOG_WARN("[Environment] weight_from: 'weight' not found in " << val);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("[Environment] Failed to load weight from " << val << ": " << e.what());
                    }
                } else {
                    // Direct numeric value
                    double weight = weightNode.as<double>();
                    mCharacter->setBodyMass(weight);
                    LOG_VERBOSE("[Environment] Set body mass from direct value: " << weight << " kg");
                }
            }
        }

        // Enable muscle force (f0) scaling based on body mass (mass^(2/3) law)
        if (muscle["scale_f0_on_weight"]) {
            bool scaleF0 = muscle["scale_f0_on_weight"].as<bool>();
            mCharacter->setScaleF0OnWeight(scaleF0);
        }

        // Apply muscle force scaling after body mass is set
        mCharacter->updateMuscleForceRatio();

        // Apply critical damping after body mass is set (Kv depends on mass matrix)
        mCharacter->updateCriticalDamping();
    }

    // === Noise Injection ===
    if (env["noise_injection"]) {
        auto ni = env["noise_injection"];
        if (ni["file"]) {
            std::string niPath = ni["file"].as<std::string>();
            std::string resolved = rm::resolve(rm::expand_pid(niPath, mGlobalPid));
            mNoiseInjector = std::make_unique<NoiseInjector>(resolved, mWorld->getTimeStep());
            LOG_INFO("[Environment] Loaded noise injection config: " << resolved);
        }
    } else {
        // Create default NoiseInjector (disabled by default)
        mNoiseInjector = std::make_unique<NoiseInjector>("", mWorld->getTimeStep());
        mNoiseInjector->setEnabled(false);
        LOG_VERBOSE("[Environment] Created default NoiseInjector (disabled)");
    }

    // === Action & GaitPhase config ===
    std::string gaitUpdateMode = "phase";
    double contactDebounceAlpha = 0.25;
    double stepMinRatio = 0.3;
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
    // All actuator types (tor, pd, mass, mass_lower) use joint-space actions
    mAction = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs() + (mPhaseDisplacementScale > 0 ? 1 : 0) + (mUseCascading ? 1 : 0));
    mNumActuatorAction = mCharacter->getSkeleton()->getNumDofs() - mCharacter->getSkeleton()->getRootJoint()->getNumDofs();

    // === Ground === (hardcoded)
    mGround = BuildFromFile(rm::resolve("@data/ground.xml"), SKEL_DEFAULT);

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
        mZeroAnkle0OnReset = sim["zero_ankle0_on_reset"].as<bool>(false);
    }

    // === Action scale ===
    if (env["action"] && env["action"]["scale"])
        mActionScale = env["action"]["scale"].as<double>(0.04);

    // === Inference per sim === (hardcoded)
    mInferencePerSim = 1;

    // === Advanced === (hardcoded)
    mCharacter->setTorqueClipping(false);
    mCharacter->setIncludeJtPinSPD(false);

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
        std::string motionPath;
        bool getStrideFromMotion = false;

        // Support both formats: motion: "uri" or motion: {file: "uri", ...}
        if (motion.IsScalar()) {
            motionPath = motion.as<std::string>();
        } else {
            motionPath = motion["file"].as<std::string>();
            mUseMirror = motion["use_mirror"].as<bool>(true);
            mLocalState = motion["local_state"].as<bool>(false);
            getStrideFromMotion = motion["get_stride_from_motion"].as<bool>(false);
        }

        std::string resolved = rm::resolve(rm::expand_pid(motionPath, mGlobalPid));

        // Auto-detect type from file extension
        std::string ext = resolved.substr(resolved.find_last_of('.') + 1);
        if (ext == "h5" || ext == "hdf") {
            HDF *new_hdf = new HDF(resolved);
            new_hdf->setRefMotion(mCharacter, mWorld);
            mMotion = new_hdf;
        }
        else if (ext == "bvh") {
            BVH *new_bvh = new BVH(resolved);
            new_bvh->setRefMotion(mCharacter, mWorld);
            mMotion = new_bvh;
        }
        else if (ext == "npz") {
            NPZ *new_npz = new NPZ(resolved);
            new_npz->setRefMotion(mCharacter, mWorld);
            mMotion = new_npz;
        }

        // Get stride from motion if configured
        if (getStrideFromMotion && mMotion) {
            HDF* hdf = dynamic_cast<HDF*>(mMotion);
            if (hdf) {
                double stride = hdf->getStrideAttribute(-1.0);
                if (stride > 0.0) {
                    mRefStride = stride;
                } else {
                    LOG_ERROR("[Environment] get_stride_from_motion=true but stride attribute not found in HDF file");
                    exit(-1);
                }
            } else {
                LOG_ERROR("[Environment] get_stride_from_motion=true but motion is not HDF format");
                exit(-1);
            }
        }
    }

    // === Two-level controller ===
    if (isTwoLevelController()) {
        Character *character = mCharacter;
        MuscleNN nn = make_muscle_nn(character->getNumMuscleRelatedDof(), getNumActuatorAction(), character->getNumMuscles(), mUseCascading, true);
        mController->setMuscleNN(nn);
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

    // === Termination reward ===
    if (env["reward"] && env["reward"]["terminated"]) {
        mRewardConfig.terminated_reward = env["reward"]["terminated"].as<double>(0.0);
    }

    // === DeepMimic/ScaDiver imitation coefficients ===
    if (env["reward"]) {
        auto reward = env["reward"];
        if (reward["ee_weight"])
            mRewardConfig.ee_weight = reward["ee_weight"].as<double>(40.0);
        if (reward["pos_weight"])
            mRewardConfig.pos_weight = reward["pos_weight"].as<double>(20.0);
        if (reward["vel_weight"])
            mRewardConfig.vel_weight = reward["vel_weight"].as<double>(10.0);
        if (reward["com_weight"])
            mRewardConfig.com_weight = reward["com_weight"].as<double>(10.0);
        if (reward["ankle_weight"])
            mRewardConfig.ankle_weight = reward["ankle_weight"].as<double>(-1.0);
        if (reward["num_ref_in_state"])
            mRewardConfig.num_ref_in_state = reward["num_ref_in_state"].as<int>(1);
        if (reward["include_ref_velocity"])
            mRewardConfig.include_ref_velocity = reward["include_ref_velocity"].as<bool>(true);
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
            auto step = loco["step"];
            if (step["weight"])
                mRewardConfig.step_weight = step["weight"].as<double>(2.0);
            if (step["clip"])
                mRewardConfig.step_clip = step["clip"].as<double>(0.075);
        } else if (loco["step_weight"]) {
            mRewardConfig.step_weight = loco["step_weight"].as<double>(2.0);
        }

        // Parse avg_vel configuration (hierarchical or flat for backward compatibility)
        if (loco["avg_vel"]) {
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

        if (disc["upper_body"]) {
            auto ub = disc["upper_body"];
            if (ub.IsMap()) {
                mDiscConfig.upper_body = ub["enabled"].as<bool>(false);
                mDiscConfig.upper_body_scale = ub["scale"].as<double>(1.0);
            } else {
                mDiscConfig.upper_body = ub.as<bool>(false);
            }
        }

        if (mDiscConfig.enabled && mUseMuscle) {
            if (mDiscConfig.upper_body) {
                int rootDof = mCharacter->getSkeleton()->getRootJoint()->getNumDofs();
                int lowerBodyDof = 18;
                mUpperBodyDim = mCharacter->getSkeleton()->getNumDofs() - rootDof - lowerBodyDof;
            }

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

Environment::~Environment()
{
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
        if (mController->hasLoadedMuscleNN())
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

    // Virtual root force: compute current kp and update ref position
    if (mVfKpStart > 0.0) {
        // Linear decay: kp_start @ t=0 → kp_end @ t=Horizon
        double t_norm = std::clamp(static_cast<double>(mSimulationStep) / static_cast<double>(mHorizon), 0.0, 1.0);
        double kp_end = mVfKpStart * mVfDiscountRate * mVfDiscountRate;
        double current_kp = mVfKpStart * (1.0 - t_norm) + kp_end * t_norm;
        mController->setVirtualRootForceKp(current_kp);

        // Extract root pose from mRefPose in WORLD frame
        // FreeJoint DOFs are body-local, need to convert to world frame
        Eigen::Isometry3d ref_transform = FreeJoint::convertToTransform(mRefPose.head<6>());
        Eigen::Vector3d ref_root_pos = ref_transform.translation();
        Eigen::Matrix3d ref_root_rot = ref_transform.linear();
        mController->setVirtualRootRefPosition(ref_root_pos);
        mController->setVirtualRootRefOrientation(ref_root_rot);
    } else {
        mController->setVirtualRootForceKp(0.0);
    }

    ActuatorType actType = mController->getActuatorType();
    if (actType == pd || actType == mass || actType == mass_lower)
    {
        Eigen::VectorXd action = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
        action.tail(actuatorAction.rows()) = actuatorAction;
        if (isMirror()) action = mCharacter->getMirrorPosition(action);
        action = mCharacter->addPositions(mRefPose, action);
        mPDTarget = action;  // Store for muscleStep()
    }
    else if (actType == tor)
    {
        Eigen::VectorXd torque = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
        torque.tail(actuatorAction.rows()) = actuatorAction;
        if (isMirror()) torque = mCharacter->getMirrorPosition(torque);
        mPendingTorque = torque;  // Store for muscleStep()
    }

    mSimulationStep++;
}

Eigen::VectorXd Environment::getFutureRefPose(int future_step)
{
    double dTime = 1.0 / mControlHz;
    Eigen::VectorXd pose;

    if (mRewardType == gaitnet) {
        double dPhase = dTime / (mMotion->getMaxTime() / mCadence);
        double basePhase = mGaitPhase->getAdaptivePhase();
        double targetPhase = basePhase + dPhase * future_step;

        pose = mMotion->getTargetPose(targetPhase);
    } else if (mRewardType == deepmimic || mRewardType == scadiver) {
        double dPhase = dTime / mMotion->getMaxTime();
        double basePhase = mGaitPhase->getAdaptivePhase();
        double targetPhase = basePhase + dPhase * future_step;
        double baseTime = mGaitPhase->getAdaptiveTime();
        double targetTime = baseTime + dTime * future_step;

        int cycleCount = mGaitPhase->getAdaptiveCycleCount(targetTime);

        pose = mMotion->getTargetPose(targetPhase);

        if (cycleCount > 0) {
            pose[3] += mMotion->getCycleDistance()[0] * cycleCount;
            pose[5] += mMotion->getCycleDistance()[2] * cycleCount;
        }
    }

    return pose;
}

void Environment::updateTargetPosAndVel(bool currentStep)
{
    double dTime = 1.0 / mControlHz;

    // gaitnet uses mCadence to scale cycle time
    double cycleTime = (mRewardType == gaitnet)
        ? mMotion->getMaxTime() / mCadence
        : mMotion->getMaxTime();
    double dPhase = dTime / cycleTime;

    double adaptPhase = (currentStep ? 0.0 : dPhase) + mGaitPhase->getAdaptivePhase();
    double adaptiveTime = (currentStep ? 0.0 : dTime) + mGaitPhase->getAdaptiveTime();

    int adaptCycleCount = mGaitPhase->getAdaptiveCycleCount(adaptiveTime);
    int nextAdaptCycleCount = mGaitPhase->getAdaptiveCycleCount(adaptiveTime + dTime);

    mRefPose = mMotion->getTargetPose(adaptPhase);
    Eigen::VectorXd nextPose = mMotion->getTargetPose(adaptPhase + dPhase);

    if (nextAdaptCycleCount > 0) {
        mRefPose[3] += mMotion->getCycleDistance()[0] * adaptCycleCount;
        mRefPose[5] += mMotion->getCycleDistance()[2] * adaptCycleCount;
        nextPose[3] += mMotion->getCycleDistance()[0] * nextAdaptCycleCount;
        nextPose[5] += mMotion->getCycleDistance()[2] * nextAdaptCycleCount;
    }

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
    // NOTE: This function assumes checkTerminated() has been called before it.
    // The calling order in postStep() must be: checkTerminated() -> checkTruncated() -> calcReward()

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

        double r_p, r_v, r_ee, r_com, r_metabolic, r_ankle;
        // Skip reward term if weight is negative (set to 1.0 for neutral effect in multiplicative rewards)
        r_ee = mRewardConfig.ee_weight < 0 ? 1.0 : exp(-mRewardConfig.ee_weight * ee_diff.squaredNorm() / ee_diff.rows());
        r_p = mRewardConfig.pos_weight < 0 ? 1.0 : exp(-mRewardConfig.pos_weight * (mImitMask * pos_diff.array().square()).sum());
        r_v = mRewardConfig.vel_weight < 0 ? 1.0 : exp(-mRewardConfig.vel_weight * (mImitMask * vel_diff.array().square()).sum());
        r_com = mRewardConfig.com_weight < 0 ? 1.0 : exp(-mRewardConfig.com_weight * com_diff.squaredNorm() / com_diff.rows());
        r_metabolic = getEnergyReward();

        // Ankle-specific position reward (TalusR and TalusL only)
        r_ankle = 1.0;
        if (mRewardConfig.ankle_weight >= 0) {
            double ankle_sq_sum = 0.0;
            auto talusR = skel->getJoint("TalusR");
            auto talusL = skel->getJoint("TalusL");
            if (talusR) {
                int idx = talusR->getIndexInSkeleton(0);
                for (size_t i = 0; i < talusR->getNumDofs(); i++)
                    ankle_sq_sum += pos_diff[idx + i] * pos_diff[idx + i];
            }
            if (talusL) {
                int idx = talusL->getIndexInSkeleton(0);
                for (size_t i = 0; i < talusL->getNumDofs(); i++)
                    ankle_sq_sum += pos_diff[idx + i] * pos_diff[idx + i];
            }
            r_ankle = exp(-mRewardConfig.ankle_weight * ankle_sq_sum);
        }

        if (mRewardType == deepmimic) r = (w_p * r_p + w_v * r_v + w_com * r_com + w_ee * r_ee + w_metabolic * r_metabolic) * r_ankle;
        else if (mRewardType == scadiver) r = (0.1 + 0.9 * r_p) * (0.1 + 0.9 * r_v) * (0.1 + 0.9 * r_com) * (0.1 + 0.9 * r_ee) * (0.1 + 0.9 * r_metabolic) * (0.1 + 0.9 * r_ankle);

        // Log individual rewards to mInfoMap (for TensorBoard)
        mInfoMap["r_ee"] = r_ee;
        mInfoMap["r_p"] = r_p;
        mInfoMap["r_v"] = r_v;
        mInfoMap["r_com"] = r_com;
        mInfoMap["r_metabolic"] = r_metabolic;
        mInfoMap["r_ankle"] = r_ankle;
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

    if (mRewardConfig.clip_step > 0 && mSimulationStep < mRewardConfig.clip_step) {
        r = std::min(r, mRewardConfig.clip_value);
    }

    // Add termination reward (penalty) if terminated
    // NOTE: Requires checkTerminated() to be called before calcReward()
    double r_terminated = 0.0;
    if (mInfoMap["terminated"] > 0.5 && mRewardConfig.terminated_reward != 0.0) {
        r_terminated = mRewardConfig.terminated_reward;
        r += r_terminated;
    }
    mInfoMap["r_terminated"] = r_terminated;

    // Always store total reward
    mInfoMap.insert(std::make_pair("r", r));

    return r;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Environment::buildPVState()
{
    if (mLocalState) {
        return buildLocalPVState();
    }

    auto skel = mCharacter->getSkeleton();
    int num_body_nodes = skel->getNumBodyNodes();
    Eigen::VectorXd p, v;
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

    return {p, v};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Environment::buildLocalPVState()
{
    auto skel = mCharacter->getSkeleton();
    int num_body_nodes = skel->getNumBodyNodes();
    Eigen::VectorXd p, v;
    p.resize(num_body_nodes * 3 + num_body_nodes * 6);
    v.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);
    p.setZero();
    v.setZero();

    // Get character's facing direction (yaw only rotation)
    Eigen::Matrix3d root_rot = skel->getBodyNode(0)->getTransform().linear();
    Eigen::Vector3d forward = root_rot.col(2);  // Z-axis is forward
    forward[1] = 0;  // Project onto ground plane
    forward.normalize();

    // Build yaw-only rotation matrix (rotate to align with world Z)
    double yaw = atan2(forward[0], forward[2]);
    Eigen::Matrix3d local_rot;
    local_rot = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitY());

    Eigen::Vector3d com = skel->getCOM();

    for (int i = 0; i < num_body_nodes; i++)
    {
        // Position relative to COM, rotated to local frame
        Eigen::Vector3d pos = skel->getBodyNode(i)->getCOM() - com;
        p.segment<3>(i * 3) = local_rot * pos;

        // Orientation in local frame
        Eigen::Matrix3d body_rot = skel->getBodyNode(i)->getTransform().linear();
        Eigen::Matrix3d local_body_rot = local_rot * body_rot;
        p.segment<6>(num_body_nodes * 3 + 6 * i) << local_body_rot(0, 0), local_body_rot(0, 1), local_body_rot(0, 2),
            local_body_rot(1, 0), local_body_rot(1, 1), local_body_rot(1, 2);

        // Linear velocity relative to COM velocity, rotated to local frame
        Eigen::Vector3d lin_vel = skel->getBodyNode(i)->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
        v.segment<3>(i * 3) = local_rot * lin_vel;

        // Angular velocity rotated to local frame
        Eigen::Vector3d ang_vel = skel->getBodyNode(i)->getAngularVelocity();
        v.segment<3>((num_body_nodes + 1) * 3 + i * 3) = 0.1 * (local_rot * ang_vel);
    }

    // COM velocity in local frame
    v.segment<3>(num_body_nodes * 3) = local_rot * skel->getCOMLinearVelocity();

    return {p, v};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Environment::getProjState(const Eigen::VectorXd minV, const Eigen::VectorXd maxV)
{
    Eigen::VectorXd state, joint_state;
    if (mRewardType == gaitnet) {
        auto skel = mCharacter->getSkeleton();
        Eigen::Vector3d com = skel->getCOM();
        com[0] = 0;
        com[2] = 0;
    
        auto [p, v] = buildPVState();
    
        Eigen::VectorXd phase = Eigen::VectorXd::Zero(1 + (mPhaseDisplacementScale > 0.0 ? 1 : 0));
        phase[0] = mGaitPhase->getAdaptivePhase();
        if (mPhaseDisplacementScale > 0.0) phase[1] = mGaitPhase->getAdaptivePhase();
        if (isMirror()) for (int i = 0; i < phase.rows(); i++) phase[i] = (phase[i] + 0.5) - (int)(phase[i] + 0.5);
    
        Eigen::VectorXd step_state = Eigen::VectorXd::Zero(0);
        step_state.resize(1);
        step_state[0] = getNextTargetFootStep()[2] - mCharacter->getSkeleton()->getCOM()[2];
    
        Eigen::VectorXd curParamState = getParamState();
        Eigen::VectorXd projState = Eigen::VectorXd::Zero(mNumParamState);
        std::vector<int> projectedParamIdx;
    
        for (int i = 0; i < minV.rows(); i++) if (abs(minV[i] - maxV[i]) > 1E-3) projectedParamIdx.push_back(i);
        for (int i = 0; i < projState.rows(); i++) projState[i] = dart::math::clip(curParamState[i], minV[i], maxV[i]);
        setParamState(projState, true);
    
        joint_state = getJointState(isMirror());
    
        // Parameter State
        Eigen::VectorXd param_state = (mUseNormalizedParamState ? getNormalizedParamState(minV, maxV, isMirror()) : getParamState(isMirror()));
        Eigen::VectorXd proj_param_state = Eigen::VectorXd::Zero(projectedParamIdx.size());
        for (int i = 0; i < projectedParamIdx.size(); i++) proj_param_state[i] = param_state[projectedParamIdx[i]];
    
        setParamState(curParamState, true);
    
        // Integration of all states
        state = Eigen::VectorXd::Zero(com.rows() + p.rows() + v.rows() + phase.rows() + step_state.rows() + joint_state.rows() + proj_param_state.rows());
        state << com, p, v, phase, step_state, joint_state, proj_param_state;
    } else if (mRewardType == deepmimic || mRewardType == scadiver) {
        auto skel = mCharacter->getSkeleton();
        auto [p, v] = buildPVState();
        const Eigen::VectorXd cur_p = skel->getPositions();
        const Eigen::VectorXd cur_v = skel->getVelocities();
        double dTime = 1.0 / mControlHz;

        // Get future reference poses
        std::vector<Eigen::VectorXd> future_poses;
        int num_poses_needed = mRewardConfig.num_ref_in_state;
        if (mRewardConfig.include_ref_velocity) {
            num_poses_needed += 1;  // Need one extra for velocity computation
        }
        for (int step = 1; step <= num_poses_needed; step++) {
            future_poses.push_back(getFutureRefPose(step));
        }

        // Build reference states for each future step
        std::vector<Eigen::VectorXd> ref_p_states;
        std::vector<Eigen::VectorXd> ref_v_states;

        for (int i = 0; i < mRewardConfig.num_ref_in_state; i++) {
            Eigen::VectorXd ref_pose = future_poses[i];
            skel->setPositions(ref_pose);

            if (mRewardConfig.include_ref_velocity) {
                Eigen::VectorXd next_pose = future_poses[i + 1];
                Eigen::VectorXd ref_vel = skel->getPositionDifferences(next_pose, ref_pose) / dTime;
                skel->setVelocities(ref_vel);
            }

            auto [p_ref, v_ref] = buildPVState();
            ref_p_states.push_back(p_ref);
            if (mRewardConfig.include_ref_velocity) {
                ref_v_states.push_back(v_ref);
            }
        }

        skel->setPositions(cur_p);
        skel->setVelocities(cur_v);

        // Compute rel_root only for first reference (step 1)
        Eigen::Isometry3d cur_root = FreeJoint::convertToTransform(cur_p.head(6));
        Eigen::Isometry3d ref_root = FreeJoint::convertToTransform(future_poses[0].head(6));
        Eigen::Isometry3d rel_tx = ref_root.inverse() * cur_root;

        // Relative position (3D) + orientation 6D (first two columns of rotation matrix)
        Eigen::VectorXd rel_root = Eigen::VectorXd::Zero(9);
        rel_root.head<3>() = rel_tx.translation();
        rel_root.segment<3>(3) << rel_tx.linear()(0, 0), rel_tx.linear()(1, 0), rel_tx.linear()(2, 0);
        rel_root.segment<3>(6) << rel_tx.linear()(0, 1), rel_tx.linear()(1, 1), rel_tx.linear()(2, 1);

        // Calculate total state size
        int ref_p_total = 0, ref_v_total = 0;
        for (const auto& rp : ref_p_states) ref_p_total += rp.rows();
        for (const auto& rv : ref_v_states) ref_v_total += rv.rows();

        state = Eigen::VectorXd::Zero(p.rows() + v.rows() + ref_p_total + ref_v_total + rel_root.rows());

        // Concatenate: [p, v, p_ref1, (v_ref1), p_ref2, (v_ref2), ..., rel_root]
        int offset = 0;
        state.segment(offset, p.rows()) = p; offset += p.rows();
        state.segment(offset, v.rows()) = v; offset += v.rows();
        for (size_t i = 0; i < ref_p_states.size(); i++) {
            state.segment(offset, ref_p_states[i].rows()) = ref_p_states[i];
            offset += ref_p_states[i].rows();
            if (mRewardConfig.include_ref_velocity) {
                state.segment(offset, ref_v_states[i].rows()) = ref_v_states[i];
                offset += ref_v_states[i].rows();
            }
        }
        state.segment(offset, rel_root.rows()) = rel_root;
    }
    
    return {state, joint_state};
}

Eigen::VectorXd Environment::getState()
{
    auto [state, joint_state] = getProjState(mParamMin, mParamMax);
    mState = state;
    mJointState = joint_state;
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

// Note: calcActivation() has been removed - logic moved to Controller::step()

void Environment::postMuscleStep()
{
    mSimulationCount++;
    mGaitPhase->step();

    if (mGaitPhase->isGaitCycleComplete()) {
        mKneeLoadingMaxCycle = mCharacter->getKneeLoadingMax();
        mCharacter->resetKneeLoadingMax();
    }

    if (mDiscConfig.enabled) {
        Eigen::VectorXf disc_obs = getDiscObs();
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
    ActuatorType actType = mController->getActuatorType();

    // Build input for Controller
    ControllerInput input;
    ControllerOutput output;
    input.pdTarget = mPDTarget;
    input.isMirror = isMirror();
    input.includeJtPinSPD = mCharacter->getIncludeJtPinSPD();

    if (actType == mass || actType == mass_lower) {
        // Copy muscle tuple to local (cache invalidated by setActivations/applyMuscleForces)
        MuscleTuple mt = mCharacter->getMuscleTuple(isMirror());
        input.muscleTuple = &mt;

        output = mController->step(input);

        // Apply muscle forces
        mCharacter->setActivations(output.activations.cast<double>());
        mCharacter->applyMuscleForces();
        
        // Apply remaining torque (upper body for mass_lower, none for mass)
        if (actType == mass_lower) {
            mCharacter->addTorque(output.torque);
        }

        // Training data sampling
        if (thread_safe_uniform(0.0, 1.0) < 1.0 / static_cast<double>(mNumSubSteps) || !mTupleFilled) {
            mRandomMuscleTuple = mt;
            mTupleFilled = true;
        }

        // Accumulate mean activation for tensorboard logging
        mMeanActivation += output.activations.cwiseAbs().mean();

    } else if (actType == tor) {
        input.torque = mPendingTorque;
        output = mController->step(input);
        mCharacter->applyTorque(output.torque);

    } else if (actType == pd) {
        output = mController->step(input);
        mCharacter->applyTorque(output.torque);
    }

    // Apply virtual root force computed in world frame SPD (position only, no orientation)
    if (mVfKpStart > 0.0) {
        Eigen::Vector3d force = output.torque.segment<3>(3);
        if (force.squaredNorm() > 1e-12) {
            auto rootBody = mCharacter->getSkeleton()->getRootBodyNode();
            rootBody->addExtForce(force, Eigen::Vector3d::Zero(), false, true);  // world frame
        }
    }

    if (mNoiseInjector) mNoiseInjector->step(mCharacter);
    mCharacter->step();  // Metabolic tracking only
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
    
    // Add mean_activation to info map for tensorboard logging (always enabled)
    // Average over all substeps
    mInfoMap["mean_activation"] = mMeanActivation / static_cast<double>(mNumSubSteps);
    
    // Reset disc_obs filled flag for next control step
    mDiscObsFilled = false;
    
    // Check and cache termination/truncation status
    // NOTE: checkTerminated() must be called before calcReward() for termination reward
    checkTerminated();
    checkTruncated();
    mReward = calcReward();
}

void Environment::reset(double phase)
{
    // Clear info map (includes termination/truncation status)
    mInfoMap.clear();

    mTupleFilled = false;
    mDiscObsFilled = false;
    mMeanActivation = 0.0;
    mSimulationStep = 0;
    mSimulationCount = 0;
    mKneeLoadingMaxCycle = 0.0;

    // Reset virtual force state
    mController->setVirtualRootForceKp(0.0);
    mController->setVirtualRootRefPosition(Eigen::Vector3d::Zero());

    // Reset Initial Time
    double time = 0.0;
    if (phase >= 0.0 && phase <= 1.0) {
        // Use specified phase (0.0 to 1.0)
        time = phase * mMotion->getMaxTime() / mCadence;
    }
    else if (mRewardType == deepmimic || mRewardType == scadiver) {
        time = thread_safe_uniform(1E-2, mMotion->getMaxTime() - 1E-2);
    }
    else if (mRewardType == gaitnet)
    {
        time = (thread_safe_uniform(0.0, 1.0) > 0.5 ? 0.5 : 0.0) + thread_safe_uniform(-0.1, 0.1);
        time *= mMotion->getMaxTime() / mCadence;
        time = abs(time);
    }    
    
    // Collision Detector Reset
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->getConstraintSolver()->clearLastCollisionResult();

    mWorld->setTime(time);
    mGaitPhase->reset(time);

    // Reset Skeletons
    mCharacter->getSkeleton()->setPositions(mCharacter->getSkeleton()->getPositions().setZero());
    mCharacter->getSkeleton()->setVelocities(mCharacter->getSkeleton()->getVelocities().setZero());

    mCharacter->getSkeleton()->clearConstraintImpulses();
    mCharacter->getSkeleton()->clearInternalForces();
    mCharacter->getSkeleton()->clearExternalForces();

    mGaitPhase->setPhaseAction(0.0);

    // Initial Pose Setting
    updateTargetPosAndVel(true);
    
    if(mRewardType == gaitnet)
    {
        // mTargetPositions.segment(6, 18) *= (mStride * (mCharacter->getGlobalRatio()));
        mTargetVelocities.head(24) *= (mStride * (mCharacter->getGlobalRatio()));
    }
    
    mCharacter->getSkeleton()->setPositions(mRefPose);

    // Zero ankle 0 DOF on reset if flag is set
    if (mZeroAnkle0OnReset) {
        auto skel = mCharacter->getSkeleton();
        Eigen::VectorXd pos = skel->getPositions();
        auto talusR = skel->getJoint("TalusR");
        auto talusL = skel->getJoint("TalusL");
        if (talusR) pos[talusR->getIndexInSkeleton(0)] = 0.0;
        if (talusL) pos[talusL->getIndexInSkeleton(0)] = 0.0;
        skel->setPositions(pos);
    }
    mGaitPhase->resetFootPos();

    mCharacter->getSkeleton()->setVelocities(mTargetVelocities);

    updateTargetPosAndVel();

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
    
    mCharacter->heightCalibration(mWorld);

    // Set termination Y limit to 60% of initial character height
    double initialHeight = mCharacter->getSkeleton()->getCOM()[1];
    mLimitY = 0.6 * initialHeight;

    // Pose In ROM
    Eigen::VectorXd cur_pos = mCharacter->getSkeleton()->getPositions();
    Eigen::VectorXd rom_min = mCharacter->getSkeleton()->getPositionLowerLimits();
    Eigen::VectorXd rom_max = mCharacter->getSkeleton()->getPositionUpperLimits();
    cur_pos = cur_pos.cwiseMax(rom_min).cwiseMin(rom_max);
    mCharacter->getSkeleton()->setPositions(cur_pos);

    mCharacter->setZeroForces();
    mPendingTorque = Eigen::VectorXd::Zero(mCharacter->getSkeleton()->getNumDofs());
    if (mUseMuscle) {
        mCharacter->setActivations(mCharacter->getActivations().setZero());
        mCharacter->resetStep();
    }

    mCharacter->clearLogs();

    mDragStartX = mCharacter->getSkeleton()->getCOM()[0];

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
    ActuatorType actType = mController->getActuatorType();

    // For torque-based actuators (pd, tor), use torque energy only
    if (actType == pd || actType == tor) {
        energy = mCharacter->getTorqueEnergy();
    }
    // For muscle-based actuators, check separate flag
    else if (mRewardConfig.flags & REWARD_SEP_TORQUE_ENERGY) {
        energy = mCharacter->getMetabolicEnergy();
    }
    else {
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
    double stride_duration = mMotion->getMaxTime() / mCadence;
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
    return joint_state * 0.008;
}

void Environment::setParamState(Eigen::VectorXd _param_state, bool onlyMuscle)
{
    int idx = 0;
    // skeleton parameter
    if (!onlyMuscle)
    {
        std::vector<std::pair<std::string, double>> skel_info;
        for (auto name : mParamName)
        {
            // gait parameter
            if (name.find("stride") != std::string::npos) mStride = _param_state[idx];
            if (name.find("cadence") != std::string::npos) mCadence = _param_state[idx];
            if (name.find("skeleton") != std::string::npos) skel_info.push_back(std::make_pair((name.substr(9)), _param_state[idx]));
            if (name.find("torsion") != std::string::npos) skel_info.push_back(std::make_pair(name, _param_state[idx]));

            idx++;
        }
        // Only call setSkelParam if there are skeleton parameters to apply
        if (!skel_info.empty()) {
            mCharacter->setSkelParam(skel_info);
        }

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

void Environment::setNormalizedParamState(Eigen::VectorXd _param_state, bool onlyMuscle)
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
        mCharacter->setSkelParam(skel_info);

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

void Environment::maskImitJoint(const std::string& jointName)
{
    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(jointName);
    if (joint && joint->getNumDofs() > 0) {
        // Mask ALL DOFs of this joint (handles multi-DOF joints like ball joints)
        for (size_t i = 0; i < joint->getNumDofs(); i++) {
            mImitMask[joint->getIndexInSkeleton(i)] = 0.0;
        }
    }
}

void Environment::demaskImitJoint(const std::string& jointName)
{
    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(jointName);
    if (joint && joint->getNumDofs() > 0) {
        // Demask ALL DOFs of this joint (restore to active)
        for (size_t i = 0; i < joint->getNumDofs(); i++) {
            mImitMask[joint->getIndexInSkeleton(i)] = 1.0;
        }
    }
}

void Environment::createNoiseInjector(const std::string& config_path)
{
    if (mNoiseInjector) {
        LOG_WARN("[Environment] NoiseInjector already exists, recreating with new config");
    }

    mNoiseInjector = std::make_unique<NoiseInjector>(config_path, mWorld->getTimeStep());
    LOG_INFO("[Environment] NoiseInjector created" << (config_path.empty() ? " with default parameters" : " from config: " + config_path));
}

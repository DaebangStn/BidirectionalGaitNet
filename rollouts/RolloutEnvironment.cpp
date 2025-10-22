#include "RolloutEnvironment.h"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>

RolloutEnvironment::RolloutEnvironment(const std::string& metadata_path) {
    mEnv.initialize(metadata_path);
    mEnv.reset();
}

RolloutEnvironment::~RolloutEnvironment() = default;

void RolloutEnvironment::LoadRecordConfig(const std::string& yaml_path) {
    mRecordConfig = RecordConfig::LoadFromYAML(yaml_path);

    // Apply MetabolicType override if enabled
    if (mRecordConfig.metabolic.enabled) {
        const std::string& type = mRecordConfig.metabolic.type;
        if (type == "LEGACY") {
            mEnv.getCharacter()->setMetabolicType(LEGACY);
        } else if (type == "A") {
            mEnv.getCharacter()->setMetabolicType(A);
        } else if (type == "A2") {
            mEnv.getCharacter()->setMetabolicType(A2);
        } else if (type == "MA") {
            mEnv.getCharacter()->setMetabolicType(MA);
        } else if (type == "MA2") {
            mEnv.getCharacter()->setMetabolicType(MA2);
        } else {
            std::cerr << "Warning: Unknown MetabolicType '" << type
                      << "', using LEGACY" << std::endl;
            mEnv.getCharacter()->setMetabolicType(LEGACY);
        }
        std::cout << "MetabolicType set to: " << type << std::endl;
    }

    // Also load target cycles from YAML if present
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        if (root["sample"] && root["sample"]["cycle"]) {
            mTargetCycles = root["sample"]["cycle"].as<int>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Note: Using default target cycles: " << mTargetCycles << std::endl;
    }
}

void RolloutEnvironment::Reset(double phase) {
    mEnv.reset(phase);
}

Eigen::VectorXd RolloutEnvironment::GetState() {
    return mEnv.getState();
}

void RolloutEnvironment::SetAction(const Eigen::VectorXd& action) {
    mEnv.setAction(action);
}

void RolloutEnvironment::Step(RolloutRecord* record) {
    // Call environment step (no pGraphData)
    for (int i = 0; i < mEnv.getNumSubSteps(); i++) {
        mEnv.muscleStep();
        RecordStep(record);
    }
    mEnv.postStep();
}

void RolloutEnvironment::SetMuscleNetworkWeight(py::object weights) {
    mEnv.setMuscleNetworkWeight(weights);
}

void RolloutEnvironment::RecordStep(RolloutRecord* record) {
    std::unordered_map<std::string, float> data;

    // Basic fields (always recorded)
    data["step"] = mEnv.getSimulationCount();
    data["time"] = mEnv.getWorldTime();
    data["phase"] = mEnv.getNormalizedPhase();
    data["cycle"] = mEnv.getWorldPhaseCount();

    auto skel = mEnv.getCharacter()->getSkeleton();

    // Contact and GRF fields
    if (mRecordConfig.foot.enabled) {
        Eigen::Vector2i contact = mEnv.getIsContact();
        Eigen::Vector2d grf = mEnv.getFootGRF();

        if (mRecordConfig.foot.contact_left) {
            data["contact/left"] = static_cast<float>(contact[0]);
        }
        if (mRecordConfig.foot.contact_right) {
            data["contact/right"] = static_cast<float>(contact[1]);
        }
        if (mRecordConfig.foot.grf_left) {
            data["grf/left"] = grf[0];
        }
        if (mRecordConfig.foot.grf_right) {
            data["grf/right"] = grf[1];
        }
    }

    // Kinematics fields
    if (mRecordConfig.kinematics.enabled) {
        // All joint positions as a single vector
        if (mRecordConfig.kinematics.all) {
            record->addVector("motions", mEnv.getSimulationCount() - 1, skel->getPositions().cast<float>());
        }

        // Root position
        if (mRecordConfig.kinematics.root) {
            auto root_body = skel->getRootBodyNode();
            Eigen::Vector3d root_pos = root_body->getCOM();
            data["root/x"] = root_pos[0];
            data["root/y"] = root_pos[1];
            data["root/z"] = root_pos[2];
        }

        // Angle fields
        if (mRecordConfig.kinematics.angle.enabled) {
            if (mRecordConfig.kinematics.angle.hip) {
                float angle = skel->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
                data["angle/HipR"] = -angle;  // Negate as per Environment.cpp:1035
            }
            if (mRecordConfig.kinematics.angle.hip_ir) {
                float angle = skel->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
                data["angle/HipIRR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.hip_ab) {
                float angle = skel->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
                data["angle/HipAbR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.knee) {
                float angle = skel->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
                data["angle/KneeR"] = angle;  // No negation
            }
            if (mRecordConfig.kinematics.angle.ankle) {
                float angle = skel->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
                data["angle/AnkleR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_tilt) {
                float angle = skel->getJoint("Pelvis")->getPosition(0) * 180.0 / M_PI;
                data["angle/Tilt"] = angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_rotation) {
                float angle = skel->getJoint("Pelvis")->getPosition(1) * 180.0 / M_PI;
                data["angle/Rotation"] = angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_obliquity) {
                float angle = skel->getJoint("Pelvis")->getPosition(2) * 180.0 / M_PI;
                data["angle/Obliquity"] = angle;
            }
        }

        // Angular velocity fields
        if (mRecordConfig.kinematics.anvel.enabled) {
            if (mRecordConfig.kinematics.anvel.hip) {
                data["anvel/HipR"] = skel->getJoint("FemurR")->getVelocity(0);
            }
            if (mRecordConfig.kinematics.anvel.knee) {
                data["anvel/KneeR"] = skel->getJoint("TibiaR")->getVelocity(0);
            }
            if (mRecordConfig.kinematics.anvel.ankle) {
                data["anvel/AnkleR"] = skel->getJoint("TalusR")->getVelocity(0);
            }
        }
    }

    // Metabolic energy recording
    if (mRecordConfig.metabolic.enabled) {
        float stepEnergy = mEnv.getCharacter()->getMetabolicStepEnergy();
        data["metabolic/step_energy"] = stepEnergy;
    }

    record->add(mEnv.getSimulationCount() - 1, data);
}

int RolloutEnvironment::GetCycleCount() {
    return mEnv.getWorldPhaseCount();
}

int RolloutEnvironment::IsEndOfEpisode() {
    return mEnv.isEOE();
}

std::vector<std::string> RolloutEnvironment::GetRecordFields() const {
    int skeleton_dof = const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getNumDofs();
    return RolloutRecord::FieldsFromConfig(mRecordConfig, skeleton_dof);
}

int RolloutEnvironment::GetSkeletonDOF() const {
    return const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getNumDofs();
}

double RolloutEnvironment::GetMass() const {
    return const_cast<Environment&>(mEnv).getCharacter()->getSkeleton()->getMass();
}

int RolloutEnvironment::GetSimulationHz() {
    return mEnv.getSimulationHz();
}

int RolloutEnvironment::GetControlHz() {
    return mEnv.getControlHz();
}

double RolloutEnvironment::GetWorldTime() {
    return mEnv.getWorldTime();
}

double RolloutEnvironment::GetNormalizedPhase() {
    return mEnv.getNormalizedPhase();
}

int RolloutEnvironment::GetWorldPhaseCount() {
    return mEnv.getWorldPhaseCount();
}

Eigen::VectorXd RolloutEnvironment::InterpolatePose(const Eigen::VectorXd& pose1,
                                                     const Eigen::VectorXd& pose2,
                                                     double t,
                                                     bool extrapolate_root) {
    return mEnv.getCharacter()->interpolatePose(pose1, pose2, t, extrapolate_root);
}

void RolloutEnvironment::SetParameters(const std::map<std::string, double>& params) {
    if (params.empty()) {
        // Empty parameters - sample from default distribution (matches updateParamState)
        mEnv.updateParamState();
        return;
    }

    // Get parameter names from environment
    const std::vector<std::string>& param_names = mEnv.getParamName();

    // Get default parameter values (1.0 for skeleton scaling, etc.)
    Eigen::VectorXd param_state = mEnv.getParamDefault();

    // Fill in provided parameters (overriding defaults)
    for (size_t i = 0; i < param_names.size(); ++i) {
        auto it = params.find(param_names[i]);
        if (it != params.end()) {
            param_state[i] = it->second;
        }
    }

    // Set the parameter state (normalized=false, onlyMuscle=false, doOptimization=true)
    mEnv.setParamState(param_state, false, true);
}

std::vector<std::string> RolloutEnvironment::GetParameterNames() {
    const std::vector<std::string>& param_names = mEnv.getParamName();
    return std::vector<std::string>(param_names.begin(), param_names.end());
}

Eigen::VectorXd RolloutEnvironment::GetParamState(bool isMirror) {
    return mEnv.getParamState(isMirror);
}

Eigen::VectorXd RolloutEnvironment::GetParamDefault() {
    return mEnv.getParamDefault();
}

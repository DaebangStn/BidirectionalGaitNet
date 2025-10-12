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

void RolloutEnvironment::Reset() {
    mEnv.reset();
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
    std::unordered_map<std::string, double> data;

    // Basic fields (always recorded)
    data["step"] = mEnv.getSimulationStep();
    data["time"] = mEnv.getWorldTime();
    data["phase"] = mEnv.getNormalizedPhase();
    data["cycle"] = mEnv.getWorldPhaseCount();

    auto skel = mEnv.getCharacter(0)->getSkeleton();

    // Contact and GRF fields
    if (mRecordConfig.foot.enabled) {
        Eigen::Vector2i contact = mEnv.getIsContact();
        Eigen::Vector2d grf = mEnv.getFootGRF();

        if (mRecordConfig.foot.contact_left) {
            data["contact_left"] = static_cast<double>(contact[0]);
        }
        if (mRecordConfig.foot.contact_right) {
            data["contact_right"] = static_cast<double>(contact[1]);
        }
        if (mRecordConfig.foot.grf_left) {
            data["grf_left"] = grf[0];
        }
        if (mRecordConfig.foot.grf_right) {
            data["grf_right"] = grf[1];
        }
    }

    // Kinematics fields
    if (mRecordConfig.kinematics.enabled) {
        // All joint positions as a single vector
        if (mRecordConfig.kinematics.all) {
            record->addVector("motions", mEnv.getSimulationStep() - 1, skel->getPositions());
        }

        // Root position
        if (mRecordConfig.kinematics.root) {
            auto root_body = skel->getRootBodyNode();
            Eigen::Vector3d root_pos = root_body->getCOM();
            data["root_x"] = root_pos[0];
            data["root_y"] = root_pos[1];
            data["root_z"] = root_pos[2];
        }

        // Angle fields
        if (mRecordConfig.kinematics.angle.enabled) {
            if (mRecordConfig.kinematics.angle.hip) {
                double angle = skel->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
                data["angle_HipR"] = -angle;  // Negate as per Environment.cpp:1035
            }
            if (mRecordConfig.kinematics.angle.hip_ir) {
                double angle = skel->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
                data["angle_HipIRR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.hip_ab) {
                double angle = skel->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
                data["angle_HipAbR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.knee) {
                double angle = skel->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
                data["angle_KneeR"] = angle;  // No negation
            }
            if (mRecordConfig.kinematics.angle.ankle) {
                double angle = skel->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
                data["angle_AnkleR"] = -angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_tilt) {
                double angle = skel->getJoint("Pelvis")->getPosition(0) * 180.0 / M_PI;
                data["angle_Tilt"] = angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_rotation) {
                double angle = skel->getJoint("Pelvis")->getPosition(1) * 180.0 / M_PI;
                data["angle_Rotation"] = angle;
            }
            if (mRecordConfig.kinematics.angle.pelvic_obliquity) {
                double angle = skel->getJoint("Pelvis")->getPosition(2) * 180.0 / M_PI;
                data["angle_Obliquity"] = angle;
            }
        }

        // Angular velocity fields
        if (mRecordConfig.kinematics.anvel.enabled) {
            if (mRecordConfig.kinematics.anvel.hip) {
                data["anvel_HipR"] = skel->getJoint("FemurR")->getVelocity(0);
            }
            if (mRecordConfig.kinematics.anvel.knee) {
                data["anvel_KneeR"] = skel->getJoint("TibiaR")->getVelocity(0);
            }
            if (mRecordConfig.kinematics.anvel.ankle) {
                data["anvel_AnkleR"] = skel->getJoint("TalusR")->getVelocity(0);
            }
        }
    }

    record->add(mEnv.getSimulationStep() - 1, data);
}

int RolloutEnvironment::GetCycleCount() {
    return mEnv.getWorldPhaseCount();
}

int RolloutEnvironment::IsEndOfEpisode() {
    return mEnv.isEOE();
}

std::vector<std::string> RolloutEnvironment::GetRecordFields() const {
    int skeleton_dof = const_cast<Environment&>(mEnv).getCharacter(0)->getSkeleton()->getNumDofs();
    return RolloutRecord::FieldsFromConfig(mRecordConfig, skeleton_dof);
}

int RolloutEnvironment::GetSkeletonDOF() const {
    return const_cast<Environment&>(mEnv).getCharacter(0)->getSkeleton()->getNumDofs();
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

void RolloutEnvironment::SetParameters(const std::map<std::string, double>& params) {
    if (params.empty()) {
        // Empty parameters - sample from default distribution (matches updateParamState)
        mEnv.updateParamState();
        return;
    }

    // Get parameter names from environment
    const std::vector<std::string>& param_names = mEnv.getParamName();

    // Create parameter vector with default values (0.0)
    Eigen::VectorXd param_state = Eigen::VectorXd::Zero(param_names.size());

    // Fill in provided parameters
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

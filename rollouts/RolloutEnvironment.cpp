#include "RolloutEnvironment.h"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>

RolloutEnvironment::RolloutEnvironment(const std::string& metadata_path) {
    mEnv.initialize(metadata_path);
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
    mSimulationStepCount = 0;
}

Eigen::VectorXd RolloutEnvironment::GetState() {
    return mEnv.getState();
}

void RolloutEnvironment::SetAction(const Eigen::VectorXd& action) {
    mEnv.setAction(action);
}

void RolloutEnvironment::Step(RolloutRecord* record) {
    // Call environment step (no pGraphData)
    mEnv.step();
    mSimulationStepCount++;
    
    // Record data if record buffer provided
    if (record) {
        RecordStep(record);
    }
}

void RolloutEnvironment::RecordStep(RolloutRecord* record) {
    std::unordered_map<std::string, double> data;
    
    // Basic fields (always recorded)
    data["step"] = mSimulationStepCount;
    data["time"] = mEnv.getWorldTime();
    data["phase"] = mEnv.getNormalizedPhase();
    data["cycle"] = mEnv.getWorldPhaseCount();
    
    auto skel = mEnv.getCharacter(0)->getSkeleton();
    
    // Angle fields
    if (mRecordConfig.angle.enabled) {
        if (mRecordConfig.angle.hip) {
            double angle = skel->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
            data["angle_HipR"] = -angle;  // Negate as per Environment.cpp:1035
        }
        if (mRecordConfig.angle.hip_ir) {
            double angle = skel->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
            data["angle_HipIRR"] = -angle;
        }
        if (mRecordConfig.angle.hip_ab) {
            double angle = skel->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
            data["angle_HipAbR"] = -angle;
        }
        if (mRecordConfig.angle.knee) {
            double angle = skel->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
            data["angle_KneeR"] = angle;  // No negation
        }
        if (mRecordConfig.angle.ankle) {
            double angle = skel->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
            data["angle_AnkleR"] = -angle;
        }
        if (mRecordConfig.angle.pelvic_tilt) {
            double angle = skel->getJoint("Pelvis")->getPosition(0) * 180.0 / M_PI;
            data["angle_Tilt"] = angle;
        }
        if (mRecordConfig.angle.pelvic_rotation) {
            double angle = skel->getJoint("Pelvis")->getPosition(1) * 180.0 / M_PI;
            data["angle_Rotation"] = angle;
        }
        if (mRecordConfig.angle.pelvic_obliquity) {
            double angle = skel->getJoint("Pelvis")->getPosition(2) * 180.0 / M_PI;
            data["angle_Obliquity"] = angle;
        }
    }
    
    // Velocity fields
    if (mRecordConfig.velocity.enabled) {
        if (mRecordConfig.velocity.hip) {
            data["velocity_HipR"] = skel->getJoint("FemurR")->getVelocity(0);
        }
        if (mRecordConfig.velocity.knee) {
            data["velocity_KneeR"] = skel->getJoint("TibiaR")->getVelocity(0);
        }
        if (mRecordConfig.velocity.ankle) {
            data["velocity_AnkleR"] = skel->getJoint("TalusR")->getVelocity(0);
        }
    }
    
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
    
    record->add(mSimulationStepCount - 1, data);
}

int RolloutEnvironment::GetCycleCount() {
    return mEnv.getWorldPhaseCount();
}

int RolloutEnvironment::IsEndOfEpisode() {
    return mEnv.isEOE();
}

std::vector<std::string> RolloutEnvironment::GetRecordFields() const {
    return RolloutRecord::FieldsFromConfig(mRecordConfig);
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


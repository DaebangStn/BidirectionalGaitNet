#include "RolloutRecord.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <iostream>

RecordConfig RecordConfig::LoadFromYAML(const std::string& yaml_path) {
    RecordConfig config;
    
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        
        if (!root["record"]) {
            std::cerr << "Warning: No 'record' section in " << yaml_path << std::endl;
            return config;
        }
        
        YAML::Node record = root["record"];
        
        // Angle configuration
        if (record["angle"] && record["angle"]["enabled"].as<bool>(false)) {
            config.angle.enabled = true;
            config.angle.hip = record["angle"]["hip"].as<bool>(false);
            config.angle.hip_ir = record["angle"]["hip_ir"].as<bool>(false);
            config.angle.hip_ab = record["angle"]["hip_ab"].as<bool>(false);
            config.angle.knee = record["angle"]["knee"].as<bool>(false);
            config.angle.ankle = record["angle"]["ankle"].as<bool>(false);
            config.angle.pelvic_tilt = record["angle"]["pelvic_tilt"].as<bool>(false);
            config.angle.pelvic_rotation = record["angle"]["pelvic_rotation"].as<bool>(false);
            config.angle.pelvic_obliquity = record["angle"]["pelvic_obliquity"].as<bool>(false);
        }
        
        // Velocity configuration
        if (record["velocity"] && record["velocity"]["enabled"].as<bool>(false)) {
            config.velocity.enabled = true;
            config.velocity.hip = record["velocity"]["hip"].as<bool>(false);
            config.velocity.knee = record["velocity"]["knee"].as<bool>(false);
            config.velocity.ankle = record["velocity"]["ankle"].as<bool>(false);
        }
        
        // Foot configuration
        if (record["foot"].as<bool>(false)) {
            config.foot.enabled = true;
            config.foot.contact_left = true;
            config.foot.contact_right = true;
            config.foot.grf_left = true;
            config.foot.grf_right = true;
        }
        
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading record config: " << e.what() << std::endl;
    }
    
    return config;
}

std::vector<std::string> RolloutRecord::FieldsFromConfig(const RecordConfig& config) {
    std::vector<std::string> fields;
    
    // Basic fields (always included)
    fields.push_back("step");
    fields.push_back("time");
    fields.push_back("phase");
    fields.push_back("cycle");
    
    // Angle fields
    if (config.angle.enabled) {
        if (config.angle.hip) fields.push_back("angle_HipR");
        if (config.angle.hip_ir) fields.push_back("angle_HipIRR");
        if (config.angle.hip_ab) fields.push_back("angle_HipAbR");
        if (config.angle.knee) fields.push_back("angle_KneeR");
        if (config.angle.ankle) fields.push_back("angle_AnkleR");
        if (config.angle.pelvic_tilt) fields.push_back("angle_Tilt");
        if (config.angle.pelvic_rotation) fields.push_back("angle_Rotation");
        if (config.angle.pelvic_obliquity) fields.push_back("angle_Obliquity");
    }
    
    // Velocity fields
    if (config.velocity.enabled) {
        if (config.velocity.hip) fields.push_back("velocity_HipR");
        if (config.velocity.knee) fields.push_back("velocity_KneeR");
        if (config.velocity.ankle) fields.push_back("velocity_AnkleR");
    }
    
    // Foot contact/GRF fields
    if (config.foot.enabled) {
        if (config.foot.contact_left) fields.push_back("contact_left");
        if (config.foot.contact_right) fields.push_back("contact_right");
        if (config.foot.grf_left) fields.push_back("grf_left");
        if (config.foot.grf_right) fields.push_back("grf_right");
    }
    
    return fields;
}

RolloutRecord::RolloutRecord(const std::vector<std::string>& field_names)
    : mFieldNames(field_names), mNcol(field_names.size()), mNrow(0) {
    
    // Build field to index map
    for (size_t i = 0; i < field_names.size(); ++i) {
        mFieldToIdx[field_names[i]] = i;
    }
    
    // Initialize with initial chunk
    mData = Eigen::MatrixXd::Zero(DATA_CHUNK_SIZE, mNcol);
}

RolloutRecord::RolloutRecord(const RecordConfig& config)
    : RolloutRecord(FieldsFromConfig(config)) {}

void RolloutRecord::resize_if_needed(unsigned int requested_size) {
    if (requested_size > static_cast<unsigned int>(mData.rows())) {
        unsigned int old_rows = mData.rows();
        unsigned int new_rows = ((requested_size / DATA_CHUNK_SIZE) + 1) * DATA_CHUNK_SIZE;
        
        mData.conservativeResize(new_rows, Eigen::NoChange);
        mData.block(old_rows, 0, new_rows - old_rows, mNcol).setZero();
    }
}

void RolloutRecord::add(unsigned int sim_step, const std::unordered_map<std::string, double>& data) {
    resize_if_needed(sim_step + 1);
    mNrow = std::max(mNrow, sim_step + 1);
    
    for (const auto& [field, value] : data) {
        auto it = mFieldToIdx.find(field);
        if (it != mFieldToIdx.end()) {
            mData(sim_step, it->second) = value;
        }
    }
}

void RolloutRecord::reset() {
    mNrow = 0;
    mData.setZero();
}


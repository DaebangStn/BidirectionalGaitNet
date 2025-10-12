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

        // Foot configuration
        if (record["foot"].as<bool>(false)) {
            config.foot.enabled = true;
            config.foot.contact_left = true;
            config.foot.contact_right = true;
            config.foot.grf_left = true;
            config.foot.grf_right = true;
        }

        // Kinematics configuration
        if (record["kinematics"] && record["kinematics"]["enabled"].as<bool>(false)) {
            config.kinematics.enabled = true;

            // All joint positions
            config.kinematics.all = record["kinematics"]["all"].as<bool>(false);

            // Root position
            config.kinematics.root = record["kinematics"]["root"].as<bool>(false);

            // Angle configuration
            if (record["kinematics"]["angle"] && record["kinematics"]["angle"]["enabled"].as<bool>(false)) {
                config.kinematics.angle.enabled = true;
                config.kinematics.angle.hip = record["kinematics"]["angle"]["hip"].as<bool>(false);
                config.kinematics.angle.hip_ir = record["kinematics"]["angle"]["hip_ir"].as<bool>(false);
                config.kinematics.angle.hip_ab = record["kinematics"]["angle"]["hip_ab"].as<bool>(false);
                config.kinematics.angle.knee = record["kinematics"]["angle"]["knee"].as<bool>(false);
                config.kinematics.angle.ankle = record["kinematics"]["angle"]["ankle"].as<bool>(false);
                config.kinematics.angle.pelvic_tilt = record["kinematics"]["angle"]["pelvic_tilt"].as<bool>(false);
                config.kinematics.angle.pelvic_rotation = record["kinematics"]["angle"]["pelvic_rotation"].as<bool>(false);
                config.kinematics.angle.pelvic_obliquity = record["kinematics"]["angle"]["pelvic_obliquity"].as<bool>(false);
            }

            // Angular velocity configuration
            if (record["kinematics"]["anvel"] && record["kinematics"]["anvel"]["enabled"].as<bool>(false)) {
                config.kinematics.anvel.enabled = true;
                config.kinematics.anvel.hip = record["kinematics"]["anvel"]["hip"].as<bool>(false);
                config.kinematics.anvel.knee = record["kinematics"]["anvel"]["knee"].as<bool>(false);
                config.kinematics.anvel.ankle = record["kinematics"]["anvel"]["ankle"].as<bool>(false);
            }
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading record config: " << e.what() << std::endl;
    }
    
    return config;
}

std::vector<std::string> RolloutRecord::FieldsFromConfig(const RecordConfig& config, int skeleton_dof) {
    std::vector<std::string> fields;

    // Basic fields (always included)
    fields.push_back("step");
    fields.push_back("time");
    fields.push_back("phase");
    fields.push_back("cycle");

    // Contact/GRF fields
    if (config.foot.enabled) {
        if (config.foot.contact_left) fields.push_back("contact_left");
        if (config.foot.contact_right) fields.push_back("contact_right");
        if (config.foot.grf_left) fields.push_back("grf_left");
        if (config.foot.grf_right) fields.push_back("grf_right");
    }

    // Kinematics fields
    if (config.kinematics.enabled) {
        if (config.kinematics.all) {
            fields.push_back("motions");
        }

        // Root position
        if (config.kinematics.root) {
            fields.push_back("root_x");
            fields.push_back("root_y");
            fields.push_back("root_z");
        }

        // Angle fields
        if (config.kinematics.angle.enabled) {
            if (config.kinematics.angle.hip) fields.push_back("angle_HipR");
            if (config.kinematics.angle.hip_ir) fields.push_back("angle_HipIRR");
            if (config.kinematics.angle.hip_ab) fields.push_back("angle_HipAbR");
            if (config.kinematics.angle.knee) fields.push_back("angle_KneeR");
            if (config.kinematics.angle.ankle) fields.push_back("angle_AnkleR");
            if (config.kinematics.angle.pelvic_tilt) fields.push_back("angle_Tilt");
            if (config.kinematics.angle.pelvic_rotation) fields.push_back("angle_Rotation");
            if (config.kinematics.angle.pelvic_obliquity) fields.push_back("angle_Obliquity");
        }

        // Angular velocity fields
        if (config.kinematics.anvel.enabled) {
            if (config.kinematics.anvel.hip) fields.push_back("anvel_HipR");
            if (config.kinematics.anvel.knee) fields.push_back("anvel_KneeR");
            if (config.kinematics.anvel.ankle) fields.push_back("anvel_AnkleR");
        }
    }

    return fields;
}

RolloutRecord::RolloutRecord(const std::vector<std::string>& field_names)
    : mFieldNames(field_names), mNcol(field_names.size()), mNrow(0) {
    for (size_t i = 0; i < field_names.size(); ++i) {
        mFieldToIdx[field_names[i]] = i;
    }
    resize_if_needed(0);
}

RolloutRecord::RolloutRecord(const RecordConfig& config)
    : RolloutRecord(FieldsFromConfig(config)) {}

void RolloutRecord::resize_if_needed(unsigned int requested_size) {
    if (requested_size >= mData.rows()) {
        Eigen::MatrixXd new_data(mData.rows() + DATA_CHUNK_SIZE, mNcol);
        if (mNrow > 0) {
            new_data.topRows(mNrow) = mData.topRows(mNrow);
        }
        mData = std::move(new_data);
    }
}

void RolloutRecord::add(unsigned int sim_step, const std::unordered_map<std::string, double>& data) {
    resize_if_needed(mNrow + 1);

    for (const auto& [key, value] : data) {
        auto it = mFieldToIdx.find(key);
        if (it != mFieldToIdx.end()) {
            mData(mNrow, it->second) = value;
        }
    }
    mNrow++;
}

void RolloutRecord::addVector(const std::string& key, int step, const Eigen::VectorXd& data) {
    resize_if_needed(step + 1);
    auto it = mFieldToIdx.find(key);
    if (it != mFieldToIdx.end()) {
        mData(step, it->second) = data(0);
    }
    if (step >= mNrow) {
        mNrow = step + 1;
    }
}

void RolloutRecord::reset() {
    mNrow = 0;
}

#include "SurgeryOperation.h"
#include "SurgeryExecutor.h"
#include <sstream>

namespace PMuscle {

// ============================================================================
// ResetMusclesOp
// ============================================================================

bool ResetMusclesOp::execute(SurgeryExecutor* executor) {
    executor->resetMuscles();
    return true;
}

YAML::Node ResetMusclesOp::toYAML() const {
    YAML::Node node;
    node["type"] = "reset_muscles";
    return node;
}

std::string ResetMusclesOp::getDescription() const {
    return "Reset all muscles to original state";
}

std::unique_ptr<SurgeryOperation> ResetMusclesOp::fromYAML(const YAML::Node& node) {
    return std::make_unique<ResetMusclesOp>();
}

// ============================================================================
// DistributePassiveForceOp
// ============================================================================

bool DistributePassiveForceOp::execute(SurgeryExecutor* executor) {
    return executor->distributePassiveForce(mMuscles, mReferenceMuscle, mJointAngles);
}

YAML::Node DistributePassiveForceOp::toYAML() const {
    YAML::Node node;
    node["type"] = "distribute_passive_force";
    node["muscles"] = mMuscles;
    node["reference_muscle"] = mReferenceMuscle;
    
    // Add joint angles if present
    if (!mJointAngles.empty()) {
        YAML::Node joints_node;
        for (const auto& [joint_name, angles] : mJointAngles) {
            if (angles.size() == 1) {
                joints_node[joint_name] = angles[0];
            } else {
                std::vector<double> angle_vec(angles.data(), angles.data() + angles.size());
                joints_node[joint_name] = angle_vec;
            }
        }
        node["joint_angles"] = joints_node;
    }
    
    return node;
}

std::string DistributePassiveForceOp::getDescription() const {
    std::ostringstream oss;
    oss << "Distribute passive force from '" << mReferenceMuscle 
        << "' to " << mMuscles.size() << " muscle(s)";
    if (!mJointAngles.empty()) {
        oss << " (at " << mJointAngles.size() << " joint angle(s))";
    }
    return oss.str();
}

std::unique_ptr<SurgeryOperation> DistributePassiveForceOp::fromYAML(const YAML::Node& node) {
    std::vector<std::string> muscles = node["muscles"].as<std::vector<std::string>>();
    std::string reference = node["reference_muscle"].as<std::string>();
    
    // Parse joint angles if present
    std::map<std::string, Eigen::VectorXd> joint_angles;
    if (node["joint_angles"]) {
        const YAML::Node& joints_node = node["joint_angles"];
        for (YAML::const_iterator it = joints_node.begin(); it != joints_node.end(); ++it) {
            std::string joint_name = it->first.as<std::string>();
            
            if (it->second.IsSequence()) {
                std::vector<double> values = it->second.as<std::vector<double>>();
                Eigen::VectorXd angles(values.size());
                for (size_t i = 0; i < values.size(); ++i) {
                    angles[i] = values[i];
                }
                joint_angles[joint_name] = angles;
            } else {
                Eigen::VectorXd angles(1);
                angles[0] = it->second.as<double>();
                joint_angles[joint_name] = angles;
            }
        }
    }
    
    return std::make_unique<DistributePassiveForceOp>(muscles, reference, joint_angles);
}

// ============================================================================
// RelaxPassiveForceOp
// ============================================================================

bool RelaxPassiveForceOp::execute(SurgeryExecutor* executor) {
    return executor->relaxPassiveForce(mMuscles, mJointAngles);
}

YAML::Node RelaxPassiveForceOp::toYAML() const {
    YAML::Node node;
    node["type"] = "relax_passive_force";
    node["muscles"] = mMuscles;
    
    // Add joint angles if present
    if (!mJointAngles.empty()) {
        YAML::Node joints_node;
        for (const auto& [joint_name, angles] : mJointAngles) {
            if (angles.size() == 1) {
                joints_node[joint_name] = angles[0];
            } else {
                std::vector<double> angle_vec(angles.data(), angles.data() + angles.size());
                joints_node[joint_name] = angle_vec;
            }
        }
        node["joint_angles"] = joints_node;
    }
    
    return node;
}

std::string RelaxPassiveForceOp::getDescription() const {
    std::ostringstream oss;
    oss << "Relax passive force for " << mMuscles.size() << " muscle(s)";
    if (!mJointAngles.empty()) {
        oss << " (at " << mJointAngles.size() << " joint angle(s))";
    }
    return oss.str();
}

std::unique_ptr<SurgeryOperation> RelaxPassiveForceOp::fromYAML(const YAML::Node& node) {
    std::vector<std::string> muscles = node["muscles"].as<std::vector<std::string>>();
    
    // Parse joint angles if present
    std::map<std::string, Eigen::VectorXd> joint_angles;
    if (node["joint_angles"]) {
        const YAML::Node& joints_node = node["joint_angles"];
        for (YAML::const_iterator it = joints_node.begin(); it != joints_node.end(); ++it) {
            std::string joint_name = it->first.as<std::string>();
            
            if (it->second.IsSequence()) {
                std::vector<double> values = it->second.as<std::vector<double>>();
                Eigen::VectorXd angles(values.size());
                for (size_t i = 0; i < values.size(); ++i) {
                    angles[i] = values[i];
                }
                joint_angles[joint_name] = angles;
            } else {
                Eigen::VectorXd angles(1);
                angles[0] = it->second.as<double>();
                joint_angles[joint_name] = angles;
            }
        }
    }
    
    return std::make_unique<RelaxPassiveForceOp>(muscles, joint_angles);
}

// ============================================================================
// RemoveAnchorOp
// ============================================================================

bool RemoveAnchorOp::execute(SurgeryExecutor* executor) {
    return executor->removeAnchorFromMuscle(mMuscle, mAnchorIndex);
}

YAML::Node RemoveAnchorOp::toYAML() const {
    YAML::Node node;
    node["type"] = "remove_anchor";
    node["muscle"] = mMuscle;
    node["anchor_index"] = mAnchorIndex;
    return node;
}

std::string RemoveAnchorOp::getDescription() const {
    std::ostringstream oss;
    oss << "Remove anchor #" << mAnchorIndex << " from '" << mMuscle << "'";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> RemoveAnchorOp::fromYAML(const YAML::Node& node) {
    std::string muscle = node["muscle"].as<std::string>();
    int anchor_index = node["anchor_index"].as<int>();
    return std::make_unique<RemoveAnchorOp>(muscle, anchor_index);
}

// ============================================================================
// CopyAnchorOp
// ============================================================================

bool CopyAnchorOp::execute(SurgeryExecutor* executor) {
    return executor->copyAnchorToMuscle(mFromMuscle, mFromIndex, mToMuscle);
}

YAML::Node CopyAnchorOp::toYAML() const {
    YAML::Node node;
    node["type"] = "copy_anchor";
    node["from_muscle"] = mFromMuscle;
    node["from_anchor_index"] = mFromIndex;
    node["to_muscle"] = mToMuscle;
    return node;
}

std::string CopyAnchorOp::getDescription() const {
    std::ostringstream oss;
    oss << "Copy anchor #" << mFromIndex << " from '" << mFromMuscle 
        << "' to '" << mToMuscle << "'";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> CopyAnchorOp::fromYAML(const YAML::Node& node) {
    std::string from_muscle = node["from_muscle"].as<std::string>();
    int from_index = node["from_anchor_index"].as<int>();
    std::string to_muscle = node["to_muscle"].as<std::string>();
    return std::make_unique<CopyAnchorOp>(from_muscle, from_index, to_muscle);
}

// ============================================================================
// EditAnchorPositionOp
// ============================================================================

bool EditAnchorPositionOp::execute(SurgeryExecutor* executor) {
    return executor->editAnchorPosition(mMuscle, mAnchorIndex, mPosition);
}

YAML::Node EditAnchorPositionOp::toYAML() const {
    YAML::Node node;
    node["type"] = "edit_anchor_position";
    node["muscle"] = mMuscle;
    node["anchor_index"] = mAnchorIndex;
    node["position"].push_back(mPosition[0]);
    node["position"].push_back(mPosition[1]);
    node["position"].push_back(mPosition[2]);
    return node;
}

std::string EditAnchorPositionOp::getDescription() const {
    std::ostringstream oss;
    oss << "Edit anchor #" << mAnchorIndex << " position in '" << mMuscle 
        << "' to [" << mPosition[0] << ", " << mPosition[1] << ", " << mPosition[2] << "]";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> EditAnchorPositionOp::fromYAML(const YAML::Node& node) {
    std::string muscle = node["muscle"].as<std::string>();
    int anchor_index = node["anchor_index"].as<int>();
    std::vector<double> pos_vec = node["position"].as<std::vector<double>>();
    Eigen::Vector3d position(pos_vec[0], pos_vec[1], pos_vec[2]);
    return std::make_unique<EditAnchorPositionOp>(muscle, anchor_index, position);
}

// ============================================================================
// EditAnchorWeightsOp
// ============================================================================

bool EditAnchorWeightsOp::execute(SurgeryExecutor* executor) {
    return executor->editAnchorWeights(mMuscle, mAnchorIndex, mWeights);
}

YAML::Node EditAnchorWeightsOp::toYAML() const {
    YAML::Node node;
    node["type"] = "edit_anchor_weights";
    node["muscle"] = mMuscle;
    node["anchor_index"] = mAnchorIndex;
    node["weights"] = mWeights;
    return node;
}

std::string EditAnchorWeightsOp::getDescription() const {
    std::ostringstream oss;
    oss << "Edit anchor #" << mAnchorIndex << " weights in '" << mMuscle 
        << "' (" << mWeights.size() << " weights)";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> EditAnchorWeightsOp::fromYAML(const YAML::Node& node) {
    std::string muscle = node["muscle"].as<std::string>();
    int anchor_index = node["anchor_index"].as<int>();
    std::vector<double> weights = node["weights"].as<std::vector<double>>();
    return std::make_unique<EditAnchorWeightsOp>(muscle, anchor_index, weights);
}

// ============================================================================
// AddBodyNodeToAnchorOp
// ============================================================================

bool AddBodyNodeToAnchorOp::execute(SurgeryExecutor* executor) {
    return executor->addBodyNodeToAnchor(mMuscle, mAnchorIndex, mBodyNode, mWeight);
}

YAML::Node AddBodyNodeToAnchorOp::toYAML() const {
    YAML::Node node;
    node["type"] = "add_bodynode_to_anchor";
    node["muscle"] = mMuscle;
    node["anchor_index"] = mAnchorIndex;
    node["bodynode"] = mBodyNode;
    node["weight"] = mWeight;
    return node;
}

std::string AddBodyNodeToAnchorOp::getDescription() const {
    std::ostringstream oss;
    oss << "Add body node '" << mBodyNode << "' to anchor #" << mAnchorIndex 
        << " in '" << mMuscle << "' (weight: " << mWeight << ")";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> AddBodyNodeToAnchorOp::fromYAML(const YAML::Node& node) {
    std::string muscle = node["muscle"].as<std::string>();
    int anchor_index = node["anchor_index"].as<int>();
    std::string bodynode = node["bodynode"].as<std::string>();
    double weight = node["weight"].as<double>();
    return std::make_unique<AddBodyNodeToAnchorOp>(muscle, anchor_index, bodynode, weight);
}

// ============================================================================
// RemoveBodyNodeFromAnchorOp
// ============================================================================

bool RemoveBodyNodeFromAnchorOp::execute(SurgeryExecutor* executor) {
    return executor->removeBodyNodeFromAnchor(mMuscle, mAnchorIndex, mBodyNodeIndex);
}

YAML::Node RemoveBodyNodeFromAnchorOp::toYAML() const {
    YAML::Node node;
    node["type"] = "remove_bodynode_from_anchor";
    node["muscle"] = mMuscle;
    node["anchor_index"] = mAnchorIndex;
    node["bodynode_index"] = mBodyNodeIndex;
    return node;
}

std::string RemoveBodyNodeFromAnchorOp::getDescription() const {
    std::ostringstream oss;
    oss << "Remove body node #" << mBodyNodeIndex << " from anchor #" << mAnchorIndex 
        << " in '" << mMuscle << "'";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> RemoveBodyNodeFromAnchorOp::fromYAML(const YAML::Node& node) {
    std::string muscle = node["muscle"].as<std::string>();
    int anchor_index = node["anchor_index"].as<int>();
    int bodynode_index = node["bodynode_index"].as<int>();
    return std::make_unique<RemoveBodyNodeFromAnchorOp>(muscle, anchor_index, bodynode_index);
}

// ============================================================================
// ExportMusclesOp
// ============================================================================

bool ExportMusclesOp::execute(SurgeryExecutor* executor) {
    try {
        executor->exportMuscles(mFilepath);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Surgery] Export failed: " << e.what() << std::endl;
        return false;
    }
}

YAML::Node ExportMusclesOp::toYAML() const {
    YAML::Node node;
    node["type"] = "export_muscles";
    node["filepath"] = mFilepath;
    return node;
}

std::string ExportMusclesOp::getDescription() const {
    return "Export muscles to '" + mFilepath + "'";
}

std::unique_ptr<SurgeryOperation> ExportMusclesOp::fromYAML(const YAML::Node& node) {
    std::string filepath = node["filepath"].as<std::string>();
    return std::make_unique<ExportMusclesOp>(filepath);
}

} // namespace PMuscle


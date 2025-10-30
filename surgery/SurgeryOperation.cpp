#include "SurgeryOperation.h"
#include "SurgeryExecutor.h"
#include "Log.h"
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
        LOG_ERROR("[Surgery] Export failed: " << e.what());
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

// ═══════════════════════════════════════════════════════════════════════════
// ExportSkeletonOp Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool ExportSkeletonOp::execute(SurgeryExecutor* executor) {
    try {
        executor->exportSkeleton(mFilepath);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("[Surgery] Skeleton export failed: " << e.what());
        return false;
    }
}

YAML::Node ExportSkeletonOp::toYAML() const {
    YAML::Node node;
    node["type"] = "export_skeleton";
    node["filepath"] = mFilepath;
    return node;
}

std::string ExportSkeletonOp::getDescription() const {
    return "Export skeleton to '" + mFilepath + "'";
}

std::unique_ptr<SurgeryOperation> ExportSkeletonOp::fromYAML(const YAML::Node& node) {
    std::string filepath = node["filepath"].as<std::string>();
    return std::make_unique<ExportSkeletonOp>(filepath);
}

// ═══════════════════════════════════════════════════════════════════════════
// RotateJointOffsetOp Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool RotateJointOffsetOp::execute(SurgeryExecutor* executor) {
    return executor->rotateJointOffset(mJointName, mAxis, mAngle, mPreservePosition);
}

YAML::Node RotateJointOffsetOp::toYAML() const {
    YAML::Node node;
    node["type"] = "rotate_joint_offset";
    node["joint_name"] = mJointName;

    node["axis"].push_back(mAxis[0]);
    node["axis"].push_back(mAxis[1]);
    node["axis"].push_back(mAxis[2]);

    node["angle"] = mAngle;
    node["preserve_position"] = mPreservePosition;
    return node;
}

std::string RotateJointOffsetOp::getDescription() const {
    std::ostringstream oss;
    oss << "Rotate joint offset for '" << mJointName << "' by " << mAngle << " rad";
    if (mPreservePosition) {
        oss << " (preserve position)";
    }
    return oss.str();
}

std::unique_ptr<SurgeryOperation> RotateJointOffsetOp::fromYAML(const YAML::Node& node) {
    std::string joint_name = node["joint_name"].as<std::string>();

    std::vector<double> axis_vec = node["axis"].as<std::vector<double>>();
    Eigen::Vector3d axis(axis_vec[0], axis_vec[1], axis_vec[2]);

    double angle = node["angle"].as<double>();

    // Backward compatible: default to false if not present
    bool preserve_position = false;
    if (node["preserve_position"]) {
        preserve_position = node["preserve_position"].as<bool>();
    }

    return std::make_unique<RotateJointOffsetOp>(joint_name, axis, angle, preserve_position);
}

// ═══════════════════════════════════════════════════════════════════════════
// RotateAnchorPointsOp Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool RotateAnchorPointsOp::execute(SurgeryExecutor* executor) {
    return executor->rotateAnchorPoints(mMuscleName, mRefAnchorIndex,
                                       mSearchDirection, mRotationAxis, mAngle);
}

YAML::Node RotateAnchorPointsOp::toYAML() const {
    YAML::Node node;
    node["type"] = "rotate_anchor_points";
    node["muscle_name"] = mMuscleName;
    node["ref_anchor_index"] = mRefAnchorIndex;

    node["search_direction"].push_back(mSearchDirection[0]);
    node["search_direction"].push_back(mSearchDirection[1]);
    node["search_direction"].push_back(mSearchDirection[2]);

    node["rotation_axis"].push_back(mRotationAxis[0]);
    node["rotation_axis"].push_back(mRotationAxis[1]);
    node["rotation_axis"].push_back(mRotationAxis[2]);

    node["angle"] = mAngle;
    return node;
}

std::string RotateAnchorPointsOp::getDescription() const {
    std::ostringstream oss;
    oss << "Rotate anchor points on muscle '" << mMuscleName << "' (ref anchor #"
        << mRefAnchorIndex << ") by " << mAngle << " rad";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> RotateAnchorPointsOp::fromYAML(const YAML::Node& node) {
    std::string muscle_name = node["muscle_name"].as<std::string>();
    int ref_anchor_index = node["ref_anchor_index"].as<int>();

    std::vector<double> search_dir_vec = node["search_direction"].as<std::vector<double>>();
    Eigen::Vector3d search_direction(search_dir_vec[0], search_dir_vec[1], search_dir_vec[2]);

    std::vector<double> rot_axis_vec = node["rotation_axis"].as<std::vector<double>>();
    Eigen::Vector3d rotation_axis(rot_axis_vec[0], rot_axis_vec[1], rot_axis_vec[2]);

    double angle = node["angle"].as<double>();

    return std::make_unique<RotateAnchorPointsOp>(muscle_name, ref_anchor_index,
                                                   search_direction, rotation_axis, angle);
}

// ═══════════════════════════════════════════════════════════════════════════
// FDOCombinedOp Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool FDOCombinedOp::execute(SurgeryExecutor* executor) {
    return executor->executeFDO(mRefMuscle, mRefAnchorIndex,
                               mSearchDirection, mRotationAxis, mAngle);
}

YAML::Node FDOCombinedOp::toYAML() const {
    YAML::Node node;
    node["type"] = "fdo_combined";
    node["ref_muscle"] = mRefMuscle;
    node["ref_anchor_index"] = mRefAnchorIndex;
    // Note: target_bodynode is no longer stored - obtained from anchor at execution time

    node["search_direction"].push_back(mSearchDirection[0]);
    node["search_direction"].push_back(mSearchDirection[1]);
    node["search_direction"].push_back(mSearchDirection[2]);

    node["rotation_axis"].push_back(mRotationAxis[0]);
    node["rotation_axis"].push_back(mRotationAxis[1]);
    node["rotation_axis"].push_back(mRotationAxis[2]);

    node["angle"] = mAngle;
    return node;
}

std::string FDOCombinedOp::getDescription() const {
    std::ostringstream oss;
    oss << "FDO Combined Surgery: ref_muscle='" << mRefMuscle
        << "' (anchor #" << mRefAnchorIndex
        << "), angle=" << (mAngle * 180.0 / M_PI) << " deg";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> FDOCombinedOp::fromYAML(const YAML::Node& node) {
    std::string ref_muscle = node["ref_muscle"].as<std::string>();
    int ref_anchor_index = node["ref_anchor_index"].as<int>();
    // Note: target_bodynode is no longer needed - obtained from anchor at execution time
    // Old YAML files with target_bodynode field will simply ignore it

    std::vector<double> search_dir_vec = node["search_direction"].as<std::vector<double>>();
    Eigen::Vector3d search_direction(search_dir_vec[0], search_dir_vec[1], search_dir_vec[2]);

    std::vector<double> rot_axis_vec = node["rotation_axis"].as<std::vector<double>>();
    Eigen::Vector3d rotation_axis(rot_axis_vec[0], rot_axis_vec[1], rot_axis_vec[2]);

    double angle = node["angle"].as<double>();

    return std::make_unique<FDOCombinedOp>(ref_muscle, ref_anchor_index,
                                           search_direction, rotation_axis, angle);
}

// ═══════════════════════════════════════════════════════════════════════════
// WeakenMuscleOp Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool WeakenMuscleOp::execute(SurgeryExecutor* executor) {
    return executor->weakenMuscles(mMuscles, mStrengthRatio);
}

YAML::Node WeakenMuscleOp::toYAML() const {
    YAML::Node node;
    node["type"] = "weaken_muscle";
    node["muscles"] = mMuscles;
    node["strength_ratio"] = mStrengthRatio;
    return node;
}

std::string WeakenMuscleOp::getDescription() const {
    std::ostringstream oss;
    int percentage = static_cast<int>(mStrengthRatio * 100);
    oss << "Weaken " << mMuscles.size() << " muscle(s) to "
        << percentage << "% strength";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> WeakenMuscleOp::fromYAML(const YAML::Node& node) {
    std::vector<std::string> muscles = node["muscles"].as<std::vector<std::string>>();
    double strength_ratio = node["strength_ratio"].as<double>();
    return std::make_unique<WeakenMuscleOp>(muscles, strength_ratio);
}

// ============================================================================
// ApplyPosePresetOp
// ============================================================================

bool ApplyPosePresetOp::execute(SurgeryExecutor* executor) {
    return executor->applyPosePresetByName(mPresetName);
}

YAML::Node ApplyPosePresetOp::toYAML() const {
    YAML::Node node;
    node["type"] = "apply_pose_preset";
    node["preset"] = mPresetName;
    return node;
}

std::string ApplyPosePresetOp::getDescription() const {
    return "Apply pose preset: '" + mPresetName + "'";
}

std::unique_ptr<SurgeryOperation> ApplyPosePresetOp::fromYAML(const YAML::Node& node) {
    std::string preset_name = node["preset"].as<std::string>();
    return std::make_unique<ApplyPosePresetOp>(preset_name);
}

} // namespace PMuscle


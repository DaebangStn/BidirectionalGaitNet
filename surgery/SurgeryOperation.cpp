#include "SurgeryOperation.h"
#include "SurgeryExecutor.h"
#include "optimizer/ContractureOptimizer.h"
#include "Log.h"
#include <sstream>
#include <fstream>
#include <filesystem>

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
    node["type"] = "scale_muscle_strength";  // Use new name (backward compatible with "weaken_muscle")
    node["muscles"] = mMuscles;
    node["strength_ratio"] = mStrengthRatio;
    return node;
}

std::string WeakenMuscleOp::getDescription() const {
    std::ostringstream oss;
    int percentage = static_cast<int>(mStrengthRatio * 100);
    std::string action = (mStrengthRatio < 1.0) ? "Weaken" : (mStrengthRatio > 1.0) ? "Strengthen" : "Scale";
    oss << action << " " << mMuscles.size() << " muscle(s) to "
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

// ============================================================================
// OptimizeWaypointsOp
// ============================================================================

bool OptimizeWaypointsOp::execute(SurgeryExecutor* executor) {
    // Load reference character from default paths
    // Note: This uses the same character as subject for reference (backward compatibility)
    // For proper reference character, use MusclePersonalizerApp which loads a separate reference
    Character* subject = executor->getCharacter();

    // Build Config from operation parameters
    WaypointOptimizer::Config config;
    config.maxIterations = mMaxIterations;
    config.numSampling = mNumSampling;
    config.lambdaShape = mLambdaShape;
    config.lambdaLengthCurve = mLambdaLengthCurve;
    config.fixOriginInsertion = mFixOriginInsertion;
    // Other params use defaults

    return executor->optimizeWaypoints(mMuscleNames, mHDFMotionPath, config, subject);
}

YAML::Node OptimizeWaypointsOp::toYAML() const {
    YAML::Node node;
    node["type"] = "optimize_waypoints";
    node["muscles"] = mMuscleNames;
    node["hdf_motion_path"] = mHDFMotionPath;
    node["max_iterations"] = mMaxIterations;
    node["num_sampling"] = mNumSampling;
    node["lambda_shape"] = mLambdaShape;
    node["lambda_length_curve"] = mLambdaLengthCurve;
    node["fix_origin_insertion"] = mFixOriginInsertion;
    return node;
}

std::string OptimizeWaypointsOp::getDescription() const {
    std::ostringstream oss;
    oss << "Optimize waypoints for " << mMuscleNames.size() << " muscle(s) "
        << "(motion: " << mHDFMotionPath << ", max_iter: " << mMaxIterations << ")";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> OptimizeWaypointsOp::fromYAML(const YAML::Node& node) {
    std::vector<std::string> muscles = node["muscles"].as<std::vector<std::string>>();
    std::string hdf_motion_path = node["hdf_motion_path"].as<std::string>();

    // Optional parameters with defaults
    int max_iterations = node["max_iterations"] ? node["max_iterations"].as<int>() : 10000;
    int num_sampling = node["num_sampling"] ? node["num_sampling"].as<int>() : 10;
    double lambda_shape = node["lambda_shape"] ? node["lambda_shape"].as<double>() : 0.1;
    double lambda_length_curve = node["lambda_length_curve"] ? node["lambda_length_curve"].as<double>() : 0.1;
    bool fix_origin_insertion = node["fix_origin_insertion"] ? node["fix_origin_insertion"].as<bool>() : true;

    return std::make_unique<OptimizeWaypointsOp>(muscles, hdf_motion_path,
                                                  max_iterations, num_sampling, lambda_shape,
                                                  lambda_length_curve, fix_origin_insertion);
}

// ============================================================================
// MirrorAnchorPositionsOp
// ============================================================================

bool MirrorAnchorPositionsOp::execute(SurgeryExecutor* executor) {
    return executor->mirrorAnchorPositions(mMuscles);
}

YAML::Node MirrorAnchorPositionsOp::toYAML() const {
    YAML::Node node;
    node["type"] = "mirror_anchor_positions";
    if (!mMuscles.empty()) {
        node["muscles"] = mMuscles;
    }
    return node;
}

std::string MirrorAnchorPositionsOp::getDescription() const {
    if (mMuscles.empty()) {
        return "Mirror anchor positions for all L/R muscle pairs";
    }
    std::ostringstream oss;
    oss << "Mirror anchor positions for " << mMuscles.size() << " muscle pair(s)";
    return oss.str();
}

std::unique_ptr<SurgeryOperation> MirrorAnchorPositionsOp::fromYAML(const YAML::Node& node) {
    if (node["muscles"]) {
        auto muscles = node["muscles"].as<std::vector<std::string>>();
        return std::make_unique<MirrorAnchorPositionsOp>(muscles);
    }
    return std::make_unique<MirrorAnchorPositionsOp>();
}

// ============================================================================
// ContractureOptOp
// ============================================================================

bool ContractureOptOp::execute(SurgeryExecutor* executor) {
    Character* character = executor->getCharacter();
    if (!character) {
        LOG_ERROR("[ContractureOptOp] No character loaded");
        return false;
    }

    auto skel = character->getSkeleton();

    // Load ROM trial configs
    std::vector<ROMTrialConfig> rom_configs;
    std::vector<std::string> trial_names;
    for (const auto& trial : mROMTrials) {
        std::string path = "data/config/rom/" + trial.name + ".yaml";
        try {
            auto rom = ContractureOptimizer::loadROMConfig(path, skel);
            if (trial.angle_deg != 0.0) {
                rom.rom_angle = rom.cd_neg ? -trial.angle_deg : trial.angle_deg;
            }
            rom_configs.push_back(rom);
            trial_names.push_back(trial.name);
        } catch (const std::exception& e) {
            LOG_ERROR("[ContractureOptOp] Failed to load ROM config: " << path << " - " << e.what());
        }
    }

    if (rom_configs.empty()) {
        LOG_ERROR("[ContractureOptOp] No valid ROM configs loaded");
        return false;
    }

    // Build inline muscle groups YAML
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;

    if (!mSearchGroupMap.empty()) {
        // Multi-group mode: each search group has its own set of muscles
        emitter << YAML::Key << "search_groups" << YAML::Value << YAML::BeginMap;
        for (const auto& [group_name, group_muscles] : mSearchGroupMap) {
            emitter << YAML::Key << group_name << YAML::Value << YAML::BeginSeq;
            for (const auto& m : group_muscles) emitter << m;
            emitter << YAML::EndSeq;
        }
        emitter << YAML::EndMap;

        // Each muscle is its own optimization group
        emitter << YAML::Key << "optimization_groups" << YAML::Value << YAML::BeginMap;
        for (const auto& [group_name, group_muscles] : mSearchGroupMap) {
            for (const auto& m : group_muscles) {
                emitter << YAML::Key << m << YAML::Value << YAML::BeginSeq << m << YAML::EndSeq;
            }
        }
        emitter << YAML::EndMap;

        // All search groups in one mapping entry for joint N-dim grid search
        emitter << YAML::Key << "grid_search_mapping" << YAML::Value << YAML::BeginSeq;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "trials" << YAML::Value << YAML::BeginSeq;
        for (const auto& t : trial_names) emitter << t;
        emitter << YAML::EndSeq;
        emitter << YAML::Key << "groups" << YAML::Value << YAML::BeginSeq;
        for (const auto& [group_name, _] : mSearchGroupMap) emitter << group_name;
        emitter << YAML::EndSeq;
        emitter << YAML::EndMap;
        emitter << YAML::EndSeq;
    } else {
        // Legacy single-group mode
        emitter << YAML::Key << "search_groups" << YAML::Value << YAML::BeginMap;
        emitter << YAML::Key << mSearchGroup << YAML::Value << YAML::BeginSeq;
        for (const auto& m : mMuscles) emitter << m;
        emitter << YAML::EndSeq;
        emitter << YAML::EndMap;

        emitter << YAML::Key << "optimization_groups" << YAML::Value << YAML::BeginMap;
        for (const auto& m : mMuscles) {
            emitter << YAML::Key << m << YAML::Value << YAML::BeginSeq << m << YAML::EndSeq;
        }
        emitter << YAML::EndMap;

        emitter << YAML::Key << "grid_search_mapping" << YAML::Value << YAML::BeginSeq;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "trials" << YAML::Value << YAML::BeginSeq;
        for (const auto& t : trial_names) emitter << t;
        emitter << YAML::EndSeq;
        emitter << YAML::Key << "groups" << YAML::Value << YAML::BeginSeq << mSearchGroup << YAML::EndSeq;
        emitter << YAML::EndMap;
        emitter << YAML::EndSeq;
    }

    emitter << YAML::EndMap;

    std::string tmp_path = "/tmp/contracture_opt_groups.yaml";
    { std::ofstream ofs(tmp_path); ofs << emitter.c_str(); }

    ContractureOptimizer optimizer;
    optimizer.loadMuscleGroups(tmp_path, character);

    YAML::Node tmp_config = YAML::LoadFile(tmp_path);
    if (tmp_config["grid_search_mapping"]) {
        std::vector<GridSearchMapping> mappings;
        for (const auto& entry : tmp_config["grid_search_mapping"]) {
            GridSearchMapping gm;
            if (entry["trials"]) for (const auto& t : entry["trials"]) gm.trials.push_back(t.as<std::string>());
            if (entry["groups"]) for (const auto& g : entry["groups"]) gm.groups.push_back(g.as<std::string>());
            mappings.push_back(gm);
        }
        optimizer.setGridSearchMapping(mappings);
    }

    ContractureOptimizer::Config opt_config;
    opt_config.maxIterations = mMaxIterations;
    opt_config.minRatio = mMinRatio;
    opt_config.maxRatio = mMaxRatio;
    opt_config.gridSearchBegin = mGridBegin;
    opt_config.gridSearchEnd = mGridEnd;
    opt_config.gridSearchInterval = mGridInterval;

    if (mParamType == "lt_rel") {
        opt_config.paramType = ContractureOptimizer::OptParam::LT_REL;
    } else {
        opt_config.paramType = ContractureOptimizer::OptParam::LM_CONTRACT;
    }

    auto result = optimizer.optimize(character, rom_configs, opt_config);
    std::filesystem::remove(tmp_path);

    LOG_INFO("[ContractureOptOp] Optimization complete (" << mSearchGroup << ", " << mParamType
             << "): " << result.muscle_results.size() << " muscles modified");
    for (const auto& mr : result.muscle_results) {
        LOG_INFO("  " << mr.muscle_name << ": " << mr.lm_contract_before << " -> " << mr.lm_contract_after
                 << " (ratio=" << mr.ratio << ")");
    }

    return true;
}

YAML::Node ContractureOptOp::toYAML() const {
    YAML::Node node;
    node["type"] = "contracture_opt";

    if (!mSearchGroupMap.empty()) {
        for (const auto& [name, muscles] : mSearchGroupMap) {
            node["search_groups"][name] = muscles;
        }
    } else {
        node["search_group"] = mSearchGroup;
        node["muscles"] = mMuscles;
    }

    for (const auto& trial : mROMTrials) {
        YAML::Node t;
        t["name"] = trial.name;
        t["angle"] = trial.angle_deg;
        node["rom_trials"].push_back(t);
    }

    node["param_type"] = mParamType;
    node["max_iterations"] = mMaxIterations;
    node["min_ratio"] = mMinRatio;
    node["max_ratio"] = mMaxRatio;
    node["grid_begin"] = mGridBegin;
    node["grid_end"] = mGridEnd;
    node["grid_interval"] = mGridInterval;
    return node;
}

std::string ContractureOptOp::getDescription() const {
    std::ostringstream oss;
    if (!mSearchGroupMap.empty()) {
        size_t total_muscles = 0;
        std::string group_names;
        for (const auto& [name, muscles] : mSearchGroupMap) {
            if (!group_names.empty()) group_names += "+";
            group_names += name;
            total_muscles += muscles.size();
        }
        oss << "Contracture optimization (" << group_names << ", " << mParamType
            << ", " << total_muscles << " muscles, " << mROMTrials.size() << " trials)";
    } else {
        oss << "Contracture optimization (" << mSearchGroup << ", " << mParamType
            << ", " << mMuscles.size() << " muscles, " << mROMTrials.size() << " trials)";
    }
    return oss.str();
}

std::unique_ptr<SurgeryOperation> ContractureOptOp::fromYAML(const YAML::Node& node) {
    std::vector<ContractureOptOp::ROMTrialParam> rom_trials;
    if (node["rom_trials"]) {
        for (const auto& t : node["rom_trials"]) {
            ROMTrialParam param;
            param.name = t["name"].as<std::string>();
            param.angle_deg = t["angle"] ? t["angle"].as<double>() : 0.0;
            rom_trials.push_back(param);
        }
    }

    std::string param_type = node["param_type"] ? node["param_type"].as<std::string>() : "lm_contract";
    int max_iterations = node["max_iterations"] ? node["max_iterations"].as<int>() : 100;
    double min_ratio = node["min_ratio"] ? node["min_ratio"].as<double>() : 0.5;
    double max_ratio = node["max_ratio"] ? node["max_ratio"].as<double>() : 2.0;
    double grid_begin = node["grid_begin"] ? node["grid_begin"].as<double>() : 0.5;
    double grid_end = node["grid_end"] ? node["grid_end"].as<double>() : 2.0;
    double grid_interval = node["grid_interval"] ? node["grid_interval"].as<double>() : 0.05;

    // New multi-group format
    if (node["search_groups"]) {
        std::map<std::string, std::vector<std::string>> search_groups;
        for (const auto& sg : node["search_groups"]) {
            std::string name = sg.first.as<std::string>();
            auto muscles = sg.second.as<std::vector<std::string>>();
            search_groups[name] = muscles;
        }
        return std::make_unique<ContractureOptOp>(
            search_groups, rom_trials, param_type,
            max_iterations, min_ratio, max_ratio, grid_begin, grid_end, grid_interval);
    }

    // Legacy single-group format
    std::string search_group = node["search_group"].as<std::string>();
    auto muscles = node["muscles"].as<std::vector<std::string>>();
    return std::make_unique<ContractureOptOp>(
        search_group, muscles, rom_trials, param_type,
        max_iterations, min_ratio, max_ratio, grid_begin, grid_end, grid_interval);
}

} // namespace PMuscle


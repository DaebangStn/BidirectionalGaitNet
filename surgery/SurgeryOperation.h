#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

namespace PMuscle {

// Forward declaration
class SurgeryExecutor;

// Base class for all surgery operations
class SurgeryOperation {
public:
    virtual ~SurgeryOperation() = default;
    
    // Execute the operation
    virtual bool execute(SurgeryExecutor* executor) = 0;
    
    // Serialize to YAML node
    virtual YAML::Node toYAML() const = 0;
    
    // Get human-readable description
    virtual std::string getDescription() const = 0;
    
    // Get operation type name
    virtual std::string getType() const = 0;
};

// Reset all muscles to original state
class ResetMusclesOp : public SurgeryOperation {
public:
    ResetMusclesOp() = default;
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "reset_muscles"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
};

// Distribute passive force coefficient from reference muscle to others
class DistributePassiveForceOp : public SurgeryOperation {
public:
    DistributePassiveForceOp(const std::vector<std::string>& muscles, const std::string& reference,
                             const std::map<std::string, Eigen::VectorXd>& joint_angles = {})
        : mMuscles(muscles), mReferenceMuscle(reference), mJointAngles(joint_angles) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "distribute_passive_force"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::vector<std::string> mMuscles;
    std::string mReferenceMuscle;
    std::map<std::string, Eigen::VectorXd> mJointAngles;  // Joint configuration for operation
};

// Relax passive force for selected muscles
class RelaxPassiveForceOp : public SurgeryOperation {
public:
    RelaxPassiveForceOp(const std::vector<std::string>& muscles,
                        const std::map<std::string, Eigen::VectorXd>& joint_angles = {})
        : mMuscles(muscles), mJointAngles(joint_angles) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "relax_passive_force"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::vector<std::string> mMuscles;
    std::map<std::string, Eigen::VectorXd> mJointAngles;  // Joint configuration for operation
};

// Remove anchor from muscle
class RemoveAnchorOp : public SurgeryOperation {
public:
    RemoveAnchorOp(const std::string& muscle, int anchor_index)
        : mMuscle(muscle), mAnchorIndex(anchor_index) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "remove_anchor"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::string mMuscle;
    int mAnchorIndex;
};

// Copy anchor from one muscle to another
class CopyAnchorOp : public SurgeryOperation {
public:
    CopyAnchorOp(const std::string& from_muscle, int from_index, const std::string& to_muscle)
        : mFromMuscle(from_muscle), mFromIndex(from_index), mToMuscle(to_muscle) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "copy_anchor"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::string mFromMuscle;
    int mFromIndex;
    std::string mToMuscle;
};

// Edit anchor position
class EditAnchorPositionOp : public SurgeryOperation {
public:
    EditAnchorPositionOp(const std::string& muscle, int anchor_index, const Eigen::Vector3d& position)
        : mMuscle(muscle), mAnchorIndex(anchor_index), mPosition(position) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "edit_anchor_position"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::string mMuscle;
    int mAnchorIndex;
    Eigen::Vector3d mPosition;
};

// Edit anchor weights
class EditAnchorWeightsOp : public SurgeryOperation {
public:
    EditAnchorWeightsOp(const std::string& muscle, int anchor_index, const std::vector<double>& weights)
        : mMuscle(muscle), mAnchorIndex(anchor_index), mWeights(weights) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "edit_anchor_weights"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::string mMuscle;
    int mAnchorIndex;
    std::vector<double> mWeights;
};

// Add body node to anchor
class AddBodyNodeToAnchorOp : public SurgeryOperation {
public:
    AddBodyNodeToAnchorOp(const std::string& muscle, int anchor_index, 
                          const std::string& bodynode, double weight)
        : mMuscle(muscle), mAnchorIndex(anchor_index), 
          mBodyNode(bodynode), mWeight(weight) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "add_bodynode_to_anchor"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::string mMuscle;
    int mAnchorIndex;
    std::string mBodyNode;
    double mWeight;
};

// Remove body node from anchor
class RemoveBodyNodeFromAnchorOp : public SurgeryOperation {
public:
    RemoveBodyNodeFromAnchorOp(const std::string& muscle, int anchor_index, int bodynode_index)
        : mMuscle(muscle), mAnchorIndex(anchor_index), mBodyNodeIndex(bodynode_index) {}
    
    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "remove_bodynode_from_anchor"; }
    
    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);
    
private:
    std::string mMuscle;
    int mAnchorIndex;
    int mBodyNodeIndex;
};

// Export muscles to file
class ExportMusclesOp : public SurgeryOperation {
public:
    ExportMusclesOp(const std::string& filepath)
        : mFilepath(filepath) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "export_muscles"; }

    // Allow overriding filepath (e.g., from CLI)
    void setFilepath(const std::string& filepath) { mFilepath = filepath; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mFilepath;
};

// Export skeleton to file
class ExportSkeletonOp : public SurgeryOperation {
public:
    ExportSkeletonOp(const std::string& filepath)
        : mFilepath(filepath) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "export_skeleton"; }

    // Allow overriding filepath (e.g., from CLI)
    void setFilepath(const std::string& filepath) { mFilepath = filepath; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mFilepath;
};

// Rotate joint offset and frame (FDO operation part 1)
class RotateJointOffsetOp : public SurgeryOperation {
public:
    RotateJointOffsetOp(const std::string& joint_name, const Eigen::Vector3d& axis, double angle, bool preserve_position = false)
        : mJointName(joint_name), mAxis(axis), mAngle(angle), mPreservePosition(preserve_position) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "rotate_joint_offset"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mJointName;
    Eigen::Vector3d mAxis;
    double mAngle;
    bool mPreservePosition;
};

// Rotate muscle anchor points (FDO operation part 2)
class RotateAnchorPointsOp : public SurgeryOperation {
public:
    RotateAnchorPointsOp(const std::string& muscle_name, int ref_anchor_index,
                         const Eigen::Vector3d& search_direction,
                         const Eigen::Vector3d& rotation_axis, double angle)
        : mMuscleName(muscle_name), mRefAnchorIndex(ref_anchor_index),
          mSearchDirection(search_direction), mRotationAxis(rotation_axis), mAngle(angle) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "rotate_anchor_points"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mMuscleName;
    int mRefAnchorIndex;
    Eigen::Vector3d mSearchDirection;
    Eigen::Vector3d mRotationAxis;
    double mAngle;
};

// FDO Combined Surgery Operation
// Target bodynode is obtained from the reference anchor (must be single-LBS)
class FDOCombinedOp : public SurgeryOperation {
public:
    FDOCombinedOp(const std::string& ref_muscle, int ref_anchor_index,
                  const Eigen::Vector3d& search_direction,
                  const Eigen::Vector3d& rotation_axis, double angle)
        : mRefMuscle(ref_muscle), mRefAnchorIndex(ref_anchor_index),
          mSearchDirection(search_direction),
          mRotationAxis(rotation_axis), mAngle(angle) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "fdo_combined"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mRefMuscle;
    int mRefAnchorIndex;
    Eigen::Vector3d mSearchDirection;
    Eigen::Vector3d mRotationAxis;
    double mAngle;
};

// Weaken muscle by reducing force capacity
class WeakenMuscleOp : public SurgeryOperation {
public:
    WeakenMuscleOp(const std::vector<std::string>& muscles, double strength_ratio)
        : mMuscles(muscles), mStrengthRatio(strength_ratio) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "weaken_muscle"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::vector<std::string> mMuscles;
    double mStrengthRatio;  // 0.0 to 1.0, where 1.0 = full strength, 0.0 = paralyzed
};

// Apply pose preset to skeleton
class ApplyPosePresetOp : public SurgeryOperation {
public:
    ApplyPosePresetOp(const std::string& preset_name)
        : mPresetName(preset_name) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "apply_pose_preset"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mPresetName;  // "supine", "standing", "prone", etc.
};

// Optimize muscle waypoint positions using Ceres solver
// Preserves force directions and length-angle curve characteristics
// Each muscle uses itself as the reference
class OptimizeWaypointsOp : public SurgeryOperation {
public:
    OptimizeWaypointsOp(const std::vector<std::string>& muscle_names,
                        const std::string& hdf_motion_path,
                        int max_iterations = 10000,
                        int num_sampling = 10,
                        double lambda_shape = 0.1,
                        double lambda_length_curve = 0.1,
                        bool fix_origin_insertion = true)
        : mMuscleNames(muscle_names),
          mHDFMotionPath(hdf_motion_path),
          mMaxIterations(max_iterations),
          mNumSampling(num_sampling),
          mLambdaShape(lambda_shape),
          mLambdaLengthCurve(lambda_length_curve),
          mFixOriginInsertion(fix_origin_insertion) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "optimize_waypoints"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::vector<std::string> mMuscleNames;     // Muscles to optimize
    std::string mHDFMotionPath;                // Path to HDF motion file
    int mMaxIterations;                        // Maximum Ceres iterations
    int mNumSampling;                          // Number of pose samples from motion
    double mLambdaShape;                       // Weight for force direction preservation
    double mLambdaLengthCurve;                 // Weight for length curve preservation
    bool mFixOriginInsertion;                  // Fix first and last anchors
};

// ═══════════════════════════════════════════════════════════════════════════
// MirrorAnchorPositionsOp - Symmetrize anchor positions between L_/R_ pairs
// ═══════════════════════════════════════════════════════════════════════════
class MirrorAnchorPositionsOp : public SurgeryOperation {
public:
    // Mirror all L_/R_ pairs
    MirrorAnchorPositionsOp() : mMuscles({}) {}

    // Mirror specific muscles (base names without L_/R_ prefix)
    explicit MirrorAnchorPositionsOp(const std::vector<std::string>& muscleBaseNames)
        : mMuscles(muscleBaseNames) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "mirror_anchor_positions"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::vector<std::string> mMuscles;  // Base names (empty = all pairs)
};

// ============================================================================
// ContractureOptOp — Run contracture optimization (for TAL/DHL scripting)
// ============================================================================

class ContractureOptOp : public SurgeryOperation {
public:
    struct ROMTrialParam {
        std::string name;       // e.g. "dorsi_k0_L"
        double angle_deg;       // ROM angle override (0 = use default from config)
    };

    // Single search group constructor (backward compatible)
    ContractureOptOp(const std::string& search_group,
                     const std::vector<std::string>& muscles,
                     const std::vector<ROMTrialParam>& rom_trials,
                     const std::string& param_type,
                     int max_iterations = 100,
                     double min_ratio = 0.5,
                     double max_ratio = 2.0,
                     double grid_begin = 0.5,
                     double grid_end = 2.0,
                     double grid_interval = 0.05)
        : mSearchGroup(search_group),
          mMuscles(muscles),
          mROMTrials(rom_trials),
          mParamType(param_type),
          mMaxIterations(max_iterations),
          mMinRatio(min_ratio),
          mMaxRatio(max_ratio),
          mGridBegin(grid_begin),
          mGridEnd(grid_end),
          mGridInterval(grid_interval) {}

    // Multiple search groups constructor (for 2D+ joint grid search)
    ContractureOptOp(const std::map<std::string, std::vector<std::string>>& search_groups,
                     const std::vector<ROMTrialParam>& rom_trials,
                     const std::string& param_type,
                     int max_iterations = 100,
                     double min_ratio = 0.5,
                     double max_ratio = 2.0,
                     double grid_begin = 0.5,
                     double grid_end = 2.0,
                     double grid_interval = 0.05)
        : mSearchGroupMap(search_groups),
          mROMTrials(rom_trials),
          mParamType(param_type),
          mMaxIterations(max_iterations),
          mMinRatio(min_ratio),
          mMaxRatio(max_ratio),
          mGridBegin(grid_begin),
          mGridEnd(grid_end),
          mGridInterval(grid_interval) {}

    bool execute(SurgeryExecutor* executor) override;
    YAML::Node toYAML() const override;
    std::string getDescription() const override;
    std::string getType() const override { return "contracture_opt"; }

    static std::unique_ptr<SurgeryOperation> fromYAML(const YAML::Node& node);

private:
    std::string mSearchGroup;                                    // single group name (legacy)
    std::vector<std::string> mMuscles;                           // flat muscle list (legacy)
    std::map<std::string, std::vector<std::string>> mSearchGroupMap;  // multi-group: name -> muscles
    std::vector<ROMTrialParam> mROMTrials;
    std::string mParamType;   // "lt_rel" or "lm_contract"
    int mMaxIterations;
    double mMinRatio, mMaxRatio;
    double mGridBegin, mGridEnd, mGridInterval;
};

} // namespace PMuscle


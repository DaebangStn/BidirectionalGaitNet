#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <Eigen/Core>
#include "Character.h"
#include "DARTHelper.h"
#include "optimizer/WaypointOptimizer.h"

namespace PMuscle {

// Shared type for contracture trial input (used by both App and Executor)
struct ContractureTrialInput {
    std::string name;
    double rom_angle;  // degrees
};

// Modification history record for export metadata
struct MuscleModificationRecord {
    // Contracture estimation input
    std::vector<ContractureTrialInput> contracture_trials;

    // Waypoint optimization input
    std::string waypoint_motion_file;
    std::vector<std::string> waypoint_muscles;
};

// Progress callback for waypoint optimization: (current, total, muscleName)
using WaypointProgressCallback = std::function<void(int, int, const std::string&)>;

// Result callback for waypoint optimization: (current, total, muscleName, result)
using WaypointResultCallback = std::function<void(int, int, const std::string&, const WaypointOptResult&)>;

/**
 * @brief Base class for surgery operations execution
 * 
 * Provides core functionality for loading characters and executing
 * surgery operations without GUI dependencies.
 */
class SurgeryExecutor {
public:
    SurgeryExecutor(const std::string& generator_context = "unknown");
    virtual ~SurgeryExecutor();
    
    // Character loading
    void loadCharacter(const std::string& skel_path, const std::string& muscle_path, 
                       ActuatorType actuator_type = mus);
    
    // Character accessor
    Character* getCharacter() { return mCharacter; }
    const Character* getCharacter() const { return mCharacter; }
    
    // Pose preset application
    void applyPosePreset(const std::map<std::string, Eigen::VectorXd>& joint_angles);
    bool applyPosePresetByName(const std::string& preset_name);

    // Surgery operations (virtual to allow GUI-specific overrides)
    virtual void resetMuscles(const std::string& muscle_xml_path = "");
    virtual bool distributePassiveForce(const std::vector<std::string>& muscles, 
                                const std::string& reference,
                                const std::map<std::string, Eigen::VectorXd>& joint_angles = {});
    virtual bool relaxPassiveForce(const std::vector<std::string>& muscles,
                          const std::map<std::string, Eigen::VectorXd>& joint_angles = {});
    virtual bool removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex);
    virtual bool copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex, const std::string& toMuscle);
    virtual bool editAnchorPosition(const std::string& muscle, int anchor_index, const Eigen::Vector3d& position);
    virtual bool editAnchorWeights(const std::string& muscle, int anchor_index, const std::vector<double>& weights);
    virtual bool addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                            const std::string& bodynode_name, double weight);
    virtual bool removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index, int bodynode_index);
    virtual void exportMuscles(const std::string& path);
    virtual void exportSkeleton(const std::string& path);
    virtual bool rotateJointOffset(const std::string& joint_name, const Eigen::Vector3d& axis, double angle, bool preserve_position = false);
    virtual bool rotateAnchorPoints(const std::string& muscle_name, int ref_anchor_index,
                           const Eigen::Vector3d& search_direction,
                           const Eigen::Vector3d& rotation_axis, double angle);

    // FDO Combined Surgery (joint + anchor rotation)
    // Target bodynode is obtained from the reference anchor (must be single-LBS)
    virtual bool executeFDO(const std::string& ref_muscle, int ref_anchor_index,
                           const Eigen::Vector3d& search_direction,
                           const Eigen::Vector3d& rotation_axis, double angle);

    // Muscle weakening
    virtual bool weakenMuscles(const std::vector<std::string>& muscles, double strength_ratio);

    // Mirror anchor positions between L_/R_ muscle pairs
    // If muscleBaseNames is empty, mirrors all pairs
    virtual bool mirrorAnchorPositions(const std::vector<std::string>& muscleBaseNames = {});

    // Waypoint optimization using Ceres solver
    // Optimizes muscle waypoint positions to preserve force directions and length-angle curves
    // reference_character: Character with reference muscles (ideal behavior from standard character)
    // progressCallback: optional callback (current, total, muscleName) for progress updates
    virtual bool optimizeWaypoints(const std::vector<std::string>& muscle_names,
                                   const std::string& hdf_motion_path,
                                   const WaypointOptimizer::Config& config,
                                   Character* reference_character,
                                   WaypointProgressCallback progressCallback = nullptr);

    // Waypoint optimization with results - returns curve data for visualization
    // reference_character: Character with reference muscles (ideal behavior from standard character)
    // characterMutex: optional mutex to lock during skeleton/muscle access (for thread safety)
    // resultCallback: callback (current, total, muscleName, result) for progress + result updates
    virtual std::vector<WaypointOptResult> optimizeWaypointsWithResults(
                                   const std::vector<std::string>& muscle_names,
                                   const std::string& hdf_motion_path,
                                   const WaypointOptimizer::Config& config,
                                   Character* reference_character,
                                   std::mutex* characterMutex = nullptr,
                                   WaypointResultCallback resultCallback = nullptr);

    // Compute which anchors will be affected by rotation operation
    // Throws std::runtime_error if ref_anchor has multiple bodynodes
    std::vector<AnchorReference> computeAffectedAnchors(
        const AnchorReference& ref_anchor,
        const Eigen::Vector3d& search_direction) const;

    // Helper methods
    Eigen::Isometry3d getBodyNodeZeroPoseTransform(dart::dynamics::BodyNode* bn);
    bool validateAnchorReferencesBodynode(const std::string& muscle_name, int anchor_index,
                                         const std::string& bodynode_name);
    dart::dynamics::Joint* getChildJoint(dart::dynamics::BodyNode* bodynode);

    // Metadata helper methods
    std::pair<std::string, std::string> getGitInfo() const;
    std::string getCurrentTimestamp() const;
    std::string getSkeletonName() const;
    std::string getMuscleName() const;

    // Filename helpers - get base filename without path or extension
    std::string getSkeletonBaseName() const;
    std::string getMuscleBaseName() const;

    // Modification record for export metadata
    void setModificationRecord(const MuscleModificationRecord& record);

private:
    // Muscle modification history (for export metadata)
    MuscleModificationRecord mModificationRecord;
    // Muscle export helper functions
    void exportMusclesXML(const std::string& path);
    void exportMusclesYAML(const std::string& path);

    // Skeleton export helper functions
    void exportSkeletonXML(const std::string& path);
    void exportSkeletonYAML(const std::string& path);
    std::string formatRotationMatrix(const Eigen::Matrix3d& R);
    std::string formatVector3d(const Eigen::Vector3d& v);
    std::string formatMatrixYAML(const Eigen::Matrix3d& M);
    std::string formatVectorYAML(const Eigen::Vector3d& v);
    std::pair<std::string, Eigen::Vector3d> getShapeInfo(dart::dynamics::ShapePtr shape);
    std::string getJointTypeString(dart::dynamics::Joint* joint);
    std::string formatJointLimits(dart::dynamics::Joint* joint, bool isLower);
    std::string formatJointLimitsYAML(dart::dynamics::Joint* joint, bool isLower);
    std::string formatJointParams(dart::dynamics::Joint* joint, const std::string& param);
    std::string formatJointParamsYAML(dart::dynamics::Joint* joint, const std::string& param);
    bool validateSkeletonExport(const std::string& exported_path);

    // Metadata preservation helpers
    Eigen::VectorXd string_to_vectorXd(const char* str, int expected_size);
    std::string formatVectorXd(const Eigen::VectorXd& vec);
    std::string formatVectorXdYAML(const Eigen::VectorXd& vec);

protected:
    Character* mCharacter;
    std::string mSubjectSkeletonPath;  // Subject skeleton being operated on
    std::string mSubjectMusclePath;    // Subject muscle being operated on
    std::string mGeneratorContext;     // "physical_exam" or "surgery-tool: script.yaml"
};

} // namespace PMuscle


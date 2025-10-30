#pragma once

#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include "Character.h"
#include "DARTHelper.h"

namespace PMuscle {

/**
 * @brief Base class for surgery operations execution
 * 
 * Provides core functionality for loading characters and executing
 * surgery operations without GUI dependencies.
 */
class SurgeryExecutor {
public:
    SurgeryExecutor();
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

private:
    // Muscle export helper functions
    void exportMusclesXML(const std::string& path);
    void exportMusclesYAML(const std::string& path);

    // Skeleton export helper functions
    std::string formatRotationMatrix(const Eigen::Matrix3d& R);
    std::string formatVector3d(const Eigen::Vector3d& v);
    std::pair<std::string, Eigen::Vector3d> getShapeInfo(dart::dynamics::ShapePtr shape);
    std::string getJointTypeString(dart::dynamics::Joint* joint);
    std::string formatJointLimits(dart::dynamics::Joint* joint, bool isLower);
    std::string formatJointParams(dart::dynamics::Joint* joint, const std::string& param);
    bool validateSkeletonExport(const std::string& exported_path);

    // Metadata preservation helpers
    Eigen::VectorXd string_to_vectorXd(const char* str, int expected_size);
    std::string formatVectorXd(const Eigen::VectorXd& vec);

protected:
    Character* mCharacter;
    std::string mOriginalSkeletonPath;  // Cached for metadata preservation
    std::string mOriginalMusclePath;    // Cached for future reference
};

} // namespace PMuscle


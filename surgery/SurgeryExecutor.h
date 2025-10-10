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
    
    // Surgery operations (virtual to allow GUI-specific overrides)
    virtual void resetMuscles();
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
    
    // Helper method
    Eigen::Isometry3d getBodyNodeZeroPoseTransform(dart::dynamics::BodyNode* bn);

protected:
    Character* mCharacter;
};

} // namespace PMuscle


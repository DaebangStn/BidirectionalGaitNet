#include "SurgeryExecutor.h"
#include "rm/rm.hpp"
#include "Log.h"
#include "DARTHelper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <atomic>
#include <unordered_map>

#include "optimizer/WaypointOptimizer.h"
#include "HDF.h"

namespace PMuscle {

SurgeryExecutor::SurgeryExecutor(const std::string& generator_context)
    : mCharacter(nullptr), mGeneratorContext(generator_context) {
}

SurgeryExecutor::~SurgeryExecutor() {
    // Note: mCharacter is not deleted here to avoid linker issues
    // Derived classes should manage Character lifetime if they own it
}

void SurgeryExecutor::setModificationRecord(const MuscleModificationRecord& record) {
    mModificationRecord = record;
}

void SurgeryExecutor::loadCharacter(const std::string& skel_path, const std::string& muscle_path) {
    // Store subject skeleton/muscle paths
    mSubjectSkeletonPath = skel_path;
    mSubjectMusclePath = muscle_path;

    // Resolve URIs
    std::string resolved_skel = rm::resolve(skel_path);
    std::string resolved_muscle = rm::resolve(muscle_path);

    // Create character
    mCharacter = new Character(resolved_skel, true);

    // Load muscles
    mCharacter->setMuscles(resolved_muscle);

    // Zero muscle activations
    if (mCharacter->getMuscles().size() > 0) {
        mCharacter->setActivations(mCharacter->getActivations().setZero());
    }
}

void SurgeryExecutor::applyPosePreset(const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        LOG_ERROR("No character loaded");
        return;
    }

    auto skeleton = mCharacter->getSkeleton();

    for (const auto& [joint_name, angles] : joint_angles) {
        auto joint = skeleton->getJoint(joint_name);
        if (!joint) {
            LOG_ERROR("Joint not found: " << joint_name);
            continue;
        }

        if (angles.size() != joint->getNumDofs()) {
            LOG_ERROR("Joint " << joint_name << " expects " << joint->getNumDofs() << " DOFs, got " << angles.size());
            continue;
        }

        joint->setPositions(angles);
    }

    LOG_INFO("Pose preset applied");
}

bool SurgeryExecutor::applyPosePresetByName(const std::string& preset_name) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] No character loaded");
        return false;
    }

    auto skel = mCharacter->getSkeleton();

    if (preset_name == "supine") {
        // Supine pose: laying on back, face up
        // Reset all joints first
        skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

        // Rotate to lay on back
        auto root = skel->getRootJoint();
        if (root->getNumDofs() >= 6) {
            Eigen::VectorXd root_pos = root->getPositions();
            root_pos[0] = -M_PI / 2.0;  // Roll rotation (X axis)
            root_pos[4] = 0.1;          // Table height (Y translation)
            root->setPositions(root_pos);
        }

        LOG_INFO("[Surgery] Applied pose preset: supine");
        return true;

    } else if (preset_name == "standing") {
        // Standing pose: all joints at zero (default stance)
        skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

        LOG_INFO("[Surgery] Applied pose preset: standing");
        return true;

    } else if (preset_name == "prone") {
        // Prone pose: laying on front, face down
        skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

        auto root = skel->getRootJoint();
        if (root->getNumDofs() >= 6) {
            Eigen::VectorXd root_pos = root->getPositions();
            root_pos[0] = M_PI / 2.0;   // Roll rotation (X axis) - opposite direction from supine
            root_pos[4] = 0.1;          // Table height (Y translation)
            root->setPositions(root_pos);
        }

        LOG_INFO("[Surgery] Applied pose preset: prone");
        return true;

    } else {
        LOG_ERROR("[Surgery] Unknown pose preset: " << preset_name);
        return false;
    }
}

void SurgeryExecutor::resetMuscles(const std::string& muscle_xml_path) {
    if (!mCharacter) return;

    // If XML path provided, reload muscles from file (resets anchors + parameters)
    if (!muscle_xml_path.empty()) {
        // Resolve URI before loading
        std::string resolved_path = rm::resolve(muscle_xml_path);

        LOG_INFO("[Surgery] Reloading muscles from XML: " << resolved_path);

        // CRITICAL: Muscle XML files store anchor positions in global coordinates
        // assuming the skeleton is in ZERO POSE. We must temporarily set the skeleton
        // to zero pose while loading so that global->local coordinate conversion is correct.

        // 1. Save current skeleton state
        auto skeleton = mCharacter->getSkeleton();
        Eigen::VectorXd saved_positions = skeleton->getPositions();
        Eigen::VectorXd saved_velocities = skeleton->getVelocities();

        // 2. Clear existing muscles
        mCharacter->clearMuscles();

        // 3. Set skeleton to zero pose (the pose muscles were authored in)
        skeleton->setPositions(Eigen::VectorXd::Zero(skeleton->getNumDofs()));
        skeleton->setVelocities(Eigen::VectorXd::Zero(skeleton->getNumDofs()));

        // 4. Load muscles (global positions will be correctly converted to local)
        mCharacter->setMuscles(resolved_path);

        // 5. Restore original skeleton state
        skeleton->setPositions(saved_positions);
        skeleton->setVelocities(saved_velocities);

        LOG_INFO("[Surgery] Muscles reloaded from XML (anchors + parameters reset)");
        return;
    }

    // Otherwise, just reset parameters (existing behavior - anchors unchanged)
    LOG_INFO("[Surgery] Resetting muscle parameters only...");

    auto muscles = mCharacter->getMuscles();
    int resetCount = 0;
    for (auto muscle : muscles) {
        muscle->change_f(1.0);
        muscle->change_l(1.0);
        muscle->SetTendonOffset(0.0);
        resetCount++;
    }

    LOG_INFO("[Surgery] Muscle parameter reset complete. Reset " << resetCount << " muscles.");
}

bool SurgeryExecutor::distributePassiveForce(const std::vector<std::string>& muscles,
                                            const std::string& reference,
                                            const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }

    // Apply joint angles if specified
    if (!joint_angles.empty()) {
        applyPosePreset(joint_angles);
        LOG_INFO("[Surgery] Applied " << joint_angles.size() << " joint angle(s)");
    }
    
    auto all_muscles = mCharacter->getMuscles();
    
    // Find reference muscle
    Muscle* refMuscle = nullptr;
    for (auto m : all_muscles) {
        if (m->name == reference) {
            refMuscle = m;
            break;
        }
    }
    
    if (!refMuscle) {
        LOG_ERROR("[Surgery] Error: Reference muscle '" << reference << "' not found!");
        return false;
    }
    
    double refCoeff = refMuscle->lm_norm;
    int modifiedCount = 0;
    
    // Apply to all selected muscles
    for (auto m : all_muscles) {
        if (std::find(muscles.begin(), muscles.end(), m->name) != muscles.end()) {
            m->SetLmNorm(refCoeff);
            modifiedCount++;
        }
    }
    
    LOG_INFO("[Surgery] Distributed passive force coefficient " << refCoeff << " from '" << reference << "' to " << modifiedCount << " muscles");
    
    return true;
}

bool SurgeryExecutor::relaxPassiveForce(const std::vector<std::string>& muscles,
                                       const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }
    
    // Apply joint angles if specified
    if (!joint_angles.empty()) {
        applyPosePreset(joint_angles);
        LOG_INFO("[Surgery] Applied " << joint_angles.size() << " joint angle(s)");
    }

    auto all_muscles = mCharacter->getMuscles();
    int relaxedCount = 0;
    
    // Apply relaxation to selected muscles
    for (auto m : all_muscles) {
        if (std::find(muscles.begin(), muscles.end(), m->name) != muscles.end()) {
            m->RelaxPassiveForce();
            relaxedCount++;
        }
    }
    
    LOG_INFO("[Surgery] Applied relaxation to " << relaxedCount << " muscles");
    
    return true;
}

bool SurgeryExecutor::removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }
    
    auto muscles = mCharacter->getMuscles();
    Muscle* targetMuscle = nullptr;
    for (auto m : muscles) {
        if (m->name == muscleName) {
            targetMuscle = m;
            break;
        }
    }
    
    if (!targetMuscle) {
        LOG_ERROR("[Surgery] Error: Muscle '" << muscleName << "' not found!");
        return false;
    }

    if (anchorIndex < 0 || anchorIndex >= targetMuscle->mAnchors.size()) {
        LOG_ERROR("[Surgery] Error: Invalid anchor index " << anchorIndex);
        return false;
    }

    LOG_INFO("[Surgery] Removing anchor #" << anchorIndex << " from muscle '" << muscleName << "'...");
    
    // Remove the anchor
    delete targetMuscle->mAnchors[anchorIndex];  // Free memory
    targetMuscle->mAnchors.erase(targetMuscle->mAnchors.begin() + anchorIndex);
    
    // Recalculate muscle parameters
    targetMuscle->SetMuscle();

    LOG_INFO("[Surgery] Anchor removal complete. Muscle now has " << targetMuscle->mAnchors.size() << " anchors.");
    
    return true;
}

bool SurgeryExecutor::copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex, const std::string& toMuscle) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }
    
    auto muscles = mCharacter->getMuscles();
    Muscle* sourceMuscle = nullptr;
    Muscle* targetMuscle = nullptr;
    
    for (auto m : muscles) {
        if (m->name == fromMuscle) {
            sourceMuscle = m;
        }
        if (m->name == toMuscle) {
            targetMuscle = m;
        }
    }
    
    if (!sourceMuscle) {
        LOG_ERROR("[Surgery] Error: Source muscle '" << fromMuscle << "' not found!");
        return false;
    }

    if (!targetMuscle) {
        LOG_ERROR("[Surgery] Error: Target muscle '" << toMuscle << "' not found!");
        return false;
    }

    auto sourceAnchors = sourceMuscle->GetAnchors();
    if (fromIndex < 0 || fromIndex >= sourceAnchors.size()) {
        LOG_ERROR("[Surgery] Error: Invalid anchor index " << fromIndex);
        return false;
    }

    auto sourceAnchor = sourceAnchors[fromIndex];

    LOG_INFO("[Surgery] Copying anchor #" << fromIndex << " from '" << fromMuscle << "' to '" << toMuscle << "'...");
    
    // Create a deep copy of the anchor
    std::vector<dart::dynamics::BodyNode*> newBodyNodes = sourceAnchor->bodynodes;
    std::vector<Eigen::Vector3d> newLocalPositions = sourceAnchor->local_positions;
    std::vector<double> newWeights = sourceAnchor->weights;
    
    Anchor* newAnchor = new Anchor(newBodyNodes, newLocalPositions, newWeights);
    
    // Add the new anchor to the target muscle
    targetMuscle->mAnchors.push_back(newAnchor);
    
    // Recalculate muscle parameters
    targetMuscle->SetMuscle();

    LOG_INFO("[Surgery] Anchor copied successfully. Target muscle now has " << targetMuscle->mAnchors.size() << " anchors.");

    // Display info about the copied anchor
    if (!newBodyNodes.empty()) {
        LOG_INFO("[Surgery]   Copied anchor attached to: " << newBodyNodes[0]->getName());
        if (newBodyNodes.size() > 1) {
            LOG_INFO("[Surgery]   (LBS with " << newBodyNodes.size() << " body nodes)");
        }
    }
    
    return true;
}

bool SurgeryExecutor::editAnchorPosition(const std::string& muscle, int anchor_index, 
                                        const Eigen::Vector3d& position) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }
    
    auto muscles = mCharacter->getMuscles();
    Muscle* targetMuscle = nullptr;
    for (auto m : muscles) {
        if (m->name == muscle) {
            targetMuscle = m;
            break;
        }
    }
    
    if (!targetMuscle) {
        LOG_ERROR("[Surgery] Error: Muscle '" << muscle << "' not found!");
        return false;
    }

    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        LOG_ERROR("[Surgery] Error: Invalid anchor index " << anchor_index);
        return false;
    }

    auto anchor = anchors[anchor_index];

    // Update anchor position for ALL bodynodes in this anchor
    for (size_t i = 0; i < anchor->local_positions.size(); ++i) {
        anchor->local_positions[i] = position;
    }

    targetMuscle->SetMuscle();

    LOG_INFO("[Surgery] Updated position for anchor #" << anchor_index << " in '" << muscle << "' to [" << position[0] << ", " << position[1] << ", " << position[2] << "]");
    
    return true;
}

bool SurgeryExecutor::editAnchorWeights(const std::string& muscle, int anchor_index,
                                       const std::vector<double>& weights) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }

    auto muscles = mCharacter->getMuscles();
    Muscle* targetMuscle = nullptr;
    for (auto m : muscles) {
        if (m->name == muscle) {
            targetMuscle = m;
            break;
        }
    }

    if (!targetMuscle) {
        LOG_ERROR("[Surgery] Error: Muscle '" << muscle << "' not found!");
        return false;
    }

    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        LOG_ERROR("[Surgery] Error: Invalid anchor index " << anchor_index);
        return false;
    }
    
    auto anchor = anchors[anchor_index];
    
    if (weights.size() != anchor->weights.size()) {
        std::cerr << "[Surgery] Error: Weight count mismatch. Expected " << anchor->weights.size()
                  << ", got " << weights.size() << std::endl;
        return false;
    }
    
    // Update weights in anchor
    for (size_t i = 0; i < weights.size(); ++i) {
        anchor->weights[i] = weights[i];
    }
    
    targetMuscle->SetMuscle();

    LOG_INFO("[Surgery] Updated weights for anchor #" << anchor_index << " in '" << muscle << "'");

    return true;
}

bool SurgeryExecutor::addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                                         const std::string& bodynode_name, double weight) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }

    auto muscles = mCharacter->getMuscles();
    Muscle* targetMuscle = nullptr;
    for (auto m : muscles) {
        if (m->name == muscle) {
            targetMuscle = m;
            break;
        }
    }

    if (!targetMuscle) {
        LOG_ERROR("[Surgery] Error: Muscle '" << muscle << "' not found!");
        return false;
    }

    auto skel = mCharacter->getSkeleton();
    auto newBodyNode = skel->getBodyNode(bodynode_name);
    if (!newBodyNode) {
        LOG_ERROR("[Surgery] Error: Body node '" << bodynode_name << "' not found!");
        return false;
    }

    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        LOG_ERROR("[Surgery] Error: Invalid anchor index " << anchor_index);
        return false;
    }
    
    auto anchor = anchors[anchor_index];
    
    // Check if bodynode already exists in anchor
    for (auto bn : anchor->bodynodes) {
        if (bn == newBodyNode) {
            LOG_ERROR("[Surgery] Error: Body node already exists in this anchor!");
            return false;
        }
    }
    
    // Add new bodynode with same local position as first one
    anchor->bodynodes.push_back(newBodyNode);
    anchor->weights.push_back(weight);
    
    if (!anchor->local_positions.empty()) {
        anchor->local_positions.push_back(anchor->local_positions[0]);
    } else {
        anchor->local_positions.push_back(Eigen::Vector3d::Zero());
    }
    
    targetMuscle->SetMuscle();
    
    std::cout << "[Surgery] Added body node '" << bodynode_name << "' to anchor #" << anchor_index 
              << " in '" << muscle << "' with weight " << weight << std::endl;
    
    return true;
}

bool SurgeryExecutor::removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index,
                                              int bodynode_index) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }

    auto muscles = mCharacter->getMuscles();
    Muscle* targetMuscle = nullptr;
    for (auto m : muscles) {
        if (m->name == muscle) {
            targetMuscle = m;
            break;
        }
    }

    if (!targetMuscle) {
        LOG_ERROR("[Surgery] Error: Muscle '" << muscle << "' not found!");
        return false;
    }

    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        LOG_ERROR("[Surgery] Error: Invalid anchor index " << anchor_index);
        return false;
    }

    auto anchor = anchors[anchor_index];

    if (anchor->bodynodes.size() <= 1) {
        LOG_ERROR("[Surgery] Error: Cannot remove last body node from anchor!");
        return false;
    }

    if (bodynode_index < 0 || bodynode_index >= anchor->bodynodes.size()) {
        LOG_ERROR("[Surgery] Error: Invalid body node index " << bodynode_index);
        return false;
    }

    // Remove this bodynode from anchor
    std::string removed_name = anchor->bodynodes[bodynode_index]->getName();
    anchor->bodynodes.erase(anchor->bodynodes.begin() + bodynode_index);
    anchor->weights.erase(anchor->weights.begin() + bodynode_index);
    anchor->local_positions.erase(anchor->local_positions.begin() + bodynode_index);

    targetMuscle->SetMuscle();

    LOG_INFO("[Surgery] Removed body node '" << removed_name << "' from anchor #" << anchor_index << " in '" << muscle << "'");
    
    return true;
}

Eigen::Isometry3d SurgeryExecutor::getBodyNodeZeroPoseTransform(dart::dynamics::BodyNode* bn) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();

    // Build chain from body node to root
    std::vector<dart::dynamics::BodyNode*> chain;
    dart::dynamics::BodyNode* current = bn;
    while (current != nullptr) {
        chain.push_back(current);
        current = current->getParentBodyNode();
    }

    // Walk from root down to target body node, accumulating transforms
    // chain is in reverse order (bn -> ... -> root), so iterate backwards
    for (int i = chain.size() - 1; i >= 0; --i) {
        auto body = chain[i];
        auto joint = body->getParentJoint();

        if (joint == nullptr) continue;  // Skip root joint

        // Get joint's fixed transforms
        Eigen::Isometry3d parentTransform = joint->getTransformFromParentBodyNode();
        Eigen::Isometry3d childTransform = joint->getTransformFromChildBodyNode();

        // Get joint transform with zero DOF values (reference pose)
        Eigen::VectorXd zeroPos = Eigen::VectorXd::Zero(joint->getNumDofs());

        // Save current joint positions
        Eigen::VectorXd currentPos = joint->getPositions();

        // Temporarily set to zero to get the transform
        joint->setPositions(zeroPos);
        Eigen::Isometry3d jointTransform = joint->getRelativeTransform();

        // Restore current positions
        joint->setPositions(currentPos);

        // Accumulate transform
        transform = transform * parentTransform * jointTransform * childTransform.inverse();
    }

    return transform;
}

void SurgeryExecutor::exportMuscles(const std::string& path) {
    if (!mCharacter) {
        throw std::runtime_error("No character loaded");
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        throw std::runtime_error("No muscles found in character");
    }

    // Resolve URI path if needed
    std::string resolved_path = rm::resolve(path);

    // Auto-detect format from file extension
    std::string ext;
    size_t dot_pos = resolved_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = resolved_path.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    LOG_INFO("[Surgery] Saving muscle configuration to: " << resolved_path);

    if (ext == ".yaml" || ext == ".yml") {
        exportMusclesYAML(resolved_path);
    } else {
        exportMusclesXML(resolved_path);  // Default to XML for backward compatibility
    }

    // Update subject muscle path to the exported file (use original URI path, not resolved)
    mSubjectMusclePath = path;
    LOG_INFO("[Surgery] Updated subject muscle path to: " << mSubjectMusclePath);
}

void SurgeryExecutor::exportMusclesXML(const std::string& path) {
    if (!mCharacter) {
        throw std::runtime_error("No character loaded");
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        throw std::runtime_error("No muscles found in character");
    }

    std::ofstream mfs(path);
    if (!mfs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    LOG_INFO("[Surgery] Saving muscle configuration to: " << path);

    // Save current skeleton state
    auto skel = mCharacter->getSkeleton();
    Eigen::VectorXd saved_positions = skel->getPositions();

    // Move to zero pose (all joint angles = 0)
    Eigen::VectorXd zero_positions = Eigen::VectorXd::Zero(skel->getNumDofs());
    skel->setPositions(zero_positions);

    // Write metadata section as XML comment
    auto [git_hash, git_message] = getGitInfo();
    mfs << "<!-- " << std::endl;
    mfs << "<Metadata>" << std::endl;
    mfs << "  <generator>" << mGeneratorContext << "</generator>" << std::endl;
    mfs << "  <timestamp>" << getCurrentTimestamp() << "</timestamp>" << std::endl;
    mfs << "  <version>v1</version>" << std::endl;
    mfs << "  <skeleton>" << getSkeletonName() << "</skeleton>" << std::endl;
    if (!git_hash.empty()) {
        mfs << "  <git_commit>" << git_hash << "</git_commit>" << std::endl;
        if (!git_message.empty()) {
            mfs << "  <git_message>" << git_message << "</git_message>" << std::endl;
        }
    }
    mfs << "</Metadata>" << std::endl;
    mfs << "-->" << std::endl;

    mfs << "<Muscle>" << std::endl;

    for (auto m : muscles) {
        std::string name = m->name;
        double f0 = m->f0;
        double l_m0 = m->lm_contract;
        double l_t0 = m->lt_rel;

        mfs << "    <Unit name=\"" << name
            << "\" f0=\"" << f0
            << "\" lm=\"" << l_m0
            << "\" lt=\"" << l_t0
            << "\">" << std::endl;

        for (auto anchor : m->GetAnchors()) {
            // Use first body node (index 0) for consistency with symmetry checking
            // The LBS system may have multiple body nodes, but for XML export we use the first
            auto body_node = anchor->bodynodes[0];
            std::string body_name = body_node->getName();

            // Get LOCAL position (pose-independent)
            Eigen::Vector3d local_position = anchor->local_positions[0];

            // Get body node's transform in zero pose (skeleton is now in zero pose)
            Eigen::Isometry3d zero_pose_transform = body_node->getWorldTransform();

            // Transform to global position in zero pose
            Eigen::Vector3d glob_position = zero_pose_transform * local_position;

            mfs << "        <Waypoint body=\"" << body_name
                << "\" p=\"" << glob_position[0] << " "
                << glob_position[1] << " "
                << glob_position[2] << " \"/>" << std::endl;
        }

        mfs << "    </Unit>" << std::endl;
    }

    mfs << "</Muscle>" << std::endl;
    mfs.close();

    // Restore original skeleton state
    skel->setPositions(saved_positions);

    LOG_INFO("[Surgery] Successfully saved " << muscles.size() << " muscles to " << path);
}

// Helper function to compute body node size for normalization
static Eigen::Vector3d getBodyNodeSize(dart::dynamics::BodyNode* body_node) {
    if (!body_node || body_node->getNumShapeNodes() == 0) {
        return Eigen::Vector3d(1.0, 1.0, 1.0);  // Default fallback
    }

    // Use the first shape node (primary shape)
    auto shape = body_node->getShapeNode(0)->getShape();

    if (auto box = std::dynamic_pointer_cast<dart::dynamics::BoxShape>(shape)) {
        return box->getSize();
    } else if (auto sphere = std::dynamic_pointer_cast<dart::dynamics::SphereShape>(shape)) {
        double radius = sphere->getRadius();
        return Eigen::Vector3d(radius, radius, radius);
    } else if (auto capsule = std::dynamic_pointer_cast<dart::dynamics::CapsuleShape>(shape)) {
        double radius = capsule->getRadius();
        double height = capsule->getHeight();
        return Eigen::Vector3d(radius, height, radius);
    } else if (auto cylinder = std::dynamic_pointer_cast<dart::dynamics::CylinderShape>(shape)) {
        double radius = cylinder->getRadius();
        double height = cylinder->getHeight();
        return Eigen::Vector3d(radius, height, radius);
    } else if (auto mesh = std::dynamic_pointer_cast<dart::dynamics::MeshShape>(shape)) {
        Eigen::Vector3d scale = mesh->getScale();
        // Use scale as a proxy for size (assumes unit mesh)
        return scale.cwiseAbs().cwiseMax(Eigen::Vector3d(1e-6, 1e-6, 1e-6));
    }

    // Fallback for unknown shape types
    return Eigen::Vector3d(1.0, 1.0, 1.0);
}

void SurgeryExecutor::exportMusclesYAML(const std::string& path) {
    if (!mCharacter) {
        throw std::runtime_error("No character loaded");
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        throw std::runtime_error("No muscles found in character");
    }

    std::ofstream mfs(path);
    if (!mfs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    LOG_INFO("[Surgery] Saving muscle configuration to YAML: " << path);

    // Write metadata section
    auto [git_hash, git_message] = getGitInfo();
    mfs << "metadata:" << std::endl;
    mfs << "  generator: \"" << mGeneratorContext << "\"" << std::endl;
    mfs << "  timestamp: \"" << getCurrentTimestamp() << "\"" << std::endl;
    mfs << "  version: v1" << std::endl;
    mfs << "  skeleton: \"" << getSkeletonName() << "\"" << std::endl;
    mfs << "  muscle_from: \"" << getMuscleName() << "\"" << std::endl;
    if (!git_hash.empty()) {
        mfs << "  git_commit: \"" << git_hash << "\"" << std::endl;
        if (!git_message.empty()) {
            mfs << "  git_message: \"" << git_message << "\"" << std::endl;
        }
    }
    mfs << std::endl;

    // Write modifications section if any modifications were recorded
    bool hasContracture = !mModificationRecord.contracture_trials.empty();
    bool hasWaypoint = !mModificationRecord.waypoint_muscles.empty();

    if (hasContracture || hasWaypoint) {
        mfs << "modifications:" << std::endl;

        if (hasContracture) {
            mfs << "  contracture_estimation:" << std::endl;
            mfs << "    trials:" << std::endl;
            for (const auto& trial : mModificationRecord.contracture_trials) {
                mfs << "      - {name: \"" << trial.name << "\", rom_angle: "
                    << std::fixed << std::setprecision(1) << trial.rom_angle << "}" << std::endl;
            }
            const auto& cc = mModificationRecord.contracture_config;
            mfs << "    parameters:" << std::endl;
            mfs << "      max_iterations: " << cc.maxIterations << std::endl;
            mfs << "      outer_iterations: " << cc.outerIterations << std::endl;
            mfs << std::fixed << std::setprecision(2);
            mfs << "      min_ratio: " << cc.minRatio << std::endl;
            mfs << "      max_ratio: " << cc.maxRatio << std::endl;
            mfs << "      grid_begin: " << cc.gridSearchBegin << std::endl;
            mfs << "      grid_end: " << cc.gridSearchEnd << std::endl;
            mfs << "      grid_interval: " << cc.gridSearchInterval << std::endl;
            mfs << std::setprecision(3);
            mfs << "      lambda_ratio_reg: " << cc.lambdaRatioReg << std::endl;
            mfs << "      lambda_torque_reg: " << cc.lambdaTorqueReg << std::endl;
            mfs << "      lambda_line_reg: " << cc.lambdaLineReg << std::endl;
        }

        if (hasWaypoint) {
            mfs << "  waypoint_optimization:" << std::endl;
            mfs << "    motion_file: \"" << mModificationRecord.waypoint_motion_file << "\"" << std::endl;
            mfs << "    muscles: [";
            for (size_t i = 0; i < mModificationRecord.waypoint_muscles.size(); ++i) {
                if (i > 0) mfs << ", ";
                mfs << "\"" << mModificationRecord.waypoint_muscles[i] << "\"";
            }
            mfs << "]" << std::endl;
            const auto& wc = mModificationRecord.waypoint_config;
            mfs << "    parameters:" << std::endl;
            mfs << "      max_iterations: " << wc.maxIterations << std::endl;
            mfs << "      num_sampling: " << wc.numSampling << std::endl;
            mfs << std::fixed << std::setprecision(3);
            mfs << "      lambda_shape: " << wc.lambdaShape << std::endl;
            mfs << "      lambda_length_curve: " << wc.lambdaLengthCurve << std::endl;
            mfs << "      weight_phase: " << wc.weightPhase << std::endl;
            mfs << "      weight_delta: " << wc.weightDelta << std::endl;
            mfs << "      weight_samples: " << wc.weightSamples << std::endl;
            mfs << "      num_phase_samples: " << wc.numPhaseSamples << std::endl;
            mfs << "      loss_power: " << wc.lossPower << std::endl;
            mfs << "      num_parallel: " << wc.numParallel << std::endl;
            mfs << "      length_type: " << (wc.lengthType == LengthCurveType::NORMALIZED ? "normalized" : "mtu") << std::endl;
            mfs << "      fix_origin_insertion: " << (wc.fixOriginInsertion ? "true" : "false") << std::endl;
            mfs << "      adaptive_sample_weight: " << (wc.adaptiveSampleWeight ? "true" : "false") << std::endl;
            mfs << "      multi_dof_joint_sweep: " << (wc.multiDofJointSweep ? "true" : "false") << std::endl;
            mfs << "      max_displacement: " << wc.maxDisplacement << std::endl;
            mfs << "      max_displacement_origin_insertion: " << wc.maxDisplacementOriginInsertion << std::endl;
            mfs << std::scientific << std::setprecision(1);
            mfs << "      function_tolerance: " << wc.functionTolerance << std::endl;
            mfs << "      gradient_tolerance: " << wc.gradientTolerance << std::endl;
            mfs << "      parameter_tolerance: " << wc.parameterTolerance << std::endl;
        }

        mfs << std::endl;
    }

    // Sort muscles by L/R pairs (maintain symmetry structure for Character loading)
    // Expected format: L_muscle1, R_muscle1, L_muscle2, R_muscle2, ...
    std::vector<Muscle*> sorted_muscles = muscles;
    std::sort(sorted_muscles.begin(), sorted_muscles.end(),
              [](Muscle* a, Muscle* b) {
                  // Extract base name (without L_/R_ prefix)
                  auto get_base = [](const std::string& name) -> std::string {
                      if (name.length() > 2 && (name[0] == 'L' || name[0] == 'R') && name[1] == '_') {
                          return name.substr(2);
                      }
                      return name;
                  };

                  std::string base_a = get_base(a->name);
                  std::string base_b = get_base(b->name);

                  // First sort by base name
                  if (base_a != base_b) {
                      return base_a < base_b;
                  }

                  // Then L before R for same base name
                  return a->name < b->name;
              });

    // Calculate max body node name length for visual alignment
    size_t max_body_len = 0;
    for (auto m : sorted_muscles) {
        for (auto anchor : m->GetAnchors()) {
            for (size_t i = 0; i < anchor->bodynodes.size(); ++i) {
                size_t len = anchor->bodynodes[i]->getName().length();
                if (len > max_body_len) max_body_len = len;
            }
        }
    }

    // Write YAML header
    mfs << "muscles:" << std::endl;

    // Write each muscle with compact flow-style formatting
    for (auto m : sorted_muscles) {
        std::string name = m->name;
        double f0 = m->f0;
        double lm = m->lm_contract;
        double lt = m->lt_rel;

        // Safety check: lm_contract must be positive (minimum 0.01)
        // Zero or negative values indicate a bug in the optimization pipeline
        if (lm < 0.01) {
            LOG_ERROR("[Surgery] BUG: Muscle " << name << " has invalid lm_contract: " << lm
                      << " - This indicates muscles not in any group lost their values. "
                      << "Setting to 1.0 as fallback.");
            lm = 1.0;  // Safe fallback to prevent downstream errors
        }

        // Start muscle entry with properties
        mfs << "  - {name: " << name
            << ", f0: " << std::fixed << std::setprecision(2) << f0
            << ", lm_contract: " << std::fixed << std::setprecision(5) << lm
            << ", lt_rel: " << std::fixed << std::setprecision(5) << lt << "," << std::endl;

        // Waypoints array (flow style)
        mfs << "     waypoints: [" << std::endl;

        // Write each anchor as nested array
        bool first_anchor = true;
        for (auto anchor : m->GetAnchors()) {
            if (anchor->bodynodes.empty()) continue;

            if (!first_anchor) mfs << "," << std::endl;
            first_anchor = false;

            // Start anchor array
            mfs << "       [";

            // Export ALL bodynodes in this anchor (multi-LBS support)
            for (size_t i = 0; i < anchor->bodynodes.size(); ++i) {
                auto body_node = anchor->bodynodes[i];
                std::string body_name = body_node->getName();
                Eigen::Vector3d local_pos = anchor->local_positions[i];
                double weight = anchor->weights[i];

                // Normalize local position by body node size
                Eigen::Vector3d bn_size = getBodyNodeSize(body_node);
                Eigen::Vector3d normalized_pos = local_pos.cwiseQuotient(bn_size);

                if (i > 0) mfs << ", ";

                // Format body entry in flow style with padded body name
                mfs << "{body: " << std::left << std::setw(max_body_len) << body_name << ", p: [";

                // Format coordinates with 5 decimals and sign alignment (normalized)
                for (int j = 0; j < 3; ++j) {
                    if (j > 0) mfs << ", ";
                    // Add leading space for positive numbers for alignment
                    if (normalized_pos[j] >= 0.0) {
                        mfs << " " << std::fixed << std::setprecision(5) << normalized_pos[j];
                    } else {
                        mfs << std::fixed << std::setprecision(5) << normalized_pos[j];
                    }
                }

                mfs << "], w: " << std::fixed << std::setprecision(4) << weight << "}";
            }

            // Close anchor array
            mfs << "]";
        }

        // Close waypoints array and muscle entry
        mfs << std::endl << "     ]}" << std::endl;
    }

    mfs.close();

    LOG_INFO("[Surgery] Successfully saved " << sorted_muscles.size() << " muscles to " << path << " (YAML format)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Skeleton Export Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

std::string SurgeryExecutor::formatRotationMatrix(const Eigen::Matrix3d& R) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << R(0,0) << " " << R(0,1) << " " << R(0,2) << " "
        << R(1,0) << " " << R(1,1) << " " << R(1,2) << " "
        << R(2,0) << " " << R(2,1) << " " << R(2,2);
    return oss.str();
}

std::string SurgeryExecutor::formatVector3d(const Eigen::Vector3d& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << v[0] << " " << v[1] << " " << v[2];
    return oss.str();
}

std::string SurgeryExecutor::formatMatrixYAML(const Eigen::Matrix3d& M) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    // Format each row with proper spacing
    oss << "[[";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << M(0, i);
    }
    oss << "], [";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << M(1, i);
    }
    oss << "], [";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << M(2, i);
    }
    oss << "]]";

    return oss.str();
}

std::string SurgeryExecutor::formatVectorYAML(const Eigen::Vector3d& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "[";
    for (int i = 0; i < 3; i++) {
        if (i > 0) oss << ", ";
        oss << std::setw(7) << v[i];
    }
    oss << "]";
    return oss.str();
}

std::pair<std::string, Eigen::Vector3d> SurgeryExecutor::getShapeInfo(dart::dynamics::ShapePtr shape) {
    if (auto box = std::dynamic_pointer_cast<dart::dynamics::BoxShape>(shape)) {
        return {"Box", box->getSize()};
    } else if (auto sphere = std::dynamic_pointer_cast<dart::dynamics::SphereShape>(shape)) {
        double r = sphere->getRadius();
        return {"Sphere", Eigen::Vector3d(r, r, r)};
    } else if (auto capsule = std::dynamic_pointer_cast<dart::dynamics::CapsuleShape>(shape)) {
        double r = capsule->getRadius();
        double h = capsule->getHeight();
        return {"Capsule", Eigen::Vector3d(r, h, r)};
    } else if (auto cylinder = std::dynamic_pointer_cast<dart::dynamics::CylinderShape>(shape)) {
        double r = cylinder->getRadius();
        double h = cylinder->getHeight();
        return {"Cylinder", Eigen::Vector3d(r, h, r)};
    } else if (auto mesh = std::dynamic_pointer_cast<dart::dynamics::MeshShape>(shape)) {
        return {"Mesh", Eigen::Vector3d(0, 0, 0)};
    }
    return {"Box", Eigen::Vector3d(0.1, 0.1, 0.1)};  // Default fallback
}

std::string SurgeryExecutor::getJointTypeString(dart::dynamics::Joint* joint) {
    if (dynamic_cast<dart::dynamics::FreeJoint*>(joint)) {
        return "Free";
    } else if (dynamic_cast<dart::dynamics::BallJoint*>(joint)) {
        return "Ball";
    } else if (dynamic_cast<dart::dynamics::RevoluteJoint*>(joint)) {
        return "Revolute";
    } else if (dynamic_cast<dart::dynamics::PrismaticJoint*>(joint)) {
        return "Prismatic";
    } else if (dynamic_cast<dart::dynamics::WeldJoint*>(joint)) {
        return "Weld";
    }
    return "Ball";  // Default fallback
}

std::string SurgeryExecutor::formatJointLimits(dart::dynamics::Joint* joint, bool isLower) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    size_t numDofs = joint->getNumDofs();
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << " ";
        if (isLower) {
            oss << joint->getPositionLowerLimit(i);
        } else {
            oss << joint->getPositionUpperLimit(i);
        }
    }
    return oss.str();
}

std::string SurgeryExecutor::formatJointLimitsYAML(dart::dynamics::Joint* joint, bool isLower) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "[";

    size_t numDofs = joint->getNumDofs();
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << ", ";
        oss << std::setw(7);
        if (isLower) {
            oss << joint->getPositionLowerLimit(i);
        } else {
            oss << joint->getPositionUpperLimit(i);
        }
    }
    oss << "]";
    return oss.str();
}

std::string SurgeryExecutor::formatJointParams(dart::dynamics::Joint* joint, const std::string& param) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);

    size_t numDofs = joint->getNumDofs();
    const Eigen::VectorXd* gains = nullptr;
    if (mCharacter) {
        if (param == "kp") {
            gains = &mCharacter->getKpVector();
        } else if (param == "kv") {
            gains = &mCharacter->getKvVector();
        }
    }
    Eigen::Index baseIndex =
        (numDofs > 0 && joint->getSkeleton()) ? static_cast<Eigen::Index>(joint->getIndexInSkeleton(0)) : 0;
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << " ";
        double value = 0.0;
        Eigen::Index gainIndex = baseIndex + static_cast<Eigen::Index>(i);
        if (gains && gainIndex < gains->size()) {
            value = (*gains)[gainIndex];
        } else if (param == "kp") {
            value = joint->getSpringStiffness(i);
        } else if (param == "kv") {
            value = joint->getDampingCoefficient(i);
        }
        oss << value;
    }
    return oss.str();
}

std::string SurgeryExecutor::formatJointParamsYAML(dart::dynamics::Joint* joint, const std::string& param) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";

    size_t numDofs = joint->getNumDofs();
    const Eigen::VectorXd* gains = nullptr;
    if (mCharacter) {
        if (param == "kp") {
            gains = &mCharacter->getKpVector();
        } else if (param == "kv") {
            gains = &mCharacter->getKvVector();
        }
    }
    Eigen::Index baseIndex =
        (numDofs > 0 && joint->getSkeleton()) ? static_cast<Eigen::Index>(joint->getIndexInSkeleton(0)) : 0;
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << ", ";
        oss << std::setw(5);
        double value = 0.0;
        Eigen::Index gainIndex = baseIndex + static_cast<Eigen::Index>(i);
        if (gains && gainIndex < gains->size()) {
            value = (*gains)[gainIndex];
        } else if (param == "kp") {
            value = joint->getSpringStiffness(i);
        } else if (param == "kv") {
            value = joint->getDampingCoefficient(i);
        }
        oss << value;
    }
    oss << "]";
    return oss.str();
}

// Parse space-separated string to VectorXd
Eigen::VectorXd SurgeryExecutor::string_to_vectorXd(const char* str, int expected_size) {
    std::vector<double> values;
    std::istringstream iss(str);
    double val;
    while (iss >> val) {
        values.push_back(val);
    }

    Eigen::VectorXd result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = values[i];
    }
    return result;
}

// Format VectorXd to space-separated string
std::string SurgeryExecutor::formatVectorXd(const Eigen::VectorXd& vec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    for (int i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << " ";
        oss << vec[i];
    }
    return oss.str();
}

std::string SurgeryExecutor::formatVectorXdYAML(const Eigen::VectorXd& vec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";
    for (int i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << std::setw(5) << vec[i];
    }
    oss << "]";
    return oss.str();
}


void SurgeryExecutor::exportSkeleton(const std::string& path) {
    if (!mCharacter) {
        throw std::runtime_error("No character loaded");
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        throw std::runtime_error("No skeleton found in character");
    }

    // Resolve URI path if needed
    std::string resolved_path = rm::resolve(path);

    // Detect format from file extension
    std::string ext;
    size_t dot_pos = resolved_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = resolved_path.substr(dot_pos);
        // Convert to lowercase for comparison
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    LOG_INFO("[Surgery] Saving skeleton configuration to: " << resolved_path);

    if (ext == ".yaml" || ext == ".yml") {
        exportSkeletonYAML(resolved_path);
    } else {
        // Default to XML for .xml or unrecognized extensions
        exportSkeletonXML(resolved_path);
    }

    // Update subject skeleton path to the exported file (use original URI path, not resolved)
    mSubjectSkeletonPath = path;
    LOG_INFO("[Surgery] Updated subject skeleton path to: " << mSubjectSkeletonPath);
}

void SurgeryExecutor::exportSkeletonXML(const std::string& path) {
    auto skel = mCharacter->getSkeleton();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Save current skeleton state
    Eigen::VectorXd saved_positions = skel->getPositions();

    // Move to zero pose (all joint angles = 0)
    Eigen::VectorXd zero_positions = Eigen::VectorXd::Zero(skel->getNumDofs());
    skel->setPositions(zero_positions);

    // Get metadata from Character instance (parsed during construction)
    const auto& contactFlags = mCharacter->getContactFlags();
    const auto& objFileLabels = mCharacter->getObjFileLabels();
    const auto& bvhMap = mCharacter->getBVHMap();
    const auto& endEffectors = mCharacter->getEndEffectors();

    // Write XML header
    ofs << "<!-- Exported skeleton configuration -->" << std::endl;
    ofs << std::endl;
    ofs << "<Skeleton name=\"" << skel->getName() << "\">" << std::endl;

    // Iterate through all body nodes and write Node elements
    auto bodyNodes = skel->getBodyNodes();
    for (auto bn : bodyNodes) {
        std::string nodeName = bn->getName();
        auto parent = bn->getParentBodyNode();
        std::string parentName = parent ? parent->getName() : "None";

        ofs << "    <Node name=\"" << nodeName << "\" parent=\"" << parentName << "\"";

        // PRESERVE: Endeffector flag from original XML
        bool isEndEffector = std::find(endEffectors.begin(), endEffectors.end(), bn) != endEffectors.end();
        if (isEndEffector) {
            ofs << " endeffector=\"True\"";
        }

        ofs << " >" << std::endl;

        // Body element
        if (bn->getNumShapeNodes() > 0) {
            auto shapeNode = bn->getShapeNode(0);
            auto shape = shapeNode->getShape();
            auto [shapeType, shapeSize] = getShapeInfo(shape);

            double mass = bn->getMass();

            // Get visual aspect for color
            dart::dynamics::VisualAspect* visualAspect = shapeNode->getVisualAspect();
            Eigen::Vector4d color = visualAspect ? visualAspect->getRGBA() : Eigen::Vector4d(0.6, 0.6, 1.5, 1.0);

            // Get transform in zero pose
            Eigen::Isometry3d bodyTransform = bn->getWorldTransform();

            // Try to get mesh filename from properties (if stored during load)
            std::string meshFile = "";
            // Note: DART may not store obj filename, we'll leave empty if not available

            ofs << "        <Body type=\"" << shapeType << "\" mass=\"" << mass << "\" size=\""
                << formatVector3d(shapeSize) << "\" contact=\"";

            // PRESERVE: Contact label from original XML (not DART's default)
            std::string contact_label = "On";  // default
            if (contactFlags.count(nodeName)) {
                contact_label = contactFlags.at(nodeName);
            }
            ofs << contact_label << "\" color=\""
                << color[0] << " " << color[1] << " " << color[2] << " " << color[3] << "\"";

            // PRESERVE: obj filename from original XML
            if (objFileLabels.count(nodeName)) {
                ofs << " obj=\"" << objFileLabels.at(nodeName) << "\"";
            } else if (auto meshShape = std::dynamic_pointer_cast<dart::dynamics::MeshShape>(shape)) {
                // Fallback: try to get mesh URI from DART
                std::string meshPath = meshShape->getMeshUri();
                if (!meshPath.empty()) {
                    size_t lastSlash = meshPath.find_last_of("/\\");
                    std::string meshFilename = (lastSlash != std::string::npos) ?
                        meshPath.substr(lastSlash + 1) : meshPath;
                    ofs << " obj=\"" << meshFilename << "\"";
                }
            }

            ofs << ">" << std::endl;

            // Body Transformation
            ofs << "            <Transformation linear=\""
                << formatRotationMatrix(bodyTransform.linear())
                << "\" translation=\""
                << formatVector3d(bodyTransform.translation()) << " \"/>" << std::endl;

            ofs << "        </Body>" << std::endl;
        }

        // Joint element
        auto joint = bn->getParentJoint();
        if (joint) {
            std::string jointType = getJointTypeString(joint);

            ofs << "        <Joint type=\"" << jointType << "\"";

            // PRESERVE: BVH mapping from original XML
            if (bvhMap.count(nodeName)) {
                const auto& bvhList = bvhMap.at(nodeName);
                ofs << " bvh=\"";
                for (size_t i = 0; i < bvhList.size(); ++i) {
                    ofs << bvhList[i];
                    if (i < bvhList.size() - 1) ofs << " ";
                }
                ofs << "\"";
            }

            // Add axis for Revolute/Prismatic joints
            if (auto revJoint = dynamic_cast<dart::dynamics::RevoluteJoint*>(joint)) {
                Eigen::Vector3d axis = revJoint->getAxis();
                ofs << " axis =\"" << formatVector3d(axis) << "\"";
            } else if (auto prisJoint = dynamic_cast<dart::dynamics::PrismaticJoint*>(joint)) {
                Eigen::Vector3d axis = prisJoint->getAxis();
                ofs << " axis =\"" << formatVector3d(axis) << "\"";
            }

            // Add joint limits for non-Free joints
            if (jointType != "Free" && joint->getNumDofs() > 0) {
                ofs << " lower=\"" << formatJointLimits(joint, true) << "\"";
                ofs << " upper=\"" << formatJointLimits(joint, false) << "\"";

                // Export kp/kv from DART (may differ from original XML if modified)
                ofs << " kp=\"" << formatJointParams(joint, "kp") << "\"";
                ofs << " kv=\"" << formatJointParams(joint, "kv") << "\"";
            }

            ofs << ">" << std::endl;

            // Joint Transformation
            // Get local transform from parent body frame
            Eigen::Isometry3d localTransform = joint->getTransformFromParentBodyNode();

            // Compute global transform by combining with parent body's global transform
            Eigen::Isometry3d globalTransform = localTransform;
            if (parent) {
                // Multiply parent body's global transform with joint's local transform
                Eigen::Isometry3d parentGlobalTransform = parent->getWorldTransform();
                globalTransform = parentGlobalTransform * localTransform;
            }

            ofs << "            <Transformation linear=\""
                << formatRotationMatrix(globalTransform.linear())
                << "\" translation=\""
                << formatVector3d(globalTransform.translation()) << "\"/>" << std::endl;

            ofs << "        </Joint>" << std::endl;
        }

        ofs << "    </Node>" << std::endl;
    }

    ofs << "</Skeleton>" << std::endl;
    ofs << std::endl;

    ofs.close();

    // Restore original skeleton state
    skel->setPositions(saved_positions);

    LOG_INFO("[Surgery] Successfully saved skeleton with " << bodyNodes.size() << " nodes to " << path);

    // Run automated validation
    if (!validateSkeletonExport(path)) {
        LOG_WARN("[Surgery] Skeleton export validation failed - please check the exported file");
    }
}

void SurgeryExecutor::exportSkeletonYAML(const std::string& path) {
    // Export skeleton in YAML format with local transforms:
    // - Body R/t: relative to parent joint frame
    // - Joint R/t: relative to parent body frame

    auto skel = mCharacter->getSkeleton();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Save current skeleton state
    Eigen::VectorXd saved_positions = skel->getPositions();

    // Move to zero pose (all joint angles = 0)
    Eigen::VectorXd zero_positions = Eigen::VectorXd::Zero(skel->getNumDofs());
    skel->setPositions(zero_positions);

    // Get metadata from Character instance (parsed during construction)
    const auto& contactFlags = mCharacter->getContactFlags();
    const auto& objFileLabels = mCharacter->getObjFileLabels();
    const auto& bvhMap = mCharacter->getBVHMap();
    const auto& endEffectors = mCharacter->getEndEffectors();

    // Write metadata section
    auto [git_hash, git_message] = getGitInfo();
    ofs << "metadata:" << std::endl;
    ofs << "  generator: \"" << mGeneratorContext << "\"" << std::endl;
    ofs << "  timestamp: \"" << getCurrentTimestamp() << "\"" << std::endl;
    ofs << "  version: v1" << std::endl;
    ofs << "  skeleton_from: \"" << getSkeletonName() << "\"" << std::endl;
    if (!git_hash.empty()) {
        ofs << "  git_commit: \"" << git_hash << "\"" << std::endl;
        if (!git_message.empty()) {
            ofs << "  git_message: \"" << git_message << "\"" << std::endl;
        }
    }
    ofs << std::endl;

    // Write skeleton section
    ofs << "skeleton:" << std::endl;
    ofs << "  name: \"" << skel->getName() << "\"" << std::endl;
    ofs << "  nodes:" << std::endl;

    // Iterate through all body nodes and write in flattened YAML format
    auto bodyNodes = skel->getBodyNodes();
    for (auto bn : bodyNodes) {
        std::string nodeName = bn->getName();
        auto parent = bn->getParentBodyNode();
        std::string parentName = parent ? parent->getName() : "None";

        // Start node entry with flow-style
        ofs << "    - {name: " << nodeName << ", parent: " << parentName;

        // PRESERVE: Endeffector flag from original XML
        bool isEndEffector = std::find(endEffectors.begin(), endEffectors.end(), bn) != endEffectors.end();
        if (isEndEffector) {
            ofs << ", ee: True";
        } else {
            ofs << ", ee: false";
        }

        // Body properties
        if (bn->getNumShapeNodes() > 0) {
            auto shapeNode = bn->getShapeNode(0);
            auto shape = shapeNode->getShape();
            auto [shapeType, shapeSize] = getShapeInfo(shape);
            double mass = bn->getMass();

            // Get visual aspect for color
            dart::dynamics::VisualAspect* visualAspect = shapeNode->getVisualAspect();
            Eigen::Vector4d color = visualAspect ? visualAspect->getRGBA() : Eigen::Vector4d(0.6, 0.6, 1.5, 1.0);

            // Get local transform relative to parent joint frame
            Eigen::Isometry3d bodyTransform = bn->getRelativeTransform();

            // PRESERVE: Contact label from original XML
            std::string contact_label = "On";
            if (contactFlags.count(nodeName)) {
                contact_label = contactFlags.at(nodeName);
            }
            bool contact_bool = (contact_label != "Off");

            ofs << ", " << std::endl << "       body: {type: " << shapeType << ", mass: "
                << std::fixed << std::setprecision(1) << mass
                << ", size: " << formatVectorYAML(shapeSize)
                << ", contact: " << (contact_bool ? "true" : "false");

            // PRESERVE: obj filename from original XML
            if (objFileLabels.count(nodeName)) {
                ofs << ", obj: \"" << objFileLabels.at(nodeName) << "\"";
            } else if (auto meshShape = std::dynamic_pointer_cast<dart::dynamics::MeshShape>(shape)) {
                std::string meshPath = meshShape->getMeshUri();
                if (!meshPath.empty()) {
                    size_t lastSlash = meshPath.find_last_of("/\\");
                    std::string meshFilename = (lastSlash != std::string::npos) ?
                        meshPath.substr(lastSlash + 1) : meshPath;
                    ofs << ", obj: \"" << meshFilename << "\"";
                }
            }

            ofs << "," << std::endl << "       R: " << formatMatrixYAML(bodyTransform.linear())
                << "," << std::endl << "       t: " << formatVectorYAML(bodyTransform.translation()) << "}";
        }

        // Joint properties
        auto joint = bn->getParentJoint();
        if (joint) {
            std::string jointType = getJointTypeString(joint);

            ofs << ", " << std::endl << std::endl << "       joint: {type: " << jointType;

            // PRESERVE: BVH mapping from original XML
            if (bvhMap.count(nodeName)) {
                const auto& bvhList = bvhMap.at(nodeName);
                ofs << ", bvh: ";
                for (size_t i = 0; i < bvhList.size(); ++i) {
                    ofs << bvhList[i];
                    if (i < bvhList.size() - 1) ofs << " ";
                }
            }

            // Add axis for Revolute/Prismatic joints
            if (auto revJoint = dynamic_cast<dart::dynamics::RevoluteJoint*>(joint)) {
                Eigen::Vector3d axis = revJoint->getAxis();
                ofs << ", axis: " << formatVectorYAML(axis);
            } else if (auto prisJoint = dynamic_cast<dart::dynamics::PrismaticJoint*>(joint)) {
                Eigen::Vector3d axis = prisJoint->getAxis();
                ofs << ", axis: " << formatVectorYAML(axis);
            }

            // Add joint limits for non-Free joints
            if (jointType != "Free" && joint->getNumDofs() > 0) {
                ofs << ", " << std::endl << "       lower: " << formatJointLimitsYAML(joint, true);
                ofs << "," << " upper: " << formatJointLimitsYAML(joint, false);

                // Export kp/kv from DART (may differ from original XML if modified)
                ofs << "," << std::endl << "       kp: " << formatJointParamsYAML(joint, "kp");
                ofs << "," << " kv: " << formatJointParamsYAML(joint, "kv");
            }

            // Joint Transformation (local: relative to parent body frame)
            Eigen::Isometry3d jointTransform = joint->getTransformFromParentBodyNode();
            ofs << "," << std::endl << "       R: " << formatMatrixYAML(jointTransform.linear())
                << "," << std::endl << "       t: " << formatVectorYAML(jointTransform.translation()) << "}";
        }

        ofs << "}" << std::endl << std::endl;
    }

    ofs.close();

    // Restore original skeleton state
    skel->setPositions(saved_positions);

    LOG_INFO("[Surgery] Successfully saved skeleton with " << bodyNodes.size() << " nodes to " << path);
}

bool SurgeryExecutor::validateSkeletonExport(const std::string& exported_path) {
    LOG_INFO("[Surgery] Validating skeleton export: " << exported_path);

    // Check if file exists and is readable
    std::ifstream ifs(exported_path);
    if (!ifs.is_open()) {
        LOG_ERROR("[Surgery] Validation failed: Cannot open exported file");
        return false;
    }

    // Read entire file content
    std::string line;
    std::string content;
    while (std::getline(ifs, line)) {
        content += line + "\n";
    }
    ifs.close();

    // Basic structural checks
    if (content.find("<Skeleton") == std::string::npos) {
        LOG_ERROR("[Surgery] Validation failed: Missing <Skeleton> tag");
        return false;
    }

    if (content.find("</Skeleton>") == std::string::npos) {
        LOG_ERROR("[Surgery] Validation failed: Missing </Skeleton> closing tag");
        return false;
    }

    // Count Node elements
    size_t nodeCount = 0;
    size_t pos = 0;
    while ((pos = content.find("<Node", pos)) != std::string::npos) {
        nodeCount++;
        pos++;
    }

    auto skel = mCharacter->getSkeleton();
    size_t expectedNodes = skel->getBodyNodes().size();

    if (nodeCount != expectedNodes) {
        LOG_WARN("[Surgery] Node count mismatch: expected " << expectedNodes << ", found " << nodeCount);
        return false;
    }

    LOG_INFO("[Surgery] Validation passed: " << nodeCount << " nodes found");
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint and Anchor Rotation Operations
// ═══════════════════════════════════════════════════════════════════════════

bool SurgeryExecutor::rotateJointOffset(const std::string& joint_name,
                                        const Eigen::Vector3d& axis,
                                        double angle,
                                        bool preserve_position) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded");
        return false;
    }

    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(joint_name);

    if (!joint) {
        LOG_ERROR("[Surgery] Error: Joint '" << joint_name << "' not found");
        return false;
    }

    LOG_INFO("[Surgery] Rotating joint offset: " << joint_name);
    LOG_INFO("[Surgery]   Axis: [" << axis[0] << ", " << axis[1] << ", " << axis[2] << "]");
    LOG_INFO("[Surgery]   Angle: " << angle << " rad (" << (angle * 180.0 / M_PI) << " deg)");
    LOG_INFO("[Surgery]   Preserve position: " << (preserve_position ? "true" : "false"));

    // Get current transform from parent body frame to this joint frame
    Eigen::Isometry3d T_parent_to_joint = joint->getTransformFromParentBodyNode();

    // Create rotation transform
    Eigen::AngleAxisd rotation(angle, axis);
    Eigen::Isometry3d R = Eigen::Isometry3d::Identity();
    R.linear() = rotation.toRotationMatrix();

    Eigen::Isometry3d T_new;
    if (preserve_position) {
        // Preserve position: only rotate the orientation component
        T_new = T_parent_to_joint;
        T_new.linear() = R.linear() * T_parent_to_joint.linear();
    } else {
        // Original behavior: Apply rotation to both orientation and position
        T_new = R * T_parent_to_joint;
    }

    // Set new transform
    joint->setTransformFromParentBodyNode(T_new);

    // Update all muscles to reflect new skeleton geometry
    auto muscles = mCharacter->getMuscles();
    for (auto muscle : muscles) {
        muscle->SetMuscle();
    }

    LOG_INFO("[Surgery] Updated " << muscles.size() << " muscles after joint rotation");
    LOG_INFO("[Surgery] Joint offset rotation complete");

    return true;
}

std::vector<AnchorReference> SurgeryExecutor::computeAffectedAnchors(
    const AnchorReference& ref_anchor,
    const Eigen::Vector3d& search_direction) const {

    std::vector<AnchorReference> affected;

    if (!mCharacter) {
        return affected;
    }

    auto muscles = mCharacter->getMuscles();

    // Find reference muscle using fast cache lookup
    Muscle* ref_muscle = mCharacter->getMuscleByName(ref_anchor.muscle_name);
    if (!ref_muscle) {
        LOG_ERROR("[Surgery] Reference muscle not found: " << ref_anchor.muscle_name);
        return affected;
    }

    auto ref_anchors = ref_muscle->GetAnchors();
    if (ref_anchor.anchor_index < 0 || ref_anchor.anchor_index >= ref_anchors.size()) {
        LOG_ERROR("[Surgery] Invalid reference anchor index: " << ref_anchor.anchor_index);
        return affected;
    }

    auto ref_anchor_ptr = ref_anchors[ref_anchor.anchor_index];

    // Validate: reference anchor must have exactly one bodynode
    if (ref_anchor_ptr->bodynodes.empty()) {
        LOG_ERROR("[Surgery] Reference anchor has no bodynodes");
        return affected;
    }

    if (ref_anchor_ptr->bodynodes.size() != 1) {
        throw std::runtime_error("[Surgery] Reference anchor must have exactly 1 bodynode (single-LBS only). Found: " +
                               std::to_string(ref_anchor_ptr->bodynodes.size()));
    }

    auto ref_bodynode = ref_anchor_ptr->bodynodes[0];
    Eigen::Vector3d ref_local_pos = ref_anchor_ptr->local_positions[0];

    // Normalize search direction
    Eigen::Vector3d search_dir = search_direction.normalized();

    // Search through ALL muscles for affected anchors
    for (auto muscle : muscles) {
        auto anchors = muscle->GetAnchors();

        for (int i = 0; i < anchors.size(); ++i) {
            auto anchor = anchors[i];

            // Skip reference anchor itself
            if (muscle->name == ref_anchor.muscle_name && i == ref_anchor.anchor_index) {
                continue;
            }

            // Find bodynode index for this anchor on the reference bodynode
            int bodynode_index = -1;
            for (int j = 0; j < anchor->bodynodes.size(); ++j) {
                if (anchor->bodynodes[j] == ref_bodynode) {
                    bodynode_index = j;
                    break;
                }
            }

            if (bodynode_index >= 0) {
                // Get local position relative to shared bodynode
                Eigen::Vector3d other_local_pos = anchor->local_positions[bodynode_index];

                // Compute difference in LOCAL space
                Eigen::Vector3d local_diff = other_local_pos - ref_local_pos;

                // Check if anchor is in search direction (LOCAL space filtering)
                double dot_product = local_diff.dot(search_dir);
                if (dot_product > 0.0) {
                    // This anchor will be affected
                    affected.emplace_back(muscle->name, i, bodynode_index);
                }
            }
        }
    }
    return affected;
}

bool SurgeryExecutor::rotateAnchorPoints(const std::string& muscle_name,
                                         int ref_anchor_index,
                                         const Eigen::Vector3d& search_direction,
                                         const Eigen::Vector3d& rotation_axis,
                                         double angle) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded");
        return false;
    }

    LOG_INFO("[Surgery] Rotating anchor points on muscle '" << muscle_name << "'");
    LOG_INFO("[Surgery]   Reference anchor index: " << ref_anchor_index);
    LOG_INFO("[Surgery]   Search direction: [" << search_direction[0] << ", " << search_direction[1] << ", " << search_direction[2] << "]");
    LOG_INFO("[Surgery]   Rotation axis: [" << rotation_axis[0] << ", " << rotation_axis[1] << ", " << rotation_axis[2] << "]");
    LOG_INFO("[Surgery]   Angle: " << angle << " rad (" << (angle * 180.0 / M_PI) << " deg)");

    // Create AnchorReference for the reference anchor
    // Note: We assume bodynode_index = 0 (single-LBS), validation happens in computeAffectedAnchors
    AnchorReference ref_anchor(muscle_name, ref_anchor_index, 0);

    // Compute affected anchors using shared method
    std::vector<AnchorReference> affected_anchors;
    try {
        affected_anchors = computeAffectedAnchors(ref_anchor, search_direction);
    } catch (const std::runtime_error& e) {
        LOG_ERROR("[Surgery] " << e.what());
        return false;
    }

    if (affected_anchors.empty()) {
        LOG_WARN("[Surgery] No affected anchors found");
        return true;  // Not an error, just nothing to do
    }

    // Get reference muscle and anchor for rotation center
    Muscle* ref_muscle = mCharacter->getMuscleByName(muscle_name);
    if (!ref_muscle) {
        LOG_ERROR("[Surgery] Reference muscle not found: " << muscle_name);
        return false;
    }

    auto ref_anchors = ref_muscle->GetAnchors();
    auto ref_anchor_ptr = ref_anchors[ref_anchor_index];
    Eigen::Vector3d ref_local_pos = ref_anchor_ptr->local_positions[0];

    // Create rotation matrix in bodynode's local frame
    Eigen::AngleAxisd rotation(angle, rotation_axis);
    Eigen::Matrix3d R_local = rotation.toRotationMatrix();

    // Apply rotation to all affected anchors
    std::set<Muscle*> modified_muscles;

    for (const auto& anchor_ref : affected_anchors) {
        Muscle* muscle = mCharacter->getMuscleByName(anchor_ref.muscle_name);
        if (!muscle) {
            LOG_WARN("[Surgery] Muscle not found: " << anchor_ref.muscle_name);
            continue;
        }

        auto anchor = muscle->GetAnchors()[anchor_ref.anchor_index];
        Eigen::Vector3d current_local_pos = anchor->local_positions[anchor_ref.bodynode_index];

        // Compute relative position in local frame
        Eigen::Vector3d relative_local = current_local_pos - ref_local_pos;

        // Apply rotation in local frame
        Eigen::Vector3d rotated_local = R_local * relative_local;

        // Update only the local position for the specific bodynode
        anchor->local_positions[anchor_ref.bodynode_index] = ref_local_pos + rotated_local;

        modified_muscles.insert(muscle);
    }

    // Update all modified muscles
    for (auto muscle : modified_muscles) {
        muscle->SetMuscle();
    }

    LOG_INFO("[Surgery] Anchor rotation complete.");
    LOG_INFO("[Surgery]   Rotated " << affected_anchors.size() << " anchors");
    LOG_INFO("[Surgery]   Updated " << modified_muscles.size() << " muscles");

    return !affected_anchors.empty();
}

// ═══════════════════════════════════════════════════════════════════════════
// FDO Combined Surgery (Joint + Anchor Rotation)
// ═══════════════════════════════════════════════════════════════════════════

bool SurgeryExecutor::validateAnchorReferencesBodynode(const std::string& muscle_name,
                                                        int anchor_index,
                                                        const std::string& bodynode_name) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded");
        return false;
    }

    // Find muscle
    Muscle* muscle = nullptr;
    auto muscles = mCharacter->getMuscles();
    for (auto m : muscles) {
        if (m->name == muscle_name) {
            muscle = m;
            break;
        }
    }

    if (!muscle) {
        std::cerr << "[Surgery] Error: Muscle '" << muscle_name << "' not found" << std::endl;
        return false;
    }

    // Validate anchor index
    auto anchors = muscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        std::cerr << "[Surgery] Error: Anchor index " << anchor_index << " out of range [0, "
                 << anchors.size() << ")" << std::endl;
        return false;
    }

    auto anchor = anchors[anchor_index];

    // Check if anchor has LBS attachment to target bodynode
    bool has_attachment = false;
    int attachment_idx = -1;
    for (int i = 0; i < anchor->bodynodes.size(); ++i) {
        if (anchor->bodynodes[i]->getName() == bodynode_name) {
            has_attachment = true;
            attachment_idx = i;
            break;
        }
    }

    if (!has_attachment) {
        // Print detailed warning with actual attachments
        std::cerr << "\n[Surgery] WARNING: Validation failed!" << std::endl;
        std::cerr << "[Surgery]   Muscle: " << muscle_name << std::endl;
        std::cerr << "[Surgery]   Anchor: #" << anchor_index << std::endl;
        std::cerr << "[Surgery]   Expected attachment to bodynode: " << bodynode_name << std::endl;
        std::cerr << "[Surgery]   Actual LBS attachments (" << anchor->bodynodes.size() << "):" << std::endl;
        for (int i = 0; i < anchor->bodynodes.size(); ++i) {
            double weight = (i < anchor->weights.size()) ? anchor->weights[i] : 0.0;
            std::cerr << "[Surgery]     [" << i << "] " << anchor->bodynodes[i]->getName()
                     << " (weight: " << weight << ")" << std::endl;
        }
        std::cerr << "[Surgery]   Operation aborted.\n" << std::endl;
        return false;
    }

    std::cout << "[Surgery] Validation passed: Anchor references target bodynode '"
             << bodynode_name << "'" << std::endl;
    return true;
}

dart::dynamics::Joint* SurgeryExecutor::getChildJoint(dart::dynamics::BodyNode* bodynode) {
    if (!bodynode) {
        return nullptr;
    }

    // Get number of child bodynodes
    size_t num_children = bodynode->getNumChildBodyNodes();

    if (num_children == 0) {
        std::cout << "[Surgery] Info: Bodynode '" << bodynode->getName()
                 << "' has no child joints" << std::endl;
        return nullptr;
    }

    if (num_children > 1) {
        std::cout << "[Surgery] Warning: Bodynode '" << bodynode->getName()
                 << "' has multiple child joints (" << num_children
                 << "), using first child" << std::endl;
    }

    // Get first child bodynode
    auto child_bn = bodynode->getChildBodyNode(0);
    if (!child_bn) {
        return nullptr;
    }

    // Get the joint connecting parent to child
    auto child_joint = child_bn->getParentJoint();

    std::cout << "[Surgery] Found child joint: '" << child_joint->getName()
             << "' (type: " << child_joint->getType() << ")" << std::endl;

    return child_joint;
}

bool SurgeryExecutor::executeFDO(const std::string& ref_muscle,
                                 int ref_anchor_index,
                                 const Eigen::Vector3d& search_direction,
                                 const Eigen::Vector3d& rotation_axis,
                                 double angle) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded");
        return false;
    }

    // Get reference muscle and anchor
    Muscle* muscle = mCharacter->getMuscleByName(ref_muscle);
    if (!muscle) {
        LOG_ERROR("[Surgery] Reference muscle not found: " << ref_muscle);
        return false;
    }

    auto anchors = muscle->GetAnchors();
    if (ref_anchor_index < 0 || ref_anchor_index >= anchors.size()) {
        LOG_ERROR("[Surgery] Invalid anchor index: " << ref_anchor_index);
        return false;
    }

    auto ref_anchor = anchors[ref_anchor_index];

    // Validate: reference anchor must have exactly one bodynode (single-LBS)
    if (ref_anchor->bodynodes.empty()) {
        LOG_ERROR("[Surgery] Reference anchor has no bodynodes");
        return false;
    }

    if (ref_anchor->bodynodes.size() != 1) {
        LOG_ERROR("[Surgery] FDO requires single-LBS anchor (found " << ref_anchor->bodynodes.size() << " bodynodes)");
        return false;
    }

    // Get target bodynode from reference anchor
    auto target_bn = ref_anchor->bodynodes[0];
    std::string target_bodynode = target_bn->getName();

    std::cout << "\n[Surgery] ═══════════════════════════════════════════════════" << std::endl;
    std::cout << "[Surgery] Executing FDO Combined Surgery" << std::endl;
    std::cout << "[Surgery] ═══════════════════════════════════════════════════" << std::endl;
    std::cout << "[Surgery]   Reference muscle: " << ref_muscle << std::endl;
    std::cout << "[Surgery]   Reference anchor: #" << ref_anchor_index << std::endl;
    std::cout << "[Surgery]   Target bodynode: " << target_bodynode << " (from anchor)" << std::endl;
    std::cout << "[Surgery]   Rotation angle: " << (angle * 180.0 / M_PI) << " degrees" << std::endl;

    // Step 1: Get child joint of target bodynode
    std::cout << "\n[Surgery] Step 1: Finding child joint of target bodynode..." << std::endl;
    auto child_joint = getChildJoint(target_bn);
    if (!child_joint) {
        std::cerr << "[Surgery] Error: No child joint found for bodynode '" << target_bodynode << "'" << std::endl;
        return false;
    }

    // Step 2: Rotate child joint
    std::cout << "\n[Surgery] Step 2: Rotating child joint..." << std::endl;
    if (!rotateJointOffset(child_joint->getName(), rotation_axis, angle)) {
        std::cerr << "[Surgery] Error: Failed to rotate child joint" << std::endl;
        return false;
    }

    // Step 3: Rotate anchors on target bodynode
    std::cout << "\n[Surgery] Step 3: Rotating anchors on target bodynode..." << std::endl;
    if (!rotateAnchorPoints(ref_muscle, ref_anchor_index, search_direction, rotation_axis, angle)) {
        std::cerr << "[Surgery] Error: Failed to rotate anchor points" << std::endl;
        return false;
    }

    std::cout << "\n[Surgery] ═══════════════════════════════════════════════════" << std::endl;
    std::cout << "[Surgery] FDO Combined Surgery Complete!" << std::endl;
    std::cout << "[Surgery] ═══════════════════════════════════════════════════\n" << std::endl;

    return true;
}

bool SurgeryExecutor::weakenMuscles(const std::vector<std::string>& muscles, double strength_ratio) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }

    if (strength_ratio <= 0.0) {
        LOG_ERROR("[Surgery] Error: Strength ratio must be positive, got " << strength_ratio);
        return false;
    }

    auto all_muscles = mCharacter->getMuscles();
    int modifiedCount = 0;

    for (auto m : all_muscles) {
        if (std::find(muscles.begin(), muscles.end(), m->name) != muscles.end()) {
            m->change_f(strength_ratio);
            modifiedCount++;
        }
    }

    if (modifiedCount == 0) {
        LOG_ERROR("[Surgery] Error: No muscles matched the provided names");
        return false;
    }

    std::string action = (strength_ratio < 1.0) ? "Weakened" : (strength_ratio > 1.0) ? "Strengthened" : "Scaled";
    LOG_INFO("[Surgery] " << action << " " << modifiedCount << " muscle(s) to "
             << (strength_ratio * 100) << "% strength");

    return true;
}

bool SurgeryExecutor::mirrorAnchorPositions(const std::vector<std::string>& muscleBaseNames) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] No character loaded!");
        return false;
    }

    auto skel = mCharacter->getSkeleton();

    // Get pelvis X as symmetry plane
    auto* pelvis = skel->getBodyNode("Pelvis");
    if (!pelvis) {
        LOG_ERROR("[Surgery] Pelvis body node not found!");
        return false;
    }
    double symmetryX = pelvis->getTransform().translation().x();

    auto& muscles = mCharacter->getMuscles();

    // Build L/R muscle map
    std::map<std::string, Muscle*> leftMuscles, rightMuscles;
    for (auto* m : muscles) {
        if (m->name.size() > 2 && m->name[1] == '_') {
            std::string base = m->name.substr(2);
            if (m->name[0] == 'L') leftMuscles[base] = m;
            else if (m->name[0] == 'R') rightMuscles[base] = m;
        }
    }

    int mirrored = 0;
    for (auto& [base, leftM] : leftMuscles) {
        auto it = rightMuscles.find(base);
        if (it == rightMuscles.end()) continue;

        // Filter by muscleBaseNames if provided
        if (!muscleBaseNames.empty()) {
            if (std::find(muscleBaseNames.begin(), muscleBaseNames.end(), base) == muscleBaseNames.end())
                continue;
        }

        Muscle* rightM = it->second;
        auto& leftAnchors = leftM->GetAnchors();
        auto& rightAnchors = rightM->GetAnchors();

        if (leftAnchors.size() != rightAnchors.size()) {
            LOG_WARN("[Surgery] Anchor count mismatch for " << base << ": L=" << leftAnchors.size() << ", R=" << rightAnchors.size());
            continue;
        }

        for (size_t i = 0; i < leftAnchors.size(); i++) {
            // Get global positions
            Eigen::Vector3d leftGlobal = leftAnchors[i]->GetPoint();
            Eigen::Vector3d rightGlobal = rightAnchors[i]->GetPoint();

            // Mirror right global across symmetry plane (pelvis X)
            Eigen::Vector3d rightMirrored = rightGlobal;
            rightMirrored.x() = 2.0 * symmetryX - rightGlobal.x();

            // Average the global positions
            Eigen::Vector3d avgGlobal = (leftGlobal + rightMirrored) / 2.0;

            // New left global = average
            Eigen::Vector3d newLeftGlobal = avgGlobal;

            // New right global = mirror of average across symmetry plane
            Eigen::Vector3d newRightGlobal = avgGlobal;
            newRightGlobal.x() = 2.0 * symmetryX - avgGlobal.x();

            // Convert back to local positions for each body node
            auto& lpos = leftAnchors[i]->local_positions;
            auto& rpos = rightAnchors[i]->local_positions;
            auto& lbns = leftAnchors[i]->bodynodes;
            auto& rbns = rightAnchors[i]->bodynodes;

            for (size_t j = 0; j < lpos.size(); j++) {
                lpos[j] = lbns[j]->getTransform().inverse() * newLeftGlobal;
            }
            for (size_t j = 0; j < rpos.size(); j++) {
                rpos[j] = rbns[j]->getTransform().inverse() * newRightGlobal;
            }
        }

        leftM->SetMuscle();
        rightM->SetMuscle();
        mirrored++;
        LOG_INFO("[Surgery] Mirrored anchor positions for " << base);
    }

    LOG_INFO("[Surgery] Mirrored " << mirrored << " muscle pair(s) (symmetry X=" << symmetryX << ")");
    return mirrored > 0;
}

bool SurgeryExecutor::optimizeWaypoints(const std::vector<std::string>& muscle_names,
                                        const std::string& /* hdf_motion_path - deprecated */,
                                        const WaypointOptimizer::Config& config,
                                        Character* reference_character,
                                        WaypointProgressCallback progressCallback) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return false;
    }

    if (!reference_character) {
        LOG_ERROR("[Surgery] Error: No reference character provided!");
        return false;
    }

    // Get skeletons
    auto subject_skeleton = mCharacter->getSkeleton();
    auto reference_skeleton = reference_character->getSkeleton();
    if (!subject_skeleton || !reference_skeleton) {
        LOG_ERROR("[Surgery] Error: No skeleton available!");
        return false;
    }

    // Save original poses (restore after all muscles processed)
    Eigen::VectorXd subject_original_pose = subject_skeleton->getPositions();
    Eigen::VectorXd reference_original_pose = reference_skeleton->getPositions();

    // Create optimizer
    WaypointOptimizer optimizer;

    // Optimize each muscle using reference character's muscles
    int optimizedCount = 0;
    int muscleIndex = 0;
    int totalMuscles = static_cast<int>(muscle_names.size());

    for (const auto& muscle_name : muscle_names) {
        // Call progress callback before processing this muscle
        if (progressCallback) {
            progressCallback(muscleIndex, totalMuscles, muscle_name);
        }

        // Find subject muscle
        Muscle* subject_muscle = mCharacter->getMuscleByName(muscle_name);
        if (!subject_muscle) {
            LOG_WARN("[Surgery] Warning: Subject muscle '" << muscle_name << "' not found, skipping");
            muscleIndex++;
            continue;
        }

        // Find reference muscle from reference character
        Muscle* reference_muscle = reference_character->getMuscleByName(muscle_name);
        if (!reference_muscle) {
            LOG_WARN("[Surgery] Warning: Reference muscle '" << muscle_name << "' not found, skipping");
            muscleIndex++;
            continue;
        }

        // Optimize by sweeping most relevant DOF
        WaypointOptResult result = optimizer.optimizeMuscle(
            subject_muscle, reference_muscle,
            reference_skeleton, subject_skeleton, config);

        if (result.success) {
            optimizedCount++;
        } else {
            LOG_WARN("[Surgery] Optimization failed for muscle: " << muscle_name);
        }

        muscleIndex++;
    }

    // Restore original poses
    subject_skeleton->setPositions(subject_original_pose);
    reference_skeleton->setPositions(reference_original_pose);

    // Final progress callback
    if (progressCallback) {
        progressCallback(totalMuscles, totalMuscles, "Complete");
    }

    if (optimizedCount == 0) {
        LOG_ERROR("[Surgery] Error: No muscles were successfully optimized");
        return false;
    }

    LOG_INFO("[Surgery] Waypoint optimization completed for " << optimizedCount
             << " out of " << muscle_names.size() << " muscle(s)");

    return true;
}


std::vector<WaypointOptResult> SurgeryExecutor::optimizeWaypointsWithResults(
                                        const std::vector<std::string>& muscle_names,
                                        const std::string& hdf_motion_path,
                                        const WaypointOptimizer::Config& config,
                                        Character* reference_character,
                                        std::mutex* characterMutex,
                                        WaypointResultCallback resultCallback) {
    std::vector<WaypointOptResult> results;

    if (!mCharacter) {
        LOG_ERROR("[Surgery] Error: No character loaded!");
        return results;
    }

    if (!reference_character) {
        LOG_ERROR("[Surgery] Error: No reference character provided!");
        return results;
    }

    // Get skeletons
    auto subject_skeleton = mCharacter->getSkeleton();
    auto reference_skeleton = reference_character->getSkeleton();
    if (!subject_skeleton || !reference_skeleton) {
        LOG_ERROR("[Surgery] Error: No skeleton available!");
        return results;
    }

    // Save original poses (restore after all muscles processed)
    Eigen::VectorXd subject_original_pose = subject_skeleton->getPositions();
    Eigen::VectorXd reference_original_pose = reference_skeleton->getPositions();

    // Load HDF and set both characters to first frame pose
    if (!hdf_motion_path.empty()) {
        try {
            HDF hdf(hdf_motion_path);
            Eigen::VectorXd first_frame_pose = hdf.getPose(0);

            // Set both skeletons to first frame pose
            if (first_frame_pose.size() == subject_skeleton->getNumDofs()) {
                subject_skeleton->setPositions(first_frame_pose);
                LOG_VERBOSE("[Surgery] Subject skeleton set to HDF first frame pose");
            } else {
                LOG_WARN("[Surgery] HDF DOF mismatch for subject: " << first_frame_pose.size()
                         << " vs " << subject_skeleton->getNumDofs());
            }

            if (first_frame_pose.size() == reference_skeleton->getNumDofs()) {
                reference_skeleton->setPositions(first_frame_pose);
                LOG_VERBOSE("[Surgery] Reference skeleton set to HDF first frame pose");
            } else {
                LOG_WARN("[Surgery] HDF DOF mismatch for reference: " << first_frame_pose.size()
                         << " vs " << reference_skeleton->getNumDofs());
            }
        } catch (const std::exception& e) {
            LOG_WARN("[Surgery] Failed to load HDF motion: " << e.what());
            LOG_WARN("[Surgery] Continuing with current skeleton poses");
        }
    }

    int totalMuscles = static_cast<int>(muscle_names.size());

    // Pre-sized results vector
    results.resize(totalMuscles);
    std::atomic<int> next_work{0};
    std::atomic<int> completed_count{0};

    // Mutex to protect skeleton/muscle cloning (DART's cloneSkeleton is NOT thread-safe)
    std::mutex cloneMutex;

    // Worker function - clones once per thread, processes multiple muscles
    auto worker_func = [&]() {
        dart::dynamics::SkeletonPtr worker_subject_skel;
        dart::dynamics::SkeletonPtr worker_ref_skel;
        Eigen::VectorXd subject_pose, ref_pose;

        // Clone skeletons ONCE per worker (protected by mutex - DART not thread-safe)
        {
            std::lock_guard<std::mutex> lock(cloneMutex);
            worker_subject_skel = subject_skeleton->cloneSkeleton();
            worker_ref_skel = reference_skeleton->cloneSkeleton();
            subject_pose = subject_skeleton->getPositions();
            ref_pose = reference_skeleton->getPositions();
        }
        worker_subject_skel->setPositions(subject_pose);
        worker_ref_skel->setPositions(ref_pose);

        // Clone ALL muscles ONCE per worker (map by name)
        std::unordered_map<std::string, Muscle*> worker_subject_muscles;
        std::unordered_map<std::string, Muscle*> worker_ref_muscles;

        // Muscle cloning also protected (reads from shared Character's muscles/bodynodes)
        {
            std::lock_guard<std::mutex> lock(cloneMutex);
            for (const auto& name : muscle_names) {
                Muscle* orig_subj = mCharacter->getMuscleByName(name);
                Muscle* orig_ref = reference_character->getMuscleByName(name);
                if (orig_subj && orig_ref) {
                    worker_subject_muscles[name] = orig_subj->clone(worker_subject_skel);
                    worker_ref_muscles[name] = orig_ref->clone(worker_ref_skel);
                }
            }
        }

        // Process work items
        WaypointOptimizer optimizer;
        while (true) {
            int work_idx = next_work.fetch_add(1);
            if (work_idx >= totalMuscles) break;

            const std::string& muscle_name = muscle_names[work_idx];
            Muscle* subj_muscle = worker_subject_muscles[muscle_name];
            Muscle* ref_muscle = worker_ref_muscles[muscle_name];

            if (!subj_muscle || !ref_muscle) {
                results[work_idx].muscle_name = muscle_name;
                results[work_idx].success = false;
                LOG_WARN("[Surgery] Warning: Muscle '" << muscle_name << "' not found, skipping");
            } else {
                results[work_idx] = optimizer.optimizeMuscle(
                    subj_muscle, ref_muscle,
                    worker_ref_skel, worker_subject_skel, config);

                // Store optimized positions in result (NO sync during work)
                if (results[work_idx].success) {
                    auto& anchors = subj_muscle->GetAnchors();
                    results[work_idx].optimized_anchor_positions.resize(anchors.size());
                    for (size_t i = 0; i < anchors.size(); ++i) {
                        results[work_idx].optimized_anchor_positions[i] =
                            anchors[i]->local_positions;
                    }
                }
            }

            int completed = ++completed_count;
            if (resultCallback) {
                resultCallback(completed, totalMuscles, muscle_name, results[work_idx]);
            }

            if (results[work_idx].success) {
                LOG_VERBOSE("[Surgery] Successfully optimized muscle: " << muscle_name);
            } else {
                LOG_WARN("[Surgery] Optimization failed for muscle: " << muscle_name);
            }
        }

        // Cleanup worker's cloned muscles
        for (auto& [name, muscle] : worker_subject_muscles) delete muscle;
        for (auto& [name, muscle] : worker_ref_muscles) delete muscle;
    };

    // Execute with numParallel threads
    if (config.numParallel > 1 && totalMuscles > 1) {
        std::vector<std::thread> threads;
        int num_threads = std::min(config.numParallel, totalMuscles);
        LOG_VERBOSE("[Surgery] Running waypoint optimization with " << num_threads << " threads");
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker_func);
        }
        for (auto& t : threads) {
            t.join();
        }
    } else {
        // Sequential execution (same worker logic, just single-threaded)
        worker_func();
    }

    // Batch sync optimized positions back to original muscles (AFTER all workers done)
    if (characterMutex) {
        std::lock_guard<std::mutex> lock(*characterMutex);
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].success && !results[i].optimized_anchor_positions.empty()) {
                Muscle* orig = mCharacter->getMuscleByName(results[i].muscle_name);
                if (orig) {
                    auto& anchors = orig->GetAnchors();
                    for (size_t a = 0; a < anchors.size(); ++a) {
                        if (a < results[i].optimized_anchor_positions.size()) {
                            anchors[a]->local_positions = results[i].optimized_anchor_positions[a];
                        }
                    }
                    orig->UpdateGeometry();
                }
            }
        }
    } else {
        // No mutex provided, still do the sync
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].success && !results[i].optimized_anchor_positions.empty()) {
                Muscle* orig = mCharacter->getMuscleByName(results[i].muscle_name);
                if (orig) {
                    auto& anchors = orig->GetAnchors();
                    for (size_t a = 0; a < anchors.size(); ++a) {
                        if (a < results[i].optimized_anchor_positions.size()) {
                            anchors[a]->local_positions = results[i].optimized_anchor_positions[a];
                        }
                    }
                    orig->UpdateGeometry();
                }
            }
        }
    }

    // Restore original poses
    subject_skeleton->setPositions(subject_original_pose);
    reference_skeleton->setPositions(reference_original_pose);

    LOG_INFO("[Surgery] Waypoint optimization with results completed for " << results.size()
             << " muscle(s)");

    return results;
}

// ============================================================================
// Metadata Helper Methods
// ============================================================================

std::pair<std::string, std::string> SurgeryExecutor::getGitInfo() const {
    std::string hash, message;

    // Get commit hash: git rev-parse HEAD
    FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            hash = std::string(buffer);
            // Trim whitespace
            size_t end = hash.find_last_not_of(" \n\r\t");
            if (end != std::string::npos) {
                hash = hash.substr(0, end + 1);
            }
        }
        pclose(pipe);
    }

    // Get commit message: git log -1 --pretty=%B
    pipe = popen("git log -1 --pretty=%B 2>/dev/null", "r");
    if (pipe) {
        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            message = std::string(buffer);
            // Trim whitespace
            size_t end = message.find_last_not_of(" \n\r\t");
            if (end != std::string::npos) {
                message = message.substr(0, end + 1);
            }
        }
        pclose(pipe);
    }

    return {hash, message};
}

std::string SurgeryExecutor::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string SurgeryExecutor::getSkeletonName() const {
    if (mSubjectSkeletonPath.empty()) {
        return "unknown";
    }

    // Extract filename from path (handle both / and \ separators)
    size_t lastSlash = mSubjectSkeletonPath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return mSubjectSkeletonPath.substr(lastSlash + 1);
    }

    return mSubjectSkeletonPath;
}

std::string SurgeryExecutor::getMuscleName() const {
    if (mSubjectMusclePath.empty()) {
        return "unknown";
    }

    // Extract filename from path (handle both / and \ separators)
    size_t lastSlash = mSubjectMusclePath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return mSubjectMusclePath.substr(lastSlash + 1);
    }

    return mSubjectMusclePath;
}

std::string SurgeryExecutor::getSkeletonBaseName() const {
    std::string filename = getSkeletonName();

    // Remove extension
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        return filename.substr(0, lastDot);
    }

    return filename;
}

std::string SurgeryExecutor::getMuscleBaseName() const {
    std::string filename = getMuscleName();

    // Remove extension
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        return filename.substr(0, lastDot);
    }

    return filename;
}

} // namespace PMuscle

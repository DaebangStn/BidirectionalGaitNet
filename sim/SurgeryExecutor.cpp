#include "SurgeryExecutor.h"
#include "UriResolver.h"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace PMuscle {

SurgeryExecutor::SurgeryExecutor() 
    : mCharacter(nullptr) {
}

SurgeryExecutor::~SurgeryExecutor() {
    // Note: mCharacter is not deleted here to avoid linker issues
    // Derived classes should manage Character lifetime if they own it
}

void SurgeryExecutor::loadCharacter(const std::string& skel_path, const std::string& muscle_path,
                                   ActuatorType actuator_type) {
    // Resolve URIs
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();

    std::string resolved_skel = resolver.resolve(skel_path);
    std::string resolved_muscle = resolver.resolve(muscle_path);

    std::cout << "Loading skeleton: " << resolved_skel << std::endl;
    std::cout << "Loading muscle: " << resolved_muscle << std::endl;

    // Create character (without world/environment dependencies)
    mCharacter = new Character(resolved_skel, 300.0, 40.0, 5.0, true);
    mCharacter->setMuscles(resolved_muscle);
    mCharacter->setActuatorType(actuator_type);

    // Zero muscle activations
    if (mCharacter->getMuscles().size() > 0) {
        mCharacter->setActivations(mCharacter->getActivations().setZero());
    }

    std::cout << "Character loaded successfully" << std::endl;
}

void SurgeryExecutor::applyPosePreset(const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        std::cerr << "No character loaded" << std::endl;
        return;
    }

    auto skeleton = mCharacter->getSkeleton();

    for (const auto& [joint_name, angles] : joint_angles) {
        auto joint = skeleton->getJoint(joint_name);
        if (!joint) {
            std::cerr << "Joint not found: " << joint_name << std::endl;
            continue;
        }

        if (angles.size() != joint->getNumDofs()) {
            std::cerr << "Joint " << joint_name << " expects "
                     << joint->getNumDofs() << " DOFs, got "
                     << angles.size() << std::endl;
            continue;
        }

        joint->setPositions(angles);
    }

    std::cout << "Pose preset applied" << std::endl;
}

void SurgeryExecutor::resetMuscles() {
    if (!mCharacter) return;

    std::cout << "[Surgery] Resetting all muscles to original state..." << std::endl;

    auto muscles = mCharacter->getMuscles();
    int resetCount = 0;
    for (auto muscle : muscles) {
        muscle->change_f(1.0);
        muscle->change_l(1.0);
        muscle->SetTendonOffset(0.0);
        resetCount++;
    }

    std::cout << "[Surgery] Muscle reset complete. Reset " << resetCount << " muscles." << std::endl;
}

bool SurgeryExecutor::distributePassiveForce(const std::vector<std::string>& muscles, 
                                            const std::string& reference,
                                            const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
        return false;
    }
    
    // Apply joint angles if specified
    if (!joint_angles.empty()) {
        applyPosePreset(joint_angles);
        std::cout << "[Surgery] Applied " << joint_angles.size() << " joint angle(s)" << std::endl;
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
        std::cerr << "[Surgery] Error: Reference muscle '" << reference << "' not found!" << std::endl;
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
    
    std::cout << "[Surgery] Distributed passive force coefficient " << refCoeff
              << " from '" << reference << "' to " << modifiedCount << " muscles" << std::endl;
    
    return true;
}

bool SurgeryExecutor::relaxPassiveForce(const std::vector<std::string>& muscles,
                                       const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
        return false;
    }
    
    // Apply joint angles if specified
    if (!joint_angles.empty()) {
        applyPosePreset(joint_angles);
        std::cout << "[Surgery] Applied " << joint_angles.size() << " joint angle(s)" << std::endl;
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
    
    std::cout << "[Surgery] Applied relaxation to " << relaxedCount << " muscles" << std::endl;
    
    return true;
}

bool SurgeryExecutor::removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
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
        std::cerr << "[Surgery] Error: Muscle '" << muscleName << "' not found!" << std::endl;
        return false;
    }
    
    if (anchorIndex < 0 || anchorIndex >= targetMuscle->mAnchors.size()) {
        std::cerr << "[Surgery] Error: Invalid anchor index " << anchorIndex << std::endl;
        return false;
    }
    
    std::cout << "[Surgery] Removing anchor #" << anchorIndex 
              << " from muscle '" << muscleName << "'..." << std::endl;
    
    // Remove the anchor
    delete targetMuscle->mAnchors[anchorIndex];  // Free memory
    targetMuscle->mAnchors.erase(targetMuscle->mAnchors.begin() + anchorIndex);
    
    // Recalculate muscle parameters
    targetMuscle->SetMuscle();

    std::cout << "[Surgery] Anchor removal complete. Muscle now has "
              << targetMuscle->mAnchors.size() << " anchors." << std::endl;
    
    return true;
}

bool SurgeryExecutor::copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex, const std::string& toMuscle) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
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
        std::cerr << "[Surgery] Error: Source muscle '" << fromMuscle << "' not found!" << std::endl;
        return false;
    }
    
    if (!targetMuscle) {
        std::cerr << "[Surgery] Error: Target muscle '" << toMuscle << "' not found!" << std::endl;
        return false;
    }
    
    auto sourceAnchors = sourceMuscle->GetAnchors();
    if (fromIndex < 0 || fromIndex >= sourceAnchors.size()) {
        std::cerr << "[Surgery] Error: Invalid anchor index " << fromIndex << std::endl;
        return false;
    }
    
    auto sourceAnchor = sourceAnchors[fromIndex];
    
    std::cout << "[Surgery] Copying anchor #" << fromIndex 
              << " from '" << fromMuscle << "' to '" << toMuscle << "'..." << std::endl;
    
    // Create a deep copy of the anchor
    std::vector<dart::dynamics::BodyNode*> newBodyNodes = sourceAnchor->bodynodes;
    std::vector<Eigen::Vector3d> newLocalPositions = sourceAnchor->local_positions;
    std::vector<double> newWeights = sourceAnchor->weights;
    
    Anchor* newAnchor = new Anchor(newBodyNodes, newLocalPositions, newWeights);
    
    // Add the new anchor to the target muscle
    targetMuscle->mAnchors.push_back(newAnchor);
    
    // Recalculate muscle parameters
    targetMuscle->SetMuscle();

    std::cout << "[Surgery] Anchor copied successfully. Target muscle now has "
              << targetMuscle->mAnchors.size() << " anchors." << std::endl;
    
    // Display info about the copied anchor
    if (!newBodyNodes.empty()) {
        std::cout << "[Surgery]   Copied anchor attached to: " 
                  << newBodyNodes[0]->getName() << std::endl;
        if (newBodyNodes.size() > 1) {
            std::cout << "[Surgery]   (LBS with " << newBodyNodes.size() << " body nodes)" << std::endl;
        }
    }
    
    return true;
}

bool SurgeryExecutor::editAnchorPosition(const std::string& muscle, int anchor_index, 
                                        const Eigen::Vector3d& position) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
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
        std::cerr << "[Surgery] Error: Muscle '" << muscle << "' not found!" << std::endl;
        return false;
    }
    
    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        std::cerr << "[Surgery] Error: Invalid anchor index " << anchor_index << std::endl;
        return false;
    }
    
    auto anchor = anchors[anchor_index];
    
    // Update anchor position for ALL bodynodes in this anchor
    for (size_t i = 0; i < anchor->local_positions.size(); ++i) {
        anchor->local_positions[i] = position;
    }
    
    targetMuscle->SetMuscle();
    
    std::cout << "[Surgery] Updated position for anchor #" << anchor_index 
              << " in '" << muscle << "' to [" << position.transpose() << "]" << std::endl;
    
    return true;
}

bool SurgeryExecutor::editAnchorWeights(const std::string& muscle, int anchor_index,
                                       const std::vector<double>& weights) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
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
        std::cerr << "[Surgery] Error: Muscle '" << muscle << "' not found!" << std::endl;
        return false;
    }
    
    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        std::cerr << "[Surgery] Error: Invalid anchor index " << anchor_index << std::endl;
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
    
    std::cout << "[Surgery] Updated weights for anchor #" << anchor_index 
              << " in '" << muscle << "'" << std::endl;
    
    return true;
}

bool SurgeryExecutor::addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                                         const std::string& bodynode_name, double weight) {
    if (!mCharacter) {
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
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
        std::cerr << "[Surgery] Error: Muscle '" << muscle << "' not found!" << std::endl;
        return false;
    }
    
    auto skel = mCharacter->getSkeleton();
    auto newBodyNode = skel->getBodyNode(bodynode_name);
    if (!newBodyNode) {
        std::cerr << "[Surgery] Error: Body node '" << bodynode_name << "' not found!" << std::endl;
        return false;
    }
    
    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        std::cerr << "[Surgery] Error: Invalid anchor index " << anchor_index << std::endl;
        return false;
    }
    
    auto anchor = anchors[anchor_index];
    
    // Check if bodynode already exists in anchor
    for (auto bn : anchor->bodynodes) {
        if (bn == newBodyNode) {
            std::cerr << "[Surgery] Error: Body node already exists in this anchor!" << std::endl;
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
        std::cerr << "[Surgery] Error: No character loaded!" << std::endl;
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
        std::cerr << "[Surgery] Error: Muscle '" << muscle << "' not found!" << std::endl;
        return false;
    }
    
    auto anchors = targetMuscle->GetAnchors();
    if (anchor_index < 0 || anchor_index >= anchors.size()) {
        std::cerr << "[Surgery] Error: Invalid anchor index " << anchor_index << std::endl;
        return false;
    }
    
    auto anchor = anchors[anchor_index];
    
    if (anchor->bodynodes.size() <= 1) {
        std::cerr << "[Surgery] Error: Cannot remove last body node from anchor!" << std::endl;
        return false;
    }
    
    if (bodynode_index < 0 || bodynode_index >= anchor->bodynodes.size()) {
        std::cerr << "[Surgery] Error: Invalid body node index " << bodynode_index << std::endl;
        return false;
    }
    
    // Remove this bodynode from anchor
    std::string removed_name = anchor->bodynodes[bodynode_index]->getName();
    anchor->bodynodes.erase(anchor->bodynodes.begin() + bodynode_index);
    anchor->weights.erase(anchor->weights.begin() + bodynode_index);
    anchor->local_positions.erase(anchor->local_positions.begin() + bodynode_index);
    
    targetMuscle->SetMuscle();
    
    std::cout << "[Surgery] Removed body node '" << removed_name << "' from anchor #" 
              << anchor_index << " in '" << muscle << "'" << std::endl;
    
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

    std::ofstream mfs(path);
    if (!mfs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::cout << "[Surgery] Saving muscle configuration to: " << path << std::endl;

    // Save current skeleton state
    auto skel = mCharacter->getSkeleton();
    Eigen::VectorXd saved_positions = skel->getPositions();

    // Move to zero pose (all joint angles = 0)
    Eigen::VectorXd zero_positions = Eigen::VectorXd::Zero(skel->getNumDofs());
    skel->setPositions(zero_positions);

    mfs << "<Muscle>" << std::endl;

    for (auto m : muscles) {
        std::string name = m->name;
        double f0 = m->f0;
        double l_m0 = m->lm_opt;
        double l_t0 = m->lt_rel;
        double pen_angle = m->pen_angle;

        mfs << "    <Unit name=\"" << name
            << "\" f0=\"" << f0
            << "\" lm=\"" << l_m0
            << "\" lt=\"" << l_t0
            << "\" pen_angle=\"" << pen_angle
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

    std::cout << "[Surgery] Successfully saved " << muscles.size()
              << " muscles to " << path << std::endl;
}

} // namespace PMuscle


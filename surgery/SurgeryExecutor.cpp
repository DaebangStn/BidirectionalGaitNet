#include "SurgeryExecutor.h"
#include "UriResolver.h"
#include "Log.h"
#include "DARTHelper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace PMuscle {

SurgeryExecutor::SurgeryExecutor(const std::string& generator_context)
    : mCharacter(nullptr), mGeneratorContext(generator_context) {
}

SurgeryExecutor::~SurgeryExecutor() {
    // Note: mCharacter is not deleted here to avoid linker issues
    // Derived classes should manage Character lifetime if they own it
}

void SurgeryExecutor::loadCharacter(const std::string& skel_path, const std::string& muscle_path,
                                   ActuatorType actuator_type) {
    // Cache original paths for metadata preservation
    mOriginalSkeletonPath = skel_path;
    mOriginalMusclePath = muscle_path;

    // Resolve URIs
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();

    std::string resolved_skel = resolver.resolve(skel_path);
    std::string resolved_muscle = resolver.resolve(muscle_path);

    LOG_INFO("Loading skeleton: " << resolved_skel);
    LOG_INFO("Loading muscle: " << resolved_muscle);

    // Create character
    mCharacter = new Character(resolved_skel, 300.0, 40.0, 5.0, true);

    // Load muscles
    mCharacter->setMuscles(resolved_muscle);
    mCharacter->setActuatorType(actuator_type);

    // Zero muscle activations
    if (mCharacter->getMuscles().size() > 0) {
        mCharacter->setActivations(mCharacter->getActivations().setZero());
    }

    LOG_INFO("Character loaded successfully");
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
        URIResolver& resolver = URIResolver::getInstance();
        resolver.initialize();
        std::string resolved_path = resolver.resolve(muscle_xml_path);

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
    // Auto-detect format from file extension
    size_t len = path.length();
    bool is_yaml = (len >= 5 && path.substr(len - 5) == ".yaml") ||
                   (len >= 4 && path.substr(len - 4) == ".yml");

    if (is_yaml) {
        exportMusclesYAML(path);
    } else {
        exportMusclesXML(path);  // Default to XML for backward compatibility
    }
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

    LOG_INFO("[Surgery] Successfully saved " << muscles.size() << " muscles to " << path);
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
        double lm = m->lm_opt;
        double lt = m->lt_rel;

        // Start muscle entry with properties
        mfs << "  - {name: " << name
            << ", f0: " << std::fixed << std::setprecision(2) << f0
            << ", lm: " << std::fixed << std::setprecision(2) << lm
            << ", lt: " << std::fixed << std::setprecision(2) << lt << "," << std::endl;

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

                if (i > 0) mfs << ", ";

                // Format body entry in flow style with padded body name
                mfs << "{body: " << std::left << std::setw(max_body_len) << body_name << ", p: [";

                // Format coordinates with 5 decimals and sign alignment
                for (int j = 0; j < 3; ++j) {
                    if (j > 0) mfs << ", ";
                    // Add leading space for positive numbers for alignment
                    if (local_pos[j] >= 0.0) {
                        mfs << " " << std::fixed << std::setprecision(5) << local_pos[j];
                    } else {
                        mfs << std::fixed << std::setprecision(5) << local_pos[j];
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
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << " ";
        if (param == "kp") {
            oss << joint->getSpringStiffness(i);
        } else if (param == "kv") {
            oss << joint->getDampingCoefficient(i);
        }
    }
    return oss.str();
}

std::string SurgeryExecutor::formatJointParamsYAML(dart::dynamics::Joint* joint, const std::string& param) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";

    size_t numDofs = joint->getNumDofs();
    for (size_t i = 0; i < numDofs; ++i) {
        if (i > 0) oss << ", ";
        oss << std::setw(5);
        if (param == "kp") {
            oss << joint->getSpringStiffness(i);
        } else if (param == "kv") {
            oss << joint->getDampingCoefficient(i);
        }
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

// ═══════════════════════════════════════════════════════════════════════════
// Skeleton Metadata Preservation
// ═══════════════════════════════════════════════════════════════════════════

struct SkeletonMetadata {
    std::map<std::string, std::string> joint_bvh_mappings;     // joint_name → bvh attribute
    std::map<std::string, std::string> node_endeffector_flags; // node_name → "True"/"False"
    std::map<std::string, Eigen::VectorXd> joint_kp_original;  // joint_name → kp values
    std::map<std::string, Eigen::VectorXd> joint_kv_original;  // joint_name → kv values
    std::map<std::string, std::string> body_contact_labels;    // node_name → "On"/"Off"
    std::map<std::string, std::string> body_obj_files;         // node_name → "Pelvis.obj"
};

static SkeletonMetadata parseOriginalSkeletonMetadata(const std::string& xml_path) {
    SkeletonMetadata metadata;

    if (xml_path.empty()) {
        return metadata;  // Empty metadata if no path
    }

    // Resolve URI before loading
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();
    std::string resolved_path = resolver.resolve(xml_path);

    // Parse XML with TinyXML2
    TiXmlDocument doc;
    if (doc.LoadFile(resolved_path.c_str()) != tinyxml2::XML_SUCCESS) {
        LOG_WARN("[Surgery] Failed to load original skeleton XML for metadata: " << resolved_path);
        return metadata;
    }

    TiXmlElement* skeleton_elem = doc.FirstChildElement("Skeleton");
    if (!skeleton_elem) {
        LOG_WARN("[Surgery] No <Skeleton> element found in: " << resolved_path);
        return metadata;
    }

    // Iterate through all <Node> elements
    for (TiXmlElement* node = skeleton_elem->FirstChildElement("Node");
         node;
         node = node->NextSiblingElement("Node")) {

        const char* node_name = node->Attribute("name");
        if (!node_name) continue;

        // 1. Parse endeffector flag
        const char* endeffector = node->Attribute("endeffector");
        if (endeffector) {
            metadata.node_endeffector_flags[node_name] = endeffector;
        }

        // 2. Parse Body metadata
        TiXmlElement* body = node->FirstChildElement("Body");
        if (body) {
            const char* contact = body->Attribute("contact");
            if (contact) {
                metadata.body_contact_labels[node_name] = contact;
            }

            const char* obj = body->Attribute("obj");
            if (obj) {
                metadata.body_obj_files[node_name] = obj;
            }
        }

        // 3. Parse Joint metadata
        TiXmlElement* joint = node->FirstChildElement("Joint");
        if (joint) {
            const char* bvh = joint->Attribute("bvh");
            if (bvh) {
                metadata.joint_bvh_mappings[node_name] = bvh;
            }

            const char* kp_str = joint->Attribute("kp");
            if (kp_str) {
                std::istringstream iss(kp_str);
                std::vector<double> kp_vals;
                double val;
                while (iss >> val) {
                    kp_vals.push_back(val);
                }
                Eigen::VectorXd kp_vec(kp_vals.size());
                for (size_t i = 0; i < kp_vals.size(); ++i) {
                    kp_vec[i] = kp_vals[i];
                }
                metadata.joint_kp_original[node_name] = kp_vec;
            }

            const char* kv_str = joint->Attribute("kv");
            if (kv_str) {
                std::istringstream iss(kv_str);
                std::vector<double> kv_vals;
                double val;
                while (iss >> val) {
                    kv_vals.push_back(val);
                }
                Eigen::VectorXd kv_vec(kv_vals.size());
                for (size_t i = 0; i < kv_vals.size(); ++i) {
                    kv_vec[i] = kv_vals[i];
                }
                metadata.joint_kv_original[node_name] = kv_vec;
            }
        }
    }

    LOG_INFO("[Surgery] Loaded metadata from original skeleton XML: "
             << metadata.joint_bvh_mappings.size() << " BVH mappings, "
             << metadata.node_endeffector_flags.size() << " endeffector flags, "
             << metadata.joint_kp_original.size() << " kp/kv values");

    return metadata;
}

void SurgeryExecutor::exportSkeleton(const std::string& path) {
    if (!mCharacter) {
        throw std::runtime_error("No character loaded");
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        throw std::runtime_error("No skeleton found in character");
    }

    // Detect format from file extension
    std::string ext;
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = path.substr(dot_pos);
        // Convert to lowercase for comparison
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    LOG_INFO("[Surgery] Saving skeleton configuration to: " << path);

    if (ext == ".yaml" || ext == ".yml") {
        exportSkeletonYAML(path);
    } else {
        // Default to XML for .xml or unrecognized extensions
        exportSkeletonXML(path);
    }
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

    // Parse original XML for metadata preservation
    SkeletonMetadata metadata = parseOriginalSkeletonMetadata(mOriginalSkeletonPath);

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
        if (metadata.node_endeffector_flags.count(nodeName)) {
            ofs << " endeffector=\"" << metadata.node_endeffector_flags.at(nodeName) << "\"";
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
            if (metadata.body_contact_labels.count(nodeName)) {
                contact_label = metadata.body_contact_labels.at(nodeName);
            }
            ofs << contact_label << "\" color=\""
                << color[0] << " " << color[1] << " " << color[2] << " " << color[3] << "\"";

            // PRESERVE: obj filename from original XML
            if (metadata.body_obj_files.count(nodeName)) {
                ofs << " obj=\"" << metadata.body_obj_files.at(nodeName) << "\"";
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
            if (metadata.joint_bvh_mappings.count(nodeName)) {
                ofs << " bvh=\"" << metadata.joint_bvh_mappings.at(nodeName) << "\"";
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

                // PRESERVE: Original kp/kv from XML (not DART's 0.0/5.0 defaults)
                if (metadata.joint_kp_original.count(nodeName)) {
                    ofs << " kp=\"" << formatVectorXd(metadata.joint_kp_original.at(nodeName)) << "\"";
                } else {
                    ofs << " kp=\"" << formatJointParams(joint, "kp") << "\"";
                }

                if (metadata.joint_kv_original.count(nodeName)) {
                    ofs << " kv=\"" << formatVectorXd(metadata.joint_kv_original.at(nodeName)) << "\"";
                } else {
                    ofs << " kv=\"" << formatJointParams(joint, "kv") << "\"";
                }
            }

            ofs << ">" << std::endl;

            // Joint Transformation
            Eigen::Isometry3d jointTransform = joint->getTransformFromParentBodyNode();
            ofs << "            <Transformation linear=\""
                << formatRotationMatrix(jointTransform.linear())
                << "\" translation=\""
                << formatVector3d(jointTransform.translation()) << "\"/>" << std::endl;

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

    // Parse original XML for metadata preservation
    SkeletonMetadata metadata = parseOriginalSkeletonMetadata(mOriginalSkeletonPath);

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
        if (metadata.node_endeffector_flags.count(nodeName)) {
            ofs << ", ee: " << metadata.node_endeffector_flags.at(nodeName);
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
            if (metadata.body_contact_labels.count(nodeName)) {
                contact_label = metadata.body_contact_labels.at(nodeName);
            }
            bool contact_bool = (contact_label != "Off");

            ofs << ", " << std::endl << "       body: {type: " << shapeType << ", mass: "
                << std::fixed << std::setprecision(1) << mass
                << ", size: " << formatVectorYAML(shapeSize)
                << ", contact: " << (contact_bool ? "true" : "false");

            // PRESERVE: obj filename from original XML
            if (metadata.body_obj_files.count(nodeName)) {
                ofs << ", obj: \"" << metadata.body_obj_files.at(nodeName) << "\"";
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
            if (metadata.joint_bvh_mappings.count(nodeName)) {
                ofs << ", bvh: " << metadata.joint_bvh_mappings.at(nodeName);
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
                ofs << ", upper: " << formatJointLimitsYAML(joint, false);

                // PRESERVE: Original kp/kv from XML
                if (metadata.joint_kp_original.count(nodeName)) {
                    ofs << ", kp: " << formatVectorXdYAML(metadata.joint_kp_original.at(nodeName));
                } else {
                    ofs << ", kp: " << formatJointParamsYAML(joint, "kp");
                }

                if (metadata.joint_kv_original.count(nodeName)) {
                    ofs << ", kv: " << formatVectorXdYAML(metadata.joint_kv_original.at(nodeName));
                } else {
                    ofs << ", kv: " << formatJointParamsYAML(joint, "kv");
                }
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
    if (mOriginalSkeletonPath.empty()) {
        return "unknown";
    }

    // Extract filename from path (handle both / and \ separators)
    size_t lastSlash = mOriginalSkeletonPath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return mOriginalSkeletonPath.substr(lastSlash + 1);
    }

    return mOriginalSkeletonPath;
}

std::string SurgeryExecutor::getMuscleName() const {
    if (mOriginalMusclePath.empty()) {
        return "unknown";
    }

    // Extract filename from path (handle both / and \ separators)
    size_t lastSlash = mOriginalMusclePath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        return mOriginalMusclePath.substr(lastSlash + 1);
    }

    return mOriginalMusclePath;
}

} // namespace PMuscle


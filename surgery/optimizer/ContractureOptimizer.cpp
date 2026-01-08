// Contracture Optimizer - Implementation
#include "ContractureOptimizer.h"
#include "rm/rm.hpp"
#include "Log.h"
#include <yaml-cpp/yaml.h>
#include <ceres/ceres.h>
#include <dart/dynamics/BallJoint.hpp>
#include <regex>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <functional>

namespace PMuscle {

// Ceres cost functor for torque matching (uses numeric differentiation)
struct ContractureOptimizer::TorqueResidual : public ceres::CostFunction {
    TorqueResidual(
        Character* character,
        const PoseData& pose,
        const std::map<int, std::vector<int>>& muscle_groups,
        const std::map<int, double>& base_lm_contract,
        int num_groups
    ) : character_(character),
        pose_(pose),
        muscle_groups_(muscle_groups),
        base_lm_contract_(base_lm_contract) {
        // Set parameter block sizes
        mutable_parameter_block_sizes()->push_back(num_groups);
        set_num_residuals(1);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        // Extract ratios from parameters
        int num_groups = static_cast<int>(muscle_groups_.size());

        // Apply ratios to muscles (temporarily)
        auto muscles = character_->getMuscles();
        std::vector<double> old_lm_contract(muscles.size());

        // Save ALL muscle lm_contract values before modification
        // (critical: non-group muscles must be preserved during restore)
        for (size_t i = 0; i < muscles.size(); ++i) {
            old_lm_contract[i] = muscles[i]->lm_contract;
        }

        for (const auto& [group_id, muscle_ids] : muscle_groups_) {
            double ratio = parameters[0][group_id];
            for (int m_idx : muscle_ids) {

                auto it = base_lm_contract_.find(m_idx);
                double base = (it != base_lm_contract_.end()) ? it->second : muscles[m_idx]->lm_contract;

                muscles[m_idx]->lm_contract = base * ratio;
                muscles[m_idx]->RefreshMuscleParams();
            }
        }

        // Set pose
        character_->getSkeleton()->setPositions(pose_.q);

        // Update muscle geometry
        for (auto& muscle : muscles) {
            muscle->UpdateGeometry();
        }

        // Compute predicted passive torque
        // For composite axis (abd_knee): use Y-axis projection with dof_offset=1
        // For simple DOF: use specific DOF filtering with pose_.joint_dof
        int dof_offset = pose_.use_composite_axis ? 1 : pose_.joint_dof;
        bool use_global_y = pose_.use_composite_axis;
        double tau_pred = computePassiveTorque(character_, pose_.joint_idx, false, dof_offset, use_global_y);

        // Restore original lm_contract values
        for (size_t i = 0; i < muscles.size(); ++i) {
            muscles[i]->lm_contract = old_lm_contract[i];
            muscles[i]->RefreshMuscleParams();
        }

        // Compute residual
        residuals[0] = std::sqrt(pose_.weight) * (tau_pred - pose_.tau_obs);

        // Compute Jacobian numerically if requested
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            const double h = 1e-6;  // Finite difference step
            for (int g = 0; g < num_groups; ++g) {
                // Perturb parameter
                std::vector<double> params_plus(parameters[0], parameters[0] + num_groups);
                params_plus[g] += h;
                const double* params_plus_ptr = params_plus.data();

                // Apply perturbed ratios
                for (const auto& [group_id, muscle_ids] : muscle_groups_) {
                    double ratio = params_plus_ptr[group_id];
                    for (int m_idx : muscle_ids) {
                        auto it = base_lm_contract_.find(m_idx);
                        double base = (it != base_lm_contract_.end()) ? it->second : old_lm_contract[m_idx];
                        muscles[m_idx]->lm_contract = base * ratio;
                        muscles[m_idx]->RefreshMuscleParams();
                    }
                }

                character_->getSkeleton()->setPositions(pose_.q);
                for (auto& muscle : muscles) {
                    muscle->UpdateGeometry();
                }
                double tau_plus = computePassiveTorque(character_, pose_.joint_idx, false, dof_offset, use_global_y);

                // Restore
                for (size_t i = 0; i < muscles.size(); ++i) {
                    muscles[i]->lm_contract = old_lm_contract[i];
                    muscles[i]->RefreshMuscleParams();
                }

                double residual_plus = std::sqrt(pose_.weight) * (tau_plus - pose_.tau_obs);
                jacobians[0][g] = (residual_plus - residuals[0]) / h;
            }
        }

        return true;
    }

private:
    Character* character_;
    PoseData pose_;
    std::map<int, std::vector<int>> muscle_groups_;
    std::map<int, double> base_lm_contract_;
};

// Ratio regularization: penalizes deviation from ratio=1.0
struct RatioRegCost {
    int group_idx;
    double sqrt_lambda;

    RatioRegCost(int g, double sl)
        : group_idx(g), sqrt_lambda(sl) {}

    bool operator()(const double* const* parameters, double* residual) const {
        residual[0] = sqrt_lambda * (parameters[0][group_idx] - 1.0);
        return true;
    }
};

// Torque regularization: penalizes passive torque magnitude
// This functor computes group passive torque at a specific trial pose
struct TorqueRegCost {
    Character* character;
    PoseData pose;
    int group_id;
    std::vector<int> muscle_indices;
    std::map<int, double> base_lm_contract;
    double sqrt_lambda;

    TorqueRegCost(Character* c, const PoseData& p, int g,
                  const std::vector<int>& m_ids,
                  const std::map<int, double>& base_lm,
                  double sl)
        : character(c), pose(p), group_id(g), muscle_indices(m_ids),
          base_lm_contract(base_lm), sqrt_lambda(sl) {}

    bool operator()(const double* const* parameters, double* residual) const {
        if (!character) {
            residual[0] = 0.0;
            return true;
        }

        auto skeleton = character->getSkeleton();
        auto& muscles = character->getMuscles();

        // Apply ratio to this group's muscles
        double ratio = parameters[0][group_id];
        std::vector<double> old_lm_contract(muscles.size());
        for (int m_idx : muscle_indices) {
            old_lm_contract[m_idx] = muscles[m_idx]->lm_contract;
            auto it = base_lm_contract.find(m_idx);
            double base = (it != base_lm_contract.end()) ? it->second : muscles[m_idx]->lm_contract;
            muscles[m_idx]->lm_contract = base * ratio;
            muscles[m_idx]->RefreshMuscleParams();
        }

        // Set pose and update geometry
        skeleton->setPositions(pose.q);
        for (auto& m : muscles) {
            m->UpdateGeometry();
        }

        // Compute group passive torque at target DOF (or composite axis)
        auto* joint = skeleton->getJoint(pose.joint_idx);
        int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
        int num_dofs = static_cast<int>(joint->getNumDofs());

        double group_torque = 0.0;

        if (pose.use_composite_axis) {
            // Composite mode: project 3D torque onto axis
            for (int m_idx : muscle_indices) {
                auto* muscle = muscles[m_idx];
                Eigen::VectorXd jtp = muscle->GetRelatedJtp();
                const auto& related_indices = muscle->related_dof_indices;

                Eigen::Vector3d torque_vec = Eigen::Vector3d::Zero();
                for (size_t i = 0; i < related_indices.size(); ++i) {
                    int global_dof = related_indices[i];
                    int local_dof = global_dof - first_dof;
                    if (local_dof >= 0 && local_dof < num_dofs && local_dof < 3) {
                        torque_vec[local_dof] = jtp[i];
                    }
                }
                group_torque += torque_vec.dot(pose.composite_axis);
            }
        } else {
            // Single-DOF mode: existing logic
            int target_dof = first_dof + pose.joint_dof;
            for (int m_idx : muscle_indices) {
                auto* muscle = muscles[m_idx];
                Eigen::VectorXd jtp = muscle->GetRelatedJtp();
                const auto& related_indices = muscle->related_dof_indices;

                for (size_t i = 0; i < related_indices.size(); ++i) {
                    if (related_indices[i] == target_dof) {
                        group_torque += jtp[i];
                        break;
                    }
                }
            }
        }

        // Restore original lm_contract
        for (int m_idx : muscle_indices) {
            muscles[m_idx]->lm_contract = old_lm_contract[m_idx];
            muscles[m_idx]->RefreshMuscleParams();
        }

        residual[0] = sqrt_lambda * group_torque;
        return true;
    }
};


ROMTrialConfig ContractureOptimizer::loadROMConfig(
    const std::string& yaml_path,
    dart::dynamics::SkeletonPtr skeleton) {

    ROMTrialConfig config;

    std::string resolved = rm::resolve(yaml_path);
    YAML::Node node = YAML::LoadFile(resolved);

    config.name = node["name"].as<std::string>("");
    config.description = node["description"].as<std::string>("");

    // Parse pose from YAML and convert to full skeleton positions
    if (node["pose"] && skeleton) {
        // Start from zero pose
        Eigen::VectorXd positions = Eigen::VectorXd::Zero(skeleton->getNumDofs());

        for (const auto& joint_node : node["pose"]) {
            std::string joint_name = joint_node.first.as<std::string>();
            auto* joint = skeleton->getJoint(joint_name);
            if (!joint) continue;

            YAML::Node values = joint_node.second;
            int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
            size_t num_dofs = joint->getNumDofs();

            // Determine how many DOFs are rotational vs translational
            // FreeJoint: 6 DOFs (0-2 rotation, 3-5 translation)
            // BallJoint: 3 DOFs (all rotation)
            // RevoluteJoint: 1 DOF (rotation)
            // PrismaticJoint: 1 DOF (translation)
            size_t num_rot_dofs = num_dofs;  // Default: all are rotation
            std::string joint_type = joint->getType();
            if (joint_type == "FreeJoint") {
                num_rot_dofs = 3;  // First 3 are rotation, last 3 are translation
            } else if (joint_type == "PrismaticJoint") {
                num_rot_dofs = 0;  // All translation
            }

            if (values.IsSequence()) {
                for (size_t i = 0; i < values.size() && i < num_dofs; ++i) {
                    double val = values[i].as<double>();
                    if (i < num_rot_dofs) {
                        // Rotation DOF: convert degrees to radians
                        positions[first_dof + i] = val * M_PI / 180.0;
                    } else {
                        // Translation DOF: use value directly (meters)
                        positions[first_dof + i] = val;
                    }
                }
            } else {
                double val = values.as<double>();
                if (num_rot_dofs > 0) {
                    positions[first_dof] = val * M_PI / 180.0;
                } else {
                    positions[first_dof] = val;
                }
            }
        }
        config.pose = positions;
    }

    // Load target joint (simplified format - no angle_sweep section)
    config.joint = node["joint"].as<std::string>("");

    // Parse dof field - can be int or string (composite DOF type)
    if (node["dof"]) {
        try {
            // Try parsing as integer first
            config.dof_index = node["dof"].as<int>();
            config.is_composite_dof = false;
        } catch (const YAML::BadConversion&) {
            // It's a string - composite DOF type (e.g., "abd_knee")
            config.dof_type = node["dof"].as<std::string>();
            config.is_composite_dof = true;
            config.dof_index = 0;  // Not used for composite
        }
    } else if (node["dof_index"]) {
        // Backward compatibility with old format
        config.dof_index = node["dof_index"].as<int>(0);
        config.is_composite_dof = false;
    } else {
        // Default to DOF 0 if neither specified
        config.dof_index = 0;
        config.is_composite_dof = false;
    }

    // Load torque cutoff value (renamed from torque)
    config.torque_cutoff = node["torque_cutoff"].as<double>(
        node["torque"].as<double>(15.0));  // Backward compat: try old "torque" key

    // ROM angle defaults to 0 - populated later from clinical data or manual input
    config.rom_angle = 0.0;

    // Load exam sweep parameters (for PhysicalExam)
    if (node["exam"]) {
        auto exam = node["exam"];
        config.angle_min = exam["angle_min"].as<double>(-90.0);
        config.angle_max = exam["angle_max"].as<double>(90.0);
        config.num_steps = exam["num_steps"].as<int>(100);
        config.angle_step = exam["angle_step"].as<double>(1.0);
    }

    // Load clinical_data reference
    if (node["clinical_data"]) {
        config.cd_side = node["clinical_data"]["side"].as<std::string>("");
        config.cd_joint = node["clinical_data"]["joint"].as<std::string>("");
        config.cd_field = node["clinical_data"]["field"].as<std::string>("");
        config.cd_neg = node["clinical_data"]["neg"].as<bool>(false);
        config.cd_cutoff = node["clinical_data"]["cutoff"].as<double>(-1.0);
    }

    // Note: uniform_search_group removed - now using centralized grid_search_mapping

    // Load IK parameters for composite DOF
    config.shank_scale = node["shank_scale"].as<double>(0.7);

    return config;
}


int ContractureOptimizer::loadMuscleGroups(const std::string& yaml_path, Character* character) {
    mMuscleGroups.clear();
    mGroupNames.clear();

    if (!character) {
        std::cerr << "[ContractureOptimizer] No character for loadMuscleGroups" << std::endl;
        return 0;
    }

    // Build muscle name -> index map
    const auto& muscles = character->getMuscles();
    std::map<std::string, int> name_to_idx;
    for (size_t i = 0; i < muscles.size(); ++i) {
        name_to_idx[muscles[i]->name] = static_cast<int>(i);
    }

    // Load YAML
    std::string resolved = rm::resolve(yaml_path);
    YAML::Node root = YAML::LoadFile(resolved);

    int group_id = 0;
    for (const auto& group_node : root) {
        std::string group_name = group_node.first.as<std::string>();
        YAML::Node muscle_list = group_node.second;

        if (!muscle_list.IsSequence()) {
            std::cerr << "[ContractureOptimizer] Group '" << group_name
                      << "' is not a sequence, skipping" << std::endl;
            continue;
        }

        std::vector<int> muscle_indices;
        for (const auto& muscle_node : muscle_list) {
            std::string muscle_name = muscle_node.as<std::string>();

            auto it = name_to_idx.find(muscle_name);
            if (it != name_to_idx.end()) {
                muscle_indices.push_back(it->second);
            } else {
                std::cerr << "[ContractureOptimizer] Muscle '" << muscle_name
                          << "' not found in character, skipping" << std::endl;
            }
        }

        if (!muscle_indices.empty()) {
            mMuscleGroups[group_id] = muscle_indices;
            mGroupNames[group_id] = group_name;
            group_id++;
        }
    }

    std::cout << "[ContractureOptimizer] Loaded " << mMuscleGroups.size()
              << " muscle groups from " << yaml_path << std::endl;

    return static_cast<int>(mMuscleGroups.size());
}


int ContractureOptimizer::getJointIndex(
    dart::dynamics::SkeletonPtr skeleton,
    const std::string& joint_name) {

    for (size_t i = 0; i < skeleton->getNumJoints(); ++i) {
        if (skeleton->getJoint(i)->getName() == joint_name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}


double ContractureOptimizer::computePassiveTorque(Character* character, int joint_idx, bool verbose,
                                                  int dof_offset, bool use_global_y) {
    if (!character) return 0.0;
    if (character->getMuscles().empty()) return 0.0;

    auto skel = character->getSkeleton();
    auto joint = skel->getJoint(joint_idx);
    if (!joint) return 0.0;

    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int num_dofs = static_cast<int>(joint->getNumDofs());

    const auto& muscles = character->getMuscles();

    // Mode 1: Global Y-axis projection (for abd_knee composite DOF)
    // Compute physical torque about joint center and project onto global Y axis
    if (dof_offset >= 0 && use_global_y && num_dofs == 3) {
        auto* child_body = joint->getChildBodyNode();

        // Get joint center in world coordinates
        Eigen::Vector3d joint_center = child_body->getTransform().translation();

        // Build set of descendant bodies (bodies affected by this joint)
        std::set<dart::dynamics::BodyNode*> descendant_bodies;
        std::function<void(dart::dynamics::BodyNode*)> collect_descendants;
        collect_descendants = [&](dart::dynamics::BodyNode* bn) {
            descendant_bodies.insert(bn);
            for (size_t i = 0; i < bn->getNumChildBodyNodes(); ++i) {
                collect_descendants(bn->getChildBodyNode(i));
            }
        };
        collect_descendants(child_body);

        double total_torque = 0.0;
        for (auto& muscle : muscles) {
            Eigen::Vector3d torque_world = muscle->GetPassiveTorqueAboutPoint(
                joint_center, &descendant_bodies);
            double contribution = torque_world.y();
            total_torque += contribution;

            if (verbose && std::abs(contribution) > 1e-6) {
                LOG_INFO("    " << muscle->name << " Y-torque=" << contribution << " Nm");
            }
        }
        return total_torque;
    }

    // Mode 2: Specific DOF only (for simple single-DOF sweeps)
    // Filter to only the specified DOF, ensuring L/R symmetry
    if (dof_offset >= 0 && !use_global_y) {
        int target_dof = first_dof + dof_offset;
        double total_torque = 0.0;
        for (auto& muscle : muscles) {
            Eigen::VectorXd jtp = muscle->GetRelatedJtp();
            const auto& related_indices = muscle->related_dof_indices;

            for (size_t i = 0; i < related_indices.size(); ++i) {
                if (related_indices[i] == target_dof) {
                    total_torque += jtp[i];
                    if (verbose && std::abs(jtp[i]) > 1e-6) {
                        LOG_INFO("    " << muscle->name << " DOF " << target_dof
                                 << " jtp=" << jtp[i] << " Nm");
                    }
                    break;  // Only one contribution per muscle per DOF
                }
            }
        }
        return total_torque;
    }

    // Mode 3: Sum all DOFs in joint (default, for backward compatibility)
    double total_torque = 0.0;
    for (auto& muscle : muscles) {
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        const auto& related_indices = muscle->related_dof_indices;

        for (size_t i = 0; i < related_indices.size(); ++i) {
            int global_dof = related_indices[i];
            if (global_dof >= first_dof && global_dof < first_dof + num_dofs) {
                total_torque += jtp[i];
                if (verbose && std::abs(jtp[i]) > 1e-6) {
                    LOG_INFO("    " << muscle->name << " DOF " << global_dof
                             << " jtp=" << jtp[i] << " Nm");
                }
            }
        }
    }

    return total_torque;
}


Eigen::Vector3d ContractureOptimizer::computeAbdKneeAxis(
    dart::dynamics::SkeletonPtr skeleton,
    int hip_joint_idx) {

    auto* hip_joint = skeleton->getJoint(hip_joint_idx);
    if (!hip_joint) {
        LOG_WARN("computeAbdKneeAxis: Hip joint not found");
        return Eigen::Vector3d::UnitZ();
    }

    // Get hip joint center (femur body origin)
    auto* femur_body = hip_joint->getChildBodyNode();
    if (!femur_body) {
        LOG_WARN("computeAbdKneeAxis: Femur body not found");
        return Eigen::Vector3d::UnitZ();
    }
    Eigen::Vector3d hip_pos = femur_body->getWorldTransform().translation();

    // Get knee joint center (child of femur)
    if (femur_body->getNumChildJoints() == 0) {
        LOG_WARN("computeAbdKneeAxis: Femur has no child joints");
        return Eigen::Vector3d::UnitZ();
    }
    auto* knee_joint = femur_body->getChildJoint(0);
    if (!knee_joint) {
        LOG_WARN("computeAbdKneeAxis: Knee joint not found");
        return Eigen::Vector3d::UnitZ();
    }

    auto* tibia_body = knee_joint->getChildBodyNode();
    if (!tibia_body) {
        LOG_WARN("computeAbdKneeAxis: Tibia body not found");
        return Eigen::Vector3d::UnitZ();
    }
    Eigen::Vector3d knee_pos = tibia_body->getWorldTransform().translation();

    // Hip-knee vector (world frame)
    Eigen::Vector3d hip_knee = (knee_pos - hip_pos);
    if (hip_knee.norm() < 1e-6) {
        LOG_WARN("computeAbdKneeAxis: Hip and knee at same position");
        return Eigen::Vector3d::UnitZ();
    }
    hip_knee.normalize();

    // World Y axis
    Eigen::Vector3d world_y = Eigen::Vector3d::UnitY();

    // Rotation axis = cross product of world Y and hip-knee vector
    // This gives the axis perpendicular to the plane of abduction measurement
    Eigen::Vector3d axis_world = world_y.cross(hip_knee);
    if (axis_world.norm() < 1e-6) {
        // hip_knee is parallel to Y, use X axis as fallback
        axis_world = Eigen::Vector3d::UnitX();
    }
    axis_world.normalize();

    // Transform to joint-local frame for correct torque projection
    Eigen::Matrix3d joint_rotation = femur_body->getWorldTransform().linear();
    Eigen::Vector3d local_axis = joint_rotation.transpose() * axis_world;

    return local_axis.normalized();
}


double ContractureOptimizer::computeKneeAngleForVerticalShank(
    dart::dynamics::SkeletonPtr skeleton,
    int hip_joint_idx,
    const Eigen::Vector3d& hip_positions) {

    // Get joint references
    auto* hip_joint = skeleton->getJoint(hip_joint_idx);
    if (!hip_joint) return 0.0;
    auto* femur_body = hip_joint->getChildBodyNode();
    if (!femur_body || femur_body->getNumChildJoints() == 0) return 0.0;
    auto* knee_joint = femur_body->getChildJoint(0);
    if (!knee_joint) return 0.0;
    auto* tibia_body = knee_joint->getChildBodyNode();
    if (!tibia_body || tibia_body->getNumChildJoints() == 0) return 0.0;
    auto* ankle_joint = tibia_body->getChildJoint(0);
    if (!ankle_joint) return 0.0;

    // Save positions
    Eigen::VectorXd old_pos = skeleton->getPositions();
    int hip_dof_start = static_cast<int>(hip_joint->getIndexInSkeleton(0));
    int knee_dof_idx = static_cast<int>(knee_joint->getIndexInSkeleton(0));

    // Apply hip rotation, set knee to 0
    skeleton->setPosition(hip_dof_start, hip_positions[0]);
    skeleton->setPosition(hip_dof_start + 1, hip_positions[1]);
    skeleton->setPosition(hip_dof_start + 2, hip_positions[2]);
    skeleton->setPosition(knee_dof_idx, 0.0);

    // Get shank direction at knee=0 using joint positions directly
    Eigen::Vector3d knee_pos = (knee_joint->getParentBodyNode()->getWorldTransform() *
                                knee_joint->getTransformFromParentBodyNode()).translation();
    Eigen::Vector3d ankle_pos = (ankle_joint->getParentBodyNode()->getWorldTransform() *
                                 ankle_joint->getTransformFromParentBodyNode()).translation();
    Eigen::Vector3d shank_init = (ankle_pos - knee_pos).normalized();

    // Compute angle to vertical (-Y) using dot product
    Eigen::Vector3d shank_target = -Eigen::Vector3d::UnitY();
    double cos_angle = shank_init.dot(shank_target);
    double knee_angle = std::acos(cos_angle);

    // Restore positions
    skeleton->setPositions(old_pos);

    return knee_angle;
}


AbdKneePoseResult ContractureOptimizer::computeAbdKneePose(
    dart::dynamics::SkeletonPtr skeleton,
    int hip_joint_idx,
    double rom_angle_deg,
    bool is_left_leg,
    double shank_scale) {

    AbdKneePoseResult result;
    result.hip_positions.setZero();
    result.knee_angle = 0.0;
    result.success = false;

    double alpha = rom_angle_deg * M_PI / 180.0;

    // Get joint references
    auto* hip_joint = skeleton->getJoint(hip_joint_idx);
    if (!hip_joint) {
        LOG_WARN("computeAbdKneePose: Hip joint not found");
        return result;
    }

    auto* femur_body = hip_joint->getChildBodyNode();
    if (!femur_body) {
        LOG_WARN("computeAbdKneePose: Femur body not found");
        return result;
    }

    if (femur_body->getNumChildJoints() == 0) {
        LOG_WARN("computeAbdKneePose: No knee joint found");
        return result;
    }

    auto* knee_joint = femur_body->getChildJoint(0);
    auto* tibia_body = knee_joint->getChildBodyNode();
    if (!tibia_body) {
        LOG_WARN("computeAbdKneePose: Tibia body not found");
        return result;
    }

    // Get ankle joint for segment length calculation
    if (tibia_body->getNumChildJoints() == 0) {
        LOG_WARN("computeAbdKneePose: No ankle joint found");
        return result;
    }
    auto* ankle_joint = tibia_body->getChildJoint(0);
    auto* talus_body = ankle_joint->getChildBodyNode();
    if (!talus_body) {
        LOG_WARN("computeAbdKneePose: Talus body not found");
        return result;
    }

    // Get segment lengths by computing distance between actual joint positions
    // Save current positions
    Eigen::VectorXd orig_pos = skeleton->getPositions();
    skeleton->setPositions(Eigen::VectorXd::Zero(skeleton->getNumDofs()));

    // Get joint world positions directly using joint transforms
    Eigen::Vector3d hip_world = (hip_joint->getParentBodyNode()->getWorldTransform() *
                                 hip_joint->getTransformFromParentBodyNode()).translation();
    Eigen::Vector3d knee_world = (knee_joint->getParentBodyNode()->getWorldTransform() *
                                  knee_joint->getTransformFromParentBodyNode()).translation();
    Eigen::Vector3d ankle_world = (ankle_joint->getParentBodyNode()->getWorldTransform() *
                                   ankle_joint->getTransformFromParentBodyNode()).translation();

    // Restore original positions
    skeleton->setPositions(orig_pos);

    // Compute segment lengths from actual joint-to-joint distances
    double thigh_length = (knee_world - hip_world).norm();
    double shank_length = (ankle_world - knee_world).norm();

    // Scale shank for IK geometry constraint
    // (actual skeleton may have shank > thigh which violates abd_knee geometry)
    shank_length *= shank_scale;

    // Geometry constraint: with foot at pelvis level and shank vertical,
    // the horizontal distance from hip to knee projection is:
    // d = sqrt(thigh^2 - shank^2)
    if (shank_length > thigh_length) {
        LOG_WARN("computeAbdKneePose: Shank longer than thigh - invalid geometry");
        return result;
    }
    double d = std::sqrt(thigh_length * thigh_length - shank_length * shank_length);

    // Target thigh direction in world frame (supine pose):
    // Shank is vertical (+Y), knee is ABOVE hip by shank_length
    // Abduction angle alpha is measured from vertical to hip-ankle line
    // Geometry:
    //   x = ±shank * tan(alpha)  (lateral: +X for left leg, -X for right leg)
    //   d² = thigh² - shank² = x² + z²  (horizontal projection constraint)
    //   z = sqrt(d² - x²)  (forward displacement)
    double sign = is_left_leg ? 1.0 : -1.0;
    double x = sign * shank_length * std::tan(alpha);
    double x_sq = x * x;
    double d_sq = d * d;
    if (x_sq > d_sq) {
        LOG_WARN("computeAbdKneePose: Abduction angle too large - |x|=" << std::abs(x) << " > d=" << d);
        return result;
    }
    double z = std::sqrt(d_sq - x_sq);

    // Knee position relative to hip: (x, shank_length, z)
    // Thigh direction = knee_pos / thigh_length
    Eigen::Vector3d target_knee_rel(x, shank_length, z);  // relative to hip
    Eigen::Vector3d target_thigh_world = target_knee_rel / thigh_length;  // Normalize

    // Initial thigh direction in supine pose with hip DOFs at zero
    // Compute from actual joint positions (not assuming local -Y)

    // Save current pose and set hip to zero
    Eigen::VectorXd saved_pos = skeleton->getPositions();
    int hip_dof_start = static_cast<int>(hip_joint->getIndexInSkeleton(0));
    skeleton->setPosition(hip_dof_start, 0.0);
    skeleton->setPosition(hip_dof_start + 1, 0.0);
    skeleton->setPosition(hip_dof_start + 2, 0.0);

    // Get actual thigh direction from hip-to-knee vector (with hip DOFs at zero)
    // Use joint world positions directly
    Eigen::Vector3d hip_pos_init = (hip_joint->getParentBodyNode()->getWorldTransform() *
                                    hip_joint->getTransformFromParentBodyNode()).translation();
    Eigen::Vector3d knee_pos_init = (knee_joint->getParentBodyNode()->getWorldTransform() *
                                     knee_joint->getTransformFromParentBodyNode()).translation();
    Eigen::Vector3d initial_thigh_world = (knee_pos_init - hip_pos_init).normalized();

    // Get joint frame rotation in world coordinates
    // The BallJoint rotation operates in the JOINT frame, not the parent body frame
    // Transform chain: thigh_world = R_parent * R_parent_to_joint * R_hip * ... * thigh_local
    // So we need: joint_frame = R_parent * R_parent_to_joint
    Eigen::Matrix3d parent_rot = hip_joint->getParentBodyNode()->getWorldTransform().linear();
    Eigen::Matrix3d parent_to_joint = hip_joint->getTransformFromParentBodyNode().linear();
    Eigen::Matrix3d joint_frame = parent_rot * parent_to_joint;

    // Transform thigh directions to joint frame
    Eigen::Vector3d initial_thigh_joint = joint_frame.transpose() * initial_thigh_world;
    Eigen::Vector3d target_thigh_joint = joint_frame.transpose() * target_thigh_world;

    // Compute rotation in joint frame
    // Step 1: Base rotation from initial_thigh to target_thigh (shortest path)
    Eigen::Quaterniond q_base = Eigen::Quaterniond::FromTwoVectors(
        initial_thigh_joint, target_thigh_joint);
    Eigen::Matrix3d R_base = q_base.toRotationMatrix();

    // Step 2: Compute twist angle around thigh axis to make knee axis horizontal
    // Apply base rotation to skeleton to get knee axis directly from skeleton state
    Eigen::Vector3d base_hip_pos = dart::dynamics::BallJoint::convertToPositions(R_base);
    skeleton->setPosition(hip_dof_start, base_hip_pos[0]);
    skeleton->setPosition(hip_dof_start + 1, base_hip_pos[1]);
    skeleton->setPosition(hip_dof_start + 2, base_hip_pos[2]);

    // Get knee axis directly from skeleton state (more reliable than manual transform chain)
    Eigen::Matrix3d femur_to_knee = knee_joint->getTransformFromParentBodyNode().linear();
    Eigen::Vector3d knee_axis_after_base = (femur_body->getWorldTransform().linear() * femur_to_knee.col(0)).normalized();

    // We need to twist around target_thigh_world to make knee_axis_final · Y = 0
    // Using Rodrigues formula: k_rot = k*cos(θ) + (a×k)*sin(θ) + a*(a·k)*(1-cos(θ))
    // where a = target_thigh_world, k = knee_axis_after_base
    // Constraint: k_rot · Y = 0
    Eigen::Vector3d a = target_thigh_world;
    Eigen::Vector3d k = knee_axis_after_base;
    Eigen::Vector3d y = Eigen::Vector3d::UnitY();

    double A = k.dot(y);                        // k · Y
    double B = (a.cross(k)).dot(y);             // (a × k) · Y
    double C = a.dot(k);                        // a · k
    double D = a.dot(y);                        // a · Y

    // Equation: (A - C*D)*cos(θ) + B*sin(θ) + C*D = 0
    double P = A - C * D;
    double Q = B;
    double R = C * D;

    // Solve: P*cos(θ) + Q*sin(θ) = -R
    // Solution: θ = atan2(Q, P) ± acos(-R / sqrt(P² + Q²))
    double twist_angle = 0.0;
    double denom = std::sqrt(P * P + Q * Q);
    if (denom > 1e-9) {
        double cos_val = -R / denom;
        cos_val = std::max(-1.0, std::min(1.0, cos_val));  // Clamp to [-1, 1]
        double phi = std::atan2(Q, P);
        // Choose solution that gives smaller twist angle
        double theta1 = phi + std::acos(cos_val);
        double theta2 = phi - std::acos(cos_val);
        twist_angle = (std::abs(theta1) < std::abs(theta2)) ? theta1 : theta2;
    }

    // Apply twist rotation: R_twist around target_thigh_world
    Eigen::AngleAxisd twist(twist_angle, target_thigh_world);
    Eigen::Matrix3d R_twist = twist.toRotationMatrix();

    // Transform twist to joint frame and combine with base rotation
    Eigen::Matrix3d R_twist_joint = joint_frame.transpose() * R_twist * joint_frame;
    Eigen::Matrix3d R_hip_local = R_twist_joint * R_base;

    // Convert to axis-angle using DART's BallJoint method
    result.hip_positions = dart::dynamics::BallJoint::convertToPositions(R_hip_local);

    // Restore saved positions before computing knee angle
    skeleton->setPositions(saved_pos);

    // Compute knee angle to keep shank vertical
    result.knee_angle = computeKneeAngleForVerticalShank(
        skeleton, hip_joint_idx, result.hip_positions);

    result.success = true;
    return result;
}


std::vector<PoseData> ContractureOptimizer::buildPoseData(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs) {

    std::vector<PoseData> data;
    auto skeleton = character->getSkeleton();

    for (const auto& config : rom_configs) {
        // Start with base pose from config (already full skeleton positions)
        if (config.pose.size() == 0) {
            std::cerr << "[ContractureOptimizer] Empty pose for config: " << config.name << std::endl;
            continue;
        }

        Eigen::VectorXd positions = config.pose;
        int joint_idx = getJointIndex(skeleton, config.joint);
        if (joint_idx < 0) {
            std::cerr << "[ContractureOptimizer] Joint not found: " << config.joint << std::endl;
            continue;
        }

        auto* joint = skeleton->getJoint(joint_idx);
        if (!joint || config.dof_index >= static_cast<int>(joint->getNumDofs())) {
            std::cerr << "[ContractureOptimizer] Invalid DOF index for joint: " << config.joint << std::endl;
            continue;
        }

        // Set ROM joint to clinical angle (single point, not sweep)
        // For composite DOF, the pose from YAML is the final pose - no single-DOF override
        int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
        if (!config.is_composite_dof) {
            double angle_rad = config.rom_angle * M_PI / 180.0;
            positions[first_dof + config.dof_index] = angle_rad;
        }
        // For composite DOF, rom_angle is informational only - pose must be fully specified in YAML

        // Record single pose data point
        PoseData point;
        point.joint_idx = joint_idx;
        point.joint_dof = config.dof_index;
        point.q = positions;
        point.tau_obs = config.torque_cutoff;
        point.weight = 1.0;

        // Handle composite DOF types
        if (config.is_composite_dof) {
            point.use_composite_axis = true;

            if (config.dof_type == "abd_knee") {
                // Determine if left or right leg from joint name
                bool is_left = (config.joint.find("L") != std::string::npos);

                // IMPORTANT: Set skeleton to base pose BEFORE computing IK
                // This ensures the pelvis supine rotation is applied
                skeleton->setPositions(positions);

                // Compute IK pose from clinical rom_angle
                AbdKneePoseResult ik_result = computeAbdKneePose(
                    skeleton, joint_idx, config.rom_angle, is_left, config.shank_scale);

                if (ik_result.success) {
                    // Apply hip joint positions (axis-angle)
                    int hip_dof_start = static_cast<int>(joint->getIndexInSkeleton(0));
                    positions[hip_dof_start] = ik_result.hip_positions[0];
                    positions[hip_dof_start + 1] = ik_result.hip_positions[1];
                    positions[hip_dof_start + 2] = ik_result.hip_positions[2];

                    // Apply knee angle
                    auto* femur_body = joint->getChildBodyNode();
                    if (femur_body && femur_body->getNumChildJoints() > 0) {
                        auto* knee_joint = femur_body->getChildJoint(0);
                        int knee_dof_idx = static_cast<int>(knee_joint->getIndexInSkeleton(0));
                        positions[knee_dof_idx] = ik_result.knee_angle;
                    }

                    // Update pose in PoseData
                    point.q = positions;

                } else {
                    LOG_WARN("[Contracture] abd_knee IK failed for trial: " << config.name);
                }

                // Set pose and compute composite axis
                skeleton->setPositions(positions);
                point.composite_axis = computeAbdKneeAxis(skeleton, joint_idx);
            } else {
                // Set pose first to compute correct joint positions for axis calculation
                skeleton->setPositions(positions);
                LOG_WARN("[Contracture] Unknown composite DOF type: " << config.dof_type);
                point.composite_axis = Eigen::Vector3d::UnitZ();
            }
        } else {
            point.use_composite_axis = false;
            point.composite_axis = Eigen::Vector3d::Zero();
        }

        data.push_back(point);
    }

    return data;
}


int ContractureOptimizer::findGroupIdByName(const std::string& name) const {
    for (const auto& [id, gname] : mGroupNames) {
        if (gname == name) return id;
    }
    return -1;
}


double ContractureOptimizer::findBestInitialRatio(
    Character* character,
    const PoseData& pose,
    int group_id,
    const std::map<int, double>& base_lm_contract,
    const Config& config) {

    auto& muscles = character->getMuscles();
    auto skeleton = character->getSkeleton();

    // Get muscle indices for this group
    auto group_it = mMuscleGroups.find(group_id);
    if (group_it == mMuscleGroups.end()) {
        return 1.0;  // Default if group not found
    }
    const auto& muscle_indices = group_it->second;

    double best_ratio = 1.0;
    double best_error = std::numeric_limits<double>::max();

    // Store original lm_contract values
    std::vector<double> original_lm_contract(muscles.size());
    for (size_t i = 0; i < muscles.size(); ++i) {
        original_lm_contract[i] = muscles[i]->lm_contract;
    }

    // Grid search over ratio values
    for (double ratio = config.gridSearchBegin;
         ratio <= config.gridSearchEnd + 1e-6;
         ratio += config.gridSearchInterval) {

        // Apply ratio to this group's muscles
        for (int m_idx : muscle_indices) {
            auto it = base_lm_contract.find(m_idx);
            double base = (it != base_lm_contract.end()) ? it->second : muscles[m_idx]->lm_contract;
            muscles[m_idx]->lm_contract = base * ratio;
            muscles[m_idx]->RefreshMuscleParams();
        }

        // Set pose and update muscle geometry
        setPoseAndUpdateGeometry(character, pose.q);

        // Compute passive torque (use specific DOF for symmetry)
        int dof_offset = pose.use_composite_axis ? 1 : pose.joint_dof;
        bool use_global_y = pose.use_composite_axis;
        double computed_torque = computePassiveTorque(character, pose.joint_idx, false, dof_offset, use_global_y);
        double error = std::abs(computed_torque - pose.tau_obs);

        if (config.verbose) {
            LOG_INFO("[GridSearch] ratio=" << ratio << " torque=" << computed_torque
                     << " target=" << pose.tau_obs << " error=" << error);
        }

        if (error < best_error) {
            best_error = error;
            best_ratio = ratio;
        }
    }

    // Restore original lm_contract values
    for (size_t i = 0; i < muscles.size(); ++i) {
        muscles[i]->lm_contract = original_lm_contract[i];
        muscles[i]->RefreshMuscleParams();
    }

    return best_ratio;
}


std::vector<MuscleGroupResult> ContractureOptimizer::optimize(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs,
    const Config& config) {

    std::vector<MuscleGroupResult> results;

    if (!character) {
        std::cerr << "[ContractureOptimizer] No character provided" << std::endl;
        return results;
    }

    if (mMuscleGroups.empty()) {
        std::cerr << "[ContractureOptimizer] No muscle groups configured. "
                  << "Call loadMuscleGroups() or detectMuscleGroups() first." << std::endl;
        return results;
    }

    int num_groups = static_cast<int>(mMuscleGroups.size());
    std::cout << "[ContractureOptimizer] Optimizing " << num_groups << " muscle groups" << std::endl;

    // Store base lm_contract values
    const auto& muscles = character->getMuscles();
    std::map<int, double> base_lm_contract;
    for (size_t i = 0; i < muscles.size(); ++i) {
        base_lm_contract[static_cast<int>(i)] = muscles[i]->lm_contract;
    }

    // Build pose data
    std::vector<PoseData> pose_data = buildPoseData(character, rom_configs);

    if (pose_data.empty()) {
        std::cerr << "[ContractureOptimizer] No pose data loaded" << std::endl;
        return results;
    }

    std::cout << "[ContractureOptimizer] Built " << pose_data.size() << " pose data points" << std::endl;

    // Initialize parameters (all ratios = 1.0)
    std::vector<double> x(num_groups, 1.0);

    // Build mapping from group_id to trial index (for passive torque computation)
    std::map<int, size_t> group_to_trial = buildGroupToTrialMapping(rom_configs, pose_data);

    // Grid search initialization using centralized mapping
    runGridSearchInitialization(character, rom_configs, pose_data, base_lm_contract, config, mGridSearchMapping, x);

    // Build torque matrix BEFORE optimization: torque_matrix_before[group_id][trial_idx]
    std::map<int, std::vector<double>> torque_matrix_before;
    for (int g = 0; g < num_groups; ++g) {
        torque_matrix_before[g].resize(pose_data.size());
        for (size_t t = 0; t < pose_data.size(); ++t) {
            torque_matrix_before[g][t] = computeGroupTorqueAtTrial(character, pose_data, g, t);
        }
    }

    // Build Ceres problem
    ceres::Problem problem;

    // Add torque residual blocks
    for (const auto& pose : pose_data) {
        auto* cost = new TorqueResidual(character, pose, mMuscleGroups, base_lm_contract, num_groups);
        problem.AddResidualBlock(cost, nullptr, x.data());
    }

    // Ratio regularization: penalize deviation from ratio=1.0
    if (config.lambdaRatioReg > 0.0) {
        double sqrt_lambda = std::sqrt(config.lambdaRatioReg);
        for (int g = 0; g < num_groups; ++g) {
            auto* reg_cost = new ceres::DynamicNumericDiffCostFunction<RatioRegCost>(
                new RatioRegCost(g, sqrt_lambda));
            reg_cost->AddParameterBlock(num_groups);
            reg_cost->SetNumResiduals(1);
            problem.AddResidualBlock(reg_cost, nullptr, x.data());
        }
        if (config.verbose) {
            LOG_INFO("[Contracture] Ratio regularization: lambda=" << config.lambdaRatioReg);
        }
    }

    // Torque regularization: penalize passive torque magnitude per group/trial
    if (config.lambdaTorqueReg > 0.0) {
        double sqrt_lambda = std::sqrt(config.lambdaTorqueReg);
        for (int g = 0; g < num_groups; ++g) {
            auto grp_it = mMuscleGroups.find(g);
            if (grp_it == mMuscleGroups.end()) continue;

            for (size_t t = 0; t < pose_data.size(); ++t) {
                auto* reg_cost = new ceres::DynamicNumericDiffCostFunction<TorqueRegCost>(
                    new TorqueRegCost(character, pose_data[t], g,
                                      grp_it->second, base_lm_contract, sqrt_lambda));
                reg_cost->AddParameterBlock(num_groups);
                reg_cost->SetNumResiduals(1);
                problem.AddResidualBlock(reg_cost, nullptr, x.data());
            }
        }
        if (config.verbose) {
            LOG_INFO("[Contracture] Torque regularization: lambda=" << config.lambdaTorqueReg);
        }
    }

    // Set bounds
    for (int g = 0; g < num_groups; ++g) {
        problem.SetParameterLowerBound(x.data(), g, config.minRatio);
        problem.SetParameterUpperBound(x.data(), g, config.maxRatio);
    }

    // Solver options
    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    options.linear_solver_type = (num_groups < 20)
        ? ceres::DENSE_QR
        : ceres::SPARSE_NORMAL_CHOLESKY;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-8;
    options.minimizer_progress_to_stdout = false;  // Disable Ceres verbosity

    // Log initial parameters if verbose (table format: base_name | R | L)
    if (config.verbose) {
        logInitialParameterTable(x);
    }

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Apply optimized ratios to muscles
    for (int g = 0; g < num_groups; ++g) {
        auto grp_it = mMuscleGroups.find(g);
        if (grp_it == mMuscleGroups.end()) continue;

        for (int m_idx : grp_it->second) {
            muscles[m_idx]->lm_contract = base_lm_contract[m_idx] * x[g];
            muscles[m_idx]->RefreshMuscleParams();
        }
    }

    // Build torque matrix AFTER optimization: torque_matrix_after[group_id][trial_idx]
    std::map<int, std::vector<double>> torque_matrix_after;
    for (int g = 0; g < num_groups; ++g) {
        torque_matrix_after[g].resize(pose_data.size());
        for (size_t t = 0; t < pose_data.size(); ++t) {
            torque_matrix_after[g][t] = computeGroupTorqueAtTrial(character, pose_data, g, t);
        }
    }

    // Build results
    // Compute averaged ratios for biarticular muscles
    std::map<int, double> averaged_ratios = computeBiarticularAverages(x, muscles, config.verbose);

    // Log final parameters if verbose
    if (config.verbose) {
        logParameterTable("Final (after optimization)", x, rom_configs,
                          &torque_matrix_before, &torque_matrix_after);
        LOG_INFO("[Contracture] Iterations: " << summary.iterations.size()
                    << ", Final cost: " << summary.final_cost);
    }

    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        MuscleGroupResult result;

        // Get group name from member map
        auto name_it = mGroupNames.find(group_id);
        result.group_name = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(group_id);
        result.ratio = x[group_id];

        for (int m_idx : muscle_ids) {
            result.muscle_names.push_back(muscles[m_idx]->name);

            // Use averaged ratio for biarticular muscles, group ratio otherwise
            double ratio = x[group_id];
            auto avg_it = averaged_ratios.find(m_idx);
            if (avg_it != averaged_ratios.end()) {
                ratio = avg_it->second;
            }

            double new_lm_contract = base_lm_contract[m_idx] * ratio;
            result.lm_contract_values.push_back(new_lm_contract);
        }

        results.push_back(result);
    }

    return results;
}


ContractureOptResult ContractureOptimizer::optimizeWithResults(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs,
    const Config& config) {

    ContractureOptResult result;

    if (!character) {
        std::cerr << "[ContractureOptimizer] No character provided" << std::endl;
        return result;
    }

    if (mMuscleGroups.empty()) {
        std::cerr << "[ContractureOptimizer] No muscle groups configured" << std::endl;
        return result;
    }

    auto skeleton = character->getSkeleton();
    const auto& muscles = character->getMuscles();

    // ============================================================
    // Pre-compute data once before outer iterations
    // (includes BEFORE state capture, pose data, grid search)
    // ============================================================
    PreOptimizationData preOpt = preOptimization(character, rom_configs, config);

    if (preOpt.pose_data.empty()) {
        std::cerr << "[ContractureOptimizer] No pose data loaded" << std::endl;
        return result;
    }

    // Copy BEFORE trial results to result (captured in preOptimization)
    result.trial_results = preOpt.trial_results_before;

    // ============================================================
    // Run optimization (with outer iterations for biarticular convergence)
    // ============================================================
    for (int outer = 0; outer < config.outerIterations; ++outer) {
        result.group_results = optimizeWithPrecomputed(character, rom_configs, config, preOpt, outer);

        if (result.group_results.empty()) {
            std::cerr << "[ContractureOptimizer] Optimization failed at outer iteration "
                      << (outer+1) << std::endl;
            return result;
        }

        // Apply averaged results to muscles (next iteration sees these as base)
        applyResults(character, result.group_results);

        // Log progress: joint torque summary for each trial
        std::cout << "[Outer " << (outer+1) << "/" << config.outerIterations << "]";
        for (size_t t = 0; t < preOpt.pose_data.size(); ++t) {
            const auto& pose = preOpt.pose_data[t];
            setPoseAndUpdateGeometry(character, pose.q);
            auto* joint = skeleton->getJoint(pose.joint_idx);
            int dof_offset = pose.use_composite_axis ? 1 : pose.joint_dof;
            bool use_global_y = pose.use_composite_axis;
            double torque = computePassiveTorque(character, pose.joint_idx, false, dof_offset, use_global_y);
            std::cout << " " << joint->getName() << "[" << pose.joint_dof << "]="
                      << std::fixed << std::setprecision(2) << torque << "Nm"
                      << " (target=" << pose.tau_obs << ")";
        }
        std::cout << std::endl;
    }

    // Compute cumulative group ratios (mean of final/initial across muscles in group)
    computeCumulativeGroupRatios(muscles, preOpt.lm_contract_before, result.group_results);

    // ============================================================
    // Capture AFTER state (lm_contract per muscle)
    // ============================================================
    for (int m_idx : preOpt.all_muscle_indices) {
        MuscleContractureResult m_result;
        m_result.muscle_name = muscles[m_idx]->name;
        m_result.muscle_idx = m_idx;
        m_result.lm_contract_before = preOpt.lm_contract_before.at(m_idx);
        m_result.lm_contract_after = muscles[m_idx]->lm_contract;
        m_result.ratio = m_result.lm_contract_after / m_result.lm_contract_before;
        result.muscle_results.push_back(m_result);
    }

    // ============================================================
    // Capture AFTER passive torques per trial
    // ============================================================
    for (size_t t = 0; t < result.trial_results.size() && t < preOpt.pose_data.size(); ++t) {
        auto& trial_result = result.trial_results[t];
        const auto& pose = preOpt.pose_data[t];

        // Set pose and update muscle geometry
        setPoseAndUpdateGeometry(character, trial_result.pose);

        // Compute AFTER total passive torque
        if (config.verbose) LOG_INFO("[Contracture] Trial '" << trial_result.trial_name << "' AFTER torque:");
        {
            int dof_offset = pose.use_composite_axis ? 1 : pose.joint_dof;
            bool use_global_y = pose.use_composite_axis;
            trial_result.computed_torque_after = computePassiveTorque(character, pose.joint_idx, config.verbose, dof_offset, use_global_y);
        }

        // Per-muscle contribution AFTER
        captureMuscleTorqueContributions(character, pose, preOpt.all_muscle_indices,
                                         trial_result.muscle_torques_after,
                                         trial_result.muscle_forces_after);
    }

    result.converged = true;

    // Copy 1D grid search results
    result.grid_search_1d_results = mGridSearch1DResults;

    std::cout << "[ContractureOptimizer] optimizeWithResults complete: "
              << result.muscle_results.size() << " muscles, "
              << result.trial_results.size() << " trials" << std::endl;

    return result;
}


void ContractureOptimizer::applyResults(
    Character* character,
    const std::vector<MuscleGroupResult>& results) {

    if (!character) return;

    auto& muscles = character->getMuscles();

    for (const auto& result : results) {
        for (size_t i = 0; i < result.muscle_names.size(); ++i) {
            const std::string& name = result.muscle_names[i];
            double lm_contract = result.lm_contract_values[i];

            // Find muscle by name
            for (auto& muscle : muscles) {
                if (muscle->name == name) {
                    muscle->lm_contract = lm_contract;
                    muscle->RefreshMuscleParams();
                    break;
                }
            }
        }
    }

    std::cout << "[ContractureOptimizer] Applied results to " << results.size() << " muscle groups" << std::endl;
}


std::map<int, std::vector<int>> ContractureOptimizer::findBiarticularMuscles() const {
    std::map<int, std::vector<int>> biarticular;  // muscle_idx -> group_ids

    // Build reverse map: muscle_idx -> list of groups containing it
    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        for (int m_idx : muscle_ids) {
            biarticular[m_idx].push_back(group_id);
        }
    }

    // Filter to only keep muscles in multiple groups
    for (auto it = biarticular.begin(); it != biarticular.end(); ) {
        if (it->second.size() <= 1) {
            it = biarticular.erase(it);
        } else {
            ++it;
        }
    }

    return biarticular;
}


// ============================================================
// Private Helper Methods
// ============================================================

bool ContractureOptimizer::groupMatchesJoint(
    const std::string& group_name,
    const std::string& joint_name) const
{
    // Determine side from group name (_l or _r suffix)
    bool group_is_left = (group_name.size() > 2 && group_name.substr(group_name.size() - 2) == "_l");
    bool group_is_right = (group_name.size() > 2 && group_name.substr(group_name.size() - 2) == "_r");

    // Determine side from joint name (ends with L or R)
    bool joint_is_left = (!joint_name.empty() && joint_name.back() == 'L');
    bool joint_is_right = (!joint_name.empty() && joint_name.back() == 'R');

    // Side must match
    if (group_is_left && !joint_is_left) return false;
    if (group_is_right && !joint_is_right) return false;

    // Match joint type to muscle group pattern
    // Tibia* -> knee joint
    if (joint_name.find("Tibia") != std::string::npos) {
        return (group_name.find("knee") != std::string::npos);
    }
    // Femur* -> hip joint
    if (joint_name.find("Femur") != std::string::npos) {
        return (group_name.find("hip") != std::string::npos);
    }
    // Talus* -> ankle joint (plantarflexor, dorsiflexor)
    if (joint_name.find("Talus") != std::string::npos) {
        return (group_name.find("plantarflexor") != std::string::npos ||
                group_name.find("dorsiflexor") != std::string::npos);
    }
    return false;
}

void ContractureOptimizer::setPoseAndUpdateGeometry(
    Character* character,
    const Eigen::VectorXd& q) const
{
    auto skeleton = character->getSkeleton();
    skeleton->setPositions(q);

    for (auto& muscle : character->getMuscles()) {
        muscle->UpdateGeometry();
    }
}

double ContractureOptimizer::computeGroupTorqueAtTrial(
    Character* character,
    const std::vector<PoseData>& pose_data,
    int group_id,
    size_t trial_idx) const
{
    if (trial_idx >= pose_data.size()) return 0.0;

    const auto& pose = pose_data[trial_idx];
    auto skeleton = character->getSkeleton();
    const auto& muscles = character->getMuscles();

    // Set pose and update muscle geometry
    setPoseAndUpdateGeometry(character, pose.q);

    // Sum passive torques from muscles in this group at the trial's target DOF
    auto* joint = skeleton->getJoint(pose.joint_idx);
    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int target_dof = first_dof + pose.joint_dof;

    double group_torque = 0.0;
    auto grp_it = mMuscleGroups.find(group_id);
    if (grp_it != mMuscleGroups.end()) {
        for (int m_idx : grp_it->second) {
            auto* muscle = muscles[m_idx];
            Eigen::VectorXd jtp = muscle->GetRelatedJtp();
            const auto& related_indices = muscle->related_dof_indices;

            for (size_t i = 0; i < related_indices.size(); ++i) {
                if (related_indices[i] == target_dof) {
                    group_torque += jtp[i];
                    break;
                }
            }
        }
    }
    return group_torque;
}

void ContractureOptimizer::computeCumulativeGroupRatios(
    const std::vector<Muscle*>& muscles,
    const std::map<int, double>& lm_contract_before,
    std::vector<MuscleGroupResult>& group_results) const
{
    for (auto& grp : group_results) {
        double sum_ratio = 0.0;
        int count = 0;
        for (const std::string& muscle_name : grp.muscle_names) {
            // Find muscle index by name
            for (size_t m = 0; m < muscles.size(); ++m) {
                if (muscles[m]->name == muscle_name) {
                    auto it = lm_contract_before.find(static_cast<int>(m));
                    double initial = (it != lm_contract_before.end()) ? it->second : 1.0;
                    double final_val = muscles[m]->lm_contract;
                    sum_ratio += final_val / initial;
                    count++;
                    break;
                }
            }
        }
        grp.ratio = (count > 0) ? sum_ratio / count : 1.0;
    }
}

void ContractureOptimizer::captureMuscleTorqueContributions(
    Character* character,
    const PoseData& pose,
    const std::set<int>& muscle_indices,
    std::vector<std::pair<std::string, double>>& out_torques,
    std::vector<std::pair<std::string, double>>& out_forces) const
{
    auto skeleton = character->getSkeleton();
    const auto& muscles = character->getMuscles();

    auto* joint = skeleton->getJoint(pose.joint_idx);
    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int target_dof = first_dof + pose.joint_dof;

    for (int m_idx : muscle_indices) {
        auto* muscle = muscles[m_idx];
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        const auto& related_indices = muscle->related_dof_indices;

        double contrib = 0.0;
        for (size_t i = 0; i < related_indices.size(); ++i) {
            if (related_indices[i] == target_dof) {
                contrib = jtp[i];
                break;
            }
        }
        out_torques.push_back({muscle->name, contrib});

        // Capture passive force (f_p * f0)
        double passive_force = muscle->Getf_p() * muscle->f0;
        out_forces.push_back({muscle->name, passive_force});
    }
}

void ContractureOptimizer::runGridSearchInitialization(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs,
    const std::vector<PoseData>& pose_data,
    const std::map<int, double>& base_lm_contract,
    const Config& config,
    const std::vector<GridSearchMapping>& mapping,
    std::vector<double>& x)
{
    if (mapping.empty()) {
        LOG_INFO("[Contracture] No grid search mapping configured, skipping initialization");
        return;
    }

    // Build a map from trial name to index for quick lookup
    std::map<std::string, size_t> trial_name_to_index;
    for (size_t i = 0; i < rom_configs.size(); ++i) {
        trial_name_to_index[rom_configs[i].name] = i;
    }

    // Track which groups have been initialized
    std::set<int> initialized_groups;

    // Process each mapping entry
    for (const auto& entry : mapping) {
        if (entry.trials.empty() || entry.groups.empty()) continue;

        // Find trial indices that are present in rom_configs
        std::vector<size_t> trial_indices;
        for (const auto& trial_name : entry.trials) {
            auto it = trial_name_to_index.find(trial_name);
            if (it != trial_name_to_index.end()) {
                trial_indices.push_back(it->second);
            }
        }

        if (trial_indices.empty()) {
            // No matching trials in current config, skip this mapping
            continue;
        }

        // Find group IDs
        std::vector<int> group_ids;
        std::vector<std::string> group_names_found;
        for (const auto& group_name : entry.groups) {
            int gid = findGroupIdByName(group_name);
            if (gid >= 0) {
                // Skip if already initialized
                if (initialized_groups.count(gid) == 0) {
                    group_ids.push_back(gid);
                    group_names_found.push_back(group_name);
                }
            } else {
                LOG_WARN("[Contracture] Grid search mapping: group '" << group_name << "' not found");
            }
        }

        if (group_ids.empty()) continue;

        // Log which trials and groups are being searched
        std::ostringstream trials_str, groups_str;
        for (size_t i = 0; i < trial_indices.size(); ++i) {
            if (i > 0) trials_str << ", ";
            trials_str << rom_configs[trial_indices[i]].name;
        }
        for (size_t i = 0; i < group_names_found.size(); ++i) {
            if (i > 0) groups_str << ", ";
            groups_str << group_names_found[i];
        }
        LOG_INFO("[Contracture] Grid search: trials=[" << trials_str.str()
                 << "] -> groups=[" << groups_str.str() << "]");

        // Run joint grid search
        auto best_ratios = runJointGridSearch(
            character, trial_indices, group_ids, pose_data, base_lm_contract, config);

        // Apply results to x
        for (const auto& [gid, ratio] : best_ratios) {
            x[gid] = ratio;
            initialized_groups.insert(gid);
        }
    }
}

std::map<int, double> ContractureOptimizer::runJointGridSearch(
    Character* character,
    const std::vector<size_t>& trial_indices,
    const std::vector<int>& group_ids,
    const std::vector<PoseData>& pose_data,
    const std::map<int, double>& base_lm_contract,
    const Config& config)
{
    std::map<int, double> result;
    if (group_ids.empty() || trial_indices.empty()) {
        return result;
    }

    auto& muscles = character->getMuscles();
    const size_t num_groups = group_ids.size();

    // Store original lm_contract values
    std::vector<double> original_lm_contract(muscles.size());
    for (size_t i = 0; i < muscles.size(); ++i) {
        original_lm_contract[i] = muscles[i]->lm_contract;
    }

    // Build grid values
    std::vector<double> grid_values;
    for (double r = config.gridSearchBegin; r <= config.gridSearchEnd + 1e-6; r += config.gridSearchInterval) {
        grid_values.push_back(r);
    }
    const size_t grid_size = grid_values.size();

    // Initialize best tracking
    double best_total_error = std::numeric_limits<double>::max();
    std::vector<double> best_ratios(num_groups, 1.0);

    // N-dimensional grid search (nested loops for up to 3 groups)
    // For simplicity, support 1, 2, or 3 groups
    auto evaluateRatios = [&](const std::vector<double>& ratios) -> double {
        // Apply ratios to all groups
        for (size_t g = 0; g < num_groups; ++g) {
            int gid = group_ids[g];
            auto group_it = mMuscleGroups.find(gid);
            if (group_it == mMuscleGroups.end()) continue;

            for (int m_idx : group_it->second) {
                auto it = base_lm_contract.find(m_idx);
                double base = (it != base_lm_contract.end()) ? it->second : muscles[m_idx]->lm_contract;
                muscles[m_idx]->lm_contract = base * ratios[g];
                muscles[m_idx]->RefreshMuscleParams();
            }
        }

        // Compute total squared error across all trials
        double total_error = 0.0;
        for (size_t ti : trial_indices) {
            if (ti >= pose_data.size()) continue;
            const auto& pose = pose_data[ti];

            // Set pose and update muscle geometry
            setPoseAndUpdateGeometry(character, pose.q);

            // Compute passive torque (use specific DOF for symmetry)
            int dof_offset = pose.use_composite_axis ? 1 : pose.joint_dof;
            bool use_global_y = pose.use_composite_axis;
            double computed_torque = computePassiveTorque(character, pose.joint_idx, false, dof_offset, use_global_y);

            double error = computed_torque - pose.tau_obs;
            total_error += error * error;
        }

        return total_error;
    };

    // Grid search based on number of groups
    if (num_groups == 1) {
        // Capture 1D grid search results
        GridSearch1DResult gs_result;
        auto name_it = mGroupNames.find(group_ids[0]);
        gs_result.group_name = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(group_ids[0]);

        int best_idx = 0;
        for (size_t i = 0; i < grid_values.size(); ++i) {
            double r0 = grid_values[i];
            std::vector<double> ratios = {r0};
            double error = evaluateRatios(ratios);
            gs_result.ratios.push_back(r0);
            gs_result.errors.push_back(error);
            if (error < best_total_error) {
                best_total_error = error;
                best_ratios = ratios;
                best_idx = static_cast<int>(i);
            }
        }
        gs_result.best_idx = best_idx;
        gs_result.best_ratio = best_ratios[0];
        gs_result.best_error = best_total_error;
        mGridSearch1DResults.push_back(gs_result);

        // Print 1D error vector if verbose
        if (config.verbose) {
            std::ostringstream oss;
            oss << "[Contracture] Error vector (" << gs_result.group_name << "): ";
            oss << std::fixed << std::setprecision(1);
            for (size_t i = 0; i < gs_result.ratios.size(); ++i) {
                if (i > 0) oss << " ";
                oss << std::setprecision(2) << gs_result.ratios[i] << "(";
                if (gs_result.errors[i] > 1000.0) {
                    oss << ">1000";
                } else {
                    oss << std::setprecision(1) << gs_result.errors[i];
                }
                oss << ")";
                if (static_cast<int>(i) == gs_result.best_idx) oss << "*";
            }
            LOG_INFO(oss.str());
        }
    } else if (num_groups == 2) {
        // Store error matrix for verbose output
        std::vector<std::vector<double>> error_matrix(grid_size, std::vector<double>(grid_size));
        for (size_t i0 = 0; i0 < grid_size; ++i0) {
            for (size_t i1 = 0; i1 < grid_size; ++i1) {
                std::vector<double> ratios = {grid_values[i0], grid_values[i1]};
                double error = evaluateRatios(ratios);
                error_matrix[i0][i1] = error;
                if (error < best_total_error) {
                    best_total_error = error;
                    best_ratios = ratios;
                }
            }
        }
        // Print error matrix if verbose
        if (config.verbose) {
            auto name0_it = mGroupNames.find(group_ids[0]);
            auto name1_it = mGroupNames.find(group_ids[1]);
            std::string gname0 = (name0_it != mGroupNames.end()) ? name0_it->second : "Group_" + std::to_string(group_ids[0]);
            std::string gname1 = (name1_it != mGroupNames.end()) ? name1_it->second : "Group_" + std::to_string(group_ids[1]);

            // Find best indices
            size_t best_i0 = 0, best_i1 = 0;
            for (size_t i0 = 0; i0 < grid_size; ++i0) {
                for (size_t i1 = 0; i1 < grid_size; ++i1) {
                    if (std::abs(grid_values[i0] - best_ratios[0]) < 1e-6 &&
                        std::abs(grid_values[i1] - best_ratios[1]) < 1e-6) {
                        best_i0 = i0;
                        best_i1 = i1;
                    }
                }
            }

            std::ostringstream mat;
            mat << "[Contracture] Error matrix (" << gname0 << " x " << gname1 << "):\n";
            mat << std::fixed << std::setprecision(2);
            // Header row (group1 ratios)
            mat << "         ";
            for (double r1 : grid_values) {
                mat << std::setw(8) << r1;
            }
            mat << "\n";
            // Data rows (group0 ratios)
            for (size_t i0 = 0; i0 < grid_size; ++i0) {
                mat << std::setw(8) << grid_values[i0] << " ";
                for (size_t i1 = 0; i1 < grid_size; ++i1) {
                    double err = error_matrix[i0][i1];
                    bool is_best = (i0 == best_i0 && i1 == best_i1);
                    if (err > 1000.0) {
                        mat << std::setw(7) << ">1000" << (is_best ? "*" : " ");
                    } else {
                        mat << std::setw(7) << std::setprecision(2) << err << (is_best ? "*" : " ");
                    }
                }
                mat << "\n";
            }
            LOG_INFO(mat.str());
        }
    } else if (num_groups >= 3) {
        // For 3+ groups, limit to first 3 to avoid combinatorial explosion
        for (double r0 : grid_values) {
            for (double r1 : grid_values) {
                for (double r2 : grid_values) {
                    std::vector<double> ratios = {r0, r1, r2};
                    // Pad with 1.0 for any additional groups
                    for (size_t g = 3; g < num_groups; ++g) {
                        ratios.push_back(1.0);
                    }
                    double error = evaluateRatios(ratios);
                    if (error < best_total_error) {
                        best_total_error = error;
                        best_ratios = ratios;
                    }
                }
            }
        }
    }

    // Restore original lm_contract values
    for (size_t i = 0; i < muscles.size(); ++i) {
        muscles[i]->lm_contract = original_lm_contract[i];
        muscles[i]->RefreshMuscleParams();
    }

    // Build result map
    for (size_t g = 0; g < num_groups; ++g) {
        result[group_ids[g]] = best_ratios[g];
    }

    // Log result
    std::ostringstream oss;
    oss << "[Contracture] Joint grid search: ";
    for (size_t g = 0; g < num_groups; ++g) {
        if (g > 0) oss << ", ";
        auto name_it = mGroupNames.find(group_ids[g]);
        std::string gname = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(group_ids[g]);
        oss << gname << "=" << std::fixed << std::setprecision(2) << best_ratios[g];
    }
    oss << " (error=" << std::fixed << std::setprecision(4) << best_total_error << ")";
    LOG_INFO(oss.str());

    return result;
}

std::map<int, double> ContractureOptimizer::computeBiarticularAverages(
    const std::vector<double>& x,
    const std::vector<Muscle*>& muscles,
    bool verbose) const
{
    auto biarticular = findBiarticularMuscles();  // muscle_idx -> vector of group_ids
    std::map<int, double> averaged_ratios;  // muscle_idx -> averaged ratio

    bool biarticular_header_printed = false;

    for (const auto& [m_idx, group_ids] : biarticular) {
        // Only average "touched" groups (ratio != 1.0)
        double sum = 0.0;
        int touched_count = 0;
        std::vector<std::string> group_details;

        for (int gid : group_ids) {
            auto name_it = mGroupNames.find(gid);
            std::string gname = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(gid);
            std::ostringstream oss;
            oss << gname << "=" << std::fixed << std::setprecision(4) << x[gid];

            // Check if this group was touched (ratio != 1.0)
            if (std::abs(x[gid] - 1.0) > 1e-6) {
                sum += x[gid];
                touched_count++;
                oss << "*";  // Mark touched groups
            }
            group_details.push_back(oss.str());
        }

        // Average only touched values; if none touched, use 1.0
        if (touched_count > 0) {
            averaged_ratios[m_idx] = sum / static_cast<double>(touched_count);
        } else {
            averaged_ratios[m_idx] = 1.0;
        }

        // Only log muscles with 2+ touched groups (true biarticular conflicts)
        if (verbose && touched_count >= 2) {
            if (!biarticular_header_printed) {
                LOG_INFO("[Contracture] Biarticular muscles (averaging 2+ touched groups):");
                biarticular_header_printed = true;
            }
            std::ostringstream details;
            for (size_t i = 0; i < group_details.size(); ++i) {
                if (i > 0) details << ", ";
                details << group_details[i];
            }
            LOG_INFO("  " << muscles[m_idx]->name << ": [" << details.str()
                     << "] -> avg=" << std::fixed << std::setprecision(4) << averaged_ratios[m_idx]);
        }
    }

    return averaged_ratios;
}

void ContractureOptimizer::logParameterTable(
    const std::string& title,
    const std::vector<double>& x,
    const std::vector<ROMTrialConfig>& rom_configs,
    const std::map<int, std::vector<double>>* torque_before,
    const std::map<int, std::vector<double>>* torque_after) const
{
    int num_groups = static_cast<int>(mMuscleGroups.size());

    LOG_INFO("[Contracture] " << title << " parameters:");

    // Find max group name length
    size_t max_gname_len = 12;
    for (int g = 0; g < num_groups; ++g) {
        auto name_it = mGroupNames.find(g);
        if (name_it != mGroupNames.end()) {
            max_gname_len = std::max(max_gname_len, name_it->second.size());
        }
    }

    // Build trial name list if torque matrices provided
    std::vector<std::string> trial_names;
    if (torque_before && torque_after) {
        for (size_t t = 0; t < rom_configs.size(); ++t) {
            trial_names.push_back(rom_configs[t].name);
        }
    }

    // Print header
    {
        std::ostringstream hdr;
        hdr << std::left << std::setw(max_gname_len) << "group" << " | ratio  ";
        for (const auto& tname : trial_names) {
            hdr << " | " << std::setw(std::max((size_t)11, tname.size())) << tname;
        }
        LOG_INFO(hdr.str());
        LOG_INFO(std::string(hdr.str().size(), '-'));
    }

    // Print each group row
    for (int g = 0; g < num_groups; ++g) {
        auto name_it = mGroupNames.find(g);
        std::string gname = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(g);

        std::ostringstream row;
        row << std::left << std::setw(max_gname_len) << gname << " | "
            << std::fixed << std::setprecision(4) << std::setw(6) << x[g];

        // Add torque columns if matrices provided
        if (torque_before && torque_after) {
            for (size_t t = 0; t < trial_names.size(); ++t) {
                auto it_bf = torque_before->find(g);
                auto it_af = torque_after->find(g);
                double bf = (it_bf != torque_before->end() && t < it_bf->second.size()) ? it_bf->second[t] : 0.0;
                double af = (it_af != torque_after->end() && t < it_af->second.size()) ? it_af->second[t] : 0.0;

                if (std::abs(bf) > 0.01 || std::abs(af) > 0.01) {
                    std::ostringstream cell;
                    cell << std::fixed << std::setprecision(1) << bf << "→" << af;
                    row << " | " << std::setw(std::max((size_t)11, trial_names[t].size())) << cell.str();
                } else {
                    row << " | " << std::setw(std::max((size_t)11, trial_names[t].size())) << "-";
                }
            }
        }
        LOG_INFO(row.str());
    }
}

std::map<int, size_t> ContractureOptimizer::buildGroupToTrialMapping(
    const std::vector<ROMTrialConfig>& rom_configs,
    const std::vector<PoseData>& pose_data) const
{
    std::map<int, size_t> group_to_trial;

    // Build trial name to index lookup
    std::map<std::string, size_t> trial_name_to_index;
    for (size_t i = 0; i < rom_configs.size(); ++i) {
        trial_name_to_index[rom_configs[i].name] = i;
    }

    // First pass: use centralized grid search mapping
    for (const auto& entry : mGridSearchMapping) {
        // Find the first matching trial
        size_t trial_idx = std::numeric_limits<size_t>::max();
        for (const auto& trial_name : entry.trials) {
            auto it = trial_name_to_index.find(trial_name);
            if (it != trial_name_to_index.end()) {
                trial_idx = it->second;
                break;
            }
        }
        if (trial_idx == std::numeric_limits<size_t>::max()) continue;

        // Map all groups in this entry to this trial
        for (const auto& group_name : entry.groups) {
            int gid = findGroupIdByName(group_name);
            if (gid >= 0) {
                group_to_trial[gid] = trial_idx;
            }
        }
    }

    // Second pass: infer from joint name for groups without mapping
    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        if (group_to_trial.count(group_id)) continue;

        auto name_it = mGroupNames.find(group_id);
        if (name_it == mGroupNames.end()) continue;
        const std::string& group_name = name_it->second;

        for (size_t i = 0; i < rom_configs.size() && i < pose_data.size(); ++i) {
            if (groupMatchesJoint(group_name, rom_configs[i].joint)) {
                group_to_trial[group_id] = i;
                break;
            }
        }
    }

    return group_to_trial;
}

void ContractureOptimizer::logInitialParameterTable(const std::vector<double>& x) const
{
    int num_groups = static_cast<int>(mMuscleGroups.size());

    LOG_INFO("[Contracture] Initial parameters (after grid search):");

    // Collect values by base name (strip _l/_r suffix)
    std::map<std::string, std::pair<double, double>> init_table;  // base_name -> (R, L)
    for (int g = 0; g < num_groups; ++g) {
        auto name_it = mGroupNames.find(g);
        std::string name = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(g);

        std::string base_name = name;
        bool is_left = false, is_right = false;
        if (name.size() > 2 && name.substr(name.size() - 2) == "_l") {
            base_name = name.substr(0, name.size() - 2);
            is_left = true;
        } else if (name.size() > 2 && name.substr(name.size() - 2) == "_r") {
            base_name = name.substr(0, name.size() - 2);
            is_right = true;
        }

        if (init_table.find(base_name) == init_table.end()) {
            init_table[base_name] = {1.0, 1.0};  // default R=1, L=1
        }
        if (is_right) init_table[base_name].first = x[g];
        else if (is_left) init_table[base_name].second = x[g];
        else init_table[base_name] = {x[g], x[g]};  // no suffix, same for both
    }

    // Find max base_name length for alignment
    size_t max_len = 10;
    for (const auto& [base, _] : init_table) {
        max_len = std::max(max_len, base.size());
    }

    // Print header
    std::ostringstream header;
    header << std::left << std::setw(max_len) << "group" << " |    R    |    L";
    LOG_INFO(header.str());
    LOG_INFO(std::string(max_len + 20, '-'));

    // Print rows
    for (const auto& [base, vals] : init_table) {
        std::ostringstream row;
        row << std::left << std::setw(max_len) << base << " | "
            << std::fixed << std::setprecision(4) << std::setw(7) << vals.first << " | "
            << std::fixed << std::setprecision(4) << std::setw(7) << vals.second;
        LOG_INFO(row.str());
    }
}

ContractureOptimizer::PreOptimizationData ContractureOptimizer::preOptimization(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs,
    const Config& config)
{
    PreOptimizationData preOpt;

    // Clear previous grid search results
    mGridSearch1DResults.clear();

    if (!character || mMuscleGroups.empty()) return preOpt;

    const auto& muscles = character->getMuscles();
    int num_groups = static_cast<int>(mMuscleGroups.size());

    // ============================================================
    // 1. Capture BEFORE state (lm_contract per muscle in all groups)
    // ============================================================
    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        for (int m_idx : muscle_ids) {
            preOpt.lm_contract_before[m_idx] = muscles[m_idx]->lm_contract;
            preOpt.all_muscle_indices.insert(m_idx);
        }
    }

    // ============================================================
    // 2. Build pose data (constant across iterations)
    // ============================================================
    preOpt.pose_data = buildPoseData(character, rom_configs);

    // ============================================================
    // 3. Capture BEFORE passive torques per trial
    // ============================================================
    for (size_t t = 0; t < rom_configs.size() && t < preOpt.pose_data.size(); ++t) {
        const auto& rom_config = rom_configs[t];
        const auto& pose = preOpt.pose_data[t];

        TrialTorqueResult trial_result;
        trial_result.trial_name = rom_config.name;
        trial_result.joint = rom_config.joint;
        trial_result.dof_index = rom_config.dof_index;
        trial_result.observed_torque = rom_config.torque_cutoff;
        trial_result.pose = pose.q;

        // Set pose and update muscle geometry
        setPoseAndUpdateGeometry(character, pose.q);

        // Compute BEFORE total passive torque
        if (config.verbose) {
            LOG_INFO("[Contracture] Trial '" << rom_config.name << "' BEFORE torque:");
        }
        {
            int dof_offset = pose.use_composite_axis ? 1 : pose.joint_dof;
            bool use_global_y = pose.use_composite_axis;
            trial_result.computed_torque_before = computePassiveTorque(character, pose.joint_idx, config.verbose, dof_offset, use_global_y);
        }

        // Per-muscle contribution BEFORE
        captureMuscleTorqueContributions(character, pose, preOpt.all_muscle_indices,
                                         trial_result.muscle_torques_before,
                                         trial_result.muscle_forces_before);

        preOpt.trial_results_before.push_back(trial_result);
    }

    // ============================================================
    // 4. Build group-to-trial mapping (constant across iterations)
    // ============================================================
    preOpt.group_to_trial = buildGroupToTrialMapping(rom_configs, preOpt.pose_data);

    // ============================================================
    // 5. Initialize x and run grid search
    // ============================================================
    preOpt.initial_x.resize(num_groups, 1.0);
    runGridSearchInitialization(character, rom_configs, preOpt.pose_data,
                                preOpt.lm_contract_before, config, mGridSearchMapping, preOpt.initial_x);

    // ============================================================
    // 6. Log initial parameters
    // ============================================================
    if (config.verbose) {
        logInitialParameterTable(preOpt.initial_x);
    }

    return preOpt;
}

std::vector<MuscleGroupResult> ContractureOptimizer::optimizeWithPrecomputed(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs,
    const Config& config,
    const PreOptimizationData& preOpt,
    int outer_iteration)
{
    std::vector<MuscleGroupResult> results;

    if (!character || mMuscleGroups.empty() || preOpt.pose_data.empty()) {
        return results;
    }

    const auto& muscles = character->getMuscles();
    int num_groups = static_cast<int>(mMuscleGroups.size());

    // Capture current lm_contract as base (may differ from initial after previous iterations)
    std::map<int, double> base_lm_contract;
    for (size_t i = 0; i < muscles.size(); ++i) {
        base_lm_contract[static_cast<int>(i)] = muscles[i]->lm_contract;
    }

    // Initialize parameters:
    // - First iteration (outer_iteration=0): use grid search results
    // - Subsequent iterations: start at 1.0 (base already has previous results applied)
    std::vector<double> x;
    if (outer_iteration == 0) {
        x = preOpt.initial_x;
    } else {
        x.resize(num_groups, 1.0);
    }

    // Build torque matrix BEFORE optimization: torque_matrix_before[group_id][trial_idx]
    std::map<int, std::vector<double>> torque_matrix_before;
    for (int g = 0; g < num_groups; ++g) {
        torque_matrix_before[g].resize(preOpt.pose_data.size());
        for (size_t t = 0; t < preOpt.pose_data.size(); ++t) {
            torque_matrix_before[g][t] = computeGroupTorqueAtTrial(character, preOpt.pose_data, g, t);
        }
    }

    // Build Ceres problem
    ceres::Problem problem;

    // Add torque residual blocks
    for (const auto& pose : preOpt.pose_data) {
        auto* cost = new TorqueResidual(character, pose, mMuscleGroups, base_lm_contract, num_groups);
        problem.AddResidualBlock(cost, nullptr, x.data());
    }

    // Ratio regularization: penalize deviation from ratio=1.0
    if (config.lambdaRatioReg > 0.0) {
        double sqrt_lambda = std::sqrt(config.lambdaRatioReg);
        for (int g = 0; g < num_groups; ++g) {
            auto* reg_cost = new ceres::DynamicNumericDiffCostFunction<RatioRegCost>(
                new RatioRegCost(g, sqrt_lambda));
            reg_cost->AddParameterBlock(num_groups);
            reg_cost->SetNumResiduals(1);
            problem.AddResidualBlock(reg_cost, nullptr, x.data());
        }
    }

    // Torque regularization: penalize passive torque magnitude per group/trial
    if (config.lambdaTorqueReg > 0.0) {
        double sqrt_lambda = std::sqrt(config.lambdaTorqueReg);
        for (int g = 0; g < num_groups; ++g) {
            auto grp_it = mMuscleGroups.find(g);
            if (grp_it == mMuscleGroups.end()) continue;

            for (size_t t = 0; t < preOpt.pose_data.size(); ++t) {
                auto* reg_cost = new ceres::DynamicNumericDiffCostFunction<TorqueRegCost>(
                    new TorqueRegCost(character, preOpt.pose_data[t], g,
                                      grp_it->second, base_lm_contract, sqrt_lambda));
                reg_cost->AddParameterBlock(num_groups);
                reg_cost->SetNumResiduals(1);
                problem.AddResidualBlock(reg_cost, nullptr, x.data());
            }
        }
    }

    // Set bounds
    for (int g = 0; g < num_groups; ++g) {
        problem.SetParameterLowerBound(x.data(), g, config.minRatio);
        problem.SetParameterUpperBound(x.data(), g, config.maxRatio);
    }

    // Solver options
    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    options.linear_solver_type = (num_groups < 20)
        ? ceres::DENSE_QR
        : ceres::SPARSE_NORMAL_CHOLESKY;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-8;
    options.minimizer_progress_to_stdout = false;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Apply optimized ratios to muscles
    for (int g = 0; g < num_groups; ++g) {
        auto grp_it = mMuscleGroups.find(g);
        if (grp_it == mMuscleGroups.end()) continue;

        for (int m_idx : grp_it->second) {
            muscles[m_idx]->lm_contract = base_lm_contract[m_idx] * x[g];
            muscles[m_idx]->RefreshMuscleParams();
        }
    }

    // Build torque matrix AFTER optimization
    std::map<int, std::vector<double>> torque_matrix_after;
    for (int g = 0; g < num_groups; ++g) {
        torque_matrix_after[g].resize(preOpt.pose_data.size());
        for (size_t t = 0; t < preOpt.pose_data.size(); ++t) {
            torque_matrix_after[g][t] = computeGroupTorqueAtTrial(character, preOpt.pose_data, g, t);
        }
    }

    // Compute averaged ratios for biarticular muscles
    std::map<int, double> averaged_ratios = computeBiarticularAverages(x, muscles, config.verbose);

    // Log final parameters if verbose
    if (config.verbose) {
        logParameterTable("Final (after optimization)", x, rom_configs,
                          &torque_matrix_before, &torque_matrix_after);
        LOG_INFO("[Contracture] Iterations: " << summary.iterations.size()
                    << ", Final cost: " << summary.final_cost);
    }

    // Build results
    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        MuscleGroupResult result;

        auto name_it = mGroupNames.find(group_id);
        result.group_name = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(group_id);
        result.ratio = x[group_id];

        for (int m_idx : muscle_ids) {
            result.muscle_names.push_back(muscles[m_idx]->name);
            // Use averaged ratio for biarticular muscles
            double effective_ratio = (averaged_ratios.count(m_idx))
                ? averaged_ratios[m_idx]
                : x[group_id];
            result.lm_contract_values.push_back(base_lm_contract[m_idx] * effective_ratio);
        }

        results.push_back(result);
    }

    return results;
}

} // namespace PMuscle

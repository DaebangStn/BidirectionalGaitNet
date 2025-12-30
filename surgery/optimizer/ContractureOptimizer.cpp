// Contracture Optimizer - Implementation
#include "ContractureOptimizer.h"
#include "rm/rm.hpp"
#include "Log.h"
#include <yaml-cpp/yaml.h>
#include <ceres/ceres.h>
#include <regex>
#include <iostream>
#include <cmath>
#include <set>

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

        for (const auto& [group_id, muscle_ids] : muscle_groups_) {
            double ratio = parameters[0][group_id];
            for (int m_idx : muscle_ids) {
                old_lm_contract[m_idx] = muscles[m_idx]->lm_contract;

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
        double tau_pred = computePassiveTorque(character_, pose_.joint_idx);

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
                double tau_plus = computePassiveTorque(character_, pose_.joint_idx);

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

            YAML::Node angles = joint_node.second;
            int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));

            if (angles.IsSequence()) {
                for (size_t i = 0; i < angles.size() && i < joint->getNumDofs(); ++i) {
                    positions[first_dof + i] = angles[i].as<double>() * M_PI / 180.0;
                }
            } else {
                positions[first_dof] = angles.as<double>() * M_PI / 180.0;
            }
        }
        config.pose = positions;
    }

    // Load target joint (simplified format - no angle_sweep section)
    config.joint = node["joint"].as<std::string>("");
    config.dof_index = node["dof_index"].as<int>(0);

    // Load single torque value
    config.torque = node["torque"].as<double>(15.0);

    // ROM angle defaults to 0 - populated later from clinical data or manual input
    config.rom_angle = 0.0;

    // Load clinical_data reference
    if (node["clinical_data"]) {
        config.cd_side = node["clinical_data"]["side"].as<std::string>("");
        config.cd_joint = node["clinical_data"]["joint"].as<std::string>("");
        config.cd_field = node["clinical_data"]["field"].as<std::string>("");
    }

    // Load uniform_search_group for grid search initialization
    config.uniform_search_group = node["uniform_search_group"].as<std::string>("");

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


double ContractureOptimizer::computePassiveTorque(Character* character, int joint_idx, bool verbose) {
    if (!character) return 0.0;
    if (character->getMuscles().empty()) return 0.0;

    auto skel = character->getSkeleton();
    auto joint = skel->getJoint(joint_idx);
    if (!joint) return 0.0;

    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int num_dofs = static_cast<int>(joint->getNumDofs());

    double total_torque = 0.0;
    const auto& muscles = character->getMuscles();
    int contributing_muscles = 0;

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
                contributing_muscles++;
            }
        }
    }

    if (verbose) {
        LOG_INFO("  Joint " << joint->getName() << " (idx=" << joint_idx
                 << ", DOFs " << first_dof << "-" << (first_dof + num_dofs - 1) << "): "
                 << contributing_muscles << " muscle contributions, total=" << total_torque << " Nm");
    }

    return total_torque;
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
        int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
        double angle_rad = config.rom_angle * M_PI / 180.0;
        positions[first_dof + config.dof_index] = angle_rad;

        // Record single pose data point
        PoseData point;
        point.joint_idx = joint_idx;
        point.joint_dof = config.dof_index;
        point.q = positions;
        point.tau_obs = config.torque;
        point.weight = 1.0;
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

        // Set skeleton pose
        skeleton->setPositions(pose.q);

        // Update muscle geometry
        for (auto& muscle : muscles) {
            muscle->UpdateGeometry();
        }

        // Compute passive torque
        double computed_torque = computePassiveTorque(character, pose.joint_idx);
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

    // Grid search for trials with uniform_search_group specified
    for (size_t i = 0; i < rom_configs.size() && i < pose_data.size(); ++i) {
        const auto& rom_config = rom_configs[i];
        if (rom_config.uniform_search_group.empty()) continue;

        // Find group_id for this uniform_search_group name
        int target_group_id = findGroupIdByName(rom_config.uniform_search_group);
        if (target_group_id < 0) {
            LOG_INFO("[Contracture] Warning: uniform_search_group '"
                     << rom_config.uniform_search_group << "' not found in muscle groups");
            continue;
        }

        // Find best initial ratio via grid search
        double best_ratio = findBestInitialRatio(
            character, pose_data[i], target_group_id, base_lm_contract, config);

        x[target_group_id] = best_ratio;
        LOG_INFO("[Contracture] Grid search for " << rom_config.uniform_search_group
                 << ": best_ratio=" << best_ratio);
    }

    // Build Ceres problem
    ceres::Problem problem;

    // Add residual blocks
    for (const auto& pose : pose_data) {
        auto* cost = new TorqueResidual(character, pose, mMuscleGroups, base_lm_contract, num_groups);
        problem.AddResidualBlock(cost, nullptr, x.data());
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

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (config.verbose) {
        LOG_INFO(summary.FullReport());
    } else {
        LOG_INFO("[ContractureOptimizer] " << summary.BriefReport());
    }

    // Build results
    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        MuscleGroupResult result;

        // Get group name from member map
        auto name_it = mGroupNames.find(group_id);
        result.group_name = (name_it != mGroupNames.end()) ? name_it->second : "Group_" + std::to_string(group_id);
        result.ratio = x[group_id];

        for (int m_idx : muscle_ids) {
            result.muscle_names.push_back(muscles[m_idx]->name);
            double new_lm_contract = base_lm_contract[m_idx] * x[group_id];
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
    // 1. Capture BEFORE state (lm_contract per muscle in all groups)
    // ============================================================
    std::map<int, double> lm_contract_before;
    std::set<int> all_muscle_indices;

    for (const auto& [group_id, muscle_ids] : mMuscleGroups) {
        for (int m_idx : muscle_ids) {
            lm_contract_before[m_idx] = muscles[m_idx]->lm_contract;
            all_muscle_indices.insert(m_idx);
        }
    }

    // ============================================================
    // 2. Capture BEFORE passive torques per trial
    // ============================================================
    std::vector<PoseData> pose_data = buildPoseData(character, rom_configs);

    for (size_t t = 0; t < rom_configs.size() && t < pose_data.size(); ++t) {
        const auto& rom_config = rom_configs[t];
        const auto& pose = pose_data[t];

        TrialTorqueResult trial_result;
        trial_result.trial_name = rom_config.name;
        trial_result.joint = rom_config.joint;
        trial_result.dof_index = rom_config.dof_index;
        trial_result.observed_torque = rom_config.torque;
        trial_result.pose = pose.q;

        // Set pose
        skeleton->setPositions(pose.q);
        for (auto& m : muscles) {
            m->UpdateGeometry();
        }

        // Compute BEFORE total passive torque
        if (config.verbose) {
            LOG_INFO("[Contracture] Trial '" << rom_config.name << "' BEFORE torque:");
        }
        trial_result.computed_torque_before = computePassiveTorque(character, pose.joint_idx, config.verbose);

        // Per-muscle contribution BEFORE
        auto* joint = skeleton->getJoint(pose.joint_idx);
        int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));

        for (int m_idx : all_muscle_indices) {
            auto* muscle = muscles[m_idx];
            Eigen::VectorXd jtp = muscle->GetRelatedJtp();
            const auto& related_indices = muscle->related_dof_indices;

            double contrib = 0.0;
            for (size_t i = 0; i < related_indices.size(); ++i) {
                if (related_indices[i] == first_dof + pose.joint_dof) {
                    contrib = jtp[i];
                    break;
                }
            }
            trial_result.muscle_torques_before.push_back({muscle->name, contrib});
            // Capture passive force (f_p * f0)
            double passive_force = muscle->Getf_p() * muscle->f0;
            trial_result.muscle_forces_before.push_back({muscle->name, passive_force});
        }

        result.trial_results.push_back(trial_result);
    }

    // ============================================================
    // 3. Run optimization
    // ============================================================
    result.group_results = optimize(character, rom_configs, config);

    if (result.group_results.empty()) {
        std::cerr << "[ContractureOptimizer] Optimization failed" << std::endl;
        return result;
    }

    // Apply results to character
    applyResults(character, result.group_results);

    // ============================================================
    // 4. Capture AFTER state (lm_contract per muscle)
    // ============================================================
    for (int m_idx : all_muscle_indices) {
        MuscleContractureResult m_result;
        m_result.muscle_name = muscles[m_idx]->name;
        m_result.muscle_idx = m_idx;
        m_result.lm_contract_before = lm_contract_before[m_idx];
        m_result.lm_contract_after = muscles[m_idx]->lm_contract;
        m_result.ratio = m_result.lm_contract_after / m_result.lm_contract_before;
        result.muscle_results.push_back(m_result);
    }

    // ============================================================
    // 5. Capture AFTER passive torques per trial
    // ============================================================
    for (size_t t = 0; t < result.trial_results.size() && t < pose_data.size(); ++t) {
        auto& trial_result = result.trial_results[t];
        const auto& pose = pose_data[t];

        // Set pose
        skeleton->setPositions(trial_result.pose);
        for (auto& m : muscles) {
            m->UpdateGeometry();
        }

        // Compute AFTER total passive torque
        if (config.verbose) {
            LOG_INFO("[Contracture] Trial '" << trial_result.trial_name << "' AFTER torque:");
        }
        trial_result.computed_torque_after = computePassiveTorque(character, pose.joint_idx, config.verbose);

        // Per-muscle contribution AFTER
        auto* joint = skeleton->getJoint(pose.joint_idx);
        int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));

        for (int m_idx : all_muscle_indices) {
            auto* muscle = muscles[m_idx];
            Eigen::VectorXd jtp = muscle->GetRelatedJtp();
            const auto& related_indices = muscle->related_dof_indices;

            double contrib = 0.0;
            for (size_t i = 0; i < related_indices.size(); ++i) {
                if (related_indices[i] == first_dof + pose.joint_dof) {
                    contrib = jtp[i];
                    break;
                }
            }
            trial_result.muscle_torques_after.push_back({muscle->name, contrib});
            // Capture passive force (f_p * f0)
            double passive_force = muscle->Getf_p() * muscle->f0;
            trial_result.muscle_forces_after.push_back({muscle->name, passive_force});
        }
    }

    result.converged = true;
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


std::vector<MuscleGroupResult> ContractureOptimizer::optimizeIterative(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs,
    const IterativeConfig& config) {

    std::vector<MuscleGroupResult> results;

    if (!character) {
        std::cerr << "[ContractureOptimizer] No character provided" << std::endl;
        return results;
    }

    if (mMuscleGroups.empty()) {
        std::cerr << "[ContractureOptimizer] No muscle groups configured" << std::endl;
        return results;
    }

    // Find biarticular muscles
    auto biarticular = findBiarticularMuscles();
    std::cout << "[ContractureOptimizer] Found " << biarticular.size()
              << " biarticular muscles" << std::endl;

    // Store original lm_contract values for restoration
    const auto& muscles = character->getMuscles();
    std::map<int, double> original_lm_contract;
    for (size_t i = 0; i < muscles.size(); ++i) {
        original_lm_contract[static_cast<int>(i)] = muscles[i]->lm_contract;
    }

    // Current lm_contract baseline (gets updated after each iteration)
    std::map<int, double> current_lm_contract = original_lm_contract;

    double max_ratio_change = std::numeric_limits<double>::max();

    for (int iter = 0; iter < config.maxOuterIterations; ++iter) {
        std::cout << "[ContractureOptimizer] Iteration " << (iter + 1)
                  << "/" << config.maxOuterIterations << std::endl;

        // Apply current baseline lm_contract values
        for (const auto& [m_idx, lm] : current_lm_contract) {
            muscles[m_idx]->lm_contract = lm;
            muscles[m_idx]->RefreshMuscleParams();
        }

        // Run single optimization
        results = optimize(character, rom_configs, config.baseConfig);

        if (results.empty()) {
            std::cerr << "[ContractureOptimizer] Optimization failed at iteration "
                      << (iter + 1) << std::endl;
            break;
        }

        // For biarticular muscles: average ratios across groups
        max_ratio_change = 0.0;
        std::map<int, double> muscle_avg_ratio;

        for (const auto& [m_idx, group_ids] : biarticular) {
            double sum_ratio = 0.0;
            for (int gid : group_ids) {
                sum_ratio += results[gid].ratio;
            }
            double avg_ratio = sum_ratio / static_cast<double>(group_ids.size());
            muscle_avg_ratio[m_idx] = avg_ratio;

            // Track max change for convergence
            double old_lm = current_lm_contract[m_idx];
            double new_lm = original_lm_contract[m_idx] * avg_ratio;
            double ratio_change = std::abs(new_lm / old_lm - 1.0);
            max_ratio_change = std::max(max_ratio_change, ratio_change);
        }

        std::cout << "[ContractureOptimizer] Max ratio change: " << max_ratio_change << std::endl;

        // Update baseline for biarticular muscles
        for (const auto& [m_idx, avg_ratio] : muscle_avg_ratio) {
            current_lm_contract[m_idx] = original_lm_contract[m_idx] * avg_ratio;
        }

        // Also update non-biarticular muscles with their group ratio
        for (const auto& result : results) {
            for (size_t i = 0; i < result.muscle_names.size(); ++i) {
                // Find muscle index by name
                for (size_t m = 0; m < muscles.size(); ++m) {
                    if (muscles[m]->name == result.muscle_names[i]) {
                        int m_idx = static_cast<int>(m);
                        // Only update if not biarticular (biarticular already updated)
                        if (biarticular.find(m_idx) == biarticular.end()) {
                            current_lm_contract[m_idx] = result.lm_contract_values[i];
                        }
                        break;
                    }
                }
            }
        }

        // Check convergence
        if (max_ratio_change < config.convergenceThreshold) {
            std::cout << "[ContractureOptimizer] Converged at iteration " << (iter + 1)
                      << " (change " << max_ratio_change << " < "
                      << config.convergenceThreshold << ")" << std::endl;
            break;
        }
    }

    // Update final results with averaged lm_contract values for biarticular muscles
    for (auto& result : results) {
        for (size_t i = 0; i < result.muscle_names.size(); ++i) {
            // Find muscle index
            for (size_t m = 0; m < muscles.size(); ++m) {
                if (muscles[m]->name == result.muscle_names[i]) {
                    int m_idx = static_cast<int>(m);
                    result.lm_contract_values[i] = current_lm_contract[m_idx];
                    break;
                }
            }
        }
    }

    // Apply final lm_contract values
    for (const auto& [m_idx, lm] : current_lm_contract) {
        muscles[m_idx]->lm_contract = lm;
        muscles[m_idx]->RefreshMuscleParams();
    }

    std::cout << "[ContractureOptimizer] Iterative optimization complete" << std::endl;

    return results;
}

} // namespace PMuscle

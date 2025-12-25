// Contracture Optimizer - Implementation
#include "ContractureOptimizer.h"
#include "rm/rm.hpp"
#include <yaml-cpp/yaml.h>
#include <ceres/ceres.h>
#include <regex>
#include <iostream>
#include <cmath>

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


ROMTrialConfig ContractureOptimizer::loadROMConfig(const std::string& yaml_path) {
    ROMTrialConfig config;

    std::string resolved = rm::resolve(yaml_path);
    YAML::Node node = YAML::LoadFile(resolved);

    config.name = node["name"].as<std::string>("");
    config.description = node["description"].as<std::string>("");

    // Load pose preset
    if (node["pose"]) {
        for (const auto& joint : node["pose"]) {
            std::string joint_name = joint.first.as<std::string>();
            YAML::Node angles = joint.second;

            Eigen::VectorXd angle_vec;
            if (angles.IsSequence()) {
                angle_vec.resize(angles.size());
                for (size_t i = 0; i < angles.size(); ++i) {
                    angle_vec[i] = angles[i].as<double>() * M_PI / 180.0;  // Convert to radians
                }
            } else {
                angle_vec.resize(1);
                angle_vec[0] = angles.as<double>() * M_PI / 180.0;
            }
            config.pose[joint_name] = angle_vec;
        }
    }

    // Load angle sweep
    if (node["angle_sweep"]) {
        config.sweep_joint = node["angle_sweep"]["joint"].as<std::string>("");
        config.sweep_dof_index = node["angle_sweep"]["dof_index"].as<int>(0);
        config.angle_min = node["angle_sweep"]["angle_min"].as<double>(0.0);
        config.angle_max = node["angle_sweep"]["angle_max"].as<double>(90.0);
        config.num_steps = node["angle_sweep"]["num_steps"].as<int>(10);
    }

    // Load observed torques
    if (node["observed_torques"]) {
        for (const auto& obs : node["observed_torques"]) {
            ObservedTorque ot;
            ot.angle = obs["angle"].as<double>(0.0);
            ot.torque = obs["torque"].as<double>(0.0);
            config.observed_torques.push_back(ot);
        }
    }

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


void ContractureOptimizer::applyPosePreset(
    dart::dynamics::SkeletonPtr skeleton,
    const std::map<std::string, Eigen::VectorXd>& pose) {

    for (const auto& [joint_name, angles] : pose) {
        auto* joint = skeleton->getJoint(joint_name);
        if (!joint) continue;

        int num_dofs = static_cast<int>(joint->getNumDofs());
        Eigen::VectorXd positions = joint->getPositions();

        for (int i = 0; i < std::min(num_dofs, static_cast<int>(angles.size())); ++i) {
            positions[i] = angles[i];
        }

        joint->setPositions(positions);
    }
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


double ContractureOptimizer::computePassiveTorque(Character* character, int joint_idx) {
    if (!character) return 0.0;
    if (character->getMuscles().empty()) return 0.0;

    auto skel = character->getSkeleton();
    auto joint = skel->getJoint(joint_idx);
    if (!joint) return 0.0;

    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int num_dofs = static_cast<int>(joint->getNumDofs());

    double total_torque = 0.0;
    const auto& muscles = character->getMuscles();

    for (auto& muscle : muscles) {
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        const auto& related_indices = muscle->related_dof_indices;

        for (size_t i = 0; i < related_indices.size(); ++i) {
            int global_dof = related_indices[i];
            if (global_dof >= first_dof && global_dof < first_dof + num_dofs) {
                total_torque += jtp[i];
            }
        }
    }

    return total_torque;
}


std::vector<PoseData> ContractureOptimizer::buildPoseData(
    Character* character,
    const std::vector<ROMTrialConfig>& rom_configs) {

    std::vector<PoseData> data;
    auto skeleton = character->getSkeleton();

    for (const auto& config : rom_configs) {
        // Apply base pose
        applyPosePreset(skeleton, config.pose);

        int joint_idx = getJointIndex(skeleton, config.sweep_joint);
        if (joint_idx < 0) {
            std::cerr << "[ContractureOptimizer] Joint not found: " << config.sweep_joint << std::endl;
            continue;
        }

        auto* joint = skeleton->getJoint(joint_idx);
        if (!joint || config.sweep_dof_index >= static_cast<int>(joint->getNumDofs())) {
            std::cerr << "[ContractureOptimizer] Invalid DOF index for joint: " << config.sweep_joint << std::endl;
            continue;
        }

        // For each observed torque point
        for (const auto& obs : config.observed_torques) {
            // Set swept joint angle
            double angle_rad = obs.angle * M_PI / 180.0;
            Eigen::VectorXd joint_pos = joint->getPositions();
            joint_pos[config.sweep_dof_index] = angle_rad;
            joint->setPositions(joint_pos);

            // Record full pose and observed torque
            PoseData point;
            point.joint_idx = joint_idx;
            point.joint_dof = config.sweep_dof_index;
            point.q = skeleton->getPositions();
            point.tau_obs = obs.torque;
            point.weight = 1.0;
            data.push_back(point);
        }
    }

    return data;
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

    // Build Ceres problem
    ceres::Problem problem;

    // Add residual blocks
    for (const auto& pose : pose_data) {
        auto* cost = new TorqueResidual(character, pose, mMuscleGroups, base_lm_contract, num_groups);

        ceres::LossFunction* loss = config.useRobustLoss
            ? new ceres::HuberLoss(1.0)
            : nullptr;

        problem.AddResidualBlock(cost, loss, x.data());
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
    options.minimizer_progress_to_stdout = config.verbose;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-8;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (config.verbose) {
        std::cout << summary.FullReport() << std::endl;
    } else {
        std::cout << "[ContractureOptimizer] " << summary.BriefReport() << std::endl;
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

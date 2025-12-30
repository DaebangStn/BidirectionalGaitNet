// Contracture Optimizer - Ceres-based muscle contracture estimation
// Optimizes lm_contract parameters based on ROM examination data
#ifndef CONTRACTURE_OPTIMIZER_H
#define CONTRACTURE_OPTIMIZER_H

#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <dart/dynamics/dynamics.hpp>
#include "Character.h"

namespace PMuscle {

/**
 * @brief ROM trial configuration loaded from YAML
 *
 * Represents a single ROM measurement at a specific pose and joint angle.
 * The rom_angle is typically populated from clinical data or manual input.
 */
struct ROMTrialConfig {
    std::string name;
    std::string description;

    // Base pose as full skeleton positions (converted from YAML on load)
    Eigen::VectorXd pose;

    // Target joint for ROM measurement
    std::string joint;
    int dof_index = 0;

    // Single ROM measurement point
    double rom_angle = 0.0;    // ROM angle in degrees (from clinical data or manual)
    double torque = 15.0;      // Observed passive torque at ROM limit (Nm)

    // Clinical data reference (for patient data linking)
    std::string cd_side;       // "left" or "right"
    std::string cd_joint;      // "hip", "knee", "ankle"
    std::string cd_field;      // field name in patient rom.yaml

    // Grid search initialization
    std::string uniform_search_group;  // Target muscle group for grid search (e.g., "plantarflexor_l")
};

/**
 * @brief Pose data point for optimization
 */
struct PoseData {
    int joint_idx;               // Global DOF index of swept joint
    int joint_dof;               // DOF index within joint
    Eigen::VectorXd q;           // Full skeleton pose
    double tau_obs;              // Observed passive torque (Nm)
    double weight;               // Measurement weight
};

/**
 * @brief Result for a muscle group
 */
struct MuscleGroupResult {
    std::string group_name;
    std::vector<std::string> muscle_names;
    double ratio;                // Optimized lm_contract ratio
    std::vector<double> lm_contract_values;  // Per-muscle lm_contract
};

/**
 * @brief Per-muscle optimization result with before/after comparison
 */
struct MuscleContractureResult {
    std::string muscle_name;
    int muscle_idx;
    double lm_contract_before;
    double lm_contract_after;
    double ratio;  // after / before
};

/**
 * @brief Per-trial passive torque result
 */
struct TrialTorqueResult {
    std::string trial_name;
    std::string joint;
    int dof_index;
    Eigen::VectorXd pose;            // Full skeleton pose at ROM angle
    double observed_torque;           // Target from clinical data
    double computed_torque_before;    // Simulated before optimization
    double computed_torque_after;     // Simulated after optimization
    // Per-muscle contribution at this trial pose
    std::vector<std::pair<std::string, double>> muscle_torques_before;
    std::vector<std::pair<std::string, double>> muscle_torques_after;
    // Per-muscle passive force (f_p * f0) at this trial pose
    std::vector<std::pair<std::string, double>> muscle_forces_before;
    std::vector<std::pair<std::string, double>> muscle_forces_after;
};

/**
 * @brief Comprehensive contracture optimization result
 */
struct ContractureOptResult {
    std::vector<MuscleGroupResult> group_results;
    std::vector<MuscleContractureResult> muscle_results;
    std::vector<TrialTorqueResult> trial_results;
    int iterations = 0;
    double final_cost = 0.0;
    bool converged = false;
};

/**
 * @brief Ceres-based contracture optimizer for muscle lm_contract parameters
 *
 * Optimizes per-group lm_contract scaling ratios to match observed passive torques
 * from ROM examinations. Uses Ceres Solver for nonlinear optimization.
 *
 * Muscle grouping: Configured via YAML or auto-detected from naming convention
 */
class ContractureOptimizer {
public:
    /**
     * @brief Configuration for single-pass optimization
     */
    struct Config {
        int maxIterations;
        double minRatio;
        double maxRatio;
        bool verbose;

        // Grid search initialization (applied when uniform_search_group is specified)
        double gridSearchBegin = 0.7;
        double gridSearchEnd = 1.3;
        double gridSearchInterval = 0.1;

        Config() : maxIterations(100), minRatio(0.7), maxRatio(1.3),
                   verbose(false) {}
    };

    /**
     * @brief Configuration for iterative optimization with biarticular averaging
     */
    struct IterativeConfig {
        Config baseConfig;
        int maxOuterIterations;      // Number of averaging iterations
        double convergenceThreshold; // Stop if max ratio change < this

        IterativeConfig() : maxOuterIterations(3), convergenceThreshold(0.01) {}
    };

    ContractureOptimizer() = default;
    ~ContractureOptimizer() = default;

    // ========== Group Configuration ==========

    /**
     * @brief Load muscle groups from YAML config file
     *
     * YAML format:
     *   group_name:
     *     - muscle_name_1
     *     - muscle_name_2
     *
     * @param yaml_path Path to muscle groups YAML file
     * @param character Character with muscles (for name->index mapping)
     * @return Number of groups loaded
     */
    int loadMuscleGroups(const std::string& yaml_path, Character* character);

    /**
     * @brief Get current muscle groups
     */
    const std::map<int, std::vector<int>>& getMuscleGroups() const { return mMuscleGroups; }

    /**
     * @brief Get current group names
     */
    const std::map<int, std::string>& getGroupNames() const { return mGroupNames; }

    /**
     * @brief Check if groups are configured
     */
    bool hasGroups() const { return !mMuscleGroups.empty(); }

    // ========== ROM Configuration ==========

    /**
     * @brief Load ROM trial configuration from YAML file
     *
     * Parses YAML and converts pose map to full skeleton positions.
     * The rom_angle is left at 0.0 and should be populated later from
     * clinical data or manual input.
     *
     * @param yaml_path Path to ROM trial YAML file
     * @param skeleton Skeleton for resolving pose to full positions (optional)
     * @return ROMTrialConfig Loaded configuration
     */
    static ROMTrialConfig loadROMConfig(
        const std::string& yaml_path,
        dart::dynamics::SkeletonPtr skeleton = nullptr);

    // ========== Optimization ==========

    /**
     * @brief Run contracture optimization using stored muscle groups
     *
     * Must call loadMuscleGroups() or detectMuscleGroups() first.
     *
     * @param character Character with muscles to optimize
     * @param rom_configs ROM trial configurations
     * @param config Optimization configuration
     * @return Vector of results per muscle group
     */
    std::vector<MuscleGroupResult> optimize(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const Config& config = Config()
    );

    /**
     * @brief Run contracture optimization with comprehensive before/after results
     *
     * Captures lm_contract and passive torque values before and after optimization
     * for visualization and analysis.
     *
     * @param character Character with muscles to optimize
     * @param rom_configs ROM trial configurations
     * @param config Optimization configuration
     * @return Comprehensive result with before/after comparison
     */
    ContractureOptResult optimizeWithResults(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const Config& config = Config()
    );

    /**
     * @brief Apply optimized ratios to character muscles
     *
     * @param character Character to modify
     * @param results Optimization results from optimize()
     */
    static void applyResults(
        Character* character,
        const std::vector<MuscleGroupResult>& results
    );

    /**
     * @brief Compute passive torque at a joint for the character
     *
     * @param character Character with muscles
     * @param joint_idx Joint index
     * @return Total passive torque from all muscles
     */
    static double computePassiveTorque(Character* character, int joint_idx, bool verbose = false);

    /**
     * @brief Build pose data from ROM configs for optimization
     *
     * @param character Character to set poses on
     * @param rom_configs ROM trial configurations
     * @return Vector of pose data points
     */
    std::vector<PoseData> buildPoseData(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs
    );

    // ========== Biarticular Muscle Handling ==========

    /**
     * @brief Find muscles that appear in multiple groups (biarticular muscles)
     *
     * @return Map of muscle_idx -> list of group_ids containing it
     */
    std::map<int, std::vector<int>> findBiarticularMuscles() const;

    /**
     * @brief Run iterative optimization with biarticular muscle averaging
     *
     * For biarticular muscles (appearing in multiple groups), averages the
     * optimized ratios across all groups containing them, then re-optimizes.
     * Converges when max ratio change < convergenceThreshold.
     *
     * @param character Character with muscles to optimize
     * @param rom_configs ROM trial configurations
     * @param config Iterative optimization configuration
     * @return Vector of results per muscle group (from final iteration)
     */
    std::vector<MuscleGroupResult> optimizeIterative(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const IterativeConfig& config = IterativeConfig()
    );

private:
    // Get joint index by name
    static int getJointIndex(
        dart::dynamics::SkeletonPtr skeleton,
        const std::string& joint_name
    );

    // Find group ID by name
    int findGroupIdByName(const std::string& name) const;

    // Grid search to find best initial ratio for a muscle group
    double findBestInitialRatio(
        Character* character,
        const PoseData& pose,
        int group_id,
        const std::map<int, double>& base_lm_contract,
        const Config& config
    );

    // Ceres cost functor
    struct TorqueResidual;

    // Member variables
    std::map<int, std::vector<int>> mMuscleGroups;  // group_id -> muscle indices
    std::map<int, std::string> mGroupNames;          // group_id -> group name
};

} // namespace PMuscle

#endif // CONTRACTURE_OPTIMIZER_H

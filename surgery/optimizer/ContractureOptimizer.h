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
 * @brief Observed torque data point from ROM examination
 */
struct ObservedTorque {
    double angle;    // Joint angle in degrees
    double torque;   // Observed passive torque in Nm
};

/**
 * @brief ROM trial configuration loaded from YAML
 */
struct ROMTrialConfig {
    std::string name;
    std::string description;

    // Base pose (joint_name -> angles vector)
    std::map<std::string, Eigen::VectorXd> pose;

    // Angle sweep parameters
    std::string sweep_joint;
    int sweep_dof_index;
    double angle_min;
    double angle_max;
    int num_steps;

    // Observed torques from clinical measurement
    std::vector<ObservedTorque> observed_torques;
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
        bool useRobustLoss;
        bool verbose;

        Config() : maxIterations(100), minRatio(0.7), maxRatio(1.3),
                   useRobustLoss(true), verbose(false) {}
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
     * @param yaml_path Path to ROM trial YAML file
     * @return ROMTrialConfig Loaded configuration
     */
    static ROMTrialConfig loadROMConfig(const std::string& yaml_path);

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
    static double computePassiveTorque(Character* character, int joint_idx);

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
    // Apply pose preset from ROM config
    void applyPosePreset(
        dart::dynamics::SkeletonPtr skeleton,
        const std::map<std::string, Eigen::VectorXd>& pose
    );

    // Get joint index by name
    static int getJointIndex(
        dart::dynamics::SkeletonPtr skeleton,
        const std::string& joint_name
    );

    // Ceres cost functor
    struct TorqueResidual;

    // Member variables
    std::map<int, std::vector<int>> mMuscleGroups;  // group_id -> muscle indices
    std::map<int, std::string> mGroupNames;          // group_id -> group name
};

} // namespace PMuscle

#endif // CONTRACTURE_OPTIMIZER_H

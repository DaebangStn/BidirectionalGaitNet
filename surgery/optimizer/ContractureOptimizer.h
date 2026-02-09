// Contracture Optimizer - Ceres-based muscle contracture estimation
// Optimizes lm_contract parameters based on ROM examination data
#ifndef CONTRACTURE_OPTIMIZER_H
#define CONTRACTURE_OPTIMIZER_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
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
    int dof_index = 0;              // Used when dof is integer (0, 1, 2)
    std::string dof_type;           // Used when dof is string (e.g., "abd_knee")
    bool is_composite_dof = false;  // True if dof_type is set

    // Single ROM measurement point
    double rom_angle = 0.0;    // ROM angle in degrees (from clinical data or manual)
    double torque_cutoff = 15.0;  // Cutoff torque for determining ROM limit (Nm)

    // Exam sweep parameters (for PhysicalExam angle sweeps)
    double angle_min = -90.0;  // Sweep min angle in degrees
    double angle_max = 90.0;   // Sweep max angle in degrees
    int num_steps = 100;       // Number of sweep steps
    double angle_step = 1.0;   // Step size for composite DOF (degrees)

    // Clinical data reference (for patient data linking)
    std::string cd_side;       // "left" or "right"
    std::string cd_joint;      // "hip", "knee", "ankle"
    std::string cd_field;      // field name in patient rom.yaml
    bool cd_neg = false;       // negate the angle from clinical data
    double cd_cutoff = -1.0;   // skip if |rom_angle| > cutoff (-1 means no cutoff)

    // Note: uniform_search_group removed - now using centralized GridSearchMapping

    // IK parameters for composite DOF
    double shank_scale = 0.7;  // Scale factor for shank length in abd_knee IK (default 0.7)
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

    // Composite DOF fields
    bool use_composite_axis = false;   // If true, use composite_axis instead of joint_dof
    Eigen::Vector3d composite_axis;    // Normalized axis for torque projection (joint-local frame)
};

/**
 * @brief Result of IK computation for abd_knee composite DOF
 *
 * Contains hip joint positions (axis-angle) and knee angle that produce
 * the desired abduction angle while keeping shank vertical.
 */
struct AbdKneePoseResult {
    Eigen::Vector3d hip_positions;  // Axis-angle for hip BallJoint
    double knee_angle;              // Radians for knee RevoluteJoint
    bool success;
};

/**
 * @brief Grid search mapping entry for joint multi-group optimization
 *
 * Maps a list of ROM trials to a list of muscle groups that should be
 * searched jointly. Enables N-dimensional grid search over multiple groups.
 */
struct GridSearchMapping {
    std::vector<std::string> trials;   // ROM trial names (e.g., ["dorsi_k0_R", "dorsi_k90_R"])
    std::vector<std::string> groups;   // Muscle group names to search jointly
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
 * @brief 1D grid search result for a single group
 */
struct GridSearch1DResult {
    std::string group_name;
    std::vector<std::string> trial_names;  // Trials used for this search
    std::vector<double> ratios;            // Grid values tested
    std::vector<double> errors;            // Error at each ratio
    int best_idx = -1;                     // Index of best ratio
    double best_ratio = 1.0;
    double best_error = 0.0;
};


/**
 * @brief Simple grid search result for a single muscle group
 *
 * Used by search() for standalone CLI-based optimization.
 */
struct SeedSearchResult {
    std::string group_name;
    double best_ratio = 1.0;
    double best_error = 0.0;
    // Per-trial torque contributions (for display)
    std::vector<std::string> trial_names;      // Trial names (for column headers)
    std::vector<double> torques_before;        // One per trial
    std::vector<double> torques_after;         // One per trial
};

/**
 * @brief Result for a search group (coarse, grid search level)
 *
 * Represents the grid search result for a coarse search group that may
 * contain multiple finer optimization groups.
 */
struct SearchGroupResult {
    std::string search_group_name;               // e.g., "hip_abductor_l"
    std::vector<std::string> opt_group_names;    // Child groups, e.g., ["gluteus_medius_l", ...]
    double ratio;                                // Best grid search ratio
    double best_error;                           // Squared error at best ratio
    std::vector<double> ratios;                  // All tested ratios
    std::vector<double> errors;                  // Error at each ratio
    int best_idx = -1;                           // Index of best ratio
};

/**
 * @brief Contracture optimization result
 *
 * Contains search group (coarse grid search) and optimization group (Ceres) results.
 */
struct ContractureOptResult {
    std::vector<SearchGroupResult> search_group_results;   // Grid search (coarse)
    std::vector<MuscleGroupResult> group_results;          // Ceres optimization (fine)
    std::vector<MuscleContractureResult> muscle_results;
    std::vector<TrialTorqueResult> trial_results;
    std::vector<GridSearch1DResult> grid_search_1d_results;  // 1D grid search results
    int iterations = 0;
    double final_cost = 0.0;
    bool converged = false;
    std::string param_name;  // "lt_rel" or "lm_contract"
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
    // Which muscle parameter to optimize
    enum class OptParam { LM_CONTRACT, LT_REL };

    struct Config {
        int maxIterations;
        double minRatio;
        double maxRatio;
        bool verbose;

        // Grid search initialization parameters
        double gridSearchBegin = 0.7;
        double gridSearchEnd = 1.3;
        double gridSearchInterval = 0.1;

        // Regularization
        double lambdaRatioReg = 0.0;   // Penalize (ratio - 1.0)^2 for each group
        double lambdaTorqueReg = 0.0;  // Penalize passive torque magnitude per group/trial
        double lambdaLineReg = 0.0;    // Penalize ratio variance among fibers of same muscle

        // Outer iterations for biarticular convergence
        int outerIterations = 1;       // Number of outer iterations (1 = single pass)

        // Which parameter to optimize (lm_contract or lt_rel)
        OptParam paramType = OptParam::LM_CONTRACT;

        // Progress callback: (iteration, cost) called after each Ceres iteration
        std::function<void(int iteration, double cost)> iterationCallback;

        Config() : maxIterations(100), minRatio(0.7), maxRatio(1.3),
                   verbose(false), lambdaRatioReg(0.0), lambdaTorqueReg(0.0),
                   lambdaLineReg(0.0), outerIterations(1),
                   paramType(OptParam::LM_CONTRACT) {}
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

    // Helpers for parameterized muscle property access (public for cost functors)
    static double getParam(Muscle* m, OptParam type);
    static void setParam(Muscle* m, OptParam type, double value);

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

    /**
     * @brief Check if tiered grouping (search + optimization) is configured
     *
     * @return true if search_groups and optimization_groups were loaded
     */
    bool hasTieredGroups() const { return !mSearchToOptGroups.empty(); }

    /**
     * @brief Get optimization group names
     */
    const std::map<int, std::string>& getOptGroupNames() const { return mOptGroupNames; }

    /**
     * @brief Get search group names
     */
    const std::map<int, std::string>& getSearchGroupNames() const { return mSearchGroupNames; }

    /**
     * @brief Get search to optimization group mapping
     */
    const std::map<int, std::vector<int>>& getSearchToOptGroups() const { return mSearchToOptGroups; }

    /**
     * @brief Get captured search group results (from last optimization)
     */
    const std::vector<SearchGroupResult>& getSearchGroupResults() const { return mSearchGroupResults; }

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

    // ========== Grid Search Mapping ==========

    /**
     * @brief Set grid search mapping for joint multi-group optimization
     *
     * @param mapping Vector of trial-to-groups mappings
     */
    void setGridSearchMapping(const std::vector<GridSearchMapping>& mapping) {
        mGridSearchMapping = mapping;
    }

    /**
     * @brief Get current grid search mapping
     */
    const std::vector<GridSearchMapping>& getGridSearchMapping() const {
        return mGridSearchMapping;
    }

    // ========== Optimization ==========

    /**
     * @brief Run contracture optimization (grid search → Ceres)
     *
     * Uses search groups for grid search initialization, then optimization groups
     * for Ceres fine-tuning. Falls back to flat optimization if tiered grouping
     * is not configured.
     *
     * @param character Character with muscles to optimize
     * @param rom_configs ROM trial configurations
     * @param config Optimization configuration
     * @return Optimization result with group ratios and torque comparisons
     */
    ContractureOptResult optimize(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const Config& config = Config()
    );

    /**
     * @brief Simple joint grid search across specified groups and all trials
     *
     * Performs N-dimensional grid search over all specified groups jointly,
     * minimizing total squared error across all trials. This is a standalone
     * method for CLI usage without the full optimization pipeline.
     *
     * @param character Character with muscles
     * @param rom_configs All ROM trials to evaluate
     * @param group_names List of muscle group names to search (from optimization_groups)
     * @param config Search configuration (gridSearchBegin/End/Interval)
     * @return Vector of SeedSearchResult (one per group)
     */
    std::vector<SeedSearchResult> seedSearch(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const std::vector<std::string>& group_names,
        const Config& config = Config()
    );

    /**
     * @brief Compute passive torque at a joint for the character
     *
     * @param character Character with muscles
     * @param joint_idx Joint index
     * @param verbose Print per-muscle contributions
     * @param dof_offset DOF index within joint (-1 = sum all DOFs, >= 0 = specific DOF)
     * @param use_global_y If true and dof_offset >= 0, use global Y-axis projection (for abd_knee)
     *                     If false and dof_offset >= 0, filter to only the specified DOF
     * @return Total passive torque from all muscles
     */
    static double computePassiveTorque(Character* character, int joint_idx, bool verbose = false,
                                       int dof_offset = -1, bool use_global_y = false);

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

public:
    // ========== Composite DOF Helpers ==========

    // Compute composite axis for abd_knee DOF type
    // Axis is perpendicular to plane containing world Y and hip-knee vector
    static Eigen::Vector3d computeAbdKneeAxis(
        dart::dynamics::SkeletonPtr skeleton,
        int hip_joint_idx);

    // Compute IK pose for abd_knee composite DOF
    // Given abduction angle, computes hip axis-angle and knee angle
    // that produce the pose with shank vertical
    static AbdKneePoseResult computeAbdKneePose(
        dart::dynamics::SkeletonPtr skeleton,
        int hip_joint_idx,
        double rom_angle_deg,
        bool is_left_leg,
        double shank_scale = 0.7);

    // Compute knee angle that makes shank point vertical (+Y)
    // after hip is set to given axis-angle positions
    static double computeKneeAngleForVerticalShank(
        dart::dynamics::SkeletonPtr skeleton,
        int hip_joint_idx,
        const Eigen::Vector3d& hip_positions);

private:
    // ========== Refactored Helper Methods ==========

    // Check if group name matches joint name pattern (side and joint type)
    bool groupMatchesJoint(const std::string& group_name, const std::string& joint_name) const;

    // Set skeleton pose and update all muscle geometry
    void setPoseAndUpdateGeometry(Character* character, const Eigen::VectorXd& q) const;

    // Compute group passive torque at a specific trial's joint
    double computeGroupTorqueSettingPose(
        Character* character,
        const std::vector<PoseData>& pose_data,
        int group_id,
        size_t trial_idx) const;

    // Compute group passive torque assuming pose is already set and geometry updated
    double computeGroupTorque(
        Character* character,
        const PoseData& pose,
        int group_id) const;

    // Log parameter table (initial or final) with optional torque matrices
    void logParameterTable(
        const std::string& title,
        const std::vector<double>& x,
        const std::vector<ROMTrialConfig>& rom_configs,
        const std::map<int, std::vector<double>>* torque_before = nullptr,
        const std::map<int, std::vector<double>>* torque_after = nullptr) const;

    // Compute averaged ratios for biarticular muscles
    std::map<int, double> computeBiarticularAverages(
        const std::vector<double>& x,
        const std::vector<Muscle*>& muscles,
        bool verbose) const;

    // Capture per-muscle torque and force contributions at a pose
    void captureMuscleTorques(
        Character* character,
        const PoseData& pose,
        const std::set<int>& muscle_indices,
        std::vector<std::pair<std::string, double>>& out_torques,
        std::vector<std::pair<std::string, double>>& out_forces) const;

    // Compute cumulative ratio per group from initial and final lm_contract
    void computeCumulativeGroupRatios(
        const std::vector<Muscle*>& muscles,
        const std::map<int, double>& lm_contract_before,
        std::vector<MuscleGroupResult>& group_results) const;

    // ========== Tiered Optimization Helpers ==========

    // Run grid search on search groups (coarse level)
    // Returns vector of SearchGroupResult with error curves per search group
    std::vector<SearchGroupResult> runGridSearchOnSearchGroups(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const std::vector<PoseData>& pose_data,
        const std::map<int, double>& base_lm_contract,
        const Config& config);

    // Initialize optimization group ratios from search group results
    // Each opt group inherits its parent search group's best ratio
    std::vector<double> initOptRatiosFromSearch(
        const std::vector<SearchGroupResult>& search_results);

    // Run Ceres optimization on optimization groups (fine level)
    // Returns vector of MuscleGroupResult per opt group
    std::vector<MuscleGroupResult> runCeresOnOptGroups(
        Character* character,
        const std::vector<ROMTrialConfig>& rom_configs,
        const std::vector<PoseData>& pose_data,
        const std::map<int, double>& base_lm_contract,
        const std::vector<double>& initial_x,
        const Config& config);

    // Find search group ID by name
    int findSearchGroupIdByName(const std::string& name) const;

    // Build fiber groups: maps base muscle name -> list of opt group indices
    // Groups fibers like vastus_intermedius0_r, vastus_intermedius1_r by base name
    std::map<std::string, std::vector<int>> buildFiberGroups() const;

    // Log initial parameters in R/L table format
    void logInitialParameterTable(const std::vector<double>& x) const;

    // Member variables
    std::map<int, std::vector<int>> mMuscleGroups;  // group_id -> muscle indices
    std::map<int, std::string> mGroupNames;          // group_id -> group name
    std::vector<GridSearchMapping> mGridSearchMapping;  // trial-to-groups mapping for grid search
    std::vector<GridSearch1DResult> mGridSearch1DResults;  // Captured 1D grid search results

    // ========== Tiered Grouping (dual-tier: search + optimization) ==========
    // Populated if search_groups and optimization_groups sections exist in YAML

    // Search groups (coarse, for grid search)
    std::map<int, std::vector<int>> mSearchGroups;      // search_id -> muscle indices (all muscles in search group)
    std::map<int, std::string> mSearchGroupNames;       // search_id -> search group name
    std::map<int, std::vector<int>> mSearchToOptGroups; // search_id -> [opt_ids] (children)

    // Optimization groups (fine, for Ceres)
    std::map<int, std::vector<int>> mOptGroups;         // opt_id -> muscle indices
    std::map<int, std::string> mOptGroupNames;          // opt_id -> opt group name
    std::map<int, int> mOptToSearchGroup;               // opt_id -> search_id (parent)
    std::map<std::string, int> mOptNameToId;            // opt_name -> opt_id (lookup helper)

    // Captured search group results (for UI visualization)
    std::vector<SearchGroupResult> mSearchGroupResults;
};

} // namespace PMuscle

#endif // CONTRACTURE_OPTIMIZER_H

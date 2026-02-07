// Waypoint Optimizer - Ceres-based muscle waypoint optimization
// Migrated from SkelGen's MuscleGenerator.cpp
#ifndef WAYPOINT_OPTIMIZER_H
#define WAYPOINT_OPTIMIZER_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include <dart/dynamics/dynamics.hpp>
#include "Character.h"  // From sim/
#include "HDF.h"        // From sim/

namespace PMuscle {

/**
 * @brief Type of muscle length to use for curve fitting
 */
enum class LengthCurveType {
    MTU_LENGTH,    // lmt - raw muscle-tendon unit length
    NORMALIZED     // lm_norm - normalized muscle fiber length
};

/**
 * @brief Muscle length curve characteristics for optimization
 */
struct LengthCurveCharacteristics {
    double min_phase;      // Phase where muscle is shortest (0.0 to 1.0)
    double max_phase;      // Phase where muscle is longest (0.0 to 1.0)
    double delta;          // Range: max_length - min_length
    double min_length;     // Actual minimum length
    double max_length;     // Actual maximum length
    std::vector<double> phase_samples;  // Length values at evenly-spaced phases
};

/**
 * @brief Result of waypoint optimization for a single muscle
 */
struct WaypointOptResult {
    std::string muscle_name;
    bool success;

    // DOF sweep info (for display)
    std::string dof_name;
    int dof_idx = -1;

    // Reference character's curve (target to match)
    std::vector<double> reference_lengths;
    LengthCurveCharacteristics reference_chars;

    // Subject character's curve BEFORE optimization
    std::vector<double> subject_before_lengths;
    LengthCurveCharacteristics subject_before_chars;

    // Subject character's curve AFTER optimization
    std::vector<double> subject_after_lengths;
    LengthCurveCharacteristics subject_after_chars;

    // Phase data (x-axis for plotting)
    std::vector<double> phases;  // 0.0 to 1.0

    // Per-phase shape metrics
    std::vector<double> shape_angle_before;   // Direction misalignment in degrees (for visualization)
    std::vector<double> shape_angle_after;    // Direction misalignment in degrees (for visualization)
    std::vector<double> shape_energy_before;  // (1-dot)² values per phase (actual optimization metric)
    std::vector<double> shape_energy_after;   // (1-dot)² values per phase (actual optimization metric)

    // Energy tracking
    double initial_shape_energy = 0.0;
    double initial_length_energy = 0.0;
    double initial_total_cost = 0.0;
    double final_shape_energy = 0.0;
    double final_length_energy = 0.0;
    double final_total_cost = 0.0;

    // Optimized waypoint positions (for deferred sync in parallel execution)
    // anchor_index -> local_position for each bodynode in anchor
    std::vector<std::vector<Eigen::Vector3d>> optimized_anchor_positions;

    // Which length type was used (for plot labeling)
    LengthCurveType length_type = LengthCurveType::MTU_LENGTH;

    // Bound hit tracking: number of waypoints that hit displacement bounds
    int num_bound_hits = 0;

    // Optimization progress: number of iterations used
    int num_iterations = 0;
};

/**
 * @brief Ceres-based waypoint optimizer for muscle routing
 *
 * Optimizes muscle waypoint positions to preserve:
 * 1. Force directions across motions (fShape energy)
 * 2. Length-angle curve characteristics (fLengthCurve energy)
 *
 * Based on SkelGen's retargetSingleMusclesWaypointsCalibrating()
 * Uses Ceres Solver instead of manual gradient descent
 */
class WaypointOptimizer {
public:
    /**
     * @brief Configuration for waypoint optimization
     */
    struct Config {
        int maxIterations;
        int numSampling;
        double lambdaShape;
        double lambdaLengthCurve;
        bool fixOriginInsertion;
        bool verbose;
        double weightPhase;       // weight for phase matching in length curve energy
        double weightDelta;       // weight for delta matching in length curve energy
        double weightSamples;     // weight for sample matching loss
        int numPhaseSamples;      // number of phase sample points (e.g., 3 → 0, 0.5, 1.0)
        int lossPower;            // Power exponent (2=squared, 3=cube, etc.)

        LengthCurveType lengthType;  // Which length metric to use for curve fitting
        int numParallel;             // Number of parallel threads (1 = sequential)
        double maxDisplacement;               // Max displacement for normal waypoints (m)
        double maxDisplacementOriginInsertion; // Max displacement for origin/insertion (m)

        // Solver tolerances
        double functionTolerance;    // Convergence tolerance on cost function change
        double gradientTolerance;    // Convergence tolerance on gradient norm
        double parameterTolerance;   // Convergence tolerance on parameter change

        bool adaptiveSampleWeight;   // Use adaptive weighting for sample matching
        bool multiDofJointSweep;     // Sweep all DOFs of best DOF's parent joint for shape energy

        Config() : maxIterations(10000), numSampling(10), lambdaShape(0.1),
                   lambdaLengthCurve(0.1), fixOriginInsertion(true), verbose(false),
                   weightPhase(1.0), weightDelta(50.0),
                   weightSamples(1.0), numPhaseSamples(3), lossPower(2),
                   lengthType(LengthCurveType::MTU_LENGTH), numParallel(1),
                   maxDisplacement(0.2), maxDisplacementOriginInsertion(0.03),
                   functionTolerance(1e-4), gradientTolerance(1e-5), parameterTolerance(1e-5),
                   adaptiveSampleWeight(false), multiDofJointSweep(false) {}
    };

    WaypointOptimizer() = default;
    ~WaypointOptimizer() = default;

    /**
     * @brief Optimize waypoint positions
     *
     * Caller is responsible for saving/restoring skeleton poses.
     *
     * @param subject_muscle Subject's muscle to optimize (will be modified)
     * @param reference_muscle Reference muscle for comparison
     * @param reference_skeleton Reference character's skeleton
     * @param subject_skeleton Subject character's skeleton
     * @param config Optimization configuration
     * @return WaypointOptResult containing reference/subject curves and success status
     */
    WaypointOptResult optimizeMuscle(
        Muscle* subject_muscle,
        Muscle* reference_muscle,
        dart::dynamics::SkeletonPtr reference_skeleton,
        dart::dynamics::SkeletonPtr subject_skeleton,
        const Config& config = Config()
    );

    /**
     * @brief Compute muscle length curve by sweeping most relevant DOF
     *
     * Finds the DOF with largest Jacobian and sweeps it across joint limits.
     * Caller manages pose save/restore.
     *
     * @param muscle Muscle to analyze (uses related_dof_indices, mCachedJs)
     * @param skeleton Skeleton for joint limits and pose
     * @param num_samples Number of samples across DOF range
     * @param length_type Which length metric to use (MTU_LENGTH or NORMALIZED)
     * @return Vector of muscle lengths at sampled DOF positions
     */
    static std::vector<double> computeMuscleLengthCurve(
        Muscle* muscle,
        dart::dynamics::SkeletonPtr skeleton,
        int num_samples = 10,
        LengthCurveType length_type = LengthCurveType::MTU_LENGTH
    );

    /**
     * @brief Analyze muscle length curve characteristics
     *
     * @param lengths Muscle lengths at different phases
     * @param numPhaseSamples Number of evenly-spaced phase samples to extract (0 = none)
     * @return Length curve characteristics (min/max phases, delta, phase_samples)
     */
    static LengthCurveCharacteristics analyzeLengthCurve(
        const std::vector<double>& lengths,
        int numPhaseSamples = 0
    );

    /**
     * @brief Get DOF count from HDF motion file
     *
     * @param hdf_filepath Path to HDF motion file
     * @return DOF count, or -1 on error
     */
    static int getMotionDOF(const std::string& hdf_filepath);

};

} // namespace PMuscle

#endif // WAYPOINT_OPTIMIZER_H

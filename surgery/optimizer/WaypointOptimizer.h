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
 * @brief Muscle length curve characteristics for optimization
 */
struct LengthCurveCharacteristics {
    double min_phase;      // Phase where muscle is shortest (0.0 to 1.0)
    double max_phase;      // Phase where muscle is longest (0.0 to 1.0)
    double delta;          // Range: max_length - min_length
    double min_length;     // Actual minimum length
    double max_length;     // Actual maximum length
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

        Config() : maxIterations(10000), numSampling(10), lambdaShape(0.1),
                   lambdaLengthCurve(0.1), fixOriginInsertion(true), verbose(false) {}
    };

    WaypointOptimizer() = default;
    ~WaypointOptimizer() = default;

    /**
     * @brief Optimize waypoint positions for a single muscle
     *
     * @param muscle Muscle to optimize (will be modified)
     * @param reference_muscle Reference muscle for comparison (standard/ideal behavior)
     * @param hdf_motion_path Path to HDF motion file for sampling
     * @param skeleton Skeleton for pose setting
     * @param config Optimization configuration
     * @return true if optimization converged, false otherwise
     */
    bool optimizeMuscle(
        Muscle* muscle,
        Muscle* reference_muscle,
        const std::string& hdf_motion_path,
        dart::dynamics::SkeletonPtr skeleton,
        const Config& config = Config()
    );

    /**
     * @brief Compute muscle length curve from HDF motion
     *
     * @param muscle Muscle to analyze
     * @param skeleton Skeleton for pose setting
     * @param hdf_motion_path Path to HDF motion file
     * @param num_samples Number of samples across motion
     * @return Vector of muscle lengths at sampled phases
     */
    static std::vector<double> computeMuscleLengthCurveFromHDF(
        Muscle* muscle,
        dart::dynamics::SkeletonPtr skeleton,
        const std::string& hdf_motion_path,
        int num_samples = 10
    );

    /**
     * @brief Analyze muscle length curve characteristics
     *
     * @param lengths Muscle lengths at different phases
     * @return Length curve characteristics (min/max phases, delta)
     */
    static LengthCurveCharacteristics analyzeLengthCurve(
        const std::vector<double>& lengths
    );

private:
    // Ceres cost functors defined in WaypointOptimizer.cpp
    struct ShapeResidual;
    struct LengthCurveResidual;
};

} // namespace PMuscle

#endif // WAYPOINT_OPTIMIZER_H

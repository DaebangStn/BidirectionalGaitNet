// Waypoint Optimizer - Ceres-based implementation
// Migrated from SkelGen/generator/MuscleGenerator.cpp

#include "WaypointOptimizer.h"
#include "Log.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <iostream>

namespace PMuscle {

// ============================================================================
// Helper Functions
// ============================================================================

std::vector<double> WaypointOptimizer::computeMuscleLengthCurveFromHDF(
    Muscle* muscle,
    dart::dynamics::SkeletonPtr skeleton,
    const std::string& hdf_filepath,
    int num_samples)
{
    // Load HDF motion file
    HDF motion(hdf_filepath);

    std::vector<double> muscle_lengths;
    muscle_lengths.reserve(num_samples);

    // Save current skeleton state
    Eigen::VectorXd original_pose = skeleton->getPositions();

    // Sample poses uniformly across the motion
    for (int i = 0; i < num_samples; ++i) {
        // Compute normalized phase (0.0 to 1.0)
        double phase = static_cast<double>(i) / (num_samples - 1);

        // Get pose from HDF at this phase (automatically interpolates)
        Eigen::VectorXd pose = motion.getPose(phase);

        // Set skeleton to this pose
        skeleton->setPositions(pose);

        // Update muscle geometry at this pose
        muscle->UpdateGeometry();

        // Get muscle length
        double l_mt = muscle->lmt;  // Muscle-tendon unit length
        muscle_lengths.push_back(l_mt);
    }

    // Restore original skeleton state
    skeleton->setPositions(original_pose);

    return muscle_lengths;
}

LengthCurveCharacteristics WaypointOptimizer::analyzeLengthCurve(
    const std::vector<double>& lengths)
{
    LengthCurveCharacteristics chars;

    // Find min and max
    auto min_it = std::min_element(lengths.begin(), lengths.end());
    auto max_it = std::max_element(lengths.begin(), lengths.end());

    chars.min_length = *min_it;
    chars.max_length = *max_it;
    chars.delta = chars.max_length - chars.min_length;

    // Compute normalized phases (0.0 to 1.0)
    int min_idx = std::distance(lengths.begin(), min_it);
    int max_idx = std::distance(lengths.begin(), max_it);

    chars.min_phase = static_cast<double>(min_idx) / (lengths.size() - 1);
    chars.max_phase = static_cast<double>(max_idx) / (lengths.size() - 1);

    return chars;
}

// ============================================================================
// Ceres Cost Functors
// ============================================================================

/**
 * @brief Shape residual - preserves force direction
 *
 * Migrated from SkelGen's fShape() function (lines 221-278)
 * Measures deviation from reference force direction
 */
struct WaypointOptimizer::ShapeResidual {
    Eigen::Vector3d ref_direction;  // Reference force direction
    double weight;

    ShapeResidual(const Eigen::Vector3d& ref_dir, double w)
        : ref_direction(ref_dir.normalized()), weight(w) {}

    template <typename T>
    bool operator()(const T* const anchor_pos, T* residuals) const {
        // Compute direction from anchor position
        // NOTE: In full implementation, this would compute direction
        // between consecutive waypoints across multiple motion phases
        //
        // For now, we use simplified residual
        // Full implementation would:
        // 1. Update muscle anchor with anchor_pos
        // 2. Sample muscle across HDF motion
        // 3. Compute force directions at each phase
        // 4. Compare with reference directions

        // Simplified: residual proportional to deviation from reference
        residuals[0] = weight * (T(anchor_pos[0]) - T(ref_direction.x()));
        residuals[1] = weight * (T(anchor_pos[1]) - T(ref_direction.y()));
        residuals[2] = weight * (T(anchor_pos[2]) - T(ref_direction.z()));

        return true;
    }
};

/**
 * @brief Length curve residual - preserves length-angle curve characteristics
 *
 * Migrated from SkelGen's fLengthCurve() function (lines 280-338)
 * Measures deviation of curve characteristics (min/max phases, delta)
 */
struct WaypointOptimizer::LengthCurveResidual {
    LengthCurveCharacteristics ref_chars;
    std::string hdf_filepath;
    int num_samples;
    dart::dynamics::SkeletonPtr skeleton;
    Muscle* muscle;
    double weight;

    LengthCurveResidual(
        const LengthCurveCharacteristics& ref,
        const std::string& motion_path,
        dart::dynamics::SkeletonPtr skel,
        Muscle* mus,
        double w,
        int samples)
        : ref_chars(ref), hdf_filepath(motion_path), num_samples(samples),
          skeleton(skel), muscle(mus), weight(w) {}

    template <typename T>
    bool operator()(const T* const anchor_pos, T* residuals) const {
        // Update anchor position (being optimized by Ceres)
        // NOTE: This is simplified - full implementation would:
        // 1. Update muscle->GetAnchors()[i]->local_positions with anchor_pos
        // 2. Call muscle->SetMuscle() to recompute geometry
        // 3. Compute length curve across HDF motion
        // 4. Analyze characteristics
        // 5. Return difference from reference

        // For now, simplified residual
        residuals[0] = weight * T(0.01);  // Placeholder
        residuals[1] = weight * T(0.01);  // Placeholder
        residuals[2] = weight * T(0.01);  // Placeholder

        return true;
    }
};

// ============================================================================
// Main Optimization Function
// ============================================================================

bool WaypointOptimizer::optimizeMuscle(
    Muscle* muscle,
    Muscle* reference_muscle,
    const std::string& hdf_motion_path,
    dart::dynamics::SkeletonPtr skeleton,
    const Config& config)
{
    LOG_INFO("[WaypointOpt] Starting waypoint optimization for muscle: " << muscle->name);

    // 1. Compute reference muscle behavior
    LOG_INFO("[WaypointOpt] Computing reference muscle length curve");
    std::vector<double> ref_lengths = computeMuscleLengthCurveFromHDF(
        reference_muscle, skeleton, hdf_motion_path, config.numSampling
    );
    LengthCurveCharacteristics ref_chars = analyzeLengthCurve(ref_lengths);

    LOG_INFO("[WaypointOpt] Reference curve: min_phase=" << ref_chars.min_phase
             << ", max_phase=" << ref_chars.max_phase
             << ", delta=" << ref_chars.delta);

    // 2. Setup Ceres problem
    ceres::Problem problem;

    // 3. Create parameter blocks for each waypoint (3D position)
    auto anchors = muscle->GetAnchors();
    std::vector<double*> waypoint_params;

    for (size_t i = 0; i < anchors.size(); ++i) {
        double* pos = new double[3];
        pos[0] = anchors[i]->local_positions[0].x();
        pos[1] = anchors[i]->local_positions[0].y();
        pos[2] = anchors[i]->local_positions[0].z();
        waypoint_params.push_back(pos);

        // 4. Fix origin/insertion if configured
        if (config.fixOriginInsertion && (i == 0 || i == anchors.size() - 1)) {
            problem.SetParameterBlockConstant(waypoint_params[i]);
            LOG_VERBOSE("[WaypointOpt] Fixed anchor " << i << " (origin/insertion)");
        }
    }

    // 5. Add residual blocks for optimization objectives
    for (size_t i = 0; i < waypoint_params.size(); ++i) {
        // Skip if fixed
        if (config.fixOriginInsertion && (i == 0 || i == anchors.size() - 1)) {
            continue;
        }

        // Add shape (force direction) residual
        // NOTE: Simplified - full implementation would compute reference direction
        Eigen::Vector3d ref_direction = Eigen::Vector3d(0, 0, 1);  // Placeholder
        ceres::CostFunction* shape_cost =
            new ceres::AutoDiffCostFunction<ShapeResidual, 3, 3>(
                new ShapeResidual(ref_direction, config.lambdaShape)
            );
        problem.AddResidualBlock(shape_cost, nullptr, waypoint_params[i]);

        // Add length curve residual
        ceres::CostFunction* length_cost =
            new ceres::AutoDiffCostFunction<LengthCurveResidual, 3, 3>(
                new LengthCurveResidual(ref_chars, hdf_motion_path, skeleton,
                                        muscle, config.lambdaLengthCurve, config.numSampling)
            );
        problem.AddResidualBlock(length_cost, nullptr, waypoint_params[i]);
    }

    // 6. Configure Ceres solver
    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = config.verbose;
    options.logging_type = config.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;

    // 7. Solve
    LOG_INFO("[WaypointOpt] Starting Ceres optimization (max " << config.maxIterations << " iterations)");
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 8. Print summary
    if (config.verbose) {
        std::cout << summary.FullReport() << std::endl;
    } else {
        LOG_INFO("[WaypointOpt] Optimization result: " << summary.BriefReport());
    }

    // 9. Update muscle anchors with optimized positions
    for (size_t i = 0; i < waypoint_params.size(); ++i) {
        auto& anchor = muscle->GetAnchors()[i];
        anchor->local_positions[0] << waypoint_params[i][0],
                                       waypoint_params[i][1],
                                       waypoint_params[i][2];
        delete[] waypoint_params[i];
    }

    // 10. Recompute muscle geometry with optimized waypoints
    muscle->SetMuscle();

    bool converged = (summary.termination_type == ceres::CONVERGENCE);
    LOG_INFO("[WaypointOpt] Optimization " << (converged ? "CONVERGED" : "FAILED"));

    return converged;
}

} // namespace PMuscle

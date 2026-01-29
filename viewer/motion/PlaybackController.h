#pragma once

#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include <cmath>

// Forward declarations
class Motion;
class RenderCharacter;
struct PlaybackViewerState;

/**
 * @brief Shared playback logic for motion processing
 *
 * Contains static utility methods extracted from RenderCkpt that are common
 * to both HDF and C3D motion processing. This centralizes the playback
 * computation logic that was previously scattered throughout RenderCkpt.cpp.
 *
 * Methods are static to avoid per-processor instantiation overhead.
 */
class PlaybackController {
public:
    // ==================== Phase Computation ====================

    /**
     * @brief Compute phase from viewer time
     * @param viewerTime Current viewer time in seconds
     * @param cycleDuration Duration of one gait cycle in seconds
     * @return Phase value [0, 1)
     */
    static double computePhase(double viewerTime, double cycleDuration);

    // ==================== Frame Computation ====================

    /**
     * @brief Compute frame float from phase using motion timestamps
     *
     * If motion has timestamps, uses binary search for accurate interpolation.
     * Otherwise, directly maps phase to frame using timesteps per cycle.
     *
     * @param motion Motion to query
     * @param phase Phase value [0, 1)
     * @return Frame float (real-valued frame index)
     */
    static double computeFrameFloat(Motion* motion, double phase);

    /**
     * @brief Wrap frame float to valid range [0, totalFrames)
     * @param frameFloat Frame float to wrap
     * @param totalFrames Total number of frames
     * @return Wrapped frame float
     */
    static double wrapFrameFloat(double frameFloat, int totalFrames);

    /**
     * @brief Compute integer frame index from frame float
     * @param frameFloat Floating-point frame position
     * @param totalFrames Total frames for clamping
     * @return Integer frame index
     */
    static int computeFrameIndex(double frameFloat, int totalFrames);

    /**
     * @brief Determine frame based on navigation mode
     *
     * Returns computed frame for SYNC mode, or manual frame index for MANUAL mode.
     *
     * @param motion Motion to query
     * @param state Playback state with navigation mode
     * @param phase Current phase
     * @return Frame float value
     */
    static double determineMotionFrame(Motion* motion,
                                       PlaybackViewerState& state,
                                       double phase);

    // ==================== Cycle Detection ====================

    /**
     * @brief Detect if cycle wrap occurred
     * @param currentFrameIdx Current frame index
     * @param lastFrameIdx Previous frame index
     * @param totalFrames Total frames in motion
     * @return true if cycle wrapped (frame decreased)
     */
    static bool detectCycleWrap(int currentFrameIdx, int lastFrameIdx, int totalFrames);

    // ==================== Cycle Accumulation ====================

    /**
     * @brief Update cycle accumulation for HDF/BVH motion
     *
     * Detects cycle wrap and adds cycle distance to accumulation.
     * This is the standard approach for skeleton-angle based motions.
     *
     * @param state Playback state (modified)
     * @param currentFrameIdx Current frame index
     */
    static void updateCycleAccumulationStandard(PlaybackViewerState& state,
                                                int currentFrameIdx);

    // ==================== Interpolation ====================

    /**
     * @brief Interpolate between two poses
     *
     * Uses skeleton-aware interpolation for proper rotation blending.
     *
     * @param p1 First pose
     * @param p2 Second pose
     * @param weight Interpolation weight [0, 1]
     * @param character Character for skeleton-aware interpolation
     * @param phaseOverflow true if interpolating across cycle boundary
     * @return Interpolated pose
     */
    static Eigen::VectorXd interpolatePose(const Eigen::VectorXd& p1,
                                           const Eigen::VectorXd& p2,
                                           double weight,
                                           RenderCharacter* character,
                                           bool phaseOverflow);

    /**
     * @brief Extract and interpolate frame from raw motion data
     *
     * Main entry point for pose evaluation. Handles:
     * - Frame extraction from raw motion data
     * - Interpolation between frames
     * - Root position offset application
     *
     * @param motion Motion to evaluate
     * @param frameFloat Frame position (with fractional part)
     * @param character Character for interpolation
     * @param cycleAccumulation Accumulated cycle offset
     * @param displayOffset Display positioning offset
     * @return Interpolated pose with offsets applied
     */
    static Eigen::VectorXd evaluatePose(Motion* motion,
                                        double frameFloat,
                                        RenderCharacter* character,
                                        const Eigen::Vector3d& cycleAccumulation,
                                        const Eigen::Vector3d& displayOffset);

    // ==================== Cycle Distance ====================

    /**
     * @brief Compute cycle distance from motion data
     *
     * Calculates the root translation per gait cycle by comparing
     * first and last frame positions.
     *
     * @param motion Motion to analyze
     * @return Cycle distance vector
     */
    static Eigen::Vector3d computeCycleDistance(Motion* motion);

    // ==================== Height Calibration ====================

    /**
     * @brief Compute height calibration offset
     *
     * Finds the vertical offset needed to place the character
     * at ground level without penetration.
     *
     * @param pose Pose to calibrate
     * @param character Character skeleton
     * @return Height offset value
     */
    static double computeHeightCalibration(const Eigen::VectorXd& pose,
                                           RenderCharacter* character);
};

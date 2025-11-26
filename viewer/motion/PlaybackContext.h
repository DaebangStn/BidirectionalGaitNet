#pragma once

#include <Eigen/Core>
#include <vector>

// Forward declarations
class Motion;
struct PlaybackViewerState;
class RenderCharacter;

/**
 * @brief Unified playback context for motion processing
 *
 * Contains all state needed to render motion at a specific point in time.
 * Replaces the separate MotionPlaybackContext and MarkerPlaybackContext
 * from the original GLFWApp implementation.
 *
 * Design Rationale:
 * - Single context handles both skeleton and marker data
 * - hasMarkers flag enables conditional marker rendering
 * - Computed once per frame, used by render()
 */
struct MotionProcessorContext {
    // ==================== Source References ====================
    Motion* motion = nullptr;               ///< Motion being played
    PlaybackViewerState* state = nullptr;   ///< Associated playback state
    RenderCharacter* character = nullptr;   ///< Character for interpolation

    // ==================== Time State ====================
    double viewerTime = 0.0;                ///< Current viewer time in seconds
    double viewerPhase = 0.0;               ///< Current viewer phase [0, 1)
    double phase = 0.0;                     ///< Computed motion phase [0, 1)
    double frameFloat = 0.0;                ///< Exact frame position (with fractional part)
    double wrappedFrameFloat = 0.0;         ///< Frame wrapped to valid range
    int frameIndex = 0;                     ///< Integer frame index

    // ==================== Motion Info ====================
    int totalFrames = 0;                    ///< Total frames in motion
    int valuesPerFrame = 0;                 ///< DOF count per frame (56 for HDF, 56 for C3D skeleton)

    // ==================== Computed Pose ====================
    Eigen::VectorXd currentPose;            ///< Evaluated skeleton pose (56 DOF)

    // ==================== Marker Data (C3D only) ====================
    std::vector<Eigen::Vector3d> currentMarkers;  ///< Marker positions (empty for HDF)
    bool hasMarkers = false;                ///< true if currentMarkers is populated

    // ==================== Cycle Tracking ====================
    bool cycleWrapped = false;              ///< true if cycle boundary was crossed this frame
    Eigen::Vector3d cycleAccumulation = Eigen::Vector3d::Zero();  ///< Accumulated root offset
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();      ///< Display positioning offset

    // ==================== Validity ====================
    bool valid = false;                     ///< true if context is usable for rendering

    // ==================== Utility Methods ====================

    /**
     * @brief Reset context to invalid state
     */
    void reset() {
        motion = nullptr;
        state = nullptr;
        character = nullptr;
        viewerTime = 0.0;
        viewerPhase = 0.0;
        phase = 0.0;
        frameFloat = 0.0;
        wrappedFrameFloat = 0.0;
        frameIndex = 0;
        totalFrames = 0;
        valuesPerFrame = 0;
        currentPose.resize(0);
        currentMarkers.clear();
        hasMarkers = false;
        cycleWrapped = false;
        cycleAccumulation.setZero();
        displayOffset.setZero();
        valid = false;
    }

    /**
     * @brief Check if context has valid skeleton pose
     */
    bool hasPose() const {
        return valid && currentPose.size() > 0;
    }
};

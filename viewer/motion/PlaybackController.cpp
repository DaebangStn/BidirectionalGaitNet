#include "PlaybackController.h"
#include "Motion.h"
#include "Character.h"
#include "../GLFWApp.h"  // For PlaybackViewerState, PlaybackNavigationMode

#include <algorithm>
#include <iostream>
#include <limits>

// ==================== Phase Computation ====================

double PlaybackController::computePhase(double viewerTime, double cycleDuration)
{
    if (cycleDuration <= 0.0)
        return 0.0;

    double phase = std::fmod(viewerTime / cycleDuration, 1.0);
    if (phase < 0.0)
        phase += 1.0;

    return phase;
}

// ==================== Frame Computation ====================

double PlaybackController::computeFrameFloat(Motion* motion, double phase)
{
    if (!motion)
        return 0.0;

    // phase: [0, 1)
    double frame_float;

    std::vector<double> timestamps = motion->getTimestamps();

    if (!timestamps.empty()) {
        // Motion with timestamps: Use actual simulation time for accurate interpolation
        double t_start = timestamps.front();
        double t_end = timestamps.back();
        double total_duration = t_end - t_start;

        if (total_duration <= 0.0)
            return 0.0;

        // Map phase [0, 1) to one gait cycle worth of time
        double motion_time = t_start + phase * total_duration;

        // Handle wrapping (keep motion_time within valid range)
        motion_time = std::fmod(motion_time - t_start, total_duration) + t_start;

        // Binary search for the frame at or after motion_time
        auto it = std::lower_bound(timestamps.begin(),
                                   timestamps.end(),
                                   motion_time);

        // Calculate frame indices and interpolation weight
        int frame_idx_right = std::distance(timestamps.begin(), it);

        // Clamp to valid range
        if (frame_idx_right >= static_cast<int>(timestamps.size())) {
            frame_idx_right = timestamps.size() - 1;
        }

        int frame_idx_left = (frame_idx_right > 0) ? frame_idx_right - 1 : 0;

        // Calculate interpolation weight based on timestamps
        double t_left = timestamps[frame_idx_left];
        double t_right = timestamps[frame_idx_right];
        double weight = (frame_idx_left == frame_idx_right) ? 0.0 :
                        (motion_time - t_left) / (t_right - t_left);

        // Set frame_float to maintain compatibility with existing code
        frame_float = frame_idx_left + weight;
    } else {
        // No timestamps: direct frame mapping using timesteps per cycle
        frame_float = phase * motion->getTimestepsPerCycle();
    }

    return frame_float;
}

double PlaybackController::wrapFrameFloat(double frameFloat, int totalFrames)
{
    if (totalFrames <= 0)
        return 0.0;

    double wrapped = frameFloat;
    if (wrapped < 0.0 || wrapped >= totalFrames) {
        wrapped = std::fmod(wrapped, static_cast<double>(totalFrames));
        if (wrapped < 0.0)
            wrapped += totalFrames;
    }

    return wrapped;
}

int PlaybackController::computeFrameIndex(double frameFloat, int totalFrames)
{
    int frameIndex = static_cast<int>(std::floor(frameFloat + 1e-9));
    return std::clamp(frameIndex, 0, totalFrames - 1);
}

double PlaybackController::determineMotionFrame(Motion* motion,
                                                PlaybackViewerState& state,
                                                double phase)
{
    if (!motion)
        return 0.0;

    if (state.navigationMode == PLAYBACK_SYNC)
        return computeFrameFloat(motion, phase);

    int total_frames = motion->getTotalTimesteps();
    if (total_frames <= 0)
        return 0.0;

    state.manualFrameIndex = std::clamp(state.manualFrameIndex, 0, total_frames - 1);
    return static_cast<double>(state.manualFrameIndex);
}

// ==================== Cycle Detection ====================

bool PlaybackController::detectCycleWrap(int currentFrameIdx, int lastFrameIdx, int totalFrames)
{
    return (currentFrameIdx < lastFrameIdx) && (lastFrameIdx < totalFrames);
}

// ==================== Cycle Accumulation ====================

void PlaybackController::updateCycleAccumulationStandard(PlaybackViewerState& state,
                                                         int currentFrameIdx)
{
    if (state.navigationMode != PLAYBACK_SYNC)
        return;

    // Detect cycle wrap and accumulate
    if (currentFrameIdx < state.lastFrameIdx) {
        state.cycleAccumulation += state.cycleDistance;
    }

    state.lastFrameIdx = currentFrameIdx;
}

// ==================== Interpolation ====================

Eigen::VectorXd PlaybackController::interpolatePose(const Eigen::VectorXd& p1,
                                                    const Eigen::VectorXd& p2,
                                                    double weight,
                                                    Character* character,
                                                    bool phaseOverflow)
{
    if (!character) {
        // Fallback to linear interpolation
        return p1 * (1.0 - weight) + p2 * weight;
    }

    // Use character's skeleton-aware interpolation
    return character->interpolatePose(p1, p2, weight, phaseOverflow);
}

Eigen::VectorXd PlaybackController::evaluatePose(Motion* motion,
                                                 double frameFloat,
                                                 Character* character,
                                                 const Eigen::Vector3d& cycleAccumulation,
                                                 const Eigen::Vector3d& displayOffset)
{
    if (!motion || !character) {
        return Eigen::VectorXd();
    }

    int valuesPerFrame = motion->getValuesPerFrame();
    int totalFrames = motion->getTotalTimesteps();

    if (totalFrames <= 0 || valuesPerFrame <= 0)
        return Eigen::VectorXd();

    // Clamp frame_float to valid range
    if (frameFloat < 0) frameFloat = 0;
    if (frameFloat >= totalFrames) frameFloat = std::fmod(frameFloat, totalFrames);

    int currentFrameIdx = static_cast<int>(frameFloat);
    currentFrameIdx = std::max(0, std::min(currentFrameIdx, totalFrames - 1));

    // Extract and interpolate frame data
    Eigen::VectorXd interpolatedFrame;
    double weight = frameFloat - std::floor(frameFloat);

    Eigen::VectorXd rawMotion = motion->getRawMotionData();

    // Safety check: ensure motion data is large enough
    int requiredSize = totalFrames * valuesPerFrame;
    if (rawMotion.size() < requiredSize) {
        std::cerr << "[PlaybackController::evaluatePose] Warning: Motion data too small! Expected "
                  << requiredSize << " but got " << rawMotion.size() << std::endl;
        return Eigen::VectorXd::Zero(valuesPerFrame);
    }

    if (weight > 1e-6) {
        int nextFrameIdx = (currentFrameIdx + 1) % totalFrames;
        Eigen::VectorXd p1 = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
        Eigen::VectorXd p2 = rawMotion.segment(nextFrameIdx * valuesPerFrame, valuesPerFrame);

        // HDF motion: use Character's skeleton-aware interpolation
        bool phaseOverflow = (nextFrameIdx < currentFrameIdx);  // Detect cycle wraparound
        interpolatedFrame = interpolatePose(p1, p2, weight, character, phaseOverflow);
    } else {
        // No interpolation needed, use exact frame
        interpolatedFrame = rawMotion.segment(currentFrameIdx * valuesPerFrame, valuesPerFrame);
    }

    // Apply position offset (HDF/BVH approach: additive)
    Eigen::VectorXd motionPos = interpolatedFrame;
    motionPos[3] += cycleAccumulation[0] + displayOffset[0];
    motionPos[4] += displayOffset[1];
    motionPos[5] += cycleAccumulation[2] + displayOffset[2];

    return motionPos;
}

// ==================== Cycle Distance ====================

Eigen::Vector3d PlaybackController::computeCycleDistance(Motion* motion)
{
    if (!motion)
        return Eigen::Vector3d::Zero();

    int totalFrames = motion->getTotalTimesteps();
    int valuesPerFrame = motion->getValuesPerFrame();

    if (totalFrames <= 1 || valuesPerFrame < 6)
        return Eigen::Vector3d::Zero();

    Eigen::VectorXd rawMotion = motion->getRawMotionData();

    if (rawMotion.size() < static_cast<Eigen::Index>(totalFrames * valuesPerFrame))
        return Eigen::Vector3d::Zero();

    // Get first and last frame root positions (indices 3, 4, 5)
    Eigen::VectorXd firstFrame = rawMotion.segment(0, valuesPerFrame);
    Eigen::VectorXd lastFrame = rawMotion.segment((totalFrames - 1) * valuesPerFrame, valuesPerFrame);

    // Calculate cycle distance with frame count correction
    double framePerCycle = static_cast<double>(totalFrames);
    Eigen::Vector3d cycleDistance;
    cycleDistance[0] = (lastFrame[3] - firstFrame[3]) * framePerCycle / (framePerCycle - 1);
    cycleDistance[1] = 0.0;  // Y (height) doesn't accumulate
    cycleDistance[2] = (lastFrame[5] - firstFrame[5]) * framePerCycle / (framePerCycle - 1);

    return cycleDistance;
}

// ==================== Height Calibration ====================

double PlaybackController::computeHeightCalibration(const Eigen::VectorXd& pose,
                                                    Character* character)
{
    if (!character || !character->getSkeleton() || pose.size() == 0)
        return 0.0;

    // Temporarily set the character to the pose we want to calibrate
    Eigen::VectorXd originalPose = character->getSkeleton()->getPositions();
    character->getSkeleton()->setPositions(pose);

    // Find the lowest body node Y position
    double lowestY = std::numeric_limits<double>::max();

    for (size_t i = 0; i < character->getSkeleton()->getNumBodyNodes(); ++i) {
        auto* bodyNode = character->getSkeleton()->getBodyNode(i);
        if (!bodyNode)
            continue;

        Eigen::Vector3d worldPos = bodyNode->getWorldTransform().translation();
        if (worldPos[1] < lowestY)
            lowestY = worldPos[1];
    }

    // Restore original pose
    character->getSkeleton()->setPositions(originalPose);

    // Return offset needed to place at ground level with small margin
    if (lowestY < std::numeric_limits<double>::max())
        return -lowestY + 0.001;  // 1mm safety margin

    return 0.0;
}

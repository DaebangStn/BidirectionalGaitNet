#pragma once

#include <algorithm>
#include <cmath>
#include "imgui.h"

// Note: This header uses PlaybackNavigationMode, MotionViewerState, and MarkerViewerState
// which are defined in GLFWApp.h. Include this file AFTER GLFWApp.h in your .cpp file.

/**
 * @file PlaybackUtils.h
 * @brief Common utility functions for motion and marker playback
 *
 * This file contains template functions and utilities shared between
 * motion and marker playback pipelines to reduce code duplication.
 */

namespace PlaybackUtils {

/**
 * @brief Wraps a frame float value to valid range [0, totalFrames)
 *
 * Handles negative values and values beyond totalFrames by wrapping.
 *
 * @param frameFloat The frame float value to wrap
 * @param totalFrames Total number of frames
 * @return Wrapped frame float in range [0, totalFrames)
 */
inline double wrapFrameFloat(double frameFloat, int totalFrames)
{
    if (totalFrames <= 0)
        return 0.0;

    // Handle negative values
    while (frameFloat < 0.0) {
        frameFloat += totalFrames;
    }

    // Handle values beyond totalFrames
    while (frameFloat >= totalFrames) {
        frameFloat -= totalFrames;
    }

    return frameFloat;
}

/**
 * @brief Clamps manual frame index to valid range [0, maxIndex]
 *
 * @param index Reference to frame index to clamp (modified in-place)
 * @param maxIndex Maximum valid frame index
 */
inline void clampManualFrameIndex(int& index, int maxIndex)
{
    index = std::clamp(index, 0, maxIndex);
}

/**
 * @brief Unified UI controls for playback navigation (radio buttons + slider)
 *
 * Displays consistent navigation controls for any playable data type.
 * Shows "Sync" and "Manual" radio buttons, and a frame slider when in manual mode.
 *
 * @tparam StateType Type of viewer state (MotionViewerState or MarkerViewerState)
 * @param label Label for the navigation section (e.g., "Motion Nav", "Marker Nav")
 * @param state Viewer state to control
 * @param maxFrameIndex Maximum frame index for manual mode slider
 * @param syncLabel Optional custom label for sync button (default: "Sync")
 * @param manualLabel Optional custom label for manual button (default: "Manual")
 */
template<typename StateType>
inline void drawPlaybackNavigationUI(const char* label,
                                     StateType& state,
                                     int maxFrameIndex,
                                     const char* syncLabel = "Sync",
                                     const char* manualLabel = "Manual")
{
    ImGui::PushID(label);  // Use label as unique ID scope

    ImGui::Text("%s:", label);
    ImGui::SameLine();

    int navMode = static_cast<int>(state.navigationMode);

    if (ImGui::RadioButton(syncLabel, &navMode, PLAYBACK_SYNC)) {
        state.navigationMode = PLAYBACK_SYNC;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(manualLabel, &navMode, PLAYBACK_MANUAL_FRAME)) {
        state.navigationMode = PLAYBACK_MANUAL_FRAME;
    }

    if (state.navigationMode == PLAYBACK_MANUAL_FRAME) {
        clampManualFrameIndex(state.manualFrameIndex, maxFrameIndex);
        ImGui::SliderInt("Frame Index", &state.manualFrameIndex, 0, maxFrameIndex);
    }

    ImGui::PopID();
}

/**
 * @brief Computes integer frame index from frame float
 *
 * Adds small epsilon to handle floating point rounding issues.
 *
 * @param frameFloat Floating-point frame index
 * @param totalFrames Total number of frames (for clamping)
 * @return Integer frame index
 */
inline int computeFrameIndex(double frameFloat, int totalFrames)
{
    int frameIndex = static_cast<int>(std::floor(frameFloat + 1e-9));
    return std::clamp(frameIndex, 0, totalFrames - 1);
}

/**
 * @brief Detects if a cycle wrap occurred (frame index decreased)
 *
 * @param currentFrameIdx Current frame index
 * @param lastFrameIdx Previous frame index
 * @param totalFrames Total frames in data source
 * @return true if cycle wrapped (currentFrameIdx < lastFrameIdx)
 */
inline bool detectCycleWrap(int currentFrameIdx, int lastFrameIdx, int totalFrames)
{
    return (currentFrameIdx < lastFrameIdx) && (lastFrameIdx < totalFrames);
}

} // namespace PlaybackUtils

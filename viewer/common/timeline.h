#ifndef TIMELINE_H
#define TIMELINE_H

#include <vector>

namespace Timeline {

/**
 * @brief Gait direction between consecutive contact phases
 */
enum class GaitDirection {
    Unknown,   // First contact or insufficient displacement
    Forward,   // Movement in +Z direction
    Backward   // Movement in -Z direction
};

/**
 * @brief Unified foot contact phase
 *
 * Replaces both FootContactPhase (MotionEditorApp) and FootLockPhase (C3D_Reader)
 */
struct FootContactPhase {
    int startFrame;
    int endFrame;
    bool isLeft;  // true = left foot, false = right foot
    GaitDirection direction = GaitDirection::Unknown;
};

/**
 * @brief Timeline configuration options
 */
struct Config {
    float height = 80.0f;           // Timeline window height
    bool showTrimMarkers = false;   // Show trim start/end markers
    int trimStart = 0;              // Trim start frame
    int trimEnd = 0;                // Trim end frame

    // Zoom state (caller-managed pointers for persistence)
    float* zoom = nullptr;          // Zoom level (1.0 = no zoom, >1 = magnified)
    float* scrollOffset = nullptr;  // Scroll offset in normalized [0,1] range
};

/**
 * @brief Timeline scrub result
 */
struct Result {
    bool scrubbed = false;   // Whether user scrubbed the timeline
    int targetFrame = 0;     // Target frame if scrubbed
};

/**
 * @brief Draw timeline trackbar with foot contact phases
 *
 * @param windowWidth   Application window width
 * @param windowHeight  Application window height
 * @param totalFrames   Total number of frames in motion
 * @param currentFrame  Current playback frame
 * @param phases        Foot contact phases to display
 * @param viewerTime    Current time for display (seconds)
 * @param isPlaying     Playback state for display
 * @param config        Optional configuration
 * @return Result       Scrub result with target frame if user interacted
 */
Result DrawTimelineTrackBar(
    int windowWidth,
    int windowHeight,
    int totalFrames,
    int currentFrame,
    const std::vector<FootContactPhase>& phases,
    float viewerTime,
    bool isPlaying,
    const Config& config = {}
);

} // namespace Timeline

#endif // TIMELINE_H

#include "timeline.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace Timeline {

Result DrawTimelineTrackBar(
    int windowWidth,
    int windowHeight,
    int totalFrames,
    int currentFrame,
    const std::vector<FootContactPhase>& phases,
    float viewerTime,
    bool isPlaying,
    const Config& config)
{
    Result result;
    const float timelineHeight = config.height;

    // Get zoom state (use defaults if not provided)
    float zoom = config.zoom ? *config.zoom : 1.0f;
    float scrollOffset = config.scrollOffset ? *config.scrollOffset : 0.0f;

    ImGui::SetNextWindowPos(ImVec2(0, windowHeight - timelineHeight));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(windowWidth), timelineHeight));
    ImGui::Begin("Timeline", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoScrollbar);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();

    float trackHeight = 30.0f;
    float trackY = canvasPos.y + 5.0f;
    float trackWidth = canvasSize.x - 20.0f;
    float trackX = canvasPos.x + 10.0f;

    // Handle zoom with Ctrl+Scroll
    bool isHovered = ImGui::IsWindowHovered();
    if (isHovered && ImGui::GetIO().KeyCtrl) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f && config.zoom) {
            // Get mouse position relative to track for zoom centering
            float mouseRelX = (ImGui::GetMousePos().x - trackX) / trackWidth;
            mouseRelX = std::clamp(mouseRelX, 0.0f, 1.0f);

            // Calculate visible range before zoom
            float visibleWidth = 1.0f / zoom;
            float viewStart = scrollOffset;
            float mouseFrame = viewStart + mouseRelX * visibleWidth;

            // Apply zoom
            float zoomFactor = (wheel > 0) ? 1.15f : (1.0f / 1.15f);
            float newZoom = std::clamp(zoom * zoomFactor, 1.0f, 50.0f);

            // Adjust scroll to keep mouse position stable
            if (config.scrollOffset && newZoom > 1.0f) {
                float newVisibleWidth = 1.0f / newZoom;
                float newScrollOffset = mouseFrame - mouseRelX * newVisibleWidth;
                newScrollOffset = std::clamp(newScrollOffset, 0.0f, 1.0f - newVisibleWidth);
                *config.scrollOffset = newScrollOffset;
            } else if (config.scrollOffset) {
                *config.scrollOffset = 0.0f;
            }

            *config.zoom = newZoom;
            zoom = newZoom;
            scrollOffset = config.scrollOffset ? *config.scrollOffset : 0.0f;
        }
    }
    // Handle horizontal scroll (without Ctrl) when zoomed
    else if (isHovered && zoom > 1.0f && config.scrollOffset) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            float visibleWidth = 1.0f / zoom;
            float scrollStep = visibleWidth * 0.2f;  // Scroll 20% of visible area
            float newOffset = scrollOffset - wheel * scrollStep;
            newOffset = std::clamp(newOffset, 0.0f, 1.0f - visibleWidth);
            *config.scrollOffset = newOffset;
            scrollOffset = newOffset;
        }
    }

    // Calculate visible frame range based on zoom
    float visibleWidth = 1.0f / zoom;
    float viewStart = scrollOffset;
    float viewEnd = viewStart + visibleWidth;

    // Helper lambda to convert frame to screen X with zoom
    auto frameToScreenX = [&](int frame) -> float {
        if (totalFrames <= 1) return trackX;
        float normalizedPos = static_cast<float>(frame) / (totalFrames - 1);
        float viewPos = (normalizedPos - viewStart) / visibleWidth;
        return trackX + viewPos * trackWidth;
    };

    // Background
    drawList->AddRectFilled(
        ImVec2(trackX, trackY),
        ImVec2(trackX + trackWidth, trackY + trackHeight),
        IM_COL32(40, 40, 40, 255)
    );
    drawList->AddRect(
        ImVec2(trackX, trackY),
        ImVec2(trackX + trackWidth, trackY + trackHeight),
        IM_COL32(80, 80, 80, 255)
    );

    // Set clipping rectangle for track content
    drawList->PushClipRect(
        ImVec2(trackX, trackY),
        ImVec2(trackX + trackWidth, trackY + trackHeight),
        true
    );

    if (totalFrames > 0) {
        // Draw foot contact phases
        if (!phases.empty()) {
            float halfHeight = trackHeight / 2.0f;
            for (const auto& phase : phases) {
                float startX = frameToScreenX(phase.startFrame);
                float endX = frameToScreenX(phase.endFrame);

                // Skip if completely outside visible area
                if (endX < trackX || startX > trackX + trackWidth) continue;

                float phaseY = phase.isLeft ? trackY : trackY + halfHeight;
                ImU32 color = phase.isLeft ? IM_COL32(80, 120, 200, 180) : IM_COL32(200, 80, 80, 180);
                drawList->AddRectFilled(ImVec2(startX, phaseY), ImVec2(endX, phaseY + halfHeight), color);

                // Draw direction arrow at start of phase (if visible)
                if (phase.direction != GaitDirection::Unknown && startX >= trackX && startX <= trackX + trackWidth) {
                    float arrowY = trackY + halfHeight;
                    float arrowSize = 5.0f;

                    ImU32 arrowColor = (phase.direction == GaitDirection::Forward)
                        ? IM_COL32(80, 220, 80, 255)
                        : IM_COL32(255, 140, 40, 255);

                    if (phase.direction == GaitDirection::Forward) {
                        drawList->AddTriangleFilled(
                            ImVec2(startX - arrowSize * 0.5f, arrowY - arrowSize),
                            ImVec2(startX - arrowSize * 0.5f, arrowY + arrowSize),
                            ImVec2(startX + arrowSize, arrowY),
                            arrowColor);
                    } else {
                        drawList->AddTriangleFilled(
                            ImVec2(startX + arrowSize * 0.5f, arrowY - arrowSize),
                            ImVec2(startX + arrowSize * 0.5f, arrowY + arrowSize),
                            ImVec2(startX - arrowSize, arrowY),
                            arrowColor);
                    }
                }
            }
            drawList->AddLine(
                ImVec2(trackX, trackY + halfHeight),
                ImVec2(trackX + trackWidth, trackY + halfHeight),
                IM_COL32(60, 60, 60, 255), 1.0f
            );
        }

        // Draw trim range markers (green)
        if (config.showTrimMarkers && (config.trimStart > 0 || config.trimEnd < totalFrames - 1)) {
            float trimStartX = frameToScreenX(config.trimStart);
            float trimEndX = frameToScreenX(config.trimEnd);
            if (trimStartX >= trackX && trimStartX <= trackX + trackWidth) {
                drawList->AddLine(ImVec2(trimStartX, trackY), ImVec2(trimStartX, trackY + trackHeight), IM_COL32(50, 200, 50, 255), 2.0f);
            }
            if (trimEndX >= trackX && trimEndX <= trackX + trackWidth) {
                drawList->AddLine(ImVec2(trimEndX, trackY), ImVec2(trimEndX, trackY + trackHeight), IM_COL32(50, 200, 50, 255), 2.0f);
            }
        }

        // Progress bar (from view start to current frame)
        float playheadX = frameToScreenX(currentFrame);
        float progressStartX = std::max(trackX, frameToScreenX(0));
        float progressEndX = std::clamp(playheadX, trackX, trackX + trackWidth);
        if (progressEndX > progressStartX) {
            drawList->AddRectFilled(
                ImVec2(progressStartX, trackY),
                ImVec2(progressEndX, trackY + trackHeight),
                IM_COL32(255, 255, 255, 40)
            );
        }

        // Playhead (if visible)
        if (playheadX >= trackX && playheadX <= trackX + trackWidth) {
            drawList->AddLine(ImVec2(playheadX, trackY - 3), ImVec2(playheadX, trackY + trackHeight + 3), IM_COL32(255, 255, 0, 255), 2.0f);
            drawList->AddTriangleFilled(
                ImVec2(playheadX - 5, trackY - 3),
                ImVec2(playheadX + 5, trackY - 3),
                ImVec2(playheadX, trackY + 4),
                IM_COL32(255, 255, 0, 255)
            );
        }
    }

    drawList->PopClipRect();

    // Click/drag to scrub
    ImGui::SetCursorScreenPos(ImVec2(trackX, trackY));
    ImGui::InvisibleButton("timeline_track", ImVec2(trackWidth, trackHeight));
    if (ImGui::IsItemActive() && totalFrames > 0 && !ImGui::GetIO().KeyCtrl) {
        float relativeX = std::clamp((ImGui::GetMousePos().x - trackX) / trackWidth, 0.0f, 1.0f);
        // Convert screen position to frame considering zoom
        float normalizedPos = viewStart + relativeX * visibleWidth;
        result.scrubbed = true;
        result.targetFrame = static_cast<int>(std::clamp(normalizedPos, 0.0f, 1.0f) * (totalFrames - 1));
    }

    // Info text with zoom indicator
    ImGui::SetCursorScreenPos(ImVec2(canvasPos.x + 10, trackY + trackHeight + 5));
    if (totalFrames > 0) {
        if (zoom > 1.01f) {
            int visibleStartFrame = static_cast<int>(viewStart * (totalFrames - 1));
            int visibleEndFrame = static_cast<int>(viewEnd * (totalFrames - 1));
            ImGui::Text("Frame: %d / %d  |  Time: %.2fs  |  %s  |  Zoom: %.1fx [%d-%d]",
                        currentFrame, totalFrames - 1, viewerTime, isPlaying ? "Playing" : "Paused",
                        zoom, visibleStartFrame, visibleEndFrame);
        } else {
            ImGui::Text("Frame: %d / %d  |  Time: %.2fs  |  %s",
                        currentFrame, totalFrames - 1, viewerTime, isPlaying ? "Playing" : "Paused");
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No motion loaded");
    }

    ImGui::End();

    return result;
}

} // namespace Timeline

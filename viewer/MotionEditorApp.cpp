#include "MotionEditorApp.h"
#include "DARTHelper.h"
#include "Log.h"
#include <rm/global.hpp>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>

using namespace dart::dynamics;
namespace fs = std::filesystem;

// =============================================================================
// Constructor / Destructor
// =============================================================================

MotionEditorApp::MotionEditorApp(const std::string& configPath)
    : ViewerAppBase("Motion Editor", 1280, 720)
    , mConfigPath(configPath.empty() ? "data/rm_config.yaml" : configPath)
{
    // Initialize Resource Manager for PID-based access (use singleton)
    try {
        mResourceManager = &rm::getManager();

        // Initialize PID Navigator with HDF file filter
        mPIDNavigator = std::make_unique<PIDNav::PIDNavigator>(
            mResourceManager,
            std::make_unique<PIDNav::HDFFileFilter>()
        );

        // Register callback for when user selects an H5 file
        mPIDNavigator->setFileSelectionCallback(
            [this](const std::string& path, const std::string& filename) {
                loadH5Motion(path);
            }
        );

        // Register callback for when PID selection changes
        mPIDNavigator->setPIDChangeCallback(
            [this](const std::string& pid) {
                scanSkeletonDirectory();
                autoDetectSkeleton();
            }
        );

        // Initial scan of available PIDs
        mPIDNavigator->scanPIDs();
    } catch (const rm::RMError& e) {
        LOG_WARN("[MotionEditor] Resource manager init failed: " << e.what());
    } catch (const std::exception& e) {
        LOG_WARN("[MotionEditor] Resource manager init failed: " << e.what());
    }

    mLastRealTime = glfwGetTime();

    LOG_INFO("[MotionEditor] Initialized");
}

MotionEditorApp::~MotionEditorApp()
{
    if (mMotion) {
        delete mMotion;
        mMotion = nullptr;
    }
    // Base class handles GLFW/ImGui cleanup
}

// =============================================================================
// ViewerAppBase Overrides
// =============================================================================

void MotionEditorApp::onInitialize()
{
    // Additional initialization after base class setup (if needed)
}

void MotionEditorApp::onFrameStart()
{
    // Update playback timing
    double currentTime = glfwGetTime();
    double dt = currentTime - mLastRealTime;
    mLastRealTime = currentTime;

    if (mIsPlaying) {
        updateViewerTime(dt * mPlaybackSpeed);
    }
}

void MotionEditorApp::updateCamera()
{
    // Camera follow character
    if (mCamera.focus == CameraFocusMode::FOLLOW_CHARACTER && mMotion != nullptr && mCharacter) {
        int currentFrameIdx = mMotionState.manualFrameIndex;
        if (mMotionState.navigationMode == ME_SYNC) {
            double frameFloat = (mViewerTime / mCycleDuration) * mMotion->getTotalTimesteps();
            currentFrameIdx = static_cast<int>(frameFloat) % mMotion->getTotalTimesteps();
        }

        Eigen::VectorXd pose = mMotion->getPose(currentFrameIdx);
        if (pose.size() >= 6) {
            mCamera.trans[0] = -(pose[3] + mMotionState.cycleAccumulation[0] + mMotionState.displayOffset[0]);
            mCamera.trans[1] = -(pose[4] + mMotionState.displayOffset[1]) - 1;
            mCamera.trans[2] = -(pose[5] + mMotionState.cycleAccumulation[2] + mMotionState.displayOffset[2]);
        }
    }
}

void MotionEditorApp::drawContent()
{
    if (mCharacter && mMotion) {
        // Draw original skeleton (blue)
        drawSkeleton(false);

        // Draw preview skeleton (orange) if transformations pending
        bool hasRotationPreview = std::abs(mPendingRotationAngle) > 0.001f;
        bool hasHeightPreview = mHeightOffsetComputed && std::abs(mComputedHeightOffset) > 0.001;
        bool hasROMPreview = mPreviewClampedPose && !mROMViolations.empty();

        if (hasRotationPreview || hasHeightPreview || hasROMPreview) {
            drawSkeleton(true);
        }

        // Draw rotation axis arrow at origin when rotating
        if (hasRotationPreview) {
            GUI::DrawArrow3D(
                Eigen::Vector3d::Zero(),              // origin
                Eigen::Vector3d(0, 1, 0),             // Y-axis direction
                1.0,                                  // length
                0.02,                                 // thickness
                Eigen::Vector4d(0.2, 0.8, 0.2, 1.0)   // green color
            );
        }
    }

    // Draw origin axis gizmo when camera is moving
    if (mMouseDown) {
        Eigen::Vector3d center = -mCamera.trans;
        GUI::DrawOriginAxisGizmo(center);
    }
}

void MotionEditorApp::drawUI()
{
    drawLeftPanel();
    drawRightPanel();
    drawTimelineTrackBar();
}

void MotionEditorApp::keyPress(int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_SPACE:
                mIsPlaying = !mIsPlaying;
                return;
            case GLFW_KEY_S:
                // Step frame(s) forward: Ctrl+S = 5 frames, S = 1 frame
                if (mMotion) {
                    mIsPlaying = false;
                    int step = (mods & GLFW_MOD_CONTROL) ? 5 : 1;
                    mMotionState.manualFrameIndex = std::min(
                        mMotionState.manualFrameIndex + step,
                        mMotion->getNumFrames() - 1);
                    mMotionState.navigationMode = ME_MANUAL_FRAME;
                }
                return;
            case GLFW_KEY_A:
                // Step frame(s) backward: Ctrl+A = 5 frames, A = 1 frame
                if (mMotion) {
                    mIsPlaying = false;
                    int step = (mods & GLFW_MOD_CONTROL) ? 5 : 1;
                    mMotionState.manualFrameIndex = std::max(
                        mMotionState.manualFrameIndex - step, 0);
                    mMotionState.navigationMode = ME_MANUAL_FRAME;
                }
                return;
            case GLFW_KEY_R:
                // Reset playback state (not camera - that's handled by base class)
                resetPlayback();
                return;
            case GLFW_KEY_1:
            case GLFW_KEY_KP_1:
                alignCameraToPlaneQuat(1);  // XY plane
                return;
            case GLFW_KEY_2:
            case GLFW_KEY_KP_2:
                alignCameraToPlaneQuat(2);  // YZ plane
                return;
            case GLFW_KEY_3:
            case GLFW_KEY_KP_3:
                alignCameraToPlaneQuat(3);  // ZX plane
                return;
            case GLFW_KEY_LEFT_BRACKET:  // [
                // Set trim start to current frame
                if (mMotion) {
                    mTrimStart = mMotionState.manualFrameIndex;
                    LOG_INFO("[MotionEditor] Trim start set to frame " << mTrimStart);
                }
                return;
            case GLFW_KEY_RIGHT_BRACKET:  // ]
                // Set trim end to current frame
                if (mMotion) {
                    mTrimEnd = mMotionState.manualFrameIndex;
                    LOG_INFO("[MotionEditor] Trim end set to frame " << mTrimEnd);
                }
                return;
            case GLFW_KEY_O:
                mAppRenderMode = static_cast<MotionEditorRenderMode>(
                    (static_cast<int>(mAppRenderMode) + 1) % 2);
                return;
        }
    }

    // Call base class for common keys (F, G, ESC)
    ViewerAppBase::keyPress(key, scancode, action, mods);
}

// =============================================================================
// Initialization
// =============================================================================

void MotionEditorApp::loadRenderConfigImpl()
{
    // Load motion_editor section from render.yaml (Template Method hook)
    // Common config (geometry, default_open_panels) already loaded by ViewerAppBase
    try {
        std::string resolved_path = rm::resolve("render.yaml");
        YAML::Node config = YAML::LoadFile(resolved_path);

        if (config["motion_editor"]) {
            if (config["motion_editor"]["foot_contact"]) {
                auto fc = config["motion_editor"]["foot_contact"];
                if (fc["velocity_threshold"])
                    mContactVelocityThreshold = fc["velocity_threshold"].as<float>();
                if (fc["min_lock_frames"])
                    mContactMinLockFrames = fc["min_lock_frames"].as<int>();
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN("[MotionEditor] Could not load render.yaml: " << e.what());
    }
}

// isPanelDefaultOpen() is inherited from ViewerAppBase

// =============================================================================
// Rendering
// =============================================================================

void MotionEditorApp::drawSkeleton(bool isPreview)
{
    if (!mCharacter || !mMotion) return;

    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    // Get current frame index
    int frameIdx = mMotionState.manualFrameIndex;
    frameIdx = std::clamp(frameIdx, 0, mMotion->getNumFrames() - 1);

    // Get pose from motion
    Eigen::VectorXd pose = mMotion->getPose(frameIdx);

    // Set color based on preview mode
    Eigen::Vector4d color;
    if (isPreview) {
        // Apply rotation transform
        if (std::abs(mPendingRotationAngle) > 0.001f) {
            pose = applyRotationToFrame(pose, mPendingRotationAngle);
        }
        // Apply height offset
        if (mHeightOffsetComputed && std::abs(mComputedHeightOffset) > 0.001) {
            pose[4] += mComputedHeightOffset;
        }
        // Apply ROM clamping if enabled
        if (mPreviewClampedPose && !mROMViolations.empty()) {
            Eigen::VectorXd rom_min = skel->getPositionLowerLimits();
            Eigen::VectorXd rom_max = skel->getPositionUpperLimits();
            pose = pose.cwiseMax(rom_min).cwiseMin(rom_max);
        }
        // Semi-transparent orange for preview
        color = Eigen::Vector4d(0.9, 0.5, 0.3, 0.6);
    } else {
        // Original blue
        color = Eigen::Vector4d(0.3, 0.6, 0.9, 1.0);
        // Store current pose for non-preview (used by evaluateMotionPose)
        mMotionState.currentPose = pose;
    }

    // Apply pose to skeleton
    if (pose.size() == skel->getNumDofs()) {
        skel->setPositions(pose);
    }

    // Draw skeleton
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    // Enable blending for semi-transparent preview
    if (isPreview) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        const dart::dynamics::BodyNode* bn = skel->getBodyNode(i);
        if (!bn) continue;

        // Get per-bone color (may highlight Talus during contact)
        Eigen::Vector4d boneColor = isPreview ? color : getRenderColor(bn, color);

        glPushMatrix();
        glMultMatrixd(bn->getTransform().data());

        bn->eachShapeNodeWith<dart::dynamics::VisualAspect>([this, &boneColor](const dart::dynamics::ShapeNode* sn) {
            if (!sn) return true;
            const auto& va = sn->getVisualAspect();
            if (!va || va->isHidden()) return true;

            glPushMatrix();
            Eigen::Affine3d tmp = sn->getRelativeTransform();
            glMultMatrixd(tmp.data());

            const auto* shape = sn->getShape().get();

            // Render primitive shapes (Primitive or Wireframe mode)
            if (mAppRenderMode == MotionEditorRenderMode::Wireframe) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(2.0f);
            }
            glColor4f(boneColor[0], boneColor[1], boneColor[2], boneColor[3]);

            if (shape->is<dart::dynamics::BoxShape>()) {
                GUI::DrawCube(static_cast<const dart::dynamics::BoxShape*>(shape)->getSize());
            } else if (shape->is<dart::dynamics::CapsuleShape>()) {
                auto* cap = static_cast<const dart::dynamics::CapsuleShape*>(shape);
                GUI::DrawCapsule(cap->getRadius(), cap->getHeight());
            } else if (shape->is<dart::dynamics::SphereShape>()) {
                GUI::DrawSphere(static_cast<const dart::dynamics::SphereShape*>(shape)->getRadius());
            } else if (shape->is<dart::dynamics::CylinderShape>()) {
                auto* cyl = static_cast<const dart::dynamics::CylinderShape*>(shape);
                GUI::DrawCylinder(cyl->getRadius(), cyl->getHeight());
            }

            // Restore fill mode and line width after wireframe
            if (mAppRenderMode == MotionEditorRenderMode::Wireframe) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glLineWidth(1.0f);
            }

            glPopMatrix();
            return true;
        });

        glPopMatrix();
    }

    // Disable blending after preview
    if (isPreview) {
        glDisable(GL_BLEND);
    }

    // Draw head facing direction arrow
    auto headNode = skel->getBodyNode("Head");
    if (headNode) {
        Eigen::Vector3d headPos = headNode->getTransform().translation();
        // Forward direction = local +Z axis (facing forward in skeleton convention)
        Eigen::Vector3d headForward = headNode->getTransform().linear() * Eigen::Vector3d(0, 0, 1);

        // Yellow arrow for head direction
        GUI::DrawArrow3D(headPos, headForward, 0.15, 0.008,
                         Eigen::Vector4d(1.0, 0.8, 0.0, 1.0));
    }
}

// =============================================================================
// UI Panels
// =============================================================================

void MotionEditorApp::drawLeftPanel()
{
    const float timelineHeight = 80.0f;
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight - timelineHeight), ImGuiCond_Once);
    ImGui::Begin("Data Loader", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    // Tab bar for PID Browser / Direct Path
    if (ImGui::BeginTabBar("DataLoaderTabs")) {
        if (ImGui::BeginTabItem("PID Browser")) {
            drawPIDBrowserTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Direct Path")) {
            drawDirectPathTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    drawSkeletonSection();
    ImGui::Separator();
    drawPlaybackSection();

    ImGui::End();
}

void MotionEditorApp::drawRightPanel()
{
    const float timelineHeight = 80.0f;
    ImGui::SetNextWindowSize(ImVec2(mRightPanelWidth, mHeight - timelineHeight), ImGuiCond_Once);
    ImGui::Begin("Data loader", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos(ImVec2(mWidth - ImGui::GetWindowSize().x, 0), ImGuiCond_Always);

    drawMotionInfoSection();
    ImGui::Separator();
    drawRotationSection();
    ImGui::Separator();
    drawHeightSection();
    ImGui::Separator();
    drawROMViolationSection();
    ImGui::Separator();
    drawFootContactSection();
    ImGui::Separator();
    drawTrimSection();
    ImGui::Separator();
    drawStrideEstimationSection();
    ImGui::Separator();
    drawExportSection();

    ImGui::End();
}

void MotionEditorApp::drawPIDBrowserTab()
{
    if (!mPIDNavigator) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "PID Navigator not initialized");
        return;
    }

    // Use the shared PIDNavigator component
    mPIDNavigator->renderInlineSelector(120, 120);
}

void MotionEditorApp::drawDirectPathTab()
{
    static char directPath[512] = {0};

    ImGui::Text("H5 Motion Path:");
    ImGui::SetNextItemWidth(-60);
    ImGui::InputText("##DirectH5Path", directPath, sizeof(directPath));
    ImGui::SameLine();
    if (ImGui::Button("Load##Direct")) {
        if (strlen(directPath) > 0) {
            loadH5Motion(directPath);
        }
    }

    ImGui::Separator();
    ImGui::Text("Skeleton Path:");
    ImGui::SetNextItemWidth(-60);
    ImGui::InputText("##DirectSkelPath", mManualSkeletonPath, sizeof(mManualSkeletonPath));
    ImGui::SameLine();
    if (ImGui::Button("Load##Skel")) {
        if (strlen(mManualSkeletonPath) > 0) {
            loadSkeleton(mManualSkeletonPath);
        }
    }
}

void MotionEditorApp::drawSkeletonSection()
{
    if (collapsingHeaderWithControls("Skeleton Config")) {
        ImGui::Checkbox("Auto-detect (PID)", &mUseAutoSkeleton);

        if (mUseAutoSkeleton) {
            if (mAutoDetectedSkeletonPath.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "No skeleton detected");
            } else {
                ImGui::TextWrapped("Path: %s", mAutoDetectedSkeletonPath.c_str());
            }
        } else {
            // PID skeleton directory browser
            bool hasPIDSelected = mPIDNavigator && mPIDNavigator->getState().selectedPID >= 0;
            if (hasPIDSelected) {
                ImGui::Text("PID Skeleton Files:");
                ImGui::SameLine();
                if (ImGui::Button("Refresh##Skel")) {
                    scanSkeletonDirectory();
                }

                // Listbox for skeleton files
                if (!mSkeletonFiles.empty()) {
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::BeginListBox("##SkeletonList", ImVec2(-1, 100))) {
                        for (size_t i = 0; i < mSkeletonFileNames.size(); ++i) {
                            bool isSelected = (mSelectedSkeletonFile == static_cast<int>(i));
                            if (ImGui::Selectable(mSkeletonFileNames[i].c_str(), isSelected)) {
                                mSelectedSkeletonFile = static_cast<int>(i);
                                // Copy path to manual path buffer and load
                                strncpy(mManualSkeletonPath, mSkeletonFiles[i].c_str(), sizeof(mManualSkeletonPath) - 1);
                                mManualSkeletonPath[sizeof(mManualSkeletonPath) - 1] = '\0';
                                loadSkeleton(mSkeletonFiles[i]);
                            }
                            if (isSelected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndListBox();
                    }
                    ImGui::TextWrapped("%s", mSkeletonDirectory.c_str());
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "No skeleton files in PID folder");
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Select a PID first");
            }

            // Manual path input as fallback
            ImGui::Separator();
            ImGui::Text("Or enter path:");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##ManualSkel", mManualSkeletonPath, sizeof(mManualSkeletonPath));
        }

        if (ImGui::Button("Reload Skeleton")) {
            std::string path = mUseAutoSkeleton ? mAutoDetectedSkeletonPath : std::string(mManualSkeletonPath);
            if (!path.empty()) {
                loadSkeleton(path);
            }
        }

        // Show current loaded skeleton
        if (mCharacter) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Loaded");
        }

        // Render mode
        ImGui::Separator();
        ImGui::Text("Render Mode:");
        int renderModeInt = static_cast<int>(mAppRenderMode);
        if (ImGui::RadioButton("Primitive", renderModeInt == 0)) mAppRenderMode = MotionEditorRenderMode::Primitive;
        ImGui::SameLine();
        if (ImGui::RadioButton("Wire", renderModeInt == 1)) mAppRenderMode = MotionEditorRenderMode::Wireframe;
    }
}

void MotionEditorApp::drawPlaybackSection()
{
    if (collapsingHeaderWithControls("Playback")) {
        // Play/Pause buttons
        if (mIsPlaying) {
            if (ImGui::Button("Pause")) mIsPlaying = false;
        } else {
            if (ImGui::Button("Play")) mIsPlaying = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            resetPlayback();
        }

        // Speed control
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("Speed", &mPlaybackSpeed, 0.1f, 3.0f, "%.1fx");

        // Frame slider
        if (mMotion) {
            int totalFrames = mMotion->getNumFrames();
            int currentFrame = mMotionState.manualFrameIndex;

            ImGui::SetNextItemWidth(-1);
            if (ImGui::SliderInt("Frame", &currentFrame, 0, totalFrames - 1)) {
                mMotionState.manualFrameIndex = currentFrame;
                mMotionState.navigationMode = ME_MANUAL_FRAME;
                mIsPlaying = false;
            }

            // Navigation mode toggle
            bool syncMode = (mMotionState.navigationMode == ME_SYNC);
            if (ImGui::Checkbox("Auto Sync", &syncMode)) {
                mMotionState.navigationMode = syncMode ? ME_SYNC : ME_MANUAL_FRAME;
            }
        }

        // Camera follow
        bool follow = (mCamera.focus == CameraFocusMode::FOLLOW_CHARACTER);
        if (ImGui::Checkbox("Camera Follow", &follow)) {
            mCamera.focus = follow ? CameraFocusMode::FOLLOW_CHARACTER : CameraFocusMode::FREE;
        }
    }
}

void MotionEditorApp::drawMotionInfoSection()
{
    if (collapsingHeaderWithControls("Motion Info")) {
        if (mMotion) {
            ImGui::Text("Frames: %d", mMotion->getNumFrames());
            ImGui::Text("FPS: %.0f", 1.0 / mMotion->getFrameTime());
            ImGui::Text("DOF: %d", mMotion->getValuesPerFrame());
            ImGui::Text("Duration: %.2f s", mMotion->getMaxTime());
            ImGui::TextWrapped("Source: %s", mMotionSourcePath.c_str());
        } else {
            ImGui::Text("No motion loaded");
        }
    }
}

void MotionEditorApp::drawTrimSection()
{
    if (collapsingHeaderWithControls("Trim")) {
        if (!mMotion) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a motion first");
            return;
        }

        int totalFrames = mMotion->getNumFrames();

        // Start frame slider
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderInt("Start Frame", &mTrimStart, 0, totalFrames - 1);

        // End frame slider
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderInt("End Frame", &mTrimEnd, 0, totalFrames - 1);

        // Ensure start <= end
        if (mTrimStart > mTrimEnd) {
            mTrimStart = mTrimEnd;
        }

        // Quick set buttons
        if (ImGui::Button("Set Start to Current")) {
            mTrimStart = mMotionState.manualFrameIndex;
        }
        ImGui::SameLine();
        if (ImGui::Button("Set End to Current")) {
            mTrimEnd = mMotionState.manualFrameIndex;
        }

        if (ImGui::Button("Reset Trim")) {
            mTrimStart = 0;
            mTrimEnd = totalFrames - 1;
        }
        ImGui::SameLine();

        // Auto Trim button - trim from first right heel strike to frame before last right heel strike
        // Example: contacts at 10-30, 100-120, 493-526 → trim to frames 10-492
        if (ImGui::Button("Auto Trim (R)")) {
            int firstRightStart = INT_MAX;
            int lastRightStart = -1;

            for (const auto& phase : mDetectedPhases) {
                if (!phase.isLeft) {  // Right foot contact
                    if (phase.startFrame < firstRightStart) {
                        firstRightStart = phase.startFrame;
                    }
                    if (phase.startFrame > lastRightStart) {
                        lastRightStart = phase.startFrame;
                    }
                }
            }

            // Trim end is one frame before the last right heel strike
            int trimEnd = lastRightStart - 1;

            if (firstRightStart != INT_MAX && trimEnd > firstRightStart) {
                mTrimStart = firstRightStart;
                mTrimEnd = trimEnd;
                applyTrim();
            } else {
                LOG_WARN("[MotionEditor] Auto Trim: Need at least 2 right heel strikes");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply Trim")) {
            applyTrim();
        }

        // Trim info
        int trimmedFrames = mTrimEnd - mTrimStart + 1;
        ImGui::Text("Trimmed: %d frames (%.2f s)", trimmedFrames, trimmedFrames * mMotion->getFrameTime());
    }
}

void MotionEditorApp::drawExportSection()
{
    if (collapsingHeaderWithControls("Export")) {
        if (!mMotion) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a motion first");
            return;
        }

        // Filename input
        ImGui::Text("Filename:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ExportFilename", mExportFilename, sizeof(mExportFilename));

        // Auto suffix checkbox
        ImGui::Checkbox("Auto-suffix \"_edited\"", &mAutoSuffix);

        // Preview output filename
        std::string filename = strlen(mExportFilename) > 0
            ? mExportFilename
            : fs::path(mMotionSourcePath).stem().string();
        if (mAutoSuffix) filename += "_edited";
        filename += ".h5";

        fs::path outputPath = fs::path(mMotionSourcePath).parent_path() / filename;
        bool fileExists = fs::exists(outputPath);

        ImGui::TextWrapped("Output: %s", filename.c_str());

        // Warn if file already exists
        if (fileExists) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Warning: File already exists!");
        }

        // Export button
        if (ImGui::Button("Export Edited Motion", ImVec2(-1, 30))) {
            exportMotion();
        }

        // Export status message
        if (!mLastExportMessage.empty()) {
            double elapsed = glfwGetTime() - mLastExportMessageTime;
            if (elapsed < 10.0) {
                bool success = mLastExportMessage.find("Success") != std::string::npos;
                ImVec4 color = success ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f) : ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
                ImGui::TextColored(color, "%s", mLastExportMessage.c_str());

                // Show URI for successful exports
                if (success && !mLastExportURI.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "URI: %s", mLastExportURI.c_str());
                }
            }
        }
    }
}

bool MotionEditorApp::collapsingHeaderWithControls(const std::string& title)
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
    if (!isPanelDefaultOpen(title)) {
        flags = 0;
    }
    return ImGui::CollapsingHeader(title.c_str(), flags);
}

void MotionEditorApp::drawRotationSection()
{
    if (collapsingHeaderWithControls("Rotation")) {
        if (!mMotion) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a motion first");
            return;
        }

        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##AngleSlider", &mPendingRotationAngle, -180.0f, 180.0f, "%.1f");

        // Row: Reset, -90, +90, float input
        if (ImGui::Button("Reset##Rotation")) {
            mPendingRotationAngle = 0.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("-90")) {
            mPendingRotationAngle = std::fmod(mPendingRotationAngle - 90.0f + 360.0f, 360.0f);
            if (mPendingRotationAngle > 180.0f) mPendingRotationAngle -= 360.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("+90")) {
            mPendingRotationAngle = std::fmod(mPendingRotationAngle + 90.0f + 360.0f, 360.0f);
            if (mPendingRotationAngle > 180.0f) mPendingRotationAngle -= 360.0f;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputFloat("##AngleInput", &mPendingRotationAngle, 0.0f, 0.0f, "%.1f");

        if (ImGui::Button("Apply Rotation", ImVec2(-1, 0))) {
            applyRotation();
        }

        // Info text
        if (std::abs(mPendingRotationAngle) > 0.001f) {
            ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.3f, 1.0f), "Preview shown (orange)");
        }
    }
}

void MotionEditorApp::drawHeightSection()
{
    if (collapsingHeaderWithControls("Height Adjustment")) {
        if (!mMotion) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a motion first");
            return;
        }

        if (!mCharacter) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Load a skeleton to calculate");
        }

        // Display computed offset
        if (mHeightOffsetComputed) {
            ImGui::Text("Computed offset: %.4f m", mComputedHeightOffset);
            ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.3f, 1.0f), "Preview shown (orange)");
        } else {
            ImGui::Text("Offset: not calculated");
        }

        // Calculate button
        if (ImGui::Button("Calculate Ground Level", ImVec2(-1, 0))) {
            if (mCharacter) {
                computeGroundLevel();
            }
        }

        // Apply button (only enabled if computed)
        bool applyDisabled = !mHeightOffsetComputed;
        if (applyDisabled) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Apply Height Adjustment", ImVec2(-1, 0))) {
            applyHeightOffset();
        }
        if (applyDisabled) {
            ImGui::EndDisabled();
        }

        // Reset button
        if (mHeightOffsetComputed) {
            if (ImGui::Button("Reset##Height")) {
                mComputedHeightOffset = 0.0;
                mHeightOffsetComputed = false;
            }
        }
    }
}

void MotionEditorApp::drawFootContactSection()
{
    if (collapsingHeaderWithControls("Foot Contact")) {
        if (!mMotion) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Load a motion first");
            return;
        }
        if (!mCharacter) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Load a skeleton to detect");
            return;
        }

        // Run button
        if (ImGui::Button("Run Detector", ImVec2(-1, 0))) {
            detectFootContacts();
        }

        // Parameters (hidden by default with TreeNodeEx)
        if (ImGui::TreeNodeEx("Parameters", 0)) {
            ImGui::Text("Velocity Threshold (m)");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputFloat("##VelThreshold", &mContactVelocityThreshold, 0.001f, 0.01f, "%.3f");
            ImGui::Text("Min Lock Frames");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputInt("##MinLockFrames", &mContactMinLockFrames);
            ImGui::TreePop();
        }

        // Results - count left and right separately
        int leftCount = 0, rightCount = 0;
        for (const auto& phase : mDetectedPhases) {
            if (phase.isLeft) leftCount++;
            else rightCount++;
        }

        // Two columns using table: Left | Right
        if (ImGui::BeginTable("##FootContactTable", 2, ImGuiTableFlags_None)) {
            ImGui::TableSetupColumn("Left", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Right", ImGuiTableColumnFlags_WidthStretch);

            // Header row
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Left (%d)", leftCount);
            ImGui::TableNextColumn();
            ImGui::Text("Right (%d)", rightCount);

            // Listbox row
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::BeginListBox("##LeftContact", ImVec2(-1, 100))) {
                for (int i = 0; i < static_cast<int>(mDetectedPhases.size()); ++i) {
                    const auto& phase = mDetectedPhases[i];
                    if (!phase.isLeft) continue;

                    char label[64];
                    snprintf(label, sizeof(label), "%d-%d (%d)##L%d",
                             phase.startFrame, phase.endFrame,
                             phase.endFrame - phase.startFrame + 1, i);

                    bool isSelected = (i == mSelectedPhase);
                    if (ImGui::Selectable(label, isSelected)) {
                        mSelectedPhase = i;
                        mMotionState.manualFrameIndex = phase.startFrame;
                        mMotionState.navigationMode = ME_MANUAL_FRAME;
                        mIsPlaying = false;
                    }
                }
                ImGui::EndListBox();
            }

            ImGui::TableNextColumn();
            if (ImGui::BeginListBox("##RightContact", ImVec2(-1, 100))) {
                for (int i = 0; i < static_cast<int>(mDetectedPhases.size()); ++i) {
                    const auto& phase = mDetectedPhases[i];
                    if (phase.isLeft) continue;

                    char label[64];
                    snprintf(label, sizeof(label), "%d-%d (%d)##R%d",
                             phase.startFrame, phase.endFrame,
                             phase.endFrame - phase.startFrame + 1, i);

                    bool isSelected = (i == mSelectedPhase);
                    if (ImGui::Selectable(label, isSelected)) {
                        mSelectedPhase = i;
                        mMotionState.manualFrameIndex = phase.startFrame;
                        mMotionState.navigationMode = ME_MANUAL_FRAME;
                        mIsPlaying = false;
                    }
                }
                ImGui::EndListBox();
            }

            ImGui::EndTable();
        }
    }
}

void MotionEditorApp::drawStrideEstimationSection()
{
    if (!collapsingHeaderWithControls("Stride Estimation")) return;

    if (!mMotion || !mCharacter) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Load motion and skeleton first");
        return;
    }

    auto skel = mCharacter->getSkeleton();

    // BodyNode selection (default: TalusR)
    const char* bodyNodes[] = {"TalusR", "TalusL", "Pelvis"};
    ImGui::Text("BodyNode:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("##StrideBodyNode", &mStrideBodyNodeIdx, bodyNodes, IM_ARRAYSIZE(bodyNodes));

    // Divider input (auto-computed from foot contacts, but editable)
    ImGui::Text("Divider:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    int dividerInput = mStrideDivider;
    if (ImGui::InputInt("##StrideDivider", &dividerInput)) {
        setStrideDivider(dividerInput);
    }

    // Calculation mode radio
    ImGui::Text("Mode:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Z only", &mStrideCalcMode, 0)) {
        computeStride();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("XZ magnitude", &mStrideCalcMode, 1)) {
        computeStride();
    }

    // Display result
    if (mComputedStride > 0.0) {
        ImGui::Text("Stride: %.4f m", mComputedStride);
    } else {
        ImGui::TextDisabled("Stride: not computed");
    }
}

void MotionEditorApp::detectFootContacts()
{
    if (!mMotion || !mCharacter) return;

    mDetectedPhases.clear();
    mSelectedPhase = -1;

    auto skel = mCharacter->getSkeleton();
    auto talusL = skel->getBodyNode("TalusL");
    auto talusR = skel->getBodyNode("TalusR");
    if (!talusL || !talusR) return;

    int numFrames = mMotion->getNumFrames();
    if (numFrames < 2) return;

    // Save skeleton state
    Eigen::VectorXd savedPositions = skel->getPositions();

    // Compute positions for all frames
    std::vector<Eigen::Vector3d> positionsL(numFrames), positionsR(numFrames);
    for (int f = 0; f < numFrames; ++f) {
        Eigen::VectorXd pose = mMotion->getPose(f);
        if (pose.size() == skel->getNumDofs()) {
            skel->setPositions(pose);
            positionsL[f] = talusL->getTransform().translation();
            positionsR[f] = talusR->getTransform().translation();
        }
    }
    skel->setPositions(savedPositions);

    // Detection lambda
    auto detectPhasesForFoot = [&](const std::vector<Eigen::Vector3d>& positions, bool isLeft) {
        int lockStart = -1;
        for (int f = 1; f < numFrames; ++f) {
            double velocity = (positions[f] - positions[f - 1]).norm();
            if (velocity < mContactVelocityThreshold) {
                if (lockStart < 0) lockStart = f;
            } else {
                if (lockStart >= 0 && (f - lockStart) >= mContactMinLockFrames) {
                    mDetectedPhases.push_back({lockStart, f - 1, isLeft});
                }
                lockStart = -1;
            }
        }
        if (lockStart >= 0 && (numFrames - lockStart) >= mContactMinLockFrames) {
            mDetectedPhases.push_back({lockStart, numFrames - 1, isLeft});
        }
    };

    detectPhasesForFoot(positionsL, true);
    detectPhasesForFoot(positionsR, false);

    // Sort by start frame
    std::sort(mDetectedPhases.begin(), mDetectedPhases.end(),
        [](const Timeline::FootContactPhase& a, const Timeline::FootContactPhase& b) {
            return a.startFrame < b.startFrame;
        });

    // Detect gait direction for each phase based on root body displacement
    if (mDetectedPhases.size() >= 2) {
        mDetectedPhases[0].direction = Timeline::GaitDirection::Unknown;

        for (size_t i = 1; i < mDetectedPhases.size(); ++i) {
            int prevFrame = mDetectedPhases[i - 1].startFrame;
            int currFrame = mDetectedPhases[i].startFrame;

            Eigen::VectorXd prevPose = mMotion->getPose(prevFrame);
            Eigen::VectorXd currPose = mMotion->getPose(currFrame);

            if (prevPose.size() >= 6 && currPose.size() >= 6) {
                double deltaZ = currPose[5] - prevPose[5];  // pose[5] = root Z
                const double threshold = 0.01;  // 1cm minimum displacement

                if (deltaZ > threshold) {
                    mDetectedPhases[i].direction = Timeline::GaitDirection::Forward;
                } else if (deltaZ < -threshold) {
                    mDetectedPhases[i].direction = Timeline::GaitDirection::Backward;
                } else {
                    mDetectedPhases[i].direction = Timeline::GaitDirection::Unknown;
                }
            }
        }
    }

    // Auto-compute stride divider based on contact counts
    int leftContactCount = 0;
    int rightContactCount = 0;
    for (const auto& phase : mDetectedPhases) {
        if (phase.isLeft) {
            leftContactCount++;
        } else {
            rightContactCount++;
        }
    }
    // Use the smaller of LCT and RCT (minimum 1)
    int divider = 1;
    if (leftContactCount > 0 && rightContactCount > 0) {
        divider = std::min(leftContactCount, rightContactCount);
    } else if (leftContactCount > 0) {
        divider = leftContactCount;
    } else if (rightContactCount > 0) {
        divider = rightContactCount;
    }
    setStrideDivider(divider);
}

void MotionEditorApp::setStrideDivider(int divider)
{
    mStrideDivider = std::max(1, divider);
    computeStride();
}

void MotionEditorApp::computeStride()
{
    if (!mMotion || !mCharacter) {
        mComputedStride = 0.0;
        return;
    }

    auto skel = mCharacter->getSkeleton();
    const char* bodyNodes[] = {"TalusR", "TalusL", "Pelvis"};
    std::string bnName = bodyNodes[mStrideBodyNodeIdx];
    auto bn = skel->getBodyNode(bnName);
    if (!bn) {
        mComputedStride = 0.0;
        return;
    }

    int numFrames = mMotion->getNumFrames();
    if (numFrames < 2) {
        mComputedStride = 0.0;
        return;
    }

    // Save skeleton state
    Eigen::VectorXd savedPositions = skel->getPositions();

    // Get first frame position
    skel->setPositions(mMotion->getPose(0));
    Eigen::Vector3d startPos = bn->getTransform().translation();

    // Get last frame position
    skel->setPositions(mMotion->getPose(numFrames - 1));
    Eigen::Vector3d endPos = bn->getTransform().translation();

    // Restore skeleton state
    skel->setPositions(savedPositions);

    // Compute stride based on mode
    if (mStrideCalcMode == 0) {
        // Z only (forward)
        mComputedStride = std::abs(endPos[2] - startPos[2]) / mStrideDivider;
    } else {
        // XZ magnitude
        double dx = endPos[0] - startPos[0];
        double dz = endPos[2] - startPos[2];
        mComputedStride = std::sqrt(dx*dx + dz*dz) / mStrideDivider;
    }
}

Eigen::Vector4d MotionEditorApp::getRenderColor(
    const dart::dynamics::BodyNode* bn,
    const Eigen::Vector4d& defaultColor) const
{
    if (mDetectedPhases.empty() || !bn) {
        return defaultColor;
    }

    int frameIdx = mMotionState.manualFrameIndex;
    std::string name = bn->getName();

    // Check if this body node should be highlighted
    for (const auto& phase : mDetectedPhases) {
        if (frameIdx >= phase.startFrame && frameIdx <= phase.endFrame) {
            if ((phase.isLeft && name == "TalusL") ||
                (!phase.isLeft && name == "TalusR")) {
                return Eigen::Vector4d(0.2, 0.9, 0.3, 1.0);  // Green for contact
            }
        }
    }

    return defaultColor;
}

// =============================================================================
// ROM Violation Detection
// =============================================================================

void MotionEditorApp::detectROMViolations()
{
    mROMViolations.clear();
    mSelectedViolation = -1;

    if (!mMotion || !mCharacter) return;

    auto skel = mCharacter->getSkeleton();
    Eigen::VectorXd rom_min = skel->getPositionLowerLimits();
    Eigen::VectorXd rom_max = skel->getPositionUpperLimits();
    int numFrames = mMotion->getNumFrames();
    int numDofs = skel->getNumDofs();

    // Skip first 6 DOFs (root FreeJoint)
    for (int dofIdx = 6; dofIdx < numDofs; ++dofIdx) {
        int violationStart = -1;
        int maxDiffFrame = -1;
        double maxDiff = 0.0;
        double maxAngle = 0.0;
        double boundHit = 0.0;
        bool isUpper = false;

        for (int f = 0; f < numFrames; ++f) {
            Eigen::VectorXd pose = mMotion->getPose(f);
            double val = pose[dofIdx];
            double diffLower = rom_min[dofIdx] - val;  // positive if below min
            double diffUpper = val - rom_max[dofIdx];  // positive if above max

            bool inViolation = (diffLower > 0 || diffUpper > 0);

            if (inViolation) {
                if (violationStart < 0) violationStart = f;

                double diff = std::max(diffLower, diffUpper);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    maxDiffFrame = f;
                    maxAngle = val;
                    isUpper = (diffUpper > diffLower);
                    boundHit = isUpper ? rom_max[dofIdx] : rom_min[dofIdx];
                }
            } else if (violationStart >= 0) {
                // End of violation range
                auto dof = skel->getDof(dofIdx);
                auto joint = dof->getJoint();
                std::string jointName = joint->getName();
                int localDofIdx = dof->getIndexInJoint();
                int numJointDofs = joint->getNumDofs();
                mROMViolations.push_back({
                    jointName, dofIdx, localDofIdx, numJointDofs,
                    violationStart, maxDiffFrame, f - 1,
                    maxAngle, boundHit, isUpper
                });
                violationStart = -1;
                maxDiff = 0.0;
            }
        }

        // Handle violation at end of motion
        if (violationStart >= 0) {
            auto dof = skel->getDof(dofIdx);
            auto joint = dof->getJoint();
            std::string jointName = joint->getName();
            int localDofIdx = dof->getIndexInJoint();
            int numJointDofs = joint->getNumDofs();
            mROMViolations.push_back({
                jointName, dofIdx, localDofIdx, numJointDofs,
                violationStart, maxDiffFrame, numFrames - 1,
                maxAngle, boundHit, isUpper
            });
        }
    }

    // Sort by start frame
    std::sort(mROMViolations.begin(), mROMViolations.end(),
        [](const ROMViolation& a, const ROMViolation& b) {
            return a.startFrame < b.startFrame;
        });

    LOG_INFO("[MotionEditor] Detected " << mROMViolations.size() << " ROM violations");
}

void MotionEditorApp::drawROMViolationSection()
{
    if (!collapsingHeaderWithControls("ROM Violations")) return;

    if (!mMotion || !mCharacter) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Load motion and skeleton first");
        return;
    }

    // Detect button
    if (ImGui::Button("Detect Violations", ImVec2(-1, 0))) {
        detectROMViolations();
    }

    // Preview checkbox
    ImGui::Checkbox("Preview Clamped Pose", &mPreviewClampedPose);

    // Violation count
    ImGui::Text("Violations: %zu", mROMViolations.size());

    // No violations message
    if (mROMViolations.empty()) {
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "No ROM violations detected");
        return;
    }

    // Violation listbox
    if (ImGui::BeginListBox("##ROMViolations", ImVec2(-1, 150))) {
        for (int i = 0; i < static_cast<int>(mROMViolations.size()); ++i) {
            const auto& v = mROMViolations[i];

            char label[128];
            const char* op = v.isUpperBound ? ">" : "<";
            double angleDeg = v.maxAngle * 180.0 / M_PI;
            double boundDeg = v.boundValue * 180.0 / M_PI;
            // Show local DOF index for multi-DOF joints (e.g., "HipR[2]")
            std::string jointDisplay = v.jointName;
            if (v.numJointDofs > 1) {
                jointDisplay += "[" + std::to_string(v.localDofIndex) + "]";
            }
            snprintf(label, sizeof(label), "%s (%d-%d-%d): %.1f° %s %.1f°##V%d",
                     jointDisplay.c_str(), v.startFrame, v.maxDiffFrame, v.endFrame,
                     angleDeg, op, boundDeg, i);

            bool isSelected = (i == mSelectedViolation);
            if (ImGui::Selectable(label, isSelected)) {
                mSelectedViolation = i;
                mMotionState.manualFrameIndex = v.startFrame;
                mMotionState.navigationMode = ME_MANUAL_FRAME;
                mIsPlaying = false;
            }
        }
        ImGui::EndListBox();
    }
}

// =============================================================================
// Data Loading
// =============================================================================

void MotionEditorApp::scanSkeletonDirectory()
{
    mSkeletonFiles.clear();
    mSkeletonFileNames.clear();
    mSelectedSkeletonFile = -1;

    if (!mResourceManager || !mPIDNavigator) {
        return;
    }

    const auto& state = mPIDNavigator->getState();
    if (state.selectedPID < 0 || state.selectedPID >= static_cast<int>(state.pidList.size())) {
        return;
    }

    const std::string& pid = state.pidList[state.selectedPID];
    std::string visit = state.preOp ? "pre" : "op1";
    std::string pattern = "@pid:" + pid + "/" + visit + "/skeleton";
    mSkeletonDirectory = pattern;

    try {
        auto files = mResourceManager->list(pattern);
        for (const auto& file : files) {
            // Only include .yaml and .xml skeleton files
            size_t len = file.size();
            bool isYaml = len > 5 && file.substr(len - 5) == ".yaml";
            bool isXml = len > 4 && file.substr(len - 4) == ".xml";
            if (isYaml || isXml) {
                // Resolve full path
                std::string uri = pattern + "/" + file;
                try {
                    auto resolved = mResourceManager->resolve(uri);
                    if (!resolved.empty() && fs::exists(resolved)) {
                        mSkeletonFiles.push_back(resolved.string());
                        mSkeletonFileNames.push_back(file);
                    }
                } catch (const rm::RMError&) {}
            }
        }
        std::sort(mSkeletonFileNames.begin(), mSkeletonFileNames.end());
        // Sort files to match names
        std::vector<size_t> indices(mSkeletonFiles.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
            return mSkeletonFileNames[a] < mSkeletonFileNames[b];
        });
        std::vector<std::string> sortedFiles, sortedNames;
        for (size_t i : indices) {
            sortedFiles.push_back(mSkeletonFiles[i]);
            sortedNames.push_back(mSkeletonFileNames[i]);
        }
        mSkeletonFiles = std::move(sortedFiles);
        mSkeletonFileNames = std::move(sortedNames);

        LOG_INFO("[MotionEditor] Found " << mSkeletonFiles.size() << " skeleton files in " << pattern);
    } catch (const rm::RMError& e) {
        LOG_WARN("[MotionEditor] Failed to scan skeleton directory: " << e.what());
    }
}

void MotionEditorApp::loadH5Motion(const std::string& path)
{
    // Unload existing motion
    if (mMotion) {
        delete mMotion;
        mMotion = nullptr;
    }

    try {
        mMotion = new HDF(path);
        mMotionSourcePath = path;

        // Prefill export filename with source file stem
        std::string stem = fs::path(path).stem().string();
        strncpy(mExportFilename, stem.c_str(), sizeof(mExportFilename) - 1);
        mExportFilename[sizeof(mExportFilename) - 1] = '\0';

        // Load skeleton if we have one auto-detected or manually set
        std::string skelPath = mUseAutoSkeleton ? mAutoDetectedSkeletonPath : std::string(mManualSkeletonPath);
        if (!skelPath.empty() && !mCharacter) {
            loadSkeleton(skelPath);
        }

        // Refresh motion state (trim, playback, foot contacts, ROM)
        refreshMotion();

        LOG_INFO("[MotionEditor] Loaded motion: " << path << " (" << mMotion->getNumFrames() << " frames)");
    } catch (const std::exception& e) {
        LOG_ERROR("[MotionEditor] Failed to load motion: " << e.what());
        mMotion = nullptr;
    }
}

void MotionEditorApp::refreshMotion()
{
    if (!mMotion) return;

    // Reset trim bounds to current motion range
    mTrimStart = 0;
    mTrimEnd = mMotion->getNumFrames() - 1;

    // Reset playback state
    mMotionState = MotionEditorViewerState();
    mMotionState.maxFrameIndex = mMotion->getNumFrames() - 1;
    mCycleDuration = mMotion->getMaxTime();

    // Reset viewer time
    mViewerTime = 0.0;
    mViewerPhase = 0.0;

    // Reset foot contact phases
    mDetectedPhases.clear();
    mSelectedPhase = -1;

    // Reset ROM violations
    mROMViolations.clear();
    mSelectedViolation = -1;

    // Auto-detect foot contacts if skeleton loaded
    if (mCharacter) {
        detectFootContacts();
        detectROMViolations();
    }
}

void MotionEditorApp::autoDetectSkeleton()
{
    mAutoDetectedSkeletonPath.clear();

    if (!mResourceManager || !mPIDNavigator) {
        return;
    }

    const auto& state = mPIDNavigator->getState();
    if (state.selectedPID < 0 || state.selectedPID >= static_cast<int>(state.pidList.size())) {
        return;
    }

    const std::string& pid = state.pidList[state.selectedPID];
    std::string visit = state.preOp ? "pre" : "op1";
    std::string uri = "@pid:" + pid + "/" + visit + "/skeleton/dynamic.yaml";

    try {
        auto resolved = mResourceManager->resolve(uri);
        if (!resolved.empty() && fs::exists(resolved)) {
            mAutoDetectedSkeletonPath = resolved.string();
            LOG_INFO("[MotionEditor] Auto-detected skeleton: " << mAutoDetectedSkeletonPath);

            // Load if using auto-detect
            if (mUseAutoSkeleton) {
                loadSkeleton(mAutoDetectedSkeletonPath);
            }
        }
    } catch (const rm::RMError&) {
        LOG_VERBOSE("[MotionEditor] No skeleton found at " << uri);
    }
}

void MotionEditorApp::loadSkeleton(const std::string& path)
{
    try {
        mCharacter = std::make_unique<RenderCharacter>(path, SKEL_COLLIDE_ALL);
        mCurrentSkeletonPath = path;

        LOG_INFO("[MotionEditor] Loaded skeleton: " << path);
    } catch (const std::exception& e) {
        LOG_ERROR("[MotionEditor] Failed to load skeleton: " << e.what());
        mCharacter.reset();
    }
}

// =============================================================================
// Playback
// =============================================================================

void MotionEditorApp::updateViewerTime(double dt)
{
    if (mCycleDuration <= 0.0) return;

    mViewerTime += dt;

    // Wrap time to cycle
    while (mViewerTime >= mCycleDuration) {
        mViewerTime -= mCycleDuration;
    }
    while (mViewerTime < 0.0) {
        mViewerTime += mCycleDuration;
    }

    mViewerPhase = mViewerTime / mCycleDuration;

    // Update frame index for sync mode
    if (mMotion && mMotionState.navigationMode == ME_SYNC) {
        double frameFloat = mViewerPhase * (mMotion->getNumFrames() - 1);
        int newFrame = static_cast<int>(frameFloat);
        newFrame = std::clamp(newFrame, 0, mMotion->getNumFrames() - 1);

        // Update cycle accumulation on wrap
        if (newFrame < mMotionState.lastFrameIdx && mMotionState.lastFrameIdx > mMotion->getNumFrames() / 2) {
            mMotionState.cycleAccumulation += mMotionState.cycleDistance;
        }
        mMotionState.lastFrameIdx = newFrame;
        mMotionState.manualFrameIndex = newFrame;
    }
}

void MotionEditorApp::evaluateMotionPose()
{
    if (!mMotion || !mCharacter) return;

    int frameIdx = mMotionState.manualFrameIndex;
    frameIdx = std::clamp(frameIdx, 0, mMotion->getNumFrames() - 1);

    Eigen::VectorXd pose = mMotion->getPose(frameIdx);
    mMotionState.currentPose = pose;

    // Apply pose to skeleton
    auto skel = mCharacter->getSkeleton();
    if (skel && pose.size() == skel->getNumDofs()) {
        skel->setPositions(pose);
    }
}

Eigen::Vector3d MotionEditorApp::computeMotionCycleDistance()
{
    if (!mMotion) return Eigen::Vector3d::Zero();

    int totalFrames = mMotion->getNumFrames();
    if (totalFrames < 2) return Eigen::Vector3d::Zero();

    Eigen::VectorXd firstPose = mMotion->getPose(0);
    Eigen::VectorXd lastPose = mMotion->getPose(totalFrames - 1);

    if (firstPose.size() >= 6 && lastPose.size() >= 6) {
        return Eigen::Vector3d(
            lastPose[3] - firstPose[3],
            0.0,  // Ignore Y (height)
            lastPose[5] - firstPose[5]
        );
    }
    return Eigen::Vector3d::Zero();
}

// =============================================================================
// Export
// =============================================================================

void MotionEditorApp::exportMotion()
{
    if (!mMotion) {
        mLastExportMessage = "Error: No motion loaded";
        mLastExportMessageTime = glfwGetTime();
        return;
    }

    HDF* hdf = dynamic_cast<HDF*>(mMotion);
    if (!hdf) {
        mLastExportMessage = "Error: Motion is not HDF format";
        mLastExportMessageTime = glfwGetTime();
        return;
    }

    // Build output filename
    std::string filename = strlen(mExportFilename) > 0
        ? mExportFilename
        : fs::path(mMotionSourcePath).stem().string();
    if (mAutoSuffix) filename += "_trimmed";

    fs::path outputPath = fs::path(mMotionSourcePath).parent_path() / (filename + ".h5");

    // Build metadata
    std::map<std::string, std::string> metadata = {
        {"source_type", "motion_editor"},
        {"source_file", fs::path(mMotionSourcePath).filename().string()},
        {"trim_start", std::to_string(mTrimStart)},
        {"trim_end", std::to_string(mTrimEnd)}
    };

    // Only add stride if computed (> 0)
    if (mComputedStride > 0.0) {
        metadata["stride"] = std::to_string(mComputedStride);
    }

    try {
        hdf->exportToFile(outputPath.string(), metadata);
        mLastExportMessage = "Success: Exported to " + outputPath.filename().string();
        mLastExportMessageTime = glfwGetTime();

        // Build PID-style URI if PID is selected
        if (mPIDNavigator) {
            const auto& state = mPIDNavigator->getState();
            if (state.selectedPID >= 0 && state.selectedPID < static_cast<int>(state.pidList.size())) {
                std::string pid = state.pidList[state.selectedPID];
                std::string visit = state.preOp ? "pre" : "op1";
                mLastExportURI = "@pid:" + pid + "/" + visit + "/motion/" + outputPath.filename().string();
            } else {
                mLastExportURI = outputPath.string();
            }
        } else {
            mLastExportURI = outputPath.string();
        }

        LOG_INFO("[MotionEditor] Exported to: " << outputPath);
        LOG_INFO("[MotionEditor] URI: " << mLastExportURI);
    } catch (const std::exception& e) {
        mLastExportMessage = "Error: " + std::string(e.what());
        mLastExportURI.clear();
        mLastExportMessageTime = glfwGetTime();
        LOG_ERROR("[MotionEditor] Export failed: " << e.what());
    }
}

// =============================================================================
// Trim
// =============================================================================

void MotionEditorApp::applyTrim()
{
    if (!mMotion) return;

    HDF* hdf = dynamic_cast<HDF*>(mMotion);
    if (!hdf) {
        LOG_WARN("[MotionEditor] Cannot trim: motion is not HDF format");
        return;
    }

    hdf->trim(mTrimStart, mTrimEnd);

    // Refresh all motion state after trim
    refreshMotion();
}

// =============================================================================
// Processing
// =============================================================================

Eigen::VectorXd MotionEditorApp::applyRotationToFrame(const Eigen::VectorXd& pose, float angleDegrees)
{
    using namespace dart::dynamics;
    Eigen::VectorXd result = pose;

    // Create Y-axis rotation transform
    double angleRad = angleDegrees * M_PI / 180.0;
    Eigen::Isometry3d yRotation = Eigen::Isometry3d::Identity();
    yRotation.rotate(Eigen::AngleAxisd(angleRad, Eigen::Vector3d::UnitY()));

    // Convert root pose to transform (first 6 DOF)
    Eigen::Isometry3d rootTransform = FreeJoint::convertToTransform(pose.head<6>());

    // Apply world rotation: new_transform = yRotation * rootTransform
    Eigen::Isometry3d newTransform = yRotation * rootTransform;

    // Convert back to pose vector
    result.head<6>() = FreeJoint::convertToPositions(newTransform);

    return result;
}

Eigen::Vector3d MotionEditorApp::getBodyNodeSize(dart::dynamics::BodyNode* bn)
{
    using namespace dart::dynamics;

    if (!bn || bn->getNumShapeNodes() == 0) {
        return Eigen::Vector3d(0.1, 0.1, 0.1);  // Default fallback
    }

    auto shape = bn->getShapeNode(0)->getShape();

    if (auto box = std::dynamic_pointer_cast<BoxShape>(shape)) {
        return box->getSize();
    } else if (auto sphere = std::dynamic_pointer_cast<SphereShape>(shape)) {
        double r = sphere->getRadius();
        return Eigen::Vector3d(r * 2, r * 2, r * 2);
    } else if (auto capsule = std::dynamic_pointer_cast<CapsuleShape>(shape)) {
        double r = capsule->getRadius();
        double h = capsule->getHeight();
        return Eigen::Vector3d(r * 2, h + r * 2, r * 2);  // Total height includes caps
    } else if (auto cylinder = std::dynamic_pointer_cast<CylinderShape>(shape)) {
        double r = cylinder->getRadius();
        double h = cylinder->getHeight();
        return Eigen::Vector3d(r * 2, h, r * 2);
    }

    return Eigen::Vector3d(0.1, 0.1, 0.1);  // Fallback
}

void MotionEditorApp::computeGroundLevel()
{
    if (!mMotion || !mCharacter) {
        LOG_WARN("[MotionEditor] Cannot compute ground level: no motion or skeleton");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    // Save current skeleton positions
    Eigen::VectorXd savedPositions = skel->getPositions();

    int numFrames = mMotion->getNumFrames();
    double totalMinY = 0.0;

    // Iterate through all frames
    for (int frame = 0; frame < numFrames; ++frame) {
        Eigen::VectorXd pose = mMotion->getPose(frame);
        if (pose.size() != skel->getNumDofs()) continue;

        skel->setPositions(pose);

        // Find minimum Y for this frame
        double frameMinY = std::numeric_limits<double>::max();

        for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
            dart::dynamics::BodyNode* bn = skel->getBodyNode(i);
            if (!bn) continue;

            // Get body node world transform
            Eigen::Isometry3d transform = bn->getTransform();
            Eigen::Vector3d pos = transform.translation();

            // Get shape size to compute bottom Y
            Eigen::Vector3d size = getBodyNodeSize(bn);

            // Compute lowest Y point (assuming Y is up, shape centered at origin)
            double bottomY = pos[1] - size[1] / 2.0;

            if (bottomY < frameMinY) {
                frameMinY = bottomY;
            }
        }

        if (frameMinY < std::numeric_limits<double>::max()) {
            totalMinY += frameMinY;
        }
    }

    // Compute mean ground level
    double meanGroundLevel = totalMinY / numFrames;

    // Offset to bring to Y=0
    mComputedHeightOffset = -meanGroundLevel;
    mHeightOffsetComputed = true;

    // Restore skeleton positions
    skel->setPositions(savedPositions);

    LOG_INFO("[MotionEditor] Computed ground level: " << meanGroundLevel
             << ", offset: " << mComputedHeightOffset);
}

void MotionEditorApp::applyRotation()
{
    if (!mMotion || std::abs(mPendingRotationAngle) < 0.001f) {
        return;
    }

    HDF* hdf = dynamic_cast<HDF*>(mMotion);
    if (!hdf) {
        LOG_WARN("[MotionEditor] Cannot apply rotation: motion is not HDF format");
        return;
    }

    // Apply rotation via HDF method
    hdf->applyYRotation(mPendingRotationAngle);

    // Reset preview angle
    mPendingRotationAngle = 0.0f;
}

void MotionEditorApp::applyHeightOffset()
{
    if (!mMotion || !mHeightOffsetComputed || std::abs(mComputedHeightOffset) < 0.001) {
        return;
    }

    HDF* hdf = dynamic_cast<HDF*>(mMotion);
    if (!hdf) {
        LOG_WARN("[MotionEditor] Cannot apply height offset: motion is not HDF format");
        return;
    }

    // Apply height offset via HDF method
    hdf->applyHeightOffset(mComputedHeightOffset);

    // Reset height state
    mComputedHeightOffset = 0.0;
    mHeightOffsetComputed = false;
}

// =============================================================================
// App-specific Helpers
// =============================================================================

void MotionEditorApp::alignCameraToPlaneQuat(int plane)
{
    Eigen::Quaterniond quat;
    switch (plane) {
        case 1: // XY plane - view from +Z
            quat = Eigen::Quaterniond::Identity();
            break;
        case 2: // YZ plane - view from +X
            quat = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY());
            break;
        case 3: // ZX plane - view from +Y (top-down)
            quat = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
            break;
    }
    mCamera.trackball.setQuaternion(quat);
}

void MotionEditorApp::resetPlayback()
{
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mMotionState.lastFrameIdx = 0;
    mMotionState.cycleAccumulation.setZero();
    mMotionState.manualFrameIndex = 0;
    mIsPlaying = false;
}

void MotionEditorApp::drawTimelineTrackBar()
{
    Timeline::Config config;
    config.height = 80.0f;
    config.showTrimMarkers = (mTrimStart > 0 || mTrimEnd < (mMotion ? mMotion->getNumFrames() - 1 : 0));
    config.trimStart = mTrimStart;
    config.trimEnd = mTrimEnd;
    config.zoom = &mTimelineZoom;
    config.scrollOffset = &mTimelineScrollOffset;

    auto result = Timeline::DrawTimelineTrackBar(
        mWidth, mHeight,
        mMotion ? mMotion->getNumFrames() : 0,
        mMotionState.manualFrameIndex,
        mDetectedPhases,
        mViewerTime, mIsPlaying,
        config
    );

    if (result.scrubbed) {
        mMotionState.navigationMode = ME_MANUAL_FRAME;
        mMotionState.manualFrameIndex = result.targetFrame;
        mIsPlaying = false;
    }
}

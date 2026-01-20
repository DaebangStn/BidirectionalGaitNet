#include "C3DProcessorApp.h"
#include "DARTHelper.h"
#include "rm/rm.hpp"
#include <rm/global.hpp>
#include "Log.h"
#include "PlotUtils.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>

using namespace dart::dynamics;
namespace fs = std::filesystem;

// Helper function for outlined text rendering (black edge + white fill)
static void drawOutlinedText(ImDrawList* drawList, ImFont* font, float fontSize,
                             const ImVec2& pos, const std::string& text,
                             ImU32 fillColor = IM_COL32(255, 255, 255, 255),
                             ImU32 outlineColor = IM_COL32(0, 0, 0, 255))
{
    // Draw outline (8 directions for clean edge)
    const float offset = 1.0f;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            drawList->AddText(font, fontSize,
                ImVec2(pos.x + dx * offset, pos.y + dy * offset),
                outlineColor, text.c_str());
        }
    }
    // Draw fill
    drawList->AddText(font, fontSize, pos, fillColor, text.c_str());
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

C3DProcessorApp::C3DProcessorApp(const std::string& configPath)
    : ViewerAppBase("C3D Processor", 1280, 720)
    , mConfigPath(configPath)
    , mC3DReader(nullptr)
    , mMotion(nullptr)
    , mRenderC3DMarkers(true)
    , mRenderExpectedMarkers(true)
    , mRenderMarkerIndices(false)
    , mViewerTime(0.0)
    , mViewerPhase(0.0)
    , mViewerPlaybackSpeed(1.0f)
    , mViewerCycleDuration(2.0 / 1.1)
    , mLastRealTime(0.0)
    , mIsPlaying(false)
    , mXmin(-10.0)
    , mPlotHideLegend(false)
{
    std::memset(mMarkerSearchFilter, 0, sizeof(mMarkerSearchFilter));

    // Load configuration from YAML file (sets paths and rendering options)
    loadConfig();

    // Initialize graph data buffer
    mGraphData = new CBufferData<double>();
    mGraphData->register_key("marker_error_mean", 1000);

    // C3D COM init
    mC3DCOM = Eigen::Vector3d::Zero();

    // Initialize ImPlot for this app
    ImPlot::CreateContext();

    // Load skeleton and create character
    // SKEL_FREE_JOINTS: All joints are FreeJoint (6 DOF) for debugging bone poses independently
    mFreeCharacter = std::make_unique<RenderCharacter>(mSkeletonPath, SKEL_COLLIDE_ALL | SKEL_FREE_JOINTS);
    mMotionCharacter = std::make_unique<RenderCharacter>(mSkeletonPath, SKEL_COLLIDE_ALL);
    RenderCharacter* characters[2] = { mFreeCharacter.get(), mMotionCharacter.get() };
    for (auto character : characters) {
        if (character) {
            character->loadMarkers(mMarkerConfigPath);
        }
    }

    // Create C3D Reader
    mC3DReader = new C3D_Reader(mFittingConfigPath, mMarkerConfigPath, mFreeCharacter.get(), mMotionCharacter.get());

    // Initialize Resource Manager for PID-based access (use singleton)
    // Must be done before scanC3DFiles() which uses mResourceManager
    try {
        mResourceManager = &rm::getManager();

        // Initialize PID Navigator with C3D filter
        mPIDNavigator = std::make_unique<PIDNav::PIDNavigator>(
            mResourceManager,
            std::make_unique<PIDNav::C3DFileFilter>()
        );

        // Register file selection callback
        mPIDNavigator->setFileSelectionCallback(
            [this](const std::string& path, const std::string& filename) {
                onPIDFileSelected(path, filename);
            }
        );

        // Register PID change callback for calibration checking
        mPIDNavigator->setPIDChangeCallback(
            [this](const std::string& pid) {
                checkForPersonalizedCalibration();
            }
        );

        // Initial PID scan
        mPIDNavigator->scanPIDs();
    } catch (const rm::RMError& e) {
        LOG_WARN("[C3DProcessor] Resource manager init failed: " << e.what());
    } catch (const std::exception& e) {
        LOG_WARN("[C3DProcessor] Resource manager init failed: " << e.what());
    }

    // Scan for C3D files and autoload first one (if enabled)
    scanC3DFiles();
    if (mAutoloadFirstC3D && !mMotionList.empty()) {
        mSelectedMotion = 0;
        loadC3DFile(mMotionList[0]);
        LOG_INFO("[C3DProcessor] Autoloaded first C3D file: " << mMotionList[0]);
    }

    mLastRealTime = glfwGetTime();

    // Reset to initial state
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mMotionState.lastFrameIdx = 0;
    mMotionState.cycleAccumulation.setZero();
    mMotionState.manualFrameIndex = 0;
    mGraphData->clear_all();

    // Refresh marker and skeleton data if motion was loaded
    if (mMotion) {
        MarkerPlaybackContext context = computeMarkerPlayback();
        evaluateMarkerPlayback(context);
    }

    LOG_INFO("[C3DProcessor] Initialized with skeleton: " << mSkeletonPath << ", markers: " << mMarkerConfigPath);
}

C3DProcessorApp::~C3DProcessorApp()
{
    if (mMotion) {
        delete mMotion;
        mMotion = nullptr;
    }
    if (mC3DReader) {
        delete mC3DReader;
        mC3DReader = nullptr;
    }
    if (mGraphData) {
        delete mGraphData;
        mGraphData = nullptr;
    }

    ImPlot::DestroyContext();
    // Base class handles GLFW/ImGui cleanup
}

void C3DProcessorApp::loadConfig()
{
    // Load all configuration from the unified YAML file
    std::string resolved_path = rm::resolve(mConfigPath);
    LOG_INFO("[C3DProcessor] Loading config from: " << resolved_path);

    try {
        YAML::Node config = YAML::LoadFile(resolved_path);

        // Character configuration
        if (config["character"]) {
            auto character = config["character"];
            mSkeletonPath = character["skeleton"].as<std::string>();
            mMarkerConfigPath = character["marker"].as<std::string>();
            mInitialMarkerPath = mMarkerConfigPath;
        } else {
            // Defaults if not specified
            mSkeletonPath = "@data/skeleton/base.yaml";
            mMarkerConfigPath = "@data/marker/static.xml";
            mInitialMarkerPath = mMarkerConfigPath;
        }

        // Fitting configuration
        if (config["fitting"]) {
            auto fitting = config["fitting"];
            if (fitting["config"]) mFittingConfigPath = fitting["config"].as<std::string>();
            if (fitting["static_config"]) mStaticConfigPath = fitting["static_config"].as<std::string>();
        } else {
            mFittingConfigPath = "@data/config/skeleton_fitting.yaml";
        }

        // Rendering configuration
        if (config["rendering"]) {
            auto rendering = config["rendering"];

            // Startup behavior
            if (rendering["autoload_first"]) mAutoloadFirstC3D = rendering["autoload_first"].as<bool>();

            // Render mode
            if (rendering["mode"]) {
                std::string mode = rendering["mode"].as<std::string>();
                if (mode == "mesh") mRenderMode = RenderMode::Mesh;
                else if (mode == "primitive") mRenderMode = RenderMode::Primitive;
                else if (mode == "wireframe") mRenderMode = RenderMode::Wireframe;
            }

            // Motion character offset
            if (rendering["motion_char_offset"]) {
                auto offset = rendering["motion_char_offset"];
                if (offset.IsSequence() && offset.size() == 3) {
                    mMotionCharacterOffset.x() = offset[0].as<double>();
                    mMotionCharacterOffset.y() = offset[1].as<double>();
                    mMotionCharacterOffset.z() = offset[2].as<double>();
                }
            }

            // Character visibility
            if (rendering["free_character"]) mRenderFreeCharacter = rendering["free_character"].as<bool>();
            if (rendering["motion_character"]) mRenderMotionCharacter = rendering["motion_character"].as<bool>();
            if (rendering["motion_char_markers"]) mRenderMotionCharMarkers = rendering["motion_char_markers"].as<bool>();

            // Marker visibility
            if (rendering["c3d_markers"]) mRenderC3DMarkers = rendering["c3d_markers"].as<bool>();
            if (rendering["skeleton_markers"]) mRenderExpectedMarkers = rendering["skeleton_markers"].as<bool>();
            if (rendering["joint_positions"]) mRenderJointPositions = rendering["joint_positions"].as<bool>();
            if (rendering["marker_labels"]) mRenderMarkerIndices = rendering["marker_labels"].as<bool>();
            if (rendering["marker_label_font_size"]) mMarkerLabelFontSize = rendering["marker_label_font_size"].as<float>();
            if (rendering["marker_alpha"]) mMarkerAlpha = rendering["marker_alpha"].as<float>();

            // Axis visualization
            if (rendering["world_axis"]) mRenderWorldAxis = rendering["world_axis"].as<bool>();
            if (rendering["skeleton_axis"]) mRenderSkeletonAxis = rendering["skeleton_axis"].as<bool>();
            if (rendering["axis_length"]) mAxisLength = rendering["axis_length"].as<float>();
        }

        LOG_INFO("[C3DProcessor] Config loaded - skeleton: " << mSkeletonPath
                 << ", marker: " << mMarkerConfigPath
                 << ", fitting: " << mFittingConfigPath);

    } catch (const std::exception& e) {
        LOG_ERROR("[C3DProcessor] Failed to load config from " << resolved_path << ": " << e.what());
        throw;
    }
}

// =============================================================================
// ViewerAppBase Overrides
// =============================================================================

void C3DProcessorApp::onInitialize()
{
    // Additional initialization after base class setup (if needed)
}

void C3DProcessorApp::onFrameStart()
{
    // Update playback timing
    double currentTime = glfwGetTime();
    double dt = currentTime - mLastRealTime;
    mLastRealTime = currentTime;

    if (mIsPlaying) {
        updateViewerTime(dt * mViewerPlaybackSpeed);
    } else {
        updateViewerTime(0.0);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void C3DProcessorApp::updateCamera()
{
    // Camera follow character (update mCamera.trans before setCamera applies it)
    if (mCamera.focus == 1 && mMotion != nullptr && mFreeCharacter) {
        // Calculate current position based on cycle accumulation
        Motion* current_motion = mMotion;
        C3DViewerState& state = mMotionState;

        double frame_float;
        if (state.navigationMode == C3D_SYNC) {
            // Use viewer time to compute frame
            frame_float = (mViewerTime / mViewerCycleDuration) * current_motion->getTotalTimesteps();
        } else {
            frame_float = static_cast<double>(state.manualFrameIndex);
        }

        int current_frame_idx = static_cast<int>(frame_float);
        int total_frames = current_motion->getTotalTimesteps();
        current_frame_idx = current_frame_idx % total_frames;

        Eigen::VectorXd raw_motion = current_motion->getRawMotionData();
        int value_per_frame = current_motion->getValuesPerFrame();
        Eigen::VectorXd current_frame = raw_motion.segment(current_frame_idx * value_per_frame, value_per_frame);

        // Motion data is already in angle format (positions 3,4,5 are root translation)
        Eigen::VectorXd current_pos = current_frame;

        // Follow root position with cycle accumulation and display offset
        mCamera.trans[0] = -(current_pos[3] + state.cycleAccumulation[0] + state.displayOffset[0]);
        mCamera.trans[1] = -(current_pos[4] + state.displayOffset[1]) - 1;
        mCamera.trans[2] = -(current_pos[5] + state.cycleAccumulation[2] + state.displayOffset[2]);
    }
}

// setCamera() removed - handled by ViewerAppBase::setCamera()

void C3DProcessorApp::loadRenderConfigImpl()
{
    // All configuration is now loaded from the unified config file in loadConfig()
    // Common config (geometry, default_open_panels) is still loaded by ViewerAppBase from render.yaml
}

// isPanelDefaultOpen() is inherited from ViewerAppBase

// =============================================================================
// Rendering
// =============================================================================

void C3DProcessorApp::drawContent()
{
    // Draw debug axes
    if (mRenderWorldAxis) drawAxis(Eigen::Isometry3d::Identity(), mAxisLength, "World");
    if (mRenderSkeletonAxis && mFreeCharacter) {
        auto skel = mFreeCharacter->getSkeleton();
        if (skel && skel->getNumBodyNodes() > 0) {
            drawAxis(skel->getBodyNode(0)->getTransform(), mAxisLength, "Root");
        }
    }

    drawSkeleton();
    drawMarkers();
    if (mMouseDown) GUI::DrawOriginAxisGizmo(-mCamera.trans);
    drawSelectedJointGizmo();
    drawSelectedBoneGizmo();
}

void C3DProcessorApp::drawUI()
{
    drawLeftPanel();
    drawRightPanel();
}

void C3DProcessorApp::drawAxis(const Eigen::Isometry3d& transform, float length, const std::string& label)
{
    glDisable(GL_LIGHTING);
    glLineWidth(3.0f);

    Eigen::Vector3d origin = transform.translation();
    Eigen::Matrix3d rot = transform.linear();

    // X axis - Red
    Eigen::Vector3d xEnd = origin + rot.col(0) * length;
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3d(origin.x(), origin.y(), origin.z());
    glVertex3d(xEnd.x(), xEnd.y(), xEnd.z());
    glEnd();

    // Y axis - Green
    Eigen::Vector3d yEnd = origin + rot.col(1) * length;
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3d(origin.x(), origin.y(), origin.z());
    glVertex3d(yEnd.x(), yEnd.y(), yEnd.z());
    glEnd();

    // Z axis - Blue
    Eigen::Vector3d zEnd = origin + rot.col(2) * length;
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3d(origin.x(), origin.y(), origin.z());
    glVertex3d(zEnd.x(), zEnd.y(), zEnd.z());
    glEnd();

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void C3DProcessorApp::drawSelectedJointGizmo()
{
    if (mJointAngleSelectedIdx < 0) return;

    glDisable(GL_LIGHTING);

    // Helper lambda to draw gizmo for one skeleton
    auto drawJointGizmoForSkeleton = [this](dart::dynamics::SkeletonPtr skel,
                                             const Eigen::Vector3d& offset,
                                             float colorTint) {
        if (!skel || mJointAngleSelectedIdx >= (int)skel->getNumJoints()) return;

        auto* joint = skel->getJoint(mJointAngleSelectedIdx);
        if (!joint) return;

        auto* childBody = joint->getChildBodyNode();
        if (!childBody) return;

        // Get joint world transform
        Eigen::Isometry3d jointWorld;
        auto* parentBody = childBody->getParentBodyNode();
        if (parentBody) {
            jointWorld = parentBody->getTransform() * joint->getTransformFromParentBodyNode();
        } else {
            jointWorld = joint->getTransformFromParentBodyNode();
        }

        // Apply offset
        jointWorld.translation() += offset;

        Eigen::Vector3d origin = jointWorld.translation();
        int numDofs = joint->getNumDofs();

        if (numDofs >= 3) {
            // FreeJoint or BallJoint: draw XYZ axis gizmo
            glPushMatrix();
            glMultMatrixd(jointWorld.data());
            GUI::DrawOriginAxisGizmo(Eigen::Vector3d::Zero(), 0.1f, colorTint);
            glPopMatrix();
        } else if (numDofs == 1) {
            // RevoluteJoint: draw rotation axis
            auto* revoluteJoint = dynamic_cast<dart::dynamics::RevoluteJoint*>(joint);
            if (revoluteJoint) {
                Eigen::Vector3d localAxis = revoluteJoint->getAxis();
                Eigen::Vector3d worldAxis = jointWorld.linear() * localAxis;

                float len = 0.15f;
                Eigen::Vector3d axisStart = origin - worldAxis * len;
                Eigen::Vector3d axisEnd = origin + worldAxis * len;

                glLineWidth(4.0f);
                // Yellow for FreeChar (colorTint=0), Orange for MotionChar (colorTint=1)
                glColor3f(1.0f, 0.7f - colorTint * 0.3f, colorTint * 0.2f);
                glBegin(GL_LINES);
                glVertex3d(axisStart.x(), axisStart.y(), axisStart.z());
                glVertex3d(axisEnd.x(), axisEnd.y(), axisEnd.z());
                glEnd();
            }
        }

        glLineWidth(1.0f);
    };

    // Draw for FreeChar
    if (mFreeCharacter) {
        drawJointGizmoForSkeleton(mFreeCharacter->getSkeleton(), Eigen::Vector3d::Zero(), 0.0f);
    }

    // Draw for MotionChar (with offset)
    if (mMotionCharacter && mRenderMotionCharacter) {
        drawJointGizmoForSkeleton(mMotionCharacter->getSkeleton(), mMotionCharacterOffset, 1.0f);
    }

    glEnable(GL_LIGHTING);
}

void C3DProcessorApp::drawSelectedBoneGizmo()
{
    if (mBonePoseSelectedIdx < 0) return;

    glDisable(GL_LIGHTING);

    // Helper lambda to draw gizmo for one skeleton's body node
    auto drawBoneGizmoForSkeleton = [this](dart::dynamics::SkeletonPtr skel,
                                            const Eigen::Vector3d& offset,
                                            float colorTint) {
        if (!skel || mBonePoseSelectedIdx >= (int)skel->getNumJoints()) return;

        // Get body node from joint (child body of selected joint)
        auto* joint = skel->getJoint(mBonePoseSelectedIdx);
        if (!joint) return;

        auto* bodyNode = joint->getChildBodyNode();
        if (!bodyNode) return;

        // Get body node's world transform
        Eigen::Isometry3d bodyWorld = bodyNode->getTransform();

        // Apply offset
        bodyWorld.translation() += offset;

        // Draw XYZ axis gizmo at body node position
        glPushMatrix();
        glMultMatrixd(bodyWorld.data());
        GUI::DrawOriginAxisGizmo(Eigen::Vector3d::Zero(), 0.12f, colorTint);
        glPopMatrix();
    };

    // Draw for FreeChar
    if (mFreeCharacter) {
        drawBoneGizmoForSkeleton(mFreeCharacter->getSkeleton(), Eigen::Vector3d::Zero(), 0.0f);
    }

    // Draw for MotionChar (with offset)
    if (mMotionCharacter && mRenderMotionCharacter) {
        drawBoneGizmoForSkeleton(mMotionCharacter->getSkeleton(), mMotionCharacterOffset, 1.0f);
    }

    glEnable(GL_LIGHTING);
}

void C3DProcessorApp::drawSkeleton()
{
    // Render mFreeCharacter (white, for bone-by-bone debugging)
    if (mFreeCharacter && mRenderFreeCharacter) {
        GUI::DrawSkeleton(mFreeCharacter->getSkeleton(),
                          Eigen::Vector4d(1.0, 1.0, 1.0, 0.9),
                          mRenderMode,
                          &mShapeRenderer);
    }

    // Render mMotionCharacter (green tint, for motion playback)
    if (mMotionCharacter && mRenderMotionCharacter) {
        // Apply offset transformation
        glPushMatrix();
        glTranslated(mMotionCharacterOffset.x(), mMotionCharacterOffset.y(), mMotionCharacterOffset.z());

        GUI::DrawSkeleton(mMotionCharacter->getSkeleton(),
                          Eigen::Vector4d(0.0, 1.0, 0.0, 0.9),
                          mRenderMode,
                          &mShapeRenderer);

        glPopMatrix();
    }
}

void C3DProcessorApp::drawMarkers()
{
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Get matrices and ImGui context for label rendering
    GLdouble modelview[16], projection[16];
    GLint viewport[4];
    ImDrawList* drawList = nullptr;
    ImFont* font = nullptr;

    if (mRenderMarkerIndices) {
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
        glGetDoublev(GL_PROJECTION_MATRIX, projection);
        glGetIntegerv(GL_VIEWPORT, viewport);
        drawList = ImGui::GetBackgroundDrawList();  // Behind ImGui panels
        font = ImGui::GetFont();
    }

    // Lambda for rendering a marker label
    auto renderLabel = [&](const Eigen::Vector3d& pos, int index, const std::string& name, float xOffset) {
        if (!mRenderMarkerIndices || !drawList) return;
        GLdouble screenX, screenY, screenZ;
        if (gluProject(pos.x(), pos.y(), pos.z(),
                      modelview, projection, viewport, &screenX, &screenY, &screenZ) == GL_TRUE) {
            if (screenZ > 0.0 && screenZ < 1.0) {
                float y = mHeight - static_cast<float>(screenY);
                std::string text = std::to_string(index) + ": " + name;
                drawOutlinedText(drawList, font, mMarkerLabelFontSize,
                    ImVec2(static_cast<float>(screenX) + xOffset, y - 10), text);
            }
        }
    };

    // 1. Data markers (from C3D capture) - green
    // Only render if we have C3D motion loaded
    if (mMotion && mMotion->getSourceType() == "c3d") {
        C3D* c3dMotion = static_cast<C3D*>(mMotion);
        const auto& dataLabels = c3dMotion->getLabels();

        // Apply cycleAccumulation + displayOffset so markers move forward with skeleton
        Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;

        if (mRenderC3DMarkers && !mMotionState.currentMarkers.empty()) {
            glColor4f(0.4f, 1.0f, 0.2f, mMarkerAlpha);
            for (size_t i = 0; i < mMotionState.currentMarkers.size(); ++i) {
                // Skip hidden markers
                if (mHiddenC3DMarkers.find(static_cast<int>(i)) != mHiddenC3DMarkers.end())
                    continue;
                const auto& marker = mMotionState.currentMarkers[i];
                if (!marker.array().isFinite().all()) continue;
                Eigen::Vector3d offsetMarker = marker + markerOffset;
                GUI::DrawSphere(offsetMarker, 0.0125);
                std::string name = (i < dataLabels.size()) ? dataLabels[i] : "";
                renderLabel(offsetMarker, static_cast<int>(i), name, -20.0f);
            }
        }
    }

    // 2. Skeleton markers (expected from skeleton pose) - red (FreeCharacter)
    if (mRenderExpectedMarkers && mFreeCharacter && mFreeCharacter->hasMarkers()) {
        glColor4f(1.0f, 0.0f, 0.0f, mMarkerAlpha);
        const auto& skelMarkers = mFreeCharacter->getMarkers();
        auto expectedMarkers = mFreeCharacter->getExpectedMarkerPositions();
        for (size_t i = 0; i < expectedMarkers.size(); ++i) {
            // Skip hidden markers
            if (mHiddenSkelMarkers.find(static_cast<int>(i)) != mHiddenSkelMarkers.end())
                continue;
            const auto& marker = expectedMarkers[i];
            if (!marker.array().isFinite().all()) continue;
            GUI::DrawSphere(marker, 0.0125);
            std::string name = (i < skelMarkers.size()) ? skelMarkers[i].name : "";
            renderLabel(marker, static_cast<int>(i), name, 5.0f);
        }
    }

    // 2b. Skeleton markers for MotionCharacter - orange (with offset)
    if (mRenderMotionCharMarkers && mMotionCharacter && mMotionCharacter->hasMarkers()) {
        glColor4f(1.0f, 0.5f, 0.0f, mMarkerAlpha);  // Orange
        const auto& skelMarkers = mMotionCharacter->getMarkers();
        auto expectedMarkers = mMotionCharacter->getExpectedMarkerPositions();
        for (size_t i = 0; i < expectedMarkers.size(); ++i) {
            // Skip hidden markers
            if (mHiddenSkelMarkers.find(static_cast<int>(i)) != mHiddenSkelMarkers.end())
                continue;
            const auto& marker = expectedMarkers[i];
            if (!marker.array().isFinite().all()) continue;
            Eigen::Vector3d offsetMarker = marker + mMotionCharacterOffset;
            GUI::DrawSphere(offsetMarker, 0.0125);
            std::string name = (i < skelMarkers.size()) ? skelMarkers[i].name : "";
            renderLabel(offsetMarker, static_cast<int>(i), name, 5.0f);
        }
    }

    // 3. Joint positions (parent joint of each body node) - purple
    if (mRenderJointPositions && mFreeCharacter) {
        auto skel = mFreeCharacter->getSkeleton();
        if (skel) {
            glColor4f(0.6f, 0.2f, 0.8f, mMarkerAlpha);
            for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
                const BodyNode* bn = skel->getBodyNode(i);
                if (!bn) continue;
                const Joint* joint = bn->getParentJoint();
                if (!joint) continue;
                // Joint position: parent body transform * joint's transform from parent
                Eigen::Isometry3d jointWorld;
                if (bn->getParentBodyNode()) {
                    jointWorld = bn->getParentBodyNode()->getTransform() *
                                 joint->getTransformFromParentBodyNode();
                } else {
                    jointWorld = joint->getTransformFromParentBodyNode();
                }
                Eigen::Vector3d pos = jointWorld.translation();
                if (!pos.array().isFinite().all()) continue;
                GUI::DrawSphere(pos, 0.0125);
            }
        }
    }

    glEnable(GL_LIGHTING);
}

// =============================================================================
// UI Panels
// =============================================================================

bool C3DProcessorApp::collapsingHeaderWithControls(const std::string& title)
{
    bool isDefaultOpen = isPanelDefaultOpen(title);
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen * isDefaultOpen;
    return ImGui::CollapsingHeader(title.c_str(), flags);
}

void C3DProcessorApp::drawLeftPanel()
{
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::Begin("Control##1", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    if (ImGui::BeginTabBar("ControlTabs")) {
        if (ImGui::BeginTabItem("Data")) {
            drawMotionListSection();
            ImGui::Separator();
            drawClinicalDataSection();
            ImGui::Separator();
            drawMarkerFittingSection();
            ImGui::Separator();
            drawSkeletonScaleSection();
            ImGui::Separator();
            drawSkeletonExportSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("View")) {
            drawPlaybackSection();
            ImGui::Separator();
            drawViewTabContent();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void C3DProcessorApp::drawMotionListSection()
{
    if (collapsingHeaderWithControls("C3D Files")) {
        // Rescan button
        if (ImGui::Button("Rescan")) {
            scanC3DFiles();
        }
        ImGui::SameLine();
        ImGui::Text("%zu files", mMotionList.size());

        // Filter input
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##C3DFilter", "Filter...", mC3DFilter, sizeof(mC3DFilter));

        // File list
        if (ImGui::BeginListBox("##C3DList", ImVec2(-1, 200))) {
            std::string filterLower(mC3DFilter);
            std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

            for (int i = 0; i < static_cast<int>(mMotionList.size()); ++i) {
                const std::string& displayName = mMotionDisplayNames[i];

                // Apply filter
                if (!filterLower.empty()) {
                    std::string nameLower = displayName;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    if (nameLower.find(filterLower) == std::string::npos) {
                        continue;
                    }
                }

                bool isSelected = (i == mSelectedMotion && mMotionSource == MotionSource::FileList);
                if (ImGui::Selectable(displayName.c_str(), isSelected)) {
                    if (i != mSelectedMotion || mMotionSource != MotionSource::FileList) {
                        mSelectedMotion = i;
                        mMotionSource = MotionSource::FileList;
                        loadC3DFile(mMotionList[i]);
                    }
                }
            }
            ImGui::EndListBox();
        }

        // Current motion info
        if (mMotion) {
            ImGui::Text("Frames: %d", mMotion->getNumFrames());
            ImGui::Text("FPS: %.1f", 1.0 / mMotion->getFrameTime());
        }
    }
}

void C3DProcessorApp::drawPlaybackSection()
{
    if (collapsingHeaderWithControls("Playback")) {
        // Play/Pause button
        if (ImGui::Button(mIsPlaying ? "Pause" : "Play")) {
            mIsPlaying = !mIsPlaying;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            mViewerTime = 0.0;
            mViewerPhase = 0.0;
            mMotionState.lastFrameIdx = 0;
            mMotionState.cycleAccumulation.setZero();
            // Reset camera pose
            mCamera.trans = Eigen::Vector3d(0.0, -0.8, 0.0);
            mCamera.zoom = 1.0;
            mCamera.trackball.setQuaternion(Eigen::Quaterniond::Identity());
        }

        // Playback speed
        if (ImGui::Button("0.3")) mViewerPlaybackSpeed = 0.3;
        ImGui::SameLine();
        if (ImGui::Button("0.5")) mViewerPlaybackSpeed = 0.5;
        ImGui::SameLine();
        if (ImGui::Button("1.0")) mViewerPlaybackSpeed = 1.0;
        ImGui::SameLine();

        ImGui::SetNextItemWidth(150);
        ImGui::SliderFloat("Speed", &mViewerPlaybackSpeed, 0.1f, 1.5f, "%.2fx");

        // Frame navigation
        if (mMotion) {
            int maxFrame = std::max(0, mMotion->getNumFrames() - 1);
            int currentFrame = mMotionState.navigationMode == C3D_MANUAL_FRAME ? mMotionState.manualFrameIndex : mMotionState.lastFrameIdx;

            ImGui::SetNextItemWidth(-100);  // Fill width, reserve space for +/- buttons
            if (ImGui::SliderInt("Frame", &currentFrame, 0, maxFrame)) {
                mMotionState.navigationMode = C3D_MANUAL_FRAME;
                mMotionState.manualFrameIndex = currentFrame;
            }
            if (mMotionState.navigationMode == C3D_MANUAL_FRAME) {
                ImGui::SameLine();
                if (ImGui::Button("+")) mMotionState.manualFrameIndex++;
                ImGui::SameLine();
                if (ImGui::Button("-")) mMotionState.manualFrameIndex--;
            }

            // Navigation mode toggle
            bool syncMode = (mMotionState.navigationMode == C3D_SYNC);
            if (ImGui::Checkbox("Sync Mode", &syncMode)) {
                mMotionState.navigationMode = syncMode ? C3D_SYNC : C3D_MANUAL_FRAME;
            }

            ImGui::SameLine();

            // Progress forward toggle
            ImGui::Checkbox("Progress Forward", &mProgressForward);
        }

        // Time display
        ImGui::Text("Time: %.2f s", mViewerTime);
        ImGui::Text("Phase: %.2f", mViewerPhase);
    }
}

void C3DProcessorApp::drawMarkerFittingSection()
{
    if (collapsingHeaderWithControls("Marker Fitting")) {
        // Static Calibration UI (enabled only when medial markers detected)
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Static Calibration");
        if (mHasMedialMarkers) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "[Medial Markers Detected]");
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!mHasMedialMarkers);
        if (ImGui::Button("Calibrate##Static")) {
            if (mC3DReader && mMotion) {
                C3D* c3dData = dynamic_cast<C3D*>(mMotion);
                if (c3dData) {
                    mStaticCalibResult = mC3DReader->calibrateStatic(c3dData, mStaticConfigPath);
                    if (mStaticCalibResult.success) {
                        LOG_INFO("[C3DProcessor] Static calibration completed successfully");
                        if (mFreeCharacter) {
                            auto& markers = mFreeCharacter->getMarkersForEdit();
                            for (auto& marker : markers) {
                                auto it = mStaticCalibResult.personalizedOffsets.find(marker.name);
                                if (it != mStaticCalibResult.personalizedOffsets.end()) {
                                    marker.offset = it->second;
                                }
                            }
                            LOG_INFO("[C3DProcessor] Applied " << mStaticCalibResult.personalizedOffsets.size()
                                     << " personalized marker offsets");
                        }
                    } else {
                        LOG_ERROR("[C3DProcessor] Static calibration failed: " << mStaticCalibResult.errorMessage);
                    }
                } else {
                    LOG_ERROR("[C3DProcessor] No C3D data available. Load a C3D file first.");
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Export##Static")) {
            if (mStaticCalibResult.success && mResourceManager && mPIDNavigator) {
                const auto& pidState = mPIDNavigator->getState();
                if (pidState.selectedPID >= 0 && pidState.selectedPID < static_cast<int>(pidState.pidList.size())) {
                    std::string pid = pidState.pidList[pidState.selectedPID];
                    std::string prePost = pidState.preOp ? "pre" : "post";
                    std::string pattern = "@pid:" + pid + "/gait/" + prePost;
                    std::string outputDir = mResourceManager->resolveDir(pattern);
                    if (!outputDir.empty()) {
                        mC3DReader->exportPersonalizedCalibration(mStaticCalibResult, outputDir);
                        LOG_INFO("[C3DProcessor] Exported personalized calibration for PID " << pid);
                        mHasPersonalizedCalibration = true;
                    }
                }
            }
        }
        ImGui::EndDisabled();

        // Show static calibration results
        if (mStaticCalibResult.success) {
            if (ImGui::TreeNode("Static Results")) {
                if (ImGui::TreeNode("Bone Scales")) {
                    for (const auto& [name, scale] : mStaticCalibResult.boneScales) {
                        ImGui::Text("%s: (%.3f, %.3f, %.3f)", name.c_str(), scale.x(), scale.y(), scale.z());
                    }
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("Personalized Offsets")) {
                    for (const auto& [name, offset] : mStaticCalibResult.personalizedOffsets) {
                        ImGui::Text("%s: (%.4f, %.4f, %.4f)", name.c_str(), offset.x(), offset.y(), offset.z());
                    }
                    ImGui::TreePop();
                }
                ImGui::TreePop();
            }
        }

        ImGui::Separator();

        // Dynamic Calibration UI (enabled only when medial markers NOT detected)
        ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.7f, 1.0f), "Dynamic Calibration");
        ImGui::SameLine();
        ImGui::BeginDisabled(mHasMedialMarkers);
        if (ImGui::Button("Calibrate and Track##Dynamic")) {
            if (mC3DReader && mMotion) {
                C3D* c3dData = dynamic_cast<C3D*>(mMotion);
                if (c3dData) {
                    mDynamicCalibResult = mC3DReader->calibrateDynamic(c3dData);
                    if (mDynamicCalibResult.success) {
                        LOG_INFO("[C3DProcessor] Dynamic calibration completed: "
                                 << mDynamicCalibResult.freePoses.size() << " frames");
                    } else {
                        LOG_ERROR("[C3DProcessor] Dynamic calibration failed: " << mDynamicCalibResult.errorMessage);
                    }
                } else {
                    LOG_ERROR("[C3DProcessor] No C3D data available. Load a C3D file first.");
                }
            }
        }
        ImGui::EndDisabled();

        // Export HDF - same line as calibrate button

        bool canExportHDF = mDynamicCalibResult.success
                         && !mDynamicCalibResult.motionPoses.empty()
                         && mResourceManager != nullptr
                         && mPIDNavigator && mPIDNavigator->getState().selectedPID >= 0;

        if (!canExportHDF) ImGui::BeginDisabled();
        if (ImGui::Button("Export HDF")) {
            exportMotionToHDF5();
        }
        if (!canExportHDF) ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputTextWithHint("##hdfname", "filename", mExportHDFName, sizeof(mExportHDFName));

        // Check if destination file exists and show warning
        if (canExportHDF) {
            const auto& pidState = mPIDNavigator->getState();
            std::string pid = pidState.pidList[pidState.selectedPID];
            std::string prePost = pidState.preOp ? "pre" : "post";
            std::string pattern = "@pid:" + pid + "/gait/" + prePost + "/h5";
            std::string outputDir = mResourceManager->resolveDir(pattern);
            if (!outputDir.empty()) {
                std::string filename = (std::strlen(mExportHDFName) > 0)
                    ? mExportHDFName
                    : fs::path(mMotionPath).stem().string();
                std::string outputPath = outputDir + "/" + filename + ".h5";
                if (fs::exists(outputPath)) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "File exists!");
                }
            }
        }

        ImGui::Separator();
        if (ImGui::Button("Clear Motion & Zero Pose")) clearMotionAndZeroPose();
    }
}

void C3DProcessorApp::drawSkeletonScaleSection()
{
    if (collapsingHeaderWithControls("Skeleton Scale")) {
        // Character selection radio buttons
        ImGui::RadioButton("Free", &mScaleCharacterSelection, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Motion", &mScaleCharacterSelection, 1);
        ImGui::Separator();

        // Select character based on radio button
        RenderCharacter* character = nullptr;
        if (mScaleCharacterSelection == 0 && mFreeCharacter) {
            character = mFreeCharacter.get();
        } else if (mScaleCharacterSelection == 1 && mMotionCharacter) {
            character = mMotionCharacter.get();
        }

        if (!character) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Selected character not loaded");
            return;
        }

        auto skel = character->getSkeleton();
        auto& skelInfos = character->getSkelInfos();

        if (!skel || skelInfos.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No skeleton info available");
            return;
        }

        bool anyChanged = false;

        // Build map of bone name -> index for quick lookup
        std::map<std::string, int> boneNameToIdx;
        for (size_t i = 0; i < skelInfos.size(); ++i) {
            boneNameToIdx[std::get<0>(skelInfos[i])] = static_cast<int>(i);
        }

        // Define paired bones (Right, Left)
        std::vector<std::pair<std::string, std::string>> pairedBones = {
            {"FemurR", "FemurL"},
            {"TibiaR", "TibiaL"},
            {"TalusR", "TalusL"},
            {"FootR", "FootL"},
            {"ArmR", "ArmL"},
            {"ForeArmR", "ForeArmL"},
            {"HandR", "HandL"},
        };

        // Track processed bones to avoid duplicates
        std::set<std::string> processedBones;

        // Helper lambda to check if values differ by 1.5x ratio
        auto isDifferent = [](double a, double b) -> bool {
            if (a == 0 && b == 0) return false;
            if (a == 0 || b == 0) return true;
            double ratio = a / b;
            return ratio > 1.5 || ratio < (1.0 / 1.5);
        };

        // Helper to get highlight color
        auto getHighlightColor = [&isDifferent](double valR, double valL) -> ImVec4 {
            if (isDifferent(valR, valL)) {
                return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red highlight
            }
            return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);  // Normal white
        };

        // Draw paired bones side-by-side
        for (const auto& [rightName, leftName] : pairedBones) {
            auto itR = boneNameToIdx.find(rightName);
            auto itL = boneNameToIdx.find(leftName);
            if (itR == boneNameToIdx.end() || itL == boneNameToIdx.end()) continue;

            int idxR = itR->second;
            int idxL = itL->second;
            auto& modInfoR = std::get<1>(skelInfos[idxR]);
            auto& modInfoL = std::get<1>(skelInfos[idxL]);

            processedBones.insert(rightName);
            processedBones.insert(leftName);

            // Extract base name (remove R/L suffix)
            std::string baseName = rightName.substr(0, rightName.length() - 1);

            ImGui::PushID(baseName.c_str());
            if (ImGui::TreeNode((baseName + " (R/L)").c_str())) {
                // Table header
                ImGui::Text("%-12s %10s %10s", "Parameter", "Right", "Left");
                ImGui::Separator();

                const char* paramNames[] = {"Scale X", "Scale Y", "Scale Z", "Uniform", "Torsion"};
                float minVals[] = {0.5f, 0.5f, 0.5f, 0.5f, -45.0f};
                float maxVals[] = {2.0f, 2.0f, 2.0f, 2.0f, 45.0f};

                for (int p = 0; p < 5; ++p) {
                    float valR = static_cast<float>(modInfoR.value[p]);
                    float valL = static_cast<float>(modInfoL.value[p]);

                    ImVec4 color = getHighlightColor(modInfoR.value[p], modInfoL.value[p]);
                    bool highlighted = isDifferent(modInfoR.value[p], modInfoL.value[p]);

                    ImGui::PushID(p);

                    // Parameter name with highlight
                    if (highlighted) {
                        ImGui::TextColored(color, "%-12s", paramNames[p]);
                    } else {
                        ImGui::Text("%-12s", paramNames[p]);
                    }

                    // Right value drag
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80);
                    if (highlighted) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.5f, 0.1f, 0.1f, 1.0f));
                    if (ImGui::DragFloat("##R", &valR, 0.01f, minVals[p], maxVals[p], "%.3f")) {
                        modInfoR.value[p] = valR;
                        anyChanged = true;
                    }
                    if (highlighted) ImGui::PopStyleColor();

                    // Left value drag
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80);
                    if (highlighted) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.5f, 0.1f, 0.1f, 1.0f));
                    if (ImGui::DragFloat("##L", &valL, 0.01f, minVals[p], maxVals[p], "%.3f")) {
                        modInfoL.value[p] = valL;
                        anyChanged = true;
                    }
                    if (highlighted) ImGui::PopStyleColor();

                    ImGui::PopID();
                }

                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        // Draw unpaired bones (center bones like Pelvis, Spine, etc.)
        if (ImGui::TreeNode("Center Bones")) {
            for (size_t i = 0; i < skelInfos.size(); ++i) {
                auto& [boneName, modInfo] = skelInfos[i];
                if (processedBones.count(boneName) > 0) continue;

                auto* bn = skel->getBodyNode(boneName);
                if (!bn) continue;

                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode(boneName.c_str())) {
                    ImGui::Text("[%.3f, %.3f, %.3f, %.3f, %.3f]",
                        modInfo.value[0], modInfo.value[1], modInfo.value[2],
                        modInfo.value[3], modInfo.value[4]);

                    const char* paramNames[] = {"Scale X", "Scale Y", "Scale Z", "Uniform", "Torsion"};
                    float minVals[] = {0.5f, 0.5f, 0.5f, 0.5f, -45.0f};
                    float maxVals[] = {2.0f, 2.0f, 2.0f, 2.0f, 45.0f};

                    for (int p = 0; p < 5; ++p) {
                        float val = static_cast<float>(modInfo.value[p]);
                        ImGui::SetNextItemWidth(120);
                        if (ImGui::DragFloat(paramNames[p], &val, 0.01f, minVals[p], maxVals[p], "%.3f")) {
                            modInfo.value[p] = val;
                            anyChanged = true;
                        }
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            ImGui::TreePop();
        }

        // Apply all bone scales when any changed
        if (anyChanged) {
            character->applySkeletonBodyNode(skelInfos, skel);
            // Also apply to the other character if both exist
            if (mFreeCharacter && mMotionCharacter && character == mFreeCharacter.get()) {
                mMotionCharacter->applySkeletonBodyNode(skelInfos, mMotionCharacter->getSkeleton());
            } else if (mFreeCharacter && mMotionCharacter && character == mMotionCharacter.get()) {
                mFreeCharacter->applySkeletonBodyNode(skelInfos, mFreeCharacter->getSkeleton());
            }
        }
    }
}

void C3DProcessorApp::drawSkeletonExportSection()
{
    if (collapsingHeaderWithControls("Skeleton Export")) {
        // Export path display
        ImGui::Text("Export to: data/skeleton/");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##exportname", mExportSkeletonName, sizeof(mExportSkeletonName));
        ImGui::SameLine();
        ImGui::Text(".yaml");

        // Export button
        if (ImGui::Button("Export Calibrated Skeleton")) {
            if (mMotionCharacter) {
                std::string outputPath = std::string("data/skeleton/") + mExportSkeletonName + ".yaml";
                mMotionCharacter->exportSkeletonYAML(outputPath);
                LOG_INFO("[C3DProcessor] Exported calibrated skeleton to: " + outputPath);
            }
        }
    }
}

void C3DProcessorApp::exportMotionToHDF5()
{
    // 1. Resolve output path via PID URI
    const auto& pidState = mPIDNavigator->getState();
    std::string pid = pidState.pidList[pidState.selectedPID];
    std::string prePost = pidState.preOp ? "pre" : "post";
    std::string pattern = "@pid:" + pid + "/gait/" + prePost + "/h5";
    std::string outputDir = mResourceManager->resolveDirCreate(pattern);

    if (outputDir.empty()) {
        LOG_ERROR("[C3DProcessor] Failed to resolve PID directory: " << pattern);
        return;
    }

    // 2. Determine filename (use C3D stem if not specified)
    std::string filename;
    if (std::strlen(mExportHDFName) > 0) {
        filename = mExportHDFName;
    } else {
        filename = fs::path(mMotionPath).stem().string();
    }
    std::string outputPath = outputDir + "/" + filename + ".h5";

    try {
        H5::H5File file(outputPath, H5F_ACC_TRUNC);

        const auto& poses = mDynamicCalibResult.motionPoses;
        int numFrames = static_cast<int>(poses.size());
        int dofPerFrame = poses.empty() ? 56 : static_cast<int>(poses[0].size());
        int frameRate = mC3DReader ? mC3DReader->getFrameRate() : 100;

        // 3. Write /motions (numFrames x DOF)
        hsize_t dims_motions[2] = {(hsize_t)numFrames, (hsize_t)dofPerFrame};
        H5::DataSpace space_motions(2, dims_motions);
        H5::DataSet ds_motions = file.createDataSet("/motions", H5::PredType::NATIVE_FLOAT, space_motions);

        std::vector<float> motionBuffer(numFrames * dofPerFrame);
        for (int f = 0; f < numFrames; ++f) {
            for (int d = 0; d < dofPerFrame; ++d) {
                motionBuffer[f * dofPerFrame + d] = static_cast<float>(poses[f][d]);
            }
        }
        ds_motions.write(motionBuffer.data(), H5::PredType::NATIVE_FLOAT);

        // 4. Write /phase (normalized 0-1)
        hsize_t dims_1d[1] = {(hsize_t)numFrames};
        H5::DataSpace space_1d(1, dims_1d);
        H5::DataSet ds_phase = file.createDataSet("/phase", H5::PredType::NATIVE_FLOAT, space_1d);

        std::vector<float> phaseBuffer(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            phaseBuffer[f] = static_cast<float>(f) / static_cast<float>(numFrames - 1);
        }
        ds_phase.write(phaseBuffer.data(), H5::PredType::NATIVE_FLOAT);

        // 5. Write /time
        H5::DataSet ds_time = file.createDataSet("/time", H5::PredType::NATIVE_FLOAT, space_1d);
        double dt = 1.0 / frameRate;
        std::vector<float> timeBuffer(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            timeBuffer[f] = static_cast<float>(f * dt);
        }
        ds_time.write(timeBuffer.data(), H5::PredType::NATIVE_FLOAT);

        // 6. Write metadata as HDF5 attributes on root group
        H5::Group root = file.openGroup("/");
        H5::DataSpace scalarSpace(H5S_SCALAR);

        // String attributes helper
        auto writeStrAttr = [&](const char* name, const std::string& value) {
            H5::StrType strType(H5::PredType::C_S1, value.size() + 1);
            H5::Attribute attr = root.createAttribute(name, strType, scalarSpace);
            attr.write(strType, value.c_str());
        };

        // Integer attributes
        auto writeIntAttr = [&](const char* name, int value) {
            H5::Attribute attr = root.createAttribute(name, H5::PredType::NATIVE_INT, scalarSpace);
            attr.write(H5::PredType::NATIVE_INT, &value);
        };

        writeStrAttr("source_type", "c3d_dynamic_calibration");
        writeStrAttr("c3d_file", fs::path(mMotionPath).filename().string());
        writeIntAttr("frame_rate", frameRate);
        writeIntAttr("num_frames", numFrames);
        writeIntAttr("dof_per_frame", dofPerFrame);
        writeStrAttr("pid", pid);
        writeStrAttr("pre_post", prePost);

        // 7. Compute and write marker tracking errors for both characters
        C3D* c3dMotion = static_cast<C3D*>(mMotion);
        const auto& config = mC3DReader->getFittingConfig();
        const auto& mappings = config.markerMappings;
        int numMarkers = static_cast<int>(mappings.size());

        if (numMarkers > 0) {
            // Create /marker_error group
            H5::Group errorGroup = file.createGroup("/marker_error");

            // Helper lambda to compute and write errors for a character
            auto writeCharacterErrors = [&](RenderCharacter* character,
                                            const std::vector<Eigen::VectorXd>& charPoses,
                                            const std::string& groupName) {
                if (!character || charPoses.empty()) return;

                H5::Group charGroup = errorGroup.createGroup(groupName);
                auto skel = character->getSkeleton();

                std::vector<float> errorBuffer(numFrames * numMarkers, 0.0f);
                std::vector<float> meanBuffer(numFrames, 0.0f);

                for (int f = 0; f < numFrames; ++f) {
                    skel->setPositions(charPoses[f]);
                    auto expectedMarkers = character->getExpectedMarkerPositions();
                    const auto& c3dMarkers = c3dMotion->getMarkers(f);
                    const auto& skelMarkers = character->getMarkers();

                    double frameErrorSum = 0.0;
                    int validCount = 0;

                    for (int m = 0; m < numMarkers; ++m) {
                        const auto& mapping = mappings[m];
                        int dataIdx = mapping.dataIndex;

                        // Find skeleton marker index
                        int skelIdx = -1;
                        for (size_t j = 0; j < skelMarkers.size(); ++j) {
                            if (skelMarkers[j].name == mapping.name) {
                                skelIdx = static_cast<int>(j);
                                break;
                            }
                        }

                        bool hasData = dataIdx >= 0 && dataIdx < (int)c3dMarkers.size();
                        bool hasSkel = skelIdx >= 0 && skelIdx < (int)expectedMarkers.size();

                        if (hasData && hasSkel) {
                            Eigen::Vector3d dataM = c3dMarkers[dataIdx];
                            Eigen::Vector3d skelM = expectedMarkers[skelIdx];
                            if (dataM.array().isFinite().all() && skelM.array().isFinite().all()) {
                                double err = (dataM - skelM).norm() * 1000.0;  // mm
                                errorBuffer[f * numMarkers + m] = static_cast<float>(err);
                                frameErrorSum += err;
                                validCount++;
                            }
                        }
                    }
                    meanBuffer[f] = validCount > 0 ? static_cast<float>(frameErrorSum / validCount) : 0.0f;
                }

                // Write data (numFrames x numMarkers)
                hsize_t dims_error[2] = {(hsize_t)numFrames, (hsize_t)numMarkers};
                H5::DataSpace space_error(2, dims_error);
                H5::DataSet ds_data = charGroup.createDataSet("data", H5::PredType::NATIVE_FLOAT, space_error);
                ds_data.write(errorBuffer.data(), H5::PredType::NATIVE_FLOAT);

                // Write mean (numFrames,)
                H5::DataSet ds_mean = charGroup.createDataSet("mean", H5::PredType::NATIVE_FLOAT, space_1d);
                ds_mean.write(meanBuffer.data(), H5::PredType::NATIVE_FLOAT);
            };

            // Write errors for both characters
            writeCharacterErrors(mFreeCharacter.get(), mDynamicCalibResult.freePoses, "free");
            writeCharacterErrors(mMotionCharacter.get(), mDynamicCalibResult.motionPoses, "motion");

            // Write marker names as root attribute
            std::string namesStr;
            for (size_t i = 0; i < mappings.size(); ++i) {
                if (i > 0) namesStr += ",";
                namesStr += mappings[i].name;
            }
            writeStrAttr("marker_names", namesStr);
            writeIntAttr("num_markers", numMarkers);
        }

        file.close();
        LOG_INFO("[C3DProcessor] Exported HDF5 motion to: " << outputPath);
        std::string hdfURI = "@pid:" + pid + "/gait/" + prePost + "/h5/" + filename + ".h5";
        LOG_INFO("[C3DProcessor] URI: " << hdfURI);

    } catch (const H5::Exception& e) {
        LOG_ERROR("[C3DProcessor] HDF5 export error: " << e.getDetailMsg());
    }
}

void C3DProcessorApp::drawRightPanel()
{
    ImGui::SetNextWindowSize(ImVec2(500, mHeight), ImGuiCond_Once);
    ImGui::Begin("Visualization", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos(ImVec2(mWidth - ImGui::GetWindowSize().x, 0), ImGuiCond_Always);

    // File Paths
    if (ImGui::CollapsingHeader("File Paths", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Skeleton: %s", mSkeletonPath.c_str());
        ImGui::TextWrapped("Markers: %s", mMarkerConfigPath.c_str());
    }

    // Camera Information
    if (ImGui::CollapsingHeader("Camera Info")) {
        ImGui::Text("Position:");
        ImGui::Text("  X: %.3f", mCamera.trans[0]);
        ImGui::Text("  Y: %.3f", mCamera.trans[1]);
        ImGui::Text("  Z: %.3f", mCamera.trans[2]);
        ImGui::Separator();
        ImGui::Text("Zoom: %.3f", mCamera.zoom);
        ImGui::Text("Perspective: %.1f deg", mCamera.persp);
        ImGui::Separator();
        ImGui::Text("Follow Mode: %s", mCamera.focus == 1 ? "ON" : "OFF");
        ImGui::Text("Trackball Rotation:");
        Eigen::Matrix3d rotMat = mCamera.trackball.getRotationMatrix();
        Eigen::Vector3d euler = rotMat.eulerAngles(2, 1, 0) * 180.0 / M_PI;
        ImGui::Text("  Yaw:   %.1f deg", euler[0]);
        ImGui::Text("  Pitch: %.1f deg", euler[1]);
        ImGui::Text("  Roll:  %.1f deg", euler[2]);
    }

    // Marker Error Plot
    drawMarkerDiffPlot();

    drawMarkerCorrespondenceTable();

    drawBonePoseSection();

    drawJointAngleSection();

    drawJointOffsetSection();

    ImGui::End();
}

void C3DProcessorApp::drawViewTabContent()
{
    // Render Mode section
    ImGui::Text("Render Mode (O):");
    int mode = static_cast<int>(mRenderMode);
    if (ImGui::RadioButton("Primitive", &mode, 0)) mRenderMode = RenderMode::Primitive;
    ImGui::SameLine();
    if (ImGui::RadioButton("Mesh", &mode, 1)) mRenderMode = RenderMode::Mesh;
    ImGui::SameLine();
    if (ImGui::RadioButton("Wire", &mode, 2)) mRenderMode = RenderMode::Wireframe;
    ImGui::Separator();

    // Free Character section
    if (ImGui::CollapsingHeader("Free Character", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Draw Free Character", &mRenderFreeCharacter);
    }

    // Motion Character section
    if (ImGui::CollapsingHeader("Motion Character", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Draw Motion Character", &mRenderMotionCharacter);
        ImGui::Checkbox("Draw Skeleton Markers", &mRenderMotionCharMarkers);

        ImGui::Text("Offset from Free Character:");
        float offset[3] = {
            static_cast<float>(mMotionCharacterOffset.x()),
            static_cast<float>(mMotionCharacterOffset.y()),
            static_cast<float>(mMotionCharacterOffset.z())
        };

        ImGui::PushItemWidth(100);
        if (ImGui::InputFloat("X##MotionOffset", &offset[0], 0.1f, 0.5f, "%.3f")) {
            mMotionCharacterOffset.x() = offset[0];
        }
        ImGui::SameLine();
        if (ImGui::InputFloat("Y##MotionOffset", &offset[1], 0.1f, 0.5f, "%.3f")) {
            mMotionCharacterOffset.y() = offset[1];
        }
        ImGui::SameLine();
        if (ImGui::InputFloat("Z##MotionOffset", &offset[2], 0.1f, 0.5f, "%.3f")) {
            mMotionCharacterOffset.z() = offset[2];
        }
        ImGui::PopItemWidth();

        if (ImGui::Button("Reset##MotionOffset")) {
            mMotionCharacterOffset.setZero();
        }
        ImGui::SameLine();
        if (ImGui::Button("X##MotionOffsetX")) {
            mMotionCharacterOffset.x() = 0.8;
            mMotionCharacterOffset.y() = 0.0;
            mMotionCharacterOffset.z() = 0.0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Z##MotionOffsetZ")) {
            mMotionCharacterOffset.x() = 0.0;
            mMotionCharacterOffset.y() = 0.0;
            mMotionCharacterOffset.z() = 0.5;
        }
    }

    // Marker Visibility section
    if (ImGui::CollapsingHeader("Marker Visibility", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Data Markers (C3D)", &mRenderC3DMarkers);
        ImGui::Checkbox("Skeleton Markers", &mRenderExpectedMarkers);
        ImGui::Checkbox("Joint Positions", &mRenderJointPositions);
        ImGui::Checkbox("Marker Labels", &mRenderMarkerIndices);
        ImGui::SliderFloat("Label Font Size", &mMarkerLabelFontSize, 10.0f, 32.0f, "%.0f");
        ImGui::SliderFloat("Marker Alpha", &mMarkerAlpha, 0.0f, 1.0f, "%.2f");
    }

    // Axis Visualization section
    if (ImGui::CollapsingHeader("Axis Visualization")) {
        ImGui::Checkbox("World Axis", &mRenderWorldAxis);
        ImGui::Checkbox("Skeleton Root Axis", &mRenderSkeletonAxis);
        ImGui::SliderFloat("Axis Length", &mAxisLength, 0.1f, 1.0f, "%.2f");
    }

    // Per-marker visibility list
    if (ImGui::CollapsingHeader("Individual Markers")) {
        // Filter input
        ImGui::InputText("Filter##RenderMarker", mRenderingMarkerFilter, sizeof(mRenderingMarkerFilter));

        // Show/Hide all buttons
        if (ImGui::Button("Show All")) {
            mHiddenC3DMarkers.clear();
            mHiddenSkelMarkers.clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Hide All")) {
            if (mMotion && mMotion->getSourceType() == "c3d") {
                C3D* c3dMotion = static_cast<C3D*>(mMotion);
                for (size_t i = 0; i < c3dMotion->getLabels().size(); ++i)
                    mHiddenC3DMarkers.insert(static_cast<int>(i));
            }
            if (mFreeCharacter && mFreeCharacter->hasMarkers()) {
                for (size_t i = 0; i < mFreeCharacter->getMarkers().size(); ++i)
                    mHiddenSkelMarkers.insert(static_cast<int>(i));
            }
        }

        ImGui::Separator();

        // Get marker data
        std::vector<std::string> c3dLabels;
        std::vector<std::string> skelLabels;

        if (mMotion && mMotion->getSourceType() == "c3d") {
            C3D* c3dMotion = static_cast<C3D*>(mMotion);
            c3dLabels = c3dMotion->getLabels();
        }
        if (mFreeCharacter && mFreeCharacter->hasMarkers()) {
            const auto& markers = mFreeCharacter->getMarkers();
            for (const auto& m : markers)
                skelLabels.push_back(m.name);
        }

        // Filter string (lowercase)
        std::string filterStr(mRenderingMarkerFilter);
        std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

        // Scrollable list
        if (ImGui::BeginChild("MarkerList", ImVec2(0, 300), true)) {
            // C3D Markers
            if (!c3dLabels.empty() && ImGui::TreeNode("C3D Markers")) {
                for (size_t i = 0; i < c3dLabels.size(); ++i) {
                    const std::string& label = c3dLabels[i];

                    // Apply filter
                    if (!filterStr.empty()) {
                        std::string labelLower = label;
                        std::transform(labelLower.begin(), labelLower.end(), labelLower.begin(), ::tolower);
                        if (labelLower.find(filterStr) == std::string::npos)
                            continue;
                    }

                    bool visible = (mHiddenC3DMarkers.find(static_cast<int>(i)) == mHiddenC3DMarkers.end());
                    std::string checkboxLabel = std::to_string(i) + ": " + label + "##c3d";
                    if (ImGui::Checkbox(checkboxLabel.c_str(), &visible)) {
                        if (visible)
                            mHiddenC3DMarkers.erase(static_cast<int>(i));
                        else
                            mHiddenC3DMarkers.insert(static_cast<int>(i));
                    }
                }
                ImGui::TreePop();
            }

            // Skeleton Markers
            if (!skelLabels.empty() && ImGui::TreeNode("Skeleton Markers")) {
                for (size_t i = 0; i < skelLabels.size(); ++i) {
                    const std::string& label = skelLabels[i];

                    // Apply filter
                    if (!filterStr.empty()) {
                        std::string labelLower = label;
                        std::transform(labelLower.begin(), labelLower.end(), labelLower.begin(), ::tolower);
                        if (labelLower.find(filterStr) == std::string::npos)
                            continue;
                    }

                    bool visible = (mHiddenSkelMarkers.find(static_cast<int>(i)) == mHiddenSkelMarkers.end());
                    std::string checkboxLabel = std::to_string(i) + ": " + label + "##skel";
                    if (ImGui::Checkbox(checkboxLabel.c_str(), &visible)) {
                        if (visible)
                            mHiddenSkelMarkers.erase(static_cast<int>(i));
                        else
                            mHiddenSkelMarkers.insert(static_cast<int>(i));
                    }
                }
                ImGui::TreePop();
            }
        }
        ImGui::EndChild();
    }
}

void C3DProcessorApp::drawMarkerDiffPlot()
{
    if (!collapsingHeaderWithControls("Marker Error")) return;
    if (!mMotion || mMotion->getSourceType() != "c3d") return;
    PlotUtils::plotMarkerError(mGraphData, mXmin, 300.0f);
}

void C3DProcessorApp::drawMarkerCorrespondenceTable()
{
    if (!collapsingHeaderWithControls("Marker Correspondence")) return;

    if (!mMotion || mMotion->getSourceType() != "c3d" || !mFreeCharacter || !mC3DReader) {
        ImGui::Text("Load a C3D file to see correspondence");
        return;
    }

    C3D* c3dMotion = static_cast<C3D*>(mMotion);
    const auto& dataLabels = c3dMotion->getLabels();
    const auto& skelMarkers = mFreeCharacter->getMarkers();

    const auto& config = mC3DReader->getFittingConfig();
    const auto& mappings = config.markerMappings;

    if (mappings.empty()) {
        ImGui::Text("No marker mappings defined in skeleton_fitting.yaml");
        return;
    }

    // Search filter
    ImGui::InputText("Filter", mMarkerSearchFilter, sizeof(mMarkerSearchFilter));

    auto expectedMarkers = mFreeCharacter->getExpectedMarkerPositions();
    Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;

    if (ImGui::BeginTable("MarkerTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 300))) {
        ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed, 30);
        ImGui::TableSetupColumn("Data", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Skel", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Diff", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < mappings.size(); ++i) {
            const auto& mapping = mappings[i];
            int dataIdx = mapping.dataIndex;

            int skelIdx = -1;
            for (size_t j = 0; j < skelMarkers.size(); ++j) {
                if (skelMarkers[j].name == mapping.name) {
                    skelIdx = static_cast<int>(j);
                    break;
                }
            }

            std::string dataLabel = (dataIdx >= 0 && dataIdx < (int)dataLabels.size())
                ? dataLabels[dataIdx] : "-";
            std::string skelLabel = mapping.name;

            // Apply filter
            if (strlen(mMarkerSearchFilter) > 0) {
                std::string filter(mMarkerSearchFilter);
                std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);
                std::string dataLower = dataLabel;
                std::string skelLower = skelLabel;
                std::transform(dataLower.begin(), dataLower.end(), dataLower.begin(), ::tolower);
                std::transform(skelLower.begin(), skelLower.end(), skelLower.begin(), ::tolower);

                if (dataLower.find(filter) == std::string::npos &&
                    skelLower.find(filter) == std::string::npos) {
                    continue;
                }
            }

            bool hasDataM = dataIdx >= 0 && dataIdx < (int)mMotionState.currentMarkers.size();
            bool hasSkelM = skelIdx >= 0 && skelIdx < (int)expectedMarkers.size();
            Eigen::Vector3d dataM = hasDataM ? mMotionState.currentMarkers[dataIdx] : Eigen::Vector3d::Zero();
            Eigen::Vector3d skelM = hasSkelM ? expectedMarkers[skelIdx] : Eigen::Vector3d::Zero();
            Eigen::Vector3d offsetDataM = dataM + markerOffset;
            bool dataValid = hasDataM && dataM.array().isFinite().all();
            bool skelValid = hasSkelM && skelM.array().isFinite().all();

            // Row 1: Labels and error norm
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", dataIdx);

            ImGui::TableNextColumn();
            ImGui::Text("%s", dataLabel.c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%s", skelLabel.c_str());

            ImGui::TableNextColumn();
            if (dataValid && skelValid) {
                double err = (offsetDataM - skelM).norm() * 1000;  // mm
                ImGui::Text("%.1f mm", err);
            } else {
                ImGui::Text("-");
            }

            // Row 2: XYZ positions and diff
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            // Empty index cell

            ImGui::TableNextColumn();
            if (dataValid) {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%.3f %.3f %.3f",
                    offsetDataM.x(), offsetDataM.y(), offsetDataM.z());
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
            }

            ImGui::TableNextColumn();
            if (skelValid) {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%.3f %.3f %.3f",
                    skelM.x(), skelM.y(), skelM.z());
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
            }

            ImGui::TableNextColumn();
            if (dataValid && skelValid) {
                Eigen::Vector3d diff = offsetDataM - skelM;
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%.3f %.3f %.3f",
                    diff.x(), diff.y(), diff.z());
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
            }
        }
        ImGui::EndTable();
    }
}

void C3DProcessorApp::drawBonePoseSection()
{
    if (!collapsingHeaderWithControls("Bone Pose")) return;

    // Get both skeletons
    dart::dynamics::SkeletonPtr freeSkel = mFreeCharacter ? mFreeCharacter->getSkeleton() : nullptr;
    dart::dynamics::SkeletonPtr motionSkel = mMotionCharacter ? mMotionCharacter->getSkeleton() : nullptr;

    if (!freeSkel && !motionSkel) {
        ImGui::Text("No skeleton available");
        return;
    }

    dart::dynamics::SkeletonPtr refSkel = freeSkel ? freeSkel : motionSkel;

    // Filter input with deselect button
    ImGui::InputText("Filter##BonePose", mBonePoseFilter, sizeof(mBonePoseFilter));
    ImGui::SameLine();
    if (ImGui::Button("Deselect##BonePose")) {
        mBonePoseSelectedIdx = -1;
    }

    // Build filtered joint list
    std::vector<std::pair<int, std::string>> filteredJoints;
    std::string filterStr(mBonePoseFilter);
    std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

    for (size_t i = 0; i < refSkel->getNumJoints(); ++i) {
        auto* joint = refSkel->getJoint(i);
        std::string name = joint->getName();
        std::string nameLower = name;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);

        if (filterStr.empty() || nameLower.find(filterStr) != std::string::npos) {
            filteredJoints.push_back({static_cast<int>(i), name});
        }
    }

    // Joint listbox
    if (ImGui::BeginListBox("##BonePoseList", ImVec2(-1, 100))) {
        for (const auto& [idx, name] : filteredJoints) {
            bool isSelected = (mBonePoseSelectedIdx == idx);
            if (ImGui::Selectable(name.c_str(), isSelected)) {
                mBonePoseSelectedIdx = idx;
            }
        }
        ImGui::EndListBox();
    }

    // Display selected bone pose for both characters (like Joint Offset section)
    if (mBonePoseSelectedIdx >= 0 && mBonePoseSelectedIdx < (int)refSkel->getNumJoints()) {
        auto* freeJoint = freeSkel ? freeSkel->getJoint(mBonePoseSelectedIdx) : nullptr;
        auto* motionJoint = motionSkel ? motionSkel->getJoint(mBonePoseSelectedIdx) : nullptr;

        auto* refJoint = freeJoint ? freeJoint : motionJoint;

        ImGui::Separator();
        ImGui::Text("Bone: %s", refJoint->getName().c_str());

        // Get poses from both skeletons
        Eigen::VectorXd freePose = freeSkel ? freeSkel->getPositions() : Eigen::VectorXd();
        Eigen::VectorXd motionPose = motionSkel ? motionSkel->getPositions() : Eigen::VectorXd();

        // Helper lambda to compute rotation matrix from axis-angle
        auto toRotationMatrix = [](const Eigen::Vector3d& axisAngle) -> Eigen::Matrix3d {
            double angle = axisAngle.norm();
            if (angle < 1e-10) return Eigen::Matrix3d::Identity();
            return Eigen::AngleAxisd(angle, axisAngle.normalized()).toRotationMatrix();
        };

        // Colors
        ImVec4 defaultColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImVec4 redColor(1.0f, 0.3f, 0.3f, 1.0f);

        auto valuesDiffer = [](double a, double b) -> bool {
            return std::abs(a - b) >= 0.0005;
        };

        // Legend
        ImGui::Text("Values: Free /");
        ImGui::SameLine();
        ImGui::TextColored(redColor, "Motion");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(red = different)");

        // Get transforms - FreeChar always has 6 DOF per joint
        Eigen::Vector3d freeTrans = Eigen::Vector3d::Zero();
        Eigen::Matrix3d freeR = Eigen::Matrix3d::Identity();
        if (freeJoint && freePose.size() > 0) {
            int dofIdx = freeJoint->getIndexInSkeleton(0);
            freeTrans = freePose.segment<3>(dofIdx + 3);
            Eigen::Vector3d freeRot = freePose.segment<3>(dofIdx);
            freeR = toRotationMatrix(freeRot);
        }

        Eigen::Vector3d motionTrans = Eigen::Vector3d::Zero();
        Eigen::Matrix3d motionR = Eigen::Matrix3d::Identity();
        if (motionJoint && motionPose.size() > 0) {
            int dofIdx = motionJoint->getIndexInSkeleton(0);
            int numDofs = motionJoint->getNumDofs();
            if (numDofs >= 6) {
                motionTrans = motionPose.segment<3>(dofIdx + 3);
                Eigen::Vector3d motionRot = motionPose.segment<3>(dofIdx);
                motionR = toRotationMatrix(motionRot);
            } else if (numDofs == 3) {
                Eigen::Vector3d motionRot = motionPose.segment<3>(dofIdx);
                motionR = toRotationMatrix(motionRot);
            } else if (numDofs == 1) {
                // For revolute joint, get rotation from body node transform
                auto* bn = motionJoint->getChildBodyNode();
                if (bn) {
                    motionR = bn->getTransform().linear();
                    motionTrans = bn->getTransform().translation();
                }
            }
        }

        // Display translation
        ImGui::Text("Translation:");
        bool xDiff = valuesDiffer(freeTrans.x(), motionTrans.x());
        bool yDiff = valuesDiffer(freeTrans.y(), motionTrans.y());
        bool zDiff = valuesDiffer(freeTrans.z(), motionTrans.z());

        ImGui::Text("  X: %.3f /", freeTrans.x());
        ImGui::SameLine();
        ImGui::TextColored(xDiff ? redColor : defaultColor, "%.3f", motionTrans.x());

        ImGui::Text("  Y: %.3f /", freeTrans.y());
        ImGui::SameLine();
        ImGui::TextColored(yDiff ? redColor : defaultColor, "%.3f", motionTrans.y());

        ImGui::Text("  Z: %.3f /", freeTrans.z());
        ImGui::SameLine();
        ImGui::TextColored(zDiff ? redColor : defaultColor, "%.3f", motionTrans.z());

        // Display rotation matrix
        ImGui::Text("Rotation:");
        for (int row = 0; row < 3; ++row) {
            bool rowDiff = valuesDiffer(freeR(row,0), motionR(row,0)) ||
                           valuesDiffer(freeR(row,1), motionR(row,1)) ||
                           valuesDiffer(freeR(row,2), motionR(row,2));

            ImGui::Text("  [%6.3f %6.3f %6.3f] /", freeR(row,0), freeR(row,1), freeR(row,2));
            ImGui::SameLine();
            ImGui::TextColored(rowDiff ? redColor : defaultColor, "[%6.3f %6.3f %6.3f]",
                motionR(row,0), motionR(row,1), motionR(row,2));
        }
    }
}

void C3DProcessorApp::drawJointAngleSection()
{
    if (!collapsingHeaderWithControls("Joint Angle")) return;

    // Get both skeletons
    dart::dynamics::SkeletonPtr freeSkel = mFreeCharacter ? mFreeCharacter->getSkeleton() : nullptr;
    dart::dynamics::SkeletonPtr motionSkel = mMotionCharacter ? mMotionCharacter->getSkeleton() : nullptr;

    if (!freeSkel && !motionSkel) {
        ImGui::Text("No skeleton available");
        return;
    }

    dart::dynamics::SkeletonPtr refSkel = freeSkel ? freeSkel : motionSkel;

    // Filter input with deselect button
    ImGui::InputText("Filter##JointAngle", mJointAngleFilter, sizeof(mJointAngleFilter));
    ImGui::SameLine();
    if (ImGui::Button("Deselect##JointAngle")) {
        mJointAngleSelectedIdx = -1;
    }
    std::string filterStr(mJointAngleFilter);
    std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

    // Build filtered joint list
    std::vector<std::pair<int, std::string>> filteredJoints;
    for (size_t i = 0; i < refSkel->getNumJoints(); ++i) {
        auto* joint = refSkel->getJoint(i);
        std::string name = joint->getName();
        std::string nameLower = name;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);

        if (filterStr.empty() || nameLower.find(filterStr) != std::string::npos) {
            filteredJoints.push_back({static_cast<int>(i), name});
        }
    }

    // Single joint listbox
    if (ImGui::BeginListBox("##JointAngleList", ImVec2(-1, 100))) {
        for (const auto& [idx, name] : filteredJoints) {
            bool isSelected = (mJointAngleSelectedIdx == idx);
            if (ImGui::Selectable(name.c_str(), isSelected)) {
                mJointAngleSelectedIdx = idx;
            }
        }
        ImGui::EndListBox();
    }

    // Display joint angle for both characters
    if (mJointAngleSelectedIdx >= 0 && mJointAngleSelectedIdx < (int)refSkel->getNumJoints()) {
        ImGui::Separator();
        ImGui::Text("Joint: %s", refSkel->getJoint(mJointAngleSelectedIdx)->getName().c_str());

        // Helper lambda to display joint angle for one skeleton
        auto displayJointAngle = [](dart::dynamics::SkeletonPtr skel, int jointIdx, const char* label, ImVec4 color) {
            if (!skel || jointIdx < 0 || jointIdx >= (int)skel->getNumJoints()) {
                ImGui::TextColored(color, "=== %s === (N/A)", label);
                return;
            }

            auto* joint = skel->getJoint(jointIdx);
            int dofIdx = joint->getIndexInSkeleton(0);
            int numDofs = joint->getNumDofs();
            Eigen::VectorXd pose = skel->getPositions();

            ImGui::TextColored(color, "=== %s ===", label);
            ImGui::Text("Type: %s (%d DOF)", joint->getType().c_str(), numDofs);

            if (numDofs >= 6) {
                // FreeJoint
                Eigen::Vector3d trans = pose.segment<3>(dofIdx + 3);
                Eigen::Vector3d rot = pose.segment<3>(dofIdx);
                double angle = rot.norm();
                Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
                if (angle > 1e-10) {
                    R = Eigen::AngleAxisd(angle, rot.normalized()).toRotationMatrix();
                }

                ImGui::Text("Translation: %.3f %.3f %.3f", trans.x(), trans.y(), trans.z());
                ImGui::Text("Rotation:");
                for (int row = 0; row < 3; ++row) {
                    ImGui::Text("  [%6.3f %6.3f %6.3f]", R(row,0), R(row,1), R(row,2));
                }

            } else if (numDofs == 3) {
                // BallJoint
                Eigen::Vector3d rot = pose.segment<3>(dofIdx);
                double angle = rot.norm();
                Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
                if (angle > 1e-10) {
                    R = Eigen::AngleAxisd(angle, rot.normalized()).toRotationMatrix();
                }

                ImGui::Text("Rotation:");
                for (int row = 0; row < 3; ++row) {
                    ImGui::Text("  [%6.3f %6.3f %6.3f]", R(row,0), R(row,1), R(row,2));
                }

            } else if (numDofs == 1) {
                // RevoluteJoint
                double angleVal = pose(dofIdx);
                ImGui::Text("Angle: %.3f rad (%.1f deg)", angleVal, angleVal * 180.0 / M_PI);
            }
        };

        // FreeChar section
        displayJointAngle(freeSkel, mJointAngleSelectedIdx, "FreeChar", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));

        ImGui::Separator();

        // MotionChar section
        displayJointAngle(motionSkel, mJointAngleSelectedIdx, "MotionChar", ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
    }
}

void C3DProcessorApp::drawJointOffsetSection()
{
    if (!collapsingHeaderWithControls("Joint Offset")) return;

    // Get both skeletons
    dart::dynamics::SkeletonPtr freeSkel = mFreeCharacter ? mFreeCharacter->getSkeleton() : nullptr;
    dart::dynamics::SkeletonPtr motionSkel = mMotionCharacter ? mMotionCharacter->getSkeleton() : nullptr;

    if (!freeSkel && !motionSkel) {
        ImGui::Text("No skeleton available");
        return;
    }

    // Use freeSkel as reference for joint list (both should have same structure)
    dart::dynamics::SkeletonPtr refSkel = freeSkel ? freeSkel : motionSkel;

    // Filter input with deselect button
    ImGui::InputText("Filter##JointOffset", mJointOffsetFilter, sizeof(mJointOffsetFilter));
    ImGui::SameLine();
    if (ImGui::Button("Deselect##JointOffset")) {
        mJointOffsetSelectedIdx = -1;
    }

    // Build filtered joint list
    std::vector<std::pair<int, std::string>> filteredJoints;
    std::string filterStr(mJointOffsetFilter);
    std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

    for (size_t i = 0; i < refSkel->getNumJoints(); ++i) {
        auto* joint = refSkel->getJoint(i);
        std::string name = joint->getName();
        std::string nameLower = name;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);

        if (filterStr.empty() || nameLower.find(filterStr) != std::string::npos) {
            filteredJoints.push_back({static_cast<int>(i), name});
        }
    }

    // Joint listbox
    if (ImGui::BeginListBox("##JointOffsetList", ImVec2(-1, 120))) {
        for (const auto& [idx, name] : filteredJoints) {
            bool isSelected = (mJointOffsetSelectedIdx == idx);
            if (ImGui::Selectable(name.c_str(), isSelected)) {
                mJointOffsetSelectedIdx = idx;
            }
        }
        ImGui::EndListBox();
    }

    // Display selected joint offset info - comparing both characters
    if (mJointOffsetSelectedIdx >= 0 && mJointOffsetSelectedIdx < (int)refSkel->getNumJoints()) {
        // Get joints from both skeletons
        auto* freeJoint = freeSkel ? freeSkel->getJoint(mJointOffsetSelectedIdx) : nullptr;
        auto* motionJoint = motionSkel ? motionSkel->getJoint(mJointOffsetSelectedIdx) : nullptr;

        auto* refJoint = freeJoint ? freeJoint : motionJoint;
        auto* parentBody = refJoint->getParentBodyNode();
        auto* childBody = refJoint->getChildBodyNode();

        ImGui::Separator();
        ImGui::Text("Joint: %s", refJoint->getName().c_str());
        ImGui::Text("Type: %s", refJoint->getType().c_str());

        // Legend
        ImGui::Text("Values: Free /");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Motion");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(red = different)");

        // Colors
        ImVec4 redColor(1.0f, 0.3f, 0.3f, 1.0f);
        ImVec4 defaultColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);

        // Helper to check if values differ at 3 decimal precision
        auto valuesDiffer = [](double a, double b) -> bool {
            return std::abs(a - b) >= 0.0005;  // 3 decimal precision threshold
        };

        // Helper to display comparison values
        auto displayTranslationCompare = [&](const char* label,
                                              const Eigen::Vector3d& freeTrans,
                                              const Eigen::Vector3d& motionTrans,
                                              bool hasFree, bool hasMotion) {
            ImGui::Text("%s", label);
            if (hasFree && hasMotion) {
                bool xDiff = valuesDiffer(freeTrans.x(), motionTrans.x());
                bool yDiff = valuesDiffer(freeTrans.y(), motionTrans.y());
                bool zDiff = valuesDiffer(freeTrans.z(), motionTrans.z());

                ImGui::Text("  X: %.3f /", freeTrans.x());
                ImGui::SameLine();
                ImGui::TextColored(xDiff ? redColor : defaultColor, "%.3f", motionTrans.x());

                ImGui::Text("  Y: %.3f /", freeTrans.y());
                ImGui::SameLine();
                ImGui::TextColored(yDiff ? redColor : defaultColor, "%.3f", motionTrans.y());

                ImGui::Text("  Z: %.3f /", freeTrans.z());
                ImGui::SameLine();
                ImGui::TextColored(zDiff ? redColor : defaultColor, "%.3f", motionTrans.z());
            } else if (hasFree) {
                ImGui::Text("  X: %.3f  Y: %.3f  Z: %.3f", freeTrans.x(), freeTrans.y(), freeTrans.z());
            } else if (hasMotion) {
                ImGui::TextColored(redColor, "  X: %.3f  Y: %.3f  Z: %.3f", motionTrans.x(), motionTrans.y(), motionTrans.z());
            }
        };

        auto displayRotationCompare = [&](const char* label,
                                           const Eigen::Matrix3d& freeRot,
                                           const Eigen::Matrix3d& motionRot,
                                           bool hasFree, bool hasMotion) {
            ImGui::Text("%s", label);
            for (int row = 0; row < 3; ++row) {
                if (hasFree && hasMotion) {
                    // Check if any value in this row differs
                    bool rowDiff = valuesDiffer(freeRot(row,0), motionRot(row,0)) ||
                                   valuesDiffer(freeRot(row,1), motionRot(row,1)) ||
                                   valuesDiffer(freeRot(row,2), motionRot(row,2));

                    ImGui::Text("  [%6.3f %6.3f %6.3f] /", freeRot(row,0), freeRot(row,1), freeRot(row,2));
                    ImGui::SameLine();
                    ImGui::TextColored(rowDiff ? redColor : defaultColor, "[%6.3f %6.3f %6.3f]", motionRot(row,0), motionRot(row,1), motionRot(row,2));
                } else if (hasFree) {
                    ImGui::Text("  [%6.3f %6.3f %6.3f]", freeRot(row,0), freeRot(row,1), freeRot(row,2));
                } else if (hasMotion) {
                    ImGui::TextColored(redColor, "  [%6.3f %6.3f %6.3f]", motionRot(row,0), motionRot(row,1), motionRot(row,2));
                }
            }
        };

        // Parent body info
        ImGui::Separator();
        if (parentBody) {
            ImGui::Text("Parent Body: %s", parentBody->getName().c_str());

            Eigen::Isometry3d freeT = freeJoint ? freeJoint->getTransformFromParentBodyNode() : Eigen::Isometry3d::Identity();
            Eigen::Isometry3d motionT = motionJoint ? motionJoint->getTransformFromParentBodyNode() : Eigen::Isometry3d::Identity();

            displayTranslationCompare("Offset from Parent (translation):",
                freeT.translation(), motionT.translation(),
                freeJoint != nullptr, motionJoint != nullptr);

            displayRotationCompare("Offset from Parent (rotation):",
                freeT.linear(), motionT.linear(),
                freeJoint != nullptr, motionJoint != nullptr);
        } else {
            ImGui::Text("Parent Body: (World/Root)");
        }

        // Child body info
        ImGui::Separator();
        if (childBody) {
            ImGui::Text("Child Body: %s", childBody->getName().c_str());

            Eigen::Isometry3d freeT = freeJoint ? freeJoint->getTransformFromChildBodyNode() : Eigen::Isometry3d::Identity();
            Eigen::Isometry3d motionT = motionJoint ? motionJoint->getTransformFromChildBodyNode() : Eigen::Isometry3d::Identity();

            displayTranslationCompare("Offset to Child (translation):",
                freeT.translation(), motionT.translation(),
                freeJoint != nullptr, motionJoint != nullptr);

            displayRotationCompare("Offset to Child (rotation):",
                freeT.linear(), motionT.linear(),
                freeJoint != nullptr, motionJoint != nullptr);
        }
    }
}

// =============================================================================
// C3D Processing
// =============================================================================

void C3DProcessorApp::scanC3DFiles()
{
    mMotionList.clear();
    mMotionDisplayNames.clear();

    if (!mResourceManager) {
        LOG_WARN("[C3DProcessor] Resource manager not initialized");
        return;
    }

    // Search paths using RM endpoints
    std::vector<std::string> searchPatterns = {"@data/c3d", "@data/motion"};

    // Temporary storage: pairs of (full_path, display_name)
    std::vector<std::pair<std::string, std::string>> fileEntries;

    for (const auto& pattern : searchPatterns) {
        try {
            // Resolve directory path
            std::string dirPath = mResourceManager->resolveDir(pattern);
            if (dirPath.empty() || !fs::exists(dirPath)) continue;

            fs::path basePath(dirPath);

            // Recursively scan for .c3d files
            for (const auto& entry : fs::recursive_directory_iterator(dirPath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".c3d") {
                        std::string fullPath = entry.path().string();
                        // Compute relative path for display
                        std::string displayName = fs::relative(entry.path(), basePath).string();
                        fileEntries.emplace_back(fullPath, displayName);
                    }
                }
            }
        } catch (const rm::RMError& e) {
            LOG_WARN("[C3DProcessor] Error scanning " << pattern << ": " << e.what());
        } catch (const std::exception& e) {
            LOG_WARN("[C3DProcessor] Error scanning " << pattern << ": " << e.what());
        }
    }

    // Sort by display name
    std::sort(fileEntries.begin(), fileEntries.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Populate parallel vectors
    for (const auto& entry : fileEntries) {
        mMotionList.push_back(entry.first);
        mMotionDisplayNames.push_back(entry.second);
    }

    LOG_INFO("[C3DProcessor] Found " << mMotionList.size() << " C3D files");
}

void C3DProcessorApp::loadC3DFile(const std::string& path)
{
    if (!mC3DReader) {
        LOG_ERROR("[C3DProcessor] C3D reader not initialized");
        return;
    }

    // Unload previous motion
    if (mMotion) {
        delete mMotion;
        mMotion = nullptr;
    }

    // Clear state
    mMotionState = C3DViewerState();
    mGraphData->clear_all();

    // Load C3D (markers only - calibration triggered separately via UI)
    mMotion = mC3DReader->loadC3DMarkersOnly(path);

    if (mMotion) {
        mMotionPath = path;
        mMotionState.maxFrameIndex = std::max(0, mMotion->getNumFrames() - 1);

        // Compute cycle distance
        C3D* c3dMotion = static_cast<C3D*>(mMotion);
        mMotionState.cycleDistance = computeMarkerCycleDistance(c3dMotion); // Character does not progress forward

        // Height calibration at first loading
        if (c3dMotion->getNumFrames() > 0) {
            auto markers = c3dMotion->getMarkers(0);
            double heightOffset = computeMarkerHeightCalibration(markers);
            mMotionState.displayOffset = Eigen::Vector3d(0, heightOffset, 0);
        }

        // Detect medial markers for static calibration
        mHasMedialMarkers = C3D_Reader::hasMedialMarkers(c3dMotion->getLabels());
        if (mHasMedialMarkers) {
            LOG_INFO("[C3DProcessor] Medial markers detected - static calibration available");
        }

        // Initialize viewer time
        mViewerTime = 0.0;
        mViewerPhase = 0.0;

        LOG_INFO("[C3DProcessor] Loaded: " << path << " (" << mMotion->getNumFrames() << " frames)");
    } else {
        LOG_ERROR("[C3DProcessor] Failed to load: " << path);
    }
}

Eigen::Vector3d C3DProcessorApp::computeMarkerCycleDistance(C3D* markerData)
{
    Eigen::Vector3d cycleDistance = Eigen::Vector3d::Zero();

    if (!markerData || markerData->getNumFrames() == 0)
        return cycleDistance;

    Eigen::Vector3d firstCentroid, lastCentroid;
    if (C3D::computeCentroid(markerData->getMarkers(0), firstCentroid) &&
        C3D::computeCentroid(markerData->getMarkers(markerData->getNumFrames() - 1), lastCentroid))
    {
        cycleDistance = lastCentroid - firstCentroid;
    }

    return cycleDistance;
}

C3DProcessorApp::MarkerPlaybackContext C3DProcessorApp::computeMarkerPlayback()
{
    MarkerPlaybackContext context;
    context.state = &mMotionState;
    context.phase = mViewerPhase;

    if (!mMotion || mMotion->getSourceType() != "c3d") return context;

    C3D* c3dMotion = static_cast<C3D*>(mMotion);
    if (c3dMotion->getNumFrames() == 0)
    {
        LOG_WARN("[C3DProcessor] No frames in motion, skipping marker playback");
        return context;
    }

    context.markers = c3dMotion;
    context.totalFrames = c3dMotion->getNumFrames();
    context.valid = true;

    C3DViewerState& markerState = *context.state;

    if (markerState.navigationMode == C3D_MANUAL_FRAME) {
        int maxFrame = std::max(0, context.totalFrames - 1);
        markerState.manualFrameIndex = std::clamp(markerState.manualFrameIndex, 0, maxFrame);
        context.frameIndex = markerState.manualFrameIndex;
        context.frameFloat = static_cast<double>(context.frameIndex);
        return context;
    }

    // Compute frame from phase
    double totalFrames = static_cast<double>(context.totalFrames);
    context.frameFloat = mViewerPhase * (totalFrames - 1);

    double wrapped = context.frameFloat;
    if (wrapped < 0.0) wrapped = 0.0;
    if (wrapped >= totalFrames) wrapped = std::fmod(wrapped, totalFrames);

    context.frameIndex = static_cast<int>(std::floor(wrapped + 1e-9));
    context.frameIndex = std::clamp(context.frameIndex, 0, context.totalFrames - 1);
    context.frameFloat = wrapped;

    return context;
}

void C3DProcessorApp::evaluateMarkerPlayback(const MarkerPlaybackContext& context)
{
    if (!context.valid || !context.markers || !context.state)
        return;

    C3DViewerState& markerState = *context.state;

    if (markerState.navigationMode == C3D_MANUAL_FRAME) {
        int maxFrame = std::max(0, context.totalFrames - 1);
        markerState.manualFrameIndex = std::clamp(markerState.manualFrameIndex, 0, maxFrame);
        markerState.currentMarkers = context.markers->getMarkers(markerState.manualFrameIndex);
        markerState.cycleAccumulation.setZero();
        markerState.lastFrameIdx = markerState.manualFrameIndex;
    } else {
        double frameFloat = context.frameFloat;
        const double totalFrames = static_cast<double>(context.totalFrames);
        double wrapped = frameFloat;
        if (wrapped < 0.0) wrapped = 0.0;
        if (wrapped >= totalFrames) wrapped = std::fmod(wrapped, totalFrames);

        int currentIdx = static_cast<int>(std::floor(wrapped + 1e-8));
        currentIdx = std::clamp(currentIdx, 0, context.totalFrames - 1);

        auto interpolated = context.markers->getInterpolatedMarkers(wrapped);
        if (interpolated.empty()) return;

        if (currentIdx < markerState.lastFrameIdx && mProgressForward) {
            markerState.cycleAccumulation += markerState.cycleDistance;
        }

        markerState.currentMarkers = std::move(interpolated);
        markerState.lastFrameIdx = currentIdx;
    }

    // Update skeleton pose from C3D
    if (mFreeCharacter && mMotion) {
        auto skel = mFreeCharacter->getSkeleton();
        if (skel && mMotion->getValuesPerFrame() == skel->getNumDofs()) {
            int frame = markerState.lastFrameIdx;
            Eigen::VectorXd pose = mMotion->getPose(frame);
            if (pose.size() == skel->getNumDofs()) {
                // Apply display offset
                pose.segment<3>(3) += markerState.displayOffset + markerState.cycleAccumulation;
                mMotionState.currentPose = pose;
                skel->setPositions(pose);
            }
        }
    }

    // Update motion character pose from converted poses
    if (mMotionCharacter && mC3DReader) {
        const auto& motionResult = mC3DReader->getMotionConversionResult();
        if (motionResult.valid && markerState.lastFrameIdx < (int)motionResult.motionPoses.size()) {
            auto skel = mMotionCharacter->getSkeleton();
            Eigen::VectorXd pose = motionResult.motionPoses[markerState.lastFrameIdx];
            if (pose.size() == skel->getNumDofs()) {
                // Apply display offset
                pose.segment<3>(3) += markerState.displayOffset + markerState.cycleAccumulation;
                skel->setPositions(pose);
            }
        }
    }
}

double C3DProcessorApp::computeMarkerHeightCalibration(const std::vector<Eigen::Vector3d>& markers)
{
    double lowest_y = std::numeric_limits<double>::max();

    for (const auto& marker : markers) {
        if (!marker.array().isFinite().all()) continue;
        if (marker[1] < lowest_y) {
            lowest_y = marker[1];
        }
    }

    if (lowest_y == std::numeric_limits<double>::max()) {
        return 0.0;
    }

    const double SAFETY_MARGIN = 1E-3;
    return -lowest_y + SAFETY_MARGIN;
}

void C3DProcessorApp::updateViewerTime(double dt)
{
    mViewerTime += dt;

    // Compute phase
    if (mViewerCycleDuration > 0.0) {
        mViewerPhase = std::fmod(mViewerTime / mViewerCycleDuration, 1.0);
        if (mViewerPhase < 0.0) mViewerPhase += 1.0;
    }

    // Compute marker playback
    MarkerPlaybackContext context = computeMarkerPlayback();
    evaluateMarkerPlayback(context);

    // Compute metrics
    computeViewerMetric();
}

void C3DProcessorApp::computeViewerMetric()
{
    if (!mMotion || mMotion->getSourceType() != "c3d" || !mFreeCharacter || !mC3DReader) return;

    const auto& skelMarkers = mFreeCharacter->getMarkers();
    const auto& config = mC3DReader->getFittingConfig();
    const auto& mappings = config.markerMappings;
    if (mappings.empty()) return;

    auto expectedMarkers = mFreeCharacter->getExpectedMarkerPositions();
    Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;

    double totalError = 0.0;
    int validCount = 0;

    for (size_t i = 0; i < mappings.size(); ++i) {
        const auto& mapping = mappings[i];
        int dataIdx = mapping.dataIndex;

        int skelIdx = -1;
        for (size_t j = 0; j < skelMarkers.size(); ++j) {
            if (skelMarkers[j].name == mapping.name) {
                skelIdx = static_cast<int>(j);
                break;
            }
        }

        bool hasDataM = dataIdx >= 0 && dataIdx < (int)mMotionState.currentMarkers.size();
        bool hasSkelM = skelIdx >= 0 && skelIdx < (int)expectedMarkers.size();
        Eigen::Vector3d dataM = hasDataM ? mMotionState.currentMarkers[dataIdx] : Eigen::Vector3d::Zero();
        Eigen::Vector3d skelM = hasSkelM ? expectedMarkers[skelIdx] : Eigen::Vector3d::Zero();
        Eigen::Vector3d offsetDataM = dataM + markerOffset;
        bool dataValid = hasDataM && dataM.array().isFinite().all();
        bool skelValid = hasSkelM && skelM.array().isFinite().all();

        if (dataValid && skelValid) {
            double err = (offsetDataM - skelM).norm() * 1000;  // mm
            totalError += err;
            validCount++;
        }
    }

    if (validCount > 0) {
        double meanError = totalError / validCount;
        mGraphData->push("marker_error_mean", meanError);
    }
}


// =============================================================================
// Input Handling (keyPress only - callbacks handled by ViewerAppBase)
// =============================================================================

void C3DProcessorApp::keyPress(int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_SPACE:
                mIsPlaying = !mIsPlaying;
                return;  // Handled
            case GLFW_KEY_S:
                // Step single frame forward
                if (mMotion) {
                    mIsPlaying = false;
                    double frameTime = mMotion->getFrameTime();
                    updateViewerTime(frameTime);
                }
                return;  // Handled
            case GLFW_KEY_R:
                resetPlaybackAndState();
                return;  // Handled
            case GLFW_KEY_1:
            case GLFW_KEY_KP_1:
                alignCameraToPlaneQuat(1);  // XY plane
                return;  // Handled
            case GLFW_KEY_2:
            case GLFW_KEY_KP_2:
                alignCameraToPlaneQuat(2);  // YZ plane
                return;  // Handled
            case GLFW_KEY_3:
            case GLFW_KEY_KP_3:
                alignCameraToPlaneQuat(3);  // ZX plane
                return;  // Handled
            case GLFW_KEY_C:
                // Reload current motion with calibration
                reloadCurrentMotion(true);
                return;  // Handled
            case GLFW_KEY_L:
                mRenderMarkerIndices = !mRenderMarkerIndices;
                return;  // Handled
            case GLFW_KEY_O:
                mRenderMode = static_cast<RenderMode>((static_cast<int>(mRenderMode) + 1) % 3);
                return;  // Handled
            case GLFW_KEY_V:
                hideVirtualMarkers();
                return;  // Handled
            case GLFW_KEY_M:
                // Toggle all marker rendering
                {
                    bool anyVisible = mRenderC3DMarkers || mRenderExpectedMarkers ||
                                      mRenderMotionCharMarkers;
                    bool newState = !anyVisible;
                    mRenderC3DMarkers = newState;
                    mRenderExpectedMarkers = newState;
                    mRenderMotionCharMarkers = newState;
                }
                return;  // Handled
        }
    }

    // Delegate common keys (F, G, ESC) to base class
    ViewerAppBase::keyPress(key, scancode, action, mods);
}

void C3DProcessorApp::alignCameraToPlaneQuat(int plane)
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

void C3DProcessorApp::resetPlaybackAndState()
{
    // Reset playback state
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mMotionState.lastFrameIdx = 0;
    mMotionState.cycleAccumulation.setZero();
    mMotionState.manualFrameIndex = 0;
    mGraphData->clear_all();

    // Refresh marker and skeleton data
    MarkerPlaybackContext context = computeMarkerPlayback();
    evaluateMarkerPlayback(context);

    // Reset camera position (use base class camera struct)
    mCamera.zoom = 1.0;
    mCamera.trans = Eigen::Vector3d(0.0, -0.8, 0.0);
    mCamera.relTrans.setZero();
    mCamera.trackball = dart::gui::Trackball();

    // Reset skeleton to zero pose
    RenderCharacter* characters[2] = { mFreeCharacter.get(), mMotionCharacter.get() };
    for (auto character : characters) {
        if (character) {
            character->resetSkeletonToDefault();
            auto skel = character->getSkeleton();
            if (skel) {
                skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));
            }
        }
    }

    // Reset currentPose to true zero (joint offset handles default height)
    if (mFreeCharacter && mFreeCharacter->getSkeleton()) {
        auto skel = mFreeCharacter->getSkeleton();
        mMotionState.currentPose = Eigen::VectorXd::Zero(skel->getNumDofs());
    }
    // MotionCharacter pose already set to zero by resetSkeletonToDefault()
}

void C3DProcessorApp::hideVirtualMarkers()
{
    if (!mMotion || mMotion->getSourceType() != "c3d") return;

    C3D* c3dMotion = static_cast<C3D*>(mMotion);
    const auto& labels = c3dMotion->getLabels();

    int hiddenCount = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        const std::string& label = labels[i];
        // Check if label starts with "V_"
        if (label.size() >= 2 && label[0] == 'V' && label[1] == '_') {
            mHiddenC3DMarkers.insert(static_cast<int>(i));
            hiddenCount++;
        }
    }

    LOG_INFO("[C3DProcessor] Hidden " << hiddenCount << " virtual markers (V_*)");
}

void C3DProcessorApp::clearMotionAndZeroPose()
{
    // Clear current motion
    if (mMotion) {
        delete mMotion;
        mMotion = nullptr;
    }
    mSelectedMotion = -1;
    mMotionPath.clear();

    // Reset motion state
    mMotionState.currentPose.setZero();
    mMotionState.currentMarkers.clear();
    mMotionState.lastFrameIdx = 0;
    mMotionState.cycleAccumulation.setZero();
    mMotionState.displayOffset.setZero();
    mMotionState.manualFrameIndex = 0;

    // Zero pose both skeletons
    RenderCharacter* characters[2] = { mFreeCharacter.get(), mMotionCharacter.get() };
    for (auto character : characters) {
        if (character) {
            auto skel = character->getSkeleton();
            if (skel) {
                skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));
            }
        }
    }

    // Reset viewer time
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mIsPlaying = false;

    LOG_INFO("[C3DProcessor] Cleared motion and set zero pose");
}

// =============================================================================
// Clinical Data (PID) Section
// =============================================================================

void C3DProcessorApp::drawClinicalDataSection()
{
    if (!collapsingHeaderWithControls("Clinical Data")) {
        return;
    }

    // PID Navigator UI
    if (!mPIDNavigator) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "PID Navigator not initialized");
        return;
    }

    mPIDNavigator->renderInlineSelector(150, 150);

    // Calibration Status Section (render below navigator)
    const auto& pidState = mPIDNavigator->getState();
    if (pidState.selectedPID >= 0) {
        ImGui::Separator();
        ImGui::Text("Static Calibration:");

        // Build pattern for calibration path
        const std::string& pid = pidState.pidList[pidState.selectedPID];
        std::string prePost = pidState.preOp ? "pre" : "post";
        std::string pattern = "@pid:" + pid + "/gait/" + prePost;

        // Use cached calibration availability (updated when PID/prePost changes)
        // Display calibration status with color coding
        if (mPersonalizedCalibrationLoaded) {
            ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "Loaded");
        } else if (mHasPersonalizedCalibration) {
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "Available");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Not Available");
        }

        // Load calibration button
        ImGui::SameLine();
        if (ImGui::Button("Load calibration")) {
            if (mHasPersonalizedCalibration && mResourceManager) {
                try {
                    // Use resolveDir for directory path (not resolve which is for files)
                    auto resolved = mResourceManager->resolveDir(pattern);
                    if (loadPersonalizedCalibration(resolved.string())) {
                        mPersonalizedCalibrationLoaded = true;
                    }
                } catch (const rm::RMError& e) {
                    LOG_ERROR("[C3DProcessor] Failed to load calibration: " << e.what());
                }
            }
        }

        // Export skeleton section
        if (!mResourceManager) return;

        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##exportCalibName", mExportCalibrationName, sizeof(mExportCalibrationName));
        ImGui::SameLine();

        // Check if export file exists
        std::string outputDir = mResourceManager->resolveDir(pattern);
        std::string skelDir = outputDir + "/skeleton";
        if (!std::filesystem::exists(skelDir)) {
            std::filesystem::create_directories(skelDir);
        }
        std::string skelPath = skelDir + "/" + mExportCalibrationName + ".yaml";
        bool fileExists = std::filesystem::exists(skelPath);

        if (fileExists) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Overwrite");
            ImGui::SameLine();
        }
        if (ImGui::Button("Export Skeleton")) {
            if (mMotionCharacter) {
                mMotionCharacter->exportSkeletonYAML(skelPath);
                LOG_INFO("[C3DProcessor] Exported skeleton to: " << skelPath);
            }
        }

        // Reset calibration button
        if (mPersonalizedCalibrationLoaded) {
            if (ImGui::Button("Reset Calibration")) {
                // Reload initial markers and reset skeleton
                if (mFreeCharacter) {
                    mFreeCharacter->loadMarkers(mInitialMarkerPath);
                    mFreeCharacter->resetSkeletonToDefault();
                }
                if (mMotionCharacter) {
                    mMotionCharacter->loadMarkers(mInitialMarkerPath);
                    mMotionCharacter->resetSkeletonToDefault();
                }
                mMarkerConfigPath = mInitialMarkerPath;
                mPersonalizedScalePath.clear();
                mPersonalizedCalibrationLoaded = false;
                LOG_INFO("[C3DProcessor] Reset calibration to initial state");
            }
        }
    }
}

bool C3DProcessorApp::loadPersonalizedCalibration(const std::string& inputDir)
{
    // Load personalized markers and body scales from given directory
    // Input files:
    //   - static_calibrated_marker.xml
    //   - static_calibrated_body_scale.yaml

    std::string markerPath = inputDir + "/static_calibrated_marker.xml";
    std::string scalePath = inputDir + "/static_calibrated_body_scale.yaml";

    // Check if files exist
    if (!std::filesystem::exists(markerPath)) {
        LOG_WARN("[Load] Marker file not found: " << markerPath);
        return false;
    }
    if (!std::filesystem::exists(scalePath)) {
        LOG_WARN("[Load] Body scale file not found: " << scalePath);
        return false;
    }

    // === 1. Load body scales ===
    if (mFreeCharacter && !mFreeCharacter->loadBodyScaleYAML(scalePath)) {
        LOG_ERROR("[Load] Failed to load body scales");
        return false;
    }

    // Apply same scales to motion character if available
    if (mMotionCharacter && mFreeCharacter) {
        auto& freeInfos = mFreeCharacter->getSkelInfos();
        auto& motionInfos = mMotionCharacter->getSkelInfos();
        for (size_t i = 0; i < freeInfos.size() && i < motionInfos.size(); ++i) {
            std::get<1>(motionInfos[i]) = std::get<1>(freeInfos[i]);
        }
        mMotionCharacter->applySkeletonBodyNode(motionInfos, mMotionCharacter->getSkeleton());
    }

    // === 2. Load personalized markers ===
    if (mFreeCharacter) {
        mFreeCharacter->loadMarkers(markerPath);
    }
    if (mMotionCharacter) {
        mMotionCharacter->loadMarkers(markerPath);
    }

    // Store loaded paths for display
    mMarkerConfigPath = markerPath;
    mPersonalizedScalePath = scalePath;

    LOG_INFO("[Load] Loaded personalized calibration from: " << inputDir);
    return true;
}

void C3DProcessorApp::onPIDFileSelected(const std::string& path,
                                         const std::string& filename)
{
    // Set motion source tracking
    mMotionSource = MotionSource::PID;

    // Load the C3D file using existing loader
    loadC3DFile(path);

    LOG_INFO("[C3DProcessor] Loaded PID C3D file: " << filename);
}

void C3DProcessorApp::checkForPersonalizedCalibration()
{
    mHasPersonalizedCalibration = false;

    if (!mPIDNavigator || !mResourceManager) return;

    const auto& state = mPIDNavigator->getState();
    if (state.selectedPID < 0) return;

    const std::string& pid = state.pidList[state.selectedPID];
    std::string prePost = state.preOp ? "pre" : "post";
    std::string pattern = "@pid:" + pid + "/gait/" + prePost;

    // Use exists() for quiet check - doesn't log errors for missing files
    bool exists1 = mResourceManager->exists(pattern + "/static_calibrated_marker.xml");
    bool exists2 = mResourceManager->exists(pattern + "/static_calibrated_body_scale.yaml");
    mHasPersonalizedCalibration = exists1 && exists2;
}

std::string C3DProcessorApp::getCurrentMotionPath() const
{
    switch (mMotionSource) {
        case MotionSource::FileList:
            if (mSelectedMotion >= 0 && mSelectedMotion < static_cast<int>(mMotionList.size())) {
                return mMotionList[mSelectedMotion];
            }
            break;
        case MotionSource::PID:
            // For PID source, return the stored mMotionPath (set by loadC3DFile)
            return mMotionPath;
        case MotionSource::None:
        default:
            break;
    }
    return "";
}

void C3DProcessorApp::reloadCurrentMotion(bool withCalibration)
{
    if (!mC3DReader || mMotionSource == MotionSource::None) {
        LOG_WARN("[C3DProcessor] No motion to reload");
        return;
    }

    std::string path = getCurrentMotionPath();
    if (path.empty()) {
        LOG_WARN("[C3DProcessor] Cannot determine current motion path");
        return;
    }

    // Unload previous motion
    if (mMotion) {
        delete mMotion;
        mMotion = nullptr;
    }

    // Clear state
    mMotionState = C3DViewerState();
    mGraphData->clear_all();

    // Reload with calibration option
    if (withCalibration) {
        C3DConversionParams params;
        params.doCalibration = true;
        mMotion = mC3DReader->loadC3D(path, params);
    } else {
        mMotion = mC3DReader->loadC3DMarkersOnly(path);
    }

    if (mMotion) {
        mMotionPath = path;
        mMotionState.maxFrameIndex = std::max(0, mMotion->getNumFrames() - 1);

        // Compute cycle distance
        C3D* c3dMotion = static_cast<C3D*>(mMotion);
        mMotionState.cycleDistance = computeMarkerCycleDistance(c3dMotion);

        // Height calibration
        if (c3dMotion->getNumFrames() > 0) {
            auto markers = c3dMotion->getMarkers(0);
            double heightOffset = computeMarkerHeightCalibration(markers);
            mMotionState.displayOffset = Eigen::Vector3d(0, heightOffset, 0);
        }

        // Initialize viewer time
        mViewerTime = 0.0;
        mViewerPhase = 0.0;

        LOG_INFO("[C3DProcessor] Reloaded: " << path << " (calibration=" << withCalibration << ")");
    } else {
        LOG_ERROR("[C3DProcessor] Failed to reload: " << path);
    }
}

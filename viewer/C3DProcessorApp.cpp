#include "C3DProcessorApp.h"
#include "DARTHelper.h"
#include "UriResolver.h"
#include "Log.h"
#include "PlotUtils.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>

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

C3DProcessorApp::C3DProcessorApp(const std::string& skeletonPath, const std::string& markerPath,
                                 const std::string& configPath)
    : mWindow(nullptr)
    , mWidth(1280)
    , mHeight(720)
    , mWindowXPos(0)
    , mWindowYPos(0)
    , mZoom(1.0)
    , mPersp(45.0)
    , mFocus(0)
    , mMouseDown(false)
    , mRotate(false)
    , mTranslate(false)
    , mMouseX(0)
    , mMouseY(0)
    , mSkeletonPath(skeletonPath)
    , mMarkerConfigPath(markerPath)
    , mFittingConfigPath(configPath)
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
    , mControlPanelWidth(400)
{
    std::memset(mMarkerSearchFilter, 0, sizeof(mMarkerSearchFilter));

    // Initialize graph data buffer
    mGraphData = new CBufferData<double>();
    mGraphData->register_key("marker_error_mean", 1000);

    // Initialize URI resolver
    PMuscle::URIResolver::getInstance().initialize();

    // Camera defaults
    mEye = Eigen::Vector3d(0.0, 0.0, 2.5);
    mUp = Eigen::Vector3d(0.0, 1.0, 0.0);
    mTrans = Eigen::Vector3d(0.0, -0.8, 0.0);
    mC3DCOM = Eigen::Vector3d::Zero();

    // Load config
    loadRenderConfig();

    // Initialize GLFW
    if (!glfwInit()) {
        LOG_ERROR("[C3DProcessor] Failed to initialize GLFW");
        exit(1);
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHintString(GLFW_X11_CLASS_NAME,    "C3Dprocessor");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "C3Dprocessor"); 
    mWindow = glfwCreateWindow(mWidth, mHeight, "C3Dprocessor", nullptr, nullptr);
    if (!mWindow) {
        LOG_ERROR("[C3DProcessor] Failed to create GLFW window");
        glfwTerminate();
        exit(1);
    }

    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOG_ERROR("[C3DProcessor] Failed to initialize GLAD");
        exit(1);
    }

    // Set up callbacks
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);
    glfwSetKeyCallback(mWindow, keyCallback);

    // Initialize ImGui
    initImGui();

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

    // Scan for C3D files and autoload first one
    scanC3DFiles();
    if (!mMotionList.empty()) {
        mSelectedMotion = 0;
        loadC3DFile(mMotionList[0]);
        LOG_INFO("[C3DProcessor] Autoloaded first C3D file: " << mMotionList[0]);
    }

    // Initialize Resource Manager for PID-based access
    try {
        mResourceManager = std::make_unique<rm::ResourceManager>("data/rm_config.yaml");
        scanPIDList();
    } catch (const rm::RMError& e) {
        LOG_WARN("[C3DProcessor] Resource manager init failed: " << e.what());
    } catch (const std::exception& e) {
        LOG_WARN("[C3DProcessor] Resource manager init failed: " << e.what());
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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

// =============================================================================
// Main Loop
// =============================================================================

void C3DProcessorApp::startLoop()
{
    while (!glfwWindowShouldClose(mWindow)) {
        glfwPollEvents();

        // Update time
        double currentTime = glfwGetTime();
        double dt = currentTime - mLastRealTime;
        mLastRealTime = currentTime;

        if (mIsPlaying) updateViewerTime(dt * mViewerPlaybackSpeed);
        else updateViewerTime(0.0);

        // Start ImGui frame (needed for marker label rendering in drawFrame)
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // OpenGL rendering
        drawFrame();

        // ImGui panels
        drawControlPanel();
        drawVisualizationPanel();
        drawRenderingPanel();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(mWindow);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void C3DProcessorApp::initGL()
{
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void C3DProcessorApp::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Load font with Korean glyph support
    ImFontConfig fontConfig;
    fontConfig.MergeMode = false;

    // Try to load Noto Sans CJK for Korean support
    const char* fontPath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";
    if (std::filesystem::exists(fontPath)) {
        // Load with full Korean range
        io.Fonts->AddFontFromFileTTF(fontPath, 16.0f, &fontConfig,
            io.Fonts->GetGlyphRangesKorean());
        LOG_INFO("[C3DProcessor] Loaded Korean font: " << fontPath);
    } else {
        // Fallback to default font
        io.Fonts->AddFontDefault();
        LOG_WARN("[C3DProcessor] Korean font not found, using default");
    }

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 150");
}

void C3DProcessorApp::initLighting()
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat light_position[] = {1.0f, 2.0f, 1.5f, 0.0f};
    GLfloat light_ambient[] = {0.3f, 0.3f, 0.3f, 1.0f};
    GLfloat light_diffuse[] = {0.7f, 0.7f, 0.7f, 1.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
}

void C3DProcessorApp::updateCamera()
{
    // Camera follow character (update mTrans before setCamera applies it)
    if (mFocus == 1 && mMotion != nullptr && mFreeCharacter) {
        // Calculate current position based on cycle accumulation
        double phase = mViewerPhase;
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
        mTrans[0] = -(current_pos[3] + state.cycleAccumulation[0] + state.displayOffset[0]);
        mTrans[1] = -(current_pos[4] + state.displayOffset[1]) - 1;
        mTrans[2] = -(current_pos[5] + state.cycleAccumulation[2] + state.displayOffset[2]);
    }
}

void C3DProcessorApp::setCamera()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, mWidth, mHeight);
    gluPerspective(mPersp, (double)mWidth / (double)mHeight, 0.1, 100.0);
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, 0.0, mUp[0], mUp[1], mUp[2]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    mTrackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
    mTrackball.setRadius(std::min(mWidth, mHeight) * 0.4);
    mTrackball.applyGLRotation();

    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0], mTrans[1], mTrans[2]);
}

void C3DProcessorApp::loadRenderConfig()
{
    try {
        PMuscle::URIResolver& resolver = PMuscle::URIResolver::getInstance();
        resolver.initialize();
        std::string resolved_path = resolver.resolve("render.yaml");

        YAML::Node config = YAML::LoadFile(resolved_path);

        if (config["geometry"]) {
            if (config["geometry"]["window"]) {
                if (config["geometry"]["window"]["width"])
                    mWidth = config["geometry"]["window"]["width"].as<int>();
                if (config["geometry"]["window"]["height"])
                    mHeight = config["geometry"]["window"]["height"].as<int>();
                if (config["geometry"]["window"]["xpos"])
                    mWindowXPos = config["geometry"]["window"]["xpos"].as<int>();
                if (config["geometry"]["window"]["ypos"])
                    mWindowYPos = config["geometry"]["window"]["ypos"].as<int>();
            }
            if (config["geometry"]["panels"]) {
                if (config["geometry"]["panels"]["control_panel_width"])
                    mControlPanelWidth = config["geometry"]["panels"]["control_panel_width"].as<int>();
            }
        }

        if (config["default_open_panels"]) {
            for (const auto& panel : config["default_open_panels"]) {
                mDefaultOpenPanels.insert(panel.as<std::string>());
            }
        }

        if (config["c3d"]) {
            if (config["c3d"]["motion_char_offset"]) {
                auto offset = config["c3d"]["motion_char_offset"];
                if (offset.IsSequence() && offset.size() == 3) {
                    mMotionCharacterOffset.x() = offset[0].as<double>();
                    mMotionCharacterOffset.y() = offset[1].as<double>();
                    mMotionCharacterOffset.z() = offset[2].as<double>();
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN("[C3DProcessor] Could not load render.yaml: " << e.what());
    }
}

bool C3DProcessorApp::isPanelDefaultOpen(const std::string& panelName) const
{
    return mDefaultOpenPanels.find(panelName) != mDefaultOpenPanels.end();
}

// =============================================================================
// Rendering
// =============================================================================

void C3DProcessorApp::drawFrame()
{
    initGL();
    initLighting();

    updateCamera();
    setCamera();
    drawGround();

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
    drawOriginAxisGizmo();
    drawSelectedJointGizmo();
    drawSelectedBoneGizmo();
}

void C3DProcessorApp::drawGround()
{
    glDisable(GL_LIGHTING);
    glColor4f(0.3f, 0.3f, 0.3f, 1.0f);

    // Draw grid
    glBegin(GL_LINES);
    for (int i = -10; i <= 10; i++) {
        glVertex3f(i * 0.5f, 0.0f, -5.0f);
        glVertex3f(i * 0.5f, 0.0f, 5.0f);
        glVertex3f(-5.0f, 0.0f, i * 0.5f);
        glVertex3f(5.0f, 0.0f, i * 0.5f);
    }
    glEnd();

    glEnable(GL_LIGHTING);
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

void C3DProcessorApp::drawOriginAxisGizmo()
{
    if (!mCameraMoving) return;

    // Draw axis at the rotation/zoom center (which is -mTrans in world space)
    Eigen::Vector3d center = -mTrans;

    glDisable(GL_LIGHTING);
    glLineWidth(3.0f);

    float len = 0.05f;

    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3d(center.x(), center.y(), center.z());
    glVertex3d(center.x() + len, center.y(), center.z());
    glEnd();

    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3d(center.x(), center.y(), center.z());
    glVertex3d(center.x(), center.y() + len, center.z());
    glEnd();

    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3d(center.x(), center.y(), center.z());
    glVertex3d(center.x(), center.y(), center.z() + len);
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
            float len = 0.1f;
            Eigen::Matrix3d rot = jointWorld.linear();

            glLineWidth(3.0f);

            // X axis - Red (with tint)
            Eigen::Vector3d xEnd = origin + rot.col(0) * len;
            glColor3f(1.0f, colorTint * 0.3f, colorTint * 0.3f);
            glBegin(GL_LINES);
            glVertex3d(origin.x(), origin.y(), origin.z());
            glVertex3d(xEnd.x(), xEnd.y(), xEnd.z());
            glEnd();

            // Y axis - Green (with tint)
            Eigen::Vector3d yEnd = origin + rot.col(1) * len;
            glColor3f(colorTint * 0.3f, 1.0f, colorTint * 0.3f);
            glBegin(GL_LINES);
            glVertex3d(origin.x(), origin.y(), origin.z());
            glVertex3d(yEnd.x(), yEnd.y(), yEnd.z());
            glEnd();

            // Z axis - Blue (with tint)
            Eigen::Vector3d zEnd = origin + rot.col(2) * len;
            glColor3f(colorTint * 0.3f, colorTint * 0.3f, 1.0f);
            glBegin(GL_LINES);
            glVertex3d(origin.x(), origin.y(), origin.z());
            glVertex3d(zEnd.x(), zEnd.y(), zEnd.z());
            glEnd();

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

        Eigen::Vector3d origin = bodyWorld.translation();
        Eigen::Matrix3d rot = bodyWorld.linear();

        // Draw XYZ axis gizmo at body node position
        float len = 0.12f;

        glLineWidth(3.0f);

        // X axis - Red (with tint)
        Eigen::Vector3d xEnd = origin + rot.col(0) * len;
        glColor3f(1.0f, colorTint * 0.3f, colorTint * 0.3f);
        glBegin(GL_LINES);
        glVertex3d(origin.x(), origin.y(), origin.z());
        glVertex3d(xEnd.x(), xEnd.y(), xEnd.z());
        glEnd();

        // Y axis - Green (with tint)
        Eigen::Vector3d yEnd = origin + rot.col(1) * len;
        glColor3f(colorTint * 0.3f, 1.0f, colorTint * 0.3f);
        glBegin(GL_LINES);
        glVertex3d(origin.x(), origin.y(), origin.z());
        glVertex3d(yEnd.x(), yEnd.y(), yEnd.z());
        glEnd();

        // Z axis - Blue (with tint)
        Eigen::Vector3d zEnd = origin + rot.col(2) * len;
        glColor3f(colorTint * 0.3f, colorTint * 0.3f, 1.0f);
        glBegin(GL_LINES);
        glVertex3d(origin.x(), origin.y(), origin.z());
        glVertex3d(zEnd.x(), zEnd.y(), zEnd.z());
        glEnd();

        glLineWidth(1.0f);
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
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    // Helper lambda to render a skeleton
    auto renderSkeleton = [this](dart::dynamics::SkeletonPtr skel, const Eigen::Vector4d& baseColor) {
        if (!skel) return;

        for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
            const BodyNode* bn = skel->getBodyNode(i);
            if (!bn) continue;

            glPushMatrix();
            glMultMatrixd(bn->getTransform().data());

            bn->eachShapeNodeWith<VisualAspect>([this, &baseColor](const ShapeNode* sn) {
                if (!sn) return true;
                const auto& va = sn->getVisualAspect();
                if (!va || va->isHidden()) return true;

                glPushMatrix();
                Eigen::Affine3d tmp = sn->getRelativeTransform();
                glMultMatrixd(tmp.data());

                const auto* shape = sn->getShape().get();

                // Render mesh (for Mesh mode)
                if (mRenderMode == RenderMode::Mesh) {
                    if (shape->is<MeshShape>()) {
                        const auto* mesh = dynamic_cast<const MeshShape*>(shape);
                        mShapeRenderer.renderMesh(mesh, false, 0.0, baseColor);
                    }
                }

                // Render primitive shapes (for Primitive and Wireframe modes)
                if (mRenderMode == RenderMode::Primitive || mRenderMode == RenderMode::Wireframe) {
                    // For wireframe mode, render primitives as wireframe
                    if (mRenderMode == RenderMode::Wireframe) {
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                        glLineWidth(2.0f);
                        glColor4f(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                    } else {
                        glColor4f(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                    }

                    if (shape->is<BoxShape>()) {
                        GUI::DrawCube(static_cast<const BoxShape*>(shape)->getSize());
                    } else if (shape->is<CapsuleShape>()) {
                        auto* cap = static_cast<const CapsuleShape*>(shape);
                        GUI::DrawCapsule(cap->getRadius(), cap->getHeight());
                    } else if (shape->is<SphereShape>()) {
                        GUI::DrawSphere(static_cast<const SphereShape*>(shape)->getRadius());
                    } else if (shape->is<CylinderShape>()) {
                        auto* cyl = static_cast<const CylinderShape*>(shape);
                        GUI::DrawCylinder(cyl->getRadius(), cyl->getHeight());
                    }

                    // Restore fill mode and line width after wireframe
                    if (mRenderMode == RenderMode::Wireframe) {
                        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                        glLineWidth(1.0f);
                    }
                }

                glPopMatrix();
                return true;
            });

            glPopMatrix();
        }
    };

    // Render mFreeCharacter (light gray, for bone-by-bone debugging)
    if (mFreeCharacter && mRenderFreeCharacter && mMotionState.currentPose.size() > 0) {
        auto skel = mFreeCharacter->getSkeleton();
        if (skel) {
            skel->setPositions(mMotionState.currentPose);
            renderSkeleton(skel, Eigen::Vector4d(0.8, 0.8, 0.8, 0.9));
        }
    }

    // Render mMotionCharacter (blue tint, for motion playback)
    // Pose will be set externally via C3D_Reader in the future
    if (mMotionCharacter && mRenderMotionCharacter) {
        auto skel = mMotionCharacter->getSkeleton();
        if (skel) {
            // Apply offset transformation
            glPushMatrix();
            glTranslated(mMotionCharacterOffset.x(), mMotionCharacterOffset.y(), mMotionCharacterOffset.z());
            renderSkeleton(skel, Eigen::Vector4d(0.0, 1.0, 0.0, 0.9));
            glPopMatrix();
        }
    }
}

void C3DProcessorApp::drawMarkers()
{
    if (!mMotion || mMotion->getSourceType() != "c3d") return;

    glDisable(GL_LIGHTING);

    C3D* c3dMotion = static_cast<C3D*>(mMotion);
    const auto& dataLabels = c3dMotion->getLabels();

    // 1. Data markers (from C3D capture) - green
    // Apply cycleAccumulation + displayOffset so markers move forward with skeleton
    Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;

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

    if (mRenderC3DMarkers && !mMotionState.currentMarkers.empty()) {
        glColor4f(0.4f, 1.0f, 0.2f, 1.0f);
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

    // 2. Skeleton markers (expected from skeleton pose) - red (FreeCharacter)
    if (mRenderExpectedMarkers && mFreeCharacter && mFreeCharacter->hasMarkers()) {
        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
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
        glColor4f(1.0f, 0.5f, 0.0f, 1.0f);  // Orange
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
            glColor4f(0.6f, 0.2f, 0.8f, 1.0f);
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

void C3DProcessorApp::drawControlPanel()
{
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("C3D Processor Control")) {
        ImGui::End();
        return; 
    }

    // Motion List Section
    drawMotionListSection();

    ImGui::Separator();

    // Clinical Data (PID) Section
    drawClinicalDataSection();

    ImGui::Separator();

    // Playback Section
    drawPlaybackSection();

    ImGui::Separator();

    // Marker Fitting Section
    drawMarkerFittingSection();

    ImGui::Separator();

    // Skeleton Scale Section
    drawSkeletonScaleSection();

    ImGui::Separator();

    // Skeleton Export Section
    drawSkeletonExportSection();

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

        // File list
        if (ImGui::BeginListBox("##C3DList", ImVec2(-1, 200))) {
            for (int i = 0; i < static_cast<int>(mMotionList.size()); ++i) {
                fs::path p(mMotionList[i]);
                std::string filename = p.filename().string();

                bool isSelected = (i == mSelectedMotion && mMotionSource == MotionSource::FileList);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
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
            mTrans = Eigen::Vector3d(0.0, -0.8, 0.0);
            mZoom = 1.0;
            mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
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

            ImGui::SetNextItemWidth(150);
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
        if (ImGui::Button("Fit Skeleton to C3D")) {
            if (mC3DReader && mMotion && mSelectedMotion >= 0) {
                C3DConversionParams params;
                params.doCalibration = true;
                mMotion = mC3DReader->loadC3D(mMotionList[mSelectedMotion], params);
                LOG_INFO("[C3DProcessor] Reloaded with calibration");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Skeleton")) {
            if (mFreeCharacter) mFreeCharacter->resetSkeletonToDefault();
        }

        // IK Refinement buttons
        if (ImGui::Button("Refine Arm IK")) {
            if (mC3DReader) {
                mC3DReader->loadSkeletonFittingConfig();
                mC3DReader->refineArmIK();
                LOG_INFO("[C3DProcessor] Arm IK refinement completed");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Refine Leg IK")) {
            if (mC3DReader) {
                mC3DReader->loadSkeletonFittingConfig();
                mC3DReader->refineLegIK();
                LOG_INFO("[C3DProcessor] Leg IK refinement completed");
            }
        }

        if (ImGui::Button("Clear Motion & Zero Pose")) {
            clearMotionAndZeroPose();
        }
    }
}

void C3DProcessorApp::drawSkeletonScaleSection()
{
    if (collapsingHeaderWithControls("Skeleton Scale")) {
        // Use mFreeCharacter as the primary source for scale info
        RenderCharacter* character = mFreeCharacter.get();
        if (!character) {
            character = mMotionCharacter.get();
        }

        if (!character) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No character loaded");
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

void C3DProcessorApp::drawVisualizationPanel()
{
    ImGui::SetNextWindowPos(ImVec2(mWidth - 500, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Visualization")) {
        ImGui::End();
        return;
    }

    // File Paths
    if (ImGui::CollapsingHeader("File Paths", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Skeleton: %s", mSkeletonPath.c_str());
        ImGui::TextWrapped("Markers: %s", mMarkerConfigPath.c_str());
    }

    // Camera Information
    if (ImGui::CollapsingHeader("Camera Info")) {
        ImGui::Text("Position:");
        ImGui::Text("  X: %.3f", mTrans[0]);
        ImGui::Text("  Y: %.3f", mTrans[1]);
        ImGui::Text("  Z: %.3f", mTrans[2]);
        ImGui::Separator();
        ImGui::Text("Zoom: %.3f", mZoom);
        ImGui::Text("Perspective: %.1f deg", mPersp);
        ImGui::Separator();
        ImGui::Text("Follow Mode: %s", mFocus == 1 ? "ON" : "OFF");
        ImGui::Text("Trackball Rotation:");
        Eigen::Matrix3d rotMat = mTrackball.getRotationMatrix();
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

void C3DProcessorApp::drawRenderingPanel()
{
    if (!mShowRenderingPanel) return;

    ImGui::SetNextWindowSize(ImVec2(350, 500), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Rendering", &mShowRenderingPanel)) {
        ImGui::End();
        return;
    }

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

    ImGui::End();
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

    // Get search paths from C3D_Reader or use default
    std::vector<std::string> searchPaths = {"data/c3d", "data/motion"};

    for (const auto& basePath : searchPaths) {
        try {
            PMuscle::URIResolver& resolver = PMuscle::URIResolver::getInstance();
            std::string resolvedPath = resolver.resolve(basePath);

            if (!fs::exists(resolvedPath)) continue;

            for (const auto& entry : fs::recursive_directory_iterator(resolvedPath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".c3d") {
                        mMotionList.push_back(entry.path().string());
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_WARN("[C3DProcessor] Error scanning " << basePath << ": " << e.what());
        }
    }

    // Sort by filename
    std::sort(mMotionList.begin(), mMotionList.end());

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

    // Load C3D
    C3DConversionParams params;
    params.doCalibration = true;
    mMotion = mC3DReader->loadC3D(path, params);

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
// Input Handling
// =============================================================================

void C3DProcessorApp::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<C3DProcessorApp*>(glfwGetWindowUserPointer(window));
    app->resize(width, height);
}

void C3DProcessorApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    auto* app = static_cast<C3DProcessorApp*>(glfwGetWindowUserPointer(window));
    app->mousePress(button, action, mods);
}

void C3DProcessorApp::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* app = static_cast<C3DProcessorApp*>(glfwGetWindowUserPointer(window));
    app->mouseMove(xpos, ypos);
}

void C3DProcessorApp::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto* app = static_cast<C3DProcessorApp*>(glfwGetWindowUserPointer(window));
    app->mouseScroll(xoffset, yoffset);
}

void C3DProcessorApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto* app = static_cast<C3DProcessorApp*>(glfwGetWindowUserPointer(window));
    app->keyPress(key, scancode, action, mods);
}

void C3DProcessorApp::resize(int width, int height)
{
    mWidth = width;
    mHeight = height;
    glViewport(0, 0, width, height);
}

void C3DProcessorApp::mousePress(int button, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (action == GLFW_PRESS) {
        mMouseDown = true;
        mCameraMoving = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = true;
            mTrackball.startBall(mMouseX, mHeight - mMouseY);
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = true;
        }
    } else if (action == GLFW_RELEASE) {
        mMouseDown = false;
        mRotate = false;
        mTranslate = false;
        mCameraMoving = false;
    }
}

void C3DProcessorApp::mouseMove(double x, double y)
{
    double dx = x - mMouseX;
    double dy = y - mMouseY;
    mMouseX = x;
    mMouseY = y;

    if (ImGui::GetIO().WantCaptureMouse) return;

    if (mRotate) {
        mTrackball.updateBall(x, mHeight - y);
    }
    if (mTranslate) {
        double scale = 0.005 / mZoom;  // Scale with zoom level
        Eigen::Matrix3d rot = mTrackball.getRotationMatrix();
        Eigen::Vector3d delta = rot.transpose() * Eigen::Vector3d(dx * scale, -dy * scale, 0.0);
        mTrans += delta;
    }
}

void C3DProcessorApp::mouseScroll(double xoff, double yoff)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    mZoom *= (1.0 + yoff * 0.1);
    mZoom = std::max(0.1, std::min(50.0, mZoom));
}

void C3DProcessorApp::keyPress(int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_SPACE:
                mIsPlaying = !mIsPlaying;
                break;
            case GLFW_KEY_S:
                // Step single frame forward
                if (mMotion) {
                    mIsPlaying = false;
                    double frameTime = mMotion->getFrameTime();
                    updateViewerTime(frameTime);
                }
                break;
            case GLFW_KEY_R:
                if (mods & GLFW_MOD_CONTROL) {
                    // Ctrl+R: Toggle Rendering panel
                    mShowRenderingPanel = !mShowRenderingPanel;
                } else {
                    reset();
                }
                break;
            case GLFW_KEY_1:
            case GLFW_KEY_KP_1:
                alignCameraToPlane(1);  // XY plane
                break;
            case GLFW_KEY_2:
            case GLFW_KEY_KP_2:
                alignCameraToPlane(2);  // YZ plane
                break;
            case GLFW_KEY_3:
            case GLFW_KEY_KP_3:
                alignCameraToPlane(3);  // ZX plane
                break;
            case GLFW_KEY_C:
                // Reload current motion with calibration
                reloadCurrentMotion(true);
                break;
            case GLFW_KEY_F:
                // Toggle camera follow
                mFocus = (mFocus == 1) ? 0 : 1;
                if (mFocus == 1) {
                    LOG_INFO("[C3DProcessor] Camera follow enabled");
                } else {
                    LOG_INFO("[C3DProcessor] Camera follow disabled");
                }
                break;
            case GLFW_KEY_L:
                mRenderMarkerIndices = !mRenderMarkerIndices;
                break;
            case GLFW_KEY_O:
                mRenderMode = static_cast<RenderMode>((static_cast<int>(mRenderMode) + 1) % 3);
                break;
            case GLFW_KEY_V:
                hideVirtualMarkers();
                break;
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
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
                break;
        }
    }
}

void C3DProcessorApp::alignCameraToPlane(int plane)
{
    Eigen::Quaterniond quat;
    switch (plane) {
        case 1: // XY plane - view from +Z
            quat = Eigen::Quaterniond::Identity();
            break;
        case 2: // YZ plane - view from +X
            quat = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY());
            break;
        case 3: // ZX plane - view from +Y (top-down)
            quat = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
            break;
    }
    mTrackball.setQuaternion(quat);
}

void C3DProcessorApp::reset()
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

    // Reset camera position
    mZoom = 1.0;
    mTrans = Eigen::Vector3d(0.0, -0.8, 0.0);
    mTrackball = dart::gui::Trackball();

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
    if (!mResourceManager) return;

    if (collapsingHeaderWithControls("Clinical Data (PID)")) {
        // PID Filter and Refresh
        ImGui::SetNextItemWidth(150);
        if (ImGui::InputText("##PIDFilter", mPIDFilter, sizeof(mPIDFilter))) {
            // Filter applied on display
        }
        ImGui::SameLine();
        if (ImGui::Button("Refresh##PID")) {
            scanPIDList();
        }
        ImGui::SameLine();
        ImGui::Text("%zu PIDs", mPIDList.size());

        // PID List
        if (ImGui::BeginListBox("##PIDList", ImVec2(-1, 150))) {
            for (int i = 0; i < static_cast<int>(mPIDList.size()); ++i) {
                const auto& pid = mPIDList[i];
                const std::string& name = (i < static_cast<int>(mPIDNames.size())) ? mPIDNames[i] : "";
                const std::string& gmfcs = (i < static_cast<int>(mPIDGMFCS.size())) ? mPIDGMFCS[i] : "";

                // Build display string: "12345678 (홍길동, II)" or "12345678 (홍길동)" or just "12345678"
                std::string displayStr;
                if (name.empty() && gmfcs.empty()) {
                    displayStr = pid;
                } else if (name.empty()) {
                    displayStr = pid + " (" + gmfcs + ")";
                } else if (gmfcs.empty()) {
                    displayStr = pid + " (" + name + ")";
                } else {
                    displayStr = pid + " (" + name + ", " + gmfcs + ")";
                }

                // Apply filter (search in PID, name, and GMFCS)
                if (mPIDFilter[0] != '\0' &&
                    pid.find(mPIDFilter) == std::string::npos &&
                    name.find(mPIDFilter) == std::string::npos &&
                    gmfcs.find(mPIDFilter) == std::string::npos) {
                    continue;
                }

                bool isSelected = (i == mSelectedPID);
                if (ImGui::Selectable(displayStr.c_str(), isSelected)) {
                    if (i != mSelectedPID) {
                        mSelectedPID = i;
                        scanPIDC3DFiles();
                    }
                }
            }
            ImGui::EndListBox();
        }

        // Pre/Post radio buttons
        if (ImGui::RadioButton("Pre-op", mPreOp)) {
            if (!mPreOp) {
                mPreOp = true;
                if (mSelectedPID >= 0) scanPIDC3DFiles();
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Post-op", !mPreOp)) {
            if (mPreOp) {
                mPreOp = false;
                if (mSelectedPID >= 0) scanPIDC3DFiles();
            }
        }

        // C3D Files section (only if PID selected)
        if (mSelectedPID >= 0 && mSelectedPID < static_cast<int>(mPIDList.size())) {
            ImGui::Separator();
            ImGui::Text("C3D Files: (%zu files)", mPIDC3DFiles.size());

            // C3D Filter
            ImGui::SetNextItemWidth(150);
            ImGui::InputText("##PIDC3DFilter", mPIDC3DFilter, sizeof(mPIDC3DFilter));

            // C3D List
            if (ImGui::BeginListBox("##PIDC3DList", ImVec2(-1, 150))) {
                for (int i = 0; i < static_cast<int>(mPIDC3DFiles.size()); ++i) {
                    const auto& filename = mPIDC3DFiles[i];

                    // Apply filter
                    if (mPIDC3DFilter[0] != '\0' && filename.find(mPIDC3DFilter) == std::string::npos) {
                        continue;
                    }

                    bool isSelected = (i == mSelectedPIDC3D && mMotionSource == MotionSource::PID);
                    if (ImGui::Selectable(filename.c_str(), isSelected)) {
                        if (i != mSelectedPIDC3D || mMotionSource != MotionSource::PID) {
                            mSelectedPIDC3D = i;
                            mMotionSource = MotionSource::PID;
                            loadPIDC3DFile(filename);
                        }
                    }
                }
                ImGui::EndListBox();
            }
        }
    }
}

void C3DProcessorApp::scanPIDList()
{
    mPIDList.clear();
    mPIDNames.clear();
    mPIDGMFCS.clear();
    mSelectedPID = -1;
    mPIDC3DFiles.clear();
    mSelectedPIDC3D = -1;

    if (!mResourceManager) return;

    try {
        // List all PIDs (directories under @pid:)
        auto entries = mResourceManager->list("@pid:");
        for (const auto& entry : entries) {
            // Entry is the PID (directory name)
            mPIDList.push_back(entry);
        }
        std::sort(mPIDList.begin(), mPIDList.end());

        // Fetch patient names and GMFCS levels for each PID
        mPIDNames.resize(mPIDList.size());
        mPIDGMFCS.resize(mPIDList.size());
        for (size_t i = 0; i < mPIDList.size(); ++i) {
            try {
                std::string nameUri = "@pid:" + mPIDList[i] + "/name";
                auto handle = mResourceManager->fetch(nameUri);
                mPIDNames[i] = handle.as_string();
            } catch (const rm::RMError&) {
                mPIDNames[i] = "";  // No name available
            }
            try {
                std::string gmfcsUri = "@pid:" + mPIDList[i] + "/gmfcs";
                auto handle = mResourceManager->fetch(gmfcsUri);
                mPIDGMFCS[i] = handle.as_string();
            } catch (const rm::RMError&) {
                mPIDGMFCS[i] = "";  // No GMFCS available
            }
        }

        LOG_INFO("[C3DProcessor] Found " << mPIDList.size() << " PIDs");
    } catch (const rm::RMError& e) {
        LOG_WARN("[C3DProcessor] Failed to list PIDs: " << e.what());
    }
}

void C3DProcessorApp::scanPIDC3DFiles()
{
    mPIDC3DFiles.clear();
    mSelectedPIDC3D = -1;

    if (!mResourceManager || mSelectedPID < 0 || mSelectedPID >= static_cast<int>(mPIDList.size())) {
        return;
    }

    const std::string& pid = mPIDList[mSelectedPID];
    std::string prePost = mPreOp ? "pre" : "post";
    std::string pattern = "@pid:" + pid + "/gait/" + prePost;

    try {
        auto files = mResourceManager->list(pattern);
        for (const auto& file : files) {
            // Filter for .c3d files only
            std::string lower = file;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.size() > 4 && lower.substr(lower.size() - 4) == ".c3d") {
                mPIDC3DFiles.push_back(file);
            }
        }
        std::sort(mPIDC3DFiles.begin(), mPIDC3DFiles.end());
        LOG_INFO("[C3DProcessor] Found " << mPIDC3DFiles.size() << " C3D files for PID " << pid << " (" << prePost << ")");
    } catch (const rm::RMError& e) {
        LOG_WARN("[C3DProcessor] Failed to list C3D files: " << e.what());
    }
}

void C3DProcessorApp::loadPIDC3DFile(const std::string& filename)
{
    if (!mResourceManager || mSelectedPID < 0) return;

    const std::string& pid = mPIDList[mSelectedPID];
    std::string prePost = mPreOp ? "pre" : "post";
    std::string uri = "@pid:" + pid + "/gait/" + prePost + "/" + filename;

    try {
        auto handle = mResourceManager->fetch(uri);
        std::filesystem::path localPath = handle.local_path();

        // Use existing loadC3DFile with the local path
        loadC3DFile(localPath.string());

        LOG_INFO("[C3DProcessor] Loaded PID C3D: " << uri);
    } catch (const rm::RMError& e) {
        LOG_ERROR("[C3DProcessor] Failed to fetch C3D: " << e.what());
    }
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
    C3DConversionParams params;
    params.doCalibration = withCalibration;
    mMotion = mC3DReader->loadC3D(path, params);

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

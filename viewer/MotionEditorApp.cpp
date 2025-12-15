#include "MotionEditorApp.h"
#include "DARTHelper.h"
#include "Log.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>

using namespace dart::dynamics;
namespace fs = std::filesystem;

// =============================================================================
// Constructor / Destructor
// =============================================================================

MotionEditorApp::MotionEditorApp(const std::string& configPath)
    : mWindow(nullptr)
    , mConfigPath(configPath.empty() ? "data/rm_config.yaml" : configPath)
{
    // Load config
    loadRenderConfig();

    // Initialize GLFW
    if (!glfwInit()) {
        LOG_ERROR("[MotionEditor] Failed to initialize GLFW");
        exit(1);
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "MotionEditor");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "MotionEditor");
    mWindow = glfwCreateWindow(mWidth, mHeight, "Motion Editor", nullptr, nullptr);
    if (!mWindow) {
        LOG_ERROR("[MotionEditor] Failed to create GLFW window");
        glfwTerminate();
        exit(1);
    }

    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOG_ERROR("[MotionEditor] Failed to initialize GLAD");
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

    // Initialize Resource Manager for PID-based access
    try {
        mResourceManager = std::make_unique<rm::ResourceManager>(mConfigPath);
        scanPIDList();
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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

// =============================================================================
// Main Loop
// =============================================================================

void MotionEditorApp::startLoop()
{
    while (!glfwWindowShouldClose(mWindow)) {
        glfwPollEvents();

        // Update time
        double currentTime = glfwGetTime();
        double dt = currentTime - mLastRealTime;
        mLastRealTime = currentTime;

        if (mIsPlaying) {
            updateViewerTime(dt * mPlaybackSpeed);
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // OpenGL rendering
        drawFrame();

        // ImGui panels
        drawLeftPanel();
        drawRightPanel();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(mWindow);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void MotionEditorApp::initGL()
{
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MotionEditorApp::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Load font with Korean glyph support
    ImFontConfig fontConfig;
    fontConfig.MergeMode = false;

    const char* fontPath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";
    if (fs::exists(fontPath)) {
        io.Fonts->AddFontFromFileTTF(fontPath, 16.0f, &fontConfig,
            io.Fonts->GetGlyphRangesKorean());
        LOG_INFO("[MotionEditor] Loaded Korean font: " << fontPath);
    } else {
        io.Fonts->AddFontDefault();
        LOG_WARN("[MotionEditor] Korean font not found, using default");
    }

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 150");
}

void MotionEditorApp::initLighting()
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

void MotionEditorApp::updateCamera()
{
    // Camera follow character
    if (mFocus == 1 && mMotion != nullptr && mCharacter) {
        int currentFrameIdx = mMotionState.manualFrameIndex;
        if (mMotionState.navigationMode == ME_SYNC) {
            double frameFloat = (mViewerTime / mCycleDuration) * mMotion->getTotalTimesteps();
            currentFrameIdx = static_cast<int>(frameFloat) % mMotion->getTotalTimesteps();
        }

        Eigen::VectorXd pose = mMotion->getPose(currentFrameIdx);
        if (pose.size() >= 6) {
            mTrans[0] = -(pose[3] + mMotionState.cycleAccumulation[0] + mMotionState.displayOffset[0]);
            mTrans[1] = -(pose[4] + mMotionState.displayOffset[1]) - 1;
            mTrans[2] = -(pose[5] + mMotionState.cycleAccumulation[2] + mMotionState.displayOffset[2]);
        }
    }
}

void MotionEditorApp::setCamera()
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

void MotionEditorApp::loadRenderConfig()
{
    try {
        std::string resolved_path = rm::resolve("render.yaml");
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
                    mControlPanelWidth = config["geometry"]["panels"]["control_panel_width"].as<float>();
            }
        }

        if (config["default_open_panels"]) {
            for (const auto& panel : config["default_open_panels"]) {
                mDefaultOpenPanels.insert(panel.as<std::string>());
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN("[MotionEditor] Could not load render.yaml: " << e.what());
    }
}

bool MotionEditorApp::isPanelDefaultOpen(const std::string& panelName) const
{
    return mDefaultOpenPanels.find(panelName) != mDefaultOpenPanels.end();
}

// =============================================================================
// Rendering
// =============================================================================

void MotionEditorApp::drawFrame()
{
    initGL();
    initLighting();

    updateCamera();
    setCamera();
    drawGround();

    if (mCharacter && mMotion) {
        drawSkeleton();
    }
}

void MotionEditorApp::drawSkeleton()
{
    if (!mCharacter || !mMotion) return;

    // Evaluate current pose
    evaluateMotionPose();

    // Draw skeleton
    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    Eigen::Vector4d color(0.3, 0.6, 0.9, 1.0);

    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        const dart::dynamics::BodyNode* bn = skel->getBodyNode(i);
        if (!bn) continue;

        glPushMatrix();
        glMultMatrixd(bn->getTransform().data());

        bn->eachShapeNodeWith<dart::dynamics::VisualAspect>([this, &color](const dart::dynamics::ShapeNode* sn) {
            if (!sn) return true;
            const auto& va = sn->getVisualAspect();
            if (!va || va->isHidden()) return true;

            glPushMatrix();
            Eigen::Affine3d tmp = sn->getRelativeTransform();
            glMultMatrixd(tmp.data());

            const auto* shape = sn->getShape().get();

            // Render primitive shapes (Primitive or Wireframe mode)
            if (mRenderMode == MotionEditorRenderMode::Wireframe) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(2.0f);
            }
            glColor4f(color[0], color[1], color[2], color[3]);

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
            if (mRenderMode == MotionEditorRenderMode::Wireframe) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glLineWidth(1.0f);
            }

            glPopMatrix();
            return true;
        });

        glPopMatrix();
    }
}

void MotionEditorApp::drawGround()
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

// =============================================================================
// UI Panels
// =============================================================================

void MotionEditorApp::drawLeftPanel()
{
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_FirstUseEver);

    ImGui::Begin("Data Loader", nullptr,
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

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
    ImGui::SetNextWindowSize(ImVec2(mRightPanelWidth, mHeight), ImGuiCond_FirstUseEver);

    ImGui::Begin("Motion Editor", nullptr, ImGuiWindowFlags_NoCollapse);

    // Update position based on current window width (allows horizontal resize)
    ImVec2 windowSize = ImGui::GetWindowSize();
    ImGui::SetWindowPos(ImVec2(mWidth - windowSize.x, 0));

    drawMotionInfoSection();
    ImGui::Separator();
    drawTrimSection();
    ImGui::Separator();
    drawExportSection();

    ImGui::End();
}

void MotionEditorApp::drawPIDBrowserTab()
{
    if (!mResourceManager) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Resource Manager not initialized");
        return;
    }

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
    if (ImGui::BeginListBox("##PIDList", ImVec2(-1, 120))) {
        for (int i = 0; i < static_cast<int>(mPIDList.size()); ++i) {
            const auto& pid = mPIDList[i];
            const std::string& name = (i < static_cast<int>(mPIDNames.size())) ? mPIDNames[i] : "";
            const std::string& gmfcs = (i < static_cast<int>(mPIDGMFCS.size())) ? mPIDGMFCS[i] : "";

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

            // Apply filter
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
                    scanH5Files();
                    autoDetectSkeleton();
                }
            }
        }
        ImGui::EndListBox();
    }

    // Pre/Post radio buttons
    if (ImGui::RadioButton("Pre-op", mPreOp)) {
        if (!mPreOp) {
            mPreOp = true;
            if (mSelectedPID >= 0) {
                scanH5Files();
                autoDetectSkeleton();
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Post-op", !mPreOp)) {
        if (mPreOp) {
            mPreOp = false;
            if (mSelectedPID >= 0) {
                scanH5Files();
                autoDetectSkeleton();
            }
        }
    }

    // H5 Files section
    if (mSelectedPID >= 0 && mSelectedPID < static_cast<int>(mPIDList.size())) {
        ImGui::Separator();
        ImGui::Text("H5 Files: (%zu files)", mH5Files.size());

        // H5 Filter
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("##H5Filter", mH5Filter, sizeof(mH5Filter));
        ImGui::SameLine();
        if (ImGui::Button("X##H5Filter")) {
            mH5Filter[0] = '\0';
        }

        // H5 List
        if (ImGui::BeginListBox("##H5List", ImVec2(-1, 120))) {
            for (int i = 0; i < static_cast<int>(mH5Files.size()); ++i) {
                const auto& filename = mH5Files[i];

                // Apply filter
                if (mH5Filter[0] != '\0' && filename.find(mH5Filter) == std::string::npos) {
                    continue;
                }

                bool isSelected = (i == mSelectedH5);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
                    if (i != mSelectedH5) {
                        mSelectedH5 = i;

                        // Build full URI and load
                        std::string pid = mPIDList[mSelectedPID];
                        std::string prePost = mPreOp ? "pre" : "post";
                        std::string uri = "@pid:" + pid + "/gait/" + prePost + "/h5/" + filename;
                        std::string resolved = mResourceManager->resolve(uri).string();
                        loadH5Motion(resolved);
                    }
                }
            }
            ImGui::EndListBox();
        }
    }
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
        ImGui::Checkbox("Auto-detect", &mUseAutoSkeleton);

        if (mUseAutoSkeleton) {
            if (mAutoDetectedSkeletonPath.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "No skeleton detected");
            } else {
                ImGui::TextWrapped("Path: %s", mAutoDetectedSkeletonPath.c_str());
            }
        } else {
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##ManualSkel", mManualSkeletonPath, sizeof(mManualSkeletonPath));
        }

        if (ImGui::Button("Reload Skeleton")) {
            std::string path = mUseAutoSkeleton ? mAutoDetectedSkeletonPath : std::string(mManualSkeletonPath);
            if (!path.empty()) {
                loadSkeleton(path);
            }
        }

        // Render mode
        ImGui::Text("Render Mode:");
        int renderModeInt = static_cast<int>(mRenderMode);
        if (ImGui::RadioButton("Primitive", renderModeInt == 0)) mRenderMode = MotionEditorRenderMode::Primitive;
        ImGui::SameLine();
        if (ImGui::RadioButton("Wire", renderModeInt == 1)) mRenderMode = MotionEditorRenderMode::Wireframe;
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
            reset();
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
        bool follow = (mFocus == 1);
        if (ImGui::Checkbox("Camera Follow", &follow)) {
            mFocus = follow ? 1 : 0;
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
        ImGui::Checkbox("Auto-suffix \"_trimmed\"", &mAutoSuffix);

        // Preview output filename
        std::string filename = strlen(mExportFilename) > 0
            ? mExportFilename
            : fs::path(mMotionSourcePath).stem().string();
        if (mAutoSuffix) filename += "_trimmed";
        filename += ".h5";

        fs::path outputPath = fs::path(mMotionSourcePath).parent_path() / filename;
        bool fileExists = fs::exists(outputPath);

        ImGui::TextWrapped("Output: %s", filename.c_str());

        // Warn if file already exists
        if (fileExists) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Warning: File already exists!");
        }

        // Export button
        if (ImGui::Button("Export to Same Folder", ImVec2(-1, 30))) {
            exportTrimmedMotion();
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

// =============================================================================
// PID Scanner Methods
// =============================================================================

void MotionEditorApp::scanPIDList()
{
    mPIDList.clear();
    mPIDNames.clear();
    mPIDGMFCS.clear();
    mSelectedPID = -1;
    mH5Files.clear();
    mSelectedH5 = -1;

    if (!mResourceManager) return;

    try {
        auto entries = mResourceManager->list("@pid:");
        for (const auto& entry : entries) {
            mPIDList.push_back(entry);
        }
        std::sort(mPIDList.begin(), mPIDList.end());

        // Fetch patient names and GMFCS levels
        mPIDNames.resize(mPIDList.size());
        mPIDGMFCS.resize(mPIDList.size());
        for (size_t i = 0; i < mPIDList.size(); ++i) {
            try {
                std::string nameUri = "@pid:" + mPIDList[i] + "/name";
                auto handle = mResourceManager->fetch(nameUri);
                mPIDNames[i] = handle.as_string();
            } catch (const rm::RMError&) {
                mPIDNames[i] = "";
            }
            try {
                std::string gmfcsUri = "@pid:" + mPIDList[i] + "/gmfcs";
                auto handle = mResourceManager->fetch(gmfcsUri);
                mPIDGMFCS[i] = handle.as_string();
            } catch (const rm::RMError&) {
                mPIDGMFCS[i] = "";
            }
        }

        LOG_INFO("[MotionEditor] Found " << mPIDList.size() << " PIDs");
    } catch (const rm::RMError& e) {
        LOG_WARN("[MotionEditor] Failed to list PIDs: " << e.what());
    }
}

void MotionEditorApp::scanH5Files()
{
    mH5Files.clear();
    mSelectedH5 = -1;

    if (!mResourceManager || mSelectedPID < 0 || mSelectedPID >= static_cast<int>(mPIDList.size())) {
        return;
    }

    const std::string& pid = mPIDList[mSelectedPID];
    std::string prePost = mPreOp ? "pre" : "post";
    std::string pattern = "@pid:" + pid + "/gait/" + prePost + "/h5";

    try {
        auto files = mResourceManager->list(pattern);
        for (const auto& file : files) {
            // Only include .h5 files
            if (file.size() > 3 && file.substr(file.size() - 3) == ".h5") {
                mH5Files.push_back(file);
            }
        }
        std::sort(mH5Files.begin(), mH5Files.end());
        LOG_INFO("[MotionEditor] Found " << mH5Files.size() << " H5 files for PID " << pid);
    } catch (const rm::RMError& e) {
        LOG_WARN("[MotionEditor] Failed to list H5 files: " << e.what());
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

        // Initialize trim range
        mTrimStart = 0;
        mTrimEnd = mMotion->getNumFrames() - 1;

        // Initialize playback state
        mMotionState = MotionEditorViewerState();
        mMotionState.maxFrameIndex = mMotion->getNumFrames() - 1;
        mCycleDuration = mMotion->getMaxTime();

        // Reset viewer time
        mViewerTime = 0.0;
        mViewerPhase = 0.0;

        // Clear export filename
        mExportFilename[0] = '\0';

        // Load skeleton if we have one auto-detected or manually set
        std::string skelPath = mUseAutoSkeleton ? mAutoDetectedSkeletonPath : std::string(mManualSkeletonPath);
        if (!skelPath.empty() && !mCharacter) {
            loadSkeleton(skelPath);
        }

        // Note: setRefMotion is not called since RenderCharacter doesn't inherit from Character
        // The motion editor doesn't need height calibration - it just plays back raw frames

        LOG_INFO("[MotionEditor] Loaded motion: " << path << " (" << mMotion->getNumFrames() << " frames)");
    } catch (const std::exception& e) {
        LOG_ERROR("[MotionEditor] Failed to load motion: " << e.what());
        mMotion = nullptr;
    }
}

void MotionEditorApp::autoDetectSkeleton()
{
    mAutoDetectedSkeletonPath.clear();

    if (!mResourceManager || mSelectedPID < 0 || mSelectedPID >= static_cast<int>(mPIDList.size())) {
        return;
    }

    const std::string& pid = mPIDList[mSelectedPID];
    std::string prePost = mPreOp ? "pre" : "post";
    std::string uri = "@pid:" + pid + "/gait/" + prePost + "/skeleton/dynamic.yaml";

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

        // Note: setRefMotion is not called since RenderCharacter doesn't inherit from Character
        // The motion editor doesn't need height calibration - it just plays back raw frames

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

void MotionEditorApp::exportTrimmedMotion()
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
        {"source_type", "motion_editor_trim"},
        {"source_file", fs::path(mMotionSourcePath).filename().string()},
        {"trim_start", std::to_string(mTrimStart)},
        {"trim_end", std::to_string(mTrimEnd)}
    };

    try {
        hdf->exportToFile(outputPath.string(), mTrimStart, mTrimEnd, metadata);
        mLastExportMessage = "Success: Exported to " + outputPath.filename().string();
        mLastExportMessageTime = glfwGetTime();

        // Build PID-style URI if PID is selected
        if (mSelectedPID >= 0 && mSelectedPID < static_cast<int>(mPIDList.size())) {
            std::string pid = mPIDList[mSelectedPID];
            std::string prePost = mPreOp ? "pre" : "post";
            mLastExportURI = "@pid:" + pid + "/gait/" + prePost + "/h5/" + outputPath.filename().string();
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
// Input Handling
// =============================================================================

void MotionEditorApp::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<MotionEditorApp*>(glfwGetWindowUserPointer(window));
    app->resize(width, height);
}

void MotionEditorApp::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    auto* app = static_cast<MotionEditorApp*>(glfwGetWindowUserPointer(window));
    app->mousePress(button, action, mods);
}

void MotionEditorApp::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* app = static_cast<MotionEditorApp*>(glfwGetWindowUserPointer(window));
    app->mouseMove(xpos, ypos);
}

void MotionEditorApp::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto* app = static_cast<MotionEditorApp*>(glfwGetWindowUserPointer(window));
    app->mouseScroll(xoffset, yoffset);
}

void MotionEditorApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto* app = static_cast<MotionEditorApp*>(glfwGetWindowUserPointer(window));
    app->keyPress(key, scancode, action, mods);
}

void MotionEditorApp::resize(int width, int height)
{
    mWidth = width;
    mHeight = height;
    glViewport(0, 0, width, height);
}

void MotionEditorApp::mousePress(int button, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (action == GLFW_PRESS) {
        mMouseDown = true;
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
    }
}

void MotionEditorApp::mouseMove(double x, double y)
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
        double scale = 0.005 / mZoom;
        Eigen::Matrix3d rot = mTrackball.getRotationMatrix();
        Eigen::Vector3d delta = rot.transpose() * Eigen::Vector3d(dx * scale, -dy * scale, 0.0);
        mTrans += delta;
    }
}

void MotionEditorApp::mouseScroll(double xoff, double yoff)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    mZoom *= (1.0 + yoff * 0.1);
    mZoom = std::max(0.1, std::min(50.0, mZoom));
}

void MotionEditorApp::keyPress(int key, int scancode, int action, int mods)
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
                    mMotionState.manualFrameIndex = std::min(
                        mMotionState.manualFrameIndex + 1,
                        mMotion->getNumFrames() - 1);
                    mMotionState.navigationMode = ME_MANUAL_FRAME;
                }
                break;
            case GLFW_KEY_R:
                reset();
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
            case GLFW_KEY_F:
                mFocus = (mFocus == 1) ? 0 : 1;
                break;
            case GLFW_KEY_LEFT_BRACKET:  // [
                // Set trim start to current frame
                if (mMotion) {
                    mTrimStart = mMotionState.manualFrameIndex;
                    LOG_INFO("[MotionEditor] Trim start set to frame " << mTrimStart);
                }
                break;
            case GLFW_KEY_RIGHT_BRACKET:  // ]
                // Set trim end to current frame
                if (mMotion) {
                    mTrimEnd = mMotionState.manualFrameIndex;
                    LOG_INFO("[MotionEditor] Trim end set to frame " << mTrimEnd);
                }
                break;
            case GLFW_KEY_O:
                mRenderMode = static_cast<MotionEditorRenderMode>(
                    (static_cast<int>(mRenderMode) + 1) % 2);
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
                break;
        }
    }
}

void MotionEditorApp::alignCameraToPlane(int plane)
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
    mTrackball.setQuaternion(quat);
}

void MotionEditorApp::reset()
{
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mMotionState.lastFrameIdx = 0;
    mMotionState.cycleAccumulation.setZero();
    mMotionState.manualFrameIndex = 0;
    mIsPlaying = false;
}

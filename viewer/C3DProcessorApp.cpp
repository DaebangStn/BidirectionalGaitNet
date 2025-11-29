#include "C3DProcessorApp.h"
#include "DARTHelper.h"
#include "UriResolver.h"
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

C3DProcessorApp::C3DProcessorApp(const std::string& skeletonPath, const std::string& markerPath)
    : mWindow(nullptr)
    , mWidth(1280)
    , mHeight(720)
    , mWindowXPos(0)
    , mWindowYPos(0)
    , mZoom(1.0)
    , mPersp(45.0)
    , mMouseDown(false)
    , mRotate(false)
    , mTranslate(false)
    , mMouseX(0)
    , mMouseY(0)
    , mSkeletonPath(skeletonPath)
    , mMarkerConfigPath(markerPath)
    , mC3DReader(nullptr)
    , mMotion(nullptr)
    , mRenderC3DMarkers(true)
    , mRenderExpectedMarkers(true)
    , mRenderMarkerIndices(true)
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

    mWindow = glfwCreateWindow(mWidth, mHeight, "C3D Processor", nullptr, nullptr);
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
    mMotionCharacter = std::make_unique<RenderCharacter>(mSkeletonPath);
    if (mMotionCharacter) {
        mMotionCharacter->loadMarkers(mMarkerConfigPath);
    }

    // Create C3D Reader
    mC3DReader = new C3D_Reader(mMarkerConfigPath, mMotionCharacter.get());

    // Scan for C3D files and autoload first one
    scanC3DFiles();
    if (!mMotionList.empty()) {
        mSelectedMotion = 0;
        loadC3DFile(mMotionList[0]);
        LOG_INFO("[C3DProcessor] Autoloaded first C3D file: " << mMotionList[0]);
    }

    mLastRealTime = glfwGetTime();

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

        if (mIsPlaying) {
            updateViewerTime(dt * mViewerPlaybackSpeed);
        }

        // OpenGL rendering
        drawFrame();

        // ImGui rendering
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawControlPanel();
        drawVisualizationPanel();

        // Draw marker labels
        if (mRenderMarkerIndices && (!mMarkerIndexLabels.empty() || !mSkelMarkerIndexLabels.empty())) {
            GLdouble modelview[16], projection[16];
            GLint viewport[4];
            glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
            glGetDoublev(GL_PROJECTION_MATRIX, projection);
            glGetIntegerv(GL_VIEWPORT, viewport);

            ImDrawList* drawList = ImGui::GetForegroundDrawList();
            ImFont* font = ImGui::GetFont();
            float fontSize = 18.0f;

            // Data markers - black text
            for (const auto& label : mMarkerIndexLabels) {
                GLdouble screenX, screenY, screenZ;
                if (gluProject(label.position.x(), label.position.y(), label.position.z(),
                              modelview, projection, viewport, &screenX, &screenY, &screenZ) == GL_TRUE) {
                    if (screenZ > 0.0 && screenZ < 1.0) {
                        float y = mHeight - static_cast<float>(screenY);
                        std::string text = std::to_string(label.index) + ": " + label.name;
                        drawList->AddText(font, fontSize, ImVec2(static_cast<float>(screenX) - 20, y - 10),
                                          IM_COL32(0, 0, 0, 255), text.c_str());
                    }
                }
            }

            // Skeleton markers - black text
            for (const auto& label : mSkelMarkerIndexLabels) {
                GLdouble screenX, screenY, screenZ;
                if (gluProject(label.position.x(), label.position.y(), label.position.z(),
                              modelview, projection, viewport, &screenX, &screenY, &screenZ) == GL_TRUE) {
                    if (screenZ > 0.0 && screenZ < 1.0) {
                        float y = mHeight - static_cast<float>(screenY);
                        std::string text = std::to_string(label.index) + ": " + label.name;
                        drawList->AddText(font, fontSize, ImVec2(static_cast<float>(screenX) + 5, y - 10),
                                          IM_COL32(0, 0, 0, 255), text.c_str());
                    }
                }
            }
        }

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
    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
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
    setCamera();

    drawGround();
    drawSkeleton();
    drawMarkers();
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

void C3DProcessorApp::drawSkeleton()
{
    if (!mMotionCharacter || mMotionState.currentPose.size() == 0) return;

    auto skel = mMotionCharacter->getSkeleton();
    if (!skel) return;

    // Set pose
    skel->setPositions(mMotionState.currentPose);

    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glColor4f(0.8f, 0.8f, 0.2f, 0.7f);

    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        const BodyNode* bn = skel->getBodyNode(i);
        if (!bn) continue;

        glPushMatrix();
        glMultMatrixd(bn->getTransform().data());

        bn->eachShapeNodeWith<VisualAspect>([this](const ShapeNode* sn) {
            if (!sn) return true;
            const auto& va = sn->getVisualAspect();
            if (!va || va->isHidden()) return true;

            glPushMatrix();
            Eigen::Affine3d tmp = sn->getRelativeTransform();
            glMultMatrixd(tmp.data());

            const auto* shape = sn->getShape().get();
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

            glPopMatrix();
            return true;
        });

        glPopMatrix();
    }
}

void C3DProcessorApp::drawMarkers()
{
    if (!mMotion || mMotion->getSourceType() != "c3d") return;

    mMarkerIndexLabels.clear();
    mSkelMarkerIndexLabels.clear();

    glDisable(GL_LIGHTING);

    C3D* c3dMotion = static_cast<C3D*>(mMotion);
    const auto& dataLabels = c3dMotion->getLabels();

    // 1. Data markers (from C3D capture) - green
    // Apply cycleAccumulation + displayOffset so markers move forward with skeleton
    Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;

    if (mRenderC3DMarkers && !mMotionState.currentMarkers.empty()) {
        glColor4f(0.4f, 1.0f, 0.2f, 1.0f);
        for (size_t i = 0; i < mMotionState.currentMarkers.size(); ++i) {
            const auto& marker = mMotionState.currentMarkers[i];
            if (!marker.array().isFinite().all()) continue;
            Eigen::Vector3d offsetMarker = marker + markerOffset;
            GUI::DrawSphere(offsetMarker, 0.0125);
            if (mRenderMarkerIndices) {
                std::string name = (i < dataLabels.size()) ? dataLabels[i] : "";
                mMarkerIndexLabels.push_back({offsetMarker, static_cast<int>(i), name});
            }
        }
    }

    // 2. Skeleton markers (expected from skeleton pose) - red
    if (mRenderExpectedMarkers && mMotionCharacter && mMotionCharacter->hasMarkers()) {
        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
        const auto& skelMarkers = mMotionCharacter->getMarkers();
        auto expectedMarkers = mMotionCharacter->getExpectedMarkerPositions();
        for (size_t i = 0; i < expectedMarkers.size(); ++i) {
            const auto& marker = expectedMarkers[i];
            if (!marker.array().isFinite().all()) continue;
            GUI::DrawSphere(marker, 0.0125);
            if (mRenderMarkerIndices) {
                std::string name = (i < skelMarkers.size()) ? skelMarkers[i].name : "";
                mSkelMarkerIndexLabels.push_back({marker, static_cast<int>(i), name});
            }
        }
    }

    glEnable(GL_LIGHTING);
}

void C3DProcessorApp::drawMarkerLabels()
{
    // Labels are drawn in startLoop after ImGui::NewFrame
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

    // Playback Section
    drawPlaybackSection();

    ImGui::Separator();

    // Marker Visibility Section
    drawMarkerVisibilitySection();

    ImGui::Separator();

    // Marker Fitting Section
    drawMarkerFittingSection();

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

                bool isSelected = (i == mSelectedMotion);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
                    if (i != mSelectedMotion) {
                        mSelectedMotion = i;
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
        }

        // Playback speed
        ImGui::SliderFloat("Speed", &mViewerPlaybackSpeed, 0.1f, 3.0f, "%.1fx");

        // Frame navigation
        if (mMotion) {
            int maxFrame = std::max(0, mMotion->getNumFrames() - 1);
            int currentFrame = mMotionState.lastFrameIdx;

            if (ImGui::SliderInt("Frame", &currentFrame, 0, maxFrame)) {
                mMotionState.navigationMode = C3D_MANUAL_FRAME;
                mMotionState.manualFrameIndex = currentFrame;
            }

            // Navigation mode toggle
            bool syncMode = (mMotionState.navigationMode == C3D_SYNC);
            if (ImGui::Checkbox("Sync Mode", &syncMode)) {
                mMotionState.navigationMode = syncMode ? C3D_SYNC : C3D_MANUAL_FRAME;
            }
        }

        // Time display
        ImGui::Text("Time: %.2f s", mViewerTime);
        ImGui::Text("Phase: %.2f", mViewerPhase);
    }
}

void C3DProcessorApp::drawMarkerVisibilitySection()
{
    if (collapsingHeaderWithControls("Marker Visibility")) {
        ImGui::Checkbox("Data Markers (C3D)", &mRenderC3DMarkers);
        ImGui::Checkbox("Skeleton Markers", &mRenderExpectedMarkers);
        ImGui::Checkbox("Marker Labels", &mRenderMarkerIndices);
    }
}

void C3DProcessorApp::drawMarkerFittingSection()
{
    if (collapsingHeaderWithControls("Marker Fitting")) {
        if (ImGui::Button("Reload Config")) {
            if (mC3DReader) {
                mC3DReader->reloadFittingConfig();
                LOG_INFO("[C3DProcessor] Reloaded fitting config");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Skeleton")) {
            if (mC3DReader) {
                mC3DReader->resetSkeletonToDefault();
                LOG_INFO("[C3DProcessor] Reset skeleton to default");
            }
        }

        // Marker error plot
        drawMarkerDiffPlot();
    }
}

void C3DProcessorApp::drawVisualizationPanel()
{
    ImGui::SetNextWindowPos(ImVec2(mWidth - 400, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Visualization")) {
        ImGui::End();
        return;
    }

    // Marker Correspondence Table
    drawMarkerCorrespondenceTable();

    ImGui::End();
}

void C3DProcessorApp::drawMarkerDiffPlot()
{
    if (!mMotion || mMotion->getSourceType() != "c3d") return;
    if (mGraphData.find("marker_error_mean") == mGraphData.end()) return;

    const auto& times = mGraphTime["marker_error_mean"];
    const auto& values = mGraphData["marker_error_mean"];

    if (times.empty() || values.empty()) return;

    ImPlot::SetNextAxisLimits(ImAxis_X1, mXmin, 0, ImGuiCond_Always);
    ImPlot::SetNextAxisLimits(ImAxis_Y1, 0.0, 0.3, ImGuiCond_Once);

    if (ImPlot::BeginPlot("Marker Error##MarkerDiff", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Time (s)", "Error (m)");

        std::vector<double> t(times.end() - std::min(times.size(), size_t(500)), times.end());
        std::vector<double> v(values.end() - std::min(values.size(), size_t(500)), values.end());

        if (!t.empty()) {
            ImPlot::PlotLine("Mean Error", t.data(), v.data(), t.size());
        }

        ImPlot::EndPlot();
    }
}

void C3DProcessorApp::drawMarkerCorrespondenceTable()
{
    if (collapsingHeaderWithControls("Marker Correspondence")) {
        if (!mMotion || mMotion->getSourceType() != "c3d" || !mMotionCharacter) {
            ImGui::Text("Load a C3D file to see correspondence");
            return;
        }

        C3D* c3dMotion = static_cast<C3D*>(mMotion);
        const auto& dataLabels = c3dMotion->getLabels();
        const auto& skelMarkers = mMotionCharacter->getMarkers();

        // Search filter
        ImGui::InputText("Filter", mMarkerSearchFilter, sizeof(mMarkerSearchFilter));

        if (ImGui::BeginTable("MarkerTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 200))) {
            ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed, 40);
            ImGui::TableSetupColumn("Data Label", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Skel Label", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Error", ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableHeadersRow();

            size_t count = std::max(dataLabels.size(), skelMarkers.size());
            for (size_t i = 0; i < count; ++i) {
                std::string dataLabel = (i < dataLabels.size()) ? dataLabels[i] : "";
                std::string skelLabel = (i < skelMarkers.size()) ? skelMarkers[i].name : "";

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

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%zu", i);

                ImGui::TableNextColumn();
                ImGui::Text("%s", dataLabel.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%s", skelLabel.c_str());

                ImGui::TableNextColumn();
                // Compute error if we have both markers
                if (i < mMotionState.currentMarkers.size() && mMotionCharacter && i < skelMarkers.size()) {
                    auto expectedMarkers = mMotionCharacter->getExpectedMarkerPositions();
                    if (i < expectedMarkers.size()) {
                        const auto& dataM = mMotionState.currentMarkers[i];
                        const auto& skelM = expectedMarkers[i];
                        if (dataM.array().isFinite().all() && skelM.array().isFinite().all()) {
                            // Apply offset to data marker for comparison
                            Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;
                            Eigen::Vector3d offsetDataM = dataM + markerOffset;
                            double err = (offsetDataM - skelM).norm() * 1000;  // mm
                            ImGui::Text("%.1f", err);
                        } else {
                            ImGui::Text("-");
                        }
                    }
                } else {
                    ImGui::Text("-");
                }
            }
            ImGui::EndTable();
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
    clearGraphData();

    // Load C3D
    C3DConversionParams params;
    mMotion = mC3DReader->loadC3D(path, params);

    if (mMotion) {
        mMotionPath = path;
        mMotionState.maxFrameIndex = std::max(0, mMotion->getNumFrames() - 1);

        // Compute cycle distance
        C3D* c3dMotion = static_cast<C3D*>(mMotion);
        mMotionState.cycleDistance = computeMarkerCycleDistance(c3dMotion);

        // Compute height calibration
        if (c3dMotion->getNumFrames() > 0) {
            auto markers = c3dMotion->getMarkers(0);
            double heightOffset = computeMarkerHeightCalibration(markers);
            mMotionState.displayOffset = Eigen::Vector3d(0, heightOffset, 0);
        }

        // Initialize viewer time
        mViewerTime = 0.0;
        mViewerPhase = 0.0;

        LOG_INFO("[C3DProcessor] Loaded: " << path);
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

    if (!mRenderC3DMarkers || !mMotion || mMotion->getSourceType() != "c3d")
        return context;

    C3D* c3dMotion = static_cast<C3D*>(mMotion);
    if (c3dMotion->getNumFrames() == 0)
        return context;

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

        if (currentIdx < markerState.lastFrameIdx) {
            markerState.cycleAccumulation += markerState.cycleDistance;
        }

        markerState.currentMarkers = std::move(interpolated);
        markerState.lastFrameIdx = currentIdx;
    }

    // Update skeleton pose from C3D
    if (mMotionCharacter && mMotion) {
        auto skel = mMotionCharacter->getSkeleton();
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
    if (!mMotion || mMotion->getSourceType() != "c3d") return;
    if (!mMotionCharacter || !mMotionCharacter->hasMarkers()) return;
    if (mMotionState.currentMarkers.empty()) return;

    auto expectedMarkers = mMotionCharacter->getExpectedMarkerPositions();

    // Offset to apply to data markers (same as skeleton offset)
    Eigen::Vector3d markerOffset = mMotionState.displayOffset + mMotionState.cycleAccumulation;

    double totalError = 0.0;
    double maxError = 0.0;
    int validCount = 0;

    size_t count = std::min(mMotionState.currentMarkers.size(), expectedMarkers.size());
    for (size_t i = 0; i < count; ++i) {
        const auto& dataMarker = mMotionState.currentMarkers[i];
        const auto& skelMarker = expectedMarkers[i];

        if (!dataMarker.array().isFinite().all()) continue;
        if (!skelMarker.array().isFinite().all()) continue;

        // Apply offset to data marker for comparison with skeleton marker
        Eigen::Vector3d offsetDataMarker = dataMarker + markerOffset;
        double error = (offsetDataMarker - skelMarker).norm();
        totalError += error;
        maxError = std::max(maxError, error);
        validCount++;
    }

    double meanError = (validCount > 0) ? totalError / validCount : 0.0;

    // Record for plotting
    double relTime = mViewerTime - std::floor(mViewerTime / 10.0) * 10.0 - 10.0;
    recordGraphData("marker_error_mean", relTime, meanError);
    recordGraphData("marker_error_max", relTime, maxError);
}

// =============================================================================
// Graph Data Helpers
// =============================================================================

void C3DProcessorApp::recordGraphData(const std::string& key, double time, double value)
{
    mGraphTime[key].push_back(time);
    mGraphData[key].push_back(value);

    // Limit buffer size
    const size_t maxSize = 1000;
    if (mGraphData[key].size() > maxSize) {
        mGraphTime[key].erase(mGraphTime[key].begin());
        mGraphData[key].erase(mGraphData[key].begin());
    }
}

void C3DProcessorApp::clearGraphData()
{
    mGraphTime.clear();
    mGraphData.clear();
}

void C3DProcessorApp::plotGraphData(const std::vector<std::string>& keys)
{
    for (const auto& key : keys) {
        if (mGraphData.find(key) == mGraphData.end()) continue;

        const auto& times = mGraphTime[key];
        const auto& values = mGraphData[key];

        if (times.empty()) continue;

        ImPlot::PlotLine(key.c_str(), times.data(), values.data(), times.size());
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
        mTrans[0] += dx * 0.005;
        mTrans[1] -= dy * 0.005;
    }
}

void C3DProcessorApp::mouseScroll(double xoff, double yoff)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    mZoom *= (1.0 + yoff * 0.1);
    mZoom = std::max(0.1, std::min(10.0, mZoom));
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
                // Reset and refresh rendered view
                mViewerTime = 0.0;
                mViewerPhase = 0.0;
                mMotionState.lastFrameIdx = 0;
                mMotionState.cycleAccumulation.setZero();
                mMotionState.manualFrameIndex = 0;
                clearGraphData();
                // Refresh marker and skeleton data
                {
                    MarkerPlaybackContext context = computeMarkerPlayback();
                    evaluateMarkerPlayback(context);
                }
                break;
            case GLFW_KEY_1:
                // Align to XY plane
                mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
                break;
            case GLFW_KEY_L:
                mRenderMarkerIndices = !mRenderMarkerIndices;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
                break;
        }
    }
}

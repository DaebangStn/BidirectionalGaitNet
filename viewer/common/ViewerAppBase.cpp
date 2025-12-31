#include "ViewerAppBase.h"
#include <rm/global.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <implot.h>

// ============================================================
// Constructor / Destructor
// ============================================================

ViewerAppBase::ViewerAppBase(const std::string& windowTitle, int width, int height)
    : mWidth(width)
    , mHeight(height)
    , mWindowTitle(windowTitle)
{
    loadRenderConfig();  // Load window geometry from render.yaml
    initGLFW();
    initImGui();

    // Initialize camera
    resetCamera();
}

ViewerAppBase::~ViewerAppBase()
{
    cleanup();
}

// ============================================================
// GLFW Initialization
// ============================================================

void ViewerAppBase::initGLFW()
{
    if (!glfwInit()) {
        std::cerr << "[ViewerAppBase] Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    // OpenGL 3.3 Compatibility Profile (needed for legacy GL functions)
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    // Create window
    mWindow = glfwCreateWindow(mWidth, mHeight, mWindowTitle.c_str(), nullptr, nullptr);
    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);
    if (!mWindow) {
        std::cerr << "[ViewerAppBase] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);  // Enable vsync

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[ViewerAppBase] Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Register callbacks
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);
    glfwSetKeyCallback(mWindow, keyCallback);

    // Get actual framebuffer size
    glfwGetFramebufferSize(mWindow, &mWidth, &mHeight);
}

// ============================================================
// ImGui Initialization
// ============================================================

void ViewerAppBase::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Load font with Korean glyph support
    ImFontConfig fontConfig;
    fontConfig.MergeMode = false;

    const char* fontPath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";
    if (std::filesystem::exists(fontPath)) {
        io.Fonts->AddFontFromFileTTF(fontPath, 16.0f, &fontConfig,
            io.Fonts->GetGlyphRangesKorean());
    } else {
        io.Fonts->AddFontDefault();
    }

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

// ============================================================
// Cleanup
// ============================================================

void ViewerAppBase::cleanup()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    if (mWindow) {
        glfwDestroyWindow(mWindow);
        mWindow = nullptr;
    }
    glfwTerminate();
}

// ============================================================
// Render Config (window geometry from render.yaml)
// ============================================================

void ViewerAppBase::loadRenderConfig()
{
    try {
        std::string resolved_path = rm::resolve("render.yaml");
        YAML::Node config = YAML::LoadFile(resolved_path);

        // Parse geometry section
        if (config["geometry"]) {
            auto geom = config["geometry"];
            if (geom["window"]) {
                auto window = geom["window"];
                if (window["width"]) mWidth = window["width"].as<int>();
                if (window["height"]) mHeight = window["height"].as<int>();
                if (window["xpos"]) mWindowXPos = window["xpos"].as<int>();
                if (window["ypos"]) mWindowYPos = window["ypos"].as<int>();
            }
            if (geom["control"]) mControlPanelWidth = geom["control"].as<int>();
            if (geom["plot"]) mPlotPanelWidth = geom["plot"].as<int>();
        }

        // Parse default_open_panels
        if (config["default_open_panels"]) {
            for (const auto& panel : config["default_open_panels"]) {
                mDefaultOpenPanels.insert(panel.as<std::string>());
            }
        }

        std::cout << "[ViewerAppBase] Loaded render.yaml: " << mWidth << "x" << mHeight
                  << " at (" << mWindowXPos << "," << mWindowYPos << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ViewerAppBase] Could not load render.yaml: " << e.what() << std::endl;
    }
}

// ============================================================
// Main Loop
// ============================================================

void ViewerAppBase::startLoop()
{
    // Load app-specific render config (derived class hook)
    loadRenderConfigImpl();

    // Call initialization hook
    onInitialize();

    while (!glfwWindowShouldClose(mWindow)) {
        glfwPollEvents();

        // Frame start hook
        onFrameStart();

        // ImGui new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // OpenGL rendering
        GUI::InitGL();
        GUI::InitLighting();

        // Camera update and apply
        updateCamera();
        setCamera();

        // Draw ground if enabled
        if (mRenderGround) GUI::DrawGroundGrid(mGroundMode);

        // Draw 3D content (pure virtual - must be overridden)
        drawContent();

        // Draw ImGui UI (pure virtual - must be overridden)
        drawUI();

        // Finalize ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(mWindow);
    }
}

// ============================================================
// Camera System
// ============================================================

void ViewerAppBase::setCamera()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, mWidth, mHeight);
    gluPerspective(mCamera.persp, getAspectRatio(), 0.1, 100.0);
    gluLookAt(mCamera.eye[0], mCamera.eye[1], mCamera.eye[2],
              0.0, 0.0, 0.0,
              mCamera.up[0], mCamera.up[1], mCamera.up[2]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Trackball rotation
    mCamera.trackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
    mCamera.trackball.setRadius(std::min(mWidth, mHeight) * 0.4);
    mCamera.trackball.applyGLRotation();

    // Scale and translate
    glScalef(mCamera.zoom, mCamera.zoom, mCamera.zoom);
    Eigen::Vector3d totalTrans = mCamera.trans + mCamera.relTrans;
    glTranslatef(totalTrans[0], totalTrans[1], totalTrans[2]);
}

void ViewerAppBase::alignCameraToPlane(int plane)
{
    // Reset trackball
    mCamera.trackball = dart::gui::Trackball();

    switch (plane) {
        case 1:  // Front view (XY plane)
            mCamera.eye = Eigen::Vector3d(0, 0, 2.5);
            mCamera.up = Eigen::Vector3d(0, 1, 0);
            break;
        case 2:  // Side view (YZ plane)
            mCamera.eye = Eigen::Vector3d(2.5, 0, 0);
            mCamera.up = Eigen::Vector3d(0, 1, 0);
            break;
        case 3:  // Top view (ZX plane)
            mCamera.eye = Eigen::Vector3d(0, 2.5, 0);
            mCamera.up = Eigen::Vector3d(0, 0, -1);
            break;
        default:
            break;
    }
}

void ViewerAppBase::resetCamera()
{
    mCamera.eye = Eigen::Vector3d(0, 0, 2.5);
    mCamera.up = Eigen::Vector3d(0, 1, 0);
    mCamera.trans = Eigen::Vector3d(0, -0.8, 0);
    mCamera.relTrans = Eigen::Vector3d::Zero();
    mCamera.zoom = 1.0;
    mCamera.persp = 45.0;
    mCamera.focus = 0;
    mCamera.trackball = dart::gui::Trackball();
}

// ============================================================
// Input Handling
// ============================================================

void ViewerAppBase::mousePress(int button, int action, int mods)
{
    // Skip if ImGui wants mouse input
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (action == GLFW_PRESS) {
        mMouseDown = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = true;
            mCamera.trackball.startBall(mMouseX, mHeight - mMouseY);
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = true;
        }
    } else if (action == GLFW_RELEASE) {
        mMouseDown = false;
        mRotate = false;
        mTranslate = false;
    }
}

void ViewerAppBase::mouseMove(double x, double y)
{
    double dx = x - mMouseX;
    double dy = y - mMouseY;
    mMouseX = x;
    mMouseY = y;

    // Skip if ImGui wants mouse input
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (mRotate) {
        mCamera.trackball.updateBall(x, mHeight - y);
    }

    if (mTranslate) {
        double scale = 0.005 / mCamera.zoom;
        Eigen::Matrix3d rot = mCamera.trackball.getRotationMatrix();
        Eigen::Vector3d delta = rot.transpose() * Eigen::Vector3d(dx * scale, -dy * scale, 0.0);

        // Update relTrans if in follow mode, otherwise update trans
        if (mCamera.focus == 1) {
            mCamera.relTrans += delta;
        } else {
            mCamera.trans += delta;
        }
    }
}

void ViewerAppBase::mouseScroll(double xoff, double yoff)
{
    // Skip if ImGui wants mouse input
    if (ImGui::GetIO().WantCaptureMouse) return;

    mCamera.zoom *= (1.0 + yoff * 0.1);
    mCamera.zoom = std::max(0.1, std::min(50.0, mCamera.zoom));
}

void ViewerAppBase::resize(int width, int height)
{
    mWidth = width;
    mHeight = height;
    glViewport(0, 0, mWidth, mHeight);
}

void ViewerAppBase::keyPress(int key, int scancode, int action, int mods)
{
    // Skip if ImGui wants keyboard input
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_1:
            case GLFW_KEY_KP_1:
                alignCameraToPlane(1);  // Front view
                break;
            case GLFW_KEY_2:
            case GLFW_KEY_KP_2:
                alignCameraToPlane(2);  // Side view
                break;
            case GLFW_KEY_3:
            case GLFW_KEY_KP_3:
                alignCameraToPlane(3);  // Top view
                break;
            case GLFW_KEY_R:
                resetCamera();
                break;
            case GLFW_KEY_F:
                mCamera.focus = (mCamera.focus == 1) ? 0 : 1;
                if (mCamera.focus == 0) {
                    mCamera.relTrans = Eigen::Vector3d::Zero();
                }
                break;
            case GLFW_KEY_G:
                mGroundMode = (mGroundMode == GroundMode::Wireframe)
                    ? GroundMode::Solid : GroundMode::Wireframe;
                break;
            case GLFW_KEY_O:
                // Cycle render mode: Primitive -> Mesh -> Wireframe -> Primitive
                mRenderMode = static_cast<RenderMode>((static_cast<int>(mRenderMode) + 1) % 3);
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
                break;
            default:
                break;
        }
    }
}

// ============================================================
// Static Callbacks
// ============================================================

void ViewerAppBase::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<ViewerAppBase*>(glfwGetWindowUserPointer(window));
    if (app) app->resize(width, height);
}

void ViewerAppBase::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    auto* app = static_cast<ViewerAppBase*>(glfwGetWindowUserPointer(window));
    if (app) app->mousePress(button, action, mods);
}

void ViewerAppBase::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* app = static_cast<ViewerAppBase*>(glfwGetWindowUserPointer(window));
    if (app) app->mouseMove(xpos, ypos);
}

void ViewerAppBase::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto* app = static_cast<ViewerAppBase*>(glfwGetWindowUserPointer(window));
    if (app) app->mouseScroll(xoffset, yoffset);
}

void ViewerAppBase::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto* app = static_cast<ViewerAppBase*>(glfwGetWindowUserPointer(window));
    if (app) app->keyPress(key, scancode, action, mods);
}

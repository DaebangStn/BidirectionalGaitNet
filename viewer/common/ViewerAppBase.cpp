#include "ViewerAppBase.h"
#include <rm/global.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <implot.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

        // Parse capture presets
        if (config["capture"]) {
            YAML::Node capture = config["capture"];
            for (auto it = capture.begin(); it != capture.end(); ++it) {
                CapturePreset p;
                p.name = it->first.as<std::string>();
                YAML::Node vals = it->second;
                p.x0 = vals["x0"].as<int>();
                p.y0 = vals["y0"].as<int>();
                p.x1 = vals["x1"].as<int>();
                p.y1 = vals["y1"].as<int>();
                mCapturePresets.push_back(p);
            }
            // Apply first preset as default
            if (!mCapturePresets.empty()) {
                mCaptureX0 = mCapturePresets[0].x0;
                mCaptureY0 = mCapturePresets[0].y0;
                mCaptureX1 = mCapturePresets[0].x1;
                mCaptureY1 = mCapturePresets[0].y1;
            }
        }

        // Parse video settings
        if (config["video"]) {
            auto video = config["video"];
            if (video["fps"]) mVideoFPS = video["fps"].as<int>();
            if (video["maxtime"]) mVideoMaxTime = video["maxtime"].as<double>();
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

        // Post-render hook (for video capture)
        onPostRender();

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
    // Note: ImGui keyboard guard is in keyCallback (static callback)
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
    // Skip if ImGui wants keyboard input (protects all derived classes)
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    auto* app = static_cast<ViewerAppBase*>(glfwGetWindowUserPointer(window));
    if (app) app->keyPress(key, scancode, action, mods);
}

// ============================================================
// Capture & Video Recording
// ============================================================

bool ViewerAppBase::captureRegionPNG(const char* filename, int x0, int y0, int x1, int y1)
{
    int width = x1 - x0;
    int height = y1 - y0;

    if (width <= 0 || height <= 0) {
        std::cerr << "[Capture] Invalid region: " << width << "x" << height << std::endl;
        return false;
    }

    // Convert to OpenGL lower-left origin
    int gl_x = x0;
    int gl_y = mHeight - y1;
    gl_y = std::max(0, gl_y);

    // Capture framebuffer
    std::vector<unsigned char> pixels(width * height * 3);
    glFinish();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(gl_x, gl_y, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip vertically (OpenGL is bottom-up, PNG is top-down)
    std::vector<unsigned char> flipped(width * height * 3);
    int stride = width * 3;
    for (int y = 0; y < height; ++y) {
        const unsigned char* src = pixels.data() + (height - 1 - y) * stride;
        unsigned char* dst = flipped.data() + y * stride;
        std::memcpy(dst, src, stride);
    }

    // Create capture directory if it doesn't exist
    std::filesystem::create_directories("capture");

    // Build full path
    std::string full_path = "capture/" + std::string(filename);

    // Write PNG
    int result = stbi_write_png(full_path.c_str(), width, height, 3, flipped.data(), stride);
    return result != 0;
}

bool ViewerAppBase::startVideoRecording(const std::string& filename, int fps)
{
    if (mFFmpegPipe) {
        stopVideoRecording();
    }

    // Create capture directory if it doesn't exist
    std::filesystem::create_directories("capture");

    // Calculate capture region dimensions
    int x0 = (int)(mWidth * 0.5) + mCaptureX0;
    int y0 = mCaptureY0;
    int x1 = (int)(mWidth * 0.5) + mCaptureX1;
    int y1 = mCaptureY1;

    // Clamp to window bounds
    x0 = std::clamp(x0, 0, mWidth);
    y0 = std::clamp(y0, 0, mHeight);
    x1 = std::clamp(x1, 0, mWidth);
    y1 = std::clamp(y1, 0, mHeight);

    int width = std::max(0, x1 - x0);
    int height = std::max(0, y1 - y0);

    // Ensure even dimensions for x264
    width = (width / 2) * 2;
    height = (height / 2) * 2;

    if (width <= 0 || height <= 0) {
        std::cerr << "[Video] Invalid capture region: " << width << "x" << height << std::endl;
        return false;
    }

    // Build ffmpeg command
    std::string full_path = "capture/" + filename;
    std::string cmd = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24";
    cmd += " -s " + std::to_string(width) + "x" + std::to_string(height);
    cmd += " -r " + std::to_string(fps);
    cmd += " -i - -c:v libx264 -pix_fmt yuv420p";
    cmd += " -crf 23 -preset medium";
    cmd += " -movflags +faststart";
    cmd += " \"" + full_path + "\" 2>/dev/null";

    mFFmpegPipe = popen(cmd.c_str(), "w");
    if (!mFFmpegPipe) {
        std::cerr << "[Video] Failed to open ffmpeg pipe for: " << full_path << std::endl;
        return false;
    }

    // Initialize recording state
    mVideoRecording = true;
    mVideoElapsedTime = 0.0;
    mVideoFrameCounter = 0;
    mVideoFrameSkip = 1;

    std::cout << "[Video] Started recording: " << full_path
              << " (fps=" << fps << ", region=" << width << "x" << height << ")" << std::endl;
    return true;
}

void ViewerAppBase::stopVideoRecording()
{
    if (mFFmpegPipe) {
        pclose(mFFmpegPipe);
        mFFmpegPipe = nullptr;
        mVideoRecording = false;

        std::cout << "[Video] Recording stopped. Duration: "
                  << std::fixed << std::setprecision(1) << mVideoElapsedTime
                  << "s, Frames: " << mVideoFrameCounter << std::endl;
    }
}

void ViewerAppBase::recordVideoFrame()
{
    if (!mVideoRecording || !mFFmpegPipe) return;

    // Check maximum recording time
    if (mVideoMaxTime > 0 && mVideoElapsedTime >= mVideoMaxTime) {
        std::cout << "[Video] Maximum recording time reached ("
                  << mVideoMaxTime << "s). Stopping recording." << std::endl;
        onMaxRecordingTimeReached();
        stopVideoRecording();
        return;
    }

    // Calculate capture region
    int x0 = (int)(mWidth * 0.5) + mCaptureX0;
    int y0 = mCaptureY0;
    int x1 = (int)(mWidth * 0.5) + mCaptureX1;
    int y1 = mCaptureY1;

    // Clamp to window bounds
    x0 = std::clamp(x0, 0, mWidth);
    y0 = std::clamp(y0, 0, mHeight);
    x1 = std::clamp(x1, 0, mWidth);
    y1 = std::clamp(y1, 0, mHeight);

    int width = std::max(0, x1 - x0);
    int height = std::max(0, y1 - y0);

    // Ensure even dimensions
    width = (width / 2) * 2;
    height = (height / 2) * 2;

    if (width <= 0 || height <= 0) return;

    // Convert to OpenGL lower-left origin
    int gl_x = x0;
    int gl_y = mHeight - (y0 + height);
    gl_y = std::max(0, gl_y);

    // Capture region framebuffer as RGB24
    std::vector<unsigned char> pixels(width * height * 3);
    std::vector<unsigned char> flipped(width * height * 3);

    glFinish();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(gl_x, gl_y, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Flip vertically (OpenGL is bottom-up, video is top-down)
    int stride = width * 3;
    for (int y = 0; y < height; ++y) {
        const unsigned char* src = pixels.data() + (height - 1 - y) * stride;
        unsigned char* dst = flipped.data() + y * stride;
        std::memcpy(dst, src, stride);
    }

    // Write frame to ffmpeg pipe
    size_t written = fwrite(flipped.data(), 1, flipped.size(), mFFmpegPipe);
    if (written != flipped.size()) {
        std::cerr << "[Video] Failed to write frame to ffmpeg pipe" << std::endl;
        stopVideoRecording();
        return;
    }
    fflush(mFFmpegPipe);

    mVideoFrameCounter++;

    // Update elapsed time (assume ~60fps rendering)
    mVideoElapsedTime += 1.0 / 60.0;
}

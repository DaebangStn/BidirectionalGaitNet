#ifndef VIEWER_APP_BASE_H
#define VIEWER_APP_BASE_H

#include "dart/gui/Trackball.hpp"
#include "ShapeRenderer.h"
#include "GLfunctions.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <Eigen/Dense>
#include <string>
#include <memory>

/**
 * @brief Base class for GLFW-based viewer applications
 *
 * Provides common functionality:
 * - GLFW window initialization and management
 * - OpenGL context setup (Compatibility Profile)
 * - ImGui initialization and frame management
 * - Camera system with trackball rotation
 * - Mouse and keyboard input handling
 * - Main loop structure
 *
 * Derived classes override:
 * - drawContent() for 3D rendering
 * - drawUI() for ImGui panels
 * - keyPress() for app-specific keyboard handling
 * - updateCamera() for camera follow modes (optional)
 */
class ViewerAppBase
{
public:
    ViewerAppBase(const std::string& windowTitle, int width = 1920, int height = 1080);
    virtual ~ViewerAppBase();

    // Main entry point - runs the application loop
    void startLoop();

protected:
    // ============================================================
    // Camera System
    // ============================================================
    struct CameraState {
        Eigen::Vector3d eye{0, 0, 2.5};
        Eigen::Vector3d up{0, 1, 0};
        Eigen::Vector3d trans{0, -0.8, 0};
        Eigen::Vector3d relTrans = Eigen::Vector3d::Zero();  // User offset for follow mode
        double zoom = 1.0;
        double persp = 45.0;
        int focus = 0;  // 0 = free camera, 1 = follow target
        dart::gui::Trackball trackball;
    };
    CameraState mCamera;

    // ============================================================
    // Input State
    // ============================================================
    bool mMouseDown = false;
    bool mRotate = false;
    bool mTranslate = false;
    double mMouseX = 0.0;
    double mMouseY = 0.0;

    // ============================================================
    // Window
    // ============================================================
    GLFWwindow* mWindow = nullptr;
    int mWidth;
    int mHeight;
    std::string mWindowTitle;

    // ============================================================
    // Rendering
    // ============================================================
    RenderMode mRenderMode = RenderMode::Mesh;
    GroundMode mGroundMode = GroundMode::Wireframe;
    ShapeRenderer mShapeRenderer;
    bool mRenderGround = true;

    // ============================================================
    // Virtual Methods (Override in Derived Classes)
    // ============================================================

    // Called before setCamera() to update camera position (e.g., follow mode)
    virtual void updateCamera() {}

    // Pure virtual - render 3D content (skeleton, muscles, markers, etc.)
    virtual void drawContent() = 0;

    // Pure virtual - render ImGui panels
    virtual void drawUI() = 0;

    // Keyboard handling - base provides common shortcuts, override for app-specific
    virtual void keyPress(int key, int scancode, int action, int mods);

    // Called once after GLFW/ImGui init, before main loop
    virtual void onInitialize() {}

    // Called each frame before rendering
    virtual void onFrameStart() {}

    // ============================================================
    // Protected Helpers
    // ============================================================

    // Apply camera transformation to OpenGL
    void setCamera();

    // Align camera to standard viewing planes
    // 1 = Front (XY), 2 = Side (YZ), 3 = Top (ZX)
    void alignCameraToPlane(int plane);

    // Reset camera to default position
    void resetCamera();

    // Get current window aspect ratio
    double getAspectRatio() const { return static_cast<double>(mWidth) / mHeight; }

private:
    // ============================================================
    // GLFW Callbacks (static)
    // ============================================================
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // ============================================================
    // Initialization
    // ============================================================
    void initGLFW();
    void initImGui();
    void cleanup();

    // ============================================================
    // Input Handling
    // ============================================================
    void mousePress(int button, int action, int mods);
    void mouseMove(double x, double y);
    void mouseScroll(double xoff, double yoff);
    void resize(int width, int height);
};

#endif // VIEWER_APP_BASE_H

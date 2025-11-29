#ifndef C3D_PROCESSOR_APP_H
#define C3D_PROCESSOR_APP_H

#include "dart/gui/Trackball.hpp"
#include "RenderCharacter.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include "C3D_Reader.h"
#include "C3D.h"
#include "Motion.h"
#include "CBufferData.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <memory>
#include <string>
#include <vector>
#include <set>

/**
 * @brief Skeleton render mode
 */
enum class RenderMode { Primitive, Mesh, Wireframe };

/**
 * @brief Playback navigation mode for C3D marker data
 */
enum C3DNavigationMode
{
    C3D_SYNC = 0,           // Automatic playback (synced with viewer time)
    C3D_MANUAL_FRAME        // Manual frame selection via slider
};

/**
 * @brief Viewer state for C3D playback
 */
struct C3DViewerState
{
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> currentMarkers;
    Eigen::VectorXd currentPose;  // Current skeleton pose
    Eigen::Vector3d cycleAccumulation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cycleDistance = Eigen::Vector3d::Zero();
    int lastFrameIdx = 0;
    int maxFrameIndex = 0;
    bool render = true;
    C3DNavigationMode navigationMode = C3D_SYNC;
    int manualFrameIndex = 0;
};

/**
 * @brief Standalone C3D processor application
 *
 * Features:
 * - Load and play C3D motion files (no HDF support)
 * - Render C3D markers and skeleton markers
 * - Marker correspondence visualization
 * - Skeleton fitting/calibration from C3D data
 * - MarkerDiff plots for fitting quality
 */
class C3DProcessorApp
{
public:
    C3DProcessorApp(const std::string& skeletonPath, const std::string& markerPath);
    ~C3DProcessorApp();

    void startLoop();
    void loadC3DFile(const std::string& path);

private:
    // GLFW window
    GLFWwindow* mWindow;
    int mWidth, mHeight;
    int mWindowXPos, mWindowYPos;

    // Camera
    Eigen::Vector3d mEye;
    Eigen::Vector3d mUp;
    Eigen::Vector3d mTrans;
    double mZoom;
    double mPersp;
    dart::gui::Trackball mTrackball;
    int mFocus;  // Camera follow mode (0=free, 1=follow skeleton COM)

    // Mouse state
    bool mMouseDown;
    bool mRotate;
    bool mTranslate;
    double mMouseX, mMouseY;

    // Character and rendering
    std::unique_ptr<RenderCharacter> mMotionCharacter;
    ShapeRenderer mShapeRenderer;
    std::string mSkeletonPath;
    std::string mMarkerConfigPath;

    // C3D Processing
    C3D_Reader* mC3DReader;
    Motion* mMotion;  // C3D motion only
    std::string mMotionPath;

    // Motion file list (C3D files only)
    std::vector<std::string> mMotionList;
    int mSelectedMotion = -1;
    std::string mDirectorySearchPath;

    // Marker rendering flags
    bool mRenderC3DMarkers;
    bool mRenderExpectedMarkers;
    bool mRenderMarkerIndices;
    bool mRenderJointPositions = false;
    float mMarkerLabelFontSize = 18.0f;

    // Axis rendering flags
    bool mRenderWorldAxis = false;
    bool mRenderSkeletonAxis = false;
    bool mCameraMoving = false;  // True while camera is being manipulated
    double mCameraMovingTimer = 0.0;  // Timer for scroll-based axis display
    float mAxisLength = 0.3f;

    // Skeleton render mode
    RenderMode mRenderMode = RenderMode::Wireframe;

    // Rendering panel
    bool mShowRenderingPanel = false;

    // Per-marker visibility (empty = all visible)
    std::set<int> mHiddenC3DMarkers;
    std::set<int> mHiddenSkelMarkers;

    // Marker search/selection
    char mMarkerSearchFilter[64] = "";
    char mRenderingMarkerFilter[64] = "";

    // Motion pose inspection
    int mPoseInspectFrame = 0;
    bool mPoseUseCurrentFrame = true;
    char mJointFilter[64] = "";
    int mSelectedJointIdx = -1;

    // Playback state
    C3DViewerState mMotionState;
    Eigen::Vector3d mC3DCOM;

    // Viewer time management
    double mViewerTime;
    double mViewerPhase;
    float mViewerPlaybackSpeed;
    double mViewerCycleDuration;
    double mLastRealTime;
    bool mIsPlaying;
    bool mProgressForward = false;  // Whether character progresses forward with cycle distance

    // Plot control
    double mXmin;
    bool mPlotHideLegend;

    // Marker playback context
    struct MarkerPlaybackContext
    {
        C3D* markers = nullptr;
        C3DViewerState* state = nullptr;
        double phase = 0.0;
        double frameFloat = 0.0;
        int frameIndex = 0;
        int totalFrames = 0;
        bool valid = false;
    };

    // Graph data buffer for plots (cyclic buffer)
    CBufferData<double>* mGraphData;

    // Configuration
    std::set<std::string> mDefaultOpenPanels;
    int mControlPanelWidth;

    // Initialization
    void initGL();
    void initImGui();
    void initLighting();
    void setCamera();
    void loadRenderConfig();

    // Rendering
    void drawFrame();
    void drawSkeleton();
    void drawMarkers();
    void drawGround();
    void drawAxis(const Eigen::Isometry3d& transform, float length, const std::string& label);
    void drawOriginAxisGizmo();

    // UI panels
    void drawControlPanel();
    void drawMotionListSection();
    void drawPlaybackSection();
    void drawMarkerFittingSection();
    void drawVisualizationPanel();
    void drawRenderingPanel();
    void drawMarkerDiffPlot();
    void drawMarkerCorrespondenceTable();
    void drawMotionPoseSection();

    // Helper for collapsing header
    bool collapsingHeaderWithControls(const std::string& title);
    bool isPanelDefaultOpen(const std::string& panelName) const;

    // C3D processing
    void scanC3DFiles();
    Eigen::Vector3d computeMarkerCycleDistance(C3D* markerData);
    MarkerPlaybackContext computeMarkerPlayback();
    void evaluateMarkerPlayback(const MarkerPlaybackContext& context);
    double computeMarkerHeightCalibration(const std::vector<Eigen::Vector3d>& markers);
    void updateViewerTime(double dt);
    void computeViewerMetric();


    // Input callbacks
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    void resize(int width, int height);
    void mousePress(int button, int action, int mods);
    void mouseMove(double x, double y);
    void mouseScroll(double xoff, double yoff);
    void keyPress(int key, int scancode, int action, int mods);
    void alignCameraToPlane(int plane);
    void reset();
};

#endif // C3D_PROCESSOR_APP_H

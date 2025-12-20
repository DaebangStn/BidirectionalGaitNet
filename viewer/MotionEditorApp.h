#ifndef MOTION_EDITOR_APP_H
#define MOTION_EDITOR_APP_H

#include "dart/gui/Trackball.hpp"
#include "RenderCharacter.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include "Motion.h"
#include "HDF.h"
#include "rm/rm.hpp"
#include "motion/PlaybackController.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <memory>
#include <string>
#include <vector>
#include <set>

/**
 * @brief Skeleton render mode
 */
enum class MotionEditorRenderMode { Primitive, Wireframe };

/**
 * @brief Playback navigation mode
 */
enum MotionEditorNavigationMode
{
    ME_SYNC = 0,           // Automatic playback (synced with viewer time)
    ME_MANUAL_FRAME        // Manual frame selection via slider
};

/**
 * @brief Viewer state for motion playback
 */
struct MotionEditorViewerState
{
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();
    Eigen::VectorXd currentPose;
    Eigen::Vector3d cycleAccumulation = Eigen::Vector3d::Zero();
    Eigen::Vector3d cycleDistance = Eigen::Vector3d::Zero();
    int lastFrameIdx = 0;
    int maxFrameIndex = 0;
    bool render = true;
    MotionEditorNavigationMode navigationMode = ME_SYNC;
    int manualFrameIndex = 0;
};

/**
 * @brief Foot contact phase data
 */
struct FootContactPhase {
    int startFrame;
    int endFrame;
    bool isLeft;
};

/**
 * @brief ROM violation data for a single DOF
 */
struct ROMViolation {
    std::string jointName;      // Joint name
    int dofIndex;               // DOF index in skeleton
    int localDofIndex;          // DOF index within joint (for multi-DOF joints)
    int numJointDofs;           // Total DOFs in this joint
    int startFrame;             // First frame of violation
    int maxDiffFrame;           // Frame with maximum violation
    int endFrame;               // Last frame of violation
    double maxAngle;            // Actual angle at max violation (rad)
    double boundValue;          // The limit that was hit
    bool isUpperBound;          // true = exceeded upper, false = below lower
};

/**
 * @brief Motion Editor Application
 *
 * Features:
 * - Load H5 motion files via PID browser or direct path
 * - Auto-detect skeleton from PID folder
 * - Visualize skeleton with motion playback
 * - Trim motion by setting start/end frames
 * - Export trimmed motion to new H5 file
 */
class MotionEditorApp
{
public:
    MotionEditorApp(const std::string& configPath = "");
    ~MotionEditorApp();

    void startLoop();

private:
    // === Window & GL ===
    GLFWwindow* mWindow;
    int mWidth = 1280, mHeight = 720;
    int mWindowXPos = 0, mWindowYPos = 0;

    // === Camera ===
    Eigen::Vector3d mEye{0, 0, 2.5};
    Eigen::Vector3d mUp{0, 1, 0};
    Eigen::Vector3d mTrans{0, -0.8, 0};
    double mZoom = 1.0;
    double mPersp = 45.0;
    dart::gui::Trackball mTrackball;
    int mFocus = 0;  // 0=fixed, 1=follow skeleton

    // === Input State ===
    bool mMouseDown = false;
    bool mRotate = false;
    bool mTranslate = false;
    bool mCameraMoving = false;
    double mMouseX = 0, mMouseY = 0;

    // === Resource Manager (singleton reference) ===
    rm::ResourceManager* mResourceManager = nullptr;
    std::string mConfigPath;

    // === PID Browser State ===
    std::vector<std::string> mPIDList;
    std::vector<std::string> mPIDNames;
    std::vector<std::string> mPIDGMFCS;
    int mSelectedPID = -1;
    char mPIDFilter[128] = {0};
    bool mPreOp = true;

    // === H5 File Browser ===
    std::vector<std::string> mH5Files;
    int mSelectedH5 = -1;
    char mH5Filter[64] = {0};

    // === Motion Data ===
    Motion* mMotion = nullptr;
    std::unique_ptr<RenderCharacter> mCharacter;
    MotionEditorViewerState mMotionState;
    std::string mMotionSourcePath;

    // === Playback Timing ===
    double mViewerTime = 0.0;
    double mViewerPhase = 0.0;
    float mPlaybackSpeed = 1.0f;
    bool mIsPlaying = false;
    double mCycleDuration = 1.0;
    double mLastRealTime = 0.0;

    // === Skeleton ===
    std::string mAutoDetectedSkeletonPath;
    bool mUseAutoSkeleton = true;
    char mManualSkeletonPath[512] = {0};
    std::string mCurrentSkeletonPath;
    std::vector<std::string> mSkeletonFiles;      // List of skeleton files in directory
    std::vector<std::string> mSkeletonFileNames;  // Display names (filename only)
    int mSelectedSkeletonFile = -1;               // Selected skeleton index
    std::string mSkeletonDirectory;               // Current skeleton directory

    // === Trim State ===
    int mTrimStart = 0;
    int mTrimEnd = 0;

    // === Export ===
    char mExportFilename[256] = {0};
    bool mAutoSuffix = true;
    std::string mLastExportMessage;
    std::string mLastExportURI;  // PID-style URI of last exported file
    double mLastExportMessageTime = 0.0;

    // === Rendering ===
    ShapeRenderer mShapeRenderer;
    MotionEditorRenderMode mRenderMode = MotionEditorRenderMode::Wireframe;
    GroundMode mGroundMode = GroundMode::Wireframe;
    float mControlPanelWidth = 350.0f;
    float mRightPanelWidth = 300.0f;

    // === Configuration ===
    std::set<std::string> mDefaultOpenPanels;

    // === Rotation Processing ===
    float mPendingRotationAngle = 0.0f;     // degrees (preview always shown when != 0)

    // === Height Processing ===
    double mComputedHeightOffset = 0.0;     // calculated offset
    bool mHeightOffsetComputed = false;     // whether calculation done

    // === Foot Contact Detection ===
    std::vector<FootContactPhase> mDetectedPhases;
    int mSelectedPhase = -1;
    float mContactVelocityThreshold = 0.01f;  // m/frame
    int mContactMinLockFrames = 5;

    // === Stride Estimation ===
    int mStrideBodyNodeIdx = 0;          // 0=TalusR, 1=TalusL, 2=Pelvis
    int mStrideDivider = 1;              // Divider for stride calculation
    int mStrideCalcMode = 0;             // 0=Z only, 1=XZ magnitude
    double mComputedStride = -1.0;       // Computed stride value (-1 = not computed)

    // === ROM Violation Detection ===
    std::vector<ROMViolation> mROMViolations;
    int mSelectedViolation = -1;
    bool mPreviewClampedPose = true;

    // === Initialization ===
    void setCamera();
    void updateCamera();
    void loadRenderConfig();

    // === Rendering ===
    void drawFrame();
    void drawSkeleton(bool isPreview = false);

    // === UI Panels ===
    void drawLeftPanel();
    void drawRightPanel();
    void drawPIDBrowserTab();
    void drawDirectPathTab();
    void drawSkeletonSection();
    void drawPlaybackSection();
    void drawMotionInfoSection();
    void drawTrimSection();
    void drawExportSection();
    void drawRotationSection();
    void drawHeightSection();
    void drawFootContactSection();
    void drawStrideEstimationSection();
    void drawROMViolationSection();

    // === Processing ===
    void detectROMViolations();

    // === Helper for collapsing header ===
    bool collapsingHeaderWithControls(const std::string& title);
    bool isPanelDefaultOpen(const std::string& panelName) const;

    // === PID Scanner Methods ===
    void scanPIDList();
    void scanH5Files();
    void scanSkeletonDirectory();
    void loadH5Motion(const std::string& path);
    void autoDetectSkeleton();
    void loadSkeleton(const std::string& path);

    // === Playback ===
    void updateViewerTime(double dt);
    void evaluateMotionPose();
    Eigen::Vector3d computeMotionCycleDistance();

    // === Export ===
    void exportMotion();

    // === Trim ===
    void applyTrim();

    // === Processing ===
    Eigen::VectorXd applyRotationToFrame(const Eigen::VectorXd& pose, float angleDegrees);
    void applyRotation();
    void computeGroundLevel();
    void applyHeightOffset();
    Eigen::Vector3d getBodyNodeSize(dart::dynamics::BodyNode* bn);
    void detectFootContacts();
    Eigen::Vector4d getRenderColor(const dart::dynamics::BodyNode* bn,
                                    const Eigen::Vector4d& defaultColor) const;

    // === Static Callbacks ===
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // === Input Handlers ===
    void resize(int width, int height);
    void mousePress(int button, int action, int mods);
    void mouseMove(double x, double y);
    void mouseScroll(double xoff, double yoff);
    void keyPress(int key, int scancode, int action, int mods);
    void alignCameraToPlane(int plane);
    void reset();
};

#endif // MOTION_EDITOR_APP_H

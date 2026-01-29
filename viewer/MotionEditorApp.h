#ifndef MOTION_EDITOR_APP_H
#define MOTION_EDITOR_APP_H

#include "common/ViewerAppBase.h"
#include "common/timeline.h"
#include "RenderCharacter.h"
#include "common/ShapeRenderer.h"
#include "Motion.h"
#include "HDF.h"
#include "rm/rm.hpp"
#include "motion/PlaybackController.h"
#include "common/PIDImGui.h"
#include <memory>
#include <string>
#include <vector>
#include <set>

/**
 * @brief Skeleton render mode (app-specific)
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
 * @brief Direction interval for consecutive frames with same walking direction
 */
struct DirectionInterval {
    int startFrame;
    int endFrame;
    Timeline::GaitDirection direction;
};

/**
 * @brief Motion Editor Application
 *
 * Inherits from ViewerAppBase for common window/camera/input handling.
 *
 * Features:
 * - Load H5 motion files via PID browser or direct path
 * - Auto-detect skeleton from PID folder
 * - Visualize skeleton with motion playback
 * - Trim motion by setting start/end frames
 * - Export trimmed motion to new H5 file
 */
class MotionEditorApp : public ViewerAppBase
{
public:
    MotionEditorApp(const std::string& configPath = "");
    ~MotionEditorApp() override;

protected:
    // ViewerAppBase overrides
    void onInitialize() override;
    void onFrameStart() override;
    void updateCamera() override;
    void drawContent() override;
    void drawUI() override;
    void keyPress(int key, int scancode, int action, int mods) override;

private:
    // === Window position (from config) ===
    int mWindowXPos = 0, mWindowYPos = 0;

    // === Resource Manager (singleton reference) ===
    rm::ResourceManager* mResourceManager = nullptr;
    std::string mConfigPath;

    // === PID Browser (using shared PIDNavigator) ===
    std::unique_ptr<PIDNav::PIDNavigator> mPIDNavigator;

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
    MotionEditorRenderMode mAppRenderMode = MotionEditorRenderMode::Wireframe;
    float mControlPanelWidth = 350.0f;
    float mRightPanelWidth = 300.0f;
    // mDefaultOpenPanels inherited from ViewerAppBase

    // === Rotation Processing ===
    float mPendingRotationAngle = 0.0f;     // degrees (preview always shown when != 0)

    // === Height Processing ===
    double mComputedHeightOffset = 0.0;     // calculated offset (single - for backward compat)
    bool mHeightOffsetComputed = false;     // whether calculation done
    std::vector<double> mIntervalHeightOffsets;  // Per-interval height offsets

    // === Foot Contact Detection ===
    std::vector<Timeline::FootContactPhase> mDetectedPhases;
    int mSelectedPhase = -1;
    float mContactVelocityThreshold = 0.01f;  // m/frame
    int mContactMinLockFrames = 5;

    // === Timeline Zoom State ===
    float mTimelineZoom = 1.0f;         // Zoom level (1.0 = no zoom)
    float mTimelineScrollOffset = 0.0f; // Scroll offset when zoomed

    // === Stride Estimation ===
    int mStrideBodyNodeIdx = 0;          // 0=TalusR, 1=TalusL, 2=Pelvis
    int mStrideDivider = 1;              // Divider for stride calculation
    int mStrideCalcMode = 0;             // 0=Z only, 1=XZ magnitude
    double mComputedStride = -1.0;       // Computed stride value (-1 = not computed)

    // === ROM Violation Detection ===
    std::vector<ROMViolation> mROMViolations;
    int mSelectedViolation = -1;
    bool mPreviewClampedPose = true;

    // === Direction Cleanup ===
    std::vector<DirectionInterval> mDirectionIntervals;
    int mSelectedDirectionInterval = -1;
    int mRotationStartFrame = 0;
    int mRotationEndFrame = 0;
    int mInterpolationFrames = 10;  // Number of frames to interpolate between intervals

    // === Initialization ===
    void loadRenderConfigImpl() override;

    // === Rendering ===
    void drawSkeleton(bool isPreview = false);

    // === UI Panels ===
    void drawLeftPanel();
    void drawRightPanel();
    void drawPIDBrowserTab();  // Updated to use PIDNavigator
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
    void drawDirectionCleanupSection();

    // === Processing ===
    void detectROMViolations();
    void detectDirectionIntervals();
    void straightenBackwardIntervals();

    // === Helper for collapsing header ===
    bool collapsingHeaderWithControls(const std::string& title);
    // isPanelDefaultOpen() inherited from ViewerAppBase

    // === Data Loading ===
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
    double computeGroundLevelForRange(int startFrame, int endFrame);
    void applyHeightOffset();
    Eigen::Vector3d getBodyNodeSize(dart::dynamics::BodyNode* bn);
    void detectFootContacts();
    void setStrideDivider(int divider);
    void computeStride();
    Eigen::Vector4d getRenderColor(const dart::dynamics::BodyNode* bn,
                                    const Eigen::Vector4d& defaultColor) const;

    // === App-specific helpers ===
    void alignCameraToPlaneQuat(int plane);  // Quaternion-based camera alignment
    void resetPlayback();                     // Reset playback state
    void refreshMotion();                     // Refresh motion state after load/trim
    void drawTimelineTrackBar();              // Draw timeline trackbar at bottom
};

#endif // MOTION_EDITOR_APP_H

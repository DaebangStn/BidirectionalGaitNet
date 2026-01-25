#ifndef C3D_PROCESSOR_APP_H
#define C3D_PROCESSOR_APP_H

#include "common/ViewerAppBase.h"
#include "RenderCharacter.h"
#include "ShapeRenderer.h"
#include "C3D_Reader.h"
#include "C3D.h"
#include "Motion.h"
#include "CBufferData.h"
#include "rm/rm.hpp"
#include "common/PIDImGui.h"
#include <implot.h>
#include <memory>
#include <string>
#include <vector>
#include <set>

/**
 * @brief Playback navigation mode for C3D marker data
 */
enum C3DNavigationMode
{
    C3D_SYNC = 0,           // Automatic playback (synced with viewer time)
    C3D_MANUAL_FRAME        // Manual frame selection via slider
};

/**
 * @brief Calibration mode for marker fitting
 * - Static: Uses mFreeCharacter only, single frame, medial markers for bone fitting
 * - Dynamic: Uses both mFreeCharacter and mMotionCharacter for full motion
 */
enum class CalibrationMode
{
    Static,     // Static calibration with medial markers (single frame)
    Dynamic     // Dynamic calibration (full motion sequence)
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
 * Inherits from ViewerAppBase for common window/camera/input handling.
 *
 * Features:
 * - Load and play C3D motion files (no HDF support)
 * - Render C3D markers and skeleton markers
 * - Marker correspondence visualization
 * - Skeleton fitting/calibration from C3D data
 * - MarkerDiff plots for fitting quality
 */
class C3DProcessorApp : public ViewerAppBase
{
public:
    explicit C3DProcessorApp(const std::string& configPath = "@data/config/c3d_processor.yaml");
    ~C3DProcessorApp() override;

    void loadC3DFile(const std::string& path);

protected:
    // ViewerAppBase overrides
    void onInitialize() override;
    void onFrameStart() override;
    void updateCamera() override;
    void drawContent() override;
    void drawUI() override;
    void keyPress(int key, int scancode, int action, int mods) override;

private:
    // === Configuration ===
    std::string mConfigPath;  // Main config file path
    void loadConfig();        // Load all settings from config file

    // === Window position (from config) ===
    int mWindowXPos = 0, mWindowYPos = 0;

    // Character and rendering
    std::unique_ptr<RenderCharacter> mFreeCharacter;   // Free joints skeleton for bone-by-bone debugging
    std::unique_ptr<RenderCharacter> mMotionCharacter; // Normal skeleton for motion playback
    std::string mSkeletonPath;
    std::string mMarkerConfigPath;
    std::string mInitialMarkerPath;  // Store initial marker path for reset
    std::string mFittingConfigPath;

    // C3D Processing
    C3D_Reader* mC3DReader;
    Motion* mMotion;  // C3D motion only
    std::string mMotionPath;
    bool mAutoloadFirstC3D = true;  // Auto-load first C3D file at startup

    // Motion source tracking
    enum class MotionSource { None, FileList, PID };
    MotionSource mMotionSource = MotionSource::None;

    // Motion file list (C3D files only)
    std::vector<std::string> mMotionList;        // Full paths for loading
    std::vector<std::string> mMotionDisplayNames; // Relative paths for display
    int mSelectedMotion = -1;
    char mC3DFilter[64] = "";                    // Filter text for C3D files
    std::string mDirectorySearchPath;

    // Resource Manager for PID-based access (singleton reference)
    rm::ResourceManager* mResourceManager = nullptr;

    // PID Navigator for clinical data browsing
    std::unique_ptr<PIDNav::PIDNavigator> mPIDNavigator;

    // Marker rendering flags
    bool mRenderC3DMarkers;
    bool mRenderExpectedMarkers;
    bool mRenderMarkerIndices;
    bool mRenderJointPositions = false;
    float mMarkerLabelFontSize = 18.0f;
    float mMarkerAlpha = 0.6f;

    // Axis rendering flags
    bool mRenderWorldAxis = false;
    bool mRenderSkeletonAxis = false;
    float mAxisLength = 0.3f;

    // Free character rendering
    bool mRenderFreeCharacter = true;

    // Motion character rendering
    bool mRenderMotionCharacter = true;
    bool mRenderMotionCharMarkers = true;
    Eigen::Vector3d mMotionCharacterOffset = Eigen::Vector3d(0.8, 0.0, 0.0);

    // Skeleton scale section character selection (0=Free, 1=Motion)
    int mScaleCharacterSelection = 0;

    // Per-marker visibility (empty = all visible)
    std::set<int> mHiddenC3DMarkers;
    std::set<int> mHiddenSkelMarkers;

    // Marker search/selection
    char mMarkerSearchFilter[64] = "";
    char mRenderingMarkerFilter[64] = "";

    // Bone pose inspection
    char mBonePoseFilter[64] = "";
    int mBonePoseSelectedIdx = -1;

    // Joint angle inspection
    char mJointAngleFilter[64] = "";
    int mJointAngleSelectedIdx = -1;

    // Joint offset inspection
    char mJointOffsetFilter[64] = "";
    int mJointOffsetSelectedIdx = -1;

    // Skeleton export
    char mExportSkeletonName[128] = "calibrated_skeleton";
    char mExportCalibrationName[128] = "dynamic";
    char mExportHDFName[256] = "";  // Empty = use C3D stem

    // Calibration mode
    CalibrationMode mCalibrationMode = CalibrationMode::Dynamic;
    bool mHasMedialMarkers = false;  // Detected from current C3D file
    bool mHasPersonalizedCalibration = false;  // For current PID/prePost
    bool mPersonalizedCalibrationLoaded = false;  // Calibration has been loaded
    StaticCalibrationResult mStaticCalibResult;
    DynamicCalibrationResult mDynamicCalibResult;
    std::string mStaticConfigPath = "@data/config/static_fitting.yaml";

    // Personalized calibration scale path (populated when loaded)
    std::string mPersonalizedScalePath;

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
    // mDefaultOpenPanels and mControlPanelWidth inherited from ViewerAppBase

    // === Initialization ===
    void loadRenderConfigImpl() override;  // Loads c3d section from render.yaml

    // === Rendering ===
    void drawSkeleton();
    void drawMarkers();
    void drawAxis(const Eigen::Isometry3d& transform, float length, const std::string& label);
    void drawSelectedJointGizmo();
    void drawSelectedBoneGizmo();

    // === UI panels ===
    void drawLeftPanel();
    void drawRightPanel();
    void drawTimelineTrackBar();
    void drawMotionListSection();
    void drawPlaybackSection();
    void drawMarkerFittingSection();
    void drawSkeletonScaleSection();
    void drawSkeletonExportSection();
    void drawViewTabContent();
    void drawMarkerDiffPlot();
    void drawMarkerCorrespondenceTable();
    void drawBonePoseSection();
    void drawJointAngleSection();
    void drawJointOffsetSection();
    void drawClinicalDataSection();

    // === HDF export ===
    void exportMotionToHDF5();

    // === PID Navigator callbacks ===
    void onPIDFileSelected(const std::string& path, const std::string& filename);
    void checkForPersonalizedCalibration();
    bool loadPersonalizedCalibration(const std::string& inputDir);

    // === Motion source helper ===
    std::string getCurrentMotionPath() const;
    void reloadCurrentMotion(bool withCalibration);

    // === Helper for collapsing header ===
    bool collapsingHeaderWithControls(const std::string& title);
    // isPanelDefaultOpen() inherited from ViewerAppBase

    // === C3D processing ===
    void scanC3DFiles();
    Eigen::Vector3d computeMarkerCycleDistance(C3D* markerData);
    MarkerPlaybackContext computeMarkerPlayback();
    void evaluateMarkerPlayback(const MarkerPlaybackContext& context);
    double computeMarkerHeightCalibration(const std::vector<Eigen::Vector3d>& markers);
    void updateViewerTime(double dt);
    void computeViewerMetric();

    // === App-specific helpers ===
    void alignCameraToPlaneQuat(int plane);  // Quaternion-based camera alignment
    void resetPlaybackAndState();            // Reset playback, graph data, and skeleton state
    void hideVirtualMarkers();
    void clearMotionAndZeroPose();
};

#endif // C3D_PROCESSOR_APP_H

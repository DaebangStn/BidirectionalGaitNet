#include "dart/gui/Trackball.hpp"
#include "Environment.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include "CBufferData.h"
#include "RenderEnvironment.h"
#include "RenderCharacter.h"
#include "Motion.h"
#include "HDF.h"
// NOTE: C3D processing moved to c3d_processor executable
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui_internal.h>
// C3D_Reader moved to c3d_processor
#include "motion/MotionProcessor.h"
#include "rm/rm.hpp"
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include "ImGuiFileDialog.h"
#include <memory>
#include <set>

struct ResizablePlot {
    std::vector<std::string> keys;
    char newKeyInput[256] = {0};
    int selectedKey = -1;
};

enum MuscleRenderingType
{
    passiveForce = 0,
    contractileForce,
    activationLevel,
    contracture,
    weakness
};

enum class SkeletonRenderMode
{
    Solid = 0,
    Wireframe,
    Mesh
};

struct RolloutStatus
{
    std::string name, settingPath, fileContents, storeKey;
    char memo[64];
    int cycle; // Remaining gait cycles. if -1, rollout continues indefinitely, if 0, rollout is finished
    bool pause;
    bool modulate;
    bool isAlarmed;
    bool store;
    bool fake_assist;

    RolloutStatus() { reset(); }

    void reset()
    {
        cycle = -1;
        pause = true;
        modulate = false;
        isAlarmed = false;
        store = false;
        fake_assist = false;
        name = "";
        settingPath = "";
        fileContents = "";
        storeKey = "";
        std::memset(memo, 0, sizeof(memo));
    }

    void step()
    {
        if (cycle > 0) cycle--;
        if (cycle == 0) pause = true;
    }
};

// Legacy ViewerMotion struct - replaced by Motion* polymorphic interface
// struct ViewerMotion
// {
//     std::string name;
//     Eigen::VectorXd param;
//     Eigen::VectorXd motion;  // Flattened motion data: NPZ: 3030 (30 frames × 101 values), HDF: variable
//     int values_per_frame = 101;  // Values per frame: NPZ=101 (6D rotation), HDF/BVH=56 (skeleton DOF), C3D=101
//     int num_frames = 30;         // Number of frames loaded: NPZ=30 (first cycle only), HDF=total frames
//     std::string source_type = "npz";  // Source format: "npz", "hdfRollout", "hdfSingle", "bvh", or "c3d"
//
//     // NPZ-specific: track total frames in file vs frames loaded
//     int npz_total_frames = 60;   // Total frames in NPZ file (60 = 2 cycles × 30 frames/cycle)
//     int npz_frames_per_cycle = 30;  // Frames per gait cycle (30)
//
//     // HDF-specific timing (for correct playback speed)
//     int hdf5_total_timesteps = 0;      // Total simulation timesteps across all cycles
//     int hdf5_timesteps_per_cycle = 0;  // Average timesteps per gait cycle (for phase mapping)
//
//     // Per-motion root positioning (clear semantics for HDF coordinate alignment)
//     Eigen::Vector3d initialRootPosition = Eigen::Vector3d::Zero();  // First frame root joint position [3,4,5]
//     Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();        // World offset: (simulated_char - motion_frame0) + overlap_prevention
//     std::vector<Eigen::Vector3d> rootBodyCOM;                       // Root body COM trajectory from root/x,y,z (HDF5 only, optional)
//     std::vector<double> timestamps;                                 // HDF5: actual simulation time for each frame (for accurate interpolation)
//
//     // Current evaluated pose (computed in updateViewerTime, used in drawPlayableMotion)
//     Eigen::VectorXd currentPose;
// };

/**
 * @brief Unified navigation mode for all playable data (motion, markers, etc.)
 */
enum PlaybackNavigationMode
{
    PLAYBACK_SYNC = 0,           // Automatic playback (synced with viewer time or other data)
    PLAYBACK_MANUAL_FRAME        // Manual frame selection via slider
};

/**
 * @brief Unified draw flags for rendering control
 */
struct DrawFlags
{
    bool character = true;
    bool pdTarget = false;
    bool refMotion = false;
    bool jointSphere = false;
    bool footStep = false;
    bool eoe = false;
    bool collision = false;
    bool noiseArrows = true;
    bool fgnSkeleton = false;
    bool obj = true;
    SkeletonRenderMode skeletonRenderMode = SkeletonRenderMode::Solid;
};

/**
 * @brief Unified viewer-specific state for playback display (motion and markers)
 *
 * Separates viewer concerns (positioning, caching, navigation) from data sources.
 * Used for both Motion* instances and standalone C3D marker data.
 *
 * Fields are optional based on data type:
 * - currentPose: Used for skeleton motion playback
 * - currentMarkers: Used for C3D or standalone marker playback
 */
struct PlaybackViewerState
{
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();        ///< World offset for display positioning
    Eigen::VectorXd currentPose;                                     ///< Last evaluated pose (cached from Motion::getPose(), optional)
    std::vector<Eigen::Vector3d> currentMarkers;                     ///< Current marker positions (for C3D or standalone markers, optional)
    Eigen::Vector3d cycleAccumulation = Eigen::Vector3d::Zero();     ///< Accumulated root translation across cycles
    Eigen::Vector3d cycleDistance = Eigen::Vector3d::Zero();         ///< Pre-computed cycle distance (for cycle wrap accumulation)
    int lastFrameIdx = 0;                                            ///< Last evaluated frame index (for wrap detection)
    int maxFrameIndex = 0;                                           ///< Maximum frame index for this data
    bool render = true;                                               ///< Whether to render this data (default: enabled)
    PlaybackNavigationMode navigationMode = PLAYBACK_SYNC;           ///< Playback mode (sync or manual frame selection)
    int manualFrameIndex = 0;                                        ///< Manual frame index when navigationMode == PLAYBACK_MANUAL_FRAME
};

class GLFWApp
{
public:
    GLFWApp(int argc, char **argv);
    ~GLFWApp();
    void startLoop();
    
private:
    py::object mns;
    py::object loading_network;

    // Helper functions for pre-computing cycle distances
    Eigen::Vector3d computeMotionCycleDistance(Motion* motion);

    // REMOVED: writeBVH(), exportBVH() - BVH format no longer supported

    void plotGraphData(const std::vector<std::string>& keys, ImAxis y_axis = ImAxis_Y1,
        std::string postfix = "", bool show_stat = false, int color_ofs = 0);

    std::map<std::string, std::map<std::string, double>>
    statGraphData(const std::vector<std::string>& keys, double xMin, double xMax);

    void plotPhaseBar(double x_min, double x_max, double y_min, double y_max);

    // Collapsing header with 2x height control (uses config system for default open)
    bool collapsingHeaderWithControls(const std::string& title);
    
    void initEnv(std::string metadata);
    void initGL();
    void update(bool isSave = false);
    void reset();
    void updateViewerTime(double dt);  // Update viewer time, phase, and motion state

    void setWindowIcon(const char* icon_path);

    // Drawing Component
    void setCamera();

    void drawSimFrame();
    void drawUIFrame();
    void drawSimControlPanel();
    void drawKinematicsControlPanel();
    void drawVisualizationPanel();
    void drawTimingPane();
    void drawTitlePanel();
    void drawResizablePlotPane();
    void drawCameraStatusSection();
    void drawPlayableMotion();

    // Helper to check if current motion is from a specific source
    bool isCurrentMotionFromSource(const std::string& sourceType, const std::string& sourceFile);

    void drawFGNControl();
    void drawBGNControl();
    // C3D controls moved to c3d_processor
    void drawMotionControl();
    void drawGVAEControl();
    void addSimulationMotion();

    void printCameraInfo();
    void initializeCameraPresets();
    void loadCameraPreset(int index);
    void alignCameraToPlane(int plane);  // 1=XY, 2=YZ, 3=ZX
    void drawGround();
    void drawCollision();

    void drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color);
    
    void drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr);

    void drawSingleBodyNode(const BodyNode *bn, const Eigen::Vector4d &color);
    void drawFootStep();
    void drawPhase(double phase, double normalized_phase);

    void drawShape(const dart::dynamics::Shape *shape, const Eigen::Vector4d &color);

    void drawAxis();
    void drawJointAxis(dart::dynamics::Joint* joint);
    void drawMuscles(MuscleRenderingType renderingType = activationLevel);

    void drawNoiseControlPanel();
    void drawNoiseVisualizations();

    void drawShadow();

    // Plot control
    float getHeelStrikeTime();

    // Mousing Function
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset);

    // Keyboard Function
    void keyboardPress(int key, int scancode, int action, int mods);

    // Camera presets
    struct CameraPreset {
        std::string description;
        Eigen::Vector3d eye;
        Eigen::Vector3d up;
        Eigen::Vector3d trans;
        double zoom;
        Eigen::Quaterniond quat;
        bool isSet;
    };
    CameraPreset mCameraPresets[3];
    int mCurrentCameraPreset;

    // Variable
    double mWidth, mHeight;
    bool mRotate, mTranslate, mZooming, mMouseDown;

    GLFWwindow *mWindow;
    RenderEnvironment *mRenderEnv;
    RenderCharacter *mMotionCharacter;


    ShapeRenderer mShapeRenderer;

    // Trackball/Camera variables
    dart::gui::Trackball mTrackball;
    double mZoom, mPersp, mMouseX, mMouseY;
    Eigen::Vector3d mTrans, mEye, mUp;
    Eigen::Vector3d mRelTrans;  // User's manual translation offset (preserved across focus modes)
    int mFocus;

    // mMotionSkeleton removed - use mMotionCharacter->getSkeleton() instead

    std::vector<std::string> mNetworkPaths;
    std::vector<Network> mNetworks;

    // Graph Data Buffer
    CBufferData<double>* mGraphData;

    // Rollout Control
    RolloutStatus mRolloutStatus;

    // Rendering Options
    DrawFlags mDrawFlags;
    bool mStochasticPolicy;

    MuscleRenderingType mMuscleRenderType;
    int mMuscleRenderTypeInt;
    float mMuscleResolution, mMuscleTransparency;

    // Noise Injector UI state
    int mNoiseMode;  // 0=None, 1=Position, 2=Force, 3=Activation, 4=All

    // Muscle Selection UI
    char mMuscleFilterText[32];
    std::vector<bool> mMuscleSelectionStates;

    // Muscle Activation Plot UI
    char mActivationFilterText[256];
    std::vector<std::string> mSelectedActivationKeys;
    bool mPlotActivationNoise;

    // Muscle Rendering Option
    std::vector<Muscle *> mSelectedMuscles;
    std::vector<bool> mRelatedDofs;

    // Using Weights
    std::vector<bool> mUseWeights;

    std::vector<std::string> mFGNList;
    std::vector<std::string> mBGNList;

    py::object mFGN;
    std::string mFGNmetadata;
    Eigen::Vector3d mFGNRootOffset;
    int selected_fgn;
    int selected_bgn;

    // Unified motion file list (paths only, loaded on-demand)
    std::vector<std::string> mMotionList;  // All motion files (HDF + C3D)
    int mSelectedMotion = -1;               // Currently selected motion index
    void scanMotionFiles();                 // Scan directories for motion files
    void loadMotionFile(const std::string& path);  // Load motion on-demand

    // Resource Manager for PID-based access
    std::unique_ptr<rm::ResourceManager> mResourceManager;

    // Clinical Data (PID) browser state
    std::vector<std::string> mPIDList;
    std::vector<std::string> mPIDNames;
    std::vector<std::string> mPIDGMFCS;
    int mSelectedPID = -1;
    char mPIDFilter[64] = "";
    bool mPreOp = true;

    // HDF files for selected PID
    std::vector<std::string> mPIDHDFFiles;
    int mSelectedPIDHDF = -1;
    char mPIDHDFFilter[64] = "";

    // Clinical Data (PID-based HDF access) methods
    void drawClinicalDataSection();
    void scanPIDList();
    void scanPIDHDFFiles();
    void loadPIDHDFFile(const std::string& filename);

    // Motion Buffer
    std::vector<Eigen::VectorXd> mMotionBuffer;
    std::vector<Eigen::Matrix3d> mJointCalibration;

    // C3D processing moved to c3d_processor executable
    std::string mSkeletonPath;  // Skeleton path from simulator metadata
    std::string mMotionPath;    // Current motion file path for reloading

    // For GVAE
    py::object mGVAE;
    bool mGVAELoaded;
    std::vector<BoneInfo> mSkelInfosForMotions;

    // Single motion architecture (new/delete pattern)
    Motion* mMotion;                            ///< Single active motion instance
    PlaybackViewerState mMotionState;           ///< Viewer state for active motion
    std::unique_ptr<MotionProcessor> mMotionProcessor;  ///< Unified motion processor

    MotionData mPredictedMotion;

    // Motion management
    void unloadMotion();                              // Unload all motions and reset parameters

    // NEW: Load parameters from currently selected motion (works with new Motion* architecture)
    void loadParametersFromCurrentMotion();           // Load parameters from mMotionsNew[mMotionIdx] to environment

    struct ViewerClock
    {
        double time = 0.0;
        double phase = 0.0;
    };

    struct MotionPlaybackContext
    {
        Motion* motion = nullptr;
        PlaybackViewerState* state = nullptr;
        RenderCharacter* character = nullptr;
        double phase = 0.0;
        double frameFloat = 0.0;
        double wrappedFrameFloat = 0.0;
        int frameIndex = 0;
        int totalFrames = 0;
        int valuesPerFrame = 0;
    };

    // MarkerPlaybackContext moved to c3d_processor

    ViewerClock updateViewerClock(double dt);
    bool computeMotionPlayback(MotionPlaybackContext& context);
    void evaluateMotionPlayback(const MotionPlaybackContext& context);

    double computeMotionPhase();
    double determineMotionFrame(Motion* motion, PlaybackViewerState& state, double phase);
    void updateMotionCycleAccumulation(Motion* current_motion,
                                       PlaybackViewerState& state,
                                       int current_frame_idx,
                                       RenderCharacter* character,
                                       int value_per_frame);

    bool mLoadSimulationOnStartup = true;  // Whether to load simulation environment on startup
    void drawMotions(Eigen::VectorXd motion, Eigen::VectorXd skel_param, Eigen::Vector3d offset = Eigen::Vector3d(-1.0,0,0), Eigen::Vector4d color = Eigen::Vector4d(0.2,0.2,0.8,0.7)) {
        if (!mMotionCharacter || !mRenderEnv) return;
        auto skel = mMotionCharacter->getSkeleton();

        // (1) Set Motion Skeleton
        double global = skel_param[2];
        for(auto& m : mSkelInfosForMotions){
            if(std::get<0>(m).find("Head") == std::string::npos)
            {
                std::get<1>(m).value[0] = global;
                std::get<1>(m).value[1] = global;
                std::get<1>(m).value[2] = global;
            }
        }


        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("FemurL")->getIndexInSkeleton()]).value[1] *= skel_param[3];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("FemurR")->getIndexInSkeleton()]).value[1] *= skel_param[4];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("TibiaL")->getIndexInSkeleton()]).value[1] *= skel_param[5];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("TibiaR")->getIndexInSkeleton()]).value[1] *= skel_param[6];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("ArmL")->getIndexInSkeleton()]).value[0] *= skel_param[7];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("ArmR")->getIndexInSkeleton()]).value[0] *= skel_param[8];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("ForeArmL")->getIndexInSkeleton()]).value[0] *= skel_param[9];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("ForeArmR")->getIndexInSkeleton()]).value[0] *= skel_param[10];

        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("FemurL")->getIndexInSkeleton()]).value[4] = skel_param[11];
        std::get<1>(mSkelInfosForMotions[skel->getBodyNode("FemurR")->getIndexInSkeleton()]).value[4] = skel_param[12];


        mRenderEnv->getCharacter()->applySkeletonBodyNode(mSkelInfosForMotions, skel);

        int pos_dof = mRenderEnv->getCharacter()->posToSixDof(mRenderEnv->getCharacter()->getSkeleton()->getPositions()).rows();
        if (motion.rows() != pos_dof * 60) {
            std::cout << "Motion Dimension is not matched" << motion.rows()  << std::endl;
            return;
        }

        // (2) Draw Skeleton according to motion
        Eigen::Vector3d pos = offset;
        glColor4f(color[0], color[1], color[2], color[3]);

        for(int i = 0; i < 60; i++)
        {
            Eigen::VectorXd skel_pos = mRenderEnv->getCharacter()->sixDofToPos(motion.segment(i*pos_dof, pos_dof));
            pos[0] += motion.segment(i * pos_dof, pos_dof)[6];
            pos[1] = motion.segment(i * pos_dof, pos_dof)[7];
            pos[2] += motion.segment(i * pos_dof, pos_dof)[8];

            skel_pos.segment(3, 3) = pos;
            if (i % 6 == 0)
                drawSkeleton(skel_pos, color);
        }

        mRenderEnv->getCharacter()->updateRefSkelParam(skel);
    }

    std::vector<Eigen::VectorXd> mTestMotion;
    // mC3DCOM moved to c3d_processor
    bool mRenderConditions;

    // Viewer independent time management
    double mViewerTime;              // Viewer's master time counter
    double mViewerPhase;             // Phase value [0, 1) for cyclic motion
    float mViewerPlaybackSpeed;     // Playback speed multiplier (1.0 = normal speed)
    double mLastPlaybackSpeed;       // Previous playback speed for change detection
    double mViewerCycleDuration;     // Duration of one motion cycle (default 2.0/1.1)
    double mLastRealTime;            // Last real time from glfwGetTime()
    double mSimulationStepDuration;  // Measured simulation step duration
    double mSimStepDurationAvg;      // Moving average of simulation step duration
    double mRealDeltaTimeAvg;        // Moving average of real frame delta time
    bool mIsPlaybackTooFast;         // Warning: playback faster than simulation can handle
    bool mProgressForward = false;   // Whether character progresses forward with cycle distance
    bool mShowTimingPane;            // Toggle for timing information pane
    bool mShowResizablePlotPane;     // Toggle for the new resizable plot pane
    bool mShowTitlePanel;            // Toggle for title panel (Ctrl+T)

    // For Resizable Plot Pane
    std::vector<ResizablePlot> mResizablePlots;
    char mResizePlotKeys[1024];
    bool mResizePlotPane, mSetResizablePlotPane;

    // Plot X-axis range
    double mXmin, mXminResizablePlotPane, mYminResizablePlotPane, mYmaxResizablePlotPane;

    // Joint Control
    void drawJointControlSection();

    // Plot title control
    bool mPlotTitle, mPlotTitleResizablePlotPane;
    std::string mCheckpointName;

    // Plot legend control
    bool mPlotHideLegend;

    // Default open panels configuration
    std::set<std::string> mDefaultOpenPanels;
    bool isPanelDefaultOpen(const std::string& panelName) const;

    // Configuration loading
    void loadRenderConfig();

    // Panel widths from config
    int mControlPanelWidth;
    int mPlotPanelWidth;

    // Window position from config
    int mWindowXPos;
    int mWindowYPos;

    // Rollout configuration
    int mDefaultRolloutCount, mRolloutCycles = -1;

    // Reset phase configuration
    double mResetPhase;  // -1.0 for randomized, 0.0-1.0 for specific phase

    // Cached metadata path
    std::string mCachedMetadata;

    // Override paths for skeleton and muscle config
    std::string mSkeletonOverride;
    std::string mMuscleOverride;

    // Helper methods for initEnv
    void loadNetworkFromPath(const std::string& path);
    void initializeMotionSkeleton();
    void initializeMotionCharacter(const std::string& metadata);  // Standalone motion character init
    void updateUnifiedKeys();
    void updateResizablePlotsFromKeys();
    void runRollout();

    // Motion navigation helper - NEW: using Motion* interface
    double computeFrameFloat(Motion* motion, double phase);
    void motionPoseEval(Motion* motion, int motionIdx, double frame_float);  // Delegates to mMotionProcessor
    double computeMotionHeightCalibration(const Eigen::VectorXd& motion_pose);
    void alignMotionToSimulation();
    void setMotion(Motion* motion);  // Helper: delete old, assign new, initialize state

    // Store muscle network state_dict for transfer to Environment
    py::object mMuscleStateDict;
};

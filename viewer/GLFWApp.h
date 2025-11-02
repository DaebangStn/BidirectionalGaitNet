#include "dart/gui/Trackball.hpp"
#include "Environment.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include "CBufferData.h"
#include "RenderEnvironment.h"
#include "Character.h"
#include "BVH_Parser.h"
#include "Motion.h"
#include "HDF.h"
#include "NPZ.h"
#include "HDFRollout.h"
#include "C3D.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include "C3D_Reader.h"
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include "ImGuiFileDialog.h"
#include <memory>

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
 * @brief Viewer-specific state for motion display
 *
 * Separates viewer concerns (positioning, caching) from motion data.
 * Each Motion* instance has a corresponding MotionViewerState.
 */
struct MotionViewerState
{
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();        ///< World offset for motion display
    Eigen::VectorXd currentPose;                                     ///< Last evaluated pose (cached from Motion::getPose())
    Eigen::Vector3d initialRootPosition = Eigen::Vector3d::Zero();   ///< Initial root position for delta calculations (HDF/BVH)
    Eigen::Vector3d cycleAccumulation = Eigen::Vector3d::Zero();     ///< Accumulated root translation across cycles
    int lastFrameIdx = 0;                                            ///< Last evaluated frame index (for wrap detection)
    PlaybackNavigationMode navigationMode = PLAYBACK_SYNC;           ///< Playback mode for this motion
    int manualFrameIndex = 0;                                        ///< Manual frame index when navigationMode == PLAYBACK_MANUAL_FRAME
    std::vector<Eigen::Vector3d> currentMarkers;                     ///< Current marker positions (for C3DMotion)
};

struct MarkerViewerState
{
    std::vector<Eigen::Vector3d> currentMarkers;
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();
    Eigen::Vector3d cycleAccumulation = Eigen::Vector3d::Zero();
    int lastFrameIdx = 0;
    PlaybackNavigationMode navigationMode = PLAYBACK_SYNC;
    int manualFrameIndex = 0;
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

    void writeBVH(const dart::dynamics::Joint *jn, std::ofstream &_f, const bool isPos = false); // Pose Or Hierarchy
    void exportBVH(const std::vector<Eigen::VectorXd> &motion, const dart::dynamics::SkeletonPtr &skel);
    
    void plotGraphData(const std::vector<std::string>& keys, ImAxis y_axis = ImAxis_Y1,
        bool show_phase = true, bool plot_avg_copy = false, std::string postfix = "",
        bool show_stat = false);

    std::map<std::string, std::map<std::string, double>>
    statGraphData(const std::vector<std::string>& keys, double xMin, double xMax);

    void plotPhaseBar(double x_min, double x_max, double y_min, double y_max);
    
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
    void drawSimVisualizationPanel();
    void drawTimingPane();
    void drawTitlePanel();
    void drawResizablePlotPane();
    void drawCameraStatusSection();
    void drawPlayableMotion();
    void drawPlayableMarkers();

    // Helper to check if current motion is from a specific source
    bool isCurrentMotionFromSource(const std::string& sourceType, const std::string& sourceFile);

    void drawFGNControl();
    void drawBGNControl();
    void drawC3DControl();
    void drawMotionControl();
    void drawGVAEControl();
    void addSimulationMotion();

    void printCameraInfo();
    void initializeCameraPresets();
    void loadCameraPreset(int index);
    void drawGround();
    void drawCollision();

    void drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color, bool isLineSkeleton = false);
    
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
    bool mRenderMode;
    double mWidth, mHeight;
    bool mRotate, mTranslate, mZooming, mMouseDown;

    GLFWwindow *mWindow;
    RenderEnvironment *mRenderEnv;
    Character *mMotionCharacter;


    ShapeRenderer mShapeRenderer;
    bool mDrawOBJ;

    // Trackball/Camera variables
    dart::gui::Trackball mTrackball;
    double mZoom, mPersp, mMouseX, mMouseY;
    Eigen::Vector3d mTrans, mEye, mUp;
    Eigen::Vector3d mRelTrans;  // User's manual translation offset (preserved across focus modes)
    int mFocus;

    // Skeleton for kinematic drawing
    dart::dynamics::SkeletonPtr mMotionSkeleton;

    std::vector<std::string> mNetworkPaths;
    std::vector<Network> mNetworks;

    // Graph Data Buffer
    CBufferData<double>* mGraphData;

    // Rollout Control
    RolloutStatus mRolloutStatus;

    // Rendering Option
    bool mDrawReferenceSkeleton;
    bool mDrawCharacter;
    bool mDrawPDTarget;
    bool mDrawJointSphere;
    bool mDrawFootStep;
    bool mStochasticPolicy;
    bool mDrawEOE;

    MuscleRenderingType mMuscleRenderType;
    int mMuscleRenderTypeInt;
    float mMuscleResolution, mMuscleTransparency;

    // Noise Injector UI state
    bool mDrawNoiseArrows;
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
    std::vector<std::string> mC3DList;

    py::object mFGN;
    std::string mFGNmetadata;
    Eigen::Vector3d mFGNRootOffset;
    int selected_fgn;
    int selected_bgn;
    int selected_c3d;
    bool mDrawFGNSkeleton;

    // Motion Buffer
    std::vector<Eigen::VectorXd> mMotionBuffer;
    std::vector<Eigen::Matrix3d> mJointCalibration;

    // C3D loading and rendering
    C3D_Reader* mC3DReader;
    std::string mSkeletonPath;  // Skeleton path from simulator metadata
    std::unique_ptr<C3D> mC3DMarkers;  // For marker-only rendering
    bool mRenderC3DMarkers;
    MarkerViewerState mMarkerState;


    // For GVAE
    py::object mGVAE;
    bool mGVAELoaded;
    std::vector<BoneInfo> mSkelInfosForMotions;

    // Polymorphic motion architecture
    std::vector<Motion*> mMotions;              ///< Polymorphic motion instances
    std::vector<MotionViewerState> mMotionStates;  ///< Viewer state per motion

    MotionData mPredictedMotion;

    int mMotionIdx;
    PlaybackNavigationMode mFallbackMotionNavigationMode;
    int mFallbackManualFrameIndex;

    // HDF5 selection (for selective param/cycle loading)
    std::vector<std::string> mHDF5Files;              // Available HDF5 files
    std::vector<std::string> mHDF5Params;             // Available params in selected file
    std::vector<std::string> mHDF5Cycles;             // Available cycles in selected param
    int mSelectedHDF5FileIdx;                         // Currently selected file
    int mSelectedHDF5ParamIdx;                        // Currently selected param (drag value)
    int mSelectedHDF5CycleIdx;                        // Currently selected cycle (drag value)
    int mMaxHDF5ParamIdx;                             // Maximum param index for selected file
    int mMaxHDF5CycleIdx;                             // Maximum cycle index for selected param
    std::string mCurrentHDF5FilePath;                 // Path to current HDF5 file
    std::string mMotionLoadError;                     // Error message for motion loading failures
    std::string mParamFailureMessage;                 // Error message for parameter failures
    std::string mLastLoadedHDF5ParamsFile;            // Track which HDF5 file's parameters are loaded
    void scanHDF5Structure();                         // Scan HDF5 file to populate params/cycles
    void loadSelectedHDF5Motion();                    // Load specific param/cycle combination
    void unloadMotion();                              // Unload all motions and reset parameters

    // NEW: Load parameters from currently selected motion (works with new Motion* architecture)
    void loadParametersFromCurrentMotion();           // Load parameters from mMotionsNew[mMotionIdx] to environment
    double computeMarkerHeightCalibration(const std::vector<Eigen::Vector3d>& markers);
    void alignMarkerToSimulation();
    void markerPoseEval(double frameFloat);

    struct ViewerClock
    {
        double time = 0.0;
        double phase = 0.0;
    };

    struct MotionPlaybackContext
    {
        Motion* motion = nullptr;
        MotionViewerState* state = nullptr;
        Character* character = nullptr;
        double phase = 0.0;
        double frameFloat = 0.0;
        double wrappedFrameFloat = 0.0;
        int frameIndex = 0;
        int totalFrames = 0;
        int valuesPerFrame = 0;
    };

    struct MarkerPlaybackContext
    {
        C3D* markers = nullptr;
        MarkerViewerState* state = nullptr;
        double phase = 0.0;
        double frameFloat = 0.0;
        int frameIndex = 0;
        int totalFrames = 0;
        bool valid = false;
    };

    ViewerClock updateViewerClock(double dt);
    bool computeMotionPlayback(MotionPlaybackContext& context);
    MarkerPlaybackContext computeMarkerPlayback(const ViewerClock& clock,
                                                const MotionPlaybackContext* motionContext);
    void evaluateMarkerPlayback(const MarkerPlaybackContext& context);
    void evaluateMotionPlayback(const MotionPlaybackContext& context);

    double computeMotionPhase();
    double determineMotionFrame(Motion* motion, MotionViewerState& state, double phase);
    void updateMotionCycleAccumulation(Motion* current_motion,
                                       MotionViewerState& state,
                                       int current_frame_idx,
                                       Character* character,
                                       int value_per_frame);

    std::string mMotionLoadMode;  // Motion loading mode: "no" to disable, otherwise loads all types (npz, hdfRollout, hdfSingle, bvh)
    void drawMotions(Eigen::VectorXd motion, Eigen::VectorXd skel_param, Eigen::Vector3d offset = Eigen::Vector3d(-1.0,0,0), Eigen::Vector4d color = Eigen::Vector4d(0.2,0.2,0.8,0.7)) {
        
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

        
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurL")->getIndexInSkeleton()]).value[1] *= skel_param[3];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurR")->getIndexInSkeleton()]).value[1] *= skel_param[4];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("TibiaL")->getIndexInSkeleton()]).value[1] *= skel_param[5];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("TibiaR")->getIndexInSkeleton()]).value[1] *= skel_param[6];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ArmL")->getIndexInSkeleton()]).value[0] *= skel_param[7];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ArmR")->getIndexInSkeleton()]).value[0] *= skel_param[8];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ForeArmL")->getIndexInSkeleton()]).value[0] *= skel_param[9];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("ForeArmR")->getIndexInSkeleton()]).value[0] *= skel_param[10];

        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurL")->getIndexInSkeleton()]).value[4] = skel_param[11];
        std::get<1>(mSkelInfosForMotions[mMotionSkeleton->getBodyNode("FemurR")->getIndexInSkeleton()]).value[4] = skel_param[12];


        mRenderEnv->getCharacter()->applySkeletonBodyNode(mSkelInfosForMotions, mMotionSkeleton);

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

        mRenderEnv->getCharacter()->updateRefSkelParam(mMotionSkeleton);
    }

    std::vector<Eigen::VectorXd> mTestMotion;
    Eigen::Vector3d mC3DCOM;
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
    bool mShowTimingPane;            // Toggle for timing information pane
    bool mShowResizablePlotPane;     // Toggle for the new resizable plot pane
    bool mShowTitlePanel;            // Toggle for title panel (Ctrl+T)

    // Motion navigation control
    int mMaxFrameIndex;                           // Maximum frame index for current motion

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

    // Helper methods for initEnv
    void loadNetworkFromPath(const std::string& path);
    void initializeMotionSkeleton();
    void loadMotionFiles();
    void loadNPZMotion();
    void loadHDFRolloutMotion();
    void loadBVHMotion();
    void loadHDFSingleMotion();
    void updateUnifiedKeys();
    void updateResizablePlotsFromKeys();
    void runRollout();

    // Motion navigation helper - NEW: using Motion* interface
    double computeFrameFloat(Motion* motion, double phase);
    void motionPoseEval(Motion* motion, int motionIdx, double frame_float);
    double computeMotionHeightCalibration(const Eigen::VectorXd& motion_pose);
    void alignMotionToSimulation();
};

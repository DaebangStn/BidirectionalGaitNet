#include "dart/gui/Trackball.hpp"
#include "Environment.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include "CBufferData.h"
#include "RenderEnvironment.h"
#include "Character.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include "C3D_Reader.h"
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>

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

struct ViewerMotion
{
    std::string name;
    Eigen::VectorXd param;
    Eigen::VectorXd motion;  // Flattened motion data: NPZ: 6060 (60 cycles × 101 frames), HDF5: variable
    int frames_per_cycle = 101;  // NPZ: timesteps per cycle (101), HDF5: skeleton DOF (56)
    int num_cycles = 60;         // NPZ: number of cycles (60), HDF5: actual gait cycles loaded
    std::string source_type = "npz";  // Source format: "npz" or "hdf5"

    // HDF5-specific timing (for correct playback speed)
    int hdf5_total_timesteps = 0;      // Total simulation timesteps across all cycles
    int hdf5_timesteps_per_cycle = 0;  // Average timesteps per gait cycle (for phase mapping)

    // Per-motion root positioning (clear semantics for HDF5 coordinate alignment)
    Eigen::Vector3d initialRootPosition = Eigen::Vector3d::Zero();  // First frame root joint position [3,4,5]
    Eigen::Vector3d displayOffset = Eigen::Vector3d::Zero();        // World offset: (simulated_char - motion_frame0) + overlap_prevention
    std::vector<Eigen::Vector3d> rootBodyCOM;                       // Root body COM trajectory from root/x,y,z (HDF5 only, optional)
    std::vector<double> timestamps;                                 // HDF5: actual simulation time for each frame (for accurate interpolation)

    // Current evaluated pose (computed in updateViewerTime, used in drawPlayableMotion)
    Eigen::VectorXd currentPose;
};

enum MotionNavigationMode
{
    NAVIGATION_VIEWER_TIME = 0,  // Automatic playback driven by viewer time
    NAVIGATION_MANUAL_FRAME      // Manual frame selection via slider
};

class GLFWApp
{
public:
    GLFWApp(int argc, char **argv, bool rendermode = true);
    ~GLFWApp();
    void startLoop();
    
private:
    py::object mns;
    py::object loading_network;

    void writeBVH(const dart::dynamics::Joint *jn, std::ofstream &_f, const bool isPos = false); // Pose Or Hierarchy
    void exportBVH(const std::vector<Eigen::VectorXd> &motion, const dart::dynamics::SkeletonPtr &skel);
    
    void plotGraphData(const std::vector<std::string>& keys, ImAxis y_axis = ImAxis_Y1,
        bool show_phase = true, bool plot_avg_copy = false, std::string postfix = "");
        
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
    void drawResizablePlotPane();
    void drawCameraStatusSection();
    void drawPlayableMotion();

    void drawFGNControl();
    void drawBGNControl();
    void drawC3DControl();
    void drawMotionControl();
    void drawGVAEControl();
    void addSimulationMotion();

    void printCameraInfo();
    void initializeCameraPresets();
    void loadCameraPreset(int index);
    void drawGround(double height);
    void drawCollision();

    void drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color, bool isLineSkeleton = false);
    
    void drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr);

    void drawSingleBodyNode(const BodyNode *bn, const Eigen::Vector4d &color);
    void drawFootStep();
    void drawPhase(double phase, double normalized_phase);

    void drawShape(const dart::dynamics::Shape *shape, const Eigen::Vector4d &color);

    void drawAxis();
    void drawMuscles(MuscleRenderingType renderingType = activationLevel);

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

    // Muscle Selection UI
    char mMuscleFilterText[32];
    std::vector<bool> mMuscleSelectionStates;

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

    // BVH Buffer

    std::vector<Eigen::VectorXd> mC3dMotion;
    int mC3DCount;

    C3D_Reader* mC3DReader;


    // For GVAE
    py::object mGVAE;
    bool mGVAELoaded;
    std::vector<BoneInfo> mSkelInfosForMotions;
    std::vector<ViewerMotion> mMotions;
    std::vector<ViewerMotion> mAddedMotions;
    Motion mPredictedMotion;

    int mMotionIdx;
    int mMotionFrameIdx;

    // Motion playback tracking (cycle accumulation and forward progress - used for both NPZ and HDF5)
    int mLastFrameIdx;
    Eigen::Vector3d mCycleAccumulation;

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
    void scanHDF5Structure();                         // Scan HDF5 file to populate params/cycles
    void loadSelectedHDF5Motion();                    // Load specific param/cycle combination

    bool mDrawMotion;
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


        mRenderEnv->getCharacter(0)->applySkeletonBodyNode(mSkelInfosForMotions, mMotionSkeleton);

        int pos_dof = mRenderEnv->getCharacter(0)->posToSixDof(mRenderEnv->getCharacter(0)->getSkeleton()->getPositions()).rows();
        if (motion.rows() != pos_dof * 60) {
            std::cout << "Motion Dimension is not matched" << motion.rows()  << std::endl;
            return;
        }

        // (2) Draw Skeleton according to motion
        Eigen::Vector3d pos = offset;
        glColor4f(color[0], color[1], color[2], color[3]);
        
        for(int i = 0; i < 60; i++)
        {
            Eigen::VectorXd skel_pos = mRenderEnv->getCharacter(0)->sixDofToPos(motion.segment(i*pos_dof, pos_dof));
            pos[0] += motion.segment(i * pos_dof, pos_dof)[6];
            pos[1] = motion.segment(i * pos_dof, pos_dof)[7];
            pos[2] += motion.segment(i * pos_dof, pos_dof)[8];

            skel_pos.segment(3, 3) = pos;
            if (i % 6 == 0)
                drawSkeleton(skel_pos, color);
        }

        mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
    }

    std::vector<Eigen::VectorXd> mTestMotion;
    Eigen::Vector3d mC3DCOM;
    bool mRenderConditions;
    bool mRenderC3D;

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

    // Motion navigation control
    MotionNavigationMode mMotionNavigationMode;  // Current navigation mode
    int mManualFrameIndex;                        // Manual frame index (when in manual mode)
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
    int mDefaultRolloutCount;

    // Cached metadata path
    std::string mCachedMetadata;

    // Helper methods for initEnv
    void loadNetworkFromPath(const std::string& path);
    void initializeMotionSkeleton();
    void loadMotionFiles();
    void updateUnifiedKeys();
    void updateResizablePlotsFromKeys();

    // Motion navigation helper
    double computeFrameFloat(const ViewerMotion& motion, double phase);
    void motionPoseEval(ViewerMotion& motion, double frame_float);
    void alignMotionToSimulation();
};
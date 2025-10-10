#ifndef __PHYSICAL_EXAM_H__
#define __PHYSICAL_EXAM_H__

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <vector>
#include <map>
#include <string>
#include <optional>
#include "dart/dart.hpp"
#include "dart/gui/Trackball.hpp"
#include "Character.h"
#include "ShapeRenderer.h"
#include "CBufferData.h"
#include "SurgeryOperation.h"
#include "../sim/SurgeryExecutor.h"
#include <memory>

namespace PMuscle {

struct ROMDataPoint {
    double force_magnitude;
    std::map<std::string, Eigen::VectorXd> joint_angles;
    double passive_force_total;
};

// Posture control target
struct PostureTarget {
    std::string bodyNodeName;
    Eigen::Vector3d referencePosition;
    Eigen::Array3i controlDimensions;  // X, Y, Z booleans (0 or 1)
    double kp;  // Proportional gain
    double ki;  // Integral gain
    Eigen::Vector3d integralError;  // Accumulated error for integral term
};

// Trial configuration
struct TrialConfig {
    std::string name;
    std::string description;
    std::map<std::string, Eigen::VectorXd> pose;

    // Force parameters
    std::string force_body_node;
    Eigen::Vector3d force_offset;
    Eigen::Vector3d force_direction;
    double force_min;
    double force_max;
    int force_steps;
    double settle_time;

    // Recording
    std::vector<std::string> record_joints;
    std::string output_file;
};

class PhysicalExam : public SurgeryExecutor {
public:
    PhysicalExam(int width = 1920, int height = 1080);
    ~PhysicalExam();

    // Initialization
    void initialize();
    void loadRenderConfig();
    void loadCharacter(const std::string& skel_path, const std::string& muscle_path, ActuatorType _actType);
    void createGround();

    // Pose and force control
    void applyPosePreset(const std::map<std::string, Eigen::VectorXd>& joint_angles);
    void applyForce(const std::string& body_node,
                   const Eigen::Vector3d& offset,
                   const Eigen::Vector3d& direction,
                   double magnitude);
    void applyConfinementForces(double magnitude);

    // Posture control
    void printBodyNodePositions();
    void parseAndPrintPostureConfig(const std::string& pastedData);
    void setupPostureTargets();
    void applyPostureControl();
    void drawPostureForces();
    void drawGraphPanel();

    // Joint angle sweep
    void setupSweepMuscles();
    void runSweep();
    void collectSweepData(double angle);
    void renderMusclePlots();
    void clearSweepData();

    // Simulation
    void stepSimulation(int steps);
    void setPaused(bool paused);  // Callback to control simulation pause state

    // Recording
    std::map<std::string, Eigen::VectorXd> recordJointAngles(
        const std::vector<std::string>& joint_names);
    double computePassiveForce();

    // Examination execution
    void loadExamSetting(const std::string& config_path);
    void runExamination(const std::string& config_path);  // Deprecated - loads and runs all trials
    void startNextTrial();
    void runCurrentTrial();
    void saveToCSV(const std::string& output_path);

    // Rendering
    void render();
    void mainLoop();
    void reset();  // Reset camera and scene

    // UI
    void drawControlPanel();  // Left panel - force controls
    void drawVisualizationPanel();  // Right panel - plots and data
    void drawSurgeryPanel();  // Surgery panel - toggleable with G key

    // Control Panel Sections
    void drawPosePresetsSection();
    void drawForceApplicationSection();
    void drawPrintInfoSection();
    void drawRecordingSection();
    void drawRenderOptionsSection();
    void drawJointControlSection();
    void drawJointAngleSweepSection();

    // Visualization Panel Sections
    void drawTrialManagementSection();
    void drawCurrentStateSection();
    void drawRecordedDataSection();
    void drawROMAnalysisSection();
    void drawCameraStatusSection();
    void drawSweepMusclePlotsSection();
    void drawMuscleInfoSection();

    // Surgery Panel Sections
    void drawDistributePassiveForceSection();
    void drawRelaxPassiveForceSection();
    void drawSaveMuscleConfigSection();
    void drawAnchorManipulationSection();
    
    // Surgery operations with GUI-specific logic (override base class)
    bool removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) override;
    bool copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex, const std::string& toMuscle) override;
    bool editAnchorPosition(const std::string& muscle, int anchor_index, const Eigen::Vector3d& position) override;
    bool editAnchorWeights(const std::string& muscle, int anchor_index, const std::vector<double>& weights) override;
    bool addBodyNodeToAnchor(const std::string& muscle, int anchor_index, const std::string& bodynode_name, double weight) override;
    bool removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index, int bodynode_index) override;
    
    // Surgery script recording and execution
    void startRecording();
    void stopRecording();
    void exportRecording(const std::string& filepath);
    void recordOperation(std::unique_ptr<SurgeryOperation> op);
    void loadSurgeryScript();
    void executeSurgeryScript(std::vector<std::unique_ptr<SurgeryOperation>>& ops);
    void showScriptPreview();

    void drawSimFrame();
    void drawGround();
    void drawSkeleton(const dart::dynamics::SkeletonPtr& skel);
    void drawSingleBodyNode(const dart::dynamics::BodyNode* bn, const Eigen::Vector4d& color);
    void drawShape(const dart::dynamics::Shape* shape, const Eigen::Vector4d& color);
    void drawMuscles();
    void drawForceArrow();
    void drawJointPassiveForces();
    void drawConfinementForces();
    void drawSelectedAnchors();

    // Camera control
    void setCamera();
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset);
    void keyboardPress(int key, int scancode, int action, int mods);
    void windowResize(int width, int height);
    
    // Camera presets
    void printCameraInfo();
    void saveCameraPreset(int index);
    void loadCameraPreset(int index);
    void initializeCameraPresets();

private:
    // DART simulation
    dart::simulation::WorldPtr mWorld;
    dart::dynamics::SkeletonPtr mGround;

    // Loaded file paths
    std::string mSkeletonPath;
    std::string mMusclePath;

    // GLFW/ImGui
    GLFWwindow* mWindow;
    int mWidth, mHeight;
    int mWindowXPos, mWindowYPos;
    int mControlPanelWidth;
    int mPlotPanelWidth;

    // Camera
    Eigen::Vector3d mEye;
    Eigen::Vector3d mUp;
    Eigen::Vector3d mTrans;
    double mZoom;
    dart::gui::Trackball mTrackball;
    bool mMouseDown;
    bool mRotate;
    bool mTranslate;
    double mMouseX, mMouseY;
    
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
    int mCurrentCameraPreset;  // Track which preset is currently active

    // Force state
    std::string mForceBodyNode;
    Eigen::Vector3d mForceOffset;
    Eigen::Vector3d mForceDirection;
    double mForceMagnitude;

    // Interactive controls
    float mForceX, mForceY, mForceZ;
    float mOffsetX, mOffsetY, mOffsetZ;
    bool mApplyingForce;
    int mSelectedBodyNode;

    // Confinement force
    bool mApplyConfinementForce;

    // Posture control
    bool mApplyPostureControl;
    std::vector<PostureTarget> mPostureTargets;
    CBufferData<double>* mGraphData;
    Eigen::VectorXd mPostureForces;

    // Joint angle sweep system
    struct JointSweepConfig {
        int joint_index;           // Which joint to sweep
        double angle_min;          // Min angle (radians)
        double angle_max;          // Max angle (radians)
        int num_steps;             // Number of sweep steps
    };

    JointSweepConfig mSweepConfig;
    std::vector<std::string> mTrackedMuscles;  // Muscles crossing swept joint
    std::vector<double> mSweepAngles;          // X-axis data (joint angles)
    std::map<std::string, bool> mMuscleVisibility;  // Track which muscles to plot
    char mMuscleFilterBuffer[256];              // Filter text buffer for muscle search
    bool mShowSweepLegend;                     // Toggle legend display in sweep plots

    // Muscle Info Panel
    char mMuscleInfoFilterBuffer[256];          // Filter text for muscle info search
    std::string mSelectedMuscleInfo;            // Currently selected muscle for info display

    // Sweep state (for non-blocking execution)
    bool mSweepRunning;                        // Is sweep currently active?
    int mSweepCurrentStep;                     // Current step in sweep
    Eigen::VectorXd mSweepOriginalPos;         // Original joint position for restoration

    // Examination state
    bool mRunning;
    bool mSimulationPaused;
    bool mSingleStep;  // Flag for single-step simulation mode
    std::string mCurrentExamName;
    std::vector<ROMDataPoint> mRecordedData;
    std::map<std::string, Eigen::VectorXd> mInitialPose;
    
    // Trial management
    std::string mExamName;
    std::string mExamDescription;
    std::vector<TrialConfig> mTrials;
    int mCurrentTrialIndex;
    bool mTrialRunning;
    int mCurrentForceStep;
    bool mExamSettingLoaded;

    // Pose presets
    int mCurrentPosePreset;  // 0=standing, 1=supine, 2=prone, 3=supine_knee_flexed
    float mPresetKneeAngle;  // For supine with knee flexion

    // Joint angle PI controller
    bool mEnableInterpolation;  // Enable/disable PI controller mode
    std::vector<std::optional<double>> mMarkedJointTargets;  // Marked joints with target angles (nullopt = unmarked)
    std::vector<double> mJointIntegralError;  // Integral error for each joint (PI controller)
    double mJointKp;  // Proportional gain for joint PI controller
    double mJointKi;  // Integral gain for joint PI controller
    double mInterpolationThreshold;  // Threshold to consider target reached (radians)

    // Examination table
    dart::dynamics::SkeletonPtr mExamTable;

    // Simulation parameters
    int mSimulationHz;
    int mControlHz;

    // Rendering
    ShapeRenderer mShapeRenderer;
    float mPassiveForceNormalizer;  // Normalization factor for passive force visualization
    float mMuscleTransparency;       // Transparency for muscle rendering
    bool mShowJointPassiveForces;   // Toggle for joint passive force arrows
    float mJointForceScale;          // Scale factor for joint force arrow visualization
    bool mShowJointForceLabels;      // Toggle for joint passive force text labels
    bool mShowPostureDebug;          // Toggle for posture control debug output
    bool mShowExamTable;             // Toggle for examination table visibility

    // Muscle Selection UI
    char mMuscleFilterText[32];
    std::vector<bool> mMuscleSelectionStates;

    // Surgery Panel
    bool mShowSurgeryPanel;          // Toggle for surgery panel visibility
    char mSaveMuscleFilename[64];   // Buffer for save muscle config filename
    bool mSavingMuscle;              // Flag to prevent duplicate saves

    // Distribute Passive Force section
    std::string mDistributeRefMuscle;                // Reference muscle name
    std::map<std::string, bool> mDistributeSelection; // Selected modifying muscles
    char mDistributeFilterBuffer[32];               // Filter text for muscle search

    // Relax Passive Force section
    std::map<std::string, bool> mRelaxSelection;     // Selected muscles to relax
    char mRelaxFilterBuffer[32];                    // Filter text for muscle search

    // Anchor Manipulation section
    char mAnchorCandidateFilterBuffer[32];          // Filter text for candidate muscle search
    char mAnchorReferenceFilterBuffer[32];          // Filter text for reference muscle search
    std::string mAnchorCandidateMuscle;              // Selected candidate muscle name
    std::string mAnchorReferenceMuscle;              // Selected reference muscle name
    int mSelectedCandidateAnchorIndex;               // Selected candidate anchor index for operations
    int mSelectedReferenceAnchorIndex;               // Selected reference anchor index for copying

    // Sweep restore option
    bool mSweepRestorePosition;                      // Whether to restore position after sweep

    // Surgery script recording and execution
    bool mRecordingSurgery;                          // Is recording active?
    std::vector<std::unique_ptr<SurgeryOperation>> mRecordedOperations;  // Recorded operations
    std::string mRecordingScriptPath;                // Default path for saving recording
    std::string mLoadScriptPath;                     // Path for loading script
    char mRecordingPathBuffer[256];                  // Buffer for recording path input
    char mLoadPathBuffer[256];                       // Buffer for load path input
    std::vector<std::unique_ptr<SurgeryOperation>> mLoadedScript;  // Loaded script for preview
    bool mShowScriptPreview;                         // Show script preview popup

    // Pose preset methods
    void setPoseStanding();
    void setPoseSupine();
    void setPoseProne();
    void setPoseSupineKneeFlexed(double knee_angle);
};

} // namespace PMuscle

#endif // __PHYSICAL_EXAM_H__

#ifndef __PHYSICAL_EXAM_H__
#define __PHYSICAL_EXAM_H__

#include <vector>
#include <map>
#include <set>
#include <string>
#include <optional>
#include <memory>
#include <chrono>
#include <H5Cpp.h>
#include <implot.h>
#include "dart/dart.hpp"
#include "Character.h"
#include "CBufferData.h"
#include "SurgeryOperation.h"
#include "SurgeryExecutor.h"
#include "SurgeryPanel.h"
#include "common/PIDImGui.h"
#include "common/ViewerAppBase.h"

namespace PMuscle {

// RenderMode is inherited from ViewerAppBase (defined in GLfunctions.h)

struct ROMDataPoint {
    double force_magnitude;
    std::map<std::string, Eigen::VectorXd> joint_angles;
    double passive_force_total;
};

// Trial mode (force sweep vs angle sweep)
enum class TrialMode { FORCE_SWEEP, ANGLE_SWEEP };

// Angle sweep trial configuration (kinematic-only sweep)
struct AngleSweepTrialConfig {
    std::string joint_name;      // e.g., "TibiaR"
    int dof_index;               // Which DOF (0 for single-DOF joints)
    double angle_min;            // Radians
    double angle_max;            // Radians
    int num_steps;               // Number of sweep steps

    // Composite DOF fields (e.g., abd_knee)
    std::string dof_type;        // "abd_knee" for composite DOF (empty = simple DOF)
    double shank_scale = 0.7;    // Scale factor for abd_knee IK
    double angle_step = 1.0;     // Step size in degrees (for composite DOF sweep)

    // Hierarchical alias for ROM summary (e.g., "hip/abduction_knee0/left")
    std::string alias;
    bool neg = false;  // Negate ROM value for display (e.g., dorsiflexion stored as negative)

    // Clinical data reference (from ROM config clinical_data section)
    std::string cd_side;    // "left" or "right"
    std::string cd_joint;   // "hip", "knee", "ankle"
    std::string cd_field;   // field name in rom.yaml (e.g., "abduction_ext_r2")
    bool cd_neg = false;    // Negate clinical value for comparison
};

// Angle sweep data point (per-step recording)
struct AngleSweepDataPoint {
    double joint_angle;                                    // Current sweep angle (radians)
    double passive_torque_total;                           // Sum of passive torques at swept joint
    double passive_torque_stiffness;                       // Passive stiffness (dtau/dtheta, Nm/rad)
    std::map<std::string, double> muscle_fp;               // Per-muscle passive force
    std::map<std::string, double> muscle_lm_norm;          // Per-muscle normalized length
    std::map<std::string, std::vector<double>> muscle_jtp; // Per-muscle joint torques (all related DOFs)
    std::map<std::string, double> muscle_jtp_dof;          // Per-muscle jtp at swept DOF only
};

// ROM (Range of Motion) metrics computed from angle sweep data
struct ROMMetrics {
    double rom_deg;                    // Functional ROM (degrees) - range before hitting threshold
    double rom_min_angle;              // Start angle of ROM (degrees)
    double rom_max_angle;              // End angle of ROM (degrees)
    double peak_stiffness;             // Peak absolute stiffness (Nm/rad)
    double peak_torque;                // Peak absolute torque (Nm)
    double angle_at_peak_stiffness;    // Angle where peak stiffness occurs (degrees)
    double angle_at_peak_torque;       // Angle where peak torque occurs (degrees)
};

// ROM threshold configuration
struct ROMThresholds {
    float max_stiffness = 100.0f;      // Maximum acceptable stiffness
    float max_torque = 50.0f;          // Maximum acceptable torque
};

// X-axis display mode for muscle plots
enum class XAxisMode { RAW_ANGLE, NORMALIZED };

// ROM metric selection - which metric determines ROM limit
enum class ROMMetric { STIFFNESS, TORQUE, EITHER, BOTH };

// Character data source for skeleton/muscle loading
using CharacterDataSource = PIDNav::CharacterDataSource;

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

    // Trial mode selection (default: FORCE_SWEEP for backward compatibility)
    TrialMode mode = TrialMode::FORCE_SWEEP;

    // Force sweep parameters (used when mode == FORCE_SWEEP)
    std::string force_body_node;
    Eigen::Vector3d force_offset;
    Eigen::Vector3d force_direction;
    double force_min;
    double force_max;
    int force_steps;
    double settle_time;

    // Angle sweep parameters (used when mode == ANGLE_SWEEP)
    AngleSweepTrialConfig angle_sweep;
    double torque_cutoff = 15.0;  // Cutoff torque for ROM limit (Nm)

    // Recording
    std::vector<std::string> record_joints;  // Used by force sweep
    std::string output_file;
};;

// Trial file info (cached from directory scan)
struct TrialFileInfo {
    std::string file_path;  // Full resolved path
    std::string name;       // Trial name (from YAML "name" field, used for display)
};

// Buffered trial data (for multi-trial management)
struct TrialDataBuffer {
    std::string trial_name;
    std::string trial_description;
    std::string alias;  // Hierarchical path for ROM summary (e.g., "hip/abduction_knee0/left")
    bool neg = false;   // Negate ROM value for display
    std::chrono::system_clock::time_point timestamp;
    std::vector<AngleSweepDataPoint> angle_sweep_data;
    std::vector<AngleSweepDataPoint> std_angle_sweep_data;
    std::vector<std::string> tracked_muscles;
    std::vector<std::string> std_tracked_muscles;
    AngleSweepTrialConfig config;
    double torque_cutoff = 15.0;  // Cutoff torque for ROM limit (Nm)
    std::vector<double> cutoff_angles;  // Angles where torque crosses cutoff (degrees)
    ROMMetrics rom_metrics;
    ROMMetrics std_rom_metrics;
    Eigen::VectorXd base_pose;  // Full skeleton pose for character positioning
    Eigen::VectorXd normative_pose;  // Full skeleton pose at normative angle

    // Clinical data reference (copied from AngleSweepTrialConfig for table rendering)
    std::string cd_side;    // "left" or "right"
    std::string cd_joint;   // "hip", "knee", "ankle"
    std::string cd_field;   // field name in rom.yaml
    bool cd_neg = false;    // Negate clinical value for comparison
};

class PhysicalExam : public ViewerAppBase, public SurgeryExecutor {
public:
    PhysicalExam(int width = 1920, int height = 1080);
    ~PhysicalExam();

    // ViewerAppBase virtual overrides
    void onInitialize() override;
    void onFrameStart() override;
    void drawContent() override;
    void drawUI() override;
    void keyPress(int key, int scancode, int action, int mods) override;

    // Initialization
    void loadRenderConfigImpl() override;
    void loadCharacter(const std::string& skel_path, const std::string& muscle_path);
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
    void renderMusclePlots();
    void clearSweepData();

    // Simulation
    void stepSimulation(int steps);
    void setPaused(bool paused);  // Callback to control simulation pause state

    // Recording
    std::map<std::string, Eigen::VectorXd> recordJointAngles(
        const std::vector<std::string>& joint_names);
    double getPassiveTorqueJoint(int joint_idx);  // Sum of passive torques at joint (all DOFs)
    double getPassiveTorqueJointGlobalY(Character* character, dart::dynamics::Joint* joint);  // Cross-product torque projected onto global Y
    double getPassiveTorqueJoint_forCharacter(Character* character, dart::dynamics::Joint* joint);  // Helper for specific character
    double getPassiveTorqueJointDof(Character* character, dart::dynamics::Joint* joint, int dof_index);  // Torque at specific DOF only

    // Pose synchronization between main and standard characters
    void setCharacterPose(const Eigen::VectorXd& positions);  // Sets positions for both main and std characters
    void setCharacterPose(const std::string& joint_name, const Eigen::VectorXd& positions);  // Sets joint positions for both characters

    // Examination execution
    void setOutputDir(const std::string& output_dir);
    void loadExamSetting(const std::string& config_path);
    void runExamination(const std::string& config_path);  // Deprecated - loads and runs all trials
    void startNextTrial();
    void runCurrentTrial();
    void saveToCSV(const std::string& output_path);
    
    // Trial file management
    void scanTrialFiles();
    TrialConfig parseTrialConfig(const YAML::Node& trial_node);
    void loadAndRunTrial(const std::string& trial_file_path);

    // Multi-trial buffer management
    void runSelectedTrials();                         // Run all selected trials
    void addTrialToBuffer(const TrialDataBuffer& buffer);  // Add trial to buffer with limit enforcement
    void loadBufferForVisualization(int buffer_index);     // Load buffer data for plotting
    void removeTrialBuffer(int buffer_index);              // Remove specific buffer
    void clearTrialBuffers();                              // Clear all buffers

    // Angle sweep trial execution (kinematic-only)
    void runAngleSweepTrial(const TrialConfig& trial);
    void collectAngleSweepData(double angle, int joint_index, bool use_global_y = false);
    void setupTrackedMusclesForAngleSweep(const std::string& joint_name);

    // ROM analysis helpers
    ROMMetrics computeROMMetrics(const std::vector<AngleSweepDataPoint>& data) const;
    std::vector<double> computeCutoffAngles(const std::vector<AngleSweepDataPoint>& data,
                                            double torque_cutoff) const;
    std::vector<double> normalizeXAxis(const std::vector<double>& x_data_deg,
                                       double rom_min_deg, double rom_max_deg) const;

    // HDF5 exam export (all trials in single file)
    std::string extractPidFromPath(const std::string& path) const;
    void initExamHDF5();
    void appendTrialToHDF5(const TrialConfig& trial);
    void exportTrialBuffersToHDF5();
    void writeAngleSweepData(H5::Group& group, const TrialConfig& trial);
    void writeAngleSweepDataForCharacter(
        H5::Group& group, 
        const TrialConfig& trial,
        const std::vector<AngleSweepDataPoint>& data,
        const std::vector<std::string>& tracked_muscles);
    void writeForceSweepData(H5::Group& group, const TrialConfig& trial);
    void runAllTrials();

    // CLI trial mode - runs trials and prints muscle comparison table
    int runTrialsCLI(const std::vector<std::string>& trial_paths,
                     bool verbose = false,
                     double torque_threshold = 0.01,
                     double length_threshold = 0.001,
                     const std::string& sort_by = "");

    // Rendering
    void reset();  // Reset camera and scene
    void resetSkeleton();  // Reset skeleton by reloading from XML

    // UI
    void drawLeftPanel();  // Left panel - force controls
    void drawRightPanel();  // Right panel - plots and data
    void drawSurgeryTabContent();  // Surgery content - now a tab in left panel

    // Control Panel Sections
    void drawClinicalDataSection();
    void drawCharacterLoadSection();
    void drawPosePresetsSection();
    void drawForceApplicationSection();
    void drawPrintInfoSection();
    void drawRecordingSection();
    void drawRenderOptionsSection();
    void drawJointControlSection();
    void drawJointAngleSweepSection();

    // Character loading helpers
    void scanSkeletonFilesForBrowse();
    void scanMuscleFilesForBrowse();
    void reloadCharacterFromBrowse();

    // Visualization Panel Sections
    void drawTrialManagementSection();
    void drawCurrentStateSection();
    void drawCameraStatusSection();
    void drawMuscleInfoSection();
    
    // Right Panel Tab Contents
    void drawBasicTabContent();
    void drawSweepTabContent();
    void drawEtcTabContent();
    void drawROMSummaryTable();
    void drawSkeleton();
    void drawMuscles();
    void drawForceArrow();
    void drawJointPassiveForces();
    void drawConfinementForces();
    void drawSelectedAnchors();

    void drawReferenceAnchor();

    // Camera control (use base class setCamera, mouse methods)
    void windowResize(int width, int height);
    
    // Camera presets
    void printCameraInfo();
    void saveCameraPreset(int index);
    void loadCameraPreset(int index);

    void selectCameraPresetInteractive();  // Interactive preset selection via stdin
    void initializeCameraPresets();

    // Utility
    std::string characterConfig() const;  // Returns "pid:XXX skeleton | muscle" string

private:
    // DART simulation
    dart::simulation::WorldPtr mWorld;
    dart::dynamics::SkeletonPtr mGround;

    // Loaded file paths
    std::string mSkeletonPath;
    std::string mMusclePath;

    // Standard character for comparison
    Character* mStdCharacter;
    std::string mStdSkeletonPath;
    std::string mStdMusclePath;

    // Standard character sweep data
    std::vector<AngleSweepDataPoint> mStdAngleSweepData;
    std::vector<std::string> mStdAngleSweepTrackedMuscles;

    // Rendering control
    bool mRenderMainCharacter;
    bool mRenderStdCharacter;
    bool mShowStdCharacterInPlots;  // Toggle for plot overlay
    bool mPlotWhiteBackground;  // Toggle for white plot background
    bool mShowTrialNameInPlots;  // Toggle for showing trial name in plot titles
    bool mShowCharacterInTitles;  // Toggle for showing character info in plot titles
    std::string mCurrentSweepName;  // Current sweep name (trial name or "GUI Sweep")

    // Camera state (mCamera inherited from ViewerAppBase)
    bool mCameraMoving = false;  // True while camera is being manipulated

    // Camera presets (uses inherited CameraState)
    struct CameraPreset {
        std::string description;
        CameraState state;  // Full camera state including trackball
        bool isSet = false;
    };
    CameraPreset mCameraPresets[10];
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
        int dof_index;             // Which DOF of the joint to sweep (0-based within joint)
        double angle_min;          // Min angle (radians)
        double angle_max;          // Max angle (radians)
        int num_steps;             // Number of sweep steps
    };

    JointSweepConfig mSweepConfig;
    std::map<std::string, bool> mMuscleVisibility;  // Track which muscles to plot
    char mMuscleFilterBuffer[256];              // Filter text buffer for muscle search

    // Muscle Info Panel
    char mMuscleInfoFilterBuffer[256];          // Filter text for muscle info search
    std::string mSelectedMuscleInfo;            // Currently selected muscle for info display

    // Sweep state (for non-blocking execution)
    bool mSweepRunning;                        // Is sweep currently active?
    int mSweepCurrentStep;                     // Current step in sweep
    Eigen::VectorXd mSweepOriginalPos;         // Original joint position for restoration

    // Joint torque plotting
    int mSelectedPlotJointIndex;               // For joint selection in torque plot (default: sweep joint)

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
    
    // Trial file scanning
    std::vector<TrialFileInfo> mAvailableTrialFiles;
    int mSelectedTrialFileIndex;
    std::vector<bool> mTrialMultiSelectStates;  // Multi-select states for trial files

    // Multi-trial buffer management
    std::vector<TrialDataBuffer> mTrialBuffers;  // All buffered trial data
    int mSelectedBufferIndex = -1;               // Currently selected buffer for plotting
    int mMaxTrialBuffers = 30;                   // Maximum buffered trials (memory management)
    bool mAutoSelectNewBuffer = true;            // Auto-select newly run trial for viewing

    // Normative ROM values (loaded from render.yaml)
    // Key: "joint/measurement" (e.g., "hip/abduction_knee0"), Value: degrees
    std::map<std::string, double> mNormativeROM;

    // Angle sweep trial data
    std::vector<AngleSweepDataPoint> mAngleSweepData;
    std::vector<std::string> mAngleSweepTrackedMuscles;
    int mAngleSweepJointIdx;  // Joint index for current angle sweep

    // ROM analysis state
    ROMThresholds mROMThresholds;              // User-configurable thresholds
    XAxisMode mXAxisMode = XAxisMode::RAW_ANGLE;  // X-axis normalization mode
    ROMMetric mROMMetric = ROMMetric::EITHER;     // Which metric determines ROM

    // HDF5 exam export
    std::string mOutputDir;        // Output directory (from command line)
    std::string mExamOutputPath;   // HDF5 output path for entire exam
    std::string mExamConfigPath;   // Original config path for naming

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

    // Rendering (mShapeRenderer inherited from ViewerAppBase)
    // Muscle color mode: 0=Passive Force, 1=Normalized Length
    int mMuscleColorMode = 1;
    float mPassiveForceNormalizer;  // Normalization factor for passive force visualization
    float mMuscleTransparency;       // Transparency for muscle rendering
    float mLmNormMin = 0.7f;         // Min lm_norm for viridis color scale
    float mLmNormMax = 1.3f;         // Max lm_norm for viridis color scale
    float mMuscleLineWidth = 5.0f;   // Muscle line thickness
    bool mShowJointPassiveForces;   // Toggle for joint passive force arrows
    float mJointForceScale;          // Scale factor for joint force arrow visualization
    bool mShowJointForceLabels;      // Toggle for joint passive force text labels
    int mTopPassiveForcesCount;      // Number of top passive forces to display in UI
    bool mShowPostureDebug;          // Toggle for posture control debug output
    bool mVerboseTorque;             // Verbose torque debug output during sweep
    bool mVisualizeSweep;            // Step-by-step sweep with N key to advance
    bool mSweepNextPressed;          // Flag for N key during visualize sweep
    bool mSweepQuitPressed;          // Flag for Q key during visualize sweep
    bool mShowExamTable;             // Toggle for examination table visibility
    bool mShowAnchorPoints;          // Toggle for anchor point visualization
    // mRenderMode inherited from ViewerAppBase

    // Muscle Selection UI
    char mMuscleFilterText[32];
    std::vector<bool> mMuscleSelectionStates;

    // Sweep restore option
    bool mSweepRestorePosition;                      // Whether to restore position after sweep

    // Pose preset methods
    void setPoseStanding();
    void setPoseSupine();
    void setPoseProne();
    void setPoseSupineKneeFlexed(double knee_angle);

    // UI helpers
    // mDefaultOpenPanels and isPanelDefaultOpen() inherited from ViewerAppBase
    bool collapsingHeaderWithControls(const std::string& title);

    // ============================================================
    // Surgery Panel (embedded surgery UI)
    // ============================================================
    std::unique_ptr<SurgeryPanel> mSurgeryPanel;

    // ============================================================
    // PID Navigator and Character Loading (Browse & Rebuild)
    // ============================================================
    std::unique_ptr<PIDNav::PIDNavigator> mPIDNavigator;

    // Independent source selection for skeleton and muscle
    CharacterDataSource mBrowseSkeletonDataSource = CharacterDataSource::DefaultData;
    CharacterDataSource mBrowseMuscleDataSource = CharacterDataSource::DefaultData;
    std::string mBrowseCharacterPID;    // PID from navigator (shared)

    std::vector<std::string> mBrowseSkeletonCandidates;
    std::vector<std::string> mBrowseMuscleCandidates;
    std::string mBrowseSkeletonPath;    // Selected skeleton path
    std::string mBrowseMusclePath;      // Selected muscle path

    // Clinical weight data
    float mClinicalWeight = 0.0f;       // kg (from clinical data)
    bool mClinicalWeightAvailable = false;

    // Clinical ROM data (loaded from @pid:{pid}/{visit}/rom.yaml)
    // Key format: "side.joint.field" (e.g., "left.hip.abduction_ext_r2")
    int mROMColorCompare = 0;  // 0 = compare to CD, 1 = compare to Norm
    std::map<std::string, std::optional<float>> mClinicalROM;
    std::string mClinicalROMPID;    // Track loaded PID for cache invalidation
    std::string mClinicalROMVisit;  // Track loaded visit for cache invalidation

    // Helper function to load clinical ROM data
    void loadClinicalROM(const std::string& pid, const std::string& visit);
    void onPIDChanged(const std::string& pid);  // Callback for PID selection changes
    void onVisitChanged(const std::string& pid, const std::string& visit);  // Callback for visit changes
};

} // namespace PMuscle

#endif // __PHYSICAL_EXAM_H__

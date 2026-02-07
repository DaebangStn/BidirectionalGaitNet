#ifndef MUSCLE_PERSONALIZER_APP_H
#define MUSCLE_PERSONALIZER_APP_H

#include "common/ViewerAppBase.h"
#include "common/PIDImGui.h"
#include "RenderCharacter.h"
#include "common/ShapeRenderer.h"
#include "rm/rm.hpp"
#include "SurgeryExecutor.h"
#include "common/imgui_common.h"
#include "optimizer/ContractureOptimizer.h"
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <optional>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

/**
 * @brief Muscle Personalizer Application
 *
 * Inherits from ViewerAppBase for common window/camera/input handling.
 *
 * Features:
 * - Weight application (f0 scaling based on target body mass)
 * - Waypoint optimization (muscle path optimization from HDF motions)
 * - Contracture estimation (Ceres-based lm_contract parameter fitting from ROM data)
 */
class MusclePersonalizerApp : public ViewerAppBase
{
public:
    MusclePersonalizerApp(const std::string& configPath = "@data/config/muscle_personalizer.yaml");
    ~MusclePersonalizerApp() override;

protected:
    // ViewerAppBase overrides
    void onInitialize() override;
    void drawContent() override;
    void drawUI() override;
    void keyPress(int key, int scancode, int action, int mods) override;

private:
    // ============================================================
    // Rendering
    // ============================================================
    ShapeRenderer mShapeRenderer;
    std::string mSkeletonPath;
    std::string mMusclePath;

    // ============================================================
    // Surgery Executor (handles all muscle operations)
    // ============================================================
    std::unique_ptr<PMuscle::SurgeryExecutor> mExecutor;

    // ============================================================
    // Reference Character (standard/ideal muscle behavior)
    // ============================================================
    Character* mReferenceCharacter = nullptr;  // Standard reference character
    std::string mReferenceSkeletonPath;        // Default skeleton path (from config)
    std::string mReferenceMusclePath;          // Default muscle path (from config)

    // ============================================================
    // Configuration
    // ============================================================
    std::string mConfigPath;
    std::string mMuscleGroupsPath;  // Path to muscle groups config (from config file)
    // mDefaultOpenPanels, mControlPanelWidth, mPlotPanelWidth inherited from ViewerAppBase

    // ============================================================
    // Resource Manager and PID Navigator
    // ============================================================
    rm::ResourceManager* mResourceManager = nullptr;
    std::unique_ptr<PIDNav::PIDNavigator> mPIDNavigator;

    // ============================================================
    // Tool 1: Weight Application
    // ============================================================
    enum class WeightSource { ClinicalData, UserInput };
    WeightSource mWeightSource = WeightSource::ClinicalData;
    float mTargetMass = 50.0f;      // kg
    float mReferenceMass = 50.0f;   // kg (from character)
    float mCurrentMass = 50.0f;     // kg (current scaled mass)
    float mAppliedRatio = 1.0f;     // Last applied scaling ratio
    float mClinicalWeight = 0.0f;   // kg (from clinical data)
    bool mClinicalWeightAvailable = false;

    // ============================================================
    // Tool 2: Waypoint Optimization
    // ============================================================
    char mHDFPath[256] = "";
    std::vector<std::string> mMotionCandidates;
    std::map<std::string, int> mMotionDOFMap;  // filename -> DOF count (-1 if unknown)
    std::vector<std::string> mAvailableMuscles;
    std::vector<bool> mSelectedMuscles;
    char mMuscleFilter[64] = "";

    // Optimization parameters
    int mWaypointMaxIterations = 10;
    int mWaypointNumSampling = 50;
    float mWaypointLambdaShape = 1.0f;
    float mWaypointLambdaLengthCurve = 1.0f;
    bool mWaypointFixOriginInsertion = true;
    bool mWaypointVerbose = false;
    float mWaypointWeightPhase = 1.0f;
    float mWaypointWeightDelta = 50.0f;
    float mWaypointWeightSamples = 1.0f;
    int mWaypointNumPhaseSamples = 3;
    int mWaypointLossPower = 2;  // 2=squared, 3=cube, etc.
    int mWaypointNumParallel = 1;  // Number of parallel threads (1 = sequential)
    bool mWaypointUseNormalizedLength = false;  // false = lmt (MTU), true = lm_norm (normalized)
    float mWaypointMaxDisplacement = 0.2f;  // Max displacement for normal waypoints (m)
    float mWaypointMaxDispOriginInsertion = 0.03f;  // Max displacement for origin/insertion (m)
    float mWaypointFunctionTolerance = 1e-4f;  // Convergence tolerance on cost function
    float mWaypointGradientTolerance = 1e-5f;  // Convergence tolerance on gradient
    float mWaypointParameterTolerance = 1e-5f;  // Convergence tolerance on parameters
    bool mWaypointAdaptiveSampleWeight = false;  // Adaptive weighting for samples
    bool mWaypointMultiDofJointSweep = false;    // Sweep all DOFs of best DOF's parent joint

    // Waypoint optimization progress (thread-safe)
    std::atomic<bool> mWaypointOptRunning{false};
    std::atomic<int> mWaypointOptCurrent{0};
    std::atomic<int> mWaypointOptTotal{0};
    std::chrono::steady_clock::time_point mWaypointOptStartTime;  // Start time for elapsed/ETA
    std::mutex mWaypointOptMutex;
    std::mutex mCharacterMutex;  // Protects skeleton/muscle access during async optimization
    std::string mWaypointOptMuscleName;
    std::unique_ptr<std::thread> mWaypointOptThread;

    // Waypoint optimization results (for curve visualization)
    std::vector<PMuscle::WaypointOptResult> mWaypointOptResults;
    std::mutex mWaypointResultsMutex;
    int mWaypointResultSelectedIdx = -1;  // Single selection index
    char mWaypointResultFilter[64] = "";

    // Sort state for waypoint results table
    enum class WaypointSortColumn { Name, ShapeEnergy, LengthEnergy, TotalEnergy };
    WaypointSortColumn mWaypointSortColumn = WaypointSortColumn::TotalEnergy;
    bool mWaypointSortAscending = false;
    int mWaypointEnergyDisplayMode = 0;  // 0=Before, 1=After, 2=Diff
    std::vector<int> mWaypointSortedIndices;  // Sorted index mapping
    bool mPlotLegendEast = true;  // true=East (right), false=West (left)
    bool mPlotHideLegend = true; // Hide legend items
    int mPlotBarsPerChart = 3;    // Max number of bars per chart (for x-label readability)
    float mResultsTableHeight = 150.0f;  // Height of results table

    // ============================================================
    // Tool 3: Contracture Estimation
    // ============================================================
    std::string mROMConfigDir;

    // ROM Trial Info (parsed from TC YAML files)
    struct ROMTrialInfo {
        std::string name;
        std::string description;
        std::string filePath;
        std::string joint;      // e.g., "TibiaL"
        int dof_index = 0;      // e.g., 0
        float torque_cutoff = 15.0f;   // Cutoff torque for ROM limit (Nm)
        // Exam sweep parameters (for PhysicalExam)
        float angle_min = -90.0f;
        float angle_max = 90.0f;
        int num_steps = 100;
        // clinical_data link
        std::string cd_side;    // "left" or "right"
        std::string cd_joint;   // "hip", "knee", "ankle"
        std::string cd_field;   // field name in rom.yaml
        bool cd_neg = false;    // negate the angle
        float cd_cutoff = -1.0f; // skip if |rom_angle| > cutoff (-1 means no cutoff)
        // resolved CD value (angle in degrees)
        std::optional<float> cd_value;
        // Normative ROM value from config file (degrees)
        std::optional<float> normative;
        // Manual ROM input (when cd_value unavailable)
        float manual_rom = 0.0f;
        // selection state
        bool selected = false;
    };
    std::vector<ROMTrialInfo> mROMTrials;
    char mROMFilter[64] = "";
    std::vector<std::string> mDefaultSelectedROMTrials;  // Default trials to pre-select

    // Patient ROM data (from @pid:{pid}/{visit}/rom.yaml)
    std::map<std::string, std::optional<float>> mPatientROMValues;  // "side.joint.field" -> value
    std::string mCurrentROMPID;  // PID for which ROM is loaded

    // ROM source selection
    enum class ROMSource { Normative, Pre, Op1 };
    ROMSource mROMSource = ROMSource::Normative;

    // Optimization parameters
    int mContractureMaxIterations = 100;
    float mContractureMinRatio = 0.7f;
    float mContractureMaxRatio = 1.2f;
    bool mContractureVerbose = true;

    // Grid search initialization
    float mContractureGridBegin = 0.7f;
    float mContractureGridEnd = 1.3f;
    float mContractureGridInterval = 0.1f;

    // Grid search mapping (trial-to-groups)
    std::vector<PMuscle::GridSearchMapping> mGridSearchMapping;

    // Regularization
    float mContractureLambdaRatioReg = 0.1f;   // Penalize (ratio - 1.0)^2
    float mContractureLambdaTorqueReg = 0.01f; // Penalize passive torque magnitude

    // Outer iterations for biarticular convergence
    int mContractureOuterIterations = 3;

    // Results (muscle groups detected after running optimization)
    struct MuscleGroupResult {
        std::string group_name;
        std::vector<std::string> muscle_names;
        double ratio;
        std::vector<double> lm_contract_values;
    };
    std::vector<MuscleGroupResult> mGroupResults;

    // Comprehensive optimization results (for visualization tab)
    std::optional<PMuscle::ContractureOptResult> mContractureOptResult;

    // Tiered optimization results (dual-tier: search groups + optimization groups)
    std::optional<PMuscle::TieredContractureOptResult> mTieredContractureResult;

    // Store configs used for contracture estimation (for export metadata)
    std::vector<PMuscle::ContractureTrialInput> mContractureUsedTrials;

    // UI state for contracture results tab
    int mContractureSelectedGroupIdx = -1;
    int mContractureSelectedTrialIdx = -1;
    char mContractureGroupFilter[64] = "";
    char mContractureTrialFilter[64] = "";
    int mContractureChartPage = 0;  // Shared pagination for muscle charts
    bool mShowLmContractChart = true;
    bool mShowTorqueChart = false;
    bool mShowForceChart = false;
    bool mShowGroupTorqueChart = false;

    // Tiered UI state (for sub-tabs)
    int mContractureSubTab = 0;        // 0=Grid Search, 1=Optimization
    int mGridSearchSelectedGroup = -1;  // Selected search group in grid search sub-tab

    // ============================================================
    // Tool 4: Symmetry Analysis (UI state)
    // ============================================================
    struct SymmetryPairInfo {
        std::string base_name;      // e.g., "Adductor_Brevis"
        std::string left_name;      // e.g., "L_Adductor_Brevis"
        std::string right_name;     // e.g., "R_Adductor_Brevis"
        bool is_symmetric;          // true if all anchors within threshold
        std::vector<bool> anchor_symmetric;  // Per-anchor symmetry status
        std::vector<double> anchor_diffs;    // Max diff per anchor
    };
    std::vector<SymmetryPairInfo> mSymmetryPairs;
    int mSymmetrySelectedIdx = -1;  // Index into mSymmetryPairs
    char mSymmetryFilter[64] = "";
    float mSymmetryThreshold = 0.001f;  // Threshold for position difference (meters)

    // Highlighted muscles for 3D rendering (from symmetry tab)
    std::string mSymmetryHighlightLeft;
    std::string mSymmetryHighlightRight;

    // Skeleton symmetry analysis
    struct SkeletonSymmetryPairInfo {
        std::string base_name;      // e.g., "Femur", "Tibia"
        std::string left_name;      // e.g., "FemurL"
        std::string right_name;     // e.g., "FemurR"
        bool is_symmetric;
        // Joint DOF comparison
        Eigen::VectorXd left_dofs;
        Eigen::VectorXd right_dofs;
        double joint_diff;          // Max DOF difference
        // BodyNode position comparison (world position)
        Eigen::Vector3d left_bn_pos;   // BodyNode world position
        Eigen::Vector3d right_bn_pos;  // BodyNode world position
        double bn_pos_diff;            // BodyNode position difference (with x mirroring)
        // Joint position comparison (world position)
        Eigen::Vector3d left_joint_pos;   // Joint world position
        Eigen::Vector3d right_joint_pos;  // Joint world position
        double joint_pos_diff;            // Joint position difference (with x mirroring)
        // Relative transforms (TransformFromParentBodyNode)
        Eigen::Vector3d left_parent_to_joint;   // Translation from parent to joint
        Eigen::Vector3d right_parent_to_joint;  // Translation from parent to joint
        double parent_to_joint_diff;            // Difference (with x mirroring)
        // Relative transforms (TransformFromChildBodyNode)
        Eigen::Vector3d left_child_to_joint;    // Translation from child to joint
        Eigen::Vector3d right_child_to_joint;   // Translation from child to joint
        double child_to_joint_diff;             // Difference (with x mirroring)
    };
    std::vector<SkeletonSymmetryPairInfo> mSkeletonSymmetryPairs;
    int mSkeletonSymmetrySelectedIdx = -1;
    int mSymmetrySubTab = 0;  // 0 = Skeleton, 1 = Muscle, 2 = Parameters

    // Muscle parameter symmetry analysis
    struct MuscleParamSymmetryPairInfo {
        std::string base_name;      // e.g., "Adductor_Brevis"
        std::string left_name;      // e.g., "L_Adductor_Brevis"
        std::string right_name;     // e.g., "R_Adductor_Brevis"
        bool is_symmetric;          // true if all params within threshold
        // Parameter values
        double left_f0, right_f0;
        double left_lm_contract, right_lm_contract;
        double left_lt_rel, right_lt_rel;
        // Differences (percentage)
        double f0_diff_pct;
        double lm_contract_diff_pct;
        double lt_rel_diff_pct;
        double max_diff_pct;        // Maximum of all diffs
    };
    std::vector<MuscleParamSymmetryPairInfo> mMuscleParamSymmetryPairs;
    int mMuscleParamSymmetrySelectedIdx = -1;
    float mMuscleParamSymmetryThreshold = 0.1f;  // 0.1% threshold
    char mMuscleParamSymmetryFilter[64] = "";

    // ============================================================
    // Rendering Flags
    // ============================================================
    bool mRenderMuscles = true;
    bool mShowAnchorPoints = false;  // Show anchor points instead of muscle cylinders
    bool mRenderReferenceCharacter = false;  // Toggle between subject and reference
    bool mShowTitlePanel = false;  // Toggle title panel (T key)

    // Muscle color mode
    enum class MuscleColorMode { Contracture, ContractureExt, Symmetry, MuscleLength };
    MuscleColorMode mMuscleColorMode = MuscleColorMode::Contracture;
    float mMuscleLengthMinValue = 0.5f;
    float mMuscleLengthMaxValue = 1.5f;

    // Visualization colormap settings (from config)
    float mContractureMinValue = 0.7f;
    float mContractureMaxValue = 1.2f;
    float mMuscleLineWidth = 5.0f;
    float mMuscleLabelFontSize = 14.0f;
    float mPlotHeight = 400.0f;  // Height of length curve plots
    int mErrorPlotYLimMode = 0;  // 0: near best, 1: overview, 2: draggable

    // Muscle selection
    std::vector<bool> mMuscleSelectionStates;
    char mMuscleFilterText[256] = "";

    // ============================================================
    // Character File Browser
    // ============================================================
    enum class CharacterDataSource { DefaultData, PatientData };

    // Independent source selection for skeleton and muscle
    CharacterDataSource mSkeletonDataSource = CharacterDataSource::DefaultData;
    CharacterDataSource mMuscleDataSource = CharacterDataSource::DefaultData;
    bool mSkeletonPreOp = true;   // true = pre, false = post
    bool mMusclePreOp = true;     // true = pre, false = post
    std::string mCharacterPID;    // PID from navigator (shared)

    std::vector<std::string> mSkeletonCandidates;
    std::vector<std::string> mMuscleCandidates;

    // ============================================================
    // Export Config
    // ============================================================
    char mExportMuscleName[64] = "base_rom";

    // ============================================================
    // Initialization
    // ============================================================
    void loadRenderConfigImpl() override;
    void loadCharacter();
    void loadReferenceCharacter();  // Load standard reference character for waypoint optimization

    // ============================================================
    // Rendering
    // ============================================================
    void drawSkeleton();
    void drawMuscles();

    // ============================================================
    // UI Panels
    // ============================================================
    void drawLeftPanel();
    void drawClinicalDataSection();
    void drawCharacterLoadSection();
    void drawWeightApplicationSection();
    void drawJointAngleSection();
    void drawWaypointOptimizationSection();
    void drawContractureEstimationSection();
    void drawRenderTab();

    void drawTitlePanel();
    void drawRightPanel();
    void drawExportSection();
    void drawWaypointCurvesTab();
    void drawContractureResultsTab();
    void drawGridSearchSubTab();      // Grid search results sub-tab (tiered)
    void drawOptimizationSubTab();    // Optimization results sub-tab (tiered)
    void drawLegacyContractureView(); // Legacy (non-tiered) contracture view
    void drawSymmetryTab();           // Right panel: status/analysis display
    void drawSkeletonSymmetrySubTab();  // Skeleton symmetry sub-tab
    void drawMuscleSymmetrySubTab();    // Muscle symmetry sub-tab
    void drawMuscleParamsSymmetrySubTab();  // Muscle parameter symmetry sub-tab
    void drawSymmetryOperationsSection();  // Left panel: mirror operations
    void computeSymmetryPairs();  // Rebuild mSymmetryPairs from current muscles
    void computeSkeletonSymmetryPairs();  // Rebuild mSkeletonSymmetryPairs
    void computeMuscleParamsSymmetryPairs();  // Rebuild mMuscleParamSymmetryPairs

    // ============================================================
    // Tool Operations (delegate to SurgeryExecutor)
    // ============================================================
    void applyWeightScaling();
    void runWaypointOptimization();
    void runWaypointOptimizationAsync();  // Background thread wrapper
    void runContractureEstimation();

    // Progress overlay for async operations
    void drawProgressOverlay();

    // Helper methods
    void scanROMConfigs();
    void applyDefaultROMValues();
    void loadPatientROM(const std::string& pid, const std::string& visit);
    void loadClinicalWeight(const std::string& pid, const std::string& visit);
    void onPIDChanged(const std::string& pid);  // Callback for PID selection changes
    void onVisitChanged(const std::string& pid, const std::string& visit);  // Callback for visit changes
    void updateROMTrialCDValues();
    std::optional<float> getEffectiveROMValue(const ROMTrialInfo& trial) const;  // Returns normative or cd_value based on mROMSource
    void refreshMuscleList();
    void scanSkeletonFiles();  // Scans based on mCharacterDataSource
    void scanMuscleFiles();    // Scans based on mCharacterDataSource
    void scanMotionFiles();
    void rescanCharacterFiles();  // Re-scan skeleton/muscle based on current source

    std::string characterConfig() const;  // Title string: pid + skeleton + muscle stems

    // Helper for collapsing header
    bool collapsingHeaderWithControls(const std::string& title);
    // isPanelDefaultOpen() inherited from ViewerAppBase
};

#endif // MUSCLE_PERSONALIZER_APP_H

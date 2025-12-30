#ifndef MUSCLE_PERSONALIZER_APP_H
#define MUSCLE_PERSONALIZER_APP_H

#include "common/ViewerAppBase.h"
#include "common/PIDNavigator.h"
#include "RenderCharacter.h"
#include "ShapeRenderer.h"
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
    // mDefaultOpenPanels and mControlPanelWidth inherited from ViewerAppBase
    int mResultsPanelWidth;

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
    bool mWaypointAnalyticalGradient = true;
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
    bool mWaypointShowAfterEnergy = false;  // true=After, false=Before
    std::vector<int> mWaypointSortedIndices;  // Sorted index mapping
    bool mPlotLegendEast = true;  // true=East (right), false=West (left)
    int mPlotBarsPerChart = 3;    // Max number of bars per chart (for x-label readability)

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
        float torque = 15.0f;   // Nm at ROM limit
        // clinical_data link
        std::string cd_side;    // "left" or "right"
        std::string cd_joint;   // "hip", "knee", "ankle"
        std::string cd_field;   // field name in rom.yaml
        // resolved CD value (angle in degrees)
        std::optional<float> cd_value;
        // Manual ROM input (when cd_value unavailable)
        float manual_rom = 0.0f;
        // selection state
        bool selected = false;
    };
    std::vector<ROMTrialInfo> mROMTrials;
    char mROMFilter[64] = "";

    // Patient ROM data (from @pid:{pid}/rom.yaml)
    std::map<std::string, std::optional<float>> mPatientROMValues;  // "side.joint.field" -> value
    std::string mCurrentROMPID;  // PID for which ROM is loaded
    bool mCurrentROMPreOp = true;

    // Optimization parameters
    int mContractureMaxIterations = 100;
    float mContractureMinRatio = 0.7f;
    float mContractureMaxRatio = 1.2f;
    bool mContractureVerbose = false;

    // Grid search initialization
    float mContractureGridBegin = 0.7f;
    float mContractureGridEnd = 1.3f;
    float mContractureGridInterval = 0.1f;

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

    // UI state for contracture results tab
    int mContractureSelectedGroupIdx = -1;
    int mContractureSelectedTrialIdx = -1;
    char mContractureGroupFilter[64] = "";
    char mContractureTrialFilter[64] = "";
    int mContractureChartPage = 0;  // Shared pagination for muscle charts

    // ============================================================
    // Rendering Flags
    // ============================================================
    bool mRenderMuscles = true;
    bool mColorByContracture = false;
    bool mRenderReferenceCharacter = false;  // Toggle between subject and reference
    float mMuscleLabelFontSize = 14.0f;
    float mPlotHeight = 400.0f;  // Height of length curve plots

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
    char mExportSkeletonName[64] = "personalized";
    char mExportMuscleName[64] = "personalized";

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
    void drawWaypointOptimizationSection();
    void drawContractureEstimationSection();
    void drawRenderTab();

    void drawRightPanel();
    void drawResultsSection();
    void drawWaypointCurvesTab();
    void drawContractureResultsTab();

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
    void loadPatientROM(const std::string& pid, bool preOp);
    void loadClinicalWeight(const std::string& pid, bool preOp);
    void updateROMTrialCDValues();
    void refreshMuscleList();
    void exportMuscleConfig();
    void scanSkeletonFiles();  // Scans based on mCharacterDataSource
    void scanMuscleFiles();    // Scans based on mCharacterDataSource
    void scanMotionFiles();
    void rescanCharacterFiles();  // Re-scan skeleton/muscle based on current source

    // Helper for collapsing header
    bool collapsingHeaderWithControls(const std::string& title);
    // isPanelDefaultOpen() inherited from ViewerAppBase
};

#endif // MUSCLE_PERSONALIZER_APP_H

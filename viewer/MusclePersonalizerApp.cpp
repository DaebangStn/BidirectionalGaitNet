#include "MusclePersonalizerApp.h"
#include "Character.h"
#include "ContractureOptimizer.h"
#include "optimizer/WaypointOptimizer.h"
#include "Log.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <implot.h>
#include <tinycolormap.hpp>
#include <ctime>

namespace fs = std::filesystem;

// ============================================================
// Constructor
// ============================================================

MusclePersonalizerApp::MusclePersonalizerApp(const std::string& configPath)
    : ViewerAppBase("Muscle Personalizer", 1920, 1080),
      mConfigPath(configPath)
{
    // Adjust camera for muscle personalizer (slightly different defaults)
    mCamera.eye = Eigen::Vector3d(0.0, 0.0, 3.0);
    mCamera.trans = Eigen::Vector3d(0.0, -0.5, 0.0);

    // Initialize export name with timestamp (mmdd_hhmm)
    std::time_t now = std::time(nullptr);
    std::tm* t = std::localtime(&now);
    char timestamp[16];
    std::strftime(timestamp, sizeof(timestamp), "%m%d_%H%M", t);
    std::snprintf(mExportMuscleName, sizeof(mExportMuscleName), "base_rom_%s", timestamp);
}


MusclePersonalizerApp::~MusclePersonalizerApp()
{
    // Wait for optimization thread to finish
    if (mWaypointOptThread && mWaypointOptThread->joinable()) {
        mWaypointOptThread->join();
    }
}

// ============================================================
// ViewerAppBase Overrides
// ============================================================

void MusclePersonalizerApp::keyPress(int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_R:
                // Reset camera
                resetCamera();
                // Reset skeleton pose to zero
                if (mExecutor && mExecutor->getCharacter()) {
                    auto skel = mExecutor->getCharacter()->getSkeleton();
                    if (skel) {
                        skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));
                    }
                }
                return;
        }
    }

    // Call base class for other keys
    ViewerAppBase::keyPress(key, scancode, action, mods);
}

void MusclePersonalizerApp::onInitialize()
{
    // Create surgery executor
    mExecutor = std::make_unique<PMuscle::SurgeryExecutor>("MusclePersonalizer");

    // Load character through surgery executor
    loadCharacter();

    // Load reference character for waypoint optimization
    loadReferenceCharacter();

    // Scan ROM configs
    scanROMConfigs();

    // Scan skeleton, muscle, and motion files for file browser
    scanSkeletonFiles();
    scanMuscleFiles();
    scanMotionFiles();

    // Initialize ResourceManager and PIDNavigator
    mResourceManager = &rm::getManager();
    if (mResourceManager) {
        // Use a simple filter that shows no files (PID selection only)
        mPIDNavigator = std::make_unique<PIDNav::PIDNavigator>(
            mResourceManager,
            nullptr  // No file filter - just for PID selection
        );
        mPIDNavigator->setPIDChangeCallback([this](const std::string& pid) {
            onPIDChanged(pid);
        });
        mPIDNavigator->setVisitChangeCallback([this](const std::string& pid, const std::string& visit) {
            onVisitChanged(pid, visit);
        });
        mPIDNavigator->scanPIDs();
    }
}

void MusclePersonalizerApp::drawContent()
{
    // Lock character access to prevent race with optimization thread
    std::lock_guard<std::mutex> lock(mCharacterMutex);

    drawSkeleton();
    drawMuscles();

    // Draw origin gizmo when camera is moving
    if (mRotate || mTranslate) {
        GUI::DrawOriginAxisGizmo(-mCamera.trans);
    }
}

void MusclePersonalizerApp::drawUI()
{
    // During optimization, only show progress overlay for faster updates
    if (mWaypointOptRunning) {
        drawProgressOverlay();
        return;
    }

    drawLeftPanel();
    drawRightPanel();

    // Draw progress overlay on top of everything (when not running, this returns early)
    drawProgressOverlay();
}

// ============================================================
// Initialization
// ============================================================

void MusclePersonalizerApp::loadRenderConfigImpl()
{
    // Common config (geometry, default_open_panels) already loaded by ViewerAppBase
    // Uses inherited mControlPanelWidth and mPlotPanelWidth from geometry.control/plot

    // Load app-specific settings from muscle_personalizer.yaml
    try {
        std::string resolved = rm::resolve(mConfigPath);
        YAML::Node config = YAML::LoadFile(resolved);

        // Default paths
        if (config["paths"]) {
            auto paths = config["paths"];
            mROMConfigDir = paths["rom_config_dir"].as<std::string>("@data/config/rom");
            mSkeletonPath = paths["skeleton_default"].as<std::string>("@data/skeleton/base.yaml");
            mMusclePath = paths["muscle_default"].as<std::string>("@data/muscle/base.yaml");
            // Reference character paths for waypoint optimization (standard/ideal muscle behavior)
            mReferenceSkeletonPath = paths["reference_skeleton"].as<std::string>("@data/skeleton/base.yaml");
            mReferenceMusclePath = paths["reference_muscle"].as<std::string>("@data/muscle/base.yaml");
            // Muscle groups config for contracture estimation
            mMuscleGroupsPath = paths["muscle_groups"].as<std::string>("@data/config/muscle_groups.yaml");
        }

        // Weight scaling defaults
        if (config["weight_scaling"]) {
            auto weight = config["weight_scaling"];
            mReferenceMass = weight["reference_mass"].as<float>(50.0f);
            mTargetMass = weight["default_target"].as<float>(50.0f);
        }

        // Waypoint optimization defaults
        if (config["waypoint_optimization"]) {
            auto wp = config["waypoint_optimization"];
            mWaypointMaxIterations = wp["max_iterations"].as<int>(100);
            mWaypointNumSampling = wp["num_sampling"].as<int>(50);
            mWaypointLambdaShape = wp["lambda_shape"].as<float>(1.0f);
            mWaypointLambdaLengthCurve = wp["lambda_length_curve"].as<float>(1.0f);
            mWaypointFixOriginInsertion = wp["fix_origin_insertion"].as<bool>(true);
            mWaypointUseNormalizedLength = wp["use_normalized_length"].as<bool>(false);
            mWaypointWeightPhase = wp["weight_phase"].as<float>(1.0f);
            mWaypointWeightDelta = wp["weight_delta"].as<float>(50.0f);
            mWaypointWeightSamples = wp["weight_samples"].as<float>(1.0f);
            mWaypointNumPhaseSamples = wp["num_phase_samples"].as<int>(3);
            mWaypointLossPower = wp["loss_power"].as<int>(2);
            // Support both correct and typo'd spelling
            if (wp["analytical_gradient"]) {
                mWaypointAnalyticalGradient = wp["analytical_gradient"].as<bool>(true);
            } else if (wp["anayltical_gradient"]) {
                mWaypointAnalyticalGradient = wp["anayltical_gradient"].as<bool>(true);
            }
            mWaypointNumParallel = wp["parallelism"].as<int>(1);
            mWaypointMaxDisplacement = wp["max_displacement"].as<float>(0.2f);
            mWaypointMaxDispOriginInsertion = wp["max_displacement_origin_insertion"].as<float>(0.03f);
            mWaypointFunctionTolerance = wp["function_tolerance"].as<float>(1e-4f);
            mWaypointGradientTolerance = wp["gradient_tolerance"].as<float>(1e-5f);
            mWaypointParameterTolerance = wp["parameter_tolerance"].as<float>(1e-5f);
            mWaypointAdaptiveSampleWeight = wp["adaptive_sample_weight"].as<bool>(false);
        }

        // Contracture estimation defaults
        if (config["contracture_estimation"]) {
            auto ce = config["contracture_estimation"];
            mContractureMaxIterations = ce["max_iterations"].as<int>(100);
            mContractureMinRatio = ce["min_ratio"].as<float>(0.7f);
            mContractureMaxRatio = ce["max_ratio"].as<float>(1.2f);
            mContractureLambdaRatioReg = ce["lambda_ratio_reg"].as<float>(0.1f);
            mContractureLambdaTorqueReg = ce["lambda_torque_reg"].as<float>(0.01f);
            mContractureOuterIterations = ce["outer_iterations"].as<int>(3);

            if (ce["grid_search"]) {
                auto gs = ce["grid_search"];
                mContractureGridBegin = gs["begin"].as<float>(0.7f);
                mContractureGridEnd = gs["end"].as<float>(1.3f);
                mContractureGridInterval = gs["interval"].as<float>(0.1f);
            }

            // Parse default ROM trials to pre-select
            mDefaultSelectedROMTrials.clear();
            if (ce["default_rom_trials"]) {
                for (const auto& trial : ce["default_rom_trials"]) {
                    mDefaultSelectedROMTrials.push_back(trial.as<std::string>());
                }
            }

            // Parse grid search mapping (trial-to-groups)
            mGridSearchMapping.clear();
            if (ce["grid_search_mapping"]) {
                for (const auto& entry : ce["grid_search_mapping"]) {
                    PMuscle::GridSearchMapping mapping;
                    if (entry["trials"]) {
                        for (const auto& t : entry["trials"]) {
                            mapping.trials.push_back(t.as<std::string>());
                        }
                    }
                    if (entry["groups"]) {
                        for (const auto& g : entry["groups"]) {
                            mapping.groups.push_back(g.as<std::string>());
                        }
                    }
                    if (!mapping.trials.empty() && !mapping.groups.empty()) {
                        mGridSearchMapping.push_back(mapping);
                    }
                }
                LOG_INFO("[MusclePersonalizer] Loaded " << mGridSearchMapping.size() << " grid search mappings");
            }
        }

        // Visualization settings
        if (config["visualization"]) {
            auto vis = config["visualization"];
            mContractureMinValue = vis["min_color_value"].as<float>(0.7f);
            mContractureMaxValue = vis["max_color_value"].as<float>(1.2f);
        }

        LOG_INFO("[MusclePersonalizer] App config loaded from " << resolved);
    }
    catch (const std::exception& e) {
        LOG_WARN("[MusclePersonalizer] Failed to load config: " << e.what());
        LOG_WARN("[MusclePersonalizer] Using default configuration");
    }
}

void MusclePersonalizerApp::loadCharacter()
{
    try {
        LOG_INFO("[MusclePersonalizer] Loading skeleton: " << mSkeletonPath);
        LOG_INFO("[MusclePersonalizer] Loading muscles: " << mMusclePath);

        // Load character through SurgeryExecutor
        mExecutor->loadCharacter(mSkeletonPath, mMusclePath);

        // Get reference mass from character
        mReferenceMass = mExecutor->getCharacter()->getSkeleton()->getMass();
        mCurrentMass = mReferenceMass;

        LOG_INFO("[MusclePersonalizer] Character loaded successfully");
        LOG_INFO("[MusclePersonalizer] Reference mass: " << mReferenceMass << " kg");

        // Refresh muscle list
        refreshMuscleList();

        // Compute symmetry pairs for the new character
        computeSymmetryPairs();
        computeSkeletonSymmetryPairs();
        computeMuscleParamsSymmetryPairs();
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Failed to load character: " << e.what());
    }
}

void MusclePersonalizerApp::loadReferenceCharacter()
{
    // Reference character is loaded from default paths and never changes
    // It represents the "ideal" muscle behavior for waypoint optimization
    try {
        std::string skelResolved = rm::resolve(mReferenceSkeletonPath);
        std::string muscleResolved = rm::resolve(mReferenceMusclePath);

        LOG_INFO("[MusclePersonalizer] Loading reference character:");
        LOG_INFO("[MusclePersonalizer]   Skeleton: " << mReferenceSkeletonPath);
        LOG_INFO("[MusclePersonalizer]   Muscles: " << mReferenceMusclePath);

        // Create reference character
        mReferenceCharacter = new Character(skelResolved, true);
        mReferenceCharacter->setMuscles(muscleResolved);

        // Zero muscle activations
        if (mReferenceCharacter->getMuscles().size() > 0) {
            mReferenceCharacter->setActivations(mReferenceCharacter->getActivations().setZero());
        }

        LOG_INFO("[MusclePersonalizer] Reference character loaded successfully");
        LOG_INFO("[MusclePersonalizer]   Reference muscles: " << mReferenceCharacter->getMuscles().size());
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Failed to load reference character: " << e.what());
        mReferenceCharacter = nullptr;
    }
}

// ============================================================
// Rendering
// ============================================================

void MusclePersonalizerApp::drawSkeleton()
{
    Character* character = nullptr;
    Eigen::Vector4d color;

    if (mRenderReferenceCharacter) {
        character = mReferenceCharacter;
        color = Eigen::Vector4d(0.5, 0.7, 1.0, 0.9);  // Blue tint for reference
    } else {
        if (mExecutor && mExecutor->getCharacter()) {
            character = mExecutor->getCharacter();
        }
        color = Eigen::Vector4d(0.8, 0.8, 0.8, 0.9);  // Gray for subject
    }

    if (!character) return;

    GUI::DrawSkeleton(character->getSkeleton(), color, mRenderMode, &mShapeRenderer);
}

void MusclePersonalizerApp::drawMuscles()
{
    if (!mRenderMuscles) return;

    // Select which character to render
    Character* character = nullptr;
    if (mRenderReferenceCharacter) {
        character = mReferenceCharacter;
    } else if (mExecutor && mExecutor->getCharacter()) {
        character = mExecutor->getCharacter();
    }
    if (!character) return;

    auto muscles = character->getMuscles();

    // Initialize selection states if needed
    if (mMuscleSelectionStates.size() != muscles.size()) {
        mMuscleSelectionStates.resize(muscles.size(), true);
    }

    // Get highlighted muscle name from waypoint results selection
    std::string highlightedMuscle;
    {
        std::lock_guard<std::mutex> lock(mWaypointResultsMutex);
        if (mWaypointResultSelectedIdx >= 0 &&
            mWaypointResultSelectedIdx < static_cast<int>(mWaypointOptResults.size())) {
            highlightedMuscle = mWaypointOptResults[mWaypointResultSelectedIdx].muscle_name;
        }
    }

    // Get highlighted muscles from symmetry tab (L/R pair)
    std::string symHighlightL = mSymmetryHighlightLeft;
    std::string symHighlightR = mSymmetryHighlightRight;

    glDisable(GL_LIGHTING);

    // Render muscles as lines
    for (size_t i = 0; i < muscles.size(); i++) {
        if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

        auto muscle = muscles[i];
        muscle->UpdateGeometry();

        // Check if this muscle is highlighted by symmetry tab
        bool isSymmetryHighlighted = (!symHighlightL.empty() &&
            (muscle->name == symHighlightL || muscle->name == symHighlightR));

        Eigen::Vector4d color;
        float lineWidth = mMuscleLineWidth;

        // Waypoint highlight (green) always takes priority
        if (!highlightedMuscle.empty() && muscle->name == highlightedMuscle) {
            color = Eigen::Vector4d(0.2, 1.0, 0.2, 1.0);  // Bright green
            lineWidth = mMuscleLineWidth + 2.0f;
        } else if (mMuscleColorMode == MuscleColorMode::Symmetry) {
            // Symmetry mode: highlight L/R pairs in blue, others in default
            if (isSymmetryHighlighted) {
                color = Eigen::Vector4d(0.2, 0.4, 1.0, 1.0);  // Blue
                lineWidth = mMuscleLineWidth + 2.0f;
            } else {
                color = Eigen::Vector4d(0.6, 0.6, 0.6, 0.5);  // Gray for non-highlighted
            }
        } else if (mMuscleColorMode == MuscleColorMode::ContractureExt) {
            // ContractureExt mode: only show extremes
            if (muscle->lmt_base > 0) {
                double ratio = muscle->lm_contract;
                if (ratio > mContractureMaxValue) {
                    color = Eigen::Vector4d(1.0, 0.3, 0.3, 1.0);  // Red: lengthened
                    lineWidth = mMuscleLineWidth + 2.0f;
                } else if (ratio < mContractureMinValue) {
                    color = Eigen::Vector4d(1.0, 0.6, 0.8, 1.0);  // Pink: contracted
                    lineWidth = mMuscleLineWidth + 2.0f;
                } else {
                    continue;  // Skip muscles in normal range
                }
            } else {
                continue;  // Skip muscles without base length
            }
        } else if (mMuscleColorMode == MuscleColorMode::MuscleLength) {
            // Muscle Length mode: viridis colormap based on lm_norm
            double lm_norm = muscle->GetLmNorm();
            double range = mMuscleLengthMaxValue - mMuscleLengthMinValue;
            double t = std::clamp((lm_norm - mMuscleLengthMinValue) / range, 0.0, 1.0);
            auto c = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis);
            color = Eigen::Vector4d(c.r(), c.g(), c.b(), 0.85);
        } else {
            // Contracture mode (default): viridis colormap
            if (muscle->lmt_base > 0) {
                double ratio = muscle->lm_contract;
                double range = mContractureMaxValue - mContractureMinValue;
                double t = std::clamp((ratio - mContractureMinValue) / range, 0.0, 1.0);
                auto c = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis);
                color = Eigen::Vector4d(c.r(), c.g(), c.b(), 0.85);
            } else {
                color = Eigen::Vector4d(1.0, 0.2, 0.6, 0.85);  // Default magenta
            }
        }

        glColor4dv(color.data());
        mShapeRenderer.renderMuscleLine(muscle, lineWidth);
    }

    // Draw anchor points if enabled (separate from muscle line rendering)
    if (mShowAnchorPoints) {
        for (size_t i = 0; i < muscles.size(); i++) {
            if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

            auto muscle = muscles[i];
            auto& anchors = muscle->GetAnchors();

            bool isSymmetryHighlighted = (!symHighlightL.empty() &&
                (muscle->name == symHighlightL || muscle->name == symHighlightR));

            // Draw anchor points and connections to bodynodes
            for (auto& anchor : anchors) {
                Eigen::Vector3d anchorPos = anchor->GetPoint();

                // Lines to bodynodes
                if (!anchor->bodynodes.empty()) {
                    if (isSymmetryHighlighted) {
                        glColor4f(0.3f, 0.5f, 1.0f, 0.8f);
                    } else {
                        glColor4f(0.0f, 0.8f, 0.0f, 0.6f);
                    }
                    glBegin(GL_LINES);
                    for (auto& bodynode : anchor->bodynodes) {
                        Eigen::Vector3d bnPos = bodynode->getWorldTransform().translation();
                        glVertex3f(anchorPos[0], anchorPos[1], anchorPos[2]);
                        glVertex3f(bnPos[0], bnPos[1], bnPos[2]);
                    }
                    glEnd();
                }

                // Anchor point sphere
                if (isSymmetryHighlighted) {
                    glColor4f(0.2f, 0.4f, 1.0f, 1.0f);
                    GUI::DrawSphere(anchorPos, 0.006);
                } else {
                    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
                    GUI::DrawSphere(anchorPos, 0.004);
                }
            }
        }
    }

    glEnable(GL_LIGHTING);
}

// ============================================================
// UI Panels
// ============================================================

void MusclePersonalizerApp::drawLeftPanel()
{
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::Begin("Control##Panel", nullptr,ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos(ImVec2(0, 0), ImGuiCond_Always);

    if (ImGui::BeginTabBar("ControlTabs")) {
        if (ImGui::BeginTabItem("Character")) {
            drawClinicalDataSection();
            drawCharacterLoadSection();
            drawExportSection();
            drawWeightApplicationSection();
            drawSymmetryOperationsSection();
            drawJointAngleSection();

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Waypoint")) {
            drawWaypointOptimizationSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Contracture")) {
            drawContractureEstimationSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Render")) {
            drawRenderTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void MusclePersonalizerApp::drawRightPanel()
{
    ImGui::SetNextWindowSize(ImVec2(mPlotPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::Begin("Data##Panel", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos(ImVec2(mWidth - ImGui::GetWindowSize().x, 0), ImGuiCond_Always);

    if (ImGui::BeginTabBar("DataTabs")) {
        if (ImGui::BeginTabItem("Waypoint Opt.")) {
            drawWaypointCurvesTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Contracture")) {
            drawContractureResultsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Symmetry")) {
            drawSymmetryTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void MusclePersonalizerApp::drawClinicalDataSection()
{
    if (collapsingHeaderWithControls("Clinical Data")) {
        if (!mPIDNavigator) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Resource Manager not available");
            return;
        }

        // PID Navigator selector (PID list only, no file sections)
        // PID changes are handled by onPIDChanged callback
        mPIDNavigator->renderUI(nullptr, 120, 0);

        // Handle deselection (e.g., after PID list refresh)
        const auto& pidState = mPIDNavigator->getState();
        if (pidState.selectedPID < 0 && !mCurrentROMPID.empty()) {
            mROMSource = ROMSource::Normative;
            mPatientROMValues.clear();
            mCurrentROMPID.clear();
            mClinicalWeightAvailable = false;
        }
    }
}

void MusclePersonalizerApp::drawCharacterLoadSection()
{
    if (collapsingHeaderWithControls("Character Loading")) {
        // Current status
        if (mExecutor && mExecutor->getCharacter()) {
            ImGui::Text("Status: Loaded (%.1f kg, %zu muscles)", mReferenceMass, mAvailableMuscles.size());
        } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Status: Not loaded");
        }

        // Update PID from navigator
        if (mPIDNavigator) {
            const auto& pidState = mPIDNavigator->getState();
            if (pidState.selectedPID >= 0) {
                mCharacterPID = pidState.pidList[pidState.selectedPID];
            }
        }

        ImGui::Separator();

        // ============ SKELETON SECTION ============
        bool skelChanged = false;
        ImGui::Text("Skeleton Source:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Default##skel", mSkeletonDataSource == CharacterDataSource::DefaultData)) {
            if (mSkeletonDataSource != CharacterDataSource::DefaultData) {
                mSkeletonDataSource = CharacterDataSource::DefaultData;
                skelChanged = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Patient##skel", mSkeletonDataSource == CharacterDataSource::PatientData)) {
            if (mSkeletonDataSource != CharacterDataSource::PatientData) {
                mSkeletonDataSource = CharacterDataSource::PatientData;
                skelChanged = true;
            }
        }

        // Patient skeleton info
        if (mSkeletonDataSource == CharacterDataSource::PatientData) {
            if (!mCharacterPID.empty()) {
                ImGui::SameLine();
                const auto& pidState = mPIDNavigator->getState();
                ImGui::Text("(%s/%s)", mCharacterPID.c_str(), pidState.getVisitDir().c_str());
            } else {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "(Select PID first)");
            }
        }

        // Del button for selected skeleton file (both Default and Patient)
        if (!mSkeletonPath.empty()) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Del##skel")) {
                try {
                    std::string resolved = mResourceManager->resolve(mSkeletonPath);
                    if (!resolved.empty() && std::filesystem::exists(resolved)) {
                        std::filesystem::remove(resolved);
                        LOG_INFO("[MusclePersonalizer] Deleted skeleton: " << resolved);
                    }
                } catch (...) {}
                mSkeletonPath.clear();
                scanSkeletonFiles();
            }
        }

        if (skelChanged) {
            scanSkeletonFiles();
            mSkeletonPath.clear();
        }

        // Skeleton file list
        std::string skelPrefix = "@data/skeleton/";
        if (mSkeletonDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            std::string visit = mPIDNavigator->getState().getVisitDir();
            skelPrefix = "@pid:" + mCharacterPID + "/" + visit + "/skeleton/";
        }

        if (ImGui::BeginListBox("##SkeletonList", ImVec2(-1, 60))) {
            for (size_t i = 0; i < mSkeletonCandidates.size(); ++i) {
                const auto& filename = mSkeletonCandidates[i];
                bool isSelected = (mSkeletonPath.find(filename) != std::string::npos);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
                    mSkeletonPath = skelPrefix + filename;
                }
            }
            ImGui::EndListBox();
        }

        ImGui::Spacing();

        // ============ MUSCLE SECTION ============
        bool muscleChanged = false;
        ImGui::Text("Muscle Source:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Default##muscle", mMuscleDataSource == CharacterDataSource::DefaultData)) {
            if (mMuscleDataSource != CharacterDataSource::DefaultData) {
                mMuscleDataSource = CharacterDataSource::DefaultData;
                muscleChanged = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Patient##muscle", mMuscleDataSource == CharacterDataSource::PatientData)) {
            if (mMuscleDataSource != CharacterDataSource::PatientData) {
                mMuscleDataSource = CharacterDataSource::PatientData;
                muscleChanged = true;
            }
        }

        // Patient muscle info
        if (mMuscleDataSource == CharacterDataSource::PatientData) {
            if (!mCharacterPID.empty()) {
                ImGui::SameLine();
                const auto& pidState = mPIDNavigator->getState();
                ImGui::Text("(%s/%s)", mCharacterPID.c_str(), pidState.getVisitDir().c_str());
            } else {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "(Select PID first)");
            }
        }

        // Del button for selected muscle file (both Default and Patient)
        if (!mMusclePath.empty()) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Del##muscle")) {
                try {
                    std::string resolved = mResourceManager->resolve(mMusclePath);
                    if (!resolved.empty() && std::filesystem::exists(resolved)) {
                        std::filesystem::remove(resolved);
                        LOG_INFO("[MusclePersonalizer] Deleted muscle: " << resolved);
                    }
                } catch (...) {}
                mMusclePath.clear();
                scanMuscleFiles();
            }
        }

        if (muscleChanged) {
            scanMuscleFiles();
            mMusclePath.clear();
        }

        // Muscle file list
        std::string musclePrefix = "@data/muscle/";
        if (mMuscleDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            std::string visit = mPIDNavigator->getState().getVisitDir();
            musclePrefix = "@pid:" + mCharacterPID + "/" + visit + "/muscle/";
        }

        if (ImGui::BeginListBox("##MuscleList", ImVec2(-1, 60))) {
            for (size_t i = 0; i < mMuscleCandidates.size(); ++i) {
                const auto& filename = mMuscleCandidates[i];
                bool isSelected = (mMusclePath.find(filename) != std::string::npos);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
                    mMusclePath = musclePrefix + filename;
                }
            }
            ImGui::EndListBox();
        }

        ImGui::Separator();

        // Rebuild button
        if (ImGui::Button("Rebuild", ImVec2(-1, 0))) {
            loadCharacter();
        }
    }
}

void MusclePersonalizerApp::drawWeightApplicationSection()
{
    if (collapsingHeaderWithControls("Weight Application")) {
        ImGui::Text("Apply f0 scaling based on target body mass");
        ImGui::Separator();

        ImGui::Text("Reference Mass: %.1f kg", mReferenceMass);
        ImGui::Text("Current Mass: %.1f kg", mCurrentMass);

        // Weight source radio buttons
        ImGui::Text("Weight Source:");
        if (ImGui::RadioButton("Clinical Data##weight", mWeightSource == WeightSource::ClinicalData)) {
            mWeightSource = WeightSource::ClinicalData;
            if (mClinicalWeightAvailable) {
                mTargetMass = mClinicalWeight;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("User Input##weight", mWeightSource == WeightSource::UserInput)) {
            mWeightSource = WeightSource::UserInput;
        }

        // Display based on weight source
        if (mWeightSource == WeightSource::ClinicalData) {
            if (mClinicalWeightAvailable) {
                ImGui::Text("Clinical Weight: %.1f kg", mClinicalWeight);
                mTargetMass = mClinicalWeight;  // Keep in sync
            } else {
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No clinical weight data available");
                ImGui::TextWrapped("Select a patient from Clinical Data section");
            }
        } else {
            // User input mode
            ImGui::SliderFloat("Target Mass (kg)", &mTargetMass, 20.0f, 120.0f);
        }

        if (mReferenceMass > 0.0f) {
            float ratio = std::pow(mTargetMass / mReferenceMass, 2.0f / 3.0f);
            ImGui::Text("f0 Scaling Ratio: %.3f", ratio);
        }

        ImGui::Separator();
        bool canApply = (mWeightSource == WeightSource::UserInput) ||
                        (mWeightSource == WeightSource::ClinicalData && mClinicalWeightAvailable);
        if (!canApply) ImGui::BeginDisabled();
        if (ImGui::Button("Apply Weight Scaling", ImVec2(-1, 0))) {
            applyWeightScaling();
        }
        if (!canApply) ImGui::EndDisabled();

        if (mAppliedRatio != 1.0f) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Last applied ratio: %.3f", mAppliedRatio);
        }
    }
}

void MusclePersonalizerApp::drawJointAngleSection()
{
    if (collapsingHeaderWithControls("Joint Angle")) {
        if (!mExecutor || !mExecutor->getCharacter()) {
            ImGui::TextDisabled("Load character first");
            return;
        }

        auto character = mExecutor->getCharacter();
        auto skel = character->getSkeleton();

        // Reset button
        if (ImGui::Button("Reset to Zero")) {
            skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));
            for (auto* m : character->getMuscles()) {
                m->UpdateGeometry();
            }
        }

        ImGui::Separator();

        // Get current positions and limits
        Eigen::VectorXd pos_lower_limit = skel->getPositionLowerLimits();
        Eigen::VectorXd pos_upper_limit = skel->getPositionUpperLimits();
        Eigen::VectorXd currentPos = skel->getPositions();
        Eigen::VectorXf pos_rad = currentPos.cast<float>();
        Eigen::VectorXf pos_deg = pos_rad * (180.0f / M_PI);

        // DOF direction labels
        const char* dof_labels[] = {"X", "Y", "Z", "tX", "tY", "tZ"};

        bool changed = false;
        int dof_idx = 0;
        for (size_t j = 0; j < skel->getNumJoints(); j++) {
            auto joint = skel->getJoint(j);
            std::string joint_name = joint->getName();
            int num_dofs = joint->getNumDofs();

            if (num_dofs == 0) continue;

            // Display joint name as a header
            ImGui::Text("%s:", joint_name.c_str());
            ImGui::Indent();

            for (int d = 0; d < num_dofs; d++) {
                // Check if this is a translation DOF (root joint DOFs 3-5 are tx, ty, tz)
                bool is_translation = (dof_idx >= 3 && dof_idx < 6);

                // Prepare limits and display value
                float lower_limit, upper_limit;
                float display_value;

                if (dof_idx < 6) {
                    // Root joint - expand limits
                    if (is_translation) {
                        lower_limit = -2.0f;
                        upper_limit = 2.0f;
                        display_value = pos_rad[dof_idx];
                    } else {
                        lower_limit = -2.0f * (180.0f / M_PI);
                        upper_limit = 2.0f * (180.0f / M_PI);
                        display_value = pos_deg[dof_idx];
                    }
                } else {
                    // Non-root joints: always rotation, convert to degrees
                    lower_limit = pos_lower_limit[dof_idx] * (180.0f / M_PI);
                    upper_limit = pos_upper_limit[dof_idx] * (180.0f / M_PI);
                    display_value = pos_deg[dof_idx];
                }

                // Create label
                std::string label;
                if (num_dofs > 1 && d < 6) {
                    label = std::string(dof_labels[d]);
                } else if (num_dofs > 1) {
                    label = "DOF " + std::to_string(d);
                } else {
                    label = "";
                }

                float prev_value = display_value;

                // Slider
                std::string slider_label = label + "##slider_" + joint_name + std::to_string(d);
                ImGui::SetNextItemWidth(200);
                const char* format = is_translation ? "%.3fm" : "%.1f°";
                ImGui::SliderFloat(slider_label.c_str(), &display_value, lower_limit, upper_limit, format);

                // InputFloat on same line
                ImGui::SameLine();
                std::string input_label = "##input_" + joint_name + std::to_string(d);
                ImGui::SetNextItemWidth(50);
                const char* input_format = is_translation ? "%.3f" : "%.1f";
                ImGui::InputFloat(input_label.c_str(), &display_value, 0.0f, 0.0f, input_format);

                // Clamp to limits
                if (display_value < lower_limit) display_value = lower_limit;
                if (display_value > upper_limit) display_value = upper_limit;

                // Update internal storage
                if (is_translation) {
                    pos_rad[dof_idx] = display_value;
                } else {
                    pos_deg[dof_idx] = display_value;
                    pos_rad[dof_idx] = display_value * (M_PI / 180.0f);
                }

                if (prev_value != display_value) {
                    changed = true;
                }

                dof_idx++;
            }

            ImGui::Unindent();
        }

        // Update positions if changed
        if (changed) {
            skel->setPositions(pos_rad.cast<double>());
            for (auto* m : character->getMuscles()) {
                m->UpdateGeometry();
            }
        }
    }
}

void MusclePersonalizerApp::drawWaypointOptimizationSection()
{
    ImGui::Text("Optimize muscle waypoints from HDF motion");
        ImGui::Separator();

        // Motion file selection with DOF compatibility
        if (ImGui::TreeNodeEx("Motion File", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Show skeleton DOF if character loaded
            int skelDof = 0;
            if (mExecutor && mExecutor->getCharacter()) {
                skelDof = mExecutor->getCharacter()->getSkeleton()->getNumDofs();
                ImGui::Text("Skeleton DOF: %d", skelDof);
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Load character to check compatibility");
            }

            // Auto-select first compatible motion file if none selected
            if (mHDFPath[0] == '\0' && skelDof > 0) {
                for (const auto& filename : mMotionCandidates) {
                    auto it = mMotionDOFMap.find(filename);
                    int motionDof = (it != mMotionDOFMap.end()) ? it->second : -1;
                    if (skelDof == 0 || motionDof == skelDof) {
                        std::string fullPath = "@data/motion/" + filename;
                        strncpy(mHDFPath, fullPath.c_str(), sizeof(mHDFPath) - 1);
                        mHDFPath[sizeof(mHDFPath) - 1] = '\0';
                        break;
                    }
                }
            }

            if (ImGui::BeginListBox("##MotionList", ImVec2(-1, 80))) {
                for (const auto& filename : mMotionCandidates) {
                    int motionDof = -1;
                    auto it = mMotionDOFMap.find(filename);
                    if (it != mMotionDOFMap.end()) {
                        motionDof = it->second;
                    }

                    bool compatible = (skelDof == 0 || motionDof == skelDof);
                    bool isSelected = (std::string(mHDFPath).find(filename) != std::string::npos);

                    // Build label with DOF info
                    std::string label = filename;
                    if (motionDof > 0) {
                        label += " (" + std::to_string(motionDof) + " DOF)";
                    }

                    if (!compatible) {
                        // Faint text for incompatible motions
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                        ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_Disabled);
                        ImGui::PopStyleColor();
                    } else {
                        if (ImGui::Selectable(label.c_str(), isSelected)) {
                            std::string fullPath = "@data/motion/" + filename;
                            strncpy(mHDFPath, fullPath.c_str(), sizeof(mHDFPath) - 1);
                            mHDFPath[sizeof(mHDFPath) - 1] = '\0';
                        }
                    }
                }
                ImGui::EndListBox();
            }
            ImGui::TreePop();
        }

        // Muscle selection
        if (ImGui::TreeNodeEx("Muscles", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (mExecutor && mExecutor->getCharacter()) {
                const auto& muscles = mExecutor->getCharacter()->getMuscles();
                ImGuiCommon::MuscleSelector("##WaypointMuscles", muscles, mSelectedMuscles,
                                            mMuscleFilter, sizeof(mMuscleFilter), 120);
            }
            ImGui::TreePop();
        }

        // Optimization Parameters
        if (ImGui::TreeNodeEx("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Two-column layout for numeric parameters
            if (ImGui::BeginTable("##waypoint_params", 2, ImGuiTableFlags_None)) {
                ImGui::TableSetupColumn("##col1", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("##col2", ImGuiTableColumnFlags_WidthStretch);

                // Row 1: Max Iterations | Num Sampling
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Max Iterations");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##max_iter", &mWaypointMaxIterations);

                ImGui::TableNextColumn();
                ImGui::Text("Num Sampling");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##num_sampling", &mWaypointNumSampling);

                // Row 2: Lambda Shape | Lambda Length
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Lambda Shape");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##lambda_shape", &mWaypointLambdaShape, 0.0f, 0.0f, "%.3f");

                ImGui::TableNextColumn();
                ImGui::Text("Lambda Length");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##lambda_length", &mWaypointLambdaLengthCurve, 0.0f, 0.0f, "%.3f");

                // Row 3: Weight Phase | Weight Delta
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Weight Phase");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##weight_phase", &mWaypointWeightPhase, 0.0f, 0.0f, "%.3f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Weight for phase matching:\n(min_phase - ref)^2 + (max_phase - ref)^2");
                }

                ImGui::TableNextColumn();
                ImGui::Text("Weight Delta");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##weight_delta", &mWaypointWeightDelta, 0.0f, 0.0f, "%.3f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Weight for delta matching:\n(delta - ref)^2 where delta = max - min");
                }

                // Row 4: Weight Samples | Num Phase Samples
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Weight Samples");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##weight_samples", &mWaypointWeightSamples, 0.0f, 0.0f, "%.3f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Weight for sample matching at N phase points");
                }

                ImGui::TableNextColumn();
                ImGui::Text("Phase Samples");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##num_phase_samples", &mWaypointNumPhaseSamples);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Number of phase points (e.g., 3 = {0, 0.5, 1.0})");
                }

                // Row 5: Loss Power | Num Parallel
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Loss Power");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##loss_power", &mWaypointLossPower);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Exponent for power loss: |x|^n (2=squared, 3=cube)");
                }

                ImGui::TableNextColumn();
                ImGui::Text("Num Parallel");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##num_parallel", &mWaypointNumParallel);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Parallel threads (1 = sequential)");
                }

                // Row 6: Max Displacement | Max Disp Origin/Insertion
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Max Disp");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##max_disp", &mWaypointMaxDisplacement, 0.0f, 0.0f, "%.3f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Max waypoint displacement from initial position (meters)");
                }

                ImGui::TableNextColumn();
                ImGui::Text("Max Disp O/I");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##max_disp_oi", &mWaypointMaxDispOriginInsertion, 0.0f, 0.0f, "%.3f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Max displacement for origin/insertion (meters)\nShould be tighter than normal waypoints");
                }

                // Row 7: Function Tol | Gradient Tol
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Func Tol");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##func_tol", &mWaypointFunctionTolerance, 0.0f, 0.0f, "%.1e");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Convergence tolerance on cost function change");
                }

                ImGui::TableNextColumn();
                ImGui::Text("Grad Tol");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##grad_tol", &mWaypointGradientTolerance, 0.0f, 0.0f, "%.1e");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Convergence tolerance on gradient norm");
                }

                // Row 8: Param Tol | Outer Iterations
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Param Tol");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputFloat("##param_tol", &mWaypointParameterTolerance, 0.0f, 0.0f, "%.1e");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Convergence tolerance on parameter change");
                }

                ImGui::EndTable();
            }

            // Checkboxes in two columns
            if (ImGui::BeginTable("##waypoint_checkboxes", 2, ImGuiTableFlags_None)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Checkbox("Fix Origin/Insertion", &mWaypointFixOriginInsertion);

                ImGui::TableNextColumn();
                ImGui::Checkbox("Analytical Gradient", &mWaypointAnalyticalGradient);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Faster but may have numerical issues.\nUncheck for numeric gradient.");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Checkbox("Verbose", &mWaypointVerbose);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Print optimization progress to console.");
                }

                ImGui::TableNextColumn();
                ImGui::Checkbox("Normalized (lm_norm)", &mWaypointUseNormalizedLength);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Use normalized muscle fiber length.\nMore accurate but slower.");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Checkbox("Adaptive Sample Weight", &mWaypointAdaptiveSampleWeight);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Weight samples by normalized length:\n"
                                      "len<1.0: w=1, len<1.2: w=2^len, else: w=5^len");
                }

                ImGui::EndTable();
            }

            ImGui::TreePop();
        }

        ImGui::Separator();

        // Disable button while optimization is running
        bool isRunning = mWaypointOptRunning.load();
        if (isRunning) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button(isRunning ? "Optimizing..." : "Optimize Waypoints", ImVec2(-1, 0))) {
            runWaypointOptimizationAsync();
        }

    if (isRunning) {
        ImGui::EndDisabled();
    }
}


void MusclePersonalizerApp::drawProgressOverlay()
{
    if (!mWaypointOptRunning) return;

    // Get display size for centering
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 displaySize = io.DisplaySize;

    // Center window
    ImVec2 windowSize(400, 150);
    ImVec2 windowPos((displaySize.x - windowSize.x) / 2,
                     (displaySize.y - windowSize.y) / 2);

    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(windowSize);
    ImGui::Begin("Optimizing...", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse);

    // Progress bar (atomic reads are thread-safe)
    int current = mWaypointOptCurrent.load();
    int total = mWaypointOptTotal.load();
    float progress = total > 0 ? static_cast<float>(current) / total : 0.0f;

    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%d / %d muscles", current, total);
    ImGui::ProgressBar(progress, ImVec2(-1, 0), overlay);

    // Current muscle name (mutex-protected)
    std::string muscleName;
    {
        std::lock_guard<std::mutex> lock(mWaypointOptMutex);
        muscleName = mWaypointOptMuscleName;
    }
    ImGui::Text("Current: %s", muscleName.c_str());

    // Elapsed time and ETA
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - mWaypointOptStartTime);
    int elapsedSec = static_cast<int>(elapsed.count());

    if (current > 0 && progress < 1.0f) {
        float avgTime = static_cast<float>(elapsedSec) / current;
        int remaining = static_cast<int>(avgTime * (total - current));
        ImGui::Text("Elapsed: %d:%02d  |  ETA: %d:%02d",
            elapsedSec / 60, elapsedSec % 60,
            remaining / 60, remaining % 60);
    } else {
        ImGui::Text("Elapsed: %d:%02d  |  ETA: --:--",
            elapsedSec / 60, elapsedSec % 60);
    }

    ImGui::End();
}

void MusclePersonalizerApp::drawContractureEstimationSection()
{
    ImGui::Text("Estimate lm_contract from ROM trials");
        ImGui::Separator();

        ImGui::Text("ROM Config Directory:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", mROMConfigDir.c_str());
        ImGui::SameLine();
        if (ImGui::Button("Scan ROM Configs")) scanROMConfigs();
        ImGui::SameLine();
        if (ImGui::Button("Default")) applyDefaultROMValues();

        // Count selected and filtered trials
        size_t selectedCount = 0;
        size_t filteredCount = 0;
        for (const auto& trial : mROMTrials) {
            bool visible = (strlen(mROMFilter) == 0 || trial.name.find(mROMFilter) != std::string::npos);
            if (visible) {
                filteredCount++;
                if (trial.selected) selectedCount++;
            }
        }
        ImGui::Text("ROM Trials: %zu / %zu selected", selectedCount, filteredCount);
        ImGui::SameLine();
        if (ImGui::SmallButton("All")) {
            for (auto& trial : mROMTrials) {
                bool visible = (strlen(mROMFilter) == 0 || trial.name.find(mROMFilter) != std::string::npos);
                if (visible) trial.selected = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) {
            for (auto& trial : mROMTrials) {
                bool visible = (strlen(mROMFilter) == 0 || trial.name.find(mROMFilter) != std::string::npos);
                if (visible) trial.selected = false;
            }
        }
        ImGui::InputText("Filter##ROM", mROMFilter, sizeof(mROMFilter));

        // ROM source selector
        ImGui::Text("ROM Source:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Normative##rom", mROMSource == ROMSource::Normative)) {
            if (mROMSource != ROMSource::Normative) {
                mROMSource = ROMSource::Normative;
                // No need to modify cd_value - getEffectiveROMValue() will use trial.normative
            }
        }
        if (mPIDNavigator) {
            const auto& pidState = mPIDNavigator->getState();
            if (pidState.selectedPID >= 0) {
                const std::string& pid = pidState.pidList[pidState.selectedPID];
                ImGui::SameLine();
                std::string preLabel = "Pre (" + pid + ")##rom";
                if (ImGui::RadioButton(preLabel.c_str(), mROMSource == ROMSource::Pre)) {
                    if (mROMSource != ROMSource::Pre) {
                        mROMSource = ROMSource::Pre;
                        loadPatientROM(pid, "pre");
                    }
                }
                ImGui::SameLine();
                std::string op1Label = "Op1 (" + pid + ")##rom";
                if (ImGui::RadioButton(op1Label.c_str(), mROMSource == ROMSource::Op1)) {
                    if (mROMSource != ROMSource::Op1) {
                        mROMSource = ROMSource::Op1;
                        loadPatientROM(pid, "op1");
                    }
                }
            }
        }

        // ROM selection list - display "TC name - CD value"
        ImGui::BeginChild("ROMList", ImVec2(0, 250), true);
        for (size_t i = 0; i < mROMTrials.size(); ++i) {
            auto& trial = mROMTrials[i];

            // Apply filter
            if (strlen(mROMFilter) > 0) {
                if (trial.name.find(mROMFilter) == std::string::npos) {
                    continue;
                }
            }

            // Format label: "TC name - value"
            auto effectiveValue = getEffectiveROMValue(trial);
            bool needsManualInput = false;

            // If CD not available, use normative as default (editable)
            if (!effectiveValue.has_value() && trial.normative.has_value()) {
                // Initialize manual_rom with normative if not yet set
                if (std::abs(trial.manual_rom) < 0.001f) {
                    mROMTrials[i].manual_rom = trial.normative.value();
                }
                needsManualInput = true;
            }

            // Build label with current value
            std::string label;
            char buf[128];
            if (effectiveValue.has_value()) {
                snprintf(buf, sizeof(buf), "%s - %.1f°", trial.name.c_str(), effectiveValue.value());
                label = buf;
            } else if (needsManualInput) {
                snprintf(buf, sizeof(buf), "%s - %.1f°", trial.name.c_str(), mROMTrials[i].manual_rom);
                label = buf;
            } else {
                label = trial.name + " - N/A";
            }

            if (ImGui::Checkbox(label.c_str(), &trial.selected)) {
                // Selection changed
            }

            // Show editable input when CD not available (using normative as default)
            if (needsManualInput) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                char input_id[64];
                snprintf(input_id, sizeof(input_id), "##manual_rom_%zu", i);
                if (ImGui::InputFloat(input_id, &mROMTrials[i].manual_rom, 0, 0, "%.1f")) {
                    // Manual ROM value updated
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Edit ROM angle (default: normative %.1f°)", trial.normative.value());
                }
            }

            // Show tooltip with details on hover
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Description: %s", trial.description.c_str());
                ImGui::Text("Joint: %s (DOF %d)", trial.joint.c_str(), trial.dof_index);
                ImGui::Text("Torque Cutoff: %.1f Nm", trial.torque_cutoff);
                if (trial.normative.has_value()) {
                    ImGui::Text("Normative: %.1f°", trial.normative.value());
                }
                if (!trial.cd_side.empty()) {
                    ImGui::Separator();
                    ImGui::Text("Clinical Data: %s.%s.%s",
                        trial.cd_side.c_str(), trial.cd_joint.c_str(), trial.cd_field.c_str());
                    if (trial.cd_value.has_value()) {
                        ImGui::Text("Patient Value: %.1f°", trial.cd_value.value());
                    } else {
                        ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Patient Value: Not available");
                    }
                }
                ImGui::EndTooltip();
            }
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Text("Optimization Parameters:");
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("Max Iterations##Contract", &mContractureMaxIterations);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("Min Ratio", &mContractureMinRatio, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("Max Ratio", &mContractureMaxRatio, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::Checkbox("Verbose##Contracture", &mContractureVerbose);

        ImGui::Text("Grid Search Range:");
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("Begin##Grid", &mContractureGridBegin, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("End##Grid", &mContractureGridEnd, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("Step##Grid", &mContractureGridInterval, 0.0f, 0.0f, "%.2f");

        ImGui::Text("Regularization:");
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("Ratio##Reg", &mContractureLambdaRatioReg, 0.0f, 0.0f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Penalize (ratio - 1.0)^2\n0 = disabled");
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("Torque##Reg", &mContractureLambdaTorqueReg, 0.0f, 0.0f, "%.3f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Penalize passive torque magnitude\n0 = disabled");
        }

        ImGui::SetNextItemWidth(120);
        ImGui::SliderInt("Outer Iters", &mContractureOuterIterations, 1, 10);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Outer iterations for biarticular convergence\n1 = single pass");
        }

    ImGui::Separator();
    if (ImGui::Button("Estimate Contracture Parameters", ImVec2(-1, 0))) runContractureEstimation();

    // Display brief optimization result summary
    if (mContractureOptResult.has_value() && !mContractureOptResult->trial_results.empty()) {
        ImGui::Spacing();

        // Calculate success rate (error <= 5%)
        int successCount = 0;
        int totalCount = static_cast<int>(mContractureOptResult->trial_results.size());
        for (const auto& trial : mContractureOptResult->trial_results) {
            double error_pct = 0.0;
            if (std::abs(trial.observed_torque) > 1e-6) {
                error_pct = std::abs(trial.computed_torque_after - trial.observed_torque)
                          / std::abs(trial.observed_torque) * 100.0;
            }
            if (error_pct <= 5.0) successCount++;
        }
        float successRate = totalCount > 0 ? (successCount * 100.0f / totalCount) : 0.0f;

        // Display header with success rate
        ImVec4 rateColor = (successRate >= 100.0f)
            ? ImVec4(0.3f, 0.9f, 0.3f, 1.0f)   // Green if 100%
            : ImVec4(0.9f, 0.6f, 0.2f, 1.0f);  // Orange otherwise
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "[Outer %d/%d]",
            mContractureOuterIterations, mContractureOuterIterations);
        ImGui::SameLine();
        ImGui::TextColored(rateColor, "Success: %d/%d (%.0f%%)", successCount, totalCount, successRate);

        // Display each trial result with wrapping
        float availableWidth = ImGui::GetContentRegionAvail().x;
        float currentX = 0.0f;

        for (const auto& trial : mContractureOptResult->trial_results) {
            // Calculate error percentage
            double error_pct = 0.0;
            if (std::abs(trial.observed_torque) > 1e-6) {
                error_pct = std::abs(trial.computed_torque_after - trial.observed_torque)
                          / std::abs(trial.observed_torque) * 100.0;
            }

            // Red if error > 5%, green otherwise
            ImVec4 color = (error_pct > 5.0)
                ? ImVec4(0.9f, 0.3f, 0.3f, 1.0f)   // Red
                : ImVec4(0.3f, 0.9f, 0.3f, 1.0f);  // Green

            char buf[64];
            snprintf(buf, sizeof(buf), "%s[%d]=%.2f(%.2f)",
                trial.joint.c_str(), trial.dof_index,
                trial.computed_torque_after, trial.observed_torque);

            // Calculate text width for wrapping
            float textWidth = ImGui::CalcTextSize(buf).x + ImGui::GetStyle().ItemSpacing.x;
            if (currentX + textWidth > availableWidth && currentX > 0) {
                // Wrap to next line
                currentX = 0.0f;
            } else if (currentX > 0) {
                ImGui::SameLine();
            }

            ImGui::TextColored(color, "%s", buf);
            currentX += textWidth;
        }
    }
}

void MusclePersonalizerApp::drawRenderTab()
{
    // Character selection toggle
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Character");
    ImGui::Separator();
    if (ImGui::RadioButton("Subject", !mRenderReferenceCharacter)) {
        mRenderReferenceCharacter = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Reference", mRenderReferenceCharacter)) {
        mRenderReferenceCharacter = true;
    }
    if (!mRenderReferenceCharacter) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Rendering subject (patient) character");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Rendering reference (standard) character");
    }

    // Skeleton Rendering
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Skeleton");
    ImGui::Separator();
    ImGui::Text("Render Mode:");
    ImGui::Indent();
    if (ImGui::RadioButton("Mesh##skel", mRenderMode == RenderMode::Mesh)) {
        mRenderMode = RenderMode::Mesh;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Primitive##skel", mRenderMode == RenderMode::Primitive)) {
        mRenderMode = RenderMode::Primitive;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Wireframe##skel", mRenderMode == RenderMode::Wireframe)) {
        mRenderMode = RenderMode::Wireframe;
    }
    ImGui::Unindent();

    // Muscle Rendering
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Muscles");
    ImGui::Separator();
    ImGui::Checkbox("Show Muscles", &mRenderMuscles);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##LineWidth", &mMuscleLineWidth, 0.0f, 0.0f, "%.1f");
    ImGui::SameLine();
    ImGui::TextDisabled("Width");
    ImGui::Checkbox("Show Anchor Points", &mShowAnchorPoints);

    // Muscle color mode
    ImGui::Text("Color Mode:");
    ImGui::Indent();
    if (ImGui::RadioButton("Contracture", mMuscleColorMode == MuscleColorMode::Contracture)) {
        mMuscleColorMode = MuscleColorMode::Contracture;
    }
    if (ImGui::RadioButton("Contracture (ext)", mMuscleColorMode == MuscleColorMode::ContractureExt)) {
        mMuscleColorMode = MuscleColorMode::ContractureExt;
    }
    if (ImGui::RadioButton("Symmetry", mMuscleColorMode == MuscleColorMode::Symmetry)) {
        mMuscleColorMode = MuscleColorMode::Symmetry;
    }
    if (ImGui::RadioButton("Muscle Length", mMuscleColorMode == MuscleColorMode::MuscleLength)) {
        mMuscleColorMode = MuscleColorMode::MuscleLength;
    }
    ImGui::Unindent();

    if (mMuscleColorMode == MuscleColorMode::Contracture ||
        mMuscleColorMode == MuscleColorMode::ContractureExt) {
        ImGui::Separator();
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("##MinVal", &mContractureMinValue, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("##MaxVal", &mContractureMaxValue, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::TextDisabled("Min/Max");
        if (mMuscleColorMode == MuscleColorMode::Contracture) {
            ImGuiCommon::ColorBarLegend("Contracture Ratio", mContractureMinValue, mContractureMaxValue,
                                        "(contracted)", "(lengthened)", "viridis");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Red: > %.2f", mContractureMaxValue);
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.8f, 1.0f), "Pink: < %.2f", mContractureMinValue);
            ImGui::TextDisabled("In range: not drawn");
        }
    } else if (mMuscleColorMode == MuscleColorMode::MuscleLength) {
        ImGui::Separator();
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("##LmMinVal", &mMuscleLengthMinValue, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("##LmMaxVal", &mMuscleLengthMaxValue, 0.0f, 0.0f, "%.2f");
        ImGui::SameLine();
        ImGui::TextDisabled("Min/Max");
        ImGuiCommon::ColorBarLegend("Normalized Length (lm_norm)", mMuscleLengthMinValue, mMuscleLengthMaxValue,
                                    "(shortened)", "(lengthened)", "viridis");
    }

    // Muscle selection
    if (mRenderMuscles && mExecutor && mExecutor->getCharacter()) {
        ImGui::Separator();
        auto muscles = mExecutor->getCharacter()->getMuscles();
        ImGuiCommon::MuscleSelector("##MuscleList", muscles, mMuscleSelectionStates,
                                    mMuscleFilterText, sizeof(mMuscleFilterText), 200.0f);
    }

    // Ground
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Ground");
    ImGui::Separator();
    ImGui::Checkbox("Show Ground Grid", &mRenderGround);
    if (mRenderGround) {
        ImGui::Text("Ground Mode:");
        ImGui::Indent();
        if (ImGui::RadioButton("Wireframe##ground", mGroundMode == GroundMode::Wireframe)) {
            mGroundMode = GroundMode::Wireframe;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Solid##ground", mGroundMode == GroundMode::Solid)) {
            mGroundMode = GroundMode::Solid;
        }
        ImGui::Unindent();
    }

    // Camera
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Camera");
    ImGui::Separator();
    float zoom = static_cast<float>(mCamera.zoom);
    if (ImGui::SliderFloat("Zoom", &zoom, 0.1f, 5.0f)) {
        mCamera.zoom = zoom;
    }
    float persp = static_cast<float>(mCamera.persp);
    if (ImGui::SliderFloat("FOV", &persp, 20.0f, 90.0f)) {
        mCamera.persp = persp;
    }

    ImGui::Separator();
    if (ImGui::Button("Reset Camera")) {
        resetCamera();
        // Apply MusclePersonalizer-specific camera defaults
        mCamera.eye = Eigen::Vector3d(0.0, 0.0, 3.0);
        mCamera.trans = Eigen::Vector3d(0.0, -0.5, 0.0);
    }
    ImGui::SameLine();
    if (ImGui::Button("Front View")) {
        mCamera.trackball = dart::gui::Trackball();
        mCamera.trans = Eigen::Vector3d(0.0, -0.5, 0.0);
        mCamera.zoom = 1.0;
    }
    if (ImGui::Button("Side View")) {
        mCamera.trackball = dart::gui::Trackball();
        mCamera.trackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
        mCamera.trackball.setRadius(std::min(mWidth, mHeight) * 0.4);
        mCamera.trackball.startBall(mWidth / 2, mHeight / 2);
        mCamera.trackball.updateBall(mWidth / 2 + mWidth * 0.25, mHeight / 2);
        mCamera.trans = Eigen::Vector3d(0.0, -0.5, 0.0);
        mCamera.zoom = 1.0;
    }
    ImGui::SameLine();
    if (ImGui::Button("Top View")) {
        mCamera.trackball = dart::gui::Trackball();
        mCamera.trackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
        mCamera.trackball.setRadius(std::min(mWidth, mHeight) * 0.4);
        mCamera.trackball.startBall(mWidth / 2, mHeight / 2);
        mCamera.trackball.updateBall(mWidth / 2, mHeight / 2 - mHeight * 0.25);
        mCamera.trans = Eigen::Vector3d(0.0, 0.0, 0.0);
        mCamera.zoom = 1.0;
    }

    // Plot Settings
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Plot");
    ImGui::Separator();
    ImGui::SetNextItemWidth(100);
    ImGui::InputFloat("Plot Height", &mPlotHeight, 10.0f, 50.0f, "%.0f");
    ImGui::Checkbox("Hide Legend", &mPlotHideLegend);

    // Legend position
    ImGui::Text("Legend:");
    ImGui::SameLine();
    int legendPos = mPlotLegendEast ? 1 : 0;
    if (ImGui::RadioButton("West (Left)", &legendPos, 0)) {
        mPlotLegendEast = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("East (Right)", &legendPos, 1)) {
        mPlotLegendEast = true;
    }

    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("Bars per Chart", &mPlotBarsPerChart);
    mPlotBarsPerChart = std::max(1, std::min(20, mPlotBarsPerChart));
}

void MusclePersonalizerApp::drawExportSection()
{
    if (!mExecutor || !mExecutor->getCharacter()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No character loaded");
        return;
    }

    // Determine export directory based on PID selection
    std::string musclePrefix = "@data/muscle/";
    std::string pathDisplay = "@data/";

    if (mPIDNavigator) {
        const auto& pidState = mPIDNavigator->getState();
        if (pidState.selectedPID >= 0) {
            const std::string& pid = pidState.pidList[pidState.selectedPID];
            std::string visit = pidState.getVisitDir();
            musclePrefix = "@pid:" + pid + "/" + visit + "/muscle/";
            pathDisplay = "@pid:" + pid + "/" + visit + "/";
        }
    }

    // Show current export path
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Export to: %s", pathDisplay.c_str());
    ImGui::Spacing();

    // Muscle export
    ImGui::Text("Muscle:  ");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-80);
    ImGui::InputText("##export_muscle_name", mExportMuscleName, sizeof(mExportMuscleName));
    ImGui::SameLine();

    std::string muscleFilename = std::string(mExportMuscleName) + ".yaml";
    fs::path muscleDir = rm::getManager().resolveDir(musclePrefix);
    fs::path musclePath = muscleDir / muscleFilename;
    bool muscleExists = fs::exists(musclePath);

    if (ImGui::Button("Export##muscle", ImVec2(-1, 0))) {
        try {
            if (musclePrefix.find("@pid:") != std::string::npos) {
                muscleDir = rm::getManager().resolveDirCreate(musclePrefix);
                musclePath = muscleDir / muscleFilename;
            }

            // Set modification record for export metadata
            PMuscle::MuscleModificationRecord record;

            // Populate from stored contracture trials
            record.contracture_trials = mContractureUsedTrials;

            // Populate from waypoint results
            if (!mWaypointOptResults.empty()) {
                // Extract filename from mHDFPath
                fs::path hdfPath(mHDFPath);
                record.waypoint_motion_file = hdfPath.filename().string();
                for (const auto& r : mWaypointOptResults) {
                    record.waypoint_muscles.push_back(r.muscle_name);
                }
            }

            mExecutor->setModificationRecord(record);
            mExecutor->exportMuscles(musclePath.string());
            LOG_INFO("[MusclePersonalizer] Muscle exported to: " << musclePrefix << muscleFilename );
        } catch (const std::exception& e) {
            LOG_ERROR("[MusclePersonalizer] Error exporting muscle: " << e.what() );
        }
    }
    if (muscleExists) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "  %s exists - will overwrite", muscleFilename.c_str());
    }
}

void MusclePersonalizerApp::drawWaypointCurvesTab()
{
    std::lock_guard<std::mutex> lock(mWaypointResultsMutex);

    if (mWaypointOptResults.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No results. Run waypoint optimization first.");
        return;
    }

    // Build muscle names list
    std::vector<std::string> muscleNames;
    muscleNames.reserve(mWaypointOptResults.size());
    for (const auto& r : mWaypointOptResults) {
        muscleNames.push_back(r.muscle_name);
    }

    // Count successes for display
    int numSuccess = 0;
    for (const auto& r : mWaypointOptResults) {
        if (r.success) numSuccess++;
    }

    // Sortable table for muscle selection
    char treeLabel[64];
    snprintf(treeLabel, sizeof(treeLabel), "Optimized result (%d/%zu)",
             numSuccess, mWaypointOptResults.size());
    if (ImGui::TreeNodeEx(treeLabel, ImGuiTreeNodeFlags_DefaultOpen)) {
        // Sort key radio buttons
        ImGui::Text("Sort by:");
        ImGui::SameLine();
        int sortCol = static_cast<int>(mWaypointSortColumn);
        if (ImGui::RadioButton("Name", &sortCol, 0)) {
            mWaypointSortColumn = WaypointSortColumn::Name;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Shape", &sortCol, 1)) {
            mWaypointSortColumn = WaypointSortColumn::ShapeEnergy;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Length", &sortCol, 2)) {
            mWaypointSortColumn = WaypointSortColumn::LengthEnergy;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Total", &sortCol, 3)) {
            mWaypointSortColumn = WaypointSortColumn::TotalEnergy;
        }

        // Sort direction radio buttons
        int sortDir = mWaypointSortAscending ? 0 : 1;
        if (ImGui::RadioButton("Asc", &sortDir, 0)) {
            mWaypointSortAscending = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Desc", &sortDir, 1)) {
            mWaypointSortAscending = false;
        }

        // Before/After energy selector
        ImGui::SameLine();
        ImGui::Text(" | ");
        ImGui::SameLine();
        int energyType = mWaypointShowAfterEnergy ? 1 : 0;
        if (ImGui::RadioButton("Before", &energyType, 0)) {
            mWaypointShowAfterEnergy = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("After", &energyType, 1)) {
            mWaypointShowAfterEnergy = true;
        }

        // Filter input
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##ResultFilter", mWaypointResultFilter, sizeof(mWaypointResultFilter));
        ImGui::SameLine();
        ImGui::TextDisabled("Filter");

        // Build sorted indices
        mWaypointSortedIndices.clear();
        for (int i = 0; i < static_cast<int>(mWaypointOptResults.size()); ++i) {
            mWaypointSortedIndices.push_back(i);
        }

        // Sort based on current column (using before/after energy as selected)
        std::stable_sort(mWaypointSortedIndices.begin(), mWaypointSortedIndices.end(),
            [this](int a, int b) {
                const auto& ra = mWaypointOptResults[a];
                const auto& rb = mWaypointOptResults[b];

                // Select energy values based on before/after toggle
                double shapeA = mWaypointShowAfterEnergy ? ra.final_shape_energy : ra.initial_shape_energy;
                double shapeB = mWaypointShowAfterEnergy ? rb.final_shape_energy : rb.initial_shape_energy;
                double lengthA = mWaypointShowAfterEnergy ? ra.final_length_energy : ra.initial_length_energy;
                double lengthB = mWaypointShowAfterEnergy ? rb.final_length_energy : rb.initial_length_energy;

                bool less = false;
                switch (mWaypointSortColumn) {
                    case WaypointSortColumn::Name:
                        less = ra.muscle_name < rb.muscle_name;
                        break;
                    case WaypointSortColumn::ShapeEnergy:
                        less = shapeA < shapeB;
                        break;
                    case WaypointSortColumn::LengthEnergy:
                        less = lengthA < lengthB;
                        break;
                    case WaypointSortColumn::TotalEnergy:
                        less = (shapeA + lengthA) < (shapeB + lengthB);
                        break;
                }
                return mWaypointSortAscending ? less : !less;
            });

        // Table with radio button, name, shape energy, length energy, total energy, iterations
        ImGuiTableFlags tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
        if (ImGui::BeginTable("ResultsTable", 6, tableFlags, ImVec2(0, 120))) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 15.0f);
            ImGui::TableSetupColumn("Muscle", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Shape E", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("Length E", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("Total E", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("Iters", ImGuiTableColumnFlags_WidthFixed, 30.0f);
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            std::string filterLower(mWaypointResultFilter);
            std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

            for (int sortedIdx : mWaypointSortedIndices) {
                const auto& r = mWaypointOptResults[sortedIdx];

                // Apply filter
                if (!filterLower.empty()) {
                    std::string nameLower = r.muscle_name;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    if (nameLower.find(filterLower) == std::string::npos) {
                        continue;
                    }
                }

                // Color: green=success, orange=success with bound hits, red=failure
                ImVec4 color;
                if (!r.success) {
                    color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);  // Red for failure
                } else if (r.num_bound_hits > 0) {
                    color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);  // Orange for bound hits
                } else {
                    color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green for success
                }

                ImGui::TableNextRow();
                ImGui::PushID(sortedIdx);

                // Column 1: Radio button (clickable)
                ImGui::TableNextColumn();
                bool isSelected = (mWaypointResultSelectedIdx == sortedIdx);
                if (ImGui::RadioButton("##radio", isSelected)) {
                    mWaypointResultSelectedIdx = sortedIdx;
                }

                // Column 2: Muscle name (with bound hit indicator)
                ImGui::TableNextColumn();
                if (r.num_bound_hits > 0) {
                    ImGui::TextColored(color, "%s [%d]", r.muscle_name.c_str(), r.num_bound_hits);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%d waypoint(s) hit displacement bounds.\n"
                                          "Consider increasing max displacement limits.",
                                          r.num_bound_hits);
                    }
                } else {
                    ImGui::TextColored(color, "%s", r.muscle_name.c_str());
                }

                // Select energy values based on before/after toggle
                double shapeE = mWaypointShowAfterEnergy ? r.final_shape_energy : r.initial_shape_energy;
                double lengthE = mWaypointShowAfterEnergy ? r.final_length_energy : r.initial_length_energy;

                // Column 3: Shape energy
                ImGui::TableNextColumn();
                ImGui::TextColored(color, "%.4f", shapeE);

                // Column 4: Length energy
                ImGui::TableNextColumn();
                ImGui::TextColored(color, "%.4f", lengthE);

                // Column 5: Total energy
                ImGui::TableNextColumn();
                ImGui::TextColored(color, "%.4f", shapeE + lengthE);

                // Column 6: Iterations
                ImGui::TableNextColumn();
                ImGui::TextColored(color, "%d", r.num_iterations);

                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        // Compute and display summary for logging/capture
        if (!mWaypointOptResults.empty()) {
            int count = static_cast<int>(mWaypointOptResults.size());

            ImGui::Separator();

            // Line 1: Key optimization parameters (compact)
            ImGui::Text("Params: iter=%d samp=%d lS=%.2f lL=%.2f wP=%.1f wD=%.1f wS=%.1f nPS=%d pow=%d disp=%.3f dispOI=%.3f %s%s%s",
                mWaypointMaxIterations, mWaypointNumSampling,
                mWaypointLambdaShape, mWaypointLambdaLengthCurve,
                mWaypointWeightPhase, mWaypointWeightDelta, mWaypointWeightSamples,
                mWaypointNumPhaseSamples, mWaypointLossPower,
                mWaypointMaxDisplacement, mWaypointMaxDispOriginInsertion,
                mWaypointFixOriginInsertion ? "fixOI " : "",
                mWaypointAnalyticalGradient ? "analGrad " : "",
                mWaypointUseNormalizedLength ? "normLen" : "lmt");

            // Line 2: Character configs (small font)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            float smallFontScale = 0.85f;
            ImGui::SetWindowFontScale(smallFontScale);

            // Extract filenames from paths for compact display
            auto extractFilename = [](const std::string& path) -> std::string {
                size_t lastSlash = path.find_last_of("/\\");
                return (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
            };

            std::string skelFile = extractFilename(mSkeletonPath);
            std::string muscleFile = extractFilename(mMusclePath);
            std::string motionFile = extractFilename(std::string(mHDFPath));

            if (!mCharacterPID.empty() && (mSkeletonDataSource == CharacterDataSource::PatientData ||
                                          mMuscleDataSource == CharacterDataSource::PatientData)) {
                ImGui::Text("Config: pid=%s skel=%s muscle=%s motion=%s",
                    mCharacterPID.c_str(), skelFile.c_str(), muscleFile.c_str(), motionFile.c_str());
            } else {
                ImGui::Text("Config: skel=%s muscle=%s motion=%s",
                    skelFile.c_str(), muscleFile.c_str(), motionFile.c_str());
            }

            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopStyleColor();

            // Line 3: Average metrics (both Before and After) with reduction ratio
            double avgShapeBefore = 0.0, avgLengthBefore = 0.0;
            double avgShapeAfter = 0.0, avgLengthAfter = 0.0;
            for (const auto& r : mWaypointOptResults) {
                avgShapeBefore += r.initial_shape_energy;
                avgLengthBefore += r.initial_length_energy;
                avgShapeAfter += r.final_shape_energy;
                avgLengthAfter += r.final_length_energy;
            }
            if (count > 0) {
                avgShapeBefore /= count; avgLengthBefore /= count;
                avgShapeAfter /= count; avgLengthAfter /= count;
            }
            double totalBefore = avgShapeBefore + avgLengthBefore;
            double totalAfter = avgShapeAfter + avgLengthAfter;
            double shapeRed = (avgShapeBefore > 1e-9) ? (1.0 - avgShapeAfter / avgShapeBefore) * 100.0 : 0.0;
            double lengthRed = (avgLengthBefore > 1e-9) ? (1.0 - avgLengthAfter / avgLengthBefore) * 100.0 : 0.0;
            double totalRed = (totalBefore > 1e-9) ? (1.0 - totalAfter / totalBefore) * 100.0 : 0.0;
            ImGui::Text("Avg[n=%d]: Before(S=%.4f L=%.4f T=%.4f) After(S=%.4f L=%.4f T=%.4f) Red(S=%.1f%% L=%.1f%% T=%.1f%%)",
                count,
                avgShapeBefore, avgLengthBefore, totalBefore,
                avgShapeAfter, avgLengthAfter, totalAfter,
                shapeRed, lengthRed, totalRed);
        }

        ImGui::TreePop();
    }

    if (mWaypointResultSelectedIdx < 0 || mWaypointResultSelectedIdx >= static_cast<int>(mWaypointOptResults.size())) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Select a muscle to view curves.");
        return;
    }

    int selectedIdx = mWaypointResultSelectedIdx;

    const auto& result = mWaypointOptResults[selectedIdx];
    ImGui::Separator();
    ImGui::Text("Muscle: %s", result.muscle_name.c_str());

    // Status indicator
    ImGui::SameLine();
    if (result.success) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "(SUCCESS)");
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "(FAILED)");
    }

    // Characteristics table with 4 columns: Metric, Reference, Subject Before, Subject After
    if (ImGui::BeginTable("Chars", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric");
        ImGui::TableSetupColumn("Reference");
        ImGui::TableSetupColumn("Before");
        ImGui::TableSetupColumn("After");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Min Phase");
        ImGui::TableNextColumn(); ImGui::Text("%.3f", result.reference_chars.min_phase);
        ImGui::TableNextColumn(); ImGui::Text("%.3f", result.subject_before_chars.min_phase);
        ImGui::TableNextColumn(); ImGui::Text("%.3f", result.subject_after_chars.min_phase);

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Max Phase");
        ImGui::TableNextColumn(); ImGui::Text("%.3f", result.reference_chars.max_phase);
        ImGui::TableNextColumn(); ImGui::Text("%.3f", result.subject_before_chars.max_phase);
        ImGui::TableNextColumn(); ImGui::Text("%.3f", result.subject_after_chars.max_phase);

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Delta (m)");
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.reference_chars.delta);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.subject_before_chars.delta);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.subject_after_chars.delta);

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Min Len (m)");
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.reference_chars.min_length);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.subject_before_chars.min_length);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.subject_after_chars.min_length);

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Max Len (m)");
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.reference_chars.max_length);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.subject_before_chars.max_length);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.subject_after_chars.max_length);

        // Mean absolute difference across phase samples
        if (!result.reference_chars.phase_samples.empty()) {
            double mean_diff_before = 0.0;
            double mean_diff_after = 0.0;
            size_t n = result.reference_chars.phase_samples.size();
            for (size_t i = 0; i < n; ++i) {
                mean_diff_before += std::abs(result.subject_before_chars.phase_samples[i]
                                             - result.reference_chars.phase_samples[i]);
                mean_diff_after += std::abs(result.subject_after_chars.phase_samples[i]
                                            - result.reference_chars.phase_samples[i]);
            }
            mean_diff_before /= n;
            mean_diff_after /= n;

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("Sample MAD");
            ImGui::TableNextColumn(); ImGui::Text("-");  // Reference is baseline
            ImGui::TableNextColumn(); ImGui::Text("%.4f", mean_diff_before);
            ImGui::TableNextColumn(); ImGui::Text("%.4f", mean_diff_after);
        }

        ImGui::EndTable();
    }

    // Energy display section
    ImGui::Spacing();
    ImGui::Text("Optimization Energy:");
    if (ImGui::BeginTable("Energy", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Energy Type");
        ImGui::TableSetupColumn("Initial");
        ImGui::TableSetupColumn("Final");
        ImGui::TableSetupColumn("Change");
        ImGui::TableHeadersRow();

        // Shape energy
        double shape_change = result.initial_shape_energy - result.final_shape_energy;
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Shape");
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.initial_shape_energy);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.final_shape_energy);
        ImGui::TableNextColumn();
        if (shape_change > 1e-6) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.4f", shape_change);
        } else if (shape_change < -1e-6) {
            ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "%.4f", shape_change);
        } else {
            ImGui::Text("%.4f", shape_change);
        }

        // Length curve energy
        double length_change = result.initial_length_energy - result.final_length_energy;
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Length");
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.initial_length_energy);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.final_length_energy);
        ImGui::TableNextColumn();
        if (length_change > 1e-6) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.4f", length_change);
        } else if (length_change < -1e-6) {
            ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "%.4f", length_change);
        } else {
            ImGui::Text("%.4f", length_change);
        }

        // Total cost
        double total_change = result.initial_total_cost - result.final_total_cost;
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Total");
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.initial_total_cost);
        ImGui::TableNextColumn(); ImGui::Text("%.4f", result.final_total_cost);
        ImGui::TableNextColumn();
        if (total_change > 1e-6) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.4f", total_change);
        } else if (total_change < -1e-6) {
            ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "%.4f", total_change);
        } else {
            ImGui::Text("%.4f", total_change);
        }

        ImGui::EndTable();
    }

    // ImPlot curve visualization with 3 curves
    if (!result.phases.empty() && !result.reference_lengths.empty()) {
        // Compute Y-axis range from all 3 curves with some padding
        double y_min = std::min({result.reference_chars.min_length,
                                  result.subject_before_chars.min_length,
                                  result.subject_after_chars.min_length}) * 0.95;
        double y_max = std::max({result.reference_chars.max_length,
                                  result.subject_before_chars.max_length,
                                  result.subject_after_chars.max_length}) * 1.05;

        // Build plot title with DOF info
        std::string plotTitle = result.muscle_name;
        if (result.dof_idx >= 0 && !result.dof_name.empty()) {
            plotTitle += " - " + result.dof_name + " (" + std::to_string(result.dof_idx) + ")";
        }
        if (ImPlot::BeginPlot(plotTitle.c_str(), ImVec2(-1, mPlotHeight), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
            const char* yAxisLabel = (result.length_type == PMuscle::LengthCurveType::NORMALIZED)
                ? "Normalized Length (lm_norm)"
                : "MTU Length (lmt)";
            ImPlot::SetupAxes("Phase", yAxisLabel);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImPlotCond_Always);  // X fixed 0-1
            ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always); // Y auto once
            if (!mPlotHideLegend)
                ImPlot::SetupLegend(mPlotLegendEast ? ImPlotLocation_NorthEast : ImPlotLocation_NorthWest);

            // Reference curve (blue) - target behavior from standard character
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
            ImPlot::PlotLine("Reference", result.phases.data(),
                            result.reference_lengths.data(), static_cast<int>(result.phases.size()));
            ImPlot::PopStyleColor();

            // Subject Before curve (orange) - subject's muscle before optimization
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.9f, 0.5f, 0.1f, 1.0f));
            ImPlot::PlotLine("Before", result.phases.data(),
                            result.subject_before_lengths.data(), static_cast<int>(result.phases.size()));
            ImPlot::PopStyleColor();

            // Subject After curve (green) - subject's muscle after optimization
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
            ImPlot::PlotLine("After", result.phases.data(),
                            result.subject_after_lengths.data(), static_cast<int>(result.phases.size()));
            ImPlot::PopStyleColor();

            ImPlot::EndPlot();
        }

        // Shape Angle Plot (Direction Alignment in degrees)
        if (!result.phases.empty() && !result.shape_angle_before.empty()) {
            ImGui::Spacing();

            // Compute Y-axis range with padding
            auto minmax_before = std::minmax_element(
                result.shape_angle_before.begin(), result.shape_angle_before.end());
            auto minmax_after = std::minmax_element(
                result.shape_angle_after.begin(), result.shape_angle_after.end());

            double shape_y_min = std::min(*minmax_before.first, *minmax_after.first) * 0.9;
            double shape_y_max = std::max(*minmax_before.second, *minmax_after.second) * 1.1;
            shape_y_min = std::max(0.0, shape_y_min);  // Angle can't be negative
            shape_y_max = std::max(shape_y_max, 1.0);  // Ensure at least 1 degree range

            if (ImPlot::BeginPlot("Direction Misalignment", ImVec2(-1, mPlotHeight), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
                ImPlot::SetupAxes("Phase", "Angle (degrees)");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, shape_y_min, shape_y_max, ImPlotCond_Always);
                if (!mPlotHideLegend)
                    ImPlot::SetupLegend(mPlotLegendEast ? ImPlotLocation_NorthEast : ImPlotLocation_NorthWest);

                // Before curve (orange)
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.9f, 0.5f, 0.1f, 1.0f));
                ImPlot::PlotLine("Before", result.phases.data(),
                                result.shape_angle_before.data(),
                                static_cast<int>(result.phases.size()));
                ImPlot::PopStyleColor();

                // After curve (green)
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
                ImPlot::PlotLine("After", result.phases.data(),
                                result.shape_angle_after.data(),
                                static_cast<int>(result.phases.size()));
                ImPlot::PopStyleColor();

                ImPlot::EndPlot();
            }
        }
    }
}

// ============================================================
// Tool Operations
// ============================================================

void MusclePersonalizerApp::applyWeightScaling()
{
    if (!mExecutor) {
        LOG_WARN("[MusclePersonalizer] No surgery executor available" );
        return;
    }

    Character* character = mExecutor->getCharacter();
    if (!character) {
        LOG_WARN("[MusclePersonalizer] No character loaded" );
        return;
    }

    if (mTargetMass <= 0.0f || mReferenceMass <= 0.0f) {
        LOG_ERROR("[MusclePersonalizer] Invalid mass values" );
        return;
    }

    // Apply body mass scaling to skeleton
    character->setBodyMass(static_cast<double>(mTargetMass));

    // Compute f0 scaling ratio: f0_new = f0_base * (mass/ref_mass)^(2/3)
    double f0_ratio = std::pow(mTargetMass / mReferenceMass, 2.0 / 3.0);

    // Apply scaling to all muscles
    const std::vector<Muscle*>& muscles = character->getMuscles();
    for (Muscle* muscle : muscles) {
        muscle->change_f(f0_ratio);
    }

    mAppliedRatio = static_cast<float>(f0_ratio);
    mCurrentMass = mTargetMass;

    LOG_INFO("[MusclePersonalizer] Applied weight scaling: "
              << mTargetMass << " kg (f0 ratio: " << f0_ratio << ")" );
}

void MusclePersonalizerApp::runWaypointOptimization()
{
    if (!mExecutor) {
        LOG_WARN("[MusclePersonalizer] No surgery executor available" );
        return;
    }

    auto* character = mExecutor->getCharacter();
    if (!character) {
        LOG_WARN("[MusclePersonalizer] No character loaded" );
        return;
    }

    if (!mReferenceCharacter) {
        LOG_WARN("[MusclePersonalizer] No reference character loaded" );
        return;
    }

    // Check HDF path
    std::string hdfPath(mHDFPath);
    if (hdfPath.empty()) {
        LOG_WARN("[MusclePersonalizer] No HDF file specified" );
        return;
    }

    // Collect selected muscle names from character
    const auto& muscles = character->getMuscles();
    std::vector<std::string> selectedMuscles;
    for (size_t i = 0; i < muscles.size() && i < mSelectedMuscles.size(); ++i) {
        if (mSelectedMuscles[i]) {
            selectedMuscles.push_back(muscles[i]->name);
        }
    }

    if (selectedMuscles.empty()) {
        LOG_WARN("[MusclePersonalizer] No muscles selected for optimization" );
        return;
    }

    LOG_INFO("[MusclePersonalizer] Starting waypoint optimization" );
    LOG_INFO("[MusclePersonalizer]   HDF: " << hdfPath );
    LOG_INFO("[MusclePersonalizer]   Muscles: " << selectedMuscles.size() );

    // Build Config struct from UI parameters
    PMuscle::WaypointOptimizer::Config config;
    config.maxIterations = mWaypointMaxIterations;
    config.numSampling = mWaypointNumSampling;
    config.lambdaShape = static_cast<double>(mWaypointLambdaShape);
    config.lambdaLengthCurve = static_cast<double>(mWaypointLambdaLengthCurve);
    config.fixOriginInsertion = mWaypointFixOriginInsertion;
    config.analyticalGradient = mWaypointAnalyticalGradient;
    config.verbose = mWaypointVerbose;
    config.weightPhase = static_cast<double>(mWaypointWeightPhase);
    config.weightDelta = static_cast<double>(mWaypointWeightDelta);
    config.weightSamples = static_cast<double>(mWaypointWeightSamples);
    config.numPhaseSamples = mWaypointNumPhaseSamples;
    config.lossPower = mWaypointLossPower;
    config.numParallel = mWaypointNumParallel;
    config.lengthType = mWaypointUseNormalizedLength
        ? PMuscle::LengthCurveType::NORMALIZED
        : PMuscle::LengthCurveType::MTU_LENGTH;

    bool success = mExecutor->optimizeWaypoints(
        selectedMuscles, hdfPath, config, mReferenceCharacter, nullptr
    );

    if (success) {
        LOG_INFO("[MusclePersonalizer] Waypoint optimization completed successfully" );
    } else {
        LOG_ERROR("[MusclePersonalizer] Waypoint optimization failed" );
    }
}


void MusclePersonalizerApp::runWaypointOptimizationAsync()
{
    // Don't start if already running
    if (mWaypointOptRunning) {
        LOG_WARN("[MusclePersonalizer] Optimization already running" );
        return;
    }

    // Join any previous thread
    if (mWaypointOptThread && mWaypointOptThread->joinable()) {
        mWaypointOptThread->join();
    }

    if (!mExecutor) {
        LOG_WARN("[MusclePersonalizer] No surgery executor available" );
        return;
    }

    auto* character = mExecutor->getCharacter();
    if (!character) {
        LOG_WARN("[MusclePersonalizer] No character loaded" );
        return;
    }

    if (!mReferenceCharacter) {
        LOG_WARN("[MusclePersonalizer] No reference character loaded" );
        return;
    }

    // Collect selected muscles (must be done on main thread)
    const auto& muscles = character->getMuscles();
    std::vector<std::string> selectedMuscles;
    for (size_t i = 0; i < muscles.size() && i < mSelectedMuscles.size(); ++i) {
        if (mSelectedMuscles[i]) {
            selectedMuscles.push_back(muscles[i]->name);
        }
    }

    if (selectedMuscles.empty()) {
        LOG_WARN("[MusclePersonalizer] No muscles selected" );
        return;
    }

    std::string hdfPath(mHDFPath);
    if (hdfPath.empty()) {
        LOG_WARN("[MusclePersonalizer] No HDF file specified" );
        return;
    }

    // Initialize progress state
    mWaypointOptRunning = true;
    mWaypointOptCurrent = 0;
    mWaypointOptTotal = static_cast<int>(selectedMuscles.size());
    mWaypointOptStartTime = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mWaypointOptMutex);
        mWaypointOptMuscleName = "Starting...";
    }

    // Clear previous results
    {
        std::lock_guard<std::mutex> lock(mWaypointResultsMutex);
        mWaypointOptResults.clear();
        mWaypointResultSelectedIdx = -1;
    }

    // Build Config struct from UI parameters
    PMuscle::WaypointOptimizer::Config config;
    config.maxIterations = mWaypointMaxIterations;
    config.numSampling = mWaypointNumSampling;
    config.lambdaShape = static_cast<double>(mWaypointLambdaShape);
    config.lambdaLengthCurve = static_cast<double>(mWaypointLambdaLengthCurve);
    config.fixOriginInsertion = mWaypointFixOriginInsertion;
    config.analyticalGradient = mWaypointAnalyticalGradient;
    config.verbose = mWaypointVerbose;
    config.weightPhase = static_cast<double>(mWaypointWeightPhase);
    config.weightDelta = static_cast<double>(mWaypointWeightDelta);
    config.weightSamples = static_cast<double>(mWaypointWeightSamples);
    config.numPhaseSamples = mWaypointNumPhaseSamples;
    config.lossPower = mWaypointLossPower;
    config.numParallel = mWaypointNumParallel;
    config.lengthType = mWaypointUseNormalizedLength
        ? PMuscle::LengthCurveType::NORMALIZED
        : PMuscle::LengthCurveType::MTU_LENGTH;
    config.maxDisplacement = static_cast<double>(mWaypointMaxDisplacement);
    config.maxDisplacementOriginInsertion = static_cast<double>(mWaypointMaxDispOriginInsertion);
    config.functionTolerance = static_cast<double>(mWaypointFunctionTolerance);
    config.gradientTolerance = static_cast<double>(mWaypointGradientTolerance);
    config.parameterTolerance = static_cast<double>(mWaypointParameterTolerance);
    config.adaptiveSampleWeight = mWaypointAdaptiveSampleWeight;

    LOG_INFO("[MusclePersonalizer] Starting async waypoint optimization" );
    LOG_INFO("[MusclePersonalizer]   HDF: " << hdfPath );
    LOG_INFO("[MusclePersonalizer]   Muscles: " << selectedMuscles.size() );
    LOG_INFO("[MusclePersonalizer]   Gradient: " << (config.analyticalGradient ? "analytical" : "numeric") );
    LOG_INFO("[MusclePersonalizer]   Weights: phase=" << config.weightPhase << ", delta=" << config.weightDelta
              << ", samples=" << config.weightSamples );
    LOG_INFO("[MusclePersonalizer]   Length type: " << (mWaypointUseNormalizedLength ? "normalized (lm_norm)" : "MTU (lmt)") );

    // Capture reference character pointer for thread
    Character* refChar = mReferenceCharacter;

    // Launch background thread (capture config by copy)
    mWaypointOptThread = std::make_unique<std::thread>([this, selectedMuscles, hdfPath, config, refChar]() {
        auto results = mExecutor->optimizeWaypointsWithResults(
            selectedMuscles, hdfPath, config,
            refChar,  // Reference character for ideal muscle behavior
            &mCharacterMutex,  // Mutex to prevent race with rendering thread
            [this](int current, int total, const std::string& name,
                   const PMuscle::WaypointOptResult& result) {
                // Update progress
                mWaypointOptCurrent = current;
                mWaypointOptTotal = total;
                {
                    std::lock_guard<std::mutex> lock(mWaypointOptMutex);
                    mWaypointOptMuscleName = name;
                }
                // Store result
                {
                    std::lock_guard<std::mutex> lock(mWaypointResultsMutex);
                    mWaypointOptResults.push_back(result);
                }
                // Wake up main thread to update display immediately
                glfwPostEmptyEvent();
            }
        );

        int successCount = 0;
        for (const auto& r : results) {
            if (r.success) successCount++;
        }
        LOG_INFO("[MusclePersonalizer] Waypoint optimization completed: "
                  << successCount << "/" << results.size() << " succeeded" );

        mWaypointOptRunning = false;
    });
}

void MusclePersonalizerApp::runContractureEstimation()
{
    if (!mExecutor || !mExecutor->getCharacter()) {
        LOG_WARN("[MusclePersonalizer] No character loaded" );
        return;
    }

    auto skeleton = mExecutor->getCharacter()->getSkeleton();

    // Collect selected ROM trials with clinical data values
    std::vector<PMuscle::ROMTrialConfig> configs;

    for (const auto& trial : mROMTrials) {
        if (!trial.selected) continue;

        try {
            // Pass skeleton to resolve pose to full VectorXd
            PMuscle::ROMTrialConfig config = PMuscle::ContractureOptimizer::loadROMConfig(
                trial.filePath, skeleton);

            // Populate rom_angle from effective value (normative or patient) or manual input
            auto effectiveValue = getEffectiveROMValue(trial);
            if (effectiveValue.has_value()) {
                // For composite DOF (e.g., abd_knee), angle is always positive
                // Only negate for simple DOF types
                if (config.is_composite_dof) {
                    config.rom_angle = effectiveValue.value();  // Always positive
                } else {
                    config.rom_angle = trial.cd_neg ? -effectiveValue.value() : effectiveValue.value();
                }
                std::string sourceType = (mROMSource == ROMSource::Normative) ? "normative" : "patient";
                LOG_INFO("[MusclePersonalizer] Using " << sourceType << " ROM: " << config.name
                          << " = " << config.rom_angle << "°"
                          << (config.is_composite_dof ? " (composite DOF, no negation)" :
                              (trial.cd_neg ? " (negated)" : "")) );
            } else if (std::abs(trial.manual_rom) > 0.001f) {
                config.rom_angle = static_cast<double>(trial.manual_rom);
                LOG_INFO("[MusclePersonalizer] Using manual ROM: " << config.name
                          << " = " << config.rom_angle << "°" );
            } else {
                LOG_WARN("[MusclePersonalizer] Skipping " << config.name
                          << ": No ROM value available (set clinical data or manual input)" );
                continue;
            }

            // Check cutoff: skip if |rom_angle| > cutoff (no muscle contraction)
            if (trial.cd_cutoff > 0 && std::abs(config.rom_angle) > trial.cd_cutoff) {
                LOG_INFO("[MusclePersonalizer] Skipping " << config.name
                          << ": ROM " << config.rom_angle << "° exceeds cutoff "
                          << trial.cd_cutoff << "° (no contracture)" );
                continue;
            }

            configs.push_back(config);
        } catch (const std::exception& e) {
            LOG_ERROR("[MusclePersonalizer] Failed to load " << trial.filePath
                      << ": " << e.what() );
        }
    }

    if (configs.empty()) {
        LOG_WARN("[MusclePersonalizer] No valid ROM configs (need clinical data or manual ROM values)" );
        return;
    }

    // Store configs for export metadata (name + rom_angle)
    mContractureUsedTrials.clear();
    for (const auto& cfg : configs) {
        mContractureUsedTrials.push_back({cfg.name, cfg.rom_angle});
    }

    // Configure optimizer
    PMuscle::ContractureOptimizer optimizer;

    // Load muscle groups from config (required)
    if (optimizer.loadMuscleGroups(mMuscleGroupsPath, mExecutor->getCharacter()) == 0) {
        LOG_ERROR("[MusclePersonalizer] Failed to load muscle groups from "
                  << mMuscleGroupsPath );
        return;
    }

    // Set grid search mapping for joint multi-group optimization
    optimizer.setGridSearchMapping(mGridSearchMapping);

    // Configure optimization
    PMuscle::ContractureOptimizer::Config optConfig;
    optConfig.maxIterations = mContractureMaxIterations;
    optConfig.minRatio = mContractureMinRatio;
    optConfig.maxRatio = mContractureMaxRatio;
    optConfig.verbose = mContractureVerbose;
    optConfig.gridSearchBegin = mContractureGridBegin;
    optConfig.gridSearchEnd = mContractureGridEnd;
    optConfig.gridSearchInterval = mContractureGridInterval;
    optConfig.lambdaRatioReg = mContractureLambdaRatioReg;
    optConfig.lambdaTorqueReg = mContractureLambdaTorqueReg;
    optConfig.outerIterations = mContractureOuterIterations;

    LOG_INFO("[MusclePersonalizer] Running optimization with results capture..." );

    // Use tiered optimization if available (dual-tier: search groups + optimization groups)
    if (optimizer.hasTieredGroups()) {
        LOG_INFO("[MusclePersonalizer] Using tiered optimization (search + optimization groups)" );
        mTieredContractureResult = optimizer.optimizeWithTieredResults(mExecutor->getCharacter(), configs, optConfig);

        if (!mTieredContractureResult.has_value() || mTieredContractureResult->opt_group_results.empty()) {
            LOG_WARN("[MusclePersonalizer] Tiered optimization returned no results" );
            mTieredContractureResult = std::nullopt;
            return;
        }

        // Convert to local result type for backwards compatibility
        mGroupResults.clear();
        for (const auto& optResult : mTieredContractureResult->opt_group_results) {
            MuscleGroupResult result;
            result.group_name = optResult.group_name;
            result.muscle_names = optResult.muscle_names;
            result.ratio = optResult.ratio;
            result.lm_contract_values = optResult.lm_contract_values;
            mGroupResults.push_back(result);
        }

        // Reset selection indices for visualization tab
        mContractureSelectedGroupIdx = mTieredContractureResult->opt_group_results.empty() ? -1 : 0;
        mContractureSelectedTrialIdx = mTieredContractureResult->trial_results.empty() ? -1 : 0;
        mGridSearchSelectedGroup = mTieredContractureResult->search_group_results.empty() ? -1 : 0;
        mContractureSubTab = 0;  // Start with Grid Search tab

        // Also set mContractureOptResult for legacy code compatibility
        mContractureOptResult = PMuscle::ContractureOptResult();
        mContractureOptResult->group_results = mTieredContractureResult->opt_group_results;
        mContractureOptResult->muscle_results = mTieredContractureResult->muscle_results;
        mContractureOptResult->trial_results = mTieredContractureResult->trial_results;

        LOG_INFO("[MusclePersonalizer] Tiered optimization complete: "
                  << mTieredContractureResult->search_group_results.size() << " search groups, "
                  << mTieredContractureResult->opt_group_results.size() << " opt groups, "
                  << mTieredContractureResult->muscle_results.size() << " muscles" );
    } else {
        // Fall back to standard optimization
        LOG_INFO("[MusclePersonalizer] Using standard optimization (no tiered groups)" );
        mTieredContractureResult = std::nullopt;

        mContractureOptResult = optimizer.optimizeWithResults(mExecutor->getCharacter(), configs, optConfig);

        if (!mContractureOptResult.has_value() || mContractureOptResult->group_results.empty()) {
            LOG_WARN("[MusclePersonalizer] Optimization returned no results" );
            mContractureOptResult = std::nullopt;
            return;
        }

        // Convert to local result type for backwards compatibility
        mGroupResults.clear();
        for (const auto& optResult : mContractureOptResult->group_results) {
            MuscleGroupResult result;
            result.group_name = optResult.group_name;
            result.muscle_names = optResult.muscle_names;
            result.ratio = optResult.ratio;
            result.lm_contract_values = optResult.lm_contract_values;
            mGroupResults.push_back(result);
        }

        // Reset selection indices for visualization tab
        mContractureSelectedGroupIdx = mContractureOptResult->group_results.empty() ? -1 : 0;
        mContractureSelectedTrialIdx = mContractureOptResult->trial_results.empty() ? -1 : 0;

        LOG_INFO("[MusclePersonalizer] Standard optimization complete: "
                  << mGroupResults.size() << " groups, "
                  << mContractureOptResult->muscle_results.size() << " muscles, "
                  << mContractureOptResult->trial_results.size() << " trials" );
    }
}

// ============================================================
// Helper Methods
// ============================================================

void MusclePersonalizerApp::scanROMConfigs()
{
    mROMTrials.clear();

    try {
        // Use resolveDir for directory paths with @ prefix
        fs::path resolved_path = rm::getManager().resolveDir(mROMConfigDir);
        std::string dir = resolved_path.string();

        if (dir.empty() || !fs::exists(dir)) {
            LOG_WARN("[MusclePersonalizer] ROM directory does not exist: " << mROMConfigDir );
            return;
        }

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".yaml") {
                try {
                    YAML::Node config = YAML::LoadFile(entry.path().string());

                    ROMTrialInfo trial;
                    trial.filePath = entry.path().string();
                    trial.name = config["name"].as<std::string>("");
                    trial.description = config["description"].as<std::string>("");
                    trial.joint = config["joint"].as<std::string>("");
                    trial.dof_index = config["dof_index"].as<int>(0);
                    // Torque cutoff (backward compat: try old "torque" key)
                    trial.torque_cutoff = config["torque_cutoff"].as<float>(
                        config["torque"].as<float>(15.0f));

                    // Parse exam section for sweep params
                    if (config["exam"]) {
                        auto exam = config["exam"];
                        trial.angle_min = exam["angle_min"].as<float>(-90.0f);
                        trial.angle_max = exam["angle_max"].as<float>(90.0f);
                        trial.num_steps = exam["num_steps"].as<int>(100);
                        // Normative ROM value (degrees)
                        if (exam["normative"]) {
                            trial.normative = exam["normative"].as<float>();
                        }
                    }

                    // Parse clinical_data section
                    if (config["clinical_data"]) {
                        auto cd = config["clinical_data"];
                        trial.cd_side = cd["side"].as<std::string>("");
                        trial.cd_joint = cd["joint"].as<std::string>("");
                        trial.cd_field = cd["field"].as<std::string>("");
                        trial.cd_neg = cd["neg"].as<bool>(false);
                        trial.cd_cutoff = cd["cutoff"].as<float>(-1.0f);
                    }

                    trial.selected = false;
                    trial.cd_value = std::nullopt;

                    mROMTrials.push_back(trial);
                }
                catch (const std::exception& e) {
                    LOG_ERROR("[MusclePersonalizer] Error parsing " << entry.path() << ": " << e.what() );
                }
            }
        }

        // Update CD values if patient ROM is loaded
        updateROMTrialCDValues();

        // Apply default ROM values for any trials without patient data
        applyDefaultROMValues();

        // Apply default selection from config
        if (!mDefaultSelectedROMTrials.empty()) {
            for (auto& trial : mROMTrials) {
                trial.selected = std::find(mDefaultSelectedROMTrials.begin(),
                                           mDefaultSelectedROMTrials.end(),
                                           trial.name) != mDefaultSelectedROMTrials.end();
            }
        }

        LOG_INFO("[MusclePersonalizer] Found " << mROMTrials.size() << " ROM configs" );
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Error scanning ROM configs: " << e.what() );
    }
}

void MusclePersonalizerApp::applyDefaultROMValues()
{
    // Apply normative ROM values from config files when no patient data is available
    // Source: Moon, Seung Jun, et al. "Normative values of physical examinations
    //         commonly used for cerebral palsy." Yonsei medical journal 58.6 (2017): 1170-1176.
    int applied = 0;
    for (auto& trial : mROMTrials) {
        // Only apply default if no value is already set (don't overwrite patient data)
        if (!trial.cd_value.has_value() && trial.normative.has_value()) {
            trial.cd_value = trial.normative.value();
            applied++;
        }
    }

    if (applied > 0) {
        std::cout << "[MusclePersonalizer] Applied default ROM values to " << applied << " trials" << std::endl;
    }
}

void MusclePersonalizerApp::loadPatientROM(const std::string& pid, const std::string& visit)
{
    mPatientROMValues.clear();
    mCurrentROMPID = pid;

    if (pid.empty() || visit.empty()) return;

    try {
        // New path: @pid:{pid}/{visit}/rom.yaml (flat structure)
        std::string romPath = "@pid:" + pid + "/" + visit + "/rom.yaml";
        std::string resolvedPath = rm::getManager().resolve(romPath);

        if (resolvedPath.empty() || !fs::exists(resolvedPath)) {
            LOG_INFO("[MusclePersonalizer] No ROM data found: " << romPath);
            return;
        }

        YAML::Node romData = YAML::LoadFile(resolvedPath);

        // Flat structure: rom.left.hip.field, rom.right.knee.field, etc.
        if (!romData["rom"]) {
            LOG_INFO("[MusclePersonalizer] No 'rom' section in ROM file: " << romPath);
            return;
        }
        auto romSection = romData["rom"];

        // Parse left and right sides
        for (const std::string& side : {"left", "right"}) {
            if (!romSection[side]) continue;
            auto sideData = romSection[side];

            // Parse each joint (hip, knee, ankle)
            for (const std::string& jointName : {"hip", "knee", "ankle"}) {
                if (!sideData[jointName]) continue;
                auto jointData = sideData[jointName];

                // Iterate through all fields in this joint
                for (auto it = jointData.begin(); it != jointData.end(); ++it) {
                    std::string fieldName = it->first.as<std::string>();
                    std::string key = side + "." + jointName + "." + fieldName;

                    // Store value if it's a number (not null)
                    if (it->second.IsScalar() && !it->second.IsNull()) {
                        try {
                            float value = it->second.as<float>();
                            mPatientROMValues[key] = value;
                        } catch (...) {
                            // Not a float value, skip
                        }
                    }
                }
            }
        }

        LOG_INFO("[MusclePersonalizer] Loaded " << mPatientROMValues.size()
                  << " ROM values for PID " << pid << " (" << visit << ")");

        // Update trial CD values
        updateROMTrialCDValues();
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Error loading patient ROM: " << e.what());
    }
}

void MusclePersonalizerApp::updateROMTrialCDValues()
{
    for (auto& trial : mROMTrials) {
        // Build key from clinical_data
        if (!trial.cd_side.empty() && !trial.cd_joint.empty() && !trial.cd_field.empty()) {
            std::string key = trial.cd_side + "." + trial.cd_joint + "." + trial.cd_field;
            auto it = mPatientROMValues.find(key);
            if (it != mPatientROMValues.end()) {
                trial.cd_value = it->second;
            } else {
                trial.cd_value = std::nullopt;
            }
        } else {
            trial.cd_value = std::nullopt;
        }
    }
}

std::optional<float> MusclePersonalizerApp::getEffectiveROMValue(const ROMTrialInfo& trial) const
{
    // Return appropriate ROM value based on current source selection
    if (mROMSource == ROMSource::Normative) {
        return trial.normative;  // Use normative from config
    } else {
        return trial.cd_value;   // Use patient data (Pre/Op1)
    }
}

void MusclePersonalizerApp::loadClinicalWeight(const std::string& pid, const std::string& visit)
{
    mClinicalWeightAvailable = false;
    mClinicalWeight = 0.0f;

    if (pid.empty()) return;

    try {
        // Load from @pid:{pid}/metadata.yaml
        std::string metaPath = "@pid:" + pid + "/metadata.yaml";
        std::string resolvedPath = rm::getManager().resolve(metaPath);

        if (resolvedPath.empty() || !fs::exists(resolvedPath)) {
            LOG_INFO("[MusclePersonalizer] No metadata found for PID: " << pid);
            return;
        }

        YAML::Node meta = YAML::LoadFile(resolvedPath);

        // Access visit (pre, op1, op2, etc.)
        std::string phaseName = visit;
        if (meta[phaseName] && meta[phaseName]["weight"]) {
            mClinicalWeight = meta[phaseName]["weight"].as<float>();
            mClinicalWeightAvailable = true;
            LOG_INFO("[MusclePersonalizer] Loaded weight: " << mClinicalWeight
                      << " kg (" << phaseName << ")" );

            // Update target mass if using clinical data
            if (mWeightSource == WeightSource::ClinicalData) {
                mTargetMass = mClinicalWeight;
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Error loading clinical weight: " << e.what() );
    }
}

void MusclePersonalizerApp::onPIDChanged(const std::string& pid)
{
    // Reset ROM source to Normative when PID changes
    mROMSource = ROMSource::Normative;
    mPatientROMValues.clear();
    mCurrentROMPID = pid;

    if (pid.empty()) {
        mClinicalWeightAvailable = false;
        return;
    }

    // Get visit from navigator state
    const auto& pidState = mPIDNavigator->getState();
    std::string visit = pidState.getVisitDir();

    // Load clinical weight
    loadClinicalWeight(pid, visit);

    // Auto-select patient skeleton (first available file)
    mCharacterPID = pid;
    mSkeletonDataSource = CharacterDataSource::PatientData;
    scanSkeletonFiles();

    // Auto-select patient muscle (first available file)
    mMuscleDataSource = CharacterDataSource::PatientData;
    scanMuscleFiles();

    // Select first skeleton/muscle files if available and auto-rebuild
    if (!mSkeletonCandidates.empty()) {
        mSkeletonPath = "@pid:" + pid + "/" + visit + "/skeleton/" + mSkeletonCandidates[0];
    }
    if (!mMuscleCandidates.empty()) {
        mMusclePath = "@pid:" + pid + "/" + visit + "/muscle/" + mMuscleCandidates[0];
    }

    if (!mSkeletonCandidates.empty() || !mMuscleCandidates.empty()) {
        loadCharacter();
    }
}

void MusclePersonalizerApp::onVisitChanged(const std::string& pid, const std::string& visit)
{
    if (pid.empty() || visit.empty()) return;

    // Load clinical weight for the new visit
    loadClinicalWeight(pid, visit);

    // Rescan files when visit changes (if using patient data)
    if (mSkeletonDataSource == CharacterDataSource::PatientData) {
        scanSkeletonFiles();
        // Update skeleton path if a file was previously selected
        if (!mSkeletonPath.empty() && !mSkeletonCandidates.empty()) {
            mSkeletonPath = "@pid:" + pid + "/" + visit + "/skeleton/" + mSkeletonCandidates[0];
        }
    }
    if (mMuscleDataSource == CharacterDataSource::PatientData) {
        scanMuscleFiles();
        // Update muscle path if a file was previously selected
        if (!mMusclePath.empty() && !mMuscleCandidates.empty()) {
            mMusclePath = "@pid:" + pid + "/" + visit + "/muscle/" + mMuscleCandidates[0];
        }
    }
}

void MusclePersonalizerApp::scanSkeletonFiles()
{
    mSkeletonCandidates.clear();

    try {
        fs::path skelDir;
        if (mSkeletonDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            // Use patient data directory: {pid_root}/{pid}/{visit}/skeleton/
            fs::path pidRoot = rm::getManager().getPidRoot();
            if (!pidRoot.empty()) {
                std::string visit = mPIDNavigator->getState().getVisitDir();
                skelDir = pidRoot / mCharacterPID / visit / "skeleton";
                if (!fs::exists(skelDir)) return;
            } else {
                LOG_WARN("[MusclePersonalizer] PID root not available");
                return;
            }
        } else {
            // Use default data directory
            skelDir = rm::getManager().resolveDir("@data/skeleton");
            if (!fs::exists(skelDir)) return;
        }

        for (const auto& entry : fs::directory_iterator(skelDir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".yaml" || ext == ".xml") {
                mSkeletonCandidates.push_back(entry.path().filename().string());
            }
        }
        std::sort(mSkeletonCandidates.begin(), mSkeletonCandidates.end());
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Error scanning skeleton files: " << e.what() );
    }
}

void MusclePersonalizerApp::scanMuscleFiles()
{
    mMuscleCandidates.clear();

    try {
        fs::path muscleDir;
        if (mMuscleDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            // Use patient data directory: {pid_root}/{pid}/{visit}/muscle/
            fs::path pidRoot = rm::getManager().getPidRoot();
            if (!pidRoot.empty()) {
                std::string visit = mPIDNavigator->getState().getVisitDir();
                muscleDir = pidRoot / mCharacterPID / visit / "muscle";
                if (!fs::exists(muscleDir)) return;
            } else {
                LOG_WARN("[MusclePersonalizer] PID root not available");
                return;
            }
        } else {
            // Use default data directory
            muscleDir = rm::getManager().resolveDir("@data/muscle");
            if (!fs::exists(muscleDir)) return;
        }

        for (const auto& entry : fs::directory_iterator(muscleDir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".yaml" || ext == ".xml") {
                mMuscleCandidates.push_back(entry.path().filename().string());
            }
        }
        std::sort(mMuscleCandidates.begin(), mMuscleCandidates.end());
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Error scanning muscle files: " << e.what() );
    }
}

void MusclePersonalizerApp::rescanCharacterFiles()
{
    scanSkeletonFiles();
    scanMuscleFiles();

    // Clear current selections when source changes
    mSkeletonPath.clear();
    mMusclePath.clear();
}

void MusclePersonalizerApp::scanMotionFiles()
{
    mMotionCandidates.clear();
    mMotionDOFMap.clear();

    try {
        fs::path motionDir = rm::getManager().resolveDir("@data/motion");
        if (!fs::exists(motionDir)) return;

        for (const auto& entry : fs::directory_iterator(motionDir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".h5" || ext == ".hdf") {
                std::string filename = entry.path().filename().string();
                mMotionCandidates.push_back(filename);

                // Read DOF from motion file
                std::string fullPath = "@data/motion/" + filename;
                std::string resolvedPath = rm::getManager().resolve(fullPath);
                int dof = PMuscle::WaypointOptimizer::getMotionDOF(resolvedPath);
                mMotionDOFMap[filename] = dof;
            }
        }
        std::sort(mMotionCandidates.begin(), mMotionCandidates.end());
    }
    catch (const std::exception& e) {
        LOG_ERROR("[MusclePersonalizer] Error scanning motion files: " << e.what() );
    }
}

void MusclePersonalizerApp::refreshMuscleList()
{
    mAvailableMuscles.clear();
    mSelectedMuscles.clear();

    if (!mExecutor || !mExecutor->getCharacter()) return;

    auto muscles = mExecutor->getCharacter()->getMuscles();
    for (const auto& muscle : muscles) {
        mAvailableMuscles.push_back(muscle->name);
        mSelectedMuscles.push_back(true);
    }

    LOG_INFO("[MusclePersonalizer] Found " << mAvailableMuscles.size() << " muscles" );
}

bool MusclePersonalizerApp::collapsingHeaderWithControls(const std::string& title)
{
    bool isDefaultOpen = isPanelDefaultOpen(title);
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
    if (!isDefaultOpen) {
        flags = ImGuiTreeNodeFlags_None;
    }
    return ImGui::CollapsingHeader(title.c_str(), flags);
}

void MusclePersonalizerApp::drawContractureResultsTab()
{
    if (!mContractureOptResult.has_value()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "No contracture optimization results available.\n"
            "Run optimization in the 'Contracture Estimation' section first.");
        return;
    }

    // Check if we have tiered results (dual-tier: search groups + optimization groups)
    bool hasTiered = mTieredContractureResult.has_value() && mTieredContractureResult->used_tiered;

    if (hasTiered) {
        // Show sub-tabs for tiered results
        if (ImGui::BeginTabBar("ContractureSubTabs")) {
            if (ImGui::BeginTabItem("Grid Search")) {
                mContractureSubTab = 0;
                drawGridSearchSubTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Optimization")) {
                mContractureSubTab = 1;
                drawOptimizationSubTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    } else {
        // Legacy view (non-tiered)
        drawLegacyContractureView();
    }
}


void MusclePersonalizerApp::drawGridSearchSubTab()
{
    if (!mTieredContractureResult.has_value()) return;

    const auto& result = mTieredContractureResult.value();

    // ===== Search Group Selector =====
    ImGui::Text("Search Groups (%zu)", result.search_group_results.size());

    if (ImGui::BeginListBox("##SearchGroupList", ImVec2(-FLT_MIN, 120))) {
        for (size_t i = 0; i < result.search_group_results.size(); ++i) {
            const auto& sr = result.search_group_results[i];
            bool selected = (mGridSearchSelectedGroup == static_cast<int>(i));
            char label[128];
            snprintf(label, sizeof(label), "%s (ratio=%.3f, err=%.1f)",
                     sr.search_group_name.c_str(), sr.ratio, sr.best_error);
            if (ImGui::Selectable(label, selected)) {
                mGridSearchSelectedGroup = static_cast<int>(i);
            }
        }
        ImGui::EndListBox();
    }

    ImGui::Separator();

    // ===== Error Curve Plot for Selected Search Group =====
    if (mGridSearchSelectedGroup >= 0 &&
        mGridSearchSelectedGroup < static_cast<int>(result.search_group_results.size())) {

        const auto& sr = result.search_group_results[mGridSearchSelectedGroup];

        ImGui::Text("Error Curve: %s", sr.search_group_name.c_str());
        ImGui::Text("Best: ratio=%.3f, error=%.2f", sr.ratio, sr.best_error);

        // Show child optimization groups
        if (!sr.opt_group_names.empty()) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Child opt groups:");
            for (const auto& name : sr.opt_group_names) {
                ImGui::BulletText("%s", name.c_str());
            }
        }

        ImGui::Spacing();

        // Plot error vs ratio
        if (!sr.ratios.empty() && sr.ratios.size() == sr.errors.size()) {
            // Find min/max for scaling
            double minErr = *std::min_element(sr.errors.begin(), sr.errors.end());
            double maxErr = *std::max_element(sr.errors.begin(), sr.errors.end());
            double minRatio = sr.ratios.front();
            double maxRatio = sr.ratios.back();
            double bestErr = (sr.best_idx >= 0 && sr.best_idx < static_cast<int>(sr.errors.size()))
                             ? sr.errors[sr.best_idx] : minErr;

            // Y-limit mode radio buttons
            ImGui::RadioButton("Near Best", &mErrorPlotYLimMode, 0); ImGui::SameLine();
            ImGui::RadioButton("Overview", &mErrorPlotYLimMode, 1); ImGui::SameLine();
            ImGui::RadioButton("Draggable", &mErrorPlotYLimMode, 2);

            if (ImPlot::BeginPlot("##ErrorCurve", ImVec2(-1, mPlotHeight))) {
                ImPlot::SetupAxes("Ratio", "Squared Error");

                // Set y-limits based on mode
                if (mErrorPlotYLimMode == 0) {
                    // Near best: 0.9*best ~ 3*best
                    ImPlot::SetupAxesLimits(minRatio - 0.05, maxRatio + 0.05,
                                            bestErr * 0.9, bestErr * 3.0, ImPlotCond_Always);
                } else if (mErrorPlotYLimMode == 1) {
                    // Overview: full range capped at 1e4
                    ImPlot::SetupAxesLimits(minRatio - 0.05, maxRatio + 0.05,
                                            minErr * 0.9, std::min(maxErr * 1.1, 1e4), ImPlotCond_Always);
                } else {
                    // Draggable: x fixed, y draggable
                    ImPlot::SetupAxisLimits(ImAxis_X1, minRatio - 0.05, maxRatio + 0.05, ImPlotCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, minErr * 0.9, std::min(maxErr * 1.1, 1e4), ImPlotCond_Once);
                }

                // Plot error curve
                ImPlot::PlotLine("Error", sr.ratios.data(), sr.errors.data(),
                                static_cast<int>(sr.ratios.size()));

                // Mark best point
                if (sr.best_idx >= 0 && sr.best_idx < static_cast<int>(sr.ratios.size())) {
                    double bx = sr.ratios[sr.best_idx];
                    double by = sr.errors[sr.best_idx];
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8, ImVec4(1, 0.3f, 0.3f, 1), 2);
                    ImPlot::PlotScatter("Best", &bx, &by, 1);
                }

                ImPlot::EndPlot();
            }
        }
    }
}


void MusclePersonalizerApp::drawOptimizationSubTab()
{
    if (!mTieredContractureResult.has_value()) return;

    const auto& result = mTieredContractureResult.value();

    // This is similar to the legacy view but shows opt_group_results
    // ===== Selectors Section =====
    if (ImGui::TreeNodeEx("Optimization Groups", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Count entries outside grid search boundary
        int outsideBoundaryCount = 0;
        for (const auto& grp : result.opt_group_results) {
            if (grp.ratio < mContractureGridBegin || grp.ratio > mContractureGridEnd) {
                outsideBoundaryCount++;
            }
        }

        ImGui::Text("Opt Groups (%zu)", result.opt_group_results.size());
        if (outsideBoundaryCount > 0) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "[%d outside grid %.2f~%.2f]",
                outsideBoundaryCount, mContractureGridBegin, mContractureGridEnd);
        }
        ImGui::InputTextWithHint("##optGrpFilter", "Filter...",
            mContractureGroupFilter, sizeof(mContractureGroupFilter));

        // Create sorted indices by |ratio - 1| descending
        std::vector<size_t> sortedIndices(result.opt_group_results.size());
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&result](size_t a, size_t b) {
                double devA = std::abs(result.opt_group_results[a].ratio - 1.0);
                double devB = std::abs(result.opt_group_results[b].ratio - 1.0);
                return devA > devB;
            });

        if (ImGui::BeginListBox("##OptGroupList", ImVec2(-FLT_MIN, 100))) {
            for (size_t idx : sortedIndices) {
                const auto& grp = result.opt_group_results[idx];
                if (strlen(mContractureGroupFilter) > 0 &&
                    grp.group_name.find(mContractureGroupFilter) == std::string::npos)
                    continue;

                bool selected = (mContractureSelectedGroupIdx == static_cast<int>(idx));
                bool outsideBoundary = (grp.ratio < mContractureGridBegin || grp.ratio > mContractureGridEnd);
                char label[128];
                if (outsideBoundary) {
                    snprintf(label, sizeof(label), "[!] %s (%.3f)", grp.group_name.c_str(), grp.ratio);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
                } else {
                    snprintf(label, sizeof(label), "%s (%.3f)", grp.group_name.c_str(), grp.ratio);
                }
                if (ImGui::Selectable(label, selected)) {
                    mContractureSelectedGroupIdx = static_cast<int>(idx);
                }
                if (outsideBoundary) {
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndListBox();
        }
        ImGui::TreePop();
    }

    ImGui::Separator();

    // Use the legacy view for the rest (charts, etc.) - it uses mContractureOptResult
    // which was populated to match opt_group_results
    drawLegacyContractureView();
}


void MusclePersonalizerApp::drawLegacyContractureView()
{
    if (!mContractureOptResult.has_value()) return;

    const auto& result = mContractureOptResult.value();

    // ===== Selectors Section (collapsible) =====
    if (ImGui::TreeNodeEx("Selectors", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Group selector - sorted by deviation from 1 (largest first)
        ImGui::Text("Muscle Groups (%zu)", result.group_results.size());
        ImGui::InputTextWithHint("##grpFilter", "Filter...",
            mContractureGroupFilter, sizeof(mContractureGroupFilter));

        // Create sorted indices by |ratio - 1| descending
        std::vector<size_t> sortedGroupIndices(result.group_results.size());
        std::iota(sortedGroupIndices.begin(), sortedGroupIndices.end(), 0);
        std::sort(sortedGroupIndices.begin(), sortedGroupIndices.end(),
            [&result](size_t a, size_t b) {
                double devA = std::abs(result.group_results[a].ratio - 1.0);
                double devB = std::abs(result.group_results[b].ratio - 1.0);
                return devA > devB;
            });

        if (ImGui::BeginListBox("##GroupList", ImVec2(-FLT_MIN, 100))) {
            for (size_t idx : sortedGroupIndices) {
                const auto& grp = result.group_results[idx];
                if (strlen(mContractureGroupFilter) > 0 &&
                    grp.group_name.find(mContractureGroupFilter) == std::string::npos)
                    continue;

                bool selected = (mContractureSelectedGroupIdx == static_cast<int>(idx));
                char label[128];
                snprintf(label, sizeof(label), "%s (%.3f)", grp.group_name.c_str(), grp.ratio);
                if (ImGui::Selectable(label, selected)) {
                    mContractureSelectedGroupIdx = static_cast<int>(idx);
                }
            }
            ImGui::EndListBox();
        }

        ImGui::Spacing();

        // Trial selector
        ImGui::Text("ROM Trials (%zu)", result.trial_results.size());
        ImGui::InputTextWithHint("##trialFilter", "Filter...",
            mContractureTrialFilter, sizeof(mContractureTrialFilter));

        if (ImGui::BeginListBox("##TrialList", ImVec2(-FLT_MIN, 150))) {
            for (size_t i = 0; i < result.trial_results.size(); ++i) {
                const auto& trial = result.trial_results[i];
                if (strlen(mContractureTrialFilter) > 0 &&
                    trial.trial_name.find(mContractureTrialFilter) == std::string::npos)
                    continue;

                // Calculate error reduction rate
                double err_before = std::abs(trial.computed_torque_before - trial.observed_torque);
                double err_after = std::abs(trial.computed_torque_after - trial.observed_torque);
                double reduction_rate = (err_before > 1e-9) ? (1.0 - err_after / err_before) * 100.0 : 0.0;

                // Color based on reduction rate: green (>50%), yellow (0-50%), red (<0%)
                ImVec4 color;
                if (reduction_rate > 50.0) {
                    color = ImVec4(0.2f, 0.9f, 0.2f, 1.0f);  // Green
                } else if (reduction_rate > 0.0) {
                    color = ImVec4(0.9f, 0.9f, 0.2f, 1.0f);  // Yellow
                } else {
                    color = ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red
                }

                bool selected = (mContractureSelectedTrialIdx == static_cast<int>(i));
                char label[256];
                snprintf(label, sizeof(label), "%s (%.1f%%)", trial.trial_name.c_str(), reduction_rate);

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                if (ImGui::Selectable(label, selected)) {
                    mContractureSelectedTrialIdx = static_cast<int>(i);
                }
                ImGui::PopStyleColor();
            }
            ImGui::EndListBox();
        }

        ImGui::TreePop();
    }

    ImGui::Separator();

    // Chart visibility checkboxes
    ImGui::Checkbox("lm_contract", &mShowLmContractChart);
    ImGui::SameLine();
    ImGui::Checkbox("Torque", &mShowTorqueChart);
    ImGui::SameLine();
    ImGui::Checkbox("Force", &mShowForceChart);
    ImGui::SameLine();
    ImGui::Checkbox("Group Torque", &mShowGroupTorqueChart);

    // Page navigation (controls all charts) - always visible
    {
        int maxPages = 1;
        if (mContractureSelectedGroupIdx >= 0 &&
            mContractureSelectedGroupIdx < static_cast<int>(result.group_results.size())) {
            const auto& grp = result.group_results[mContractureSelectedGroupIdx];
            int total = static_cast<int>(grp.muscle_names.size());
            maxPages = (total + mPlotBarsPerChart - 1) / mPlotBarsPerChart;
        }
        mContractureChartPage = std::max(0, std::min(mContractureChartPage, maxPages - 1));

        ImGui::BeginDisabled(mContractureChartPage == 0);
        if (ImGui::ArrowButton("##chart_prev", ImGuiDir_Left)) mContractureChartPage--;
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("%d/%d", mContractureChartPage + 1, maxPages);
        ImGui::SameLine();
        ImGui::BeginDisabled(mContractureChartPage >= maxPages - 1);
        if (ImGui::ArrowButton("##chart_next", ImGuiDir_Right)) mContractureChartPage++;
        ImGui::EndDisabled();
    }

    // ===== Chart 1: lm_contract before/after for selected group =====
    if (mShowLmContractChart && mContractureSelectedGroupIdx >= 0 &&
        mContractureSelectedGroupIdx < static_cast<int>(result.group_results.size())) {

        const auto& grp = result.group_results[mContractureSelectedGroupIdx];
        ImGui::Text("lm_contract: %s (ratio=%.3f)", grp.group_name.c_str(), grp.ratio);

        // Collect data for selected group muscles
        std::vector<std::string> labels;
        std::vector<double> before_vals, after_vals;

        for (const auto& m_result : result.muscle_results) {
            // Check if this muscle belongs to selected group
            bool in_group = std::find(grp.muscle_names.begin(), grp.muscle_names.end(),
                                      m_result.muscle_name) != grp.muscle_names.end();
            if (in_group) {
                labels.push_back(m_result.muscle_name);
                before_vals.push_back(m_result.lm_contract_before);
                after_vals.push_back(m_result.lm_contract_after);
            }
        }

        if (!labels.empty()) {
            // Pagination logic (shared between both charts)
            int total = static_cast<int>(labels.size());
            int maxPages = (total + mPlotBarsPerChart - 1) / mPlotBarsPerChart;
            mContractureChartPage = std::max(0, std::min(mContractureChartPage, maxPages - 1));
            int startIdx = mContractureChartPage * mPlotBarsPerChart;
            int endIdx = std::min(startIdx + mPlotBarsPerChart, total);
            int n = endIdx - startIdx;

            // Extract page data
            std::vector<const char*> tick_labels;
            std::vector<double> page_before, page_after;
            for (int i = startIdx; i < endIdx; ++i) {
                tick_labels.push_back(labels[i].c_str());
                page_before.push_back(before_vals[i]);
                page_after.push_back(after_vals[i]);
            }

            // Compute Y range from current page data
            double y_min = 0.0, y_max = 0.0;
            for (size_t i = 0; i < page_before.size(); ++i) {
                y_min = std::min(y_min, std::min(page_before[i], page_after[i]));
                y_max = std::max(y_max, std::max(page_before[i], page_after[i]));
            }
            double y_margin = (y_max - y_min) * 0.2;  // 20% margin for text labels
            y_min -= y_margin;
            y_max += y_margin;

            if (ImPlot::BeginPlot("##lm_contract_chart", ImVec2(-1, 200), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
                ImPlot::SetupAxes("Muscle", "lm_contract (m)");
                if (!mPlotHideLegend)
                    ImPlot::SetupLegend(mPlotLegendEast ? ImPlotLocation_NorthEast : ImPlotLocation_NorthWest);

                double bar_width = 0.35;

                // Fix x-axis to bar width
                ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n - 0.5, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);

                // Generate positions for grouped bars
                std::vector<double> x_before(n), x_after(n);
                for (int i = 0; i < n; ++i) {
                    x_before[i] = i - bar_width / 2.0;
                    x_after[i] = i + bar_width / 2.0;
                }

                ImPlot::SetupAxisTicks(ImAxis_X1, 0, n - 1, n, tick_labels.data());

                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));
                ImPlot::PlotBars("Before", x_before.data(), page_before.data(), n, bar_width);
                for (int i = 0; i < n; ++i) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.3f", page_before[i]);
                    ImPlot::PlotText(buf, x_before[i], page_before[i], ImVec2(0, -5));
                }

                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));
                ImPlot::PlotBars("After", x_after.data(), page_after.data(), n, bar_width);
                for (int i = 0; i < n; ++i) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.3f", page_after[i]);
                    ImPlot::PlotText(buf, x_after[i], page_after[i], ImVec2(0, -5));
                }

                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No muscle data for this group");
        }
    }

    ImGui::Spacing();

    // ===== Chart 2: Per-muscle passive torque at selected trial pose =====
    if (mShowTorqueChart && mContractureSelectedTrialIdx >= 0 &&
        mContractureSelectedTrialIdx < static_cast<int>(result.trial_results.size()) &&
        mContractureSelectedGroupIdx >= 0 &&
        mContractureSelectedGroupIdx < static_cast<int>(result.group_results.size())) {

        const auto& trial = result.trial_results[mContractureSelectedTrialIdx];
        const auto& grp = result.group_results[mContractureSelectedGroupIdx];

        ImGui::Text("Passive Torque at %s: %s", trial.trial_name.c_str(), grp.group_name.c_str());

        // Filter muscle torques to selected group
        std::vector<std::string> labels;
        std::vector<double> before_vals, after_vals;

        for (size_t i = 0; i < trial.muscle_torques_before.size(); ++i) {
            const auto& [name, before_val] = trial.muscle_torques_before[i];
            bool in_group = std::find(grp.muscle_names.begin(), grp.muscle_names.end(),
                                      name) != grp.muscle_names.end();
            if (in_group && i < trial.muscle_torques_after.size()) {
                labels.push_back(name);
                before_vals.push_back(before_val);
                after_vals.push_back(trial.muscle_torques_after[i].second);
            }
        }

        if (!labels.empty()) {
            // Use shared pagination from lm_contract chart
            int total = static_cast<int>(labels.size());
            int startIdx = mContractureChartPage * mPlotBarsPerChart;
            int endIdx = std::min(startIdx + mPlotBarsPerChart, total);
            if (startIdx >= total) {
                startIdx = 0;
                endIdx = std::min(mPlotBarsPerChart, total);
            }
            int n = endIdx - startIdx;

            // Extract page data
            std::vector<const char*> tick_labels;
            std::vector<double> page_before, page_after;
            for (int i = startIdx; i < endIdx; ++i) {
                tick_labels.push_back(labels[i].c_str());
                page_before.push_back(before_vals[i]);
                page_after.push_back(after_vals[i]);
            }

            // Compute Y range from current page data
            double y_min = 0.0, y_max = 0.0;
            for (size_t i = 0; i < page_before.size(); ++i) {
                y_min = std::min(y_min, std::min(page_before[i], page_after[i]));
                y_max = std::max(y_max, std::max(page_before[i], page_after[i]));
            }
            double y_margin = (y_max - y_min) * 0.2;  // 20% margin for text labels
            y_min -= y_margin;
            y_max += y_margin;

            if (ImPlot::BeginPlot("##torque_chart", ImVec2(-1, 200), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
                ImPlot::SetupAxes("Muscle", "Passive Torque (Nm)");
                if (!mPlotHideLegend) ImPlot::SetupLegend(mPlotLegendEast ? ImPlotLocation_NorthEast : ImPlotLocation_NorthWest);

                double bar_width = 0.35;

                // Fix x-axis to bar width
                ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n - 0.5, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);

                std::vector<double> x_before(n), x_after(n);
                for (int i = 0; i < n; ++i) {
                    x_before[i] = i - bar_width / 2.0;
                    x_after[i] = i + bar_width / 2.0;
                }

                ImPlot::SetupAxisTicks(ImAxis_X1, 0, n - 1, n, tick_labels.data());

                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));
                ImPlot::PlotBars("Before", x_before.data(), page_before.data(), n, bar_width);
                for (int i = 0; i < n; ++i) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.1f", page_before[i]);
                    ImPlot::PlotText(buf, x_before[i], page_before[i], ImVec2(0, -5));
                }

                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));
                ImPlot::PlotBars("After", x_after.data(), page_after.data(), n, bar_width);
                for (int i = 0; i < n; ++i) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.1f", page_after[i]);
                    ImPlot::PlotText(buf, x_after[i], page_after[i], ImVec2(0, -5));
                }

                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No torque data for this group/trial");
        }
    }

    ImGui::Spacing();

    // ===== Chart 3: Per-muscle passive force at selected trial pose =====
    if (mShowForceChart && mContractureSelectedTrialIdx >= 0 &&
        mContractureSelectedTrialIdx < static_cast<int>(result.trial_results.size()) &&
        mContractureSelectedGroupIdx >= 0 &&
        mContractureSelectedGroupIdx < static_cast<int>(result.group_results.size())) {

        const auto& trial = result.trial_results[mContractureSelectedTrialIdx];
        const auto& grp = result.group_results[mContractureSelectedGroupIdx];

        ImGui::Text("Passive Force at %s: %s", trial.trial_name.c_str(), grp.group_name.c_str());

        // Filter muscle forces to selected group
        std::vector<std::string> labels;
        std::vector<double> before_vals, after_vals;

        for (size_t i = 0; i < trial.muscle_forces_before.size(); ++i) {
            const auto& [name, before_val] = trial.muscle_forces_before[i];
            bool in_group = std::find(grp.muscle_names.begin(), grp.muscle_names.end(),
                                      name) != grp.muscle_names.end();
            if (in_group && i < trial.muscle_forces_after.size()) {
                labels.push_back(name);
                before_vals.push_back(before_val);
                after_vals.push_back(trial.muscle_forces_after[i].second);
            }
        }

        if (!labels.empty()) {
            // Use shared pagination from lm_contract chart
            int total = static_cast<int>(labels.size());
            int startIdx = mContractureChartPage * mPlotBarsPerChart;
            int endIdx = std::min(startIdx + mPlotBarsPerChart, total);
            if (startIdx >= total) {
                startIdx = 0;
                endIdx = std::min(mPlotBarsPerChart, total);
            }
            int n = endIdx - startIdx;

            // Extract page data
            std::vector<const char*> tick_labels;
            std::vector<double> page_before, page_after;
            for (int i = startIdx; i < endIdx; ++i) {
                tick_labels.push_back(labels[i].c_str());
                page_before.push_back(before_vals[i]);
                page_after.push_back(after_vals[i]);
            }

            // Compute Y range from current page data
            double y_min = 0.0, y_max = 0.0;
            for (size_t i = 0; i < page_before.size(); ++i) {
                y_min = std::min(y_min, std::min(page_before[i], page_after[i]));
                y_max = std::max(y_max, std::max(page_before[i], page_after[i]));
            }
            double y_margin = (y_max - y_min) * 0.2;  // 20% margin for text labels
            y_min -= y_margin;
            y_max += y_margin;

            if (ImPlot::BeginPlot("##force_chart", ImVec2(-1, 200), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
                ImPlot::SetupAxes("Muscle", "Passive Force (N)");
                if (!mPlotHideLegend)
                    ImPlot::SetupLegend(mPlotLegendEast ? ImPlotLocation_NorthEast : ImPlotLocation_NorthWest);

                double bar_width = 0.35;

                // Fix x-axis to bar width
                ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n - 0.5, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);

                std::vector<double> x_before(n), x_after(n);
                for (int i = 0; i < n; ++i) {
                    x_before[i] = i - bar_width / 2.0;
                    x_after[i] = i + bar_width / 2.0;
                }

                ImPlot::SetupAxisTicks(ImAxis_X1, 0, n - 1, n, tick_labels.data());

                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));
                ImPlot::PlotBars("Before", x_before.data(), page_before.data(), n, bar_width);
                for (int i = 0; i < n; ++i) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.1f", page_before[i]);
                    ImPlot::PlotText(buf, x_before[i], page_before[i], ImVec2(0, -5));
                }

                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));
                ImPlot::PlotBars("After", x_after.data(), page_after.data(), n, bar_width);
                for (int i = 0; i < n; ++i) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.1f", page_after[i]);
                    ImPlot::PlotText(buf, x_after[i], page_after[i], ImVec2(0, -5));
                }

                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No force data for this group/trial");
        }
    }

    ImGui::Spacing();

    // ===== Chart 3.5: Group total passive torque (sum of muscles in group) =====
    if (mShowGroupTorqueChart && mContractureSelectedTrialIdx >= 0 &&
        mContractureSelectedTrialIdx < static_cast<int>(result.trial_results.size()) &&
        mContractureSelectedGroupIdx >= 0 &&
        mContractureSelectedGroupIdx < static_cast<int>(result.group_results.size())) {

        const auto& trial = result.trial_results[mContractureSelectedTrialIdx];
        const auto& grp = result.group_results[mContractureSelectedGroupIdx];

        // Sum torques for muscles in selected group
        double sum_before = 0.0, sum_after = 0.0;
        int muscle_count = 0;

        for (size_t i = 0; i < trial.muscle_torques_before.size(); ++i) {
            const auto& [name, before_val] = trial.muscle_torques_before[i];
            bool in_group = std::find(grp.muscle_names.begin(), grp.muscle_names.end(),
                                      name) != grp.muscle_names.end();
            if (in_group && i < trial.muscle_torques_after.size()) {
                sum_before += before_val;
                sum_after += trial.muscle_torques_after[i].second;
                muscle_count++;
            }
        }

        if (muscle_count > 0) {
            ImGui::Text("Group Total Torque: %s (%d muscles)", grp.group_name.c_str(), muscle_count);

            const char* bar_labels[] = {"Before", "After"};
            double values[] = {sum_before, sum_after};

            if (ImPlot::BeginPlot("##group_torque", ImVec2(-1, 150), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
                ImPlot::SetupAxes("", "Torque (Nm)");
                ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, 1.5, ImPlotCond_Always);

                double t_min = std::min(values[0], values[1]);
                double t_max = std::max(values[0], values[1]);
                double t_margin = (t_max - t_min) * 0.15;
                if (t_margin < 0.1) t_margin = 0.1;
                ImPlot::SetupAxisLimits(ImAxis_Y1, t_min - t_margin, t_max + t_margin, ImPlotCond_Always);
                ImPlot::SetupAxisTicks(ImAxis_X1, 0, 1, 2, bar_labels);

                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));  // Blue for before
                ImPlot::PlotBars("##grp_before", &values[0], 1, 0.5, 0);

                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));  // Orange for after
                ImPlot::PlotBars("##grp_after", &values[1], 1, 0.5, 1);

                ImPlot::EndPlot();
            }

            // Show numeric values and change
            double change = sum_after - sum_before;
            double pct_change = (sum_before != 0) ? (change / std::abs(sum_before)) * 100.0 : 0.0;
            ImGui::Text("Before: %.2f Nm | After: %.2f Nm | Change: %+.2f Nm (%.1f%%)",
                       sum_before, sum_after, change, pct_change);
        }
    }

    ImGui::Spacing();

    // ===== Chart 4: Joint total passive torque before/after per trial =====
    if (mContractureSelectedTrialIdx >= 0 &&
        mContractureSelectedTrialIdx < static_cast<int>(result.trial_results.size())) {

        const auto& trial = result.trial_results[mContractureSelectedTrialIdx];

        ImGui::Text("Joint Torque: %s (%s DOF %d)",
                   trial.trial_name.c_str(), trial.joint.c_str(), trial.dof_index);

        // Summary bar: observed vs before vs after
        const char* bar_labels[] = {"Observed", "Before", "After"};
        double values[] = {
            trial.observed_torque,
            trial.computed_torque_before,
            trial.computed_torque_after
        };

        if (ImPlot::BeginPlot("##joint_torque", ImVec2(-1, 150), mPlotHideLegend ? ImPlotFlags_NoLegend : 0)) {
            ImPlot::SetupAxes("", "Torque (Nm)");
            ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, 2.5, ImPlotCond_Always);
            // Set Y limits with margin
            double t_min = std::min({values[0], values[1], values[2]});
            double t_max = std::max({values[0], values[1], values[2]});
            double t_margin = (t_max - t_min) * 0.15;
            ImPlot::SetupAxisLimits(ImAxis_Y1, t_min - t_margin, t_max + t_margin, ImPlotCond_Always);
            ImPlot::SetupAxisTicks(ImAxis_X1, 0, 2, 3, bar_labels);

            // Different colors for each bar
            ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.8f, 0.2f, 0.8f));  // Green for observed
            ImPlot::PlotBars("##observed", &values[0], 1, 0.5, 0);

            ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));  // Blue for before
            ImPlot::PlotBars("##before", &values[1], 1, 0.5, 1);

            ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));  // Orange for after
            ImPlot::PlotBars("##after", &values[2], 1, 0.5, 2);

            ImPlot::EndPlot();
        }

        // Show numeric values
        ImGui::Text("Observed: %.2f Nm | Before: %.2f Nm | After: %.2f Nm",
                   trial.observed_torque, trial.computed_torque_before, trial.computed_torque_after);

        // Error indicator
        double error_before = std::abs(trial.computed_torque_before - trial.observed_torque);
        double error_after = std::abs(trial.computed_torque_after - trial.observed_torque);
        if (error_after < error_before) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                "Improvement: %.2f Nm -> %.2f Nm (%.1f%% reduction)",
                error_before, error_after, (1.0 - error_after / error_before) * 100.0);
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
                "Error: Before %.2f Nm, After %.2f Nm", error_before, error_after);
        }
    }
}

// ============================================================
// Symmetry Analysis
// ============================================================

void MusclePersonalizerApp::computeSymmetryPairs()
{
    mSymmetryPairs.clear();
    mSymmetrySelectedIdx = -1;
    mSymmetryHighlightLeft.clear();
    mSymmetryHighlightRight.clear();

    Character* character = nullptr;
    if (mExecutor && mExecutor->getCharacter()) {
        character = mExecutor->getCharacter();
    }
    if (!character) return;

    auto skel = character->getSkeleton();
    if (!skel) return;

    // Get pelvis X as symmetry plane (same as mirrorAnchorPositions)
    auto* pelvis = skel->getBodyNode("Pelvis");
    double symmetryX = 0.0;
    if (pelvis) {
        symmetryX = pelvis->getTransform().translation().x();
    }

    auto& muscles = character->getMuscles();
    std::map<std::string, Muscle*> leftMuscles, rightMuscles;

    // Group muscles by side (L_ prefix or R_ prefix)
    for (auto* m : muscles) {
        if (m->name.size() > 2 && m->name[1] == '_') {
            std::string base = m->name.substr(2);
            if (m->name[0] == 'L') {
                leftMuscles[base] = m;
            } else if (m->name[0] == 'R') {
                rightMuscles[base] = m;
            }
        }
    }

    // Build pairs by matching base names
    for (auto& [base, leftM] : leftMuscles) {
        auto it = rightMuscles.find(base);
        if (it == rightMuscles.end()) continue;

        Muscle* rightM = it->second;
        SymmetryPairInfo info;
        info.base_name = base;
        info.left_name = leftM->name;
        info.right_name = rightM->name;

        auto& leftAnchors = leftM->GetAnchors();
        auto& rightAnchors = rightM->GetAnchors();

        // Check anchor count
        if (leftAnchors.size() != rightAnchors.size()) {
            info.is_symmetric = false;
            // Mark all as asymmetric with large diff
            for (size_t i = 0; i < std::max(leftAnchors.size(), rightAnchors.size()); i++) {
                info.anchor_symmetric.push_back(false);
                info.anchor_diffs.push_back(999.0);
            }
        } else {
            info.is_symmetric = true;
            for (size_t i = 0; i < leftAnchors.size(); i++) {
                // Get global positions using Anchor::GetPoint()
                Eigen::Vector3d leftGlobal = leftAnchors[i]->GetPoint();
                Eigen::Vector3d rightGlobal = rightAnchors[i]->GetPoint();

                // Mirror right global across pelvis X symmetry plane
                Eigen::Vector3d rightMirrored = rightGlobal;
                rightMirrored.x() = 2.0 * symmetryX - rightGlobal.x();

                // Calculate difference in global space
                double diff = (leftGlobal - rightMirrored).norm();

                info.anchor_diffs.push_back(diff);
                bool anchorSym = diff <= mSymmetryThreshold;
                info.anchor_symmetric.push_back(anchorSym);
                if (!anchorSym) {
                    info.is_symmetric = false;
                }
            }
        }
        mSymmetryPairs.push_back(info);
    }

    // Sort by base_name for consistent ordering
    std::sort(mSymmetryPairs.begin(), mSymmetryPairs.end(),
              [](const SymmetryPairInfo& a, const SymmetryPairInfo& b) {
                  return a.base_name < b.base_name;
              });
}

void MusclePersonalizerApp::drawSymmetryTab()
{
    // Sub-tabs for Skeleton, Muscle waypoint, and Muscle parameter symmetry
    if (ImGui::BeginTabBar("SymmetrySubTabs")) {
        if (ImGui::BeginTabItem("Skeleton")) {
            mSymmetrySubTab = 0;
            drawSkeletonSymmetrySubTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Waypoint")) {
            mSymmetrySubTab = 1;
            drawMuscleSymmetrySubTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Params")) {
            mSymmetrySubTab = 2;
            drawMuscleParamsSymmetrySubTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void MusclePersonalizerApp::computeSkeletonSymmetryPairs()
{
    mSkeletonSymmetryPairs.clear();
    mSkeletonSymmetrySelectedIdx = -1;

    Character* character = nullptr;
    if (mExecutor && mExecutor->getCharacter()) {
        character = mExecutor->getCharacter();
    }
    if (!character) return;

    auto skel = character->getSkeleton();
    std::map<std::string, dart::dynamics::BodyNode*> leftNodes, rightNodes;

    // Group bodynodes by L/R suffix
    for (size_t i = 0; i < skel->getNumBodyNodes(); i++) {
        auto bn = skel->getBodyNode(i);
        std::string name = bn->getName();
        if (name.empty()) continue;
        char lastChar = name.back();
        if (lastChar == 'L') {
            leftNodes[name.substr(0, name.size()-1)] = bn;
        } else if (lastChar == 'R') {
            rightNodes[name.substr(0, name.size()-1)] = bn;
        }
    }

    // Helper lambda for x-mirrored comparison
    auto mirrorX = [](const Eigen::Vector3d& v) -> Eigen::Vector3d {
        return Eigen::Vector3d(-v.x(), v.y(), v.z());
    };

    // Build pairs
    for (auto& [base, leftBN] : leftNodes) {
        auto it = rightNodes.find(base);
        if (it == rightNodes.end()) continue;

        auto rightBN = it->second;
        SkeletonSymmetryPairInfo info;
        info.base_name = base;
        info.left_name = leftBN->getName();
        info.right_name = rightBN->getName();

        // Get BodyNode world positions
        info.left_bn_pos = leftBN->getWorldTransform().translation();
        info.right_bn_pos = rightBN->getWorldTransform().translation();
        info.bn_pos_diff = (info.left_bn_pos - mirrorX(info.right_bn_pos)).norm();

        // Get joints
        auto leftJoint = leftBN->getParentJoint();
        auto rightJoint = rightBN->getParentJoint();

        info.joint_diff = 0.0;
        info.joint_pos_diff = 0.0;
        info.parent_to_joint_diff = 0.0;
        info.child_to_joint_diff = 0.0;
        info.left_joint_pos.setZero();
        info.right_joint_pos.setZero();
        info.left_parent_to_joint.setZero();
        info.right_parent_to_joint.setZero();
        info.left_child_to_joint.setZero();
        info.right_child_to_joint.setZero();

        if (leftJoint && rightJoint) {
            // Joint DOFs
            info.left_dofs = leftJoint->getPositions();
            info.right_dofs = rightJoint->getPositions();
            if (info.left_dofs.size() == info.right_dofs.size()) {
                for (int i = 0; i < info.left_dofs.size(); i++) {
                    double diff = std::abs(info.left_dofs[i] - info.right_dofs[i]);
                    info.joint_diff = std::max(info.joint_diff, diff);
                }
            }

            // Joint world position (use joint's world transform)
            // Joint position can be computed from parent bodynode + TransformFromParentBodyNode
            auto leftParentBN = leftJoint->getParentBodyNode();
            auto rightParentBN = rightJoint->getParentBodyNode();

            if (leftParentBN && rightParentBN) {
                Eigen::Isometry3d leftParentToJoint = leftJoint->getTransformFromParentBodyNode();
                Eigen::Isometry3d rightParentToJoint = rightJoint->getTransformFromParentBodyNode();

                // Joint world position = parent world transform * parentToJoint
                info.left_joint_pos = (leftParentBN->getWorldTransform() * leftParentToJoint).translation();
                info.right_joint_pos = (rightParentBN->getWorldTransform() * rightParentToJoint).translation();
                info.joint_pos_diff = (info.left_joint_pos - mirrorX(info.right_joint_pos)).norm();

                // Store relative transform translations
                info.left_parent_to_joint = leftParentToJoint.translation();
                info.right_parent_to_joint = rightParentToJoint.translation();
                info.parent_to_joint_diff = (info.left_parent_to_joint - mirrorX(info.right_parent_to_joint)).norm();
            }

            // TransformFromChildBodyNode
            Eigen::Isometry3d leftChildToJoint = leftJoint->getTransformFromChildBodyNode();
            Eigen::Isometry3d rightChildToJoint = rightJoint->getTransformFromChildBodyNode();
            info.left_child_to_joint = leftChildToJoint.translation();
            info.right_child_to_joint = rightChildToJoint.translation();
            info.child_to_joint_diff = (info.left_child_to_joint - mirrorX(info.right_child_to_joint)).norm();
        }

        // Use maximum of all position differences for symmetry check
        double maxDiff = std::max({info.bn_pos_diff, info.joint_pos_diff,
                                   info.parent_to_joint_diff, info.child_to_joint_diff});
        info.is_symmetric = (maxDiff <= mSymmetryThreshold);
        mSkeletonSymmetryPairs.push_back(info);
    }

    // Sort by base_name
    std::sort(mSkeletonSymmetryPairs.begin(), mSkeletonSymmetryPairs.end(),
              [](const SkeletonSymmetryPairInfo& a, const SkeletonSymmetryPairInfo& b) {
                  return a.base_name < b.base_name;
              });
}

void MusclePersonalizerApp::drawSkeletonSymmetrySubTab()
{
    // Threshold controls
    ImGui::Text("Threshold:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("##SkelSymThreshold", &mSymmetryThreshold, 0.0001f, 0.001f, "%.4f")) {
        mSymmetryThreshold = std::max(0.0001f, mSymmetryThreshold);
        computeSkeletonSymmetryPairs();
    }
    ImGui::SameLine();
    ImGui::Text("m");
    ImGui::SameLine();
    if (ImGui::Button("Refresh##SkelSym")) {
        computeSkeletonSymmetryPairs();
    }

    ImGui::Separator();

    // Statistics
    int asymmetricCount = 0;
    for (const auto& pair : mSkeletonSymmetryPairs) {
        if (!pair.is_symmetric) asymmetricCount++;
    }
    int totalPairs = static_cast<int>(mSkeletonSymmetryPairs.size());
    float percentage = totalPairs > 0 ? (asymmetricCount * 100.0f / totalPairs) : 0.0f;

    if (asymmetricCount > 0) {
        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f),
            "Asymmetric: %d / %d (%.1f%%)", asymmetricCount, totalPairs, percentage);
    } else {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            "All symmetric: %d pairs", totalPairs);
    }

    ImGui::Separator();

    // BodyNode pairs listbox
    ImGui::Text("BodyNode Pairs:");
    float listHeight = 200.0f;
    if (ImGui::BeginListBox("##SkelSymList", ImVec2(-FLT_MIN, listHeight))) {
        for (int i = 0; i < static_cast<int>(mSkeletonSymmetryPairs.size()); i++) {
            const auto& pair = mSkeletonSymmetryPairs[i];

            // Color based on symmetry status
            ImVec4 color = pair.is_symmetric
                ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)   // Green for symmetric
                : ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red for asymmetric

            ImGui::PushStyleColor(ImGuiCol_Text, color);

            bool isSelected = (mSkeletonSymmetrySelectedIdx == i);
            if (ImGui::Selectable(pair.base_name.c_str(), isSelected)) {
                mSkeletonSymmetrySelectedIdx = i;
            }

            ImGui::PopStyleColor();
        }
        ImGui::EndListBox();
    }

    ImGui::Separator();

    // Selected pair details
    if (mSkeletonSymmetrySelectedIdx >= 0 && mSkeletonSymmetrySelectedIdx < static_cast<int>(mSkeletonSymmetryPairs.size())) {
        const auto& pair = mSkeletonSymmetryPairs[mSkeletonSymmetrySelectedIdx];

        ImGui::Text("Selected: %s", pair.base_name.c_str());
        ImGui::Text("  L: %s   R: %s", pair.left_name.c_str(), pair.right_name.c_str());

        ImGui::Separator();

        // Scrollable details
        float detailsHeight = std::min(500.0f, ImGui::GetContentRegionAvail().y - 10);
        if (ImGui::BeginChild("SkelSymDetails", ImVec2(0, detailsHeight), true)) {
            ImGui::SetWindowFontScale(1.3f);

            // Helper lambda to get color based on diff value
            auto getDiffColor = [this](double diff) -> ImVec4 {
                if (diff <= mSymmetryThreshold) {
                    return ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
                } else {
                    return ImVec4(0.9f, 0.4f, 0.4f, 1.0f);  // Red
                }
            };

            // =============================================
            // 1. BodyNode World Positions
            // =============================================
            ImGui::Text("BodyNode World Positions:");
            ImVec4 bnColor = getDiffColor(pair.bn_pos_diff);
            ImGui::TextColored(bnColor, "  L: (%.4f, %.4f, %.4f)",
                pair.left_bn_pos.x(), pair.left_bn_pos.y(), pair.left_bn_pos.z());
            ImGui::TextColored(bnColor, "  R: (%.4f, %.4f, %.4f)",
                pair.right_bn_pos.x(), pair.right_bn_pos.y(), pair.right_bn_pos.z());
            ImGui::TextColored(bnColor, "  R mirrored: (%.4f, %.4f, %.4f)",
                -pair.right_bn_pos.x(), pair.right_bn_pos.y(), pair.right_bn_pos.z());
            ImGui::TextColored(getDiffColor(pair.bn_pos_diff), "  Diff: %.6f m", pair.bn_pos_diff);

            ImGui::Separator();

            // =============================================
            // 2. Joint World Positions
            // =============================================
            ImGui::Text("Joint World Positions:");
            ImVec4 jointPosColor = getDiffColor(pair.joint_pos_diff);
            ImGui::TextColored(jointPosColor, "  L: (%.4f, %.4f, %.4f)",
                pair.left_joint_pos.x(), pair.left_joint_pos.y(), pair.left_joint_pos.z());
            ImGui::TextColored(jointPosColor, "  R: (%.4f, %.4f, %.4f)",
                pair.right_joint_pos.x(), pair.right_joint_pos.y(), pair.right_joint_pos.z());
            ImGui::TextColored(jointPosColor, "  R mirrored: (%.4f, %.4f, %.4f)",
                -pair.right_joint_pos.x(), pair.right_joint_pos.y(), pair.right_joint_pos.z());
            ImGui::TextColored(getDiffColor(pair.joint_pos_diff), "  Diff: %.6f m", pair.joint_pos_diff);

            ImGui::Separator();

            // =============================================
            // 3. TransformFromParentBodyNode (relative)
            // =============================================
            ImGui::Text("Parent-to-Joint Transform (tx):");
            ImVec4 parentColor = getDiffColor(pair.parent_to_joint_diff);
            ImGui::TextColored(parentColor, "  L: (%.4f, %.4f, %.4f)",
                pair.left_parent_to_joint.x(), pair.left_parent_to_joint.y(), pair.left_parent_to_joint.z());
            ImGui::TextColored(parentColor, "  R: (%.4f, %.4f, %.4f)",
                pair.right_parent_to_joint.x(), pair.right_parent_to_joint.y(), pair.right_parent_to_joint.z());
            ImGui::TextColored(parentColor, "  R mirrored: (%.4f, %.4f, %.4f)",
                -pair.right_parent_to_joint.x(), pair.right_parent_to_joint.y(), pair.right_parent_to_joint.z());
            ImGui::TextColored(getDiffColor(pair.parent_to_joint_diff), "  Diff: %.6f m", pair.parent_to_joint_diff);

            ImGui::Separator();

            // =============================================
            // 4. TransformFromChildBodyNode (relative)
            // =============================================
            ImGui::Text("Child-to-Joint Transform (tx):");
            ImVec4 childColor = getDiffColor(pair.child_to_joint_diff);
            ImGui::TextColored(childColor, "  L: (%.4f, %.4f, %.4f)",
                pair.left_child_to_joint.x(), pair.left_child_to_joint.y(), pair.left_child_to_joint.z());
            ImGui::TextColored(childColor, "  R: (%.4f, %.4f, %.4f)",
                pair.right_child_to_joint.x(), pair.right_child_to_joint.y(), pair.right_child_to_joint.z());
            ImGui::TextColored(childColor, "  R mirrored: (%.4f, %.4f, %.4f)",
                -pair.right_child_to_joint.x(), pair.right_child_to_joint.y(), pair.right_child_to_joint.z());
            ImGui::TextColored(getDiffColor(pair.child_to_joint_diff), "  Diff: %.6f m", pair.child_to_joint_diff);

            ImGui::Separator();

            // =============================================
            // 5. Joint DOF comparison
            // =============================================
            ImGui::Text("Joint DOFs:");
            if (pair.left_dofs.size() > 0) {
                std::string lDofsStr = "  L: [";
                for (int i = 0; i < pair.left_dofs.size(); i++) {
                    if (i > 0) lDofsStr += ", ";
                    lDofsStr += std::to_string(pair.left_dofs[i]).substr(0, 7);
                }
                lDofsStr += "]";
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", lDofsStr.c_str());

                std::string rDofsStr = "  R: [";
                for (int i = 0; i < pair.right_dofs.size(); i++) {
                    if (i > 0) rDofsStr += ", ";
                    rDofsStr += std::to_string(pair.right_dofs[i]).substr(0, 7);
                }
                rDofsStr += "]";
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", rDofsStr.c_str());

                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "  Max DOF diff: %.6f rad", pair.joint_diff);
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "  (No DOFs)");
            }

            ImGui::SetWindowFontScale(1.0f);
        }
        ImGui::EndChild();
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Select a bodynode pair to view details");
    }
}

void MusclePersonalizerApp::drawMuscleSymmetrySubTab()
{
    // Get character for accessing muscle data
    Character* character = nullptr;
    if (mExecutor && mExecutor->getCharacter()) {
        character = mExecutor->getCharacter();
    }

    // Filter and threshold controls
    ImGui::Text("Filter:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputText("##SymFilter", mSymmetryFilter, sizeof(mSymmetryFilter))) {
        // Filter changed - no action needed, filtering is done during display
    }
    ImGui::SameLine();
    ImGui::Text("Threshold:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("##SymThreshold", &mSymmetryThreshold, 0.0001f, 0.001f, "%.4f")) {
        mSymmetryThreshold = std::max(0.0001f, mSymmetryThreshold);
        computeSymmetryPairs();  // Recompute with new threshold
    }
    ImGui::SameLine();
    ImGui::Text("m");
    ImGui::SameLine();
    if (ImGui::Button("Refresh##Sym")) {
        computeSymmetryPairs();
    }

    ImGui::Separator();

    // Statistics
    int asymmetricCount = 0;
    for (const auto& pair : mSymmetryPairs) {
        if (!pair.is_symmetric) asymmetricCount++;
    }
    int totalPairs = static_cast<int>(mSymmetryPairs.size());
    float percentage = totalPairs > 0 ? (asymmetricCount * 100.0f / totalPairs) : 0.0f;

    if (asymmetricCount > 0) {
        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f),
            "Asymmetric: %d / %d (%.1f%%)", asymmetricCount, totalPairs, percentage);
    } else {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            "All symmetric: %d pairs", totalPairs);
    }

    ImGui::Separator();

    // Muscle pairs listbox
    ImGui::Text("Muscle Pairs:");
    float listHeight = 250.0f;
    if (ImGui::BeginListBox("##SymmetryList", ImVec2(-FLT_MIN, listHeight))) {
        std::string filterLower = mSymmetryFilter;
        std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

        for (int i = 0; i < static_cast<int>(mSymmetryPairs.size()); i++) {
            const auto& pair = mSymmetryPairs[i];

            // Apply filter
            if (!filterLower.empty()) {
                std::string nameLower = pair.base_name;
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                if (nameLower.find(filterLower) == std::string::npos) continue;
            }

            // Color based on symmetry status
            ImVec4 color = pair.is_symmetric
                ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)   // Green for symmetric
                : ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red for asymmetric

            ImGui::PushStyleColor(ImGuiCol_Text, color);

            bool isSelected = (mSymmetrySelectedIdx == i);
            if (ImGui::Selectable(pair.base_name.c_str(), isSelected)) {
                mSymmetrySelectedIdx = i;
                // Update highlight for 3D rendering
                mSymmetryHighlightLeft = pair.left_name;
                mSymmetryHighlightRight = pair.right_name;
            }

            ImGui::PopStyleColor();
        }
        ImGui::EndListBox();
    }

    ImGui::Separator();

    // Selected pair details
    if (mSymmetrySelectedIdx >= 0 && mSymmetrySelectedIdx < static_cast<int>(mSymmetryPairs.size())) {
        const auto& selectedPair = mSymmetryPairs[mSymmetrySelectedIdx];

        ImGui::Text("Selected: %s", selectedPair.base_name.c_str());
        ImGui::Text("  L: %s", selectedPair.left_name.c_str());
        ImGui::Text("  R: %s", selectedPair.right_name.c_str());

        ImGui::Separator();
        ImGui::Text("Anchor Comparison:");

        // Get muscle pointers for position display
        Muscle* leftM = character ? character->getMuscleByName(selectedPair.left_name) : nullptr;
        Muscle* rightM = character ? character->getMuscleByName(selectedPair.right_name) : nullptr;

        if (leftM && rightM) {
            auto& leftAnchors = leftM->GetAnchors();
            auto& rightAnchors = rightM->GetAnchors();

            // Scrollable region for anchor details
            float anchorListHeight = std::min(500.0f, ImGui::GetContentRegionAvail().y - 20);
            if (ImGui::BeginChild("AnchorDetails", ImVec2(0, anchorListHeight), true)) {
                // Increase font size for anchor comparison
                ImGui::SetWindowFontScale(1.4f);

                for (size_t i = 0; i < selectedPair.anchor_diffs.size(); i++) {
                    bool anchorSym = selectedPair.anchor_symmetric[i];
                    double diff = selectedPair.anchor_diffs[i];

                    // Anchor header with status
                    if (anchorSym) {
                        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                            "Anchor %zu: OK", i);
                    } else {
                        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f),
                            "Anchor %zu: DIFF (%.4f m)", i, diff);
                    }

                    // Show positions and bodynode names if available
                    if (i < leftAnchors.size() && i < rightAnchors.size()) {
                        auto& lpos = leftAnchors[i]->local_positions;
                        auto& rpos = rightAnchors[i]->local_positions;
                        auto& lbodies = leftAnchors[i]->bodynodes;
                        auto& rbodies = rightAnchors[i]->bodynodes;

                        for (size_t j = 0; j < std::max(lpos.size(), rpos.size()); j++) {
                            ImVec4 textColor = anchorSym
                                ? ImVec4(0.7f, 0.7f, 0.7f, 1.0f)   // Gray for matching
                                : ImVec4(0.9f, 0.4f, 0.4f, 1.0f);  // Red for different

                            if (j < lpos.size()) {
                                std::string lBodyName = (j < lbodies.size() && lbodies[j])
                                    ? lbodies[j]->getName() : "?";
                                ImGui::TextColored(textColor, "    L[%zu] %s: (%.4f, %.4f, %.4f)",
                                    j, lBodyName.c_str(), lpos[j].x(), lpos[j].y(), lpos[j].z());
                            }
                            if (j < rpos.size()) {
                                std::string rBodyName = (j < rbodies.size() && rbodies[j])
                                    ? rbodies[j]->getName() : "?";
                                // Show mirrored value for comparison
                                ImGui::TextColored(textColor, "    R[%zu] %s: (%.4f, %.4f, %.4f) -> mirrored: (%.4f, %.4f, %.4f)",
                                    j, rBodyName.c_str(), rpos[j].x(), rpos[j].y(), rpos[j].z(),
                                    -rpos[j].x(), rpos[j].y(), rpos[j].z());
                            }
                        }
                    }

                    if (i < selectedPair.anchor_diffs.size() - 1) {
                        ImGui::Separator();
                    }
                }

                // Reset font scale
                ImGui::SetWindowFontScale(1.0f);
            }
            ImGui::EndChild();
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Muscle data not available");
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Select a muscle pair to view details");
    }

    // Clear highlight button
    if (!mSymmetryHighlightLeft.empty()) {
        ImGui::Separator();
        if (ImGui::Button("Clear 3D Highlight")) {
            mSymmetrySelectedIdx = -1;
            mSymmetryHighlightLeft.clear();
            mSymmetryHighlightRight.clear();
        }
    }
}

void MusclePersonalizerApp::computeMuscleParamsSymmetryPairs()
{
    mMuscleParamSymmetryPairs.clear();
    mMuscleParamSymmetrySelectedIdx = -1;

    Character* character = nullptr;
    if (mExecutor && mExecutor->getCharacter()) {
        character = mExecutor->getCharacter();
    }
    if (!character) return;

    auto& muscles = character->getMuscles();

    // Group muscles by L_/R_ prefix
    std::map<std::string, Muscle*> leftMuscles, rightMuscles;
    for (auto* m : muscles) {
        const std::string& name = m->name;
        if (name.size() < 2) continue;
        if (name.substr(0, 2) == "L_") {
            leftMuscles[name.substr(2)] = m;
        } else if (name.substr(0, 2) == "R_") {
            rightMuscles[name.substr(2)] = m;
        }
    }

    // Helper to compute percentage difference
    auto pctDiff = [](double a, double b) -> double {
        double avg = (std::abs(a) + std::abs(b)) / 2.0;
        if (avg < 1e-9) return 0.0;
        return std::abs(a - b) / avg * 100.0;
    };

    // Build pairs
    for (auto& [baseName, leftM] : leftMuscles) {
        auto it = rightMuscles.find(baseName);
        if (it == rightMuscles.end()) continue;

        Muscle* rightM = it->second;

        MuscleParamSymmetryPairInfo info;
        info.base_name = baseName;
        info.left_name = leftM->name;
        info.right_name = rightM->name;

        // Get parameter values
        info.left_f0 = leftM->f0;
        info.right_f0 = rightM->f0;
        info.left_lm_contract = leftM->lm_contract;
        info.right_lm_contract = rightM->lm_contract;
        info.left_lt_rel = leftM->lt_rel;
        info.right_lt_rel = rightM->lt_rel;

        // Compute percentage differences
        info.f0_diff_pct = pctDiff(info.left_f0, info.right_f0);
        info.lm_contract_diff_pct = pctDiff(info.left_lm_contract, info.right_lm_contract);
        info.lt_rel_diff_pct = pctDiff(info.left_lt_rel, info.right_lt_rel);

        info.max_diff_pct = std::max({info.f0_diff_pct, info.lm_contract_diff_pct, info.lt_rel_diff_pct});
        info.is_symmetric = (info.max_diff_pct <= mMuscleParamSymmetryThreshold);

        mMuscleParamSymmetryPairs.push_back(info);
    }

    // Sort by base_name
    std::sort(mMuscleParamSymmetryPairs.begin(), mMuscleParamSymmetryPairs.end(),
              [](const auto& a, const auto& b) { return a.base_name < b.base_name; });
}

void MusclePersonalizerApp::drawMuscleParamsSymmetrySubTab()
{
    // Threshold controls
    ImGui::Text("Threshold:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputFloat("##ParamSymThreshold", &mMuscleParamSymmetryThreshold, 1.0f, 5.0f, "%.1f%%")) {
        mMuscleParamSymmetryThreshold = std::max(0.1f, mMuscleParamSymmetryThreshold);
        computeMuscleParamsSymmetryPairs();
    }
    ImGui::SameLine();
    if (ImGui::Button("Refresh##ParamSym")) {
        computeMuscleParamsSymmetryPairs();
    }
    ImGui::SameLine();
    ImGui::Text("Filter:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##ParamSymFilter", mMuscleParamSymmetryFilter, sizeof(mMuscleParamSymmetryFilter));

    ImGui::Separator();

    // Statistics
    int asymmetricCount = 0;
    for (const auto& pair : mMuscleParamSymmetryPairs) {
        if (!pair.is_symmetric) asymmetricCount++;
    }
    int totalPairs = static_cast<int>(mMuscleParamSymmetryPairs.size());
    float percentage = totalPairs > 0 ? (asymmetricCount * 100.0f / totalPairs) : 0.0f;

    if (asymmetricCount > 0) {
        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f),
            "Asymmetric: %d / %d (%.1f%%)", asymmetricCount, totalPairs, percentage);
    } else {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            "All symmetric: %d pairs", totalPairs);
    }

    ImGui::Separator();

    // Muscle pairs listbox
    ImGui::Text("Muscle Pairs (by max param diff):");
    float listHeight = 200.0f;
    if (ImGui::BeginListBox("##ParamSymmetryList", ImVec2(-FLT_MIN, listHeight))) {
        // Sort by max_diff_pct descending for display
        std::vector<size_t> sortedIndices(mMuscleParamSymmetryPairs.size());
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
            [this](size_t a, size_t b) {
                return mMuscleParamSymmetryPairs[a].max_diff_pct > mMuscleParamSymmetryPairs[b].max_diff_pct;
            });

        // Convert filter to lowercase for case-insensitive matching
        std::string filterLower = mMuscleParamSymmetryFilter;
        std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

        for (size_t idx : sortedIndices) {
            const auto& pair = mMuscleParamSymmetryPairs[idx];

            // Apply filter
            if (!filterLower.empty()) {
                std::string nameLower = pair.base_name;
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                if (nameLower.find(filterLower) == std::string::npos) continue;
            }

            // Color based on symmetry status
            ImVec4 color = pair.is_symmetric
                ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)   // Green for symmetric
                : ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red for asymmetric

            ImGui::PushStyleColor(ImGuiCol_Text, color);

            char label[128];
            snprintf(label, sizeof(label), "%s (%.1f%%)", pair.base_name.c_str(), pair.max_diff_pct);

            bool isSelected = (mMuscleParamSymmetrySelectedIdx == static_cast<int>(idx));
            if (ImGui::Selectable(label, isSelected)) {
                mMuscleParamSymmetrySelectedIdx = static_cast<int>(idx);
                // Update highlight for 3D rendering
                mSymmetryHighlightLeft = pair.left_name;
                mSymmetryHighlightRight = pair.right_name;
            }

            ImGui::PopStyleColor();
        }
        ImGui::EndListBox();
    }

    ImGui::Separator();

    // Selected pair details
    if (mMuscleParamSymmetrySelectedIdx >= 0 &&
        mMuscleParamSymmetrySelectedIdx < static_cast<int>(mMuscleParamSymmetryPairs.size())) {
        const auto& pair = mMuscleParamSymmetryPairs[mMuscleParamSymmetrySelectedIdx];

        ImGui::Text("Selected: %s", pair.base_name.c_str());
        ImGui::Text("  L: %s", pair.left_name.c_str());
        ImGui::Text("  R: %s", pair.right_name.c_str());

        ImGui::Separator();

        // Parameter comparison table
        ImGui::SetWindowFontScale(1.2f);
        if (ImGui::BeginTable("ParamCompare", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Left", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Right", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Diff %", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableHeadersRow();

            auto drawRow = [&](const char* name, double left, double right, double diff_pct) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", name);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", left);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", right);
                ImGui::TableNextColumn();
                ImVec4 color = (diff_pct <= mMuscleParamSymmetryThreshold)
                    ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                    : ImVec4(0.9f, 0.2f, 0.2f, 1.0f);
                ImGui::TextColored(color, "%.1f", diff_pct);
            };

            drawRow("f0", pair.left_f0, pair.right_f0, pair.f0_diff_pct);
            drawRow("lm_contract", pair.left_lm_contract, pair.right_lm_contract, pair.lm_contract_diff_pct);
            drawRow("lt_rel", pair.left_lt_rel, pair.right_lt_rel, pair.lt_rel_diff_pct);

            ImGui::EndTable();
        }
        ImGui::SetWindowFontScale(1.0f);
    }

    // Clear highlight button
    if (!mSymmetryHighlightLeft.empty()) {
        ImGui::Separator();
        if (ImGui::Button("Clear 3D Highlight##ParamSym")) {
            mMuscleParamSymmetrySelectedIdx = -1;
            mSymmetryHighlightLeft.clear();
            mSymmetryHighlightRight.clear();
        }
    }
}

// ============================================================
// Symmetry Operations Section (Left Panel)
// ============================================================
void MusclePersonalizerApp::drawSymmetryOperationsSection()
{
    if (!mExecutor || !mExecutor->getCharacter()) {
        ImGui::TextDisabled("Load character first");
        return;
    }

    // Refresh button
    if (ImGui::Button("Refresh Symmetry Analysis")) {
        computeSymmetryPairs();
        computeMuscleParamsSymmetryPairs();
    }

    ImGui::Separator();

    // Statistics summary
    int asymmetricCount = 0;
    std::vector<std::string> asymmetricBaseNames;
    for (const auto& pair : mSymmetryPairs) {
        if (!pair.is_symmetric) {
            asymmetricBaseNames.push_back(pair.base_name);
            asymmetricCount++;
        }
    }
    int totalPairs = static_cast<int>(mSymmetryPairs.size());
    float percentage = totalPairs > 0 ? (asymmetricCount * 100.0f / totalPairs) : 0.0f;

    if (asymmetricCount > 0) {
        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.2f, 1.0f),
            "Asymmetric: %d / %d (%.1f%%)", asymmetricCount, totalPairs, percentage);
    } else if (totalPairs > 0) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            "All symmetric: %d pairs", totalPairs);
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No muscle pairs found");
    }

    ImGui::Separator();
    ImGui::Text("Mirror Operations:");
    ImGui::TextWrapped("Averages Y/Z values directly. For X: averages abs values and preserves original signs.");

    ImGui::Spacing();

    // Mirror selected pair (from right panel selection)
    bool hasSelection = mSymmetrySelectedIdx >= 0 && mSymmetrySelectedIdx < static_cast<int>(mSymmetryPairs.size());

    if (hasSelection) {
        const auto& selectedPair = mSymmetryPairs[mSymmetrySelectedIdx];
        ImGui::Text("Selected: %s", selectedPair.base_name.c_str());
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Select pair in Symmetry tab (right panel)");
    }

    ImGui::BeginDisabled(!hasSelection);
    if (ImGui::Button("Mirror Selected Pair", ImVec2(-1, 25))) {
        if (hasSelection) {
            const auto& selectedPair = mSymmetryPairs[mSymmetrySelectedIdx];
            std::vector<std::string> toMirror = {selectedPair.base_name};
            if (mExecutor->mirrorAnchorPositions(toMirror)) {
                computeSymmetryPairs();  // Refresh after mirroring
            }
        }
    }
    ImGui::EndDisabled();

    ImGui::Spacing();

    // Mirror all asymmetric pairs
    ImGui::BeginDisabled(asymmetricCount == 0);
    char mirrorAsymLabel[64];
    snprintf(mirrorAsymLabel, sizeof(mirrorAsymLabel), "Mirror All Asymmetric (%d)", asymmetricCount);
    if (ImGui::Button(mirrorAsymLabel, ImVec2(-1, 25))) {
        if (!asymmetricBaseNames.empty()) {
            if (mExecutor->mirrorAnchorPositions(asymmetricBaseNames)) {
                computeSymmetryPairs();  // Refresh after mirroring
            }
        }
    }
    ImGui::EndDisabled();

    // Mirror all pairs
    ImGui::BeginDisabled(totalPairs == 0);
    if (ImGui::Button("Mirror All Pairs", ImVec2(-1, 25))) {
        if (mExecutor->mirrorAnchorPositions({})) {
            computeSymmetryPairs();  // Refresh after mirroring
        }
    }
    ImGui::EndDisabled();
}

// isPanelDefaultOpen() is inherited from ViewerAppBase

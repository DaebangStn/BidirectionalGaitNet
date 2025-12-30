#include "MusclePersonalizerApp.h"
#include "Character.h"
#include "ContractureOptimizer.h"
#include "optimizer/WaypointOptimizer.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>
#include <implot.h>

namespace fs = std::filesystem;

// ============================================================
// Constructor
// ============================================================

MusclePersonalizerApp::MusclePersonalizerApp(const std::string& configPath)
    : ViewerAppBase("Muscle Personalizer", 1920, 1080),
      mConfigPath(configPath),
      mResultsPanelWidth(450)
{
    // Adjust camera for muscle personalizer (slightly different defaults)
    mCamera.eye = Eigen::Vector3d(0.0, 0.0, 3.0);
    mCamera.trans = Eigen::Vector3d(0.0, -0.5, 0.0);

    std::cout << "[MusclePersonalizer] Constructor complete" << std::endl;
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

void MusclePersonalizerApp::onInitialize()
{
    // Create surgery executor
    mExecutor = std::make_unique<PMuscle::SurgeryExecutor>("MusclePersonalizer");
    std::cout << "[MusclePersonalizer] Surgery executor created" << std::endl;

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
        mPIDNavigator->scanPIDs();
    }

    std::cout << "[MusclePersonalizer] Initialization complete" << std::endl;
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
    // Load muscle_personalizer section from render.yaml (Template Method hook)
    // Common config (geometry, default_open_panels) already loaded by ViewerAppBase
    try {
        std::string renderResolved = rm::resolve("render.yaml");
        YAML::Node renderConfig = YAML::LoadFile(renderResolved);

        if (renderConfig["muscle_personalizer"]) {
            auto mp = renderConfig["muscle_personalizer"];
            if (mp["control_panel_width"]) mControlPanelWidth = mp["control_panel_width"].as<int>();
            if (mp["results_panel_width"]) mResultsPanelWidth = mp["results_panel_width"].as<int>();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Failed to load render.yaml: " << e.what() << std::endl;
    }

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
        }

        // Contracture estimation defaults
        if (config["contracture_estimation"]) {
            auto ce = config["contracture_estimation"];
            mContractureMaxIterations = ce["max_iterations"].as<int>(100);
            mContractureMinRatio = ce["min_ratio"].as<float>(0.7f);
            mContractureMaxRatio = ce["max_ratio"].as<float>(1.2f);
            mContractureUseRobustLoss = ce["use_robust_loss"].as<bool>(true);
        }

        std::cout << "[MusclePersonalizer] App config loaded from " << resolved << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Failed to load config: " << e.what() << std::endl;
        std::cerr << "[MusclePersonalizer] Using default configuration" << std::endl;
    }
}

void MusclePersonalizerApp::loadCharacter()
{
    try {
        std::cout << "[MusclePersonalizer] Loading skeleton: " << mSkeletonPath << std::endl;
        std::cout << "[MusclePersonalizer] Loading muscles: " << mMusclePath << std::endl;

        // Load character through SurgeryExecutor
        mExecutor->loadCharacter(mSkeletonPath, mMusclePath, ActuatorType::mus);

        // Get reference mass from character
        mReferenceMass = mExecutor->getCharacter()->getSkeleton()->getMass();
        mCurrentMass = mReferenceMass;

        std::cout << "[MusclePersonalizer] Character loaded successfully" << std::endl;
        std::cout << "[MusclePersonalizer] Reference mass: " << mReferenceMass << " kg" << std::endl;

        // Refresh muscle list
        refreshMuscleList();
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Failed to load character: " << e.what() << std::endl;
    }
}

void MusclePersonalizerApp::loadReferenceCharacter()
{
    // Reference character is loaded from default paths and never changes
    // It represents the "ideal" muscle behavior for waypoint optimization
    try {
        std::string skelResolved = rm::resolve(mReferenceSkeletonPath);
        std::string muscleResolved = rm::resolve(mReferenceMusclePath);

        std::cout << "[MusclePersonalizer] Loading reference character:" << std::endl;
        std::cout << "[MusclePersonalizer]   Skeleton: " << mReferenceSkeletonPath << std::endl;
        std::cout << "[MusclePersonalizer]   Muscles: " << mReferenceMusclePath << std::endl;

        // Create reference character
        mReferenceCharacter = new Character(skelResolved, true);
        mReferenceCharacter->setMuscles(muscleResolved);
        mReferenceCharacter->setActuatorType(ActuatorType::mus);

        // Zero muscle activations
        if (mReferenceCharacter->getMuscles().size() > 0) {
            mReferenceCharacter->setActivations(mReferenceCharacter->getActivations().setZero());
        }

        std::cout << "[MusclePersonalizer] Reference character loaded successfully" << std::endl;
        std::cout << "[MusclePersonalizer]   Reference muscles: " << mReferenceCharacter->getMuscles().size() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Failed to load reference character: " << e.what() << std::endl;
        mReferenceCharacter = nullptr;
    }
}

void MusclePersonalizerApp::initializeSurgeryExecutor()
{
    // No longer needed - SurgeryExecutor is created in onInitialize
    // Kept for compatibility but does nothing
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

    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

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

    for (size_t i = 0; i < muscles.size(); i++) {
        // Skip if muscle is not selected
        if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

        auto muscle = muscles[i];
        muscle->UpdateGeometry();

        Eigen::Vector4d color;

        // Highlight selected muscle in bright green
        if (!highlightedMuscle.empty() && muscle->name == highlightedMuscle) {
            color = Eigen::Vector4d(0.2, 1.0, 0.2, 1.0);  // Bright green
        } else if (mColorByContracture && muscle->lmt_base > 0) {
            // Contracture ratio: < 1.0 means shortened, > 1.0 means lengthened
            double ratio = muscle->lmt_ref / muscle->lmt_base;
            // Clamp to [0.7, 1.3] range and normalize to [0, 1]
            double t = std::clamp((ratio - 0.7) / 0.6, 0.0, 1.0);

            // Colormap: Blue (contracted) -> Cyan -> Green -> Yellow -> Red (lengthened)
            if (t < 0.25) {
                double s = t / 0.25;
                color = Eigen::Vector4d(0.0, s, 1.0, 0.85);  // Blue to Cyan
            } else if (t < 0.5) {
                double s = (t - 0.25) / 0.25;
                color = Eigen::Vector4d(0.0, 1.0, 1.0 - s, 0.85);  // Cyan to Green
            } else if (t < 0.75) {
                double s = (t - 0.5) / 0.25;
                color = Eigen::Vector4d(s, 1.0, 0.0, 0.85);  // Green to Yellow
            } else {
                double s = (t - 0.75) / 0.25;
                color = Eigen::Vector4d(1.0, 1.0 - s, 0.0, 0.85);  // Yellow to Red
            }
        } else {
            // Default: Fluorescent magenta/pink for high visibility
            color = Eigen::Vector4d(1.0, 0.2, 0.6, 0.85);
        }

        glColor4dv(color.data());
        mShapeRenderer.renderMuscle(muscle, -1.0);
    }

    glEnable(GL_LIGHTING);
}

// ============================================================
// UI Panels
// ============================================================

void MusclePersonalizerApp::drawLeftPanel()
{
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_Once);

    ImGui::Begin("Control##Panel", nullptr,ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    if (ImGui::BeginTabBar("ControlTabs")) {
        if (ImGui::BeginTabItem("Character")) {
            drawClinicalDataSection();
            drawCharacterLoadSection();
            drawWeightApplicationSection();
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
    ImGui::SetNextWindowPos(ImVec2(mWidth - mResultsPanelWidth, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mResultsPanelWidth, mHeight), ImGuiCond_Once);

    ImGui::Begin("Data##Panel", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    if (ImGui::BeginTabBar("DataTabs")) {
        if (ImGui::BeginTabItem("Waypoint Opt.")) {
            drawWaypointCurvesTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Contracture")) {
            drawContractureResultsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Exports")) {
            drawResultsSection();
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

        // PID Navigator inline selector
        mPIDNavigator->renderInlineSelector(120, 0);  // PID list only, no file list

        // Show selected PID info
        const auto& pidState = mPIDNavigator->getState();

        // Check if PID or pre/post changed - reload patient ROM and weight
        if (pidState.selectedPID >= 0) {
            const std::string& pid = pidState.pidList[pidState.selectedPID];
            if (pid != mCurrentROMPID || pidState.preOp != mCurrentROMPreOp) {
                loadPatientROM(pid, pidState.preOp);
                loadClinicalWeight(pid, pidState.preOp);
            }
        } else if (!mCurrentROMPID.empty()) {
            // PID was deselected
            mPatientROMValues.clear();
            mCurrentROMPID.clear();
            mClinicalWeightAvailable = false;
            updateROMTrialCDValues();
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

        // Patient skeleton controls - uses preOp from PID navigator
        if (mSkeletonDataSource == CharacterDataSource::PatientData) {
            if (!mCharacterPID.empty()) {
                ImGui::SameLine();
                const auto& pidState = mPIDNavigator->getState();
                ImGui::Text("(%s/%s)", mCharacterPID.c_str(), pidState.preOp ? "pre" : "post");
            } else {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "(Select PID first)");
            }
        }

        if (skelChanged) {
            scanSkeletonFiles();
            mSkeletonPath.clear();
        }

        // Skeleton file list
        std::string skelPrefix = "@data/skeleton/";
        if (mSkeletonDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            std::string prePost = mPIDNavigator->getState().preOp ? "pre" : "post";
            skelPrefix = "@pid:" + mCharacterPID + "/gait/" + prePost + "/skeleton/";
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

        // Patient muscle controls - uses preOp from PID navigator
        if (mMuscleDataSource == CharacterDataSource::PatientData) {
            if (!mCharacterPID.empty()) {
                ImGui::SameLine();
                const auto& pidState = mPIDNavigator->getState();
                ImGui::Text("(%s/%s)", mCharacterPID.c_str(), pidState.preOp ? "pre" : "post");
            } else {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "(Select PID first)");
            }
        }

        if (muscleChanged) {
            scanMuscleFiles();
            mMusclePath.clear();
        }

        // Muscle file list
        std::string musclePrefix = "@data/muscle/";
        if (mMuscleDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            std::string prePost = mPIDNavigator->getState().preOp ? "pre" : "post";
            musclePrefix = "@pid:" + mCharacterPID + "/gait/" + prePost + "/muscle/";
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
            initializeSurgeryExecutor();
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
    ImVec2 windowSize(400, 100);
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
    snprintf(overlay, sizeof(overlay), "%d / %d", current, total);
    ImGui::ProgressBar(progress, ImVec2(-1, 0), overlay);

    // Current muscle name (mutex-protected)
    std::string muscleName;
    {
        std::lock_guard<std::mutex> lock(mWaypointOptMutex);
        muscleName = mWaypointOptMuscleName;
    }
    ImGui::Text("Current: %s", muscleName.c_str());

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

        ImGui::Text("ROM Trials: %zu found", mROMTrials.size());
        ImGui::SameLine();
        if (ImGui::SmallButton("All")) {
            for (auto& trial : mROMTrials) trial.selected = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) {
            for (auto& trial : mROMTrials) trial.selected = false;
        }
        ImGui::InputText("Filter##ROM", mROMFilter, sizeof(mROMFilter));

        // ROM selection list - display "TC name - CD value"
        ImGui::BeginChild("ROMList", ImVec2(0, 150), true);
        for (size_t i = 0; i < mROMTrials.size(); ++i) {
            auto& trial = mROMTrials[i];

            // Apply filter
            if (strlen(mROMFilter) > 0) {
                if (trial.name.find(mROMFilter) == std::string::npos) {
                    continue;
                }
            }

            // Format label: "TC name - CD value" or "TC name - N/A"
            std::string label;
            if (trial.cd_value.has_value()) {
                char buf[128];
                snprintf(buf, sizeof(buf), "%s - %.1f°", trial.name.c_str(), trial.cd_value.value());
                label = buf;
            } else {
                label = trial.name + " - N/A";
            }

            if (ImGui::Checkbox(label.c_str(), &trial.selected)) {
                // Selection changed
            }

            // If no clinical data available, show manual ROM input
            if (!trial.cd_value.has_value()) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                char input_id[64];
                snprintf(input_id, sizeof(input_id), "##manual_rom_%zu", i);
                if (ImGui::InputFloat(input_id, &mROMTrials[i].manual_rom, 0, 0, "%.1f")) {
                    // Manual ROM value updated
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Manual ROM angle (degrees)");
                }
            }

            // Show tooltip with details on hover
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Description: %s", trial.description.c_str());
                ImGui::Text("Joint: %s (DOF %d)", trial.joint.c_str(), trial.dof_index);
                ImGui::Text("Torque: %.1f Nm", trial.torque);
                if (!trial.cd_side.empty()) {
                    ImGui::Separator();
                    ImGui::Text("Clinical Data: %s.%s.%s",
                        trial.cd_side.c_str(), trial.cd_joint.c_str(), trial.cd_field.c_str());
                    if (trial.cd_value.has_value()) {
                        ImGui::Text("Value: %.1f°", trial.cd_value.value());
                    } else {
                        ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Value: Not available");
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
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("Min Ratio", &mContractureMinRatio, 0.0f, 0.0f, "%.2f");
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("Max Ratio", &mContractureMaxRatio, 0.0f, 0.0f, "%.2f");

    ImGui::Separator();
    if (ImGui::Button("Estimate Contracture Parameters", ImVec2(-1, 0))) runContractureEstimation();
}

void MusclePersonalizerApp::drawRenderTab()
{
    // Character selection toggle
    if (ImGui::CollapsingHeader("Character", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    // Skeleton Rendering
    if (ImGui::CollapsingHeader("Skeleton", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    // Muscle Rendering
    if (ImGui::CollapsingHeader("Muscles", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show Muscles", &mRenderMuscles);
        ImGui::Checkbox("Color by Contracture", &mColorByContracture);

        if (mColorByContracture) {
            ImGui::Separator();
            ImGuiCommon::ColorBarLegend("Contracture Ratio", 0.7f, 1.3f, "(contracted)", "(lengthened)");
        }

        // Muscle selection
        if (mRenderMuscles && mExecutor && mExecutor->getCharacter()) {
            ImGui::Separator();
            auto muscles = mExecutor->getCharacter()->getMuscles();
            ImGuiCommon::MuscleSelector("##MuscleList", muscles, mMuscleSelectionStates,
                                        mMuscleFilterText, sizeof(mMuscleFilterText), 200.0f);
        }
    }

    // Ground
    if (ImGui::CollapsingHeader("Ground", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    // Camera
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    // Plot Settings
    if (ImGui::CollapsingHeader("Plot", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("Plot Height", &mPlotHeight, 10.0f, 50.0f, "%.0f");
    }
}

void MusclePersonalizerApp::drawResultsSection()
{
    if (!mExecutor || !mExecutor->getCharacter()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No character loaded");
        return;
    }

    // Determine export directory based on PID selection
    std::string skelPrefix = "@data/skeleton/";
    std::string musclePrefix = "@data/muscle/";
    std::string pathDisplay = "@data/";

    if (mPIDNavigator) {
        const auto& pidState = mPIDNavigator->getState();
        if (pidState.selectedPID >= 0) {
            const std::string& pid = pidState.pidList[pidState.selectedPID];
            std::string prePost = pidState.preOp ? "pre" : "post";
            skelPrefix = "@pid:" + pid + "/gait/" + prePost + "/skeleton/";
            musclePrefix = "@pid:" + pid + "/gait/" + prePost + "/muscle/";
            pathDisplay = "@pid:" + pid + "/gait/" + prePost + "/";
        }
    }

    // Show current export path
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Export to: %s", pathDisplay.c_str());
    ImGui::Spacing();

    // Skeleton export
    ImGui::Text("Skeleton:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-80);
    ImGui::InputText("##export_skeleton_name", mExportSkeletonName, sizeof(mExportSkeletonName));
    ImGui::SameLine();

    std::string skelFilename = std::string(mExportSkeletonName) + ".yaml";
    fs::path skelDir = rm::getManager().resolveDir(skelPrefix);
    fs::path skelPath = skelDir / skelFilename;
    bool skelExists = fs::exists(skelPath);

    if (ImGui::Button("Export##skel", ImVec2(-1, 0))) {
        try {
            if (skelPrefix.find("@pid:") != std::string::npos) {
                skelDir = rm::getManager().resolveDirCreate(skelPrefix);
                skelPath = skelDir / skelFilename;
            }
            mExecutor->exportSkeleton(skelPath.string());
            std::cout << "[MusclePersonalizer] Skeleton exported to: " << skelPrefix << skelFilename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MusclePersonalizer] Error exporting skeleton: " << e.what() << std::endl;
        }
    }
    if (skelExists) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "  %s exists - will overwrite", skelFilename.c_str());
    }

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
            mExecutor->exportMuscles(musclePath.string());
            std::cout << "[MusclePersonalizer] Muscle exported to: " << musclePrefix << muscleFilename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MusclePersonalizer] Error exporting muscle: " << e.what() << std::endl;
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

    // Filterable radio button list for single muscle selection
    char treeLabel[64];
    snprintf(treeLabel, sizeof(treeLabel), "Optimized result (%d/%zu)",
             numSuccess, mWaypointOptResults.size());
    if (ImGui::TreeNodeEx(treeLabel, ImGuiTreeNodeFlags_DefaultOpen)) {
        // Color function: green for success, red for failure
        auto colorFunc = [this](int idx) -> ImVec4 {
            if (idx >= 0 && idx < static_cast<int>(mWaypointOptResults.size())) {
                return mWaypointOptResults[idx].success
                    ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                    : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
            }
            return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        };

        ImGuiCommon::FilterableRadioList(
            "##ResultRadioList", muscleNames,
            &mWaypointResultSelectedIdx,
            mWaypointResultFilter, sizeof(mWaypointResultFilter),
            120, colorFunc);
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
        if (ImPlot::BeginPlot(plotTitle.c_str(), ImVec2(-1, mPlotHeight))) {
            const char* yAxisLabel = (result.length_type == PMuscle::LengthCurveType::NORMALIZED)
                ? "Normalized Length (lm_norm)"
                : "MTU Length (lmt)";
            ImPlot::SetupAxes("Phase", yAxisLabel);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImPlotCond_Always);  // X fixed 0-1
            ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always); // Y auto once

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

        // Shape Energy Plot (Direction Alignment)
        if (!result.phases.empty() && !result.shape_energy_before.empty()) {
            ImGui::Spacing();

            // Compute Y-axis range with padding
            auto minmax_before = std::minmax_element(
                result.shape_energy_before.begin(), result.shape_energy_before.end());
            auto minmax_after = std::minmax_element(
                result.shape_energy_after.begin(), result.shape_energy_after.end());

            double shape_y_min = std::min(*minmax_before.first, *minmax_after.first) * 0.9;
            double shape_y_max = std::max(*minmax_before.second, *minmax_after.second) * 1.1;
            shape_y_min = std::max(0.0, shape_y_min);  // Angle can't be negative
            shape_y_max = std::max(shape_y_max, 1.0);  // Ensure at least 1 degree range

            if (ImPlot::BeginPlot("Shape Energy (Direction Alignment)", ImVec2(-1, mPlotHeight))) {
                ImPlot::SetupAxes("Phase", "Angle (degrees)");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, shape_y_min, shape_y_max, ImPlotCond_Always);

                // Before curve (orange)
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.9f, 0.5f, 0.1f, 1.0f));
                ImPlot::PlotLine("Before", result.phases.data(),
                                result.shape_energy_before.data(),
                                static_cast<int>(result.phases.size()));
                ImPlot::PopStyleColor();

                // After curve (green)
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
                ImPlot::PlotLine("After", result.phases.data(),
                                result.shape_energy_after.data(),
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
        std::cerr << "[MusclePersonalizer] No surgery executor available" << std::endl;
        return;
    }

    Character* character = mExecutor->getCharacter();
    if (!character) {
        std::cerr << "[MusclePersonalizer] No character loaded" << std::endl;
        return;
    }

    if (mTargetMass <= 0.0f || mReferenceMass <= 0.0f) {
        std::cerr << "[MusclePersonalizer] Invalid mass values" << std::endl;
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

    std::cout << "[MusclePersonalizer] Applied weight scaling: "
              << mTargetMass << " kg (f0 ratio: " << f0_ratio << ")" << std::endl;
}

void MusclePersonalizerApp::runWaypointOptimization()
{
    if (!mExecutor) {
        std::cerr << "[MusclePersonalizer] No surgery executor available" << std::endl;
        return;
    }

    auto* character = mExecutor->getCharacter();
    if (!character) {
        std::cerr << "[MusclePersonalizer] No character loaded" << std::endl;
        return;
    }

    if (!mReferenceCharacter) {
        std::cerr << "[MusclePersonalizer] No reference character loaded" << std::endl;
        return;
    }

    // Check HDF path
    std::string hdfPath(mHDFPath);
    if (hdfPath.empty()) {
        std::cerr << "[MusclePersonalizer] No HDF file specified" << std::endl;
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
        std::cerr << "[MusclePersonalizer] No muscles selected for optimization" << std::endl;
        return;
    }

    std::cout << "[MusclePersonalizer] Starting waypoint optimization" << std::endl;
    std::cout << "[MusclePersonalizer]   HDF: " << hdfPath << std::endl;
    std::cout << "[MusclePersonalizer]   Muscles: " << selectedMuscles.size() << std::endl;

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
        std::cout << "[MusclePersonalizer] Waypoint optimization completed successfully" << std::endl;
    } else {
        std::cerr << "[MusclePersonalizer] Waypoint optimization failed" << std::endl;
    }
}


void MusclePersonalizerApp::runWaypointOptimizationAsync()
{
    // Don't start if already running
    if (mWaypointOptRunning) {
        std::cerr << "[MusclePersonalizer] Optimization already running" << std::endl;
        return;
    }

    // Join any previous thread
    if (mWaypointOptThread && mWaypointOptThread->joinable()) {
        mWaypointOptThread->join();
    }

    if (!mExecutor) {
        std::cerr << "[MusclePersonalizer] No surgery executor available" << std::endl;
        return;
    }

    auto* character = mExecutor->getCharacter();
    if (!character) {
        std::cerr << "[MusclePersonalizer] No character loaded" << std::endl;
        return;
    }

    if (!mReferenceCharacter) {
        std::cerr << "[MusclePersonalizer] No reference character loaded" << std::endl;
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
        std::cerr << "[MusclePersonalizer] No muscles selected" << std::endl;
        return;
    }

    std::string hdfPath(mHDFPath);
    if (hdfPath.empty()) {
        std::cerr << "[MusclePersonalizer] No HDF file specified" << std::endl;
        return;
    }

    // Initialize progress state
    mWaypointOptRunning = true;
    mWaypointOptCurrent = 0;
    mWaypointOptTotal = static_cast<int>(selectedMuscles.size());
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

    std::cout << "[MusclePersonalizer] Starting async waypoint optimization" << std::endl;
    std::cout << "[MusclePersonalizer]   HDF: " << hdfPath << std::endl;
    std::cout << "[MusclePersonalizer]   Muscles: " << selectedMuscles.size() << std::endl;
    std::cout << "[MusclePersonalizer]   Gradient: " << (config.analyticalGradient ? "analytical" : "numeric") << std::endl;
    std::cout << "[MusclePersonalizer]   Weights: phase=" << config.weightPhase << ", delta=" << config.weightDelta
              << ", samples=" << config.weightSamples << std::endl;
    std::cout << "[MusclePersonalizer]   Length type: " << (mWaypointUseNormalizedLength ? "normalized (lm_norm)" : "MTU (lmt)") << std::endl;

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
        std::cout << "[MusclePersonalizer] Waypoint optimization completed: "
                  << successCount << "/" << results.size() << " succeeded" << std::endl;

        mWaypointOptRunning = false;
    });
}

void MusclePersonalizerApp::runContractureEstimation()
{
    // Hardcoded muscle groups config path
    static const std::string MUSCLE_GROUPS_CONFIG = "@data/config/muscle_groups.yaml";

    if (!mExecutor || !mExecutor->getCharacter()) {
        std::cerr << "[MusclePersonalizer] No character loaded" << std::endl;
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

            // Populate rom_angle from clinical data or manual input
            if (trial.cd_value.has_value()) {
                config.rom_angle = trial.cd_value.value();
                std::cout << "[MusclePersonalizer] Using clinical ROM: " << config.name
                          << " = " << config.rom_angle << "°" << std::endl;
            } else if (std::abs(trial.manual_rom) > 0.001f) {
                config.rom_angle = static_cast<double>(trial.manual_rom);
                std::cout << "[MusclePersonalizer] Using manual ROM: " << config.name
                          << " = " << config.rom_angle << "°" << std::endl;
            } else {
                std::cerr << "[MusclePersonalizer] Skipping " << config.name
                          << ": No ROM value available (set clinical data or manual input)" << std::endl;
                continue;
            }

            configs.push_back(config);
        } catch (const std::exception& e) {
            std::cerr << "[MusclePersonalizer] Failed to load " << trial.filePath
                      << ": " << e.what() << std::endl;
        }
    }

    if (configs.empty()) {
        std::cerr << "[MusclePersonalizer] No valid ROM configs (need clinical data or manual ROM values)" << std::endl;
        return;
    }

    // Configure optimizer
    PMuscle::ContractureOptimizer optimizer;

    // Load muscle groups from config (required)
    if (optimizer.loadMuscleGroups(MUSCLE_GROUPS_CONFIG, mExecutor->getCharacter()) == 0) {
        std::cerr << "[MusclePersonalizer] Failed to load muscle groups from "
                  << MUSCLE_GROUPS_CONFIG << std::endl;
        return;
    }

    // Configure optimization
    PMuscle::ContractureOptimizer::Config optConfig;
    optConfig.maxIterations = mContractureMaxIterations;
    optConfig.minRatio = mContractureMinRatio;
    optConfig.maxRatio = mContractureMaxRatio;
    optConfig.useRobustLoss = mContractureUseRobustLoss;
    optConfig.verbose = true;

    std::cout << "[MusclePersonalizer] Running optimization with results capture..." << std::endl;

    // Run optimization with comprehensive results capture
    mContractureOptResult = optimizer.optimizeWithResults(mExecutor->getCharacter(), configs, optConfig);

    if (!mContractureOptResult.has_value() || mContractureOptResult->group_results.empty()) {
        std::cerr << "[MusclePersonalizer] Optimization returned no results" << std::endl;
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

    std::cout << "[MusclePersonalizer] Contracture estimation complete: "
              << mGroupResults.size() << " groups, "
              << mContractureOptResult->muscle_results.size() << " muscles, "
              << mContractureOptResult->trial_results.size() << " trials" << std::endl;
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
            std::cerr << "[MusclePersonalizer] ROM directory does not exist: " << mROMConfigDir << std::endl;
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
                    trial.torque = config["torque"].as<float>(15.0f);

                    // Parse clinical_data section
                    if (config["clinical_data"]) {
                        auto cd = config["clinical_data"];
                        trial.cd_side = cd["side"].as<std::string>("");
                        trial.cd_joint = cd["joint"].as<std::string>("");
                        trial.cd_field = cd["field"].as<std::string>("");
                    }

                    trial.selected = false;
                    trial.cd_value = std::nullopt;

                    mROMTrials.push_back(trial);
                }
                catch (const std::exception& e) {
                    std::cerr << "[MusclePersonalizer] Error parsing " << entry.path() << ": " << e.what() << std::endl;
                }
            }
        }

        // Update CD values if patient ROM is loaded
        updateROMTrialCDValues();

        std::cout << "[MusclePersonalizer] Found " << mROMTrials.size() << " ROM configs" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Error scanning ROM configs: " << e.what() << std::endl;
    }
}

void MusclePersonalizerApp::loadPatientROM(const std::string& pid, bool preOp)
{
    mPatientROMValues.clear();
    mCurrentROMPID = pid;
    mCurrentROMPreOp = preOp;

    if (pid.empty()) return;

    try {
        // Try to resolve @pid:{pid}/rom.yaml
        std::string romPath = "@pid:" + pid + "/rom.yaml";
        std::string resolvedPath = rm::getManager().resolve(romPath);

        if (resolvedPath.empty() || !fs::exists(resolvedPath)) {
            std::cout << "[MusclePersonalizer] No ROM data found for PID: " << pid << std::endl;
            return;
        }

        YAML::Node romData = YAML::LoadFile(resolvedPath);

        // Select phase based on preOp flag
        std::string phaseName = preOp ? "pre_op" : "post_op";
        if (!romData[phaseName]) {
            std::cout << "[MusclePersonalizer] No " << phaseName << " data in ROM file" << std::endl;
            return;
        }

        auto phaseData = romData[phaseName];
        if (!phaseData["rom"]) return;
        auto romSection = phaseData["rom"];

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

        std::cout << "[MusclePersonalizer] Loaded " << mPatientROMValues.size()
                  << " ROM values for PID " << pid << " (" << phaseName << ")" << std::endl;

        // Update trial CD values
        updateROMTrialCDValues();
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Error loading patient ROM: " << e.what() << std::endl;
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

void MusclePersonalizerApp::loadClinicalWeight(const std::string& pid, bool preOp)
{
    mClinicalWeightAvailable = false;
    mClinicalWeight = 0.0f;

    if (pid.empty()) return;

    try {
        // Load from @pid:{pid}/gait/metadata.yaml
        std::string metaPath = "@pid:" + pid + "/gait/metadata.yaml";
        std::string resolvedPath = rm::getManager().resolve(metaPath);

        if (resolvedPath.empty() || !fs::exists(resolvedPath)) {
            std::cout << "[MusclePersonalizer] No gait metadata found for PID: " << pid << std::endl;
            return;
        }

        YAML::Node meta = YAML::LoadFile(resolvedPath);

        // Access pre or post based on preOp flag
        std::string phaseName = preOp ? "pre" : "post";
        if (meta[phaseName] && meta[phaseName]["weight"]) {
            mClinicalWeight = meta[phaseName]["weight"].as<float>();
            mClinicalWeightAvailable = true;
            std::cout << "[MusclePersonalizer] Loaded weight: " << mClinicalWeight
                      << " kg (" << phaseName << ")" << std::endl;

            // Update target mass if using clinical data
            if (mWeightSource == WeightSource::ClinicalData) {
                mTargetMass = mClinicalWeight;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Error loading clinical weight: " << e.what() << std::endl;
    }
}

void MusclePersonalizerApp::scanSkeletonFiles()
{
    mSkeletonCandidates.clear();

    try {
        fs::path skelDir;
        if (mSkeletonDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            // Use patient data directory: {pid_root}/{pid}/gait/{pre|post}/skeleton/
            fs::path pidRoot = rm::getManager().getPidRoot();
            if (!pidRoot.empty()) {
                std::string prePost = mPIDNavigator->getState().preOp ? "pre" : "post";
                skelDir = pidRoot / mCharacterPID / "gait" / prePost / "skeleton";

                // Create directory if it doesn't exist
                if (!fs::exists(skelDir)) {
                    std::cout << "[MusclePersonalizer] Creating skeleton directory: " << skelDir << std::endl;
                    fs::create_directories(skelDir);
                }
            } else {
                std::cerr << "[MusclePersonalizer] PID root not available" << std::endl;
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
        std::cerr << "[MusclePersonalizer] Error scanning skeleton files: " << e.what() << std::endl;
    }
}

void MusclePersonalizerApp::scanMuscleFiles()
{
    mMuscleCandidates.clear();

    try {
        fs::path muscleDir;
        if (mMuscleDataSource == CharacterDataSource::PatientData && !mCharacterPID.empty() && mPIDNavigator) {
            // Use patient data directory: {pid_root}/{pid}/gait/{pre|post}/muscle/
            fs::path pidRoot = rm::getManager().getPidRoot();
            if (!pidRoot.empty()) {
                std::string prePost = mPIDNavigator->getState().preOp ? "pre" : "post";
                muscleDir = pidRoot / mCharacterPID / "gait" / prePost / "muscle";

                // Create directory if it doesn't exist
                if (!fs::exists(muscleDir)) {
                    std::cout << "[MusclePersonalizer] Creating muscle directory: " << muscleDir << std::endl;
                    fs::create_directories(muscleDir);
                }
            } else {
                std::cerr << "[MusclePersonalizer] PID root not available" << std::endl;
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
        std::cerr << "[MusclePersonalizer] Error scanning muscle files: " << e.what() << std::endl;
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
        std::cerr << "[MusclePersonalizer] Error scanning motion files: " << e.what() << std::endl;
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
        mSelectedMuscles.push_back(false);  // Select all by default
    }

    std::cout << "[MusclePersonalizer] Found " << mAvailableMuscles.size() << " muscles" << std::endl;
}

void MusclePersonalizerApp::exportMuscleConfig()
{
    // TODO: Implement muscle config export
    std::cout << "[MusclePersonalizer] Muscle config export not yet implemented" << std::endl;
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

    const auto& result = mContractureOptResult.value();

    // ===== Selectors Section (collapsible) =====
    if (ImGui::TreeNodeEx("Selectors", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Group selector
        ImGui::Text("Muscle Groups (%zu)", result.group_results.size());
        ImGui::InputTextWithHint("##grpFilter", "Filter...",
            mContractureGroupFilter, sizeof(mContractureGroupFilter));

        if (ImGui::BeginListBox("##GroupList", ImVec2(-FLT_MIN, 100))) {
            for (size_t i = 0; i < result.group_results.size(); ++i) {
                const auto& grp = result.group_results[i];
                if (strlen(mContractureGroupFilter) > 0 &&
                    grp.group_name.find(mContractureGroupFilter) == std::string::npos)
                    continue;

                bool selected = (mContractureSelectedGroupIdx == static_cast<int>(i));
                char label[128];
                snprintf(label, sizeof(label), "%s (%.3f)", grp.group_name.c_str(), grp.ratio);
                if (ImGui::Selectable(label, selected)) {
                    mContractureSelectedGroupIdx = static_cast<int>(i);
                }
            }
            ImGui::EndListBox();
        }

        ImGui::Spacing();

        // Trial selector
        ImGui::Text("ROM Trials (%zu)", result.trial_results.size());
        ImGui::InputTextWithHint("##trialFilter", "Filter...",
            mContractureTrialFilter, sizeof(mContractureTrialFilter));

        if (ImGui::BeginListBox("##TrialList", ImVec2(-FLT_MIN, 100))) {
            for (size_t i = 0; i < result.trial_results.size(); ++i) {
                const auto& trial = result.trial_results[i];
                if (strlen(mContractureTrialFilter) > 0 &&
                    trial.trial_name.find(mContractureTrialFilter) == std::string::npos)
                    continue;

                bool selected = (mContractureSelectedTrialIdx == static_cast<int>(i));
                if (ImGui::Selectable(trial.trial_name.c_str(), selected)) {
                    mContractureSelectedTrialIdx = static_cast<int>(i);
                }
            }
            ImGui::EndListBox();
        }

        ImGui::TreePop();
    }

    ImGui::Separator();

    // ===== Chart 1: lm_contract before/after for selected group =====
    if (mContractureSelectedGroupIdx >= 0 &&
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
            // Create tick labels
            std::vector<const char*> tick_labels;
            for (const auto& l : labels) tick_labels.push_back(l.c_str());

            if (ImPlot::BeginPlot("##lm_contract_chart", ImVec2(-1, 200))) {
                ImPlot::SetupAxes("Muscle", "lm_contract (m)");

                double bar_width = 0.35;
                int n = static_cast<int>(labels.size());

                // Generate positions for grouped bars
                std::vector<double> x_before(n), x_after(n);
                for (int i = 0; i < n; ++i) {
                    x_before[i] = i - bar_width / 2.0;
                    x_after[i] = i + bar_width / 2.0;
                }

                ImPlot::SetupAxisTicks(ImAxis_X1, 0, n - 1, n, tick_labels.data());

                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));
                ImPlot::PlotBars("Before", x_before.data(), before_vals.data(), n, bar_width);

                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));
                ImPlot::PlotBars("After", x_after.data(), after_vals.data(), n, bar_width);

                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No muscle data for this group");
        }
    }

    ImGui::Spacing();

    // ===== Chart 2: Per-muscle passive torque at selected trial pose =====
    if (mContractureSelectedTrialIdx >= 0 &&
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
            std::vector<const char*> tick_labels;
            for (const auto& l : labels) tick_labels.push_back(l.c_str());

            if (ImPlot::BeginPlot("##torque_chart", ImVec2(-1, 200))) {
                ImPlot::SetupAxes("Muscle", "Passive Torque (Nm)");

                double bar_width = 0.35;
                int n = static_cast<int>(labels.size());

                std::vector<double> x_before(n), x_after(n);
                for (int i = 0; i < n; ++i) {
                    x_before[i] = i - bar_width / 2.0;
                    x_after[i] = i + bar_width / 2.0;
                }

                ImPlot::SetupAxisTicks(ImAxis_X1, 0, n - 1, n, tick_labels.data());

                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.4f, 0.8f, 0.8f));
                ImPlot::PlotBars("Before", x_before.data(), before_vals.data(), n, bar_width);

                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.4f, 0.2f, 0.8f));
                ImPlot::PlotBars("After", x_after.data(), after_vals.data(), n, bar_width);

                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No torque data for this group/trial");
        }
    }

    ImGui::Spacing();

    // ===== Chart 3: Joint total passive torque before/after per trial =====
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

        if (ImPlot::BeginPlot("##joint_torque", ImVec2(-1, 150))) {
            ImPlot::SetupAxes("", "Torque (Nm)");
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

// isPanelDefaultOpen() is inherited from ViewerAppBase

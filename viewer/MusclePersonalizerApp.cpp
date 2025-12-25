#include "MusclePersonalizerApp.h"
#include "Character.h"
#include "ContractureOptimizer.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// Constructor
// ============================================================

MusclePersonalizerApp::MusclePersonalizerApp(const std::string& configPath)
    : ViewerAppBase("Muscle Personalizer", 1920, 1080),
      mConfigPath(configPath),
      mControlPanelWidth(450),
      mResultsPanelWidth(450)
{
    // Load configuration (sets paths, parameters, and panel widths)
    loadConfig();

    // Set window position if configured
    if (mWindowXPos != 0 || mWindowYPos != 0) {
        glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);
    }

    // Adjust camera for muscle personalizer (slightly different defaults)
    mCamera.eye = Eigen::Vector3d(0.0, 0.0, 3.0);
    mCamera.trans = Eigen::Vector3d(0.0, -0.5, 0.0);

    std::cout << "[MusclePersonalizer] Constructor complete" << std::endl;
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

    // Scan ROM configs
    scanROMConfigs();

    std::cout << "[MusclePersonalizer] Initialization complete" << std::endl;
}

void MusclePersonalizerApp::drawContent()
{
    drawSkeleton();
    drawMuscles();

    // Draw origin gizmo when camera is moving
    if (mRotate || mTranslate) {
        GUI::DrawOriginAxisGizmo(-mCamera.trans);
    }
}

void MusclePersonalizerApp::drawUI()
{
    drawLeftPanel();
    drawRightPanel();
}

// ============================================================
// Initialization
// ============================================================

void MusclePersonalizerApp::loadConfig()
{
    // Load render settings from render.yaml
    try {
        std::string renderResolved = rm::resolve("render.yaml");
        YAML::Node renderConfig = YAML::LoadFile(renderResolved);

        // Window geometry from render.yaml
        if (renderConfig["geometry"]) {
            auto geom = renderConfig["geometry"];
            if (geom["window"]) {
                mWidth = geom["window"]["width"].as<int>(1920);
                mHeight = geom["window"]["height"].as<int>(1080);
                mWindowXPos = geom["window"]["xpos"].as<int>(0);
                mWindowYPos = geom["window"]["ypos"].as<int>(0);
            }
        }

        // Panel widths from render.yaml muscle_personalizer section
        if (renderConfig["muscle_personalizer"]) {
            auto mp = renderConfig["muscle_personalizer"];
            mControlPanelWidth = mp["control_panel_width"].as<int>(450);
            mResultsPanelWidth = mp["results_panel_width"].as<int>(450);
        }

        // Default open panels from render.yaml
        if (renderConfig["default_open_panels"]) {
            for (const auto& panel : renderConfig["default_open_panels"]) {
                mDefaultOpenPanels.insert(panel.as<std::string>());
            }
        }

        std::cout << "[MusclePersonalizer] Render config loaded from " << renderResolved << std::endl;
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
    if (!mExecutor || !mExecutor->getCharacter()) return;

    GUI::DrawSkeleton(mExecutor->getCharacter()->getSkeleton(),
                      Eigen::Vector4d(0.8, 0.8, 0.8, 0.9),
                      mRenderMode,
                      &mShapeRenderer);
}

void MusclePersonalizerApp::drawMuscles()
{
    if (!mExecutor || !mExecutor->getCharacter() || !mRenderMuscles) return;

    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    auto muscles = mExecutor->getCharacter()->getMuscles();

    // Initialize selection states if needed
    if (mMuscleSelectionStates.size() != muscles.size()) {
        mMuscleSelectionStates.resize(muscles.size(), true);
    }

    for (size_t i = 0; i < muscles.size(); i++) {
        // Skip if muscle is not selected
        if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

        auto muscle = muscles[i];
        muscle->UpdateGeometry();

        Eigen::Vector4d color;
        if (mColorByContracture && muscle->lmt_base > 0) {
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
        if (ImGui::BeginTabItem("Control")) {
            drawCharacterLoadSection();
            drawWeightApplicationSection();
            drawWaypointOptimizationSection();
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

    ImGui::Begin("Data##Panel", nullptr,ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    // Results section only (View controls moved to left panel Render tab)
    drawResultsSection();

    ImGui::End();
}

void MusclePersonalizerApp::drawCharacterLoadSection()
{
    if (collapsingHeaderWithControls("Character Loading")) {
        ImGui::Text("Skeleton: %s", mSkeletonPath.c_str());
        ImGui::Text("Muscles: %s", mMusclePath.c_str());

        if (mExecutor && mExecutor->getCharacter()) {
            ImGui::Text("Status: Loaded");
            ImGui::Text("Reference Mass: %.1f kg", mReferenceMass);
            ImGui::Text("Muscles: %zu", mAvailableMuscles.size());
        } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Status: Not loaded");
        }

        if (ImGui::Button("Reload Character")) {
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
        ImGui::SliderFloat("Target Mass (kg)", &mTargetMass, 20.0f, 120.0f);

        if (mReferenceMass > 0.0f) {
            float ratio = std::pow(mTargetMass / mReferenceMass, 2.0f / 3.0f);
            ImGui::Text("f0 Scaling Ratio: %.3f", ratio);
        }

        ImGui::Separator();
        if (ImGui::Button("Apply Weight Scaling", ImVec2(-1, 0))) {
            applyWeightScaling();
        }

        if (mAppliedRatio != 1.0f) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Last applied ratio: %.3f", mAppliedRatio);
        }
    }
}

void MusclePersonalizerApp::drawWaypointOptimizationSection()
{
    if (collapsingHeaderWithControls("Waypoint Optimization")) {
#ifndef USE_WAYPOINT_OPTIMIZATION
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Waypoint optimization not available");
        ImGui::Text("(USE_WAYPOINT_OPTIMIZATION not defined)");
        return;
#endif

        ImGui::Text("Optimize muscle waypoints from HDF motion");
        ImGui::Separator();

        ImGui::InputText("HDF File", mHDFPath, sizeof(mHDFPath));
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            // TODO: File browser
        }

        ImGui::Text("Muscles: %zu available", mAvailableMuscles.size());
        ImGui::InputText("Filter##Muscle", mMuscleFilter, sizeof(mMuscleFilter));

        // Muscle selection list
        ImGui::BeginChild("MuscleList", ImVec2(0, 150), true);
        for (size_t i = 0; i < mAvailableMuscles.size(); ++i) {
            if (strlen(mMuscleFilter) > 0) {
                if (mAvailableMuscles[i].find(mMuscleFilter) == std::string::npos) {
                    continue;
                }
            }
            bool selected = i < mSelectedMuscles.size() ? mSelectedMuscles[i] : false;
            if (ImGui::Checkbox(mAvailableMuscles[i].c_str(), &selected)) {
                if (i < mSelectedMuscles.size()) {
                    mSelectedMuscles[i] = selected;
                }
            }
        }
        ImGui::EndChild();

        // Reference muscle selection
        if (!mAvailableMuscles.empty()) {
            if (ImGui::BeginCombo("Reference Muscle",
                    mReferenceMuscleIdx >= 0 && mReferenceMuscleIdx < static_cast<int>(mAvailableMuscles.size())
                    ? mAvailableMuscles[mReferenceMuscleIdx].c_str()
                    : "Select...")) {
                for (size_t i = 0; i < mAvailableMuscles.size(); ++i) {
                    bool isSelected = (mReferenceMuscleIdx == static_cast<int>(i));
                    if (ImGui::Selectable(mAvailableMuscles[i].c_str(), isSelected)) {
                        mReferenceMuscleIdx = static_cast<int>(i);
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }

        ImGui::Separator();
        ImGui::Text("Optimization Parameters:");
        ImGui::SliderInt("Max Iterations", &mWaypointMaxIterations, 10, 500);
        ImGui::SliderInt("Num Sampling", &mWaypointNumSampling, 10, 200);
        ImGui::SliderFloat("Lambda Shape", &mWaypointLambdaShape, 0.0f, 10.0f);
        ImGui::SliderFloat("Lambda Length Curve", &mWaypointLambdaLengthCurve, 0.0f, 10.0f);
        ImGui::Checkbox("Fix Origin/Insertion", &mWaypointFixOriginInsertion);

        ImGui::Separator();
        if (ImGui::Button("Optimize Waypoints", ImVec2(-1, 0))) {
            runWaypointOptimization();
        }
    }
}

void MusclePersonalizerApp::drawContractureEstimationSection()
{
    if (collapsingHeaderWithControls("Contracture Estimation")) {
        ImGui::Text("Estimate lm_contract from ROM trials");
        ImGui::Separator();

        ImGui::Text("ROM Config Directory:");
        ImGui::TextWrapped("%s", mROMConfigDir.c_str());

        if (ImGui::Button("Scan ROM Configs")) {
            scanROMConfigs();
        }

        ImGui::Text("ROM Trials: %zu found", mROMConfigNames.size());
        ImGui::InputText("Filter##ROM", mROMFilter, sizeof(mROMFilter));

        // ROM selection list
        ImGui::BeginChild("ROMList", ImVec2(0, 150), true);
        for (size_t i = 0; i < mROMConfigNames.size(); ++i) {
            if (strlen(mROMFilter) > 0) {
                if (mROMConfigNames[i].find(mROMFilter) == std::string::npos) {
                    continue;
                }
            }
            bool selected = i < mSelectedROMs.size() ? mSelectedROMs[i] : false;
            if (ImGui::Checkbox(mROMConfigNames[i].c_str(), &selected)) {
                if (i < mSelectedROMs.size()) {
                    mSelectedROMs[i] = selected;
                }
            }
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Text("Optimization Parameters:");
        ImGui::SliderInt("Max Iterations##Contract", &mContractureMaxIterations, 10, 500);
        ImGui::SliderFloat("Min Ratio", &mContractureMinRatio, 0.3f, 1.0f);
        ImGui::SliderFloat("Max Ratio", &mContractureMaxRatio, 1.0f, 2.0f);
        ImGui::Checkbox("Use Robust Loss (Huber)", &mContractureUseRobustLoss);

        ImGui::Separator();
        if (ImGui::Button("Estimate Contracture Parameters", ImVec2(-1, 0))) {
            runContractureEstimation();
        }
    }
}

void MusclePersonalizerApp::drawRenderTab()
{
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
}

void MusclePersonalizerApp::drawResultsSection()
{
    ImGui::Text("Muscle Groups: %zu", mGroupResults.size());
    ImGui::Separator();

    for (const auto& group : mGroupResults) {
        if (ImGui::TreeNode(group.group_name.c_str())) {
            ImGui::Text("Ratio: %.3f", group.ratio);
            ImGui::Text("Muscles:");
            for (size_t i = 0; i < group.muscle_names.size(); ++i) {
                ImGui::BulletText("%s: lm_contract = %.3f",
                    group.muscle_names[i].c_str(),
                    i < group.lm_contract_values.size() ? group.lm_contract_values[i] : 0.0);
            }
            ImGui::TreePop();
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Export Muscle Config", ImVec2(-1, 0))) {
        exportMuscleConfig();
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

    // Compute f0 scaling ratio: f0_new = f0_base * (mass/ref_mass)^(2/3)
    double f0_ratio = std::pow(mTargetMass / mReferenceMass, 2.0 / 3.0);

    // Apply scaling to all muscles
    const std::vector<Muscle*>& muscles = character->getMuscles();
    for (Muscle* muscle : muscles) {
        muscle->change_f(f0_ratio);
    }

    mAppliedRatio = static_cast<float>(f0_ratio);
    mCurrentMass = mTargetMass;

    std::cout << "[MusclePersonalizer] Applied f0 scaling ratio: " << f0_ratio
              << " to " << muscles.size() << " muscles" << std::endl;
}

void MusclePersonalizerApp::runWaypointOptimization()
{
    if (!mExecutor) {
        std::cerr << "[MusclePersonalizer] No surgery executor available" << std::endl;
        return;
    }

    if (!mExecutor->getCharacter()) {
        std::cerr << "[MusclePersonalizer] No character loaded" << std::endl;
        return;
    }

    // Check HDF path
    std::string hdfPath(mHDFPath);
    if (hdfPath.empty()) {
        std::cerr << "[MusclePersonalizer] No HDF file specified" << std::endl;
        return;
    }

    // Collect selected muscle names
    std::vector<std::string> selectedMuscles;
    for (size_t i = 0; i < mAvailableMuscles.size() && i < mSelectedMuscles.size(); ++i) {
        if (mSelectedMuscles[i]) {
            selectedMuscles.push_back(mAvailableMuscles[i]);
        }
    }

    if (selectedMuscles.empty()) {
        std::cerr << "[MusclePersonalizer] No muscles selected for optimization" << std::endl;
        return;
    }

    // Get reference muscle name
    std::string refMuscle;
    if (mReferenceMuscleIdx >= 0 && mReferenceMuscleIdx < static_cast<int>(mAvailableMuscles.size())) {
        refMuscle = mAvailableMuscles[mReferenceMuscleIdx];
    } else if (!selectedMuscles.empty()) {
        refMuscle = selectedMuscles[0];  // Default to first selected
    }

    std::cout << "[MusclePersonalizer] Starting waypoint optimization" << std::endl;
    std::cout << "[MusclePersonalizer]   HDF: " << hdfPath << std::endl;
    std::cout << "[MusclePersonalizer]   Muscles: " << selectedMuscles.size() << std::endl;
    std::cout << "[MusclePersonalizer]   Reference: " << refMuscle << std::endl;

    bool success = mExecutor->optimizeWaypoints(
        selectedMuscles,
        refMuscle,
        hdfPath,
        mWaypointMaxIterations,
        mWaypointNumSampling,
        static_cast<double>(mWaypointLambdaShape),
        static_cast<double>(mWaypointLambdaLengthCurve),
        mWaypointFixOriginInsertion
    );

    if (success) {
        std::cout << "[MusclePersonalizer] Waypoint optimization completed successfully" << std::endl;
    } else {
        std::cerr << "[MusclePersonalizer] Waypoint optimization failed" << std::endl;
    }
}

void MusclePersonalizerApp::runContractureEstimation()
{
    // Hardcoded muscle groups config path
    static const std::string MUSCLE_GROUPS_CONFIG = "@data/config/muscle_groups.yaml";

    if (!mExecutor || !mExecutor->getCharacter()) {
        std::cerr << "[MusclePersonalizer] No character loaded" << std::endl;
        return;
    }

    // Collect selected ROM configs
    std::vector<std::string> selectedPaths;
    for (size_t i = 0; i < mROMConfigPaths.size() && i < mSelectedROMs.size(); ++i) {
        if (mSelectedROMs[i]) {
            selectedPaths.push_back(mROMConfigPaths[i]);
        }
    }

    if (selectedPaths.empty()) {
        std::cerr << "[MusclePersonalizer] No ROM configs selected" << std::endl;
        return;
    }

    std::cout << "[MusclePersonalizer] Loading " << selectedPaths.size() << " ROM configs..." << std::endl;

    // Load ROM configs
    std::vector<PMuscle::ROMTrialConfig> configs;
    for (const auto& path : selectedPaths) {
        try {
            configs.push_back(PMuscle::ContractureOptimizer::loadROMConfig(path));
            std::cout << "[MusclePersonalizer]   Loaded: " << path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MusclePersonalizer]   Failed: " << path << ": " << e.what() << std::endl;
        }
    }

    if (configs.empty()) {
        std::cerr << "[MusclePersonalizer] No valid ROM configs loaded" << std::endl;
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

    // Configure and run iterative optimization (handles biarticular muscles)
    PMuscle::ContractureOptimizer::IterativeConfig iterConfig;
    iterConfig.baseConfig.maxIterations = mContractureMaxIterations;
    iterConfig.baseConfig.minRatio = mContractureMinRatio;
    iterConfig.baseConfig.maxRatio = mContractureMaxRatio;
    iterConfig.baseConfig.useRobustLoss = mContractureUseRobustLoss;
    iterConfig.baseConfig.verbose = true;
    iterConfig.maxOuterIterations = 3;     // Averaging iterations for biarticular muscles
    iterConfig.convergenceThreshold = 0.01;

    std::cout << "[MusclePersonalizer] Running iterative optimization..." << std::endl;
    auto optimizerResults = optimizer.optimizeIterative(mExecutor->getCharacter(), configs, iterConfig);

    if (optimizerResults.empty()) {
        std::cerr << "[MusclePersonalizer] Optimization returned no results" << std::endl;
        return;
    }

    // Convert to local result type
    mGroupResults.clear();
    for (const auto& optResult : optimizerResults) {
        MuscleGroupResult result;
        result.group_name = optResult.group_name;
        result.muscle_names = optResult.muscle_names;
        result.ratio = optResult.ratio;
        result.lm_contract_values = optResult.lm_contract_values;
        mGroupResults.push_back(result);
    }

    // Apply results to character
    PMuscle::ContractureOptimizer::applyResults(mExecutor->getCharacter(), optimizerResults);

    std::cout << "[MusclePersonalizer] Contracture estimation complete: "
              << mGroupResults.size() << " groups optimized" << std::endl;
}

// ============================================================
// Helper Methods
// ============================================================

void MusclePersonalizerApp::scanROMConfigs()
{
    mROMConfigPaths.clear();
    mROMConfigNames.clear();
    mSelectedROMs.clear();

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
                mROMConfigPaths.push_back(entry.path().string());
                mROMConfigNames.push_back(entry.path().filename().string());
                mSelectedROMs.push_back(false);
            }
        }

        std::cout << "[MusclePersonalizer] Found " << mROMConfigNames.size() << " ROM configs" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[MusclePersonalizer] Error scanning ROM configs: " << e.what() << std::endl;
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
        mSelectedMuscles.push_back(false);
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

bool MusclePersonalizerApp::isPanelDefaultOpen(const std::string& panelName) const
{
    return mDefaultOpenPanels.find(panelName) != mDefaultOpenPanels.end();
}

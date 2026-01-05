#include "SurgeryPanel.h"
#include "SurgeryScript.h"
#include "Log.h"
#include "rm/rm.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <filesystem>

using namespace PMuscle;

SurgeryPanel::SurgeryPanel(Character* character, ShapeRenderer* renderer)
    : SurgeryExecutor()
    , mCharacter(character)
    , mShapeRenderer(renderer)
    , mRecordingSurgery(false)
    , mRecordingScriptPath("data/recorded_surgery.yaml")
    , mLoadScriptPath("data/recorded_surgery.yaml")
    , mShowScriptPreview(false)
    , mSavingMuscle(false)
    , mSelectedCandidateAnchorIndex(-1)
    , mSelectedReferenceAnchorIndex(-1)
{
    // Set character in base class
    SurgeryExecutor::mCharacter = character;
    
    // Initialize path buffers
    strncpy(mRecordingPathBuffer, mRecordingScriptPath.c_str(), sizeof(mRecordingPathBuffer) - 1);
    mRecordingPathBuffer[sizeof(mRecordingPathBuffer) - 1] = '\0';
    strncpy(mLoadPathBuffer, mLoadScriptPath.c_str(), sizeof(mLoadPathBuffer) - 1);
    mLoadPathBuffer[sizeof(mLoadPathBuffer) - 1] = '\0';

    // Initialize save filename buffer
    std::strncpy(mSaveMuscleFilename, "muscle_modified", sizeof(mSaveMuscleFilename));
    mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

    // Initialize filter buffers
    mDistributeFilterBuffer[0] = '\0';
    mRelaxFilterBuffer[0] = '\0';
    mAnchorCandidateFilterBuffer[0] = '\0';
    mAnchorReferenceFilterBuffer[0] = '\0';

    // Initialize FDO Rotate Joint Offset members
    mRotateJointComboBuffer[0] = '\0';
    mSelectedRotateJoint = "";
    mRotateJointAxis[0] = 0.0f;
    mRotateJointAxis[1] = 1.0f;  // Default Y axis
    mRotateJointAxis[2] = 0.0f;
    mRotateJointAngleDeg = 0.0f;
    mRotateJointPreservePosition = false;

    // Initialize FDO Rotate Anchor Points members
    mRotateAnchorMuscleComboBuffer[0] = '\0';
    mRotateAnchorMuscleFilterBuffer[0] = '\0';
    mSelectedRotateAnchorMuscle = "";
    mSelectedRotateAnchorIndex = -1;
    mRotateAnchorSearchDir[0] = 0.0f;
    mRotateAnchorSearchDir[1] = -1.0f;  // Default -Y direction
    mRotateAnchorSearchDir[2] = 0.0f;
    mRotateAnchorRotAxis[0] = 0.0f;
    mRotateAnchorRotAxis[1] = 1.0f;  // Default Y axis
    mRotateAnchorRotAxis[2] = 0.0f;
    mRotateAnchorAngleDeg = 0.0f;

    // Initialize FDO Combined Mode members
    mFDOMode = false;
    mSelectedFDOTargetBodynode = "";
    mFDOBodynodeFilterBuffer[0] = '\0';

    // Initialize Save Skeleton members
    std::strncpy(mSaveSkeletonFilename, "skeleton_modified", sizeof(mSaveSkeletonFilename) - 1);
    mSaveSkeletonFilename[sizeof(mSaveSkeletonFilename) - 1] = '\0';
    mSaveAsYAML = true;  // Default to YAML format
}

void SurgeryPanel::setCharacter(Character* character) {
    mCharacter = character;
    SurgeryExecutor::mCharacter = character;
}

void SurgeryPanel::invalidateMuscleCache(const std::string& muscleName) {
    if (!mCharacter || !mShapeRenderer) return;
    
    auto muscles = mCharacter->getMuscles();
    for (auto m : muscles) {
        if (m->name == muscleName) {
            mShapeRenderer->invalidateMuscleCache(m);
            break;
        }
    }
}

// ============================================================================
// SURGERY OPERATION OVERRIDES (with cache invalidation)
// ============================================================================

bool SurgeryPanel::editAnchorPosition(const std::string& muscle, int anchor_index, 
                                     const Eigen::Vector3d& position) {
    bool success = SurgeryExecutor::editAnchorPosition(muscle, anchor_index, position);
    if (success) {
        invalidateMuscleCache(muscle);
    }
    return success;
}

bool SurgeryPanel::editAnchorWeights(const std::string& muscle, int anchor_index,
                                    const std::vector<double>& weights) {
    bool success = SurgeryExecutor::editAnchorWeights(muscle, anchor_index, weights);
    if (success) {
        invalidateMuscleCache(muscle);
    }
    return success;
}

bool SurgeryPanel::addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                                      const std::string& bodynode_name, double weight) {
    bool success = SurgeryExecutor::addBodyNodeToAnchor(muscle, anchor_index, bodynode_name, weight);
    if (success) {
        invalidateMuscleCache(muscle);
    }
    return success;
}

bool SurgeryPanel::removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index,
                                           int bodynode_index) {
    bool success = SurgeryExecutor::removeBodyNodeFromAnchor(muscle, anchor_index, bodynode_index);
    if (success) {
        invalidateMuscleCache(muscle);
    }
    return success;
}

bool SurgeryPanel::removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) {
    bool success = SurgeryExecutor::removeAnchorFromMuscle(muscleName, anchorIndex);
    if (success) {
        invalidateMuscleCache(muscleName);
    }
    return success;
}

bool SurgeryPanel::copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex, 
                                     const std::string& toMuscle) {
    bool success = SurgeryExecutor::copyAnchorToMuscle(fromMuscle, fromIndex, toMuscle);
    if (success) {
        invalidateMuscleCache(toMuscle);
    }
    return success;
}

// ============================================================================
// MAIN SURGERY PANEL UI
// ============================================================================

void SurgeryPanel::drawSurgeryPanel(bool* show_panel, int window_height) {
    ImGui::SetNextWindowSize(ImVec2(450, window_height - 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(450, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Surgery Operations", show_panel);

    if (ImGui::BeginTabBar("SurgeryPanelTabs")) {
        if (ImGui::BeginTabItem("Script")) {
            drawScriptTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Distribute")) {
            drawDistributePassiveForceSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Relax")) {
            drawRelaxPassiveForceSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Anchor")) {
            drawAnchorManipulationSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("FDO")) {
            drawRotateJointOffsetSection();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            drawRotateAnchorPointsSection();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            drawFDOCombinedSection();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();

    showScriptPreview();
}

void SurgeryPanel::drawSurgeryContent() {
    // Draw just the tab bar content (for embedding in another application's tab)
    // This does NOT create its own window - it draws directly into the current context
    if (ImGui::BeginTabBar("SurgeryPanelTabs")) {
        if (ImGui::BeginTabItem("Script")) {
            drawScriptTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Distribute")) {
            drawDistributePassiveForceSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Relax")) {
            drawRelaxPassiveForceSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Anchor")) {
            drawAnchorManipulationSection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("FDO")) {
            drawRotateJointOffsetSection();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            drawRotateAnchorPointsSection();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            drawFDOCombinedSection();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Still need to show script preview popup (it's a separate window)
    showScriptPreview();
}

// ============================================================================
// SCRIPT TAB CONTENT
// ============================================================================

void SurgeryPanel::drawScriptTabContent() {
    ImGui::Text("Script Recording");
    ImGui::Separator();
    drawScriptControlsSection();

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text("Save");
    ImGui::Separator();
    drawSaveMuscleConfigSection();
    ImGui::Spacing();
    drawSaveSkeletonConfigSection();

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text("Reset");
    ImGui::Separator();
    drawResetMusclesSection();
    ImGui::Spacing();
    drawResetSkeletonSection();
}

// ============================================================================
// SURGERY SCRIPT CONTROLS
// ============================================================================

void SurgeryPanel::drawScriptControlsSection() {
    if (!mRecordingSurgery) {
        if (ImGui::Button("Start Recording", ImVec2(150, 30))) {
            startRecording();
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
        if (ImGui::Button("Stop Recording", ImVec2(150, 30))) {
            stopRecording();
        }
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "● RECORDING");
    }

    if (!mRecordedOperations.empty()) {
        ImGui::Text("Recorded operations: %zu", mRecordedOperations.size());

        ImGui::Text("Export to:");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##RecordingPath", mRecordingPathBuffer, sizeof(mRecordingPathBuffer))) {
            mRecordingScriptPath = mRecordingPathBuffer;
        }

        if (ImGui::Button("Export Recording", ImVec2(-1, 30))) {
            exportRecording(mRecordingScriptPath);
        }
    }

    ImGui::Text("Script path:");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##LoadPath", mLoadPathBuffer, sizeof(mLoadPathBuffer))) {
        mLoadScriptPath = mLoadPathBuffer;
    }

    if (ImGui::Button("Load and Preview", ImVec2(-1, 30))) {
        loadSurgeryScript(mLoadScriptPath);
    }
}

void SurgeryPanel::drawResetMusclesSection() {
    if (ImGui::Button("Reset Muscles")) {
        resetMuscles();
        
        if (mRecordingSurgery) {
            auto op = std::make_unique<ResetMusclesOp>();
            recordOperation(std::move(op));
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset all muscle properties to their original state");
    }
}

void SurgeryPanel::drawDistributePassiveForceSection() {
    ImGui::TextWrapped("Select muscles, then choose one as reference to copy its passive force coefficient to others.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        return;
    }

    // Build muscle name list
    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }

    // Two-column layout
    ImGui::Columns(2, "DistributeColumns", true);

    // LEFT PANEL: All Muscles
    ImGui::Text("All Muscles:");
    ImGui::Separator();

    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##DistributeFilter", mDistributeFilterBuffer, sizeof(mDistributeFilterBuffer));
    if (ImGui::SmallButton("Clear Filter##Distribute")) {
        mDistributeFilterBuffer[0] = '\0';
    }

    std::string filter_lower(mDistributeFilterBuffer);
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

    std::vector<std::string> filteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (filter_lower.empty() || muscle_lower.find(filter_lower) != std::string::npos) {
            filteredMuscles.push_back(muscle_name);
        }
    }

    if (ImGui::SmallButton("All##Distribute")) {
        for (const auto& muscle_name : filteredMuscles) {
            mDistributeSelection[muscle_name] = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("None##Distribute")) {
        for (const auto& muscle_name : filteredMuscles) {
            mDistributeSelection[muscle_name] = false;
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Empty##Distribute")) {
        mDistributeSelection.clear();
        mDistributeRefMuscle = "";
    }

    ImGui::BeginChild("DistributeAllMuscles", ImVec2(0, 150), true);
    for (auto& muscle_name : filteredMuscles) {
        bool isSelected = mDistributeSelection[muscle_name];
        if (ImGui::Checkbox(muscle_name.c_str(), &isSelected)) {
            mDistributeSelection[muscle_name] = isSelected;
        }
    }
    ImGui::EndChild();

    // RIGHT PANEL: Selected Muscles
    ImGui::NextColumn();
    ImGui::Text("Selected Muscles:");
    ImGui::Separator();

    std::vector<std::string> selectedMuscles;
    for (const auto& muscle_name : muscleNames) {
        if (mDistributeSelection[muscle_name]) {
            selectedMuscles.push_back(muscle_name);
        }
    }

    ImGui::Text("Count: %zu", selectedMuscles.size());

    ImGui::BeginChild("DistributeSelectedMuscles", ImVec2(0, 150), true);
    if (selectedMuscles.empty()) {
        ImGui::TextDisabled("No muscles selected");
    } else {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Choose Reference:");
        ImGui::Separator();
        for (const auto& muscle_name : selectedMuscles) {
            bool isRef = (muscle_name == mDistributeRefMuscle);
            if (ImGui::RadioButton(muscle_name.c_str(), isRef)) {
                mDistributeRefMuscle = muscle_name;
            }
        }
    }
    ImGui::EndChild();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button("Apply Distribution", ImVec2(-1, 30))) {
        if (selectedMuscles.empty()) {
            LOG_WARN("[Surgery] Error: No muscles selected!");
        } else if (mDistributeRefMuscle.empty()) {
            LOG_WARN("[Surgery] Error: No reference muscle selected!");
        } else {
            std::map<std::string, Eigen::VectorXd> currentJointAngles;
            if (mRecordingSurgery && mCharacter) {
                auto skel = mCharacter->getSkeleton();
                for (size_t i = 1; i < skel->getNumJoints(); ++i) {
                    auto joint = skel->getJoint(i);
                    if (joint->getNumDofs() > 0) {
                        currentJointAngles[joint->getName()] = joint->getPositions();
                    }
                }
            }
            
            if (distributePassiveForce(selectedMuscles, mDistributeRefMuscle)) {
                if (mRecordingSurgery) {
                    auto op = std::make_unique<DistributePassiveForceOp>(
                        selectedMuscles, mDistributeRefMuscle, currentJointAngles);
                    recordOperation(std::move(op));
                }
            }
        }
    }
}

void SurgeryPanel::drawRelaxPassiveForceSection() {
    ImGui::TextWrapped("Select muscles to relax (reduce passive forces).");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        return;
    }

    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }

    ImGui::Columns(2, "RelaxColumns", true);

    ImGui::Text("All Muscles:");
    ImGui::Separator();

    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##RelaxFilter", mRelaxFilterBuffer, sizeof(mRelaxFilterBuffer));
    if (ImGui::SmallButton("Clear Filter##Relax")) {
        mRelaxFilterBuffer[0] = '\0';
    }

    std::string filter_lower(mRelaxFilterBuffer);
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

    std::vector<std::string> filteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (filter_lower.empty() || muscle_lower.find(filter_lower) != std::string::npos) {
            filteredMuscles.push_back(muscle_name);
        }
    }

    if (ImGui::SmallButton("All##Relax")) {
        for (const auto& muscle_name : filteredMuscles) {
            mRelaxSelection[muscle_name] = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("None##Relax")) {
        for (const auto& muscle_name : filteredMuscles) {
            mRelaxSelection[muscle_name] = false;
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Empty##Relax")) {
        mRelaxSelection.clear();
    }

    ImGui::BeginChild("RelaxAllMuscles", ImVec2(0, 150), true);
    for (auto& muscle_name : filteredMuscles) {
        bool isSelected = mRelaxSelection[muscle_name];
        if (ImGui::Checkbox(muscle_name.c_str(), &isSelected)) {
            mRelaxSelection[muscle_name] = isSelected;
        }
    }
    ImGui::EndChild();

    ImGui::NextColumn();
    ImGui::Text("Muscles to Relax:");
    ImGui::Separator();

    std::vector<std::string> selectedMuscles;
    for (const auto& muscle_name : muscleNames) {
        if (mRelaxSelection[muscle_name]) {
            selectedMuscles.push_back(muscle_name);
        }
    }

    ImGui::Text("Count: %zu", selectedMuscles.size());

    ImGui::BeginChild("RelaxSelectedMuscles", ImVec2(0, 150), true);
    if (selectedMuscles.empty()) {
        ImGui::TextDisabled("No muscles selected");
    } else {
        for (const auto& muscle_name : selectedMuscles) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", muscle_name.c_str());
        }
    }
    ImGui::EndChild();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button("Apply Relaxation", ImVec2(-1, 30))) {
        if (selectedMuscles.empty()) {
            LOG_WARN("[Surgery] Error: No muscles selected!");
        } else {
            std::map<std::string, Eigen::VectorXd> currentJointAngles;
            if (mRecordingSurgery && mCharacter) {
                auto skel = mCharacter->getSkeleton();
                for (size_t i = 1; i < skel->getNumJoints(); ++i) {
                    auto joint = skel->getJoint(i);
                    if (joint->getNumDofs() > 0) {
                        currentJointAngles[joint->getName()] = joint->getPositions();
                    }
                }
            }
            
            if (relaxPassiveForce(selectedMuscles)) {
                if (mRecordingSurgery) {
                    auto op = std::make_unique<RelaxPassiveForceOp>(
                        selectedMuscles, currentJointAngles);
                    recordOperation(std::move(op));
                }
            }
        }
    }
}

void SurgeryPanel::drawSaveMuscleConfigSection() {
    ImGui::TextWrapped("Save current muscle configuration to file.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    // Format selection
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Directory: @data/muscle/");
    ImGui::SameLine();
    if (ImGui::RadioButton("YAML##MuscleFormat", mSaveAsYAML)) {
        mSaveAsYAML = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("XML##MuscleFormat", !mSaveAsYAML)) {
        mSaveAsYAML = false;
    }
    ImGui::Spacing();

    // Determine file extension based on format
    const char* extension = mSaveAsYAML ? ".yaml" : ".xml";

    // Construct full filename with extension
    std::string filenameWithExt = std::string(mSaveMuscleFilename);
    // Remove existing extension if present
    size_t lastDot = filenameWithExt.find_last_of('.');
    if (lastDot != std::string::npos) {
        std::string ext = filenameWithExt.substr(lastDot);
        if (ext == ".yaml" || ext == ".xml") {
            filenameWithExt = filenameWithExt.substr(0, lastDot);
        }
    }
    filenameWithExt += extension;

    // For save destinations, resolve directory (not file) to avoid "not found" errors
    std::string dirUri = "@data/muscle";
    auto resolvedDir = rm::getManager().resolveDirCreate(dirUri);
    std::string resolvedPath = (resolvedDir / filenameWithExt).string();

    // Save button
    if (ImGui::Button("Save##Muscle")) {
        if (!mSavingMuscle) {
            mSavingMuscle = true;
            LOG_INFO("[Surgery] Muscle Save button clicked! Resolved path: " << resolvedPath);

            try {
                exportMuscles(resolvedPath);

                if (mRecordingSurgery) {
                    std::string recordUriPath = std::string("@data/muscle/") + filenameWithExt;
                    auto op = std::make_unique<ExportMusclesOp>(recordUriPath);
                    recordOperation(std::move(op));
                    LOG_INFO("[Surgery] Recorded muscle export with name: " << recordUriPath);
                }

                LOG_INFO("[Surgery] Muscle configuration saved to: " << resolvedPath);
            } catch (const std::exception& e) {
                LOG_ERROR("[Surgery] Error saving muscle configuration: " << e.what());
            }
            mSavingMuscle = false;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Save muscle properties to %s", resolvedPath.c_str());
    }
    ImGui::SameLine();

    // Text input for leaf filename only
    ImGui::Text("Filename:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##save_muscle_filename", mSaveMuscleFilename, sizeof(mSaveMuscleFilename));

    // Check if resolved file exists
    bool fileExists = std::filesystem::exists(resolvedPath);

    // Show warning if file exists
    if (fileExists) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Warning: File already exists and will be overwritten!");
    }

    ImGui::Spacing();
}

void SurgeryPanel::drawAnchorManipulationSection() {
    ImGui::TextWrapped("Select candidate and reference muscles, then manipulate anchor points.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        return;
    }
    
    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }
    
    static float editAnchorPosX = 0.0f;
    static float editAnchorPosY = 0.0f;
    static float editAnchorPosZ = 0.0f;
    static bool editValuesLoaded = false;
    static int selectedBodyNodeIndex = 0;
    static int newBodyNodeIndex = 0;
    static float newBodyNodeWeight = 1.0f;
    static std::vector<float> editWeights;
    
    // ========================================================================
    // ROW 1: Candidate Muscle
    // ========================================================================
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Candidate Muscle:");
    ImGui::Separator();
    
    ImGui::Columns(2, "CandidateColumns", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.35f);
    ImGui::SetColumnWidth(1, ImGui::GetWindowWidth() * 0.65f);
    
    ImGui::Text("Filter:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##CandidateFilter", mAnchorCandidateFilterBuffer, sizeof(mAnchorCandidateFilterBuffer));
    if (ImGui::SmallButton("Clear##Candidate")) {
        mAnchorCandidateFilterBuffer[0] = '\0';
    }
    
    std::string candidate_filter_lower(mAnchorCandidateFilterBuffer);
    std::transform(candidate_filter_lower.begin(), candidate_filter_lower.end(), 
                   candidate_filter_lower.begin(), ::tolower);
    
    std::vector<std::string> candidateFilteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (candidate_filter_lower.empty() || muscle_lower.find(candidate_filter_lower) != std::string::npos) {
            candidateFilteredMuscles.push_back(muscle_name);
        }
    }
    
    ImGui::Spacing();
    ImGui::Text("Select Muscle:");
    ImGui::BeginChild("CandidateMuscleList", ImVec2(0, 150), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto& muscle_name : candidateFilteredMuscles) {
        bool isSelected = (muscle_name == mAnchorCandidateMuscle);
        if (ImGui::RadioButton(muscle_name.c_str(), isSelected)) {
            mAnchorCandidateMuscle = muscle_name;
            mSelectedCandidateAnchorIndex = -1;
        }
    }
    ImGui::EndChild();
    
    ImGui::NextColumn();
    
    if (!mAnchorCandidateMuscle.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Anchors: %s", mAnchorCandidateMuscle.c_str());
        
        Muscle* candidateMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mAnchorCandidateMuscle) {
                candidateMuscle = m;
                break;
            }
        }
        
        if (candidateMuscle) {
            auto anchors = candidateMuscle->GetAnchors();
            
            ImGui::Text("Total: %zu anchors", anchors.size());
            if (mSelectedCandidateAnchorIndex >= 0) {
                ImGui::SameLine();
                ImGui::Text("| Sel: Anchor #%d", mSelectedCandidateAnchorIndex);
            }
            
            ImGui::BeginChild("CandidateAnchors", ImVec2(0, 200), true);
            for (int i = 0; i < anchors.size(); ++i) {
                auto anchor = anchors[i];
                bool isSelected = (i == mSelectedCandidateAnchorIndex);
                
                std::string label = "Anchor #" + std::to_string(i);
                if (ImGui::RadioButton(label.c_str(), isSelected)) {
                    mSelectedCandidateAnchorIndex = i;
                    editValuesLoaded = false;
                }
                
                ImGui::Indent();
                if (!anchor->bodynodes.empty()) {
                    ImGui::Text("Body: %s", anchor->bodynodes[0]->getName().c_str());
                    
                    if (anchor->bodynodes.size() > 1) {
                        ImGui::Text("Weights:");
                        for (size_t j = 0; j < anchor->bodynodes.size(); ++j) {
                            ImGui::Text("  [%zu] %s: %.3f", j, 
                                       anchor->bodynodes[j]->getName().c_str(),
                                       anchor->weights[j]);
                        }
                    }
                    
                    if (!anchor->local_positions.empty()) {
                        auto pos = anchor->local_positions[0];
                        ImGui::Text("Pos: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                    }
                }
                ImGui::Unindent();
                ImGui::Spacing();
            }
            ImGui::EndChild();
        }
    } else {
        ImGui::TextDisabled("No candidate muscle selected");
    }
    
    ImGui::Columns(1);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // ========================================================================
    // ROW 2: Reference Muscle (similar structure)
    // ========================================================================
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Reference Muscle:");
    ImGui::Separator();
    
    ImGui::Columns(2, "ReferenceColumns", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.35f);
    ImGui::SetColumnWidth(1, ImGui::GetWindowWidth() * 0.65f);

    ImGui::Text("Filter:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##ReferenceFilter", mAnchorReferenceFilterBuffer, sizeof(mAnchorReferenceFilterBuffer));
    if (ImGui::SmallButton("Clear##Reference")) {
        mAnchorReferenceFilterBuffer[0] = '\0';
    }
    
    std::string reference_filter_lower(mAnchorReferenceFilterBuffer);
    std::transform(reference_filter_lower.begin(), reference_filter_lower.end(), 
                   reference_filter_lower.begin(), ::tolower);
    
    std::vector<std::string> referenceFilteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (reference_filter_lower.empty() || muscle_lower.find(reference_filter_lower) != std::string::npos) {
            referenceFilteredMuscles.push_back(muscle_name);
        }
    }
    
    ImGui::Spacing();
    ImGui::Text("Select Muscle:");
    ImGui::BeginChild("ReferenceMuscleList", ImVec2(0, 150), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto& muscle_name : referenceFilteredMuscles) {
        bool isSelected = (muscle_name == mAnchorReferenceMuscle);
        if (ImGui::RadioButton(muscle_name.c_str(), isSelected)) {
            mAnchorReferenceMuscle = muscle_name;
            mSelectedReferenceAnchorIndex = -1;
        }
    }
    ImGui::EndChild();
    
    ImGui::NextColumn();
    
    if (!mAnchorReferenceMuscle.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Anchors: %s", mAnchorReferenceMuscle.c_str());
        
        Muscle* referenceMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mAnchorReferenceMuscle) {
                referenceMuscle = m;
                break;
            }
        }
        
        if (referenceMuscle) {
            auto anchors = referenceMuscle->GetAnchors();
            
            ImGui::Text("Total: %zu anchors", anchors.size());
            if (mSelectedReferenceAnchorIndex >= 0) {
                ImGui::SameLine();
                ImGui::Text("| Sel: Anchor #%d", mSelectedReferenceAnchorIndex);
            }
            
            ImGui::BeginChild("ReferenceAnchors", ImVec2(0, 200), true);
            for (int i = 0; i < anchors.size(); ++i) {
                auto anchor = anchors[i];
                bool isSelected = (i == mSelectedReferenceAnchorIndex);
                
                std::string label = "Anchor #" + std::to_string(i);
                if (ImGui::RadioButton(label.c_str(), isSelected)) {
                    mSelectedReferenceAnchorIndex = i;
                }
                
                ImGui::Indent();
                if (!anchor->bodynodes.empty()) {
                    ImGui::Text("Body: %s", anchor->bodynodes[0]->getName().c_str());
                    
                    if (anchor->bodynodes.size() > 1) {
                        ImGui::Text("Weights:");
                        for (size_t j = 0; j < anchor->bodynodes.size(); ++j) {
                            ImGui::Text("  [%zu] %s: %.3f", j, 
                                       anchor->bodynodes[j]->getName().c_str(),
                                       anchor->weights[j]);
                        }
                    }
                    
                    if (!anchor->local_positions.empty()) {
                        auto pos = anchor->local_positions[0];
                        ImGui::Text("Pos: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                    }
                }
                ImGui::Unindent();
                ImGui::Spacing();
            }
            ImGui::EndChild();
        }
    } else {
        ImGui::TextDisabled("No reference muscle selected");
    }
    
    ImGui::Columns(1);
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // ========================================================================
    // ANCHOR EDITING SECTION
    // ========================================================================
    if (ImGui::CollapsingHeader("Edit Selected Anchor", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
        if (!mAnchorCandidateMuscle.empty() && mSelectedCandidateAnchorIndex >= 0) {
            Muscle* candidateMuscle = nullptr;
            for (auto m : muscles) {
                if (m->name == mAnchorCandidateMuscle) {
                    candidateMuscle = m;
                    break;
                }
            }
            
            if (candidateMuscle && mSelectedCandidateAnchorIndex < candidateMuscle->GetAnchors().size()) {
                auto anchor = candidateMuscle->GetAnchors()[mSelectedCandidateAnchorIndex];
                auto skel = mCharacter->getSkeleton();
                
                if (!editValuesLoaded) {
                    if (!anchor->local_positions.empty()) {
                        auto pos = anchor->local_positions[0];
                        editAnchorPosX = pos[0];
                        editAnchorPosY = pos[1];
                        editAnchorPosZ = pos[2];
                    }
                    
                    editWeights.clear();
                    for (size_t i = 0; i < anchor->weights.size(); ++i) {
                        editWeights.push_back(anchor->weights[i]);
                    }
                    
                    selectedBodyNodeIndex = 0;
                    editValuesLoaded = true;
                }
                
                // Body Nodes & Weights
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Body Nodes & Weights:");
                ImGui::Separator();
                
                ImGui::Text("Anchor has %zu body node(s)", anchor->bodynodes.size());
                
                ImGui::BeginChild("AnchorBodyNodes", ImVec2(0, 120), true);
                for (size_t i = 0; i < anchor->bodynodes.size(); ++i) {
                    ImGui::PushID(i);
                    
                    ImGui::Text("[%zu] Body: %s", i, anchor->bodynodes[i]->getName().c_str());
                    
                    ImGui::SameLine(200);
                    ImGui::Text("Weight:");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80);
                    
                    if (i < editWeights.size()) {
                        ImGui::InputFloat("##Weight", &editWeights[i], 0.01f, 0.1f, "%.3f");
                    }
                    
                    if (anchor->bodynodes.size() > 1) {
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Remove##BodyNode")) {
                            if (removeBodyNodeFromAnchor(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, static_cast<int>(i))) {
                                editWeights.erase(editWeights.begin() + i);
                                editValuesLoaded = false;
                                
                                if (mRecordingSurgery) {
                                    auto op = std::make_unique<RemoveBodyNodeFromAnchorOp>(
                                        mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, static_cast<int>(i));
                                    recordOperation(std::move(op));
                                }
                            }
                        }
                    }
                    
                    ImGui::PopID();
                }
                ImGui::EndChild();
                
                if (ImGui::Button("Apply Weights", ImVec2(150, 25))) {
                    std::vector<double> weights_double(editWeights.begin(), editWeights.end());
                    
                    if (editAnchorWeights(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, weights_double)) {
                        if (mRecordingSurgery) {
                            auto op = std::make_unique<EditAnchorWeightsOp>(
                                mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, weights_double);
                            recordOperation(std::move(op));
                        }
                    }
                }
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                // Add Body Node
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Add Body Node:");
                ImGui::Separator();
                
                std::vector<std::string> allBodyNodeNames;
                for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
                    allBodyNodeNames.push_back(skel->getBodyNode(i)->getName());
                }
                
                ImGui::Text("Select Body Node:");
                ImGui::SetNextItemWidth(150);
                if (ImGui::BeginCombo("##NewBodyNode", 
                                      newBodyNodeIndex < allBodyNodeNames.size() ? 
                                      allBodyNodeNames[newBodyNodeIndex].c_str() : "")) {
                    for (size_t i = 0; i < allBodyNodeNames.size(); ++i) {
                        bool isSelected = (newBodyNodeIndex == i);
                        if (ImGui::Selectable(allBodyNodeNames[i].c_str(), isSelected)) {
                            newBodyNodeIndex = i;
                        }
                    }
                    ImGui::EndCombo();
                }
                
                ImGui::Text("Weight:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("##NewWeight", &newBodyNodeWeight, 0.01f, 0.1f, "%.3f");
                
                if (ImGui::Button("Add Body Node to Anchor", ImVec2(-1, 25))) {
                    if (newBodyNodeIndex < allBodyNodeNames.size()) {
                        auto newBodyNode = skel->getBodyNode(newBodyNodeIndex);
                        std::string bodyNodeName = newBodyNode->getName();
                        
                        if (addBodyNodeToAnchor(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, 
                                                bodyNodeName, newBodyNodeWeight)) {
                            editWeights.push_back(newBodyNodeWeight);
                            editValuesLoaded = false;
                            
                            if (mRecordingSurgery) {
                                auto op = std::make_unique<AddBodyNodeToAnchorOp>(
                                    mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, 
                                    bodyNodeName, newBodyNodeWeight);
                                recordOperation(std::move(op));
                            }
                        } else {
                            LOG_WARN("[Surgery] Body node already exists in this anchor!");
                        }
                    }
                }
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                // Local Position
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Local Position:");
                ImGui::Separator();
                
                if (!anchor->local_positions.empty()) {
                    auto pos = anchor->local_positions[0];
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                                      "Current: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                }
                
                ImGui::Text("New Position:");
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("X##EditAnchor", &editAnchorPosX, 0.001f, 0.01f, "%.3f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("Y##EditAnchor", &editAnchorPosY, 0.001f, 0.01f, "%.3f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("Z##EditAnchor", &editAnchorPosZ, 0.001f, 0.01f, "%.3f");
                
                if (ImGui::Button("Apply Position", ImVec2(-1, 25))) {
                    Eigen::Vector3d newPos(editAnchorPosX, editAnchorPosY, editAnchorPosZ);
                    if (editAnchorPosition(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, newPos)) {
                        if (mRecordingSurgery) {
                            auto op = std::make_unique<EditAnchorPositionOp>(
                                mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, newPos);
                            recordOperation(std::move(op));
                        }
                    }
                }
                
            } else {
                ImGui::TextDisabled("Invalid anchor selection");
            }
        } else {
            ImGui::TextDisabled("Select a candidate anchor to edit");
        }
        
        ImGui::Unindent();
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // ========================================================================
    // OPERATION BUTTONS
    // ========================================================================
    
    if (ImGui::Button("Remove Selected Anchor", ImVec2(-1, 30))) {
        if (mAnchorCandidateMuscle.empty()) {
            LOG_WARN("[Surgery] Error: No candidate muscle selected!");
        } else if (mSelectedCandidateAnchorIndex < 0) {
            LOG_WARN("[Surgery] Error: No anchor selected for removal!");
        } else {
            Muscle* candidateMuscle = nullptr;
            for (auto m : muscles) {
                if (m->name == mAnchorCandidateMuscle) {
                    candidateMuscle = m;
                    break;
                }
            }
            
            if (candidateMuscle) {
                int totalAnchors = candidateMuscle->GetAnchors().size();
                
                if (totalAnchors <= 2) {
                    LOG_WARN("[Surgery] Error: Cannot remove anchor - muscle must have at least 2 anchors!");
                } else {
                    if (removeAnchorFromMuscle(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex)) {
                        if (mRecordingSurgery) {
                            auto op = std::make_unique<RemoveAnchorOp>(
                                mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex);
                            recordOperation(std::move(op));
                        }
                        mSelectedCandidateAnchorIndex = -1;
                    }
                }
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Remove selected anchor from candidate muscle (keeps at least 2)");
    }
    
    if (ImGui::Button("Copy Anchor to Candidate", ImVec2(-1, 30))) {
        if (mAnchorCandidateMuscle.empty()) {
            LOG_WARN("[Surgery] Error: No candidate muscle selected!");
        } else if (mAnchorReferenceMuscle.empty()) {
            LOG_WARN("[Surgery] Error: No reference muscle selected!");
        } else if (mSelectedReferenceAnchorIndex < 0) {
            LOG_WARN("[Surgery] Error: No reference anchor selected!");
        } else {
            if (copyAnchorToMuscle(mAnchorReferenceMuscle, mSelectedReferenceAnchorIndex, mAnchorCandidateMuscle)) {
                if (mRecordingSurgery) {
                    auto op = std::make_unique<CopyAnchorOp>(
                        mAnchorReferenceMuscle, mSelectedReferenceAnchorIndex, mAnchorCandidateMuscle);
                    recordOperation(std::move(op));
                }
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Copy selected anchor from reference muscle to candidate muscle");
    }
}

// ============================================================================
// FDO SURGERY DRAWING METHODS
// ============================================================================

void SurgeryPanel::drawRotateJointOffsetSection() {
    ImGui::TextWrapped("Rotate a joint's offset and orientation frame relative to its parent body node (FDO surgery).");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        ImGui::TextDisabled("No skeleton available");
        return;
    }

    // Build joint name list (exclude root joint)
    std::vector<std::string> jointNames;
    for (size_t i = 0; i < skel->getNumJoints(); ++i) {
        auto joint = skel->getJoint(i);
        if (joint->getParentBodyNode() != nullptr) {  // Skip root joint
            jointNames.push_back(joint->getName());
        }
    }

    if (jointNames.empty()) {
        ImGui::TextDisabled("No valid joints found");
        return;
    }

    // Joint selection combo box
    ImGui::Text("Joint Name:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::BeginCombo("##RotateJoint", mSelectedRotateJoint.empty() ? "(select joint)" : mSelectedRotateJoint.c_str())) {
        for (const auto& joint_name : jointNames) {
            bool isSelected = (joint_name == mSelectedRotateJoint);
            if (ImGui::Selectable(joint_name.c_str(), isSelected)) {
                mSelectedRotateJoint = joint_name;
                strncpy(mRotateJointComboBuffer, joint_name.c_str(), sizeof(mRotateJointComboBuffer) - 1);
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Rotation axis input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Rotation Axis (in parent frame):");
    ImGui::SameLine();

    // Radio buttons for axis selection
    if (ImGui::RadioButton("X##RotJoint", mRotateJointAxis[0] == 1.0f && mRotateJointAxis[1] == 0.0f && mRotateJointAxis[2] == 0.0f)) {
        mRotateJointAxis[0] = 1.0f; mRotateJointAxis[1] = 0.0f; mRotateJointAxis[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Y##RotJoint", mRotateJointAxis[0] == 0.0f && mRotateJointAxis[1] == 1.0f && mRotateJointAxis[2] == 0.0f)) {
        mRotateJointAxis[0] = 0.0f; mRotateJointAxis[1] = 1.0f; mRotateJointAxis[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Z##RotJoint", mRotateJointAxis[0] == 0.0f && mRotateJointAxis[1] == 0.0f && mRotateJointAxis[2] == 1.0f)) {
        mRotateJointAxis[0] = 0.0f; mRotateJointAxis[1] = 0.0f; mRotateJointAxis[2] = 1.0f;
    }

    // Angle input (degrees and radians)
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Angle:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("degrees##RotJoint", &mRotateJointAngleDeg, 1.0f, 10.0f, "%.2f");
    ImGui::SameLine();
    ImGui::Checkbox("Preserve Position##RotJointPreservePositionCheckbox", &mRotateJointPreservePosition);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Apply button
    if (ImGui::Button("Apply Joint Rotation", ImVec2(-1, 30))) {
        if (mSelectedRotateJoint.empty()) {
            LOG_INFO("[Surgery] Error: No joint selected!");
        } else {
            // Normalize axis
            Eigen::Vector3d axis(mRotateJointAxis[0], mRotateJointAxis[1], mRotateJointAxis[2]);
            double axis_norm = axis.norm();
            if (axis_norm < 1e-6) {
                LOG_INFO("[Surgery] Error: Rotation axis cannot be zero!");
            } else {
                axis.normalize();
                double angle_rad_d = mRotateJointAngleDeg * M_PI / 180.0;

                // Execute operation
                if (rotateJointOffset(mSelectedRotateJoint, axis, angle_rad_d, mRotateJointPreservePosition)) {
                    // Record operation if recording
                    if (mRecordingSurgery) {
                        auto op = std::make_unique<RotateJointOffsetOp>(
                            mSelectedRotateJoint, axis, angle_rad_d, mRotateJointPreservePosition);
                        recordOperation(std::move(op));
                    }
                }
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Rotate the selected joint's offset and orientation frame");
    }
}

void SurgeryPanel::drawRotateAnchorPointsSection() {
    ImGui::TextWrapped("Rotate muscle anchor points on a bodynode with positional filtering (FDO surgery).");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        return;
    }

    // Build muscle name list
    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }

    // Reference muscle selection with search filter
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Reference Muscle:");

    // Search filter
    ImGui::SetNextItemWidth(200);
    ImGui::InputTextWithHint("##RotateAnchorMuscleFilter", "Search muscles...",
                            mRotateAnchorMuscleFilterBuffer, sizeof(mRotateAnchorMuscleFilterBuffer));

    // Convert filter to lowercase for case-insensitive search
    std::string filter_lower(mRotateAnchorMuscleFilterBuffer);
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);

    // Build filtered muscle list
    std::vector<std::string> filteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        if (filter_lower.empty()) {
            filteredMuscles.push_back(muscle_name);
        } else {
            std::string muscle_lower = muscle_name;
            std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
            if (muscle_lower.find(filter_lower) != std::string::npos) {
                filteredMuscles.push_back(muscle_name);
            }
        }
    }

    // Muscle selection combo
    ImGui::SetNextItemWidth(-1);
    std::string preview_text = mSelectedRotateAnchorMuscle.empty()
        ? "(select muscle)"
        : mSelectedRotateAnchorMuscle;
    if (!filter_lower.empty()) {
        preview_text += " [" + std::to_string(filteredMuscles.size()) + " found]";
    }

    if (ImGui::BeginCombo("##RotateAnchorMuscle", preview_text.c_str())) {
        for (const auto& muscle_name : filteredMuscles) {
            bool isSelected = (muscle_name == mSelectedRotateAnchorMuscle);
            if (ImGui::Selectable(muscle_name.c_str(), isSelected)) {
                mSelectedRotateAnchorMuscle = muscle_name;
                mSelectedRotateAnchorIndex = -1;  // Reset anchor selection
                strncpy(mRotateAnchorMuscleComboBuffer, muscle_name.c_str(), sizeof(mRotateAnchorMuscleComboBuffer) - 1);
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Reference anchor selection (if muscle selected)
    if (!mSelectedRotateAnchorMuscle.empty()) {
        Muscle* refMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mSelectedRotateAnchorMuscle) {
                refMuscle = m;
                break;
            }
        }

        if (refMuscle) {
            auto anchors = refMuscle->GetAnchors();
            ImGui::Text("Reference Anchor:");
            ImGui::SetNextItemWidth(-1);

            std::string anchor_preview = (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < (int)anchors.size())
                ? "Anchor #" + std::to_string(mSelectedRotateAnchorIndex)
                : "(select anchor)";

            if (ImGui::BeginCombo("##RotateAnchorIndex", anchor_preview.c_str())) {
                for (int i = 0; i < (int)anchors.size(); ++i) {
                    bool isSelected = (i == mSelectedRotateAnchorIndex);
                    std::string label = "Anchor #" + std::to_string(i);
                    if (!anchors[i]->bodynodes.empty()) {
                        label += " (" + anchors[i]->bodynodes[0]->getName() + ")";
                    }
                    if (ImGui::Selectable(label.c_str(), isSelected)) {
                        // Validate: reference anchor must have exactly one bodynode
                        if (anchors[i]->bodynodes.size() != 1) {
                            LOG_ERROR("[Surgery] Reference anchor must have exactly 1 bodynode (single-LBS only). Found: "
                                     << anchors[i]->bodynodes.size());
                            mSelectedRotateAnchorIndex = -1;
                        } else {
                            mSelectedRotateAnchorIndex = i;
                        }
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            // Display anchor info
            if (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < (int)anchors.size()) {
                auto anchor = anchors[mSelectedRotateAnchorIndex];
                if (!anchor->bodynodes.empty() && !anchor->local_positions.empty()) {
                    ImGui::TextDisabled("  Body: %s", anchor->bodynodes[0]->getName().c_str());
                    auto pos = anchor->local_positions[0];
                    ImGui::TextDisabled("  Pos: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                }
            }
        }
    }

    // Search direction input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Search Direction (filter criterion):");

    // Radio buttons for direction selection
    if (ImGui::RadioButton("+X##Search", mRotateAnchorSearchDir[0] == 1.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = 1.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("-X##Search", mRotateAnchorSearchDir[0] == -1.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = -1.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("+Y##Search", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == 1.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = 1.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("-Y##Search", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == -1.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = -1.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("+Z##Search", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == 1.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = 1.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("-Z##Search", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == -1.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = -1.0f;
    }

    ImGui::TextDisabled("  Rotates anchors where dot(pos-ref, dir) > 0");

    ImGui::Spacing();

    // Rotation axis input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Rotation Axis (in bodynode frame):");
    ImGui::SameLine();

    // Radio buttons for axis selection
    if (ImGui::RadioButton("X##RotAnchor", mRotateAnchorRotAxis[0] == 1.0f && mRotateAnchorRotAxis[1] == 0.0f && mRotateAnchorRotAxis[2] == 0.0f)) {
        mRotateAnchorRotAxis[0] = 1.0f; mRotateAnchorRotAxis[1] = 0.0f; mRotateAnchorRotAxis[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Y##RotAnchor", mRotateAnchorRotAxis[0] == 0.0f && mRotateAnchorRotAxis[1] == 1.0f && mRotateAnchorRotAxis[2] == 0.0f)) {
        mRotateAnchorRotAxis[0] = 0.0f; mRotateAnchorRotAxis[1] = 1.0f; mRotateAnchorRotAxis[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Z##RotAnchor", mRotateAnchorRotAxis[0] == 0.0f && mRotateAnchorRotAxis[1] == 0.0f && mRotateAnchorRotAxis[2] == 1.0f)) {
        mRotateAnchorRotAxis[0] = 0.0f; mRotateAnchorRotAxis[1] = 0.0f; mRotateAnchorRotAxis[2] = 1.0f;
    }

    ImGui::Spacing();

    // Angle input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Rotation Angle:");
    ImGui::SameLine();

    ImGui::SetNextItemWidth(80);
    ImGui::InputFloat("degrees##RotAnchor", &mRotateAnchorAngleDeg, 1.0f, 10.0f, "%.2f");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Apply button
    if (ImGui::Button("Apply Anchor Rotation", ImVec2(-1, 30))) {
        if (mSelectedRotateAnchorMuscle.empty()) {
            LOG_WARN("[Surgery GUI] Error: No muscle selected!");
        } else if (mSelectedRotateAnchorIndex < 0) {
            LOG_WARN("[Surgery GUI] Error: No anchor selected!");
        } else {
            // Normalize vectors
            Eigen::Vector3d search_dir(mRotateAnchorSearchDir[0], mRotateAnchorSearchDir[1], mRotateAnchorSearchDir[2]);
            Eigen::Vector3d rot_axis(mRotateAnchorRotAxis[0], mRotateAnchorRotAxis[1], mRotateAnchorRotAxis[2]);

            double search_norm = search_dir.norm();
            double axis_norm = rot_axis.norm();

            if (search_norm < 1e-6) {
                LOG_WARN("[Surgery GUI] Error: Search direction cannot be zero!");
            } else if (axis_norm < 1e-6) {
                LOG_WARN("[Surgery GUI] Error: Rotation axis cannot be zero!");
            } else {
                search_dir.normalize();
                rot_axis.normalize();
                double angle_rad_d = mRotateAnchorAngleDeg * M_PI / 180.0;

                // Execute operation
                if (rotateAnchorPoints(mSelectedRotateAnchorMuscle, mSelectedRotateAnchorIndex,
                                      search_dir, rot_axis, angle_rad_d)) {
                    // Record operation if recording
                    if (mRecordingSurgery) {
                        auto op = std::make_unique<RotateAnchorPointsOp>(
                            mSelectedRotateAnchorMuscle, mSelectedRotateAnchorIndex,
                            search_dir, rot_axis, angle_rad_d);
                        recordOperation(std::move(op));
                    }
                }
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Rotate anchor points on the bodynode using positional filter");
    }
}

void SurgeryPanel::drawFDOCombinedSection() {
    ImGui::TextWrapped("FDO Combined Surgery: Automatically rotate child joint of target bone, then rotate anchors on that bone with positional filtering.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        return;
    }

    // Build muscle name list
    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }

    // Reference muscle selection with search filter
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Reference Muscle:");

    // Search filter
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##FDOMuscleFilter", "Search muscles...",
                            mRotateAnchorMuscleFilterBuffer, sizeof(mRotateAnchorMuscleFilterBuffer));

    // Convert filter to lowercase for case-insensitive search
    std::string filter_lower(mRotateAnchorMuscleFilterBuffer);
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

    // Build filtered muscle list
    std::vector<std::string> filteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        if (filter_lower.empty()) {
            filteredMuscles.push_back(muscle_name);
        } else {
            std::string muscle_lower = muscle_name;
            std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
            if (muscle_lower.find(filter_lower) != std::string::npos) {
                filteredMuscles.push_back(muscle_name);
            }
        }
    }

    // Muscle selection combo
    ImGui::SetNextItemWidth(-1);
    std::string preview_text = mSelectedRotateAnchorMuscle.empty()
        ? "(select muscle)"
        : mSelectedRotateAnchorMuscle;
    if (!filter_lower.empty()) {
        preview_text += " [" + std::to_string(filteredMuscles.size()) + " found]";
    }

    if (ImGui::BeginCombo("##FDOMuscle", preview_text.c_str())) {
        for (const auto& muscle_name : filteredMuscles) {
            bool isSelected = (muscle_name == mSelectedRotateAnchorMuscle);
            if (ImGui::Selectable(muscle_name.c_str(), isSelected)) {
                mSelectedRotateAnchorMuscle = muscle_name;
                mSelectedRotateAnchorIndex = -1;  // Reset anchor selection
                strncpy(mRotateAnchorMuscleComboBuffer, muscle_name.c_str(), sizeof(mRotateAnchorMuscleComboBuffer) - 1);
            }
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Reference anchor selection (if muscle selected)
    if (!mSelectedRotateAnchorMuscle.empty()) {
        Muscle* refMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mSelectedRotateAnchorMuscle) {
                refMuscle = m;
                break;
            }
        }

        if (refMuscle) {
            auto anchors = refMuscle->GetAnchors();
            ImGui::Text("Reference Anchor:");
            ImGui::SetNextItemWidth(-1);

            std::string anchor_preview = (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < (int)anchors.size())
                ? "Anchor #" + std::to_string(mSelectedRotateAnchorIndex)
                : "(select anchor)";

            if (ImGui::BeginCombo("##FDOAnchorIndex", anchor_preview.c_str())) {
                for (int i = 0; i < (int)anchors.size(); ++i) {
                    bool isSelected = (i == mSelectedRotateAnchorIndex);
                    std::string label = "Anchor #" + std::to_string(i);
                    if (!anchors[i]->bodynodes.empty()) {
                        label += " (" + anchors[i]->bodynodes[0]->getName() + ")";
                    }
                    if (ImGui::Selectable(label.c_str(), isSelected)) {
                        // Validate: reference anchor must have exactly one bodynode
                        if (anchors[i]->bodynodes.size() != 1) {
                            LOG_ERROR("[Surgery] Reference anchor must have exactly 1 bodynode (single-LBS only). Found: "
                                     << anchors[i]->bodynodes.size());
                            mSelectedRotateAnchorIndex = -1;
                        } else {
                            mSelectedRotateAnchorIndex = i;
                        }
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            // Display anchor info
            if (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < (int)anchors.size()) {
                auto anchor = anchors[mSelectedRotateAnchorIndex];
                if (!anchor->bodynodes.empty() && !anchor->local_positions.empty()) {
                    ImGui::TextDisabled("  Body: %s", anchor->bodynodes[0]->getName().c_str());
                    auto pos = anchor->local_positions[0];
                    ImGui::TextDisabled("  Pos: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                }
            }
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Note: Target bodynode is automatically determined from the reference anchor");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Search direction input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Search Direction (filter criterion):");

    // Radio buttons for direction selection
    if (ImGui::RadioButton("+X##FDOSearch", mRotateAnchorSearchDir[0] == 1.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = 1.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("-X##FDOSearch", mRotateAnchorSearchDir[0] == -1.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = -1.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("+Y##FDOSearch", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == 1.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = 1.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("-Y##FDOSearch", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == -1.0f && mRotateAnchorSearchDir[2] == 0.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = -1.0f; mRotateAnchorSearchDir[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("+Z##FDOSearch", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == 1.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = 1.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("-Z##FDOSearch", mRotateAnchorSearchDir[0] == 0.0f && mRotateAnchorSearchDir[1] == 0.0f && mRotateAnchorSearchDir[2] == -1.0f)) {
        mRotateAnchorSearchDir[0] = 0.0f; mRotateAnchorSearchDir[1] = 0.0f; mRotateAnchorSearchDir[2] = -1.0f;
    }

    ImGui::TextDisabled("  Rotates anchors where dot(pos-ref, dir) > 0");

    ImGui::Spacing();

    // Rotation axis input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Rotation Axis (in bodynode frame):");

    // Radio buttons for axis selection
    if (ImGui::RadioButton("X##FDORotAnchor", mRotateAnchorRotAxis[0] == 1.0f && mRotateAnchorRotAxis[1] == 0.0f && mRotateAnchorRotAxis[2] == 0.0f)) {
        mRotateAnchorRotAxis[0] = 1.0f; mRotateAnchorRotAxis[1] = 0.0f; mRotateAnchorRotAxis[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Y##FDORotAnchor", mRotateAnchorRotAxis[0] == 0.0f && mRotateAnchorRotAxis[1] == 1.0f && mRotateAnchorRotAxis[2] == 0.0f)) {
        mRotateAnchorRotAxis[0] = 0.0f; mRotateAnchorRotAxis[1] = 1.0f; mRotateAnchorRotAxis[2] = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Z##FDORotAnchor", mRotateAnchorRotAxis[0] == 0.0f && mRotateAnchorRotAxis[1] == 0.0f && mRotateAnchorRotAxis[2] == 1.0f)) {
        mRotateAnchorRotAxis[0] = 0.0f; mRotateAnchorRotAxis[1] = 0.0f; mRotateAnchorRotAxis[2] = 1.0f;
    }

    ImGui::Spacing();

    // Angle input
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Angle:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("degrees##FDORotAnchor", &mRotateAnchorAngleDeg, 1.0f, 10.0f, "%.2f");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Apply button
    if (ImGui::Button("Apply FDO Combined Surgery", ImVec2(-1, 30))) {
        if (mSelectedRotateAnchorMuscle.empty()) {
            LOG_INFO("[Surgery] Error: No muscle selected!");
        } else if (mSelectedRotateAnchorIndex < 0) {
            LOG_INFO("[Surgery] Error: No anchor selected!");
        } else {
            // Normalize vectors
            Eigen::Vector3d search_dir(mRotateAnchorSearchDir[0], mRotateAnchorSearchDir[1], mRotateAnchorSearchDir[2]);
            Eigen::Vector3d rot_axis(mRotateAnchorRotAxis[0], mRotateAnchorRotAxis[1], mRotateAnchorRotAxis[2]);

            double search_norm = search_dir.norm();
            double axis_norm = rot_axis.norm();

            if (search_norm < 1e-6) {
                LOG_INFO("[Surgery] Error: Search direction cannot be zero!");
            } else if (axis_norm < 1e-6) {
                LOG_INFO("[Surgery] Error: Rotation axis cannot be zero!");
            } else {
                search_dir.normalize();
                rot_axis.normalize();
                double angle_rad_d = mRotateAnchorAngleDeg * M_PI / 180.0;

                // Execute FDO Combined Surgery (target bodynode obtained from anchor)
                if (executeFDO(mSelectedRotateAnchorMuscle, mSelectedRotateAnchorIndex,
                              search_dir, rot_axis, angle_rad_d)) {
                    // Record operation if recording
                    if (mRecordingSurgery) {
                        auto op = std::make_unique<FDOCombinedOp>(
                            mSelectedRotateAnchorMuscle, mSelectedRotateAnchorIndex,
                            search_dir, rot_axis, angle_rad_d);
                        recordOperation(std::move(op));
                    }
                }
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Execute FDO surgery:\n1. Rotate child joint of target bone\n2. Rotate anchors ON target bone (with positional filter)");
    }
}

void SurgeryPanel::drawResetSkeletonSection() {
    ImGui::TextWrapped("Reset skeleton to original configuration.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    if (ImGui::Button("Reset Skeleton", ImVec2(-1, 30))) {
        resetSkeleton();
        LOG_INFO("[Surgery] Skeleton reset to original configuration");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset skeleton joint offsets and transforms to original configuration");
    }
}

void SurgeryPanel::drawSaveSkeletonConfigSection() {
    ImGui::TextWrapped("Save skeleton configuration to file.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    // Format selection
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Directory: @data/skeleton/");
    ImGui::SameLine();
    if (ImGui::RadioButton("YAML##SkeletonFormat", mSaveAsYAML)) {
        mSaveAsYAML = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("XML##SkeletonFormat", !mSaveAsYAML)) {
        mSaveAsYAML = false;
    }
    ImGui::Spacing();

    // Determine file extension based on format
    const char* extension = mSaveAsYAML ? ".yaml" : ".xml";

    // Construct full filename with extension
    std::string filenameWithExt = std::string(mSaveSkeletonFilename);
    // Remove existing extension if present
    size_t lastDot = filenameWithExt.find_last_of('.');
    if (lastDot != std::string::npos) {
        filenameWithExt = filenameWithExt.substr(0, lastDot);
    }
    filenameWithExt += extension;

    // For save destinations, resolve directory (not file) to avoid "not found" errors
    std::string dirUri = "@data/skeleton";
    auto resolvedDir = rm::getManager().resolveDirCreate(dirUri);
    std::string resolvedPath = (resolvedDir / filenameWithExt).string();

    // Save button
    if (ImGui::Button("Save##Skeleton")) {
        LOG_INFO("[Surgery] Skeleton Save button clicked! Resolved path: " << resolvedPath);

        try {
            exportSkeleton(resolvedPath);

            // Update muscle filename to match the skeleton filename (without extension)
            std::string skelNameWithoutExt = std::string(mSaveSkeletonFilename);
            size_t skelDot = skelNameWithoutExt.find_last_of('.');
            if (skelDot != std::string::npos) {
                skelNameWithoutExt = skelNameWithoutExt.substr(0, skelDot);
            }
            strncpy(mSaveMuscleFilename, skelNameWithoutExt.c_str(), sizeof(mSaveMuscleFilename) - 1);
            mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

            // Record operation if recording
            if (mRecordingSurgery) {
                std::string recordUriPath = std::string("@data/skeleton/") + filenameWithExt;
                auto op = std::make_unique<ExportSkeletonOp>(recordUriPath);
                recordOperation(std::move(op));
                LOG_INFO("[Surgery] Recorded skeleton export with name: " << recordUriPath);
            }

            LOG_INFO("[Surgery] Skeleton configuration saved to: " << resolvedPath);
            LOG_INFO("[Surgery] Muscle filename updated to match: " << mSaveMuscleFilename);
        } catch (const std::exception& e) {
            LOG_ERROR("[Surgery] Error saving skeleton configuration: " << e.what());
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Save skeleton properties to %s", resolvedPath.c_str());
    }
    ImGui::SameLine();

    // Text input for leaf filename only
    ImGui::Text("Filename:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##save_skeleton_filename", mSaveSkeletonFilename, sizeof(mSaveSkeletonFilename));

    // Check if resolved file exists
    bool fileExists = std::filesystem::exists(resolvedPath);

    // Show warning if file exists
    if (fileExists) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Warning: File already exists and will be overwritten!");
    }

    ImGui::Spacing();
}

// ============================================================================
// SURGERY SCRIPT RECORDING AND EXECUTION
// ============================================================================

void SurgeryPanel::startRecording() {
    mRecordedOperations.clear();
    mRecordingSurgery = true;
    LOG_INFO("[Surgery Recording] Started recording surgery operations");
}

void SurgeryPanel::stopRecording() {
    mRecordingSurgery = false;
    LOG_INFO("[Surgery Recording] Stopped recording. Captured " << mRecordedOperations.size() << " operation(s)");
}

void SurgeryPanel::exportRecording(const std::string& filepath) {
    if (mRecordedOperations.empty()) {
        LOG_WARN("[Surgery Recording] No operations to export!");
        return;
    }
    
    try {
        SurgeryScript::saveToFile(mRecordedOperations, filepath, "Recorded surgery operations");
        LOG_INFO("[Surgery Recording] Exported " << mRecordedOperations.size() << " operation(s) to " << filepath);
    } catch (const std::exception& e) {
        LOG_ERROR("[Surgery Recording] Export failed: " << e.what());
    }
}

void SurgeryPanel::recordOperation(std::unique_ptr<SurgeryOperation> op) {
    if (!mRecordingSurgery) return;
    
    LOG_INFO("[Surgery Recording] Recorded: " << op->getDescription());
    mRecordedOperations.push_back(std::move(op));
}

void SurgeryPanel::loadSurgeryScript(const std::string& filepath) {
    try {
        mLoadedScript = SurgeryScript::loadFromFile(filepath);
        
        if (mLoadedScript.empty()) {
            LOG_WARN("[Surgery Script] No operations loaded from " << filepath);
            return;
        }
        
        LOG_INFO("[Surgery Script] Loaded " << mLoadedScript.size() << " operation(s) from " << filepath);
        
        mShowScriptPreview = true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("[Surgery Script] Failed to load: " << e.what());
    }
}

void SurgeryPanel::executeSurgeryScript(std::vector<std::unique_ptr<SurgeryOperation>>& ops) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery Script] Error: No character loaded!");
        return;
    }
    
    LOG_INFO("[Surgery Script] Executing " << ops.size() << " operation(s)...");
    
    int successCount = 0;
    int failCount = 0;
    
    for (size_t i = 0; i < ops.size(); ++i) {
        LOG_INFO("[Surgery Script] Operation " << (i + 1) << "/" << ops.size() << ": " << ops[i]->getDescription());
        
        bool success = ops[i]->execute(this);
        if (success) {
            successCount++;
        } else {
            failCount++;
            LOG_ERROR("[Surgery Script] Operation " << (i + 1) << " FAILED");
        }
    }
    
    LOG_INFO("[Surgery Script] Execution complete. Success: " << successCount << ", Failed: " << failCount);
}

void SurgeryPanel::showScriptPreview() {
    if (!mShowScriptPreview || mLoadedScript.empty()) return;
    
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(400, 200), ImGuiCond_FirstUseEver);
    
    if (ImGui::Begin("Surgery Script Preview", &mShowScriptPreview)) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Script Preview");
        ImGui::Separator();
        
        ImGui::Text("Total operations: %zu", mLoadedScript.size());
        ImGui::Spacing();
        
        ImGui::BeginChild("ScriptOperations", ImVec2(0, 250), true);
        for (size_t i = 0; i < mLoadedScript.size(); ++i) {
            ImGui::Text("%zu. %s", i + 1, mLoadedScript[i]->getDescription().c_str());
        }
        ImGui::EndChild();
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        if (ImGui::Button("Execute Script", ImVec2(150, 40))) {
            executeSurgeryScript(mLoadedScript);
            mShowScriptPreview = false;
            mLoadedScript.clear();
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Cancel", ImVec2(150, 40))) {
            mShowScriptPreview = false;
            mLoadedScript.clear();
        }
    }
    ImGui::End();
}

// ============================================================================
// FDO SURGERY OVERRIDE METHODS
// ============================================================================

bool SurgeryPanel::rotateJointOffset(const std::string& joint_name, const Eigen::Vector3d& axis,
                                    double angle, bool preserve_position) {
    // Call base class implementation
    bool success = SurgeryExecutor::rotateJointOffset(joint_name, axis, angle, preserve_position);

    // No GUI-specific logic needed for joint rotation
    // (skeleton updates automatically)

    return success;
}

bool SurgeryPanel::rotateAnchorPoints(const std::string& muscle_name, int ref_anchor_index,
                                     const Eigen::Vector3d& search_direction,
                                     const Eigen::Vector3d& rotation_axis, double angle) {
    // Call base class implementation
    bool success = SurgeryExecutor::rotateAnchorPoints(muscle_name, ref_anchor_index,
                                                       search_direction, rotation_axis, angle);

    // Invalidate muscle cache for visual update
    if (success && mCharacter && mShapeRenderer) {
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            // Invalidate cache for all muscles that might have been modified
            mShapeRenderer->invalidateMuscleCache(m);
        }
    }

    return success;
}

void SurgeryPanel::exportSkeleton(const std::string& path) {
    // Call base class implementation
    SurgeryExecutor::exportSkeleton(path);
}

bool SurgeryPanel::executeFDO(const std::string& ref_muscle, int ref_anchor_index,
                             const Eigen::Vector3d& search_dir, const Eigen::Vector3d& rot_axis,
                             double angle) {
    // Call base class implementation
    bool success = SurgeryExecutor::executeFDO(ref_muscle, ref_anchor_index,
                                                search_dir, rot_axis, angle);

    // Invalidate all muscle caches for visual update after FDO
    if (success && mCharacter && mShapeRenderer) {
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            mShapeRenderer->invalidateMuscleCache(m);
        }
    }

    return success;
}

void SurgeryPanel::resetSkeleton() {
    // Note: SurgeryPanel doesn't have direct access to mWorld and loadCharacter
    // This operation must be performed by the parent application
    LOG_WARN("[SurgeryPanel] resetSkeleton called - this operation requires the parent "
             "application to reload the character. Please use the application's reset function.");

    // If we have a character, we can at least log what we would reset
    if (mCharacter) {
        auto skel = mCharacter->getSkeleton();
        if (skel) {
            LOG_INFO("[SurgeryPanel] Skeleton '" << skel->getName() << "' needs reset by parent application");
        }
    }
}


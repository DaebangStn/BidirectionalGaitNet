#include "SurgeryPanel.h"
#include "SurgeryScript.h"
#include <algorithm>
#include <iostream>
#include <cstring>

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
    std::strncpy(mSaveMuscleFilename, "data/muscle_modified.xml", sizeof(mSaveMuscleFilename));
    mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

    // Initialize filter buffers
    mDistributeFilterBuffer[0] = '\0';
    mRelaxFilterBuffer[0] = '\0';
    mAnchorCandidateFilterBuffer[0] = '\0';
    mAnchorReferenceFilterBuffer[0] = '\0';
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
    
    drawScriptControlsSection();
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    drawResetMusclesSection();
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Distribute Passive Force")) {
        drawDistributePassiveForceSection();
    }
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Relax Passive Force")) {
        drawRelaxPassiveForceSection();
    }
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Anchor Point Manipulation", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawAnchorManipulationSection();
    }
    ImGui::Spacing();
    
    if (ImGui::CollapsingHeader("Save Muscle Config", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawSaveMuscleConfigSection();
    }
    
    ImGui::End();
    
    // Show script preview popup if active
    showScriptPreview();
}

// ============================================================================
// SURGERY SCRIPT CONTROLS
// ============================================================================

void SurgeryPanel::drawScriptControlsSection() {
    if (ImGui::CollapsingHeader("Load & Record Surgery Script", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
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
        
        ImGui::Unindent();
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
    ImGui::Indent();

    ImGui::TextWrapped("Select muscles, then choose one as reference to copy its passive force coefficient to others.");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        ImGui::Unindent();
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        ImGui::Unindent();
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
            std::cout << "[Surgery] Error: No muscles selected!" << std::endl;
        } else if (mDistributeRefMuscle.empty()) {
            std::cout << "[Surgery] Error: No reference muscle selected!" << std::endl;
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

    ImGui::Unindent();
}

void SurgeryPanel::drawRelaxPassiveForceSection() {
    ImGui::Indent();

    ImGui::TextWrapped("Select muscles to relax (reduce passive forces).");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        ImGui::Unindent();
        return;
    }

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        ImGui::Unindent();
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
            std::cout << "[Surgery] Error: No muscles selected!" << std::endl;
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

    ImGui::Unindent();
}

void SurgeryPanel::drawSaveMuscleConfigSection() {
    ImGui::Indent();
    
    ImGui::TextWrapped("Save current muscle configuration to an XML file.");
    ImGui::Spacing();
    
    ImGui::Text("Output Filename:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##save_muscle_filename", mSaveMuscleFilename, sizeof(mSaveMuscleFilename));
    
    ImGui::Spacing();
    
    if (ImGui::Button("Save to File", ImVec2(-1, 30))) {
        if (!mSavingMuscle) {
            mSavingMuscle = true;
            if (mCharacter) {
                try {
                    exportMuscles(mSaveMuscleFilename);
                    
                    if (mRecordingSurgery) {
                        auto op = std::make_unique<ExportMusclesOp>(
                            std::string(mSaveMuscleFilename));
                        recordOperation(std::move(op));
                    }
                } catch (const std::exception& e) {
                    std::cout << "[Surgery] Error saving muscle configuration: " << e.what() << std::endl;
                }
            } else {
                std::cout << "[Surgery] Error: No character loaded!" << std::endl;
            }
            mSavingMuscle = false;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Save muscle properties to the specified XML file");
    }
    
    ImGui::Unindent();
}

void SurgeryPanel::drawAnchorManipulationSection() {
    ImGui::Indent();
    
    ImGui::TextWrapped("Select candidate and reference muscles, then manipulate anchor points.");
    ImGui::Spacing();
    
    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        ImGui::Unindent();
        return;
    }
    
    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) {
        ImGui::TextDisabled("No muscles found");
        ImGui::Unindent();
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
                            std::cout << "[Surgery] Body node already exists in this anchor!" << std::endl;
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
            std::cout << "[Surgery] Error: No candidate muscle selected!" << std::endl;
        } else if (mSelectedCandidateAnchorIndex < 0) {
            std::cout << "[Surgery] Error: No anchor selected for removal!" << std::endl;
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
                    std::cout << "[Surgery] Error: Cannot remove anchor - muscle must have at least 2 anchors!" << std::endl;
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
            std::cout << "[Surgery] Error: No candidate muscle selected!" << std::endl;
        } else if (mAnchorReferenceMuscle.empty()) {
            std::cout << "[Surgery] Error: No reference muscle selected!" << std::endl;
        } else if (mSelectedReferenceAnchorIndex < 0) {
            std::cout << "[Surgery] Error: No reference anchor selected!" << std::endl;
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
    
    ImGui::Unindent();
}

// ============================================================================
// SURGERY SCRIPT RECORDING AND EXECUTION
// ============================================================================

void SurgeryPanel::startRecording() {
    mRecordedOperations.clear();
    mRecordingSurgery = true;
    std::cout << "[Surgery Recording] Started recording surgery operations" << std::endl;
}

void SurgeryPanel::stopRecording() {
    mRecordingSurgery = false;
    std::cout << "[Surgery Recording] Stopped recording. Captured " 
              << mRecordedOperations.size() << " operation(s)" << std::endl;
}

void SurgeryPanel::exportRecording(const std::string& filepath) {
    if (mRecordedOperations.empty()) {
        std::cerr << "[Surgery Recording] No operations to export!" << std::endl;
        return;
    }
    
    try {
        SurgeryScript::saveToFile(mRecordedOperations, filepath, "Recorded surgery operations");
        std::cout << "[Surgery Recording] Exported " << mRecordedOperations.size() 
                  << " operation(s) to " << filepath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Surgery Recording] Export failed: " << e.what() << std::endl;
    }
}

void SurgeryPanel::recordOperation(std::unique_ptr<SurgeryOperation> op) {
    if (!mRecordingSurgery) return;
    
    std::cout << "[Surgery Recording] Recorded: " << op->getDescription() << std::endl;
    mRecordedOperations.push_back(std::move(op));
}

void SurgeryPanel::loadSurgeryScript(const std::string& filepath) {
    try {
        mLoadedScript = SurgeryScript::loadFromFile(filepath);
        
        if (mLoadedScript.empty()) {
            std::cerr << "[Surgery Script] No operations loaded from " << filepath << std::endl;
            return;
        }
        
        std::cout << "[Surgery Script] Loaded " << mLoadedScript.size() 
                  << " operation(s) from " << filepath << std::endl;
        
        mShowScriptPreview = true;
        
    } catch (const std::exception& e) {
        std::cerr << "[Surgery Script] Failed to load: " << e.what() << std::endl;
    }
}

void SurgeryPanel::executeSurgeryScript(std::vector<std::unique_ptr<SurgeryOperation>>& ops) {
    if (!mCharacter) {
        std::cerr << "[Surgery Script] Error: No character loaded!" << std::endl;
        return;
    }
    
    std::cout << "[Surgery Script] Executing " << ops.size() << " operation(s)..." << std::endl;
    
    int successCount = 0;
    int failCount = 0;
    
    for (size_t i = 0; i < ops.size(); ++i) {
        std::cout << "[Surgery Script] Operation " << (i + 1) << "/" << ops.size() 
                  << ": " << ops[i]->getDescription() << std::endl;
        
        bool success = ops[i]->execute(this);
        if (success) {
            successCount++;
        } else {
            failCount++;
            std::cerr << "[Surgery Script] Operation " << (i + 1) << " FAILED" << std::endl;
        }
    }
    
    std::cout << "[Surgery Script] Execution complete. Success: " << successCount 
              << ", Failed: " << failCount << std::endl;
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


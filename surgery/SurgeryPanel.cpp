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
{
    SurgeryExecutor::mCharacter = character;

    strncpy(mRecordingPathBuffer, mRecordingScriptPath.c_str(), sizeof(mRecordingPathBuffer) - 1);
    mRecordingPathBuffer[sizeof(mRecordingPathBuffer) - 1] = '\0';
    strncpy(mLoadPathBuffer, mLoadScriptPath.c_str(), sizeof(mLoadPathBuffer) - 1);
    mLoadPathBuffer[sizeof(mLoadPathBuffer) - 1] = '\0';

    std::strncpy(mSaveMuscleFilename, "muscle_modified", sizeof(mSaveMuscleFilename));
    mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

    std::strncpy(mSaveSkeletonFilename, "skeleton_modified", sizeof(mSaveSkeletonFilename) - 1);
    mSaveSkeletonFilename[sizeof(mSaveSkeletonFilename) - 1] = '\0';

    // SEMLS init
    strncpy(mPatientPathBuffer, "", sizeof(mPatientPathBuffer));
    strncpy(mSurgeryConfigPathBuffer, "data/config/surgery_panel.yaml", sizeof(mSurgeryConfigPathBuffer) - 1);
    mSurgeryConfigPathBuffer[sizeof(mSurgeryConfigPathBuffer) - 1] = '\0';

    loadSurgeryConfig(mSurgeryConfigPathBuffer);

    // Optimizer defaults
    mOptConfig.maxIterations = 100;
    mOptConfig.minRatio = 0.5;
    mOptConfig.maxRatio = 2.0;
    mOptConfig.gridSearchBegin = 0.5;
    mOptConfig.gridSearchEnd = 2.0;
    mOptConfig.gridSearchInterval = 0.05;
    mOptConfig.lambdaRatioReg = 0.0;
    mOptConfig.lambdaTorqueReg = 0.0;
    mOptConfig.lambdaLineReg = 0.0;
    mOptConfig.outerIterations = 1;
    mOptConfig.verbose = true;
}

void SurgeryPanel::setCharacter(Character* character) {
    mCharacter = character;
    SurgeryExecutor::mCharacter = character;
}

void SurgeryPanel::onPIDChanged(const std::string& pid, const std::string& visit) {
    mPID = pid;
    mVisit = visit;

    if (pid.empty()) {
        mPatientLoaded = false;
        return;
    }

    // Load patient metadata from PID path (always from op1)
    std::string metaUri = "@pid:" + pid + "/op1/metadata.yaml";
    std::string resolvedPath = rm::getManager().resolve(metaUri);
    if (!resolvedPath.empty() && std::filesystem::exists(resolvedPath)) {
        loadPatientMetadata(resolvedPath);
    } else {
        LOG_INFO("[SEMLS] No metadata found for PID " << pid << "/op1");
    }

    // Load ROM angles from op1/rom.yaml
    std::string romUri = "@pid:" + pid + "/op1/rom.yaml";
    std::string romPath = rm::getManager().resolve(romUri);
    if (!romPath.empty() && std::filesystem::exists(romPath)) {
        try {
            YAML::Node rom = YAML::LoadFile(romPath);
            auto loadSideAngles = [&](const YAML::Node& side, TALConfig& tal, DHLConfig& dhl) {
                if (!side) return;
                // TAL: dorsiflexion angles
                if (side["ankle"]) {
                    auto& ankle = side["ankle"];
                    for (auto& entry : tal.rom_trials) {
                        if (entry.trial_name.find("dorsi_k0") != std::string::npos) {
                            if (ankle["dorsiflexion_knee0_r2"] && !ankle["dorsiflexion_knee0_r2"].IsNull())
                                entry.angle_deg = ankle["dorsiflexion_knee0_r2"].as<float>();
                        } else if (entry.trial_name.find("dorsi_k90") != std::string::npos) {
                            if (ankle["dorsiflexion_knee90_r2"] && !ankle["dorsiflexion_knee90_r2"].IsNull())
                                entry.angle_deg = ankle["dorsiflexion_knee90_r2"].as<float>();
                        }
                    }
                }
                // DHL: popliteal angle (bilateral)
                if (side["knee"] && side["knee"]["popliteal_bilateral"] && !side["knee"]["popliteal_bilateral"].IsNull()) {
                    dhl.popliteal_angle_deg = side["knee"]["popliteal_bilateral"].as<float>();
                }
            };
            if (rom["rom"]) {
                loadSideAngles(rom["rom"]["left"], mSEMLSConfig.tal_left, mSEMLSConfig.dhl_left);
                loadSideAngles(rom["rom"]["right"], mSEMLSConfig.tal_right, mSEMLSConfig.dhl_right);
            }
            LOG_INFO("[SEMLS] Loaded ROM angles from: " << romPath);
        } catch (const std::exception& e) {
            LOG_ERROR("[SEMLS] Failed to load ROM: " << e.what());
        }
    }

    // FDO angle: difference of hip anteversion between pre and op1
    std::string preRomUri = "@pid:" + pid + "/pre/rom.yaml";
    std::string preRomPath = rm::getManager().resolve(preRomUri);
    if (!preRomPath.empty() && std::filesystem::exists(preRomPath) &&
        !romPath.empty() && std::filesystem::exists(romPath)) {
        try {
            YAML::Node preRom = YAML::LoadFile(preRomPath);
            YAML::Node op1Rom = YAML::LoadFile(romPath);
            auto loadFDOAngle = [](const YAML::Node& preSide, const YAML::Node& op1Side) -> float {
                if (!preSide || !preSide["hip"] || !preSide["hip"]["anteversion"] || preSide["hip"]["anteversion"].IsNull())
                    return 0.0f;
                if (!op1Side || !op1Side["hip"] || !op1Side["hip"]["anteversion"] || op1Side["hip"]["anteversion"].IsNull())
                    return 0.0f;
                float pre = preSide["hip"]["anteversion"].as<float>();
                float op1 = op1Side["hip"]["anteversion"].as<float>();
                return pre - op1;
            };
            if (preRom["rom"] && op1Rom["rom"]) {
                mSEMLSConfig.fdo_left.angle_deg = loadFDOAngle(preRom["rom"]["left"], op1Rom["rom"]["left"]);
                mSEMLSConfig.fdo_right.angle_deg = loadFDOAngle(preRom["rom"]["right"], op1Rom["rom"]["right"]);
                LOG_INFO("[SEMLS] FDO angles from anteversion diff: L=" << mSEMLSConfig.fdo_left.angle_deg
                         << " R=" << mSEMLSConfig.fdo_right.angle_deg);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("[SEMLS] Failed to load FDO anteversion: " << e.what());
        }
    }
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
    if (success) invalidateMuscleCache(muscle);
    return success;
}

bool SurgeryPanel::editAnchorWeights(const std::string& muscle, int anchor_index,
                                    const std::vector<double>& weights) {
    bool success = SurgeryExecutor::editAnchorWeights(muscle, anchor_index, weights);
    if (success) invalidateMuscleCache(muscle);
    return success;
}

bool SurgeryPanel::addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                                      const std::string& bodynode_name, double weight) {
    bool success = SurgeryExecutor::addBodyNodeToAnchor(muscle, anchor_index, bodynode_name, weight);
    if (success) invalidateMuscleCache(muscle);
    return success;
}

bool SurgeryPanel::removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index,
                                           int bodynode_index) {
    bool success = SurgeryExecutor::removeBodyNodeFromAnchor(muscle, anchor_index, bodynode_index);
    if (success) invalidateMuscleCache(muscle);
    return success;
}

bool SurgeryPanel::removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) {
    bool success = SurgeryExecutor::removeAnchorFromMuscle(muscleName, anchorIndex);
    if (success) invalidateMuscleCache(muscleName);
    return success;
}

bool SurgeryPanel::copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex,
                                     const std::string& toMuscle) {
    bool success = SurgeryExecutor::copyAnchorToMuscle(fromMuscle, fromIndex, toMuscle);
    if (success) invalidateMuscleCache(toMuscle);
    return success;
}

bool SurgeryPanel::rotateJointOffset(const std::string& joint_name, const Eigen::Vector3d& axis,
                                    double angle, bool preserve_position) {
    return SurgeryExecutor::rotateJointOffset(joint_name, axis, angle, preserve_position);
}

bool SurgeryPanel::rotateAnchorPoints(const std::string& muscle_name, int ref_anchor_index,
                                     const Eigen::Vector3d& search_direction,
                                     const Eigen::Vector3d& rotation_axis, double angle) {
    bool success = SurgeryExecutor::rotateAnchorPoints(muscle_name, ref_anchor_index,
                                                       search_direction, rotation_axis, angle);
    if (success && mCharacter && mShapeRenderer) {
        for (auto m : mCharacter->getMuscles())
            mShapeRenderer->invalidateMuscleCache(m);
    }
    return success;
}

void SurgeryPanel::exportSkeleton(const std::string& path) {
    SurgeryExecutor::exportSkeleton(path);
}

bool SurgeryPanel::executeFDO(const std::string& ref_muscle, int ref_anchor_index,
                             const Eigen::Vector3d& search_dir, const Eigen::Vector3d& rot_axis,
                             double angle) {
    bool success = SurgeryExecutor::executeFDO(ref_muscle, ref_anchor_index,
                                                search_dir, rot_axis, angle);
    if (success && mCharacter && mShapeRenderer) {
        for (auto m : mCharacter->getMuscles())
            mShapeRenderer->invalidateMuscleCache(m);
    }
    return success;
}

void SurgeryPanel::resetSkeleton() {
    LOG_WARN("[SurgeryPanel] resetSkeleton called - requires parent application to reload character.");
    if (mCharacter) {
        auto skel = mCharacter->getSkeleton();
        if (skel) {
            LOG_INFO("[SurgeryPanel] Skeleton '" << skel->getName() << "' needs reset by parent application");
        }
    }
}

// ============================================================================
// MAIN SURGERY PANEL UI - SEMLS Tabs
// ============================================================================

void SurgeryPanel::drawSurgeryPanel(bool* show_panel, int window_height) {
    ImGui::SetNextWindowSize(ImVec2(450, window_height - 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(450, 10), ImGuiCond_FirstUseEver);
    mDrawFDOAnchors = false;

    ImGui::Begin("SEMLS Surgery", show_panel);

    if (ImGui::BeginTabBar("SEMLSTabs")) {
        if (ImGui::BeginTabItem("Project")) {
            drawProjectTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("TAL")) {
            drawTALTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("DHL")) {
            drawDHLTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("FDO")) {
            drawFDOTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("RFT")) {
            drawRFTTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Optimizer")) {
            drawOptimizerTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
    showScriptPreview();
}

void SurgeryPanel::drawSurgeryContent() {
    mDrawFDOAnchors = false;

    if (ImGui::BeginTabBar("SEMLSTabs")) {
        if (ImGui::BeginTabItem("Project")) {
            drawProjectTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("TAL")) {
            drawTALTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("DHL")) {
            drawDHLTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("FDO")) {
            drawFDOTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("RFT")) {
            drawRFTTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Optimizer")) {
            drawOptimizerTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    showScriptPreview();
}

// ============================================================================
// SEMLS CONFIG LOADING
// ============================================================================

static void loadTALSide(const YAML::Node& node, SurgeryPanel::TALConfig& cfg) {
    if (!node) return;
    if (node["rom_trials"]) {
        cfg.rom_trials.clear();
        for (const auto& t : node["rom_trials"]) {
            SurgeryPanel::ROMTrialEntry entry;
            entry.trial_name = t["trial"].as<std::string>();
            if (t["angle"]) entry.angle_deg = t["angle"].as<float>();
            cfg.rom_trials.push_back(entry);
        }
    }
    if (node["muscles"]) {
        cfg.muscles.clear();
        for (const auto& m : node["muscles"]) cfg.muscles.push_back(m.as<std::string>());
    }
}

static void loadDHLSide(const YAML::Node& node, SurgeryPanel::DHLConfig& cfg) {
    if (!node) return;
    if (node["target_muscle"]) cfg.target_muscle = node["target_muscle"].as<std::string>();
    if (node["donor_muscle"]) cfg.donor_muscle = node["donor_muscle"].as<std::string>();
    if (node["donor_anchor"]) cfg.donor_anchor = node["donor_anchor"].as<int>();
    if (node["remove_anchors"]) {
        cfg.remove_anchors.clear();
        for (const auto& a : node["remove_anchors"]) cfg.remove_anchors.push_back(a.as<int>());
    }
    if (node["rom_trials"]) {
        cfg.rom_trials.clear();
        for (const auto& t : node["rom_trials"]) cfg.rom_trials.push_back(t.as<std::string>());
    }
    if (node["muscles"]) {
        cfg.muscles.clear();
        for (const auto& m : node["muscles"]) cfg.muscles.push_back(m.as<std::string>());
    }
    if (node["popliteal_angle"]) cfg.popliteal_angle_deg = node["popliteal_angle"].as<float>();
}

static void loadFDOSide(const YAML::Node& node, SurgeryPanel::FDOConfig& cfg) {
    if (!node) return;
    if (node["joint"]) cfg.joint = node["joint"].as<std::string>();
    if (node["ref_muscle"]) cfg.ref_muscle = node["ref_muscle"].as<std::string>();
    if (node["ref_anchor"]) cfg.ref_anchor = node["ref_anchor"].as<int>();
    if (node["rotation_axis"] && node["rotation_axis"].size() == 3) {
        for (int i = 0; i < 3; ++i) cfg.rotation_axis[i] = node["rotation_axis"][i].as<float>();
    }
    if (node["search_direction"] && node["search_direction"].size() == 3) {
        for (int i = 0; i < 3; ++i) cfg.search_direction[i] = node["search_direction"][i].as<float>();
    }
    if (node["angle_deg"]) cfg.angle_deg = node["angle_deg"].as<float>();
}

static void loadRFTSide(const YAML::Node& node, SurgeryPanel::RFTConfig& cfg) {
    if (!node) return;
    if (node["target_muscles"]) {
        cfg.target_muscles.clear();
        for (const auto& m : node["target_muscles"]) cfg.target_muscles.push_back(m.as<std::string>());
    }
    if (node["donor_muscle"]) cfg.donor_muscle = node["donor_muscle"].as<std::string>();
    if (node["remove_anchors"]) {
        cfg.remove_anchors.clear();
        for (const auto& a : node["remove_anchors"]) cfg.remove_anchors.push_back(a.as<int>());
    }
    if (node["copy_donor_anchors"]) {
        cfg.copy_donor_anchors.clear();
        for (const auto& a : node["copy_donor_anchors"]) cfg.copy_donor_anchors.push_back(a.as<int>());
    }
}

void SurgeryPanel::loadSurgeryConfig(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        if (config["tal"]) {
            loadTALSide(config["tal"]["left"], mSEMLSConfig.tal_left);
            loadTALSide(config["tal"]["right"], mSEMLSConfig.tal_right);
        }
        if (config["dhl"]) {
            loadDHLSide(config["dhl"]["left"], mSEMLSConfig.dhl_left);
            loadDHLSide(config["dhl"]["right"], mSEMLSConfig.dhl_right);
        }
        if (config["fdo"]) {
            loadFDOSide(config["fdo"]["left"], mSEMLSConfig.fdo_left);
            loadFDOSide(config["fdo"]["right"], mSEMLSConfig.fdo_right);
        }
        if (config["rft"]) {
            loadRFTSide(config["rft"]["left"], mSEMLSConfig.rft_left);
            loadRFTSide(config["rft"]["right"], mSEMLSConfig.rft_right);
        }
        mSurgeryConfigLoaded = true;
        LOG_INFO("[SEMLS] Loaded surgery config from: " << yaml_path);
    } catch (const std::exception& e) {
        LOG_ERROR("[SEMLS] Failed to load surgery config: " << e.what());
    }
}

void SurgeryPanel::loadPatientMetadata(const std::string& metadata_path) {
    try {
        mPatientMetadata = YAML::LoadFile(metadata_path);
        mPatientMetadataPath = metadata_path;
        mPatientLoaded = true;
        LOG_INFO("[SEMLS] Loaded patient metadata from: " << metadata_path);

        // Set enabled flags from patient surgery list
        if (mPatientMetadata["surgery"]) {
            // Reset all to false, then enable based on patient data
            auto& cfg = mSEMLSConfig;
            cfg.tal_left.enabled = cfg.tal_right.enabled = false;
            cfg.dhl_left.enabled = cfg.dhl_right.enabled = false;
            cfg.fdo_left.enabled = cfg.fdo_right.enabled = false;
            cfg.rft_left.enabled = cfg.rft_right.enabled = false;

            for (const auto& s : mPatientMetadata["surgery"]) {
                std::string name = s.as<std::string>();
                bool is_left  = name.size() > 3 && name.substr(name.size() - 3) == "_Lt";
                bool is_right = name.size() > 3 && name.substr(name.size() - 3) == "_Rt";
                std::string base = name;
                if (is_left || is_right) base = name.substr(0, name.size() - 3);

                if (base == "TAL") {
                    if (is_left)  cfg.tal_left.enabled = true;
                    if (is_right) cfg.tal_right.enabled = true;
                } else if (base == "DHL") {
                    if (is_left)  cfg.dhl_left.enabled = true;
                    if (is_right) cfg.dhl_right.enabled = true;
                } else if (base == "FDO" || base == "FVDO") {
                    if (is_left)  cfg.fdo_left.enabled = true;
                    if (is_right) cfg.fdo_right.enabled = true;
                } else if (base == "RFT") {
                    if (is_left)  cfg.rft_left.enabled = true;
                    if (is_right) cfg.rft_right.enabled = true;
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[SEMLS] Failed to load patient metadata: " << e.what());
    }
}

// ============================================================================
// PROJECT TAB
// ============================================================================

void SurgeryPanel::drawProjectTab() {
    // Surgery config loading
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Surgery Configuration");
    ImGui::Separator();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##SurgeryConfigPath", mSurgeryConfigPathBuffer, sizeof(mSurgeryConfigPathBuffer));
    if (ImGui::Button("Load Surgery Config", ImVec2(-1, 25))) {
        loadSurgeryConfig(mSurgeryConfigPathBuffer);
    }
    if (mSurgeryConfigLoaded) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Config loaded");
    }

    ImGui::Spacing();

    // Patient metadata loading
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Patient Metadata");
    ImGui::Separator();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##PatientPath", mPatientPathBuffer, sizeof(mPatientPathBuffer));
    if (ImGui::Button("Load Patient", ImVec2(-1, 25))) {
        loadPatientMetadata(mPatientPathBuffer);
    }
    if (mPatientLoaded) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Patient loaded");
        if (mPatientMetadata["surgery"]) {
            ImGui::Text("Surgeries:");
            for (const auto& s : mPatientMetadata["surgery"]) {
                ImGui::BulletText("%s", s.as<std::string>().c_str());
            }
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Surgery summary
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), "Surgery Summary");
    ImGui::Separator();

    auto& cfg = mSEMLSConfig;
    ImGui::Checkbox("TAL Left",  &cfg.tal_left.enabled);  ImGui::SameLine();
    ImGui::Checkbox("TAL Right", &cfg.tal_right.enabled);
    ImGui::Checkbox("DHL Left",  &cfg.dhl_left.enabled);  ImGui::SameLine();
    ImGui::Checkbox("DHL Right", &cfg.dhl_right.enabled);
    ImGui::Checkbox("FDO Left",  &cfg.fdo_left.enabled);  ImGui::SameLine();
    ImGui::Checkbox("FDO Right", &cfg.fdo_right.enabled);
    ImGui::Checkbox("RFT Left",  &cfg.rft_left.enabled);  ImGui::SameLine();
    ImGui::Checkbox("RFT Right", &cfg.rft_right.enabled);

    ImGui::Spacing();

    // Execute all
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
    if (ImGui::Button("Do SEMLS", ImVec2(-1, 40))) {
        executeSEMLS();
    }
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Spacing();

    // Script controls, Save, Reset
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Script Recording");
    ImGui::Separator();
    drawScriptControlsSection();

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Save");
    ImGui::Separator();
    drawSaveMuscleConfigSection();
    ImGui::Spacing();
    drawSaveSkeletonConfigSection();

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Reset");
    ImGui::Separator();
    drawResetMusclesSection();
    ImGui::Spacing();
    drawResetSkeletonSection();
}

// ============================================================================
// TAL TAB
// ============================================================================

static void drawTALSideConfig(const char* label, SurgeryPanel::TALConfig& cfg) {
    ImGui::PushID(label);
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), "%s", label);
    ImGui::Indent();

    ImGui::Text("ROM Trials:");
    for (size_t i = 0; i < cfg.rom_trials.size(); ++i) {
        auto& entry = cfg.rom_trials[i];
        ImGui::PushID((int)i);
        ImGui::BulletText("%s", entry.trial_name.c_str());
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("deg", &entry.angle_deg, 5.0f, 10.0f, "%.1f");
        ImGui::PopID();
    }

    ImGui::Text("Muscles:");
    for (const auto& m : cfg.muscles) {
        ImGui::BulletText("%s", m.c_str());
    }

    ImGui::Unindent();
    ImGui::PopID();
}

void SurgeryPanel::drawTALTab() {
    ImGui::TextWrapped("Tendo Achilles Lengthening - Optimizes lt_rel (tendon slack length) "
                       "to match target dorsiflexion ROM angles.");
    ImGui::Spacing();

    if (!mSurgeryConfigLoaded) {
        ImGui::TextDisabled("Load surgery config first (Project tab)");
        return;
    }

    drawTALSideConfig("TAL Left", mSEMLSConfig.tal_left);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawTALSideConfig("TAL Right", mSEMLSConfig.tal_right);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    float w = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    if (ImGui::Button("TAL (left)", ImVec2(w, 28))) { executeTAL(true); }
    ImGui::SameLine();
    if (ImGui::Button("TAL (right)", ImVec2(w, 28))) { executeTAL(false); }
    ImGui::SameLine();
    if (ImGui::Button("TAL (both)", ImVec2(w, 28))) { executeTAL(true); executeTAL(false); }

    drawOptResultDisplay("tal", mTALResult);
}

// ============================================================================
// DHL TAB
// ============================================================================

static void drawDHLSideConfig(const char* label, SurgeryPanel::DHLConfig& cfg) {
    ImGui::PushID(label);
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), "%s", label);
    ImGui::Indent();

    ImGui::Checkbox("Do Anchor Transfer", &cfg.do_anchor_transfer);
    if (cfg.do_anchor_transfer) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Step 1: Anchor Transfer");
        ImGui::Text("Target: %s", cfg.target_muscle.c_str());
        ImGui::Text("Donor: %s (anchor #%d)", cfg.donor_muscle.c_str(), cfg.donor_anchor);
        ImGui::Text("Remove anchors:");
        ImGui::SameLine();
        for (size_t i = 0; i < cfg.remove_anchors.size(); ++i) {
            if (i > 0) ImGui::SameLine();
            ImGui::Text("%d", cfg.remove_anchors[i]);
        }
    }

    ImGui::Spacing();
    ImGui::Checkbox("Do Contracture Optimization", &cfg.do_contracture_opt);
    if (cfg.do_contracture_opt) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Step 2: Contracture Optimization");
        ImGui::Text("Popliteal angle:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("deg", &cfg.popliteal_angle_deg, 10.0f, 10.0f, "%.1f");

        ImGui::Text("ROM Trials:");
        for (const auto& t : cfg.rom_trials) {
            ImGui::BulletText("%s", t.c_str());
        }
        ImGui::Text("Muscles:");
        for (const auto& m : cfg.muscles) {
            ImGui::BulletText("%s", m.c_str());
        }
    }

    ImGui::Unindent();
    ImGui::PopID();
}

void SurgeryPanel::drawDHLTab() {
    ImGui::TextWrapped("Distal Hamstring Lengthening - Anchor transfer + contracture optimization "
                       "for popliteal angle.");
    ImGui::Spacing();

    if (!mSurgeryConfigLoaded) {
        ImGui::TextDisabled("Load surgery config first (Project tab)");
        return;
    }

    drawDHLSideConfig("DHL Left", mSEMLSConfig.dhl_left);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawDHLSideConfig("DHL Right", mSEMLSConfig.dhl_right);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    float w = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    if (ImGui::Button("DHL (left)", ImVec2(w, 28))) { executeDHL(true); }
    ImGui::SameLine();
    if (ImGui::Button("DHL (right)", ImVec2(w, 28))) { executeDHL(false); }
    ImGui::SameLine();
    if (ImGui::Button("DHL (both)", ImVec2(w, 28))) { executeDHL(true); executeDHL(false); }

    drawOptResultDisplay("dhl", mDHLResult);
}

// ============================================================================
// FDO TAB
// ============================================================================

static void drawFDOSideConfig(const char* label, SurgeryPanel::FDOConfig& cfg) {
    ImGui::PushID(label);
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), "%s", label);
    ImGui::Indent();

    ImGui::Text("Joint: %s", cfg.joint.c_str());
    ImGui::Text("Ref muscle: %s (anchor #%d)", cfg.ref_muscle.c_str(), cfg.ref_anchor);
    ImGui::Text("Rotation axis: [%.0f, %.0f, %.0f]",
                 cfg.rotation_axis[0], cfg.rotation_axis[1], cfg.rotation_axis[2]);
    ImGui::Text("Search dir: [%.0f, %.0f, %.0f]",
                 cfg.search_direction[0], cfg.search_direction[1], cfg.search_direction[2]);

    ImGui::Text("Angle:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputFloat("deg", &cfg.angle_deg, 5.0f, 10.0f, "%.1f");

    ImGui::Unindent();
    ImGui::PopID();
}

void SurgeryPanel::drawFDOTab() {
    ImGui::TextWrapped("Femoral Derotational Osteotomy - Rotates joint offset and anchor points.");
    ImGui::Spacing();

    if (!mSurgeryConfigLoaded) {
        ImGui::TextDisabled("Load surgery config first (Project tab)");
        return;
    }

    auto executeFDOSide = [this](FDOConfig& cfg) {
        double angle_rad = cfg.angle_deg * M_PI / 180.0;
        Eigen::Vector3d search_dir(cfg.search_direction[0], cfg.search_direction[1], cfg.search_direction[2]);
        Eigen::Vector3d rot_axis(cfg.rotation_axis[0], cfg.rotation_axis[1], cfg.rotation_axis[2]);
        search_dir.normalize();
        rot_axis.normalize();

        bool ok = executeFDO(cfg.ref_muscle, cfg.ref_anchor, search_dir, rot_axis, angle_rad);
        if (ok && mRecordingSurgery) {
            auto op = std::make_unique<FDOCombinedOp>(
                cfg.ref_muscle, cfg.ref_anchor, search_dir, rot_axis, angle_rad);
            recordOperation(std::move(op));
        }
    };

    // Update highlighted anchors for 3D rendering
    mDrawFDOAnchors = true;
    mHighlightedAnchors.clear();
    auto collectAnchors = [this](FDOConfig& cfg) {
        if (!mCharacter || cfg.ref_muscle.empty()) return;
        Eigen::Vector3d search_dir(cfg.search_direction[0], cfg.search_direction[1], cfg.search_direction[2]);
        if (search_dir.norm() < 1e-6) return;

        // Reference anchor
        Muscle* refM = mCharacter->getMuscleByName(cfg.ref_muscle);
        if (!refM) return;
        auto refAnchors = refM->GetAnchors();
        if (cfg.ref_anchor < (int)refAnchors.size()) {
            mHighlightedAnchors.push_back({refAnchors[cfg.ref_anchor]->GetPoint(), true});
        }

        // Affected anchors
        try {
            AnchorReference ref(cfg.ref_muscle, cfg.ref_anchor, 0);
            auto affected = computeAffectedAnchors(ref, search_dir);
            for (const auto& ar : affected) {
                Muscle* m = mCharacter->getMuscleByName(ar.muscle_name);
                if (!m) continue;
                auto anchors = m->GetAnchors();
                if (ar.anchor_index < (int)anchors.size()) {
                    mHighlightedAnchors.push_back({anchors[ar.anchor_index]->GetPoint(), false});
                }
            }
        } catch (const std::runtime_error&) {}
    };
    collectAnchors(mSEMLSConfig.fdo_left);
    collectAnchors(mSEMLSConfig.fdo_right);

    drawFDOSideConfig("FDO Left", mSEMLSConfig.fdo_left);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawFDOSideConfig("FDO Right", mSEMLSConfig.fdo_right);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    float w = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    if (ImGui::Button("FDO (left)", ImVec2(w, 28))) { executeFDOSide(mSEMLSConfig.fdo_left); }
    ImGui::SameLine();
    if (ImGui::Button("FDO (right)", ImVec2(w, 28))) { executeFDOSide(mSEMLSConfig.fdo_right); }
    ImGui::SameLine();
    if (ImGui::Button("FDO (both)", ImVec2(w, 28))) {
        executeFDOSide(mSEMLSConfig.fdo_left);
        executeFDOSide(mSEMLSConfig.fdo_right);
    }
}

// ============================================================================
// RFT TAB
// ============================================================================

static void drawRFTSideConfig(const char* label, SurgeryPanel::RFTConfig& cfg) {
    ImGui::PushID(label);
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), "%s", label);
    ImGui::Indent();

    ImGui::Text("Targets:");
    for (const auto& t : cfg.target_muscles) {
        ImGui::BulletText("%s", t.c_str());
    }
    ImGui::Text("Donor: %s", cfg.donor_muscle.c_str());
    ImGui::Text("Remove anchors:");
    ImGui::SameLine();
    for (size_t i = 0; i < cfg.remove_anchors.size(); ++i) {
        if (i > 0) ImGui::SameLine();
        ImGui::Text("%d", cfg.remove_anchors[i]);
    }
    ImGui::Text("Copy donor anchors:");
    ImGui::SameLine();
    for (size_t i = 0; i < cfg.copy_donor_anchors.size(); ++i) {
        if (i > 0) ImGui::SameLine();
        ImGui::Text("%d", cfg.copy_donor_anchors[i]);
    }

    ImGui::Unindent();
    ImGui::PopID();
}

void SurgeryPanel::drawRFTTab() {
    ImGui::TextWrapped("Rectus Femoris Transfer - Removes distal RF anchors and copies Gracilis anchors.");
    ImGui::Spacing();

    if (!mSurgeryConfigLoaded) {
        ImGui::TextDisabled("Load surgery config first (Project tab)");
        return;
    }

    drawRFTSideConfig("RFT Left", mSEMLSConfig.rft_left);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    drawRFTSideConfig("RFT Right", mSEMLSConfig.rft_right);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    float w = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    if (ImGui::Button("RFT (left)", ImVec2(w, 28))) { executeRFT(true); }
    ImGui::SameLine();
    if (ImGui::Button("RFT (right)", ImVec2(w, 28))) { executeRFT(false); }
    ImGui::SameLine();
    if (ImGui::Button("RFT (both)", ImVec2(w, 28))) { executeRFT(true); executeRFT(false); }
}

// ============================================================================
// SURGERY EXECUTION
// ============================================================================

ContractureOptResult SurgeryPanel::runContractureOpt(
    const std::string& search_group_name,
    const std::vector<std::string>& muscles,
    const std::vector<std::string>& trial_names,
    const std::vector<ROMTrialConfig>& rom_configs,
    const ContractureOptimizer::Config& opt_config)
{
    // Build inline muscle groups YAML (one opt group per muscle, one search group for all)
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;

    emitter << YAML::Key << "search_groups" << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << search_group_name << YAML::Value << YAML::BeginSeq;
    for (const auto& m : muscles) emitter << m;
    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;

    emitter << YAML::Key << "optimization_groups" << YAML::Value << YAML::BeginMap;
    for (const auto& m : muscles) {
        emitter << YAML::Key << m << YAML::Value << YAML::BeginSeq << m << YAML::EndSeq;
    }
    emitter << YAML::EndMap;

    emitter << YAML::Key << "grid_search_mapping" << YAML::Value << YAML::BeginSeq;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "trials" << YAML::Value << YAML::BeginSeq;
    for (const auto& t : trial_names) emitter << t;
    emitter << YAML::EndSeq;
    emitter << YAML::Key << "groups" << YAML::Value << YAML::BeginSeq << search_group_name << YAML::EndSeq;
    emitter << YAML::EndMap;
    emitter << YAML::EndSeq;

    emitter << YAML::EndMap;

    std::string tmp_path = "/tmp/semls_muscle_groups.yaml";
    { std::ofstream ofs(tmp_path); ofs << emitter.c_str(); }

    ContractureOptimizer optimizer;
    optimizer.loadMuscleGroups(tmp_path, mCharacter);

    YAML::Node tmp_config = YAML::LoadFile(tmp_path);
    if (tmp_config["grid_search_mapping"]) {
        std::vector<GridSearchMapping> mappings;
        for (const auto& entry : tmp_config["grid_search_mapping"]) {
            GridSearchMapping gm;
            if (entry["trials"]) for (const auto& t : entry["trials"]) gm.trials.push_back(t.as<std::string>());
            if (entry["groups"]) for (const auto& g : entry["groups"]) gm.groups.push_back(g.as<std::string>());
            mappings.push_back(gm);
        }
        optimizer.setGridSearchMapping(mappings);
    }

    auto result = optimizer.optimize(mCharacter, rom_configs, opt_config);
    std::filesystem::remove(tmp_path);
    return result;
}

bool SurgeryPanel::executeTAL(bool left) {
    auto& cfg = left ? mSEMLSConfig.tal_left : mSEMLSConfig.tal_right;
    const char* side = left ? "Left" : "Right";

    if (!mCharacter || cfg.rom_trials.empty() || cfg.muscles.empty()) {
        LOG_ERROR("[TAL " << side << "] Missing character, ROM trials, or muscles");
        return false;
    }

    LOG_INFO("[TAL " << side << "] Executing with " << cfg.muscles.size() << " muscles, "
             << cfg.rom_trials.size() << " ROM trials");

    // Load ROM configs
    std::vector<ROMTrialConfig> rom_configs;
    std::vector<std::string> trial_names;
    auto skel = mCharacter->getSkeleton();
    for (const auto& entry : cfg.rom_trials) {
        std::string path = "data/config/rom/" + entry.trial_name + ".yaml";
        try {
            auto rom = ContractureOptimizer::loadROMConfig(path, skel);
            if (entry.angle_deg != 0.0f) {
                rom.rom_angle = rom.cd_neg ? -entry.angle_deg : entry.angle_deg;
            }
            rom_configs.push_back(rom);
            trial_names.push_back(entry.trial_name);
        } catch (const std::exception& e) {
            LOG_ERROR("[TAL " << side << "] Failed to load ROM config: " << path << " - " << e.what());
        }
    }

    if (rom_configs.empty()) {
        LOG_ERROR("[TAL " << side << "] No valid ROM configs loaded");
        return false;
    }

    ContractureOptimizer::Config opt_config = mOptConfig;
    opt_config.paramType = ContractureOptimizer::OptParam::LT_REL;

    auto result = runContractureOpt("plantarflexor", cfg.muscles, trial_names, rom_configs, opt_config);
    mTALResult = result;

    if (mRecordingSurgery) {
        std::vector<ContractureOptOp::ROMTrialParam> rom_params;
        for (const auto& entry : cfg.rom_trials)
            rom_params.push_back({entry.trial_name, (double)entry.angle_deg});
        auto op = std::make_unique<ContractureOptOp>(
            "plantarflexor", cfg.muscles, rom_params, "lt_rel",
            opt_config.maxIterations, opt_config.minRatio, opt_config.maxRatio,
            opt_config.gridSearchBegin, opt_config.gridSearchEnd, opt_config.gridSearchInterval);
        recordOperation(std::move(op));
    }

    LOG_INFO("[TAL " << side << "] Optimization complete. " << result.muscle_results.size() << " muscles modified.");
    for (const auto& mr : result.muscle_results) {
        LOG_INFO("  " << mr.muscle_name << ": " << mr.lm_contract_before << " -> " << mr.lm_contract_after
                 << " (ratio=" << mr.ratio << ")");
    }

    return true;
}

bool SurgeryPanel::executeDHL(bool left) {
    auto& cfg = left ? mSEMLSConfig.dhl_left : mSEMLSConfig.dhl_right;
    const char* side = left ? "Left" : "Right";

    if (!mCharacter) {
        LOG_ERROR("[DHL " << side << "] No character loaded");
        return false;
    }

    // Step 1: Anchor transfer
    if (cfg.do_anchor_transfer) {
        int expectedAnchors = *std::max_element(cfg.remove_anchors.begin(), cfg.remove_anchors.end()) + 1;
        if (!validateAnchorCount(cfg.target_muscle, expectedAnchors)) return false;

        for (int anchor_idx : cfg.remove_anchors) {
            bool ok = removeAnchorFromMuscle(cfg.target_muscle, anchor_idx);
            if (!ok) {
                LOG_ERROR("[DHL " << side << "] Failed to remove anchor " << anchor_idx << " from " << cfg.target_muscle);
                return false;
            }
            if (mRecordingSurgery) {
                auto op = std::make_unique<RemoveAnchorOp>(cfg.target_muscle, anchor_idx);
                recordOperation(std::move(op));
            }
        }

        bool ok = copyAnchorToMuscle(cfg.donor_muscle, cfg.donor_anchor, cfg.target_muscle);
        if (!ok) {
            LOG_ERROR("[DHL " << side << "] Failed to copy anchor from " << cfg.donor_muscle);
            return false;
        }
        if (mRecordingSurgery) {
            auto op = std::make_unique<CopyAnchorOp>(cfg.donor_muscle, cfg.donor_anchor, cfg.target_muscle);
            recordOperation(std::move(op));
        }
        Muscle* dhlTarget = mCharacter->getMuscleByName(cfg.target_muscle);
        LOG_INFO("[DHL " << side << "] " << cfg.target_muscle << ": removed "
                 << cfg.remove_anchors.size() << " anchors, copied #" << cfg.donor_anchor
                 << " from " << cfg.donor_muscle << " (" << (dhlTarget ? (int)dhlTarget->GetAnchors().size() : -1) << " anchors)");
    }

    // Step 2: Contracture optimization (lm_contract)
    if (cfg.do_contracture_opt && !cfg.rom_trials.empty() && !cfg.muscles.empty()) {
        std::vector<ROMTrialConfig> rom_configs;
        auto skel = mCharacter->getSkeleton();
        for (const auto& trial_name : cfg.rom_trials) {
            std::string path = "data/config/rom/" + trial_name + ".yaml";
            try {
                auto rom = ContractureOptimizer::loadROMConfig(path, skel);
                if (cfg.popliteal_angle_deg != 0.0f) {
                    rom.rom_angle = rom.cd_neg ? -cfg.popliteal_angle_deg : cfg.popliteal_angle_deg;
                }
                rom_configs.push_back(rom);
            } catch (const std::exception& e) {
                LOG_ERROR("[DHL " << side << "] Failed to load ROM config: " << path);
            }
        }

        if (!rom_configs.empty()) {
            ContractureOptimizer::Config opt_config = mOptConfig;
            opt_config.paramType = ContractureOptimizer::OptParam::LM_CONTRACT;

            auto result = runContractureOpt("hamstrings", cfg.muscles, cfg.rom_trials, rom_configs, opt_config);
            mDHLResult = result;

            if (mRecordingSurgery) {
                std::vector<ContractureOptOp::ROMTrialParam> rom_params;
                for (const auto& trial_name : cfg.rom_trials)
                    rom_params.push_back({trial_name, (double)cfg.popliteal_angle_deg});
                auto op = std::make_unique<ContractureOptOp>(
                    "hamstrings", cfg.muscles, rom_params, "lm_contract",
                    opt_config.maxIterations, opt_config.minRatio, opt_config.maxRatio,
                    opt_config.gridSearchBegin, opt_config.gridSearchEnd, opt_config.gridSearchInterval);
                recordOperation(std::move(op));
            }

            LOG_INFO("[DHL " << side << "] Contracture optimization complete. " << result.muscle_results.size() << " muscles modified.");
            for (const auto& mr : result.muscle_results) {
                LOG_INFO("  " << mr.muscle_name << ": " << mr.lm_contract_before << " -> " << mr.lm_contract_after
                         << " (ratio=" << mr.ratio << ")");
            }
        }
    }

    return true;
}

bool SurgeryPanel::executeRFT(bool left) {
    auto& cfg = left ? mSEMLSConfig.rft_left : mSEMLSConfig.rft_right;
    const char* side = left ? "Left" : "Right";

    if (!mCharacter) {
        LOG_ERROR("[RFT " << side << "] No character loaded");
        return false;
    }

    int expectedAnchors = *std::max_element(cfg.remove_anchors.begin(), cfg.remove_anchors.end()) + 1;
    for (const auto& target : cfg.target_muscles) {
        if (!validateAnchorCount(target, expectedAnchors)) return false;
    }

    for (const auto& target : cfg.target_muscles) {
        // Remove anchors from target (descending order)
        for (int anchor_idx : cfg.remove_anchors) {
            bool ok = removeAnchorFromMuscle(target, anchor_idx);
            if (!ok) {
                LOG_ERROR("[RFT " << side << "] Failed to remove anchor " << anchor_idx << " from " << target);
                return false;
            }
            if (mRecordingSurgery) {
                auto op = std::make_unique<RemoveAnchorOp>(target, anchor_idx);
                recordOperation(std::move(op));
            }
        }

        // Copy donor anchors (ascending order, appended)
        for (int donor_anchor_idx : cfg.copy_donor_anchors) {
            bool ok = copyAnchorToMuscle(cfg.donor_muscle, donor_anchor_idx, target);
            if (!ok) {
                LOG_ERROR("[RFT " << side << "] Failed to copy anchor " << donor_anchor_idx
                          << " from " << cfg.donor_muscle);
                return false;
            }
            if (mRecordingSurgery) {
                auto op = std::make_unique<CopyAnchorOp>(cfg.donor_muscle, donor_anchor_idx, target);
                recordOperation(std::move(op));
            }
        }
        Muscle* rftTarget = mCharacter->getMuscleByName(target);
        LOG_INFO("[RFT " << side << "] " << target << ": removed "
                 << cfg.remove_anchors.size() << " anchors, copied "
                 << cfg.copy_donor_anchors.size() << " from " << cfg.donor_muscle
                 << " (" << (rftTarget ? (int)rftTarget->GetAnchors().size() : -1) << " anchors)");
    }

    return true;
}

void SurgeryPanel::executeSEMLS() {
    LOG_INFO("[SEMLS] Executing enabled surgeries...");

    auto& cfg = mSEMLSConfig;

    // TAL
    if (cfg.tal_left.enabled)  executeTAL(true);
    if (cfg.tal_right.enabled) executeTAL(false);

    // DHL
    if (cfg.dhl_left.enabled)  executeDHL(true);
    if (cfg.dhl_right.enabled) executeDHL(false);

    // FDO
    if (cfg.fdo_left.enabled) {
        auto& fdo = cfg.fdo_left;
        double angle_rad = fdo.angle_deg * M_PI / 180.0;
        Eigen::Vector3d search_dir(fdo.search_direction[0], fdo.search_direction[1], fdo.search_direction[2]);
        Eigen::Vector3d rot_axis(fdo.rotation_axis[0], fdo.rotation_axis[1], fdo.rotation_axis[2]);
        search_dir.normalize();
        rot_axis.normalize();
        executeFDO(fdo.ref_muscle, fdo.ref_anchor, search_dir, rot_axis, angle_rad);
        if (mRecordingSurgery) {
            auto op = std::make_unique<FDOCombinedOp>(fdo.ref_muscle, fdo.ref_anchor, search_dir, rot_axis, angle_rad);
            recordOperation(std::move(op));
        }
    }
    if (cfg.fdo_right.enabled) {
        auto& fdo = cfg.fdo_right;
        double angle_rad = fdo.angle_deg * M_PI / 180.0;
        Eigen::Vector3d search_dir(fdo.search_direction[0], fdo.search_direction[1], fdo.search_direction[2]);
        Eigen::Vector3d rot_axis(fdo.rotation_axis[0], fdo.rotation_axis[1], fdo.rotation_axis[2]);
        search_dir.normalize();
        rot_axis.normalize();
        executeFDO(fdo.ref_muscle, fdo.ref_anchor, search_dir, rot_axis, angle_rad);
        if (mRecordingSurgery) {
            auto op = std::make_unique<FDOCombinedOp>(fdo.ref_muscle, fdo.ref_anchor, search_dir, rot_axis, angle_rad);
            recordOperation(std::move(op));
        }
    }

    // RFT
    if (cfg.rft_left.enabled)  executeRFT(true);
    if (cfg.rft_right.enabled) executeRFT(false);

    LOG_INFO("[SEMLS] All surgeries complete.");
}

// ============================================================================
// OPTIMIZATION RESULT DISPLAY (shared by TAL/DHL)
// ============================================================================

void SurgeryPanel::drawOptResultDisplay(const std::string& id_suffix,
                                        const std::optional<ContractureOptResult>& result) {
    if (!result.has_value() || result->trial_results.empty()) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Success rate
    int successCount = 0;
    int totalCount = static_cast<int>(result->trial_results.size());
    for (const auto& trial : result->trial_results) {
        double error_pct = 0.0;
        if (std::abs(trial.observed_torque) > 1e-6) {
            error_pct = std::abs(trial.computed_torque_after - trial.observed_torque)
                      / std::abs(trial.observed_torque) * 100.0;
        }
        if (error_pct <= 5.0) successCount++;
    }
    float successRate = totalCount > 0 ? (successCount * 100.0f / totalCount) : 0.0f;

    ImVec4 rateColor = (successRate >= 100.0f)
        ? ImVec4(0.3f, 0.9f, 0.3f, 1.0f)
        : ImVec4(0.9f, 0.6f, 0.2f, 1.0f);
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Parameter: %s", result->param_name.c_str());
    ImGui::SameLine();
    ImGui::TextColored(rateColor, "  Torque match: %d/%d (%.0f%%)", successCount, totalCount, successRate);

    // Per-trial torque table
    ImGuiTableFlags tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                 ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    std::string trialTableId = "##trials_" + id_suffix;
    if (ImGui::BeginTable(trialTableId.c_str(), 2, tableFlags)) {
        ImGui::TableSetupColumn("Trial", ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableSetupColumn("Torque (before->after, t:target)", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (const auto& trial : result->trial_results) {
            double error_pct = 0.0;
            if (std::abs(trial.observed_torque) > 1e-6) {
                error_pct = std::abs(trial.computed_torque_after - trial.observed_torque)
                          / std::abs(trial.observed_torque) * 100.0;
            }
            ImVec4 color = (error_pct > 5.0)
                ? ImVec4(0.9f, 0.3f, 0.3f, 1.0f)
                : ImVec4(0.3f, 0.9f, 0.3f, 1.0f);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(color, "%s", trial.trial_name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(color, "(%.1f->%.1f, t:%.1f)",
                trial.computed_torque_before, trial.computed_torque_after, trial.observed_torque);
        }
        ImGui::EndTable();
    }

    // Per-group ratio table
    if (!result->group_results.empty()) {
        std::string groupTableId = "##groups_" + id_suffix;
        if (ImGui::BeginTable(groupTableId.c_str(), 3, tableFlags)) {
            ImGui::TableSetupColumn("Muscle", ImGuiTableColumnFlags_WidthFixed, 220.0f);
            ImGui::TableSetupColumn("Ratio", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("Before -> After", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            for (const auto& grp : result->group_results) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", grp.group_name.c_str());
                ImGui::TableNextColumn();
                ImVec4 ratioColor = (std::abs(grp.ratio - 1.0) > 0.05)
                    ? ImVec4(1.0f, 0.8f, 0.3f, 1.0f)
                    : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                ImGui::TextColored(ratioColor, "%.3f", grp.ratio);
                ImGui::TableNextColumn();
                // Find matching muscle result for before/after values
                for (const auto& mr : result->muscle_results) {
                    if (mr.muscle_name == grp.group_name) {
                        ImGui::Text("%.4f -> %.4f", mr.lm_contract_before, mr.lm_contract_after);
                        break;
                    }
                }
            }
            ImGui::EndTable();
        }
    }
}

// ============================================================================
// OPTIMIZER TAB
// ============================================================================

void SurgeryPanel::drawOptimizerTab() {
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Contracture Optimizer Hyperparameters");
    ImGui::Separator();
    ImGui::Spacing();

    float w = 120.0f;

    // Ratio bounds
    ImGui::Text("Ratio Bounds");
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Min Ratio##opt", &mOptConfig.minRatio, 0.05, 0.1, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Max Ratio##opt", &mOptConfig.maxRatio, 0.05, 0.1, "%.2f");

    ImGui::Spacing();

    // Grid search
    ImGui::Text("Grid Search");
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Begin##gs", &mOptConfig.gridSearchBegin, 0.05, 0.1, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("End##gs", &mOptConfig.gridSearchEnd, 0.05, 0.1, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Step##gs", &mOptConfig.gridSearchInterval, 0.01, 0.05, "%.3f");

    ImGui::Spacing();

    // Ceres solver
    ImGui::Text("Ceres Solver");
    ImGui::SetNextItemWidth(w);
    ImGui::InputInt("Max Iterations##opt", &mOptConfig.maxIterations, 10, 50);
    ImGui::SetNextItemWidth(w);
    ImGui::InputInt("Outer Iterations##opt", &mOptConfig.outerIterations, 1, 1);

    ImGui::Spacing();

    // Regularization
    ImGui::Text("Regularization");
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Ratio Reg##opt", &mOptConfig.lambdaRatioReg, 0.1, 1.0, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Torque Reg##opt", &mOptConfig.lambdaTorqueReg, 0.1, 1.0, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(w);
    ImGui::InputDouble("Line Reg##opt", &mOptConfig.lambdaLineReg, 0.1, 1.0, "%.2f");

    ImGui::Spacing();
    ImGui::Checkbox("Verbose##opt", &mOptConfig.verbose);

    ImGui::Spacing();
    ImGui::Separator();
    if (ImGui::Button("Reset to Defaults")) {
        mOptConfig.maxIterations = 100;
        mOptConfig.minRatio = 0.5;
        mOptConfig.maxRatio = 2.0;
        mOptConfig.gridSearchBegin = 0.5;
        mOptConfig.gridSearchEnd = 2.0;
        mOptConfig.gridSearchInterval = 0.05;
        mOptConfig.lambdaRatioReg = 0.0;
        mOptConfig.lambdaTorqueReg = 0.0;
        mOptConfig.lambdaLineReg = 0.0;
        mOptConfig.outerIterations = 1;
        mOptConfig.verbose = true;
    }
}

// ============================================================================
// KEPT UI SECTIONS
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

void SurgeryPanel::drawSaveMuscleConfigSection() {
    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    std::string musclePrefix = "@data/muscle/";
    if (!mPID.empty()) {
        musclePrefix = "@pid:" + mPID + "/" + mVisit + "/muscle/";
    }

    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Muscle: %s", musclePrefix.c_str());

    std::string filenameWithExt = std::string(mSaveMuscleFilename);
    size_t lastDot = filenameWithExt.find_last_of('.');
    if (lastDot != std::string::npos) {
        std::string ext = filenameWithExt.substr(lastDot);
        if (ext == ".yaml" || ext == ".xml") filenameWithExt = filenameWithExt.substr(0, lastDot);
    }
    filenameWithExt += ".yaml";

    auto resolvedDir = rm::getManager().resolveDirCreate(musclePrefix);
    std::string resolvedPath = (resolvedDir / filenameWithExt).string();

    if (ImGui::Button("Save##Muscle")) {
        if (!mSavingMuscle) {
            mSavingMuscle = true;
            try {
                exportMuscles(resolvedPath);
                if (mRecordingSurgery) {
                    auto op = std::make_unique<ExportMusclesOp>(musclePrefix + filenameWithExt);
                    recordOperation(std::move(op));
                }
                LOG_INFO("[Surgery] Muscle configuration saved to: " << resolvedPath);
            } catch (const std::exception& e) {
                LOG_ERROR("[Surgery] Error saving muscle configuration: " << e.what());
            }
            mSavingMuscle = false;
        }
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##save_muscle_filename", mSaveMuscleFilename, sizeof(mSaveMuscleFilename));

    if (std::filesystem::exists(resolvedPath)) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "  %s exists - will overwrite", filenameWithExt.c_str());
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
    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        return;
    }

    std::string skelPrefix = "@data/skeleton/";
    if (!mPID.empty()) {
        skelPrefix = "@pid:" + mPID + "/" + mVisit + "/skeleton/";
    }

    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Skeleton: %s", skelPrefix.c_str());

    std::string filenameWithExt = std::string(mSaveSkeletonFilename);
    size_t lastDot = filenameWithExt.find_last_of('.');
    if (lastDot != std::string::npos) filenameWithExt = filenameWithExt.substr(0, lastDot);
    filenameWithExt += ".yaml";

    auto resolvedDir = rm::getManager().resolveDirCreate(skelPrefix);
    std::string resolvedPath = (resolvedDir / filenameWithExt).string();

    if (ImGui::Button("Save##Skeleton")) {
        try {
            exportSkeleton(resolvedPath);

            std::string skelNameWithoutExt = std::string(mSaveSkeletonFilename);
            size_t skelDot = skelNameWithoutExt.find_last_of('.');
            if (skelDot != std::string::npos) skelNameWithoutExt = skelNameWithoutExt.substr(0, skelDot);
            strncpy(mSaveMuscleFilename, skelNameWithoutExt.c_str(), sizeof(mSaveMuscleFilename) - 1);
            mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

            if (mRecordingSurgery) {
                auto op = std::make_unique<ExportSkeletonOp>(skelPrefix + filenameWithExt);
                recordOperation(std::move(op));
            }
            LOG_INFO("[Surgery] Skeleton configuration saved to: " << resolvedPath);
        } catch (const std::exception& e) {
            LOG_ERROR("[Surgery] Error saving skeleton configuration: " << e.what());
        }
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##save_skeleton_filename", mSaveSkeletonFilename, sizeof(mSaveSkeletonFilename));

    if (std::filesystem::exists(resolvedPath)) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "  %s exists - will overwrite", filenameWithExt.c_str());
    }
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
    int successCount = 0, failCount = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        LOG_INFO("[Surgery Script] Operation " << (i + 1) << "/" << ops.size() << ": " << ops[i]->getDescription());
        if (ops[i]->execute(this)) successCount++;
        else { failCount++; LOG_ERROR("[Surgery Script] Operation " << (i + 1) << " FAILED"); }
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

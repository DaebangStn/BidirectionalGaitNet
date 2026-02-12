#ifndef SURGERY_PANEL_H
#define SURGERY_PANEL_H

#include "SurgeryExecutor.h"
#include "SurgeryOperation.h"
#include "Character.h"
#include "common/ShapeRenderer.h"
#include "optimizer/ContractureOptimizer.h"
#include "Log.h"
#include <imgui.h>
#include <yaml-cpp/yaml.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <optional>

namespace PMuscle {

/**
 * SurgeryPanel - SEMLS surgery planning UI
 *
 * Provides 5 tabs: Project, TAL, DHL, FDO, RFT
 * Inherits from SurgeryExecutor for cache-invalidating overrides.
 */
class SurgeryPanel : public SurgeryExecutor {
public:
    SurgeryPanel(Character* character, ShapeRenderer* renderer);
    ~SurgeryPanel() = default;

    // Main UI rendering
    void drawSurgeryPanel(bool* show_panel, int window_height);
    void drawSurgeryContent();

    void setCharacter(Character* character);
    void onPIDChanged(const std::string& pid, const std::string& visit);

    // Surgery operation overrides (with cache invalidation)
    bool editAnchorPosition(const std::string& muscle, int anchor_index,
                           const Eigen::Vector3d& position) override;
    bool editAnchorWeights(const std::string& muscle, int anchor_index,
                          const std::vector<double>& weights) override;
    bool addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                            const std::string& bodynode_name, double weight) override;
    bool removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index,
                                 int bodynode_index) override;
    bool removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) override;
    bool copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex,
                           const std::string& toMuscle) override;

    // FDO surgery operation overrides (with cache invalidation)
    bool rotateJointOffset(const std::string& joint_name, const Eigen::Vector3d& axis,
                          double angle, bool preserve_position = false) override;
    bool rotateAnchorPoints(const std::string& muscle_name, int ref_anchor_index,
                           const Eigen::Vector3d& search_direction,
                           const Eigen::Vector3d& rotation_axis, double angle) override;
    void exportSkeleton(const std::string& path) override;

    // FDO combined surgery
    bool executeFDO(const std::string& ref_muscle, int ref_anchor_index,
                   const Eigen::Vector3d& search_dir, const Eigen::Vector3d& rot_axis,
                   double angle) override;

    void resetSkeleton();

    // Script recording
    bool isRecording() const { return mRecordingSurgery; }
    void startRecording();
    void stopRecording();
    void exportRecording(const std::string& filepath);
    void loadSurgeryScript(const std::string& filepath);

    // --- SEMLS Surgery Config ---
    struct ROMTrialEntry {
        std::string trial_name;
        float angle_deg = 0.0f;
    };
    struct TALConfig {
        std::vector<ROMTrialEntry> rom_trials;
        std::vector<std::string> muscles;
        bool enabled = true;
    };
    struct DHLConfig {
        std::string target_muscle;
        std::string donor_muscle;
        int donor_anchor = 3;
        std::vector<int> remove_anchors;
        std::vector<std::string> rom_trials;
        std::vector<std::string> muscles;
        float popliteal_angle_deg = 0.0f;
        bool enabled = true;
        bool do_anchor_transfer = true;
        bool do_contracture_opt = true;
    };
    struct FDOConfig {
        std::string joint;
        std::string ref_muscle;
        int ref_anchor = 0;
        float rotation_axis[3] = {0, 1, 0};
        float search_direction[3] = {0, -1, 0};
        float angle_deg = 0.0f;
        bool enabled = true;
    };
    struct RFTConfig {
        std::vector<std::string> target_muscles;
        std::string donor_muscle;
        std::vector<int> remove_anchors;
        std::vector<int> copy_donor_anchors;
        bool enabled = true;
    };
    struct SEMLSConfig {
        TALConfig tal_left, tal_right;
        DHLConfig dhl_left, dhl_right;
        FDOConfig fdo_left, fdo_right;
        RFTConfig rft_left, rft_right;
    };

    // Load default surgery parameters from YAML
    void loadSurgeryConfig(const std::string& yaml_path);
    void loadPatientMetadata(const std::string& metadata_path);

    // Execute individual surgeries
    bool executeTAL(bool left);
    bool executeDHL(bool left);
    bool executeRFT(bool left);

    // Execute all enabled surgeries in SEMLS order
    void executeSEMLS();

    // Highlighted anchors for 3D rendering (populated by FDO tab)
    struct HighlightAnchor {
        Eigen::Vector3d position;
        bool is_reference;  // true = ref anchor (red), false = affected (sky blue)
    };
    const std::vector<HighlightAnchor>& getHighlightedAnchors() const { return mHighlightedAnchors; }
    bool shouldDrawFDOAnchors() const { return mDrawFDOAnchors; }

private:
    // SEMLS tab draw methods
    void drawProjectTab();
    void drawTALTab();
    void drawDHLTab();
    void drawFDOTab();
    void drawRFTTab();
    void drawOptimizerTab();

    // Kept UI sections (used in Project tab)
    void drawScriptControlsSection();
    void drawSaveMuscleConfigSection();
    void drawSaveSkeletonConfigSection();
    void drawResetMusclesSection();
    void drawResetSkeletonSection();

    // Script preview popup
    void showScriptPreview();
    void executeSurgeryScript(std::vector<std::unique_ptr<SurgeryOperation>>& ops);
    void recordOperation(std::unique_ptr<SurgeryOperation> op);

    void invalidateMuscleCache(const std::string& muscleName);
    void drawOptResultDisplay(const std::string& id_suffix, const std::optional<ContractureOptResult>& result);

    // Shared contracture optimization helper for TAL/DHL
    ContractureOptResult runContractureOpt(
        const std::string& search_group_name,
        const std::vector<std::string>& muscles,
        const std::vector<std::string>& trial_names,
        const std::vector<ROMTrialConfig>& rom_configs,
        const ContractureOptimizer::Config& opt_config);

private:
    Character* mExpCharacter;
    ShapeRenderer* mShapeRenderer;

    // Script recording
    bool mRecordingSurgery;
    std::vector<std::unique_ptr<SurgeryOperation>> mRecordedOperations;
    std::string mRecordingScriptPath;
    std::string mLoadScriptPath;
    char mRecordingPathBuffer[256];
    char mLoadPathBuffer[256];

    // Loaded script for preview/execution
    std::vector<std::unique_ptr<SurgeryOperation>> mLoadedScript;
    bool mShowScriptPreview;

    // Save muscle config
    char mSaveMuscleFilename[256];
    bool mSavingMuscle;

    // Save Skeleton section
    char mSaveSkeletonFilename[64];

    // PID state for save paths
    std::string mPID;
    std::string mVisit;

    // --- SEMLS state ---
    std::vector<HighlightAnchor> mHighlightedAnchors;
    bool mDrawFDOAnchors = false;
    SEMLSConfig mSEMLSConfig;
    ContractureOptimizer mContractureOptimizer;
    std::string mPatientMetadataPath;
    YAML::Node mPatientMetadata;
    bool mPatientLoaded = false;
    char mPatientPathBuffer[256];
    char mSurgeryConfigPathBuffer[256];
    bool mSurgeryConfigLoaded = false;

    // Optimizer hyperparameters and results
    ContractureOptimizer::Config mOptConfig;
    std::optional<ContractureOptResult> mTALResult;
    std::optional<ContractureOptResult> mDHLResult;
};

} // namespace PMuscle

#endif // SURGERY_PANEL_H

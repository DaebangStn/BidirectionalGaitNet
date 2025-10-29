#ifndef SURGERY_PANEL_H
#define SURGERY_PANEL_H

#include "SurgeryExecutor.h"
#include "SurgeryOperation.h"
#include "Character.h"
#include "ShapeRenderer.h"
#include "Log.h"
#include <imgui.h>
#include <map>
#include <vector>
#include <string>
#include <memory>

namespace PMuscle {

/**
 * SurgeryPanel - UI and execution for muscle surgery operations
 * 
 * Inherits from SurgeryExecutor to provide GUI-specific overrides
 * with cache invalidation for visual updates.
 */
class SurgeryPanel : public SurgeryExecutor {
public:
    SurgeryPanel(Character* character, ShapeRenderer* renderer);
    ~SurgeryPanel() = default;

    // Main UI rendering
    void drawSurgeryPanel(bool* show_panel, int window_height);

    // Update character reference (needed when character is reloaded)
    void setCharacter(Character* character);

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

    // Script recording
    bool isRecording() const { return mRecordingSurgery; }
    void startRecording();
    void stopRecording();
    void exportRecording(const std::string& filepath);
    void loadSurgeryScript(const std::string& filepath);

    // Getters for rendering (anchor visualization)
    std::string getCandidateMuscle() const { return mAnchorCandidateMuscle; }
    int getCandidateAnchorIndex() const { return mSelectedCandidateAnchorIndex; }
    std::string getReferenceMuscle() const { return mAnchorReferenceMuscle; }
    int getReferenceAnchorIndex() const { return mSelectedReferenceAnchorIndex; }

private:
    // UI section rendering
    void drawScriptControlsSection();
    void drawResetMusclesSection();
    void drawDistributePassiveForceSection();
    void drawRelaxPassiveForceSection();
    void drawAnchorManipulationSection();
    void drawSaveMuscleConfigSection();
    
    // Script preview popup
    void showScriptPreview();
    void executeSurgeryScript(std::vector<std::unique_ptr<SurgeryOperation>>& ops);
    void recordOperation(std::unique_ptr<SurgeryOperation> op);

    // Helper - invalidate muscle cache for rendering update
    void invalidateMuscleCache(const std::string& muscleName);

private:
    // Character and rendering
    Character* mCharacter;
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

    // Distribute passive force section
    char mDistributeFilterBuffer[128];
    std::map<std::string, bool> mDistributeSelection;
    std::string mDistributeRefMuscle;

    // Relax passive force section
    char mRelaxFilterBuffer[128];
    std::map<std::string, bool> mRelaxSelection;

    // Anchor manipulation section
    char mAnchorCandidateFilterBuffer[128];
    char mAnchorReferenceFilterBuffer[128];
    std::string mAnchorCandidateMuscle;
    std::string mAnchorReferenceMuscle;
    int mSelectedCandidateAnchorIndex;
    int mSelectedReferenceAnchorIndex;
};

} // namespace PMuscle

#endif // SURGERY_PANEL_H


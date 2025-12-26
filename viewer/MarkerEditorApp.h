#ifndef MARKER_EDITOR_APP_H
#define MARKER_EDITOR_APP_H

#include "common/ViewerAppBase.h"
#include "RenderCharacter.h"
#include <memory>
#include <string>
#include <vector>
#include <cfloat>
#include <assimp/scene.h>

// Undo state for fit operations
struct FitUndoState {
    size_t bodyNodeIndex;
    Eigen::Vector3d oldSize;
};

/**
 * @brief Standalone marker editor application
 *
 * Inherits from ViewerAppBase for common window/camera/input handling.
 *
 * Features:
 * - Load skeleton and render body nodes
 * - Load marker set from XML and render as spheres
 * - ImGui panel for editing: select, move offset, change bone, rename, duplicate, delete
 * - Export to new marker set XML
 */
class MarkerEditorApp : public ViewerAppBase
{
public:
    MarkerEditorApp(int argc, char** argv);
    ~MarkerEditorApp() override = default;

protected:
    // ViewerAppBase overrides
    void onInitialize() override;
    void drawContent() override;
    void drawUI() override;
    void keyPress(int key, int scancode, int action, int mods) override;

private:
    // Window position (from render.yaml)
    int mWindowXPos = 0;
    int mWindowYPos = 0;

    // Skeleton and markers
    std::unique_ptr<RenderCharacter> mCharacter;
    std::string mSkeletonPath;
    std::string mMarkerPath;
    std::string mExportPath;

    // Editor state
    int mSelectedMarkerIndex = -1;
    int mReferenceMarkerIndex = -1;  // For alignment feature (-1 if none)
    bool mShowLabels = true;
    bool mShowPlaneXY = false;
    bool mShowPlaneYZ = false;
    bool mShowPlaneZX = false;
    float mPlaneOpacity = 0.5f;
    char mSearchFilter[256] = {0};
    char mNewMarkerName[64] = "NewMarker";
    int mNewMarkerBoneIndex = 0;

    // Marker visibility
    std::vector<bool> mMarkerVisible;

    // Body node visibility
    std::vector<bool> mBodyNodeVisible;
    char mBodyNodeFilter[128] = {0};
    int mSelectedBodyNodeIndex = -1;

    // Skeleton export
    char mExportSkeletonPath[256] = "data/skeleton/edited.yaml";

    // Cached matrices for label drawing
    GLdouble mModelview[16];
    GLdouble mProjection[16];
    GLint mViewport[4];

    // Body node names cache
    std::vector<std::string> mBodyNodeNames;

    // Undo state for fit operations
    std::vector<FitUndoState> mFitUndoStack;

    // Rendering
    void drawSkeleton();
    void drawMarkers();
    void drawMarkerLabels();

    // ImGui UI
    void drawEditorPanel();
    void drawMarkerListSection();
    void drawSelectedMarkerSection();
    void drawAddMarkerPopup();
    void drawBodyNodesSection();

    // File operations
    void loadSkeleton(const std::string& path);
    void loadMarkers(const std::string& path);
    void exportMarkers(const std::string& path);

    // Helpers
    void updateBodyNodeNames();

    // Geometry editing
    Eigen::Vector3d computeMeshAABB(const dart::dynamics::BodyNode* bn);
    void fitBoxToMesh(size_t bodyNodeIndex, bool clearUndoStack = true);
    void fitVisibleBoxesToMesh();
    void undoFit();
    bool canUndoFit() const;
    void exportSkeleton(const std::string& path);

    // Marker alignment
    void alignMarkerAxis(int axis);
    void mirrorMarkerAxis(int axis);
};

#endif // MARKER_EDITOR_APP_H

#ifndef MARKER_EDITOR_APP_H
#define MARKER_EDITOR_APP_H

#include "dart/gui/Trackball.hpp"
#include "RenderCharacter.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
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
 * Features:
 * - Load skeleton and render body nodes
 * - Load marker set from XML and render as spheres
 * - ImGui panel for editing: select, move offset, change bone, rename, duplicate, delete
 * - Export to new marker set XML
 */
class MarkerEditorApp
{
public:
    MarkerEditorApp(int argc, char** argv);
    ~MarkerEditorApp();

    void startLoop();

private:
    // GLFW window
    GLFWwindow* mWindow;
    int mWidth, mHeight;
    int mWindowXPos, mWindowYPos;

    // Camera
    Eigen::Vector3d mEye;
    Eigen::Vector3d mUp;
    Eigen::Vector3d mTrans;
    double mZoom;
    double mPersp;
    dart::gui::Trackball mTrackball;

    // Mouse state
    bool mMouseDown;
    bool mRotate;
    bool mTranslate;
    double mMouseX, mMouseY;

    // Skeleton and markers
    std::unique_ptr<RenderCharacter> mCharacter;
    std::string mSkeletonPath;
    std::string mMarkerPath;
    std::string mExportPath;

    // Editor state
    int mSelectedMarkerIndex;
    int mReferenceMarkerIndex;  // For alignment feature (-1 if none)
    bool mShowLabels;
    bool mShowPlaneXY, mShowPlaneYZ, mShowPlaneZX;  // Axis plane visibility
    float mPlaneOpacity;  // Plane transparency (0.0 - 1.0)
    RenderMode mRenderMode;  // Primitive, Mesh, or Overlay (key O to cycle)
    char mSearchFilter[256];
    char mNewMarkerName[64];
    int mNewMarkerBoneIndex;

    // Marker visibility
    std::vector<bool> mMarkerVisible;

    // Body node visibility
    std::vector<bool> mBodyNodeVisible;
    char mBodyNodeFilter[128];  // Search filter for body nodes
    int mSelectedBodyNodeIndex;  // Selected body node for geometry editing

    // Skeleton export
    char mExportSkeletonPath[256];

    // Rendering
    ShapeRenderer mShapeRenderer;

    // Cached matrices for label drawing (stored during drawMarkers, used in drawMarkerLabels)
    GLdouble mModelview[16];
    GLdouble mProjection[16];
    GLint mViewport[4];

    // Body node names cache
    std::vector<std::string> mBodyNodeNames;

    // Undo state for fit operations
    std::vector<FitUndoState> mFitUndoStack;

    // Initialization
    void initGL();
    void initImGui();
    void initLighting();
    void setCamera();

    // Rendering
    void drawFrame();
    void drawSkeleton();
    void drawMarkers();
    void drawMarkerLabels();  // Draw labels after ImGui::NewFrame

    // ImGui UI
    void drawEditorPanel();
    void drawMarkerListSection();
    void drawSelectedMarkerSection();
    void drawAddMarkerPopup();
    void drawBodyNodesSection();

    // Input callbacks (static wrappers + member functions)
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    void resize(int width, int height);
    void mousePress(int button, int action, int mods);
    void mouseMove(double x, double y);
    void mouseScroll(double xoff, double yoff);
    void keyPress(int key, int scancode, int action, int mods);

    // File operations
    void loadSkeleton(const std::string& path);
    void loadMarkers(const std::string& path);
    void exportMarkers(const std::string& path);

    // Helpers
    void updateBodyNodeNames();
    void loadRenderConfig();

    // Geometry editing
    Eigen::Vector3d computeMeshAABB(const dart::dynamics::BodyNode* bn);
    void fitBoxToMesh(size_t bodyNodeIndex, bool clearUndoStack = true);
    void fitVisibleBoxesToMesh();
    void undoFit();
    bool canUndoFit() const;
    void exportSkeleton(const std::string& path);

    // Camera
    void alignCameraToPlane(int plane);

    // Marker alignment
    void alignMarkerAxis(int axis);
    void mirrorMarkerAxis(int axis);
};

#endif // MARKER_EDITOR_APP_H

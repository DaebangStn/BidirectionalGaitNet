#ifndef MUSCLE_PERSONALIZER_APP_H
#define MUSCLE_PERSONALIZER_APP_H

#include "dart/gui/Trackball.hpp"
#include "RenderCharacter.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include "rm/rm.hpp"
#include "SurgeryExecutor.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "common/imgui_common.h"
#include <memory>
#include <string>
#include <vector>
#include <set>

/**
 * @brief Muscle Personalizer Application
 *
 * Features:
 * - Weight application (f0 scaling based on target body mass)
 * - Waypoint optimization (muscle path optimization from HDF motions)
 * - Contracture estimation (Ceres-based lm_contract parameter fitting from ROM data)
 */
class MusclePersonalizerApp
{
public:
    MusclePersonalizerApp(const std::string& configPath = "@data/config/muscle_personalizer.yaml");
    ~MusclePersonalizerApp();

    void startLoop();

private:
    // ============================================================
    // GLFW Window
    // ============================================================
    GLFWwindow* mWindow;
    int mWidth, mHeight;
    int mWindowXPos, mWindowYPos;

    // ============================================================
    // Camera
    // ============================================================
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

    // ============================================================
    // Rendering
    // ============================================================
    ShapeRenderer mShapeRenderer;
    std::string mSkeletonPath;
    std::string mMusclePath;
    RenderMode mRenderMode = RenderMode::Wireframe;
    GroundMode mGroundMode = GroundMode::Wireframe;

    // ============================================================
    // Surgery Executor (handles all muscle operations)
    // ============================================================
    std::unique_ptr<PMuscle::SurgeryExecutor> mExecutor;

    // ============================================================
    // Configuration
    // ============================================================
    std::string mConfigPath;
    std::set<std::string> mDefaultOpenPanels;
    int mControlPanelWidth;
    int mResultsPanelWidth;

    // ============================================================
    // Tool 1: Weight Application
    // ============================================================
    float mTargetMass = 50.0f;      // kg
    float mReferenceMass = 50.0f;   // kg (from character)
    float mCurrentMass = 50.0f;     // kg (current scaled mass)
    float mAppliedRatio = 1.0f;     // Last applied scaling ratio

    // ============================================================
    // Tool 2: Waypoint Optimization
    // ============================================================
    char mHDFPath[256] = "";
    std::vector<std::string> mAvailableMuscles;
    std::vector<bool> mSelectedMuscles;
    char mMuscleFilter[64] = "";
    int mReferenceMuscleIdx = -1;

    // Optimization parameters
    int mWaypointMaxIterations = 100;
    int mWaypointNumSampling = 50;
    float mWaypointLambdaShape = 1.0f;
    float mWaypointLambdaLengthCurve = 1.0f;
    bool mWaypointFixOriginInsertion = true;

    // ============================================================
    // Tool 3: Contracture Estimation
    // ============================================================
    std::string mROMConfigDir;
    std::vector<std::string> mROMConfigPaths;        // Full paths
    std::vector<std::string> mROMConfigNames;        // Display names
    std::vector<bool> mSelectedROMs;
    char mROMFilter[64] = "";

    // Optimization parameters
    int mContractureMaxIterations = 100;
    float mContractureMinRatio = 0.7f;
    float mContractureMaxRatio = 1.2f;
    bool mContractureUseRobustLoss = true;

    // Results (muscle groups detected after running optimization)
    struct MuscleGroupResult {
        std::string group_name;
        std::vector<std::string> muscle_names;
        double ratio;
        std::vector<double> lm_contract_values;
    };
    std::vector<MuscleGroupResult> mGroupResults;

    // ============================================================
    // Rendering Flags
    // ============================================================
    bool mRenderMuscles = true;
    bool mRenderGround = true;
    bool mColorByContracture = false;
    float mMuscleLabelFontSize = 14.0f;

    // Muscle selection
    std::vector<bool> mMuscleSelectionStates;
    char mMuscleFilterText[256] = "";

    // ============================================================
    // Initialization
    // ============================================================
    void initCamera();
    void setCamera();
    void loadConfig();
    void loadCharacter();
    void initializeSurgeryExecutor();

    // ============================================================
    // Rendering
    // ============================================================
    void drawFrame();
    void drawSkeleton();
    void drawMuscles();

    // ============================================================
    // UI Panels
    // ============================================================
    void drawLeftPanel();
    void drawCharacterLoadSection();
    void drawWeightApplicationSection();
    void drawWaypointOptimizationSection();
    void drawContractureEstimationSection();
    void drawRenderTab();

    void drawRightPanel();
    void drawViewSection();
    void drawResultsSection();

    // ============================================================
    // Tool Operations (delegate to SurgeryExecutor)
    // ============================================================
    void applyWeightScaling();
    void runWaypointOptimization();
    void runContractureEstimation();

    // Helper methods
    void scanROMConfigs();
    void refreshMuscleList();
    void exportMuscleConfig();

    // Helper for collapsing header
    bool collapsingHeaderWithControls(const std::string& title);
    bool isPanelDefaultOpen(const std::string& panelName) const;

    // ============================================================
    // Input callbacks
    // ============================================================
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
    void reset();
};

#endif // MUSCLE_PERSONALIZER_APP_H

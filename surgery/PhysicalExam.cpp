#include <glad/glad.h>
#include "PhysicalExam.h"
#include "Log.h"
#include "UriResolver.h"
#include "GLfunctions.h"
#include "SurgeryScript.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <imgui.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <GL/glu.h>

using namespace PMuscle;

// ============================================================================
// EXAMINATION BED PARAMETERS
// ============================================================================
// Typical physical therapy examination table dimensions (in meters)
constexpr double BED_WIDTH = 1.2;      // Width (X-axis): 1.2m 
constexpr double BED_HEIGHT = 0.15;    // Thickness (Y-axis): 0.15m
constexpr double BED_LENGTH = 2.5;     // Length (Z-axis): 2.5m
constexpr double BED_POSITION_Y = 0.60; // Elevation above ground: 0.60m

PhysicalExam::PhysicalExam(int width, int height)
    : SurgeryExecutor("physical_exam")
    , mWindow(nullptr)
    , mWidth(width)
    , mHeight(height)
    , mWindowXPos(0)
    , mWindowYPos(0)
    , mControlPanelWidth(250)
    , mPlotPanelWidth(350)
    , mMouseDown(false)
    , mRotate(false)
    , mTranslate(false)
    , mMouseX(0)
    , mMouseY(0)
    , mZoom(1.0)
    , mForceMagnitude(0.0)
    , mForceX(0.0f)
    , mForceY(1.0f)
    , mForceZ(0.0f)
    , mOffsetX(0.0f)
    , mOffsetY(0.0f)
    , mOffsetZ(0.0f)
    , mApplyingForce(false)
    , mSelectedBodyNode(0)
    , mApplyConfinementForce(false)
    , mRunning(false)
    , mSimulationPaused(true)  // Start paused
    , mSingleStep(false)
    , mCurrentPosePreset(0)
    , mPresetKneeAngle(90.0f)
    , mSimulationHz(900)
    , mControlHz(30)
    , mCurrentCameraPreset(0)
    , mCurrentTrialIndex(-1)
    , mTrialRunning(false)
    , mCurrentForceStep(0)
    , mExamSettingLoaded(false)
    , mPassiveForceNormalizer(10.0f)  // Default normalization factor
    , mMuscleTransparency(1.0f)        // Default muscle transparency (0.0-1.0)
    , mShowJointPassiveForces(true)   // Show joint passive forces by default
    , mJointForceScale(0.01f)          // Default scale for force arrows
    , mShowJointForceLabels(false)     // Labels off by default
    , mShowPostureDebug(false)        // Posture control debug off by default
    , mShowExamTable(false)            // Show examination table by default
    , mShowAnchorPoints(true)         // Anchor point visualization off by default
    , mDrawOBJ(true)                  // Draw mesh shapes by default
    , mApplyPostureControl(true)
    , mGraphData(nullptr)
    , mEnableInterpolation(false)
    , mJointKp(500.0)               // Proportional gain
    , mJointKi(50.0)                // Integral gain
    , mInterpolationThreshold(0.01)   // 0.01 radians threshold
    , mShowSurgeryPanel(false)      // Surgery panel hidden by default
    , mSavingMuscle(false)          // Not currently saving
    , mSweepRestorePosition(false)   // Restore position after sweep by default
    , mRecordingSurgery(false)       // Not recording by default
    , mRecordingScriptPath("")       // Will be constructed from buffer
    , mLoadScriptPath("")            // Will be constructed from buffer
    , mShowScriptPreview(false)     // Script preview hidden by default
    , mUseMuscle(true)              // Default to true for backward compatibility
{
    mForceBodyNode = "FemurR";  // Default body node
    mMuscleFilterBuffer[0] = '\0';  // Initialize filter buffer as empty string
    mShowSweepLegend = true;  // Show legend by default

    // Initialize path buffers with simple default names (no path, no extension)
    strncpy(mRecordingPathBuffer, "recorded_surgery", sizeof(mRecordingPathBuffer) - 1);
    mRecordingPathBuffer[sizeof(mRecordingPathBuffer) - 1] = '\0';
    strncpy(mLoadPathBuffer, "recorded_surgery", sizeof(mLoadPathBuffer) - 1);
    mLoadPathBuffer[sizeof(mLoadPathBuffer) - 1] = '\0';

    // Muscle Selection UI
    std::memset(mMuscleFilterText, 0, sizeof(mMuscleFilterText));

    // Initialize muscle info panel
    mMuscleInfoFilterBuffer[0] = '\0';  // Initialize muscle info filter buffer
    mSelectedMuscleInfo = "";  // No muscle selected initially

    // Initialize surgery panel filename buffers with default leaf names only
    std::strncpy(mSaveMuscleFilename, "muscle_modified", sizeof(mSaveMuscleFilename));
    mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

    std::strncpy(mSaveSkeletonFilename, "skeleton_modified", sizeof(mSaveSkeletonFilename));
    mSaveSkeletonFilename[sizeof(mSaveSkeletonFilename) - 1] = '\0';

    // Initialize surgery section filter buffers
    mDistributeFilterBuffer[0] = '\0';
    mRelaxFilterBuffer[0] = '\0';
    mAnchorCandidateFilterBuffer[0] = '\0';
    mAnchorReferenceFilterBuffer[0] = '\0';
    mSelectedCandidateAnchorIndex = -1;
    mSelectedReferenceAnchorIndex = -1;

    // Initialize FDO surgery sections
    mRotateJointComboBuffer[0] = '\0';
    mSelectedRotateJoint = "";
    mRotateJointAxis[0] = 0.0f;
    mRotateJointAxis[1] = 1.0f;  // Default: Y-axis
    mRotateJointAxis[2] = 0.0f;
    mRotateJointAngleDeg = 10.0f;  // Default: 10 degrees
    mRotateJointPreservePosition = true;  // Default: original behavior

    mRotateAnchorMuscleComboBuffer[0] = '\0';
    mRotateAnchorMuscleFilterBuffer[0] = '\0';
    mSelectedRotateAnchorMuscle = "";
    mSelectedRotateAnchorIndex = -1;
    mRotateAnchorSearchDir[0] = 0.0f;
    mRotateAnchorSearchDir[1] = 1.0f;  // Default: Y direction (below reference)
    mRotateAnchorSearchDir[2] = 0.0f;
    mRotateAnchorRotAxis[0] = 0.0f;
    mRotateAnchorRotAxis[1] = 1.0f;  // Default: Y-axis
    mRotateAnchorRotAxis[2] = 0.0f;
    mRotateAnchorAngleDeg = 10.0f;  // Default: 10 degrees

    // Initialize FDO combined mode
    mFDOMode = false;
    mSelectedFDOTargetBodynode = "";
    mFDOBodynodeFilterBuffer[0] = '\0';

    // Initialize camera presets from preset definitions below
    initializeCameraPresets();
    // Initialize graph data for posture control
    mGraphData = new CBufferData<double>();

    // Initialize sweep configuration
    mSweepConfig.joint_index = 0;
    mSweepConfig.angle_min = -1.57;  // -90 degrees
    mSweepConfig.angle_max = 1.57;   // +90 degrees
    mSweepConfig.num_steps = 50;
    mSweepRunning = false;
    mSweepCurrentStep = 0;

    // Load render config from YAML (geometry section only, skip glfwapp)
    loadRenderConfig();
}

PhysicalExam::~PhysicalExam() {
    // Note: Character has no destructor, so we don't delete it
    // It will be cleaned up when the program exits

    if (mGraphData) {
        delete mGraphData;
        mGraphData = nullptr;
    }

    if (mWindow) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(mWindow);
        glfwTerminate();
    }
}

void PhysicalExam::loadRenderConfig() {
    try {
        PMuscle::URIResolver& resolver = PMuscle::URIResolver::getInstance();
        resolver.initialize();
        std::string resolved_path = resolver.resolve("render.yaml");

        LOG_INFO("[Config] Loading render config from: " << resolved_path);

        YAML::Node config = YAML::LoadFile(resolved_path);

        // Load geometry settings ONLY (skip glfwapp section)
        if (config["geometry"]) {
            if (config["geometry"]["window"]) {
                if (config["geometry"]["window"]["width"])
                    mWidth = config["geometry"]["window"]["width"].as<int>();
                if (config["geometry"]["window"]["height"])
                    mHeight = config["geometry"]["window"]["height"].as<int>();
                if (config["geometry"]["window"]["xpos"])
                    mWindowXPos = config["geometry"]["window"]["xpos"].as<int>();
                if (config["geometry"]["window"]["ypos"])
                    mWindowYPos = config["geometry"]["window"]["ypos"].as<int>();
            }

            if (config["geometry"]["control"])
                mControlPanelWidth = config["geometry"]["control"].as<int>();

            if (config["geometry"]["plot"])
                mPlotPanelWidth = config["geometry"]["plot"].as<int>();
        }

        LOG_INFO("[Config] Loaded - Window: " << mWidth << "x" << mHeight
                  << " at (" << mWindowXPos << "," << mWindowYPos << ")"
                  << ", Control: " << mControlPanelWidth
                  << ", Plot: " << mPlotPanelWidth);

    } catch (const std::exception& e) {
        LOG_ERROR("[Config] Warning: Could not load render.yaml: " << e.what());
        LOG_ERROR("[Config] Using default values.");
    }
}

void PhysicalExam::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "MuscleSurgery");
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    mWindow = glfwCreateWindow(mWidth, mHeight, "MuscleSurgery", nullptr, nullptr);
    if (!mWindow) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Set window position from config
    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOG_ERROR("Failed to initialize GLAD");
        exit(EXIT_FAILURE);
    }

    // Initialize GLUT (needed for glutSolidCube/glutSolidSphere in rendering)
    int argc = 1;
    char* argv[1] = {(char*)"physical_exam"};
    glutInit(&argc, argv);

    // Set up GLFW callbacks for mouse and keyboard
    glfwSetWindowUserPointer(mWindow, this);

    auto keyCallback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard) {
            PhysicalExam* app = static_cast<PhysicalExam*>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
    };
    glfwSetKeyCallback(mWindow, keyCallback);

    auto cursorPosCallback = [](GLFWwindow* window, double xpos, double ypos) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            PhysicalExam* app = static_cast<PhysicalExam*>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
    };
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);

    auto mouseButtonCallback = [](GLFWwindow* window, int button, int action, int mods) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            PhysicalExam* app = static_cast<PhysicalExam*>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
    };
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);

    auto scrollCallback = [](GLFWwindow* window, double xoffset, double yoffset) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            PhysicalExam* app = static_cast<PhysicalExam*>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
        }
    };
    glfwSetScrollCallback(mWindow, scrollCallback);

    auto framebufferSizeCallback = [](GLFWwindow* window, int width, int height) {
        PhysicalExam* app = static_cast<PhysicalExam*>(glfwGetWindowUserPointer(window));
        app->windowResize(width, height);
    };
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Initialize trackball
    mTrackball.setTrackball(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5), mWidth * 0.5);
    mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
    loadCameraPreset(0);

    // Initialize DART world
    mWorld = dart::simulation::World::create();
    mWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));
    mWorld->setTimeStep(1.0 / mSimulationHz);

    // Create ground and examination table
    createGround();

    // Create large examination bed (elevated platform for physical therapy)
    mExamTable = dart::dynamics::Skeleton::create("exam_bed");
    dart::dynamics::BodyNode::Properties bed_props;
    bed_props.mName = "bed_body";

    // Use parametric bed dimensions
    dart::dynamics::ShapePtr bed_shape(
        new dart::dynamics::BoxShape(Eigen::Vector3d(BED_WIDTH, BED_HEIGHT, BED_LENGTH)));

    auto bed_pair = mExamTable->createJointAndBodyNodePair<dart::dynamics::WeldJoint>(
        nullptr, dart::dynamics::WeldJoint::Properties(), bed_props);

    // Create shape node with collision enabled
    auto bed_shape_node = bed_pair.second->createShapeNodeWith<
        dart::dynamics::VisualAspect,
        dart::dynamics::CollisionAspect,
        dart::dynamics::DynamicsAspect>(bed_shape);

    // Set collision properties for the bed
    auto bed_collision_aspect = bed_shape_node->getCollisionAspect();
    if (bed_collision_aspect) {
        bed_collision_aspect->setCollidable(true);
    }

    // Position bed at examination height
    Eigen::Isometry3d bed_tf = Eigen::Isometry3d::Identity();
    bed_tf.translation() = Eigen::Vector3d(0.0, BED_POSITION_Y, 0.0);
    mExamTable->getJoint(0)->setTransformFromParentBodyNode(bed_tf);

    mWorld->addSkeleton(mExamTable);

    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    // Light settings - reduced specular
    GLfloat light_position[] = {10.0f, 10.0f, 10.0f, 0.0f};
    GLfloat light_ambient[] = {0.1f, 0.1f, 0.1f, 1.0f};
    GLfloat light_diffuse[] = {0.3f, 0.3f, 0.3f, 1.0f};
    GLfloat light_specular[] = {0.0f, 0.0f, 0.0f, 1.0f};  // No specular highlights
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

    // Material settings - matte finish (low shininess)
    GLfloat mat_specular[] = {0.0f, 0.0f, 0.0f, 1.0f};  // No specular reflection
    GLfloat mat_shininess[] = {0.0f};  // No shininess (matte)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

    LOG_INFO("Physical Examination initialized");
}

void PhysicalExam::createGround() {
    // Create ground as a large flat box
    mGround = dart::dynamics::Skeleton::create("ground");

    dart::dynamics::BodyNode::Properties body_props;
    body_props.mName = "ground_body";

    dart::dynamics::ShapePtr ground_shape(
        new dart::dynamics::BoxShape(Eigen::Vector3d(10.0, 0.1, 10.0)));

    auto pair = mGround->createJointAndBodyNodePair<dart::dynamics::WeldJoint>(
        nullptr, dart::dynamics::WeldJoint::Properties(), body_props);

    pair.second->createShapeNodeWith<dart::dynamics::VisualAspect,
                                     dart::dynamics::CollisionAspect,
                                     dart::dynamics::DynamicsAspect>(ground_shape);

    // Position ground at y = 0
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() = Eigen::Vector3d(0.0, -0.05, 0.0);
    mGround->getJoint(0)->setTransformFromParentBodyNode(tf);

    mWorld->addSkeleton(mGround);
}

void PhysicalExam::loadCharacter(const std::string& skel_path, const std::string& muscle_path, ActuatorType _actType) {
    // Store file paths for UI display
    mSkeletonPath = skel_path;
    mMusclePath = muscle_path;

    // Store subject skeleton/muscle paths in base class for consistent metadata tracking
    mSubjectSkeletonPath = skel_path;
    mSubjectMusclePath = muscle_path;

    // Resolve URIs
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();

    std::string resolved_skel = resolver.resolve(skel_path);

    LOG_INFO("Loading skeleton: " << resolved_skel);

    // Create character
    mCharacter = new Character(resolved_skel);

    // Load muscles if path is provided and mUseMuscle is true
    if (!muscle_path.empty() && mUseMuscle) {
        std::string resolved_muscle = resolver.resolve(muscle_path);
        LOG_INFO("Loading muscle: " << resolved_muscle);
        mCharacter->setMuscles(resolved_muscle);

        // Zero muscle activations
        if (mCharacter->getMuscles().size() > 0) {
            mCharacter->setActivations(mCharacter->getActivations().setZero());
        }
    } else {
        LOG_INFO("Skipping muscle loading" << (mUseMuscle ? " (no muscle path provided)" : " (muscles disabled)"));
        mUseMuscle = false;  // Ensure mUseMuscle is false if no muscles loaded
    }

    // Set actuator type
    mCharacter->setActuatorType(_actType);

    // Add to world
    mWorld->addSkeleton(mCharacter->getSkeleton());

    // Set initial pose to supine (laying on back on examination bed)
    setPoseSupine();

    // Setup posture control targets (must be called after character is loaded)
    setupPostureTargets();

    // Initialize marked joint targets and PI controller state (all unmarked initially)
    auto skel = mCharacter->getSkeleton();
    mMarkedJointTargets.resize(skel->getNumDofs(), std::nullopt);
    mJointIntegralError.resize(skel->getNumDofs(), 0.0);
    LOG_INFO("Initialized joint PI controller system (" << skel->getNumDofs() << " DOFs)");

    // Initialize muscle selection states
    auto muscles = mCharacter->getMuscles();
    mMuscleSelectionStates.clear();
    if (muscles.size() > 0) {
        mMuscleSelectionStates.resize(muscles.size(), true);  // All muscles selected by default
    }

    LOG_INFO("Character loaded successfully in supine position" << (muscle_path.empty() ? " (skeleton only)" : ""));
}

void PhysicalExam::applyPosePreset(const std::map<std::string, Eigen::VectorXd>& joint_angles) {
    if (!mCharacter) {
        LOG_ERROR("No character loaded");
        return;
    }

    auto skeleton = mCharacter->getSkeleton();

    for (const auto& [joint_name, angles] : joint_angles) {
        auto joint = skeleton->getJoint(joint_name);
        if (!joint) {
            LOG_ERROR("Joint not found: " << joint_name);
            continue;
        }

        if (angles.size() != joint->getNumDofs()) {
            LOG_ERROR("Joint " << joint_name << " expects "
                     << joint->getNumDofs() << " DOFs, got "
                     << angles.size());
            continue;
        }

        joint->setPositions(angles);
    }

    // Store as initial pose for reset
    mInitialPose = joint_angles;

    LOG_INFO("Pose preset applied");
}

void PhysicalExam::applyForce(const std::string& body_node,
                             const Eigen::Vector3d& offset,
                             const Eigen::Vector3d& direction,
                             double magnitude) {
    if (!mCharacter) {
        LOG_ERROR("No character loaded");
        return;
    }

    auto bn = mCharacter->getSkeleton()->getBodyNode(body_node);
    if (!bn) {
        LOG_ERROR("Body node not found: " << body_node);
        return;
    }

    // Apply external force
    Eigen::Vector3d force = direction.normalized() * magnitude;
    bn->addExtForce(force, offset, false, true);
}

void PhysicalExam::applyConfinementForces(double magnitude) {
    if (!mCharacter) return;

    const char* confinementBodies[] = {"Pelvis", "Torso", "ShoulderR", "ShoulderL"};
    Eigen::Vector3d downwardForce(0.0, -magnitude, 0.0);  // Downward force
    Eigen::Vector3d zeroOffset(0.0, 0.0, 0.0);

    for (const char* bodyName : confinementBodies) {
        auto bn = mCharacter->getSkeleton()->getBodyNode(bodyName);
        if (bn) {
            bn->addExtForce(downwardForce, zeroOffset, false, true);
        }
    }
}

void PhysicalExam::stepSimulation(int steps) {
    for (int i = 0; i < steps; ++i) {
        mCharacter->step();
        mWorld->step();
    }
}

void PhysicalExam::setPaused(bool paused) {
    if (mSimulationPaused == paused) return;  // No change

    mSimulationPaused = paused;
    LOG_INFO("Simulation " << (mSimulationPaused ? "PAUSED" : "RESUMED"));

    // When pausing with PI controller enabled, update marked targets and reset integral errors
    if (mSimulationPaused && mEnableInterpolation && mCharacter) {
        auto skel = mCharacter->getSkeleton();
        Eigen::VectorXd currentPos = skel->getPositions();
        for (int i = 0; i < mMarkedJointTargets.size(); ++i) {
            if (mMarkedJointTargets[i].has_value()) {
                mMarkedJointTargets[i] = currentPos[i];
                mJointIntegralError[i] = 0.0;  // Reset integral error on pause
            }
        }
        LOG_INFO("Updated marked target angles to current positions and reset integral errors");
    }
}

std::map<std::string, Eigen::VectorXd> PhysicalExam::recordJointAngles(
    const std::vector<std::string>& joint_names) {

    std::map<std::string, Eigen::VectorXd> angles;

    if (!mCharacter) return angles;

    auto skeleton = mCharacter->getSkeleton();

    for (const auto& joint_name : joint_names) {
        auto joint = skeleton->getJoint(joint_name);
        if (joint) {
            angles[joint_name] = joint->getPositions();
        }
    }

    return angles;
}

double PhysicalExam::computePassiveForce() {
    if (!mCharacter) return 0.0;
    if (!mUseMuscle) return 0.0;  // Guard: no muscles loaded

    double total_force = 0.0;
    auto muscles = mCharacter->getMuscles();

    for (auto& muscle : muscles) {
        total_force += muscle->Getf_p();  // Use passive force, not active
    }

    return total_force;
}

void PhysicalExam::loadExamSetting(const std::string& config_path) {
    // Resolve URI if needed
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();
    std::string resolved_config_path = resolver.resolve(config_path);
    
    LOG_INFO("Loading exam setting from: " << resolved_config_path);
    
    // Parse YAML configuration
    YAML::Node config = YAML::LoadFile(resolved_config_path);

    mExamName = config["name"].as<std::string>();
    mExamDescription = config["description"] ? config["description"].as<std::string>() : "";
    
    std::string skeleton_path = config["character"]["skeleton"].as<std::string>();
    std::string muscle_path = config["character"]["muscle"] ? config["character"]["muscle"].as<std::string>() : "";
    mUseMuscle = config["character"]["use_muscle"] ? config["character"]["use_muscle"].as<bool>() : true;  // Default to true for backward compatibility
    std::string _actTypeString = config["character"]["actuator"].as<std::string>();
    ActuatorType _actType = getActuatorType(_actTypeString);

    LOG_INFO("Exam setting: " << mExamName);
    if (!mExamDescription.empty()) {
        LOG_INFO("Description: " << mExamDescription);
    }

    // Load character (muscle loading controlled by mUseMuscle option)
    if (!mUseMuscle) {
        muscle_path = "";  // Clear muscle path if mUseMuscle is false
        LOG_INFO("Muscle loading disabled by configuration");
    }
    loadCharacter(skeleton_path, muscle_path, _actType);
    
    // Parse trials
    mTrials.clear();
    if (config["trials"]) {
        for (size_t i = 0; i < config["trials"].size(); ++i) {
            const YAML::Node trial_ref = config["trials"][i];
            TrialConfig trial;
            
            // Check if this is a file reference or inline trial
            YAML::Node trial_node;
            if (trial_ref["file"]) {
                // Load trial from external file
                std::string trial_file = trial_ref["file"].as<std::string>();
                
                // Resolve URI if needed
                URIResolver& resolver = URIResolver::getInstance();
                std::string resolved_path = resolver.resolve(trial_file);
                
                LOG_INFO("  Loading trial from file: " << resolved_path);
                trial_node = YAML::LoadFile(resolved_path);
            } else {
                // Inline trial definition - use the node directly
                trial_node = trial_ref;
            }
            
            // Parse trial configuration
            trial.name = trial_node["name"].as<std::string>();
            trial.description = trial_node["description"] ? 
                trial_node["description"].as<std::string>() : "";
            
            // Parse pose
            YAML::Node pose_node = trial_node["pose"];
            for (YAML::const_iterator it = pose_node.begin(); it != pose_node.end(); ++it) {
                std::string joint_name = it->first.as<std::string>();
                
                if (it->second.IsSequence()) {
                    std::vector<double> values = it->second.as<std::vector<double>>();
                    Eigen::VectorXd angles(values.size());
                    for (size_t i = 0; i < values.size(); ++i) {
                        angles[i] = values[i];
                    }
                    trial.pose[joint_name] = angles;
                } else {
                    Eigen::VectorXd angles(1);
                    angles[0] = it->second.as<double>();
                    trial.pose[joint_name] = angles;
                }
            }
            
            // Parse force configuration
            YAML::Node force_cfg = trial_node["force"];
            trial.force_body_node = force_cfg["body_node"].as<std::string>();
            std::vector<double> offset_vec = force_cfg["position_offset"].as<std::vector<double>>();
            std::vector<double> dir_vec = force_cfg["direction"].as<std::vector<double>>();
            trial.force_offset = Eigen::Vector3d(offset_vec[0], offset_vec[1], offset_vec[2]);
            trial.force_direction = Eigen::Vector3d(dir_vec[0], dir_vec[1], dir_vec[2]);
            trial.force_min = force_cfg["magnitude_min"].as<double>();
            trial.force_max = force_cfg["magnitude_max"].as<double>();
            trial.force_steps = force_cfg["magnitude_steps"].as<int>();
            trial.settle_time = force_cfg["settle_time"].as<double>();
            
            // Parse recording configuration
            trial.record_joints = trial_node["recording"]["joints"].as<std::vector<std::string>>();
            trial.output_file = trial_node["recording"]["output_file"].as<std::string>();
            
            mTrials.push_back(trial);
            LOG_INFO("  Loaded trial: " << trial.name);
        }
    }
    
    mExamSettingLoaded = true;
    mCurrentTrialIndex = -1;
    mTrialRunning = false;
    setPaused(true);
    
    if (mTrials.empty()) {
        LOG_INFO("Exam setting loaded (no trials defined - interactive mode)");
    } else {
        LOG_INFO("Exam setting loaded with " << mTrials.size() << " trial(s)");
    }
}

void PhysicalExam::startNextTrial() {
    if (!mExamSettingLoaded || mTrials.empty()) {
        LOG_ERROR("No exam setting loaded or no trials available");
        return;
    }
    
    mCurrentTrialIndex++;
    if (mCurrentTrialIndex >= static_cast<int>(mTrials.size())) {
        LOG_INFO("All trials completed!");
        mCurrentTrialIndex = mTrials.size() - 1;
        mTrialRunning = false;
        return;
    }
    
    LOG_INFO("Starting trial " << (mCurrentTrialIndex + 1) << "/" 
              << mTrials.size() << ": " 
              << mTrials[mCurrentTrialIndex].name);
    
    mTrialRunning = true;
    mCurrentForceStep = 0;
    mRecordedData.clear();
    
    // Run the trial
    runCurrentTrial();
    
    mTrialRunning = false;
    LOG_INFO("Trial completed. Results saved to: " 
              << mTrials[mCurrentTrialIndex].output_file);
}

void PhysicalExam::runCurrentTrial() {
    if (!mExamSettingLoaded || mCurrentTrialIndex < 0 || 
        mCurrentTrialIndex >= static_cast<int>(mTrials.size())) {
        LOG_ERROR("Invalid trial index");
        return;
    }
    
    const TrialConfig& trial = mTrials[mCurrentTrialIndex];
    
    // Apply initial pose
    applyPosePreset(trial.pose);
    
    // Run force sweep
    int settle_steps = trial.settle_time * mSimulationHz;
    
    for (int i = 0; i <= trial.force_steps; ++i) {
        double magnitude = trial.force_min + 
            (trial.force_max - trial.force_min) * double(i) / trial.force_steps;
        
        LOG_INFO("  Force step " << i << "/" << trial.force_steps 
                  << ": " << magnitude << " N");
        
        // Reset to initial pose
        applyPosePreset(trial.pose);
        
        // Apply force
        applyForce(trial.force_body_node, trial.force_offset, 
                  trial.force_direction, magnitude);
        
        // Let physics settle
        stepSimulation(settle_steps);
        
        // Record data
        ROMDataPoint data;
        data.force_magnitude = magnitude;
        data.joint_angles = recordJointAngles(trial.record_joints);
        data.passive_force_total = computePassiveForce();
        mRecordedData.push_back(data);
    }
    
    // Save results
    saveToCSV(trial.output_file);
}

void PhysicalExam::runExamination(const std::string& config_path) {
    // Deprecated method - loads and runs all trials automatically
    // Parse YAML configuration
    YAML::Node config = YAML::LoadFile(config_path);

    mCurrentExamName = config["name"].as<std::string>();
    LOG_INFO("Running examination: " << mCurrentExamName);

    std::string skeleton_path = config["character"]["skeleton"].as<std::string>();
    std::string muscle_path = config["character"]["muscle"].as<std::string>();
    std::string _actTypeString = config["character"]["actuator"].as<std::string>();
    ActuatorType _actType = getActuatorType(_actTypeString);

    // Load character
    loadCharacter(skeleton_path, muscle_path, _actType);

    // Parse pose preset
    std::map<std::string, Eigen::VectorXd> pose;
    YAML::Node pose_node = config["pose"];

    for (YAML::const_iterator it = pose_node.begin(); it != pose_node.end(); ++it) {
        std::string joint_name = it->first.as<std::string>();

        if (it->second.IsSequence()) {
            std::vector<double> values = it->second.as<std::vector<double>>();
            Eigen::VectorXd angles(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                angles[i] = values[i];
            }
            pose[joint_name] = angles;
        } else {
            Eigen::VectorXd angles(1);
            angles[0] = it->second.as<double>();
            pose[joint_name] = angles;
        }
    }

    applyPosePreset(pose);

    // Parse force configuration
    YAML::Node force_cfg = config["force"];
    std::string force_body = force_cfg["body_node"].as<std::string>();
    std::vector<double> offset_vec = force_cfg["position_offset"].as<std::vector<double>>();
    std::vector<double> dir_vec = force_cfg["direction"].as<std::vector<double>>();
    double min_force = force_cfg["magnitude_min"].as<double>();
    double max_force = force_cfg["magnitude_max"].as<double>();
    int steps = force_cfg["magnitude_steps"].as<int>();
    double settle_time = force_cfg["settle_time"].as<double>();

    Eigen::Vector3d offset(offset_vec[0], offset_vec[1], offset_vec[2]);
    Eigen::Vector3d direction(dir_vec[0], dir_vec[1], dir_vec[2]);

    // Parse recording configuration
    std::vector<std::string> record_joints =
        config["recording"]["joints"].as<std::vector<std::string>>();
    std::string output_file = config["recording"]["output_file"].as<std::string>();

    // Run ROM sweep
    mRecordedData.clear();
    int settle_steps = settle_time * mSimulationHz;

    for (int i = 0; i <= steps; ++i) {
        double magnitude = min_force + (max_force - min_force) * double(i) / steps;

        LOG_INFO("Step " << i << "/" << steps << ": Force = " << magnitude << " N");

        // Reset to initial pose
        applyPosePreset(pose);

        // Apply force
        applyForce(force_body, offset, direction, magnitude);

        // Let physics settle
        stepSimulation(settle_steps);

        // Record data
        ROMDataPoint data;
        data.force_magnitude = magnitude;
        data.joint_angles = recordJointAngles(record_joints);
        data.passive_force_total = computePassiveForce();
        mRecordedData.push_back(data);
    }

    // Save results
    saveToCSV(output_file);

    LOG_INFO("Examination complete. Results saved to: " << output_file);
}

void PhysicalExam::saveToCSV(const std::string& output_path) {
    std::ofstream file(output_path);

    if (!file.is_open()) {
        LOG_ERROR("Failed to open output file: " << output_path);
        return;
    }

    // Write header
    file << "force_magnitude";

    if (!mRecordedData.empty()) {
        for (const auto& [joint_name, angles] : mRecordedData[0].joint_angles) {
            for (int i = 0; i < angles.size(); ++i) {
                file << "," << joint_name;
                if (angles.size() > 1) {
                    file << "_" << i;
                }
            }
        }
    }

    file << ",passive_force_total\n";

    // Write data
    for (const auto& data : mRecordedData) {
        file << data.force_magnitude;

        for (const auto& [joint_name, angles] : data.joint_angles) {
            for (int i = 0; i < angles.size(); ++i) {
                file << "," << angles[i];
            }
        }

        file << "," << data.passive_force_total << "\n";
    }

    file.close();
}

void PhysicalExam::setCamera() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)mWidth / mHeight, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    Eigen::Vector3d eye = mEye * mZoom;
    gluLookAt(eye[0], eye[1], eye[2],
              0.0, 0.0, 0.0,
              mUp[0], mUp[1], mUp[2]);

    mTrackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
    mTrackball.setRadius(std::min(mWidth, mHeight) * 0.4);
    mTrackball.applyGLRotation();

    glTranslatef(mTrans[0], mTrans[1], mTrans[2]);
}

void PhysicalExam::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);  // Light background
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    setCamera();
    drawSimFrame();

    // ImGui UI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    drawControlPanel();  // Left panel
    drawVisualizationPanel();  // Right panel
    
    if (mShowSurgeryPanel) drawSurgeryPanel();
    showScriptPreview();  // Show script preview popup if active
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(mWindow);
    glfwPollEvents();
}

void PhysicalExam::mainLoop() {
    while (!glfwWindowShouldClose(mWindow)) {
        // Execute one sweep step if sweep is running
        if (mSweepRunning && mCharacter) {
            auto skel = mCharacter->getSkeleton();
            auto joint = skel->getJoint(mSweepConfig.joint_index);

            if (mSweepCurrentStep <= mSweepConfig.num_steps) {
                // Calculate current angle
                double angle = mSweepConfig.angle_min +
                    (mSweepConfig.angle_max - mSweepConfig.angle_min) *
                    mSweepCurrentStep / (double)mSweepConfig.num_steps;

                // Set joint position
                Eigen::VectorXd pos = joint->getPositions();
                pos[0] = angle;  // Assumes 1-DOF joint (can be extended for multi-DOF)
                joint->setPositions(pos);

                // Update muscle state (recalculate muscle lengths and forces)
                if (mUseMuscle) {  // Guard: only update muscles if loaded
                    mCharacter->getMuscleTuple();
                }

                // Collect muscle data at this angle
                collectSweepData(angle);

                mSweepCurrentStep++;
            } else {
                // Sweep complete - restore original position if enabled
                if (mSweepRestorePosition) {
                    joint->setPositions(mSweepOriginalPos);
                }
                mSweepRunning = false;
                LOG_INFO("Sweep completed. Collected " << mSweepAngles.size()
                          << " data points");
            }

            // Check for user interruption (ESC key)
            if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                if (mSweepRestorePosition) {
                    joint->setPositions(mSweepOriginalPos);
                }
                mSweepRunning = false;
                LOG_INFO("Sweep interrupted by user");
            }
        }

        // Apply force if enabled
        for (int step=0; step<5; step++) {
            // Step simulation
            if (mCharacter) mCharacter->step();

            // Check if simulation should advance
            bool shouldStep = !mSimulationPaused || mSingleStep;
            if (shouldStep) {
                // Apply PI controller for marked joints (only when simulation is running)
                if (mEnableInterpolation && mCharacter) {
                    auto skel = mCharacter->getSkeleton();
                    Eigen::VectorXd currentPos = skel->getPositions();
                    Eigen::VectorXd currentForces = skel->getForces();
                    double dt = mWorld->getTimeStep();

                    // Apply PI controller to marked joints (those with target angles)
                    for (int i = 0; i < mMarkedJointTargets.size(); ++i) {
                        if (!mMarkedJointTargets[i].has_value()) continue;  // Skip unmarked joints

                        double targetAngle = mMarkedJointTargets[i].value();
                        double error = targetAngle - currentPos[i];

                        if (std::abs(error) > mInterpolationThreshold) {
                            // Update integral error
                            mJointIntegralError[i] += error * dt;

                            // PI control: torque = Kp * error + Ki * integral_error
                            double torque = mJointKp * error + mJointKi * mJointIntegralError[i];

                            // Apply torque to joint
                            currentForces[i] += torque;
                        } else {
                            // Target reached - unmark the joint and reset integral error
                            mMarkedJointTargets[i] = std::nullopt;
                            mJointIntegralError[i] = 0.0;
                        }
                    }
                    skel->setForces(currentForces);
                }

                if (mApplyingForce && mForceMagnitude > 0.0) {
                    Eigen::Vector3d offset(mOffsetX, mOffsetY, mOffsetZ);
                    Eigen::Vector3d direction(mForceX, mForceY, mForceZ);
                    applyForce(mForceBodyNode, offset, direction, mForceMagnitude);
                }
                if (mApplyConfinementForce) applyConfinementForces(500.0);
                if (mApplyPostureControl) applyPostureControl();
                mWorld->step();
                mSingleStep = false;
            }
            mCharacter->setZeroForces();
        }
        render();
    }
}

void PhysicalExam::drawControlPanel() {
    // Left panel - matches GLFWApp layout
    ImGui::SetNextWindowSize(ImVec2(400, mHeight-10), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::Begin("Physical Examination Controls");

    drawPosePresetsSection();
    drawForceApplicationSection();
    drawPrintInfoSection();
    drawRecordingSection();
    drawRenderOptionsSection();
    drawJointControlSection();
    drawJointAngleSweepSection();
    drawTrialManagementSection();

    ImGui::End();
}

void PhysicalExam::drawVisualizationPanel() {
    // Right panel - matches GLFWApp layout
    ImGui::SetNextWindowSize(ImVec2(400, mHeight-10), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(mWidth - 410, 10), ImGuiCond_Once);
    ImGui::Begin("Visualization & Data");

    // Loaded Files Section
    if (ImGui::CollapsingHeader("Loaded Files", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (!mSkeletonPath.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Skeleton:");
            ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
            ImGui::TextWrapped("%s", mSkeletonPath.c_str());
            ImGui::PopTextWrapPos();
        } else {
            ImGui::TextDisabled("Skeleton: Not loaded");
        }

        ImGui::Spacing();

        if (!mMusclePath.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Muscle:");
            ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x);
            ImGui::TextWrapped("%s", mMusclePath.c_str());
            ImGui::PopTextWrapPos();
        } else {
            ImGui::TextDisabled("Muscle: Not loaded");
        }

        ImGui::Unindent();
    }
    ImGui::Spacing();

    drawCurrentStateSection();
    drawRecordedDataSection();
    drawROMAnalysisSection();
    drawCameraStatusSection();
    drawSweepMusclePlotsSection();
    drawMuscleInfoSection();

    // Posture control graphs
    drawGraphPanel();

    ImGui::End();
}

// ============================================================================
// SURGERY PANEL
// ============================================================================
void PhysicalExam::drawSurgeryPanel() {
    // Floating panel in center-top area
    ImGui::SetNextWindowSize(ImVec2(450, mHeight - 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(450, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Surgery Operations", &mShowSurgeryPanel);
    
    // 1. Reset Muscle Button (not a header, just a button)
    if (ImGui::Button("Reset Muscles")) {
        // Warn and stop recording if currently recording
        if (mRecordingSurgery) {
            LOG_WARN("[Surgery] Reset Muscles called while recording is active!");
            LOG_WARN("[Surgery] Stopping recording to prevent inconsistent surgery script");
            stopRecording();
        }
        resetMuscles(mMusclePath);  // Pass XML path to reload anchors from file
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset all muscle properties (anchors + parameters) to original XML state\n(WARNING: Will stop recording if active)");
    }

    ImGui::SameLine();

    // 1b. Reset Skeleton Button
    if (ImGui::Button("Reset Skeleton")) {
        // Warn and stop recording if currently recording
        if (mRecordingSurgery) {
            LOG_WARN("[Surgery] Reset Skeleton called while recording is active!");
            LOG_WARN("[Surgery] Stopping recording to prevent inconsistent surgery script");
            stopRecording();
        }
        resetSkeleton();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset skeleton by reloading from original XML files\n"
                         "(WARNING: Will stop recording if active)\n"
                         "(WARNING: This is a destructive operation - all skeletal modifications will be lost)");
    }
    ImGui::Spacing();

    // ========================================================================
    // SURGERY SCRIPT CONTROLS
    // ========================================================================
    
    // Recording Section
    if (ImGui::CollapsingHeader("Load & Record Surgery Script")) {
        ImGui::Indent();

        // Directory prefix display
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Directory: @data/surgery/");
        ImGui::Spacing();

        if (!mRecordingSurgery) {
            if (ImGui::Button("Start Rec##Surgery")) {
                startRecording();
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
            if (ImGui::Button("Stop Rec##Surgery")) {
                stopRecording();
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "● RECORDING");
        }

        if (!mRecordedOperations.empty()) {
            ImGui::Text("Recorded operations: %zu", mRecordedOperations.size());
            ImGui::Spacing();

            // Construct full filename with .yaml extension for export
            std::string recordingFilenameWithExt = std::string(mRecordingPathBuffer);
            // Remove existing extension if present
            size_t recordingLastDot = recordingFilenameWithExt.find_last_of('.');
            if (recordingLastDot != std::string::npos) {
                recordingFilenameWithExt = recordingFilenameWithExt.substr(0, recordingLastDot);
            }
            recordingFilenameWithExt += ".yaml";

            // Construct URI path and resolve it for export
            std::string recordingUriPath = std::string("@data/surgery/") + recordingFilenameWithExt;
            URIResolver& recordingResolver = URIResolver::getInstance();
            recordingResolver.initialize();
            std::string recordingResolvedPath = recordingResolver.resolve(recordingUriPath);

            // Export button
            if (ImGui::Button("Export##Surgery")) {
                exportRecording(recordingResolvedPath);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Export to %s", recordingResolvedPath.c_str());
            }
            ImGui::SameLine();

            // Filename input
            ImGui::Text("Filename:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##RecordingPath", mRecordingPathBuffer, sizeof(mRecordingPathBuffer));

            ImGui::Spacing();
        }

        // Construct full filename with .yaml extension for loading
        std::string loadFilenameWithExt = std::string(mLoadPathBuffer);
        // Remove existing extension if present
        size_t loadLastDot = loadFilenameWithExt.find_last_of('.');
        if (loadLastDot != std::string::npos) {
            loadFilenameWithExt = loadFilenameWithExt.substr(0, loadLastDot);
        }
        loadFilenameWithExt += ".yaml";

        // Construct URI path and resolve it for loading
        std::string loadUriPath = std::string("@data/surgery/") + loadFilenameWithExt;
        URIResolver& loadResolver = URIResolver::getInstance();
        loadResolver.initialize();
        std::string loadResolvedPath = loadResolver.resolve(loadUriPath);

        // Load button
        if (ImGui::Button("Load##Surgery")) {
            loadSurgeryScript();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Load from %s", loadResolvedPath.c_str());
        }
        ImGui::SameLine();

        // Filename input
        ImGui::Text("Filename:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##LoadPath", mLoadPathBuffer, sizeof(mLoadPathBuffer));

        ImGui::Unindent();
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // ========================================================================
    // SURGERY OPERATIONS
    // ========================================================================
        
    // 2. Distribute Passive Force
    if (ImGui::CollapsingHeader("Distribute Passive Force")) {
        drawDistributePassiveForceSection();
    }
    ImGui::Spacing();
    
    // 3. Relax Passive Force (CollapsingHeader)
    if (ImGui::CollapsingHeader("Relax Passive Force")) {
        drawRelaxPassiveForceSection();
    }
    ImGui::Spacing();
    
    // 5. Anchor Point Manipulation (CollapsingHeader)
    if (ImGui::CollapsingHeader("Anchor Point Manipulation")) {
        drawAnchorManipulationSection();
    }
    ImGui::Spacing();
    
    // 5. Rotate Joint Offset
    if (ImGui::CollapsingHeader("Rotate Joint Offset")) {
        drawRotateJointOffsetSection();
    }
    ImGui::Spacing();

    // 6. Rotate Anchor Points
    if (ImGui::CollapsingHeader("Rotate Anchor Points")) {
        drawRotateAnchorPointsSection();
    }
    ImGui::Spacing();

    // 7. FDO Combined Surgery
    if (ImGui::CollapsingHeader("FDO Combined Surgery")) {
        drawFDOCombinedSection();
    }

    ImGui::Spacing();
    // 9. Save Skeleton Config (CollapsingHeader)
    if (ImGui::CollapsingHeader("Save Skeleton Config", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawSaveSkeletonConfigSection();
    }
    ImGui::Spacing();

    // 8. Save Muscle Config (CollapsingHeader)
    if (ImGui::CollapsingHeader("Save Muscle Config", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawSaveMuscleConfigSection();
    }
    ImGui::Spacing();

    ImGui::End();
}

// ============================================================================
// SURGERY OPERATIONS (GUI-specific overrides)
// ============================================================================

bool PhysicalExam::editAnchorPosition(const std::string& muscle, int anchor_index, 
                                      const Eigen::Vector3d& position) {
    // Call base class implementation
    bool success = SurgeryExecutor::editAnchorPosition(muscle, anchor_index, position);
    
    // Add GUI-specific logic if successful
    if (success && mCharacter && mUseMuscle) {  // Guard: check muscles loaded
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            if (m->name == muscle) {
                mShapeRenderer.invalidateMuscleCache(m);
                break;
            }
        }
    }
    
    return success;
}

bool PhysicalExam::editAnchorWeights(const std::string& muscle, int anchor_index,
                                     const std::vector<double>& weights) {
    // Call base class implementation
    bool success = SurgeryExecutor::editAnchorWeights(muscle, anchor_index, weights);
    
    // Add GUI-specific logic if successful
    if (success && mCharacter && mUseMuscle) {  // Guard: check muscles loaded
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            if (m->name == muscle) {
                mShapeRenderer.invalidateMuscleCache(m);
                break;
            }
        }
    }
    
    return success;
}

bool PhysicalExam::addBodyNodeToAnchor(const std::string& muscle, int anchor_index,
                                       const std::string& bodynode_name, double weight) {
    // Call base class implementation
    bool success = SurgeryExecutor::addBodyNodeToAnchor(muscle, anchor_index, bodynode_name, weight);
    
    // Add GUI-specific logic if successful
    if (success && mCharacter && mUseMuscle) {  // Guard: check muscles loaded
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            if (m->name == muscle) {
                mShapeRenderer.invalidateMuscleCache(m);
                break;
            }
        }
    }
    
    return success;
}

bool PhysicalExam::removeBodyNodeFromAnchor(const std::string& muscle, int anchor_index,
                                            int bodynode_index) {
    // Call base class implementation
    bool success = SurgeryExecutor::removeBodyNodeFromAnchor(muscle, anchor_index, bodynode_index);
    
    // Add GUI-specific logic if successful
    if (success && mCharacter && mUseMuscle) {  // Guard: check muscles loaded
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            if (m->name == muscle) {
                mShapeRenderer.invalidateMuscleCache(m);
                break;
            }
        }
    }
    
    return success;
}

void PhysicalExam::drawDistributePassiveForceSection() {
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

    // Filter textbox
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##DistributeFilter", mDistributeFilterBuffer, sizeof(mDistributeFilterBuffer));
    if (ImGui::SmallButton("Clear Filter##Distribute")) {
        mDistributeFilterBuffer[0] = '\0';
    }

    // Convert filter to lowercase
    std::string filter_lower(mDistributeFilterBuffer);
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

    // Build filtered muscle list
    std::vector<std::string> filteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (filter_lower.empty() || muscle_lower.find(filter_lower) != std::string::npos) {
            filteredMuscles.push_back(muscle_name);
        }
    }

    // Select All / Deselect All / Empty Selection buttons
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

    // Muscle checkboxes (scrollable)
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

    // Build selected muscles list
    std::vector<std::string> selectedMuscles;
    for (const auto& muscle_name : muscleNames) {
        if (mDistributeSelection[muscle_name]) {
            selectedMuscles.push_back(muscle_name);
        }
    }

    ImGui::Text("Count: %zu", selectedMuscles.size());

    // Selected muscles list with radio buttons for reference selection
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

    // Apply button
    if (ImGui::Button("Apply Distribution", ImVec2(-1, 30))) {
        if (selectedMuscles.empty()) {
            LOG_INFO("[Surgery] Error: No muscles selected!");
        } else if (mDistributeRefMuscle.empty()) {
            LOG_INFO("[Surgery] Error: No reference muscle selected!");
        } else {
            // Capture current joint angles for recording
            std::map<std::string, Eigen::VectorXd> currentJointAngles;
            if (mRecordingSurgery && mCharacter) {
                auto skel = mCharacter->getSkeleton();
                for (size_t i = 1; i < skel->getNumJoints(); ++i) {  // Skip root joint
                    auto joint = skel->getJoint(i);
                    if (joint->getNumDofs() > 0) {
                        currentJointAngles[joint->getName()] = joint->getPositions();
                    }
                }
            }
            
            // Execute operation using refactored method
            if (distributePassiveForce(selectedMuscles, mDistributeRefMuscle)) {
                // Record operation if recording
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

void PhysicalExam::drawRelaxPassiveForceSection() {
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

    // Build muscle name list
    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }

    // Two-column layout
    ImGui::Columns(2, "RelaxColumns", true);

    // LEFT PANEL: All Muscles
    ImGui::Text("All Muscles:");
    ImGui::Separator();

    // Filter textbox
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##RelaxFilter", mRelaxFilterBuffer, sizeof(mRelaxFilterBuffer));
    if (ImGui::SmallButton("Clear Filter##Relax")) {
        mRelaxFilterBuffer[0] = '\0';
    }

    // Convert filter to lowercase
    std::string filter_lower(mRelaxFilterBuffer);
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

    // Build filtered muscle list
    std::vector<std::string> filteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (filter_lower.empty() || muscle_lower.find(filter_lower) != std::string::npos) {
            filteredMuscles.push_back(muscle_name);
        }
    }

    // Select All / Deselect All / Empty Selection buttons
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

    // Muscle checkboxes (scrollable)
    ImGui::BeginChild("RelaxAllMuscles", ImVec2(0, 150), true);
    for (auto& muscle_name : filteredMuscles) {
        bool isSelected = mRelaxSelection[muscle_name];
        if (ImGui::Checkbox(muscle_name.c_str(), &isSelected)) {
            mRelaxSelection[muscle_name] = isSelected;
        }
    }
    ImGui::EndChild();

    // RIGHT PANEL: Selected Muscles to Relax
    ImGui::NextColumn();
    ImGui::Text("Muscles to Relax:");
    ImGui::Separator();

    // Build selected muscles list
    std::vector<std::string> selectedMuscles;
    for (const auto& muscle_name : muscleNames) {
        if (mRelaxSelection[muscle_name]) {
            selectedMuscles.push_back(muscle_name);
        }
    }

    ImGui::Text("Count: %zu", selectedMuscles.size());

    // Selected muscles list
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

    // Apply button
    if (ImGui::Button("Apply Relaxation", ImVec2(-1, 30))) {
        if (selectedMuscles.empty()) {
            LOG_INFO("[Surgery] Error: No muscles selected!");
        } else {
            // Capture current joint angles for recording
            std::map<std::string, Eigen::VectorXd> currentJointAngles;
            if (mRecordingSurgery && mCharacter) {
                auto skel = mCharacter->getSkeleton();
                for (size_t i = 1; i < skel->getNumJoints(); ++i) {  // Skip root joint
                    auto joint = skel->getJoint(i);
                    if (joint->getNumDofs() > 0) {
                        currentJointAngles[joint->getName()] = joint->getPositions();
                    }
                }
            }
            
            // Execute operation using refactored method
            if (relaxPassiveForce(selectedMuscles)) {
                // Record operation if recording
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

void PhysicalExam::drawSaveMuscleConfigSection() {
    ImGui::Indent();

    // Format selection (static to persist across calls)
    static int saveFormat = 0;  // 0 = YAML (default), 1 = XML

    // Directory prefix display with format radio buttons
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Directory: @data/muscle/");
    ImGui::SameLine();
    ImGui::RadioButton("YAML##MuscleFormat", &saveFormat, 0);
    ImGui::SameLine();
    ImGui::RadioButton("XML##MuscleFormat", &saveFormat, 1);
    ImGui::Spacing();

    // Determine file extension based on format
    const char* extension = (saveFormat == 0) ? ".yaml" : ".xml";

    // Construct full filename with extension
    std::string filenameWithExt = std::string(mSaveMuscleFilename);
    // Remove existing extension if present
    size_t lastDot = filenameWithExt.find_last_of('.');
    if (lastDot != std::string::npos) {
        filenameWithExt = filenameWithExt.substr(0, lastDot);
    }
    filenameWithExt += extension;

    std::string uriPath = std::string("@data/muscle/") + filenameWithExt;
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();
    std::string resolvedPath = resolver.resolve(uriPath);

    // Save button (with debounce to prevent duplicate saves on double-click)
    if (ImGui::Button("Save##Muscle")) {
        LOG_INFO("[Surgery] Muscle Save button clicked! Resolved path: " << resolvedPath);

        if (!mSavingMuscle) {  // Only process if not already saving
            mSavingMuscle = true;
            if (mCharacter) {
                try {
                    exportMuscles(resolvedPath);

                    // Record operation if recording
                    // Use skeleton base name for consistent naming across skeleton/muscle
                    if (mRecordingSurgery) {
                        std::string skelBaseName = getSkeletonBaseName();
                        std::string recordUriPath = std::string("@data/muscle/") + skelBaseName + extension;
                        auto op = std::make_unique<ExportMusclesOp>(recordUriPath);
                        recordOperation(std::move(op));
                        LOG_INFO("[Surgery] Recorded muscle export with name: " << recordUriPath);
                    }
                    LOG_INFO("[Surgery] Muscle configuration saved to: " << resolvedPath);
                } catch (const std::exception& e) {
                    LOG_ERROR("[Surgery] Error saving muscle configuration: " << e.what());
                }
            } else {
                LOG_ERROR("[Surgery] Error: No character loaded!");
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

    ImGui::Unindent();
}

void PhysicalExam::drawSaveSkeletonConfigSection() {
    ImGui::Indent();

    // Format selection (static to persist across calls)
    static int saveFormat = 0;  // 0 = YAML (default), 1 = XML

    // Directory prefix display with format radio buttons
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Directory: @data/skeleton/");
    ImGui::SameLine();
    ImGui::RadioButton("YAML##SkeletonFormat", &saveFormat, 0);
    ImGui::SameLine();
    ImGui::RadioButton("XML##SkeletonFormat", &saveFormat, 1);
    ImGui::Spacing();

    // Determine file extension based on format
    const char* extension = (saveFormat == 0) ? ".yaml" : ".xml";

    // Construct full filename with extension
    std::string filenameWithExt = std::string(mSaveSkeletonFilename);
    // Remove existing extension if present
    size_t lastDot = filenameWithExt.find_last_of('.');
    if (lastDot != std::string::npos) {
        filenameWithExt = filenameWithExt.substr(0, lastDot);
    }
    filenameWithExt += extension;

    std::string uriPath = std::string("@data/skeleton/") + filenameWithExt;
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();
    std::string resolvedPath = resolver.resolve(uriPath);

    // Save button (with debounce to prevent duplicate saves on double-click)
    if (ImGui::Button("Save##Skeleton")) {
        LOG_INFO("[Surgery] Skeleton Save button clicked! Resolved path: " << resolvedPath);

        if (mCharacter) {
            try {
                exportSkeleton(resolvedPath);

                // Update muscle filename to match the skeleton filename (without extension)
                // This keeps skeleton and muscle configs in sync
                std::string skelNameWithoutExt = std::string(mSaveSkeletonFilename);
                size_t skelDot = skelNameWithoutExt.find_last_of('.');
                if (skelDot != std::string::npos) {
                    skelNameWithoutExt = skelNameWithoutExt.substr(0, skelDot);
                }
                strncpy(mSaveMuscleFilename, skelNameWithoutExt.c_str(), sizeof(mSaveMuscleFilename) - 1);
                mSaveMuscleFilename[sizeof(mSaveMuscleFilename) - 1] = '\0';

                // Record operation if recording
                // Use subject skeleton base name for consistent naming
                if (mRecordingSurgery) {
                    std::string skelBaseName = getSkeletonBaseName();
                    std::string recordUriPath = std::string("@data/skeleton/") + skelBaseName + extension;
                    auto op = std::make_unique<ExportSkeletonOp>(recordUriPath);
                    recordOperation(std::move(op));
                    LOG_INFO("[Surgery] Recorded skeleton export with name: " << recordUriPath);
                }

                LOG_INFO("[Surgery] Skeleton configuration saved to: " << resolvedPath);
                LOG_INFO("[Surgery] Muscle filename updated to match: " << mSaveMuscleFilename);
            } catch (const std::exception& e) {
                LOG_ERROR("[Surgery] Error saving skeleton configuration: " << e.what());
            }
        } else {
            LOG_ERROR("[Surgery] Error: No character loaded!");
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
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "⚠ Warning: File already exists and will be overwritten!");
    }

    ImGui::Spacing();

    ImGui::Unindent();
}

void PhysicalExam::drawAnchorManipulationSection() {
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
    
    // Build muscle name list
    std::vector<std::string> muscleNames;
    for (auto m : muscles) {
        muscleNames.push_back(m->name);
    }
    
    // Static buffers for anchor editing
    static float editAnchorPosX = 0.0f;
    static float editAnchorPosY = 0.0f;
    static float editAnchorPosZ = 0.0f;
    static bool editValuesLoaded = false;
    static int selectedBodyNodeIndex = 0;  // Which bodynode in the anchor we're editing
    static int newBodyNodeIndex = 0;  // For adding new bodynodes
    static float newBodyNodeWeight = 1.0f;  // Weight for new bodynode
    static std::vector<float> editWeights;  // Editable weights for each bodynode
    
    // ========================================================================
    // ROW 1: Candidate Muscle
    // Layout: Filter (left) | Anchor List (right)
    // ========================================================================
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Candidate Muscle:");
    ImGui::Separator();
    
    ImGui::Columns(2, "CandidateColumns", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.35f);
    ImGui::SetColumnWidth(1, ImGui::GetWindowWidth() * 0.65f);
    
    // LEFT: Filter and muscle selection
    ImGui::Text("Filter:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##CandidateFilter", mAnchorCandidateFilterBuffer, sizeof(mAnchorCandidateFilterBuffer));
    if (ImGui::SmallButton("Clear##Candidate")) {
        mAnchorCandidateFilterBuffer[0] = '\0';
    }
    
    // Convert filter to lowercase
    std::string candidate_filter_lower(mAnchorCandidateFilterBuffer);
    std::transform(candidate_filter_lower.begin(), candidate_filter_lower.end(), 
                   candidate_filter_lower.begin(), ::tolower);
    
    // Build filtered muscle list for candidate
    std::vector<std::string> candidateFilteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (candidate_filter_lower.empty() || muscle_lower.find(candidate_filter_lower) != std::string::npos) {
            candidateFilteredMuscles.push_back(muscle_name);
        }
    }
    
    // Muscle selection (radio buttons)
    ImGui::Spacing();
    ImGui::Text("Select Muscle:");
    ImGui::BeginChild("CandidateMuscleList", ImVec2(0, 150), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto& muscle_name : candidateFilteredMuscles) {
        bool isSelected = (muscle_name == mAnchorCandidateMuscle);
        if (ImGui::RadioButton(muscle_name.c_str(), isSelected)) {
            mAnchorCandidateMuscle = muscle_name;
            mSelectedCandidateAnchorIndex = -1;  // Clear selection when changing muscle
        }
    }
    ImGui::EndChild();
    
    // RIGHT: Anchor list
    ImGui::NextColumn();
    
    if (!mAnchorCandidateMuscle.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Anchors: %s", mAnchorCandidateMuscle.c_str());
        
        // Find the muscle
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
                
                // Radio button for selection
                std::string label = "Anchor #" + std::to_string(i);
                if (ImGui::RadioButton(label.c_str(), isSelected)) {
                    mSelectedCandidateAnchorIndex = i;
                    editValuesLoaded = false;  // Reset flag to reload values
                }
                
                // Display anchor info (indented)
                ImGui::Indent();
                if (!anchor->bodynodes.empty()) {
                    // Single body node case
                    if (anchor->bodynodes.size() == 1) {
                        ImGui::Text("Body: %s", anchor->bodynodes[0]->getName().c_str());
                        if (!anchor->local_positions.empty()) {
                            auto pos = anchor->local_positions[0];
                            ImGui::Text("Pos: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                        }
                    }
                    // Multi-body LBS case
                    else {
                        ImGui::Text("Bodies & Local Positions:");
                        for (size_t j = 0; j < anchor->bodynodes.size(); ++j) {
                            ImGui::Text("  [%zu] %s (w:%.3f)", j,
                                       anchor->bodynodes[j]->getName().c_str(),
                                       anchor->weights[j]);
                            if (j < anchor->local_positions.size()) {
                                auto pos = anchor->local_positions[j];
                                ImGui::Text("      Pos: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                            }
                        }
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
    // ROW 2: Reference Muscle
    // Layout: Filter (left) | Anchor List (right)
    // ========================================================================
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Reference Muscle:");
    ImGui::Separator();
    
    ImGui::Columns(2, "ReferenceColumns", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.35f);
    ImGui::SetColumnWidth(1, ImGui::GetWindowWidth() * 0.65f);

    // LEFT: Filter and muscle selection
    ImGui::Text("Filter:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##ReferenceFilter", mAnchorReferenceFilterBuffer, sizeof(mAnchorReferenceFilterBuffer));
    if (ImGui::SmallButton("Clear##Reference")) {
        mAnchorReferenceFilterBuffer[0] = '\0';
    }
    
    // Convert filter to lowercase
    std::string reference_filter_lower(mAnchorReferenceFilterBuffer);
    std::transform(reference_filter_lower.begin(), reference_filter_lower.end(), 
                   reference_filter_lower.begin(), ::tolower);
    
    // Build filtered muscle list for reference
    std::vector<std::string> referenceFilteredMuscles;
    for (const auto& muscle_name : muscleNames) {
        std::string muscle_lower = muscle_name;
        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
        if (reference_filter_lower.empty() || muscle_lower.find(reference_filter_lower) != std::string::npos) {
            referenceFilteredMuscles.push_back(muscle_name);
        }
    }
    
    // Muscle selection (radio buttons)
    ImGui::Spacing();
    ImGui::Text("Select Muscle:");
    ImGui::BeginChild("ReferenceMuscleList", ImVec2(0, 150), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto& muscle_name : referenceFilteredMuscles) {
        bool isSelected = (muscle_name == mAnchorReferenceMuscle);
        if (ImGui::RadioButton(muscle_name.c_str(), isSelected)) {
            mAnchorReferenceMuscle = muscle_name;
            mSelectedReferenceAnchorIndex = -1;  // Clear selection when changing muscle
        }
    }
    ImGui::EndChild();
    
    // RIGHT: Anchor list
    ImGui::NextColumn();
    
    if (!mAnchorReferenceMuscle.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Anchors: %s", mAnchorReferenceMuscle.c_str());
        
        // Find the muscle
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
                
                // Radio button for selection
                std::string label = "Anchor #" + std::to_string(i);
                if (ImGui::RadioButton(label.c_str(), isSelected)) {
                    mSelectedReferenceAnchorIndex = i;
                }
                
                // Display anchor info (indented)
                ImGui::Indent();
                if (!anchor->bodynodes.empty()) {
                    ImGui::Text("Body: %s", anchor->bodynodes[0]->getName().c_str());
                    
                    // Display weights if available (LBS multi-body)
                    if (anchor->bodynodes.size() > 1) {
                        ImGui::Text("Weights:");
                        for (size_t j = 0; j < anchor->bodynodes.size(); ++j) {
                            ImGui::Text("  [%zu] %s: %.3f", j, 
                                       anchor->bodynodes[j]->getName().c_str(),
                                       anchor->weights[j]);
                        }
                    }
                    
                    // Display local position
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
            // Find candidate muscle and selected anchor
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
                
                // Load current values into edit buffers on first display or when selection changes
                if (!editValuesLoaded) {
                    if (!anchor->local_positions.empty()) {
                        auto pos = anchor->local_positions[0];
                        editAnchorPosX = pos[0];
                        editAnchorPosY = pos[1];
                        editAnchorPosZ = pos[2];
                    }
                    
                    // Load weights
                    editWeights.clear();
                    for (size_t i = 0; i < anchor->weights.size(); ++i) {
                        editWeights.push_back(anchor->weights[i]);
                    }
                    
                    selectedBodyNodeIndex = 0;
                    editValuesLoaded = true;
                }
                
                // ----------------------------------------------------------------
                // SECTION 1: Body Nodes & Weights
                // ----------------------------------------------------------------
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Body Nodes & Weights:");
                ImGui::Separator();
                
                ImGui::Text("Anchor has %zu body node(s)", anchor->bodynodes.size());
                
                ImGui::BeginChild("AnchorBodyNodes", ImVec2(0, 120), true);
                for (size_t i = 0; i < anchor->bodynodes.size(); ++i) {
                    ImGui::PushID(i);
                    
                    // Body node name
                    ImGui::Text("[%zu] Body: %s", i, anchor->bodynodes[i]->getName().c_str());
                    
                    // Weight editor
                    ImGui::SameLine(200);
                    ImGui::Text("Weight:");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80);
                    
                    if (i < editWeights.size()) {
                        ImGui::InputFloat("##Weight", &editWeights[i], 0.01f, 0.1f, "%.3f");
                    }
                    
                    // Remove button (only if more than 1 bodynode)
                    if (anchor->bodynodes.size() > 1) {
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Remove##BodyNode")) {
                            std::string bodyNodeName = anchor->bodynodes[i]->getName();
                            
                            if (removeBodyNodeFromAnchor(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, static_cast<int>(i))) {
                                editWeights.erase(editWeights.begin() + i);
                                editValuesLoaded = false;
                                
                                // Record operation if recording
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
                
                // Apply Weights button
                if (ImGui::Button("Apply Weights", ImVec2(150, 25))) {
                    // Convert float weights to double
                    std::vector<double> weights_double(editWeights.begin(), editWeights.end());
                    
                    if (editAnchorWeights(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, weights_double)) {
                        // Record operation if recording
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
                
                // ----------------------------------------------------------------
                // SECTION 2: Add Body Node
                // ----------------------------------------------------------------
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Add Body Node:");
                ImGui::Separator();
                
                // Build list of all body nodes in skeleton
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
                            
                            // Record operation if recording
                            if (mRecordingSurgery) {
                                auto op = std::make_unique<AddBodyNodeToAnchorOp>(
                                    mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, 
                                    bodyNodeName, newBodyNodeWeight);
                                recordOperation(std::move(op));
                            }
                        } else {
                            LOG_INFO("[Surgery] Body node already exists in this anchor!");
                        }
                    }
                }
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                // ----------------------------------------------------------------
                // SECTION 3: Local Position
                // ----------------------------------------------------------------
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Local Position:");
                ImGui::Separator();
                
                // Display current values from anchor (read-only reference)
                if (!anchor->local_positions.empty()) {
                    auto pos = anchor->local_positions[0];
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                                      "Current: [%.3f, %.3f, %.3f]", pos[0], pos[1], pos[2]);
                }
                
                // Editable text inputs
                ImGui::Text("New Position:");
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("X##EditAnchor", &editAnchorPosX, 0.001f, 0.01f, "%.3f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("Y##EditAnchor", &editAnchorPosY, 0.001f, 0.01f, "%.3f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputFloat("Z##EditAnchor", &editAnchorPosZ, 0.001f, 0.01f, "%.3f");
                
                // Apply button
                if (ImGui::Button("Apply Position", ImVec2(-1, 25))) {
                    Eigen::Vector3d newPos(editAnchorPosX, editAnchorPosY, editAnchorPosZ);
                    if (editAnchorPosition(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex, newPos)) {
                        // Record operation if recording
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
    
    // Remove Anchor button
    if (ImGui::Button("Remove Selected Anchor", ImVec2(-1, 30))) {
        if (mAnchorCandidateMuscle.empty()) {
            LOG_INFO("[Surgery] Error: No candidate muscle selected!");
        } else if (mSelectedCandidateAnchorIndex < 0) {
            LOG_INFO("[Surgery] Error: No anchor selected for removal!");
        } else {
            // Check if we're removing the last anchor (need at least 2)
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
                    LOG_INFO("[Surgery] Error: Cannot remove anchor - muscle must have at least 2 anchors!");
                } else {
                    if (removeAnchorFromMuscle(mAnchorCandidateMuscle, mSelectedCandidateAnchorIndex)) {
                        // Record operation if recording
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
    
    // Copy Anchor button
    if (ImGui::Button("Copy Anchor to Candidate", ImVec2(-1, 30))) {
        if (mAnchorCandidateMuscle.empty()) {
            LOG_INFO("[Surgery] Error: No candidate muscle selected!");
        } else if (mAnchorReferenceMuscle.empty()) {
            LOG_INFO("[Surgery] Error: No reference muscle selected!");
        } else if (mSelectedReferenceAnchorIndex < 0) {
            LOG_INFO("[Surgery] Error: No reference anchor selected!");
        } else {
            if (copyAnchorToMuscle(mAnchorReferenceMuscle, mSelectedReferenceAnchorIndex, mAnchorCandidateMuscle)) {
                // Record operation if recording
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

bool PhysicalExam::removeAnchorFromMuscle(const std::string& muscleName, int anchorIndex) {
    // Call base class implementation
    bool success = SurgeryExecutor::removeAnchorFromMuscle(muscleName, anchorIndex);
    
    // Add GUI-specific logic if successful
    if (success && mCharacter) {
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            if (m->name == muscleName) {
                mShapeRenderer.invalidateMuscleCache(m);
                break;
            }
        }
    }
    
    return success;
}

bool PhysicalExam::copyAnchorToMuscle(const std::string& fromMuscle, int fromIndex, const std::string& toMuscle) {
    // Call base class implementation
    bool success = SurgeryExecutor::copyAnchorToMuscle(fromMuscle, fromIndex, toMuscle);
    
    // Add GUI-specific logic if successful
    if (success && mCharacter) {
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            if (m->name == toMuscle) {
                mShapeRenderer.invalidateMuscleCache(m);
                break;
            }
        }
    }
    
    return success;
}

void PhysicalExam::drawRotateJointOffsetSection() {
    ImGui::Indent();

    ImGui::TextWrapped("Rotate a joint's offset and orientation frame relative to its parent body node (FDO surgery).");
    ImGui::Spacing();

    if (!mCharacter) {
        ImGui::TextDisabled("No character loaded");
        ImGui::Unindent();
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        ImGui::TextDisabled("No skeleton available");
        ImGui::Unindent();
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
        ImGui::Unindent();
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

    // Preserve position checkbox

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

    ImGui::Unindent();
}

void PhysicalExam::drawRotateAnchorPointsSection() {
    ImGui::Indent();

    ImGui::TextWrapped("Rotate muscle anchor points on a bodynode with positional filtering (FDO surgery).");
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

            std::string preview_text = (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < anchors.size())
                ? "Anchor #" + std::to_string(mSelectedRotateAnchorIndex)
                : "(select anchor)";

            if (ImGui::BeginCombo("##RotateAnchorIndex", preview_text.c_str())) {
                for (int i = 0; i < anchors.size(); ++i) {
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
            if (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < anchors.size()) {
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

    ImGui::Unindent();
}

void PhysicalExam::drawFDOCombinedSection() {
    ImGui::Indent();

    ImGui::TextWrapped("FDO Combined Surgery: Automatically rotate child joint of target bone, then rotate anchors on that bone with positional filtering.");
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

            std::string preview_text = (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < anchors.size())
                ? "Anchor #" + std::to_string(mSelectedRotateAnchorIndex)
                : "(select anchor)";

            if (ImGui::BeginCombo("##FDOAnchorIndex", preview_text.c_str())) {
                for (int i = 0; i < anchors.size(); ++i) {
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
            if (mSelectedRotateAnchorIndex >= 0 && mSelectedRotateAnchorIndex < anchors.size()) {
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

    ImGui::Unindent();
}

// Override methods for GUI-specific logic
bool PhysicalExam::rotateJointOffset(const std::string& joint_name, const Eigen::Vector3d& axis, double angle, bool preserve_position) {
    // Call base class implementation
    bool success = SurgeryExecutor::rotateJointOffset(joint_name, axis, angle, preserve_position);

    // No GUI-specific logic needed for joint rotation
    // (skeleton updates automatically)

    return success;
}

bool PhysicalExam::rotateAnchorPoints(const std::string& muscle_name, int ref_anchor_index,
                                     const Eigen::Vector3d& search_direction,
                                     const Eigen::Vector3d& rotation_axis, double angle) {
    // Call base class implementation
    bool success = SurgeryExecutor::rotateAnchorPoints(muscle_name, ref_anchor_index,
                                                       search_direction, rotation_axis, angle);

    // Invalidate muscle cache for visual update
    if (success && mCharacter) {
        auto muscles = mCharacter->getMuscles();
        for (auto m : muscles) {
            // Invalidate cache for all muscles that might have been modified
            mShapeRenderer.invalidateMuscleCache(m);
        }
    }

    return success;
}

void PhysicalExam::drawSimFrame() {
    glEnable(GL_LIGHTING);

    drawGround();

    // Draw examination bed
    if (mShowExamTable && mExamTable) {
        auto bed_body = mExamTable->getBodyNode(0);
        Eigen::Isometry3d bed_tf = bed_body->getWorldTransform();

        glPushMatrix();
        Eigen::Matrix4d bed_mat = bed_tf.matrix();
        glMultMatrixd(bed_mat.data());

        // Green color for examination bed
        glColor3f(0.2f, 0.8f, 0.3f);

        // Draw bed as box (using parametric dimensions)
        glScalef(BED_WIDTH, BED_HEIGHT, BED_LENGTH);
        glutSolidCube(1.0);

        glPopMatrix();
    }

    if (mCharacter) {
        drawSkeleton(mCharacter->getSkeleton());
        drawMuscles();
        drawJointPassiveForces();
        drawSelectedAnchors();
        drawReferenceAnchor();
    }

    drawForceArrow();
    drawConfinementForces();
    drawPostureForces();
}

void PhysicalExam::drawGround() {
    glDisable(GL_LIGHTING);
    glColor3f(0.5f, 0.5f, 0.5f);

    // Draw grid at the same height as physics ground (y = -0.05)
    const float groundY = -0.05f;
    glBegin(GL_LINES);
    for (int i = -10; i <= 10; ++i) {
        glVertex3f(i * 0.5f, groundY, -5.0f);
        glVertex3f(i * 0.5f, groundY, 5.0f);
        glVertex3f(-5.0f, groundY, i * 0.5f);
        glVertex3f(5.0f, groundY, i * 0.5f);
    }
    glEnd();

    glEnable(GL_LIGHTING);
}

void PhysicalExam::drawSkeleton(const dart::dynamics::SkeletonPtr& skel) {
    if (!skel) return;

    Eigen::Vector4d color(0.5, 0.5, 0.5, 1.0); // Gray color
    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        auto bn = skel->getBodyNode(i);
        drawSingleBodyNode(bn, color);
    }
}

void PhysicalExam::drawSingleBodyNode(const dart::dynamics::BodyNode* bn, const Eigen::Vector4d& color) {
    if (!bn) return;

    glPushMatrix();
    glMultMatrixd(bn->getTransform().data());

    bn->eachShapeNodeWith<dart::dynamics::VisualAspect>([this, &color](const dart::dynamics::ShapeNode* sn) {
        if (!sn) return true;

        const auto& va = sn->getVisualAspect();
        if (!va || va->isHidden()) return true;

        glPushMatrix();
        Eigen::Affine3d tmp = sn->getRelativeTransform();
        glMultMatrixd(tmp.data());

        drawShape(sn->getShape().get(), color);

        glPopMatrix();
        return true;
    });

    glPopMatrix();
}

void PhysicalExam::drawShape(const dart::dynamics::Shape* shape, const Eigen::Vector4d& color) {
    if (!shape) return;
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glColor4d(color[0], color[1], color[2], color[3]);

    if (!mDrawOBJ) {
        // Draw primitive shapes (simple geometric forms)
        if (shape->is<dart::dynamics::SphereShape>()) {
            const auto* sphere = dynamic_cast<const dart::dynamics::SphereShape*>(shape);
            GUI::DrawSphere(sphere->getRadius());
        }
        else if (shape->is<dart::dynamics::BoxShape>()) {
            const auto* box = dynamic_cast<const dart::dynamics::BoxShape*>(shape);
            GUI::DrawCube(box->getSize());
        }
        else if (shape->is<dart::dynamics::CapsuleShape>()) {
            const auto* capsule = dynamic_cast<const dart::dynamics::CapsuleShape*>(shape);
            GUI::DrawCapsule(capsule->getRadius(), capsule->getHeight());
        }
        else if (shape->is<dart::dynamics::CylinderShape>()) {
            const auto* cylinder = dynamic_cast<const dart::dynamics::CylinderShape*>(shape);
            GUI::DrawCylinder(cylinder->getRadius(), cylinder->getHeight());
        }
    }
    else {
        // Draw mesh shapes (OBJ models)
        if (shape->is<dart::dynamics::MeshShape>()) {
            const auto& mesh = dynamic_cast<const dart::dynamics::MeshShape*>(shape);
            mShapeRenderer.renderMesh(mesh, false, 0.0, color);
        }
    }
}

void PhysicalExam::drawMuscles() {
    if (!mCharacter) return;
    if (!mUseMuscle) return;  // Guard: no muscles loaded

    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    // Defensive: Ensure GL_COLOR_ARRAY is disabled before muscle rendering
    // (prevents state leakage from skeleton mesh rendering with vertex colors)
    glDisableClientState(GL_COLOR_ARRAY);

    auto muscles = mCharacter->getMuscles();

    for (int i = 0; i < muscles.size(); i++) {
        auto muscle = muscles[i];

        // Skip if muscle is not selected (using same order as environment)
        if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

        // Passive force visualization (blue gradient)
        double f_p = muscle->Getf_p();
        // Normalize by user-adjustable normalizer value
        double normalized = std::min(1.0, f_p / mPassiveForceNormalizer);
        glColor4f(0.1f, 0.1f, 0.1f + 0.9f * normalized, mMuscleTransparency);

        if (mShowAnchorPoints) {
            // Render anchor points as dots with connection lines and muscle path
            glDisable(GL_LIGHTING);

            auto anchors = muscle->GetAnchors();

            // First, draw muscle path as thin black line (behind everything)
            glLineWidth(1.5f);
            glColor4f(0.0f, 0.0f, 0.0f, 0.8f);  // Thin semi-transparent black line
            glBegin(GL_LINE_STRIP);
            for (auto anchor : anchors) {
                Eigen::Vector3d anchorPos = anchor->GetPoint();
                glVertex3f(anchorPos[0], anchorPos[1], anchorPos[2]);
            }
            glEnd();

            // Now draw anchor-to-bodynode connections and anchor points on top
            for (auto anchor : anchors) {
                Eigen::Vector3d anchorPos = anchor->GetPoint();

                // Draw lines from anchor to each bodynode it's attached to
                if (!anchor->bodynodes.empty()) {
                    glLineWidth(1.5f);
                    glColor4f(0.0f, 0.8f, 0.0f, 0.6f);  // Semi-transparent green for lines
                    glBegin(GL_LINES);

                    for (auto bodynode : anchor->bodynodes) {
                        // Get bodynode center position (world transform origin)
                        Eigen::Vector3d bodynodePos = bodynode->getWorldTransform().translation();

                        // Draw line from anchor to bodynode center
                        glVertex3f(anchorPos[0], anchorPos[1], anchorPos[2]);
                        glVertex3f(bodynodePos[0], bodynodePos[1], bodynodePos[2]);
                    }

                    glEnd();
                }

                // Draw anchor point sphere
                glColor4f(0.0f, 1.0f, 0.0f, 1.0f);  // Bright green color
                glPushMatrix();
                glTranslatef(anchorPos[0], anchorPos[1], anchorPos[2]);
                glutSolidSphere(0.004, 12, 12);  // Small sphere for anchor point
                glPopMatrix();
            }

            glLineWidth(1.0f);  // Reset line width
            glEnable(GL_LIGHTING);
        } else {
            // Render muscle as line (default behavior)
            mShapeRenderer.renderMuscle(muscle, -1.0);
        }
    }

    glEnable(GL_LIGHTING);
}

void PhysicalExam::drawForceArrow() {
    if (!mCharacter || !mApplyingForce) return;

    // Get body node from UI selection
    const char* bodyNodes[] = {"Pelvis", "FemurR", "FemurL", "TibiaR", "TibiaL", "TalusR", "TalusL"};
    auto bn = mCharacter->getSkeleton()->getBodyNode(bodyNodes[mSelectedBodyNode]);
    if (!bn) return;

    glDisable(GL_LIGHTING);

    // Get force parameters directly from UI
    Eigen::Vector3d offset(mOffsetX, mOffsetY, mOffsetZ);
    Eigen::Vector3d direction(mForceX, mForceY, mForceZ);
    direction.normalize();

    // Calculate force application point in world coordinates
    Eigen::Vector3d world_pos = bn->getWorldTransform() * offset;
    Eigen::Vector3d force_vec = direction * (mForceMagnitude * 0.001); // Scale for visualization

    // Draw arrow
    glColor3f(0.6f, 0.2f, 0.8f); // purple
    glLineWidth(6.0f);

    glBegin(GL_LINES);
    glVertex3f(world_pos[0], world_pos[1], world_pos[2]);
    glVertex3f(world_pos[0] + force_vec[0],
               world_pos[1] + force_vec[1],
               world_pos[2] + force_vec[2]);
    glEnd();

    // Draw arrowhead
    Eigen::Vector3d tip = world_pos + force_vec;
    glPushMatrix();
    glTranslatef(tip[0], tip[1], tip[2]);
    glutSolidSphere(0.02, 15, 15);
    glPopMatrix();

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void PhysicalExam::drawConfinementForces() {
    if (!mCharacter || !mApplyConfinementForce) return;

    glDisable(GL_LIGHTING);
    glColor3f(0.6f, 0.2f, 0.8f);  // Purple color
    glLineWidth(6.0f);

    const char* confinementBodies[] = {"Pelvis", "Torso", "ShoulderR", "ShoulderL"};
    Eigen::Vector3d forceDirection(0.0, -1.0, 0.0);  // Downward
    double forceMagnitude = 500.0;
    double visualScale = 0.001;  // Scale for visualization

    for (const char* bodyName : confinementBodies) {
        auto bn = mCharacter->getSkeleton()->getBodyNode(bodyName);
        if (!bn) continue;

        // Get body node center of mass position in world coordinates
        Eigen::Vector3d world_pos = bn->getWorldTransform().translation();
        Eigen::Vector3d force_vec = forceDirection * (forceMagnitude * visualScale);

        // Draw arrow line
        glBegin(GL_LINES);
        glVertex3f(world_pos[0], world_pos[1], world_pos[2]);
        glVertex3f(world_pos[0] + force_vec[0],
                   world_pos[1] + force_vec[1],
                   world_pos[2] + force_vec[2]);
        glEnd();

        // Draw arrowhead
        Eigen::Vector3d tip = world_pos + force_vec;
        glPushMatrix();
        glTranslatef(tip[0], tip[1], tip[2]);
        glutSolidSphere(0.02, 15, 15);
        glPopMatrix();
    }

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void PhysicalExam::drawSelectedAnchors() {
    if (!mCharacter) return;

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) return;

    glDisable(GL_LIGHTING);

    // Draw selected candidate anchor with green dot
    if (!mAnchorCandidateMuscle.empty() && mSelectedCandidateAnchorIndex >= 0) {
        Muscle* candidateMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mAnchorCandidateMuscle) {
                candidateMuscle = m;
                break;
            }
        }

        if (candidateMuscle) {
            auto anchors = candidateMuscle->GetAnchors();
            if (mSelectedCandidateAnchorIndex < anchors.size()) {
                auto anchor = anchors[mSelectedCandidateAnchorIndex];
                Eigen::Vector3d anchorPos = anchor->GetPoint();

                // Draw green sphere at selected candidate anchor
                glColor3f(0.2f, 1.0f, 0.2f);  // Bright green
                glPushMatrix();
                glTranslatef(anchorPos[0], anchorPos[1], anchorPos[2]);
                glutSolidSphere(0.015, 15, 15);
                glPopMatrix();
            }
        }
    }

    // Draw selected reference anchor with cyan dot
    if (!mAnchorReferenceMuscle.empty() && mSelectedReferenceAnchorIndex >= 0) {
        Muscle* referenceMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mAnchorReferenceMuscle) {
                referenceMuscle = m;
                break;
            }
        }

        if (referenceMuscle) {
            auto anchors = referenceMuscle->GetAnchors();
            if (mSelectedReferenceAnchorIndex < anchors.size()) {
                auto anchor = anchors[mSelectedReferenceAnchorIndex];
                Eigen::Vector3d anchorPos = anchor->GetPoint();

                // Draw cyan sphere at selected reference anchor
                glColor3f(0.2f, 1.0f, 1.0f);  // Cyan
                glPushMatrix();
                glTranslatef(anchorPos[0], anchorPos[1], anchorPos[2]);
                glutSolidSphere(0.012, 15, 15);
                glPopMatrix();
            }
        }
    }

    glEnable(GL_LIGHTING);
}

void PhysicalExam::drawReferenceAnchor() {
    if (!mCharacter) return;

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) return;

    // Draw reference anchor for FDO rotation operations
    if (!mSelectedRotateAnchorMuscle.empty() && mSelectedRotateAnchorIndex >= 0) {
        Muscle* refMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == mSelectedRotateAnchorMuscle) {
                refMuscle = m;
                break;
            }
        }

        if (refMuscle) {
            auto anchors = refMuscle->GetAnchors();
            if (mSelectedRotateAnchorIndex < anchors.size()) {
                auto anchor = anchors[mSelectedRotateAnchorIndex];
                Eigen::Vector3d anchorPos = anchor->GetPoint();

                glDisable(GL_LIGHTING);

                // Draw red sphere at reference anchor (larger and brighter)
                glColor4f(1.0f, 0.0f, 0.0f, 1.0f);  // Bright red, fully opaque
                glPushMatrix();
                glTranslatef(anchorPos[0], anchorPos[1], anchorPos[2]);
                glutSolidSphere(0.007, 15, 15);  // Reduced size
                glPopMatrix();

                // Draw affected anchors in sky blue based on search direction
                // Use shared computeAffectedAnchors method for consistency with execution
                Eigen::Vector3d search_dir(mRotateAnchorSearchDir[0],
                                          mRotateAnchorSearchDir[1],
                                          mRotateAnchorSearchDir[2]);

                if (search_dir.norm() > 1e-6 && !anchor->bodynodes.empty()) {
                    try {
                        // Create AnchorReference for reference anchor
                        AnchorReference ref_anchor(mSelectedRotateAnchorMuscle, mSelectedRotateAnchorIndex, 0);

                        // Compute affected anchors using shared method (this class inherits from SurgeryExecutor)
                        auto affected_anchors = computeAffectedAnchors(ref_anchor, search_dir);

                        // Render affected anchors in sky blue
                        glColor4f(0.3f, 0.7f, 1.0f, 0.8f);  // Transparent sky blue

                        for (const auto& anchor_ref : affected_anchors) {
                            Muscle* muscle = mCharacter->getMuscleByName(anchor_ref.muscle_name);
                            if (!muscle) continue;

                            auto affected_anchor = muscle->GetAnchors()[anchor_ref.anchor_index];
                            Eigen::Vector3d world_pos = affected_anchor->GetPoint();

                            glPushMatrix();
                            glTranslatef(world_pos[0], world_pos[1], world_pos[2]);
                            glutSolidSphere(0.005, 15, 15);  // Reduced size
                            glPopMatrix();
                        }
                    } catch (const std::runtime_error& e) {
                        // Multi-LBS anchor - already validated during selection
                        LOG_ERROR("[PhysicalExam] " << e.what());
                    }
                }

                glEnable(GL_LIGHTING);
            }
        }
    }
}

void PhysicalExam::drawJointPassiveForces() {
    if (!mCharacter || !mShowJointPassiveForces) return;
    if (!mUseMuscle) return;  // Guard: no muscles loaded

    // Get muscle tuple to extract passive joint torques
    MuscleTuple mt = mCharacter->getMuscleTuple(false);

    auto skel = mCharacter->getSkeleton();
    int root_dof = skel->getRootJoint()->getNumDofs();

    glDisable(GL_LIGHTING);
    glLineWidth(2.5f);

    // Iterate through all joints (excluding root)
    int dof_idx = 0;
    for (size_t i = 1; i < skel->getNumJoints(); i++) {  // Start from 1 to skip root
        auto joint = skel->getJoint(i);
        int num_dofs = joint->getNumDofs();

        if (num_dofs == 0) continue;

        // Get parent body node to access joint transform
        auto parent_bn = joint->getParentBodyNode();
        if (!parent_bn) continue;

        // Get joint position in world coordinates
        // Joint is located at the transform from parent to child
        Eigen::Isometry3d parent_tf = parent_bn->getWorldTransform();
        Eigen::Isometry3d joint_local_tf = joint->getTransformFromParentBodyNode();
        Eigen::Isometry3d joint_world_tf = parent_tf * joint_local_tf;
        Eigen::Vector3d joint_world_pos = joint_world_tf.translation();

        // Draw arrow for each DOF
        for (int d = 0; d < num_dofs; d++) {
            if (dof_idx >= mt.JtP.size()) break;

            double torque = mt.JtP[dof_idx];
            double torque_magnitude = std::abs(torque);

            // Only draw if torque is significant
            if (torque_magnitude > 0.1) {
                // Get the axis for this DOF in world coordinates
                // For revolute joints, use the rotation axis
                Eigen::Vector3d axis_world;

                // Try to get axis from RevoluteJoint
                auto revolute_joint = dynamic_cast<dart::dynamics::RevoluteJoint*>(joint);
                if (revolute_joint) {
                    Eigen::Vector3d axis_local = revolute_joint->getAxis();
                    axis_world = joint_world_tf.rotation() * axis_local;
                } else {
                    // For other joint types, use standard axes (X, Y, Z)
                    Eigen::Vector3d axes[3] = {
                        Eigen::Vector3d(1, 0, 0),
                        Eigen::Vector3d(0, 1, 0),
                        Eigen::Vector3d(0, 0, 1)
                    };
                    int axis_idx = d % 3;
                    axis_world = joint_world_tf.rotation() * axes[axis_idx];
                }
                axis_world.normalize();

                // Arrow direction based on torque sign and joint axis
                Eigen::Vector3d force_vec = axis_world * (torque * mJointForceScale);

                // Color based on magnitude (red-orange gradient)
                float intensity = std::min(1.0f, static_cast<float>(torque_magnitude / 50.0));
                glColor3f(1.0f, 0.5f * (1.0f - intensity), 0.0f);  // Red to orange

                // Draw arrow line
                glBegin(GL_LINES);
                glVertex3f(joint_world_pos[0], joint_world_pos[1], joint_world_pos[2]);
                glVertex3f(joint_world_pos[0] + force_vec[0],
                           joint_world_pos[1] + force_vec[1],
                           joint_world_pos[2] + force_vec[2]);
                glEnd();

                // Draw arrowhead
                Eigen::Vector3d tip = joint_world_pos + force_vec;
                glPushMatrix();
                glTranslatef(tip[0], tip[1], tip[2]);
                glutSolidSphere(0.015, 15, 15);
                glPopMatrix();

                // Draw text label if enabled
                if (mShowJointForceLabels) {
                    // Use stroke font for scalable text rendering
                    void* font = GLUT_STROKE_ROMAN;
                    float base_font_scale = 0.0001f;  // Base scale for stroke font

                    // Format the torque value
                    char label[32];
                    snprintf(label, sizeof(label), "%.1f", torque_magnitude);

                    // Calculate text width for background sizing (stroke fonts)
                    float text_width = 0;
                    for (int i = 0; label[i] != '\0'; i++) {
                        text_width += glutStrokeWidth(font, label[i]);
                    }
                    float text_height = 119.05f;  // Height of GLUT_STROKE_ROMAN

                    // Position text slightly offset from arrow tip
                    Eigen::Vector3d label_pos = tip + Eigen::Vector3d(0.02, 0.02, 0.0);

                    // Save state
                    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
                    glDisable(GL_DEPTH_TEST);  // Always show labels on top

                    // Draw background rectangle
                    float bg_padding = 15.0f;  // Padding in stroke font units
                    glPushMatrix();
                    glTranslatef(label_pos[0], label_pos[1], label_pos[2]);

                    // Billboard effect - make text face camera
                    GLfloat modelview[16];
                    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);

                    // Set rotation part to identity (billboard)
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            if (i == j)
                                modelview[i*4+j] = 1.0f;
                            else
                                modelview[i*4+j] = 0.0f;
                        }
                    }
                    glLoadMatrixf(modelview);

                    // Apply zoom-responsive scaling to both background and text
                    float scale = base_font_scale * mZoom;
                    glScalef(scale, scale, scale);

                    // Background quad with darker semi-transparent color
                    glColor4f(0.0f, 0.0f, 0.0f, 0.85f);  // Darker semi-transparent background
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                    glBegin(GL_QUADS);
                    glVertex3f(-bg_padding, -bg_padding, 0);
                    glVertex3f(text_width + bg_padding, -bg_padding, 0);
                    glVertex3f(text_width + bg_padding, text_height + bg_padding, 0);
                    glVertex3f(-bg_padding, text_height + bg_padding, 0);
                    glEnd();

                    // Render stroke text (scales with glScalef)
                    glColor3f(1.0f, 1.0f, 1.0f);  // White text
                    glTranslatef(0, 0, 0.001f);  // Slightly in front of background
                    glLineWidth(2.0f);  // Stroke thickness

                    for (int i = 0; label[i] != '\0'; i++) {
                        glutStrokeCharacter(font, label[i]);
                    }

                    glPopMatrix();
                    glPopAttrib();  // Restore state
                }
            }

            dof_idx++;
        }
    }

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void PhysicalExam::mouseMove(double xpos, double ypos) {
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;

    mMouseX = xpos;
    mMouseY = ypos;

    if (mRotate) {
        if (deltaX != 0 || deltaY != 0) {
            mTrackball.updateBall(xpos, mHeight - ypos);
            mCurrentCameraPreset = -1;  // Mark as custom view
        }
    }

    if (mTranslate) {
        Eigen::Matrix3d rot = mTrackball.getRotationMatrix();
        // Scale translation by window size to make it reasonable (0.001 factor for smoothness)
        double scale = 0.001 / mZoom;
        mTrans += scale * rot.transpose() * Eigen::Vector3d(deltaX, -deltaY, 0.0);
        mCurrentCameraPreset = -1;  // Mark as custom view
    }
}

void PhysicalExam::mousePress(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        mMouseDown = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = true;
            mTrackball.startBall(mMouseX, mHeight - mMouseY);
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = true;
        }
    } else if (action == GLFW_RELEASE) {
        mMouseDown = false;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = false;
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = false;
        }
    }
}

void PhysicalExam::mouseScroll(double xoffset, double yoffset) {
    if (yoffset < 0) {
        mEye *= 1.05;
        mCurrentCameraPreset = -1;  // Mark as custom view
    } else if (yoffset > 0 && mEye.norm() > 0.005) {
        mEye *= 0.95;
        mCurrentCameraPreset = -1;  // Mark as custom view
    }
}

void PhysicalExam::keyboardPress(int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_SPACE) {
        setPaused(!mSimulationPaused);
    }
    else if (key == GLFW_KEY_S) {
        if (mSimulationPaused) mSingleStep = true;
    }
    else if (key == GLFW_KEY_R) {
        reset();
    }
    else if (key == GLFW_KEY_O) {
        mDrawOBJ = !mDrawOBJ;  // Toggle mesh vs primitive shape rendering
        LOG_INFO("Rendering mode: " << (mDrawOBJ ? "Mesh (OBJ)" : "Primitive shapes"));
    }
    else if (key == GLFW_KEY_1) {
        setPoseStanding();
    }
    else if (key == GLFW_KEY_2) {
        setPoseSupine();
    }
    else if (key == GLFW_KEY_3) {
        setPoseProne();
    }
    else if (key == GLFW_KEY_4) {
        setPoseSupineKneeFlexed(mPresetKneeAngle);
    }
    // Camera preset loading (hardcoded presets)
    else if (key == GLFW_KEY_8) {
        loadCameraPreset(0);  // Front view (initial)
    }
    else if (key == GLFW_KEY_9) {
        loadCameraPreset(1);  // Side view (right)
    }
    else if (key == GLFW_KEY_0) {
        loadCameraPreset(2);  // Top view
    }
    else if (key == GLFW_KEY_G) {
        mShowSurgeryPanel = !mShowSurgeryPanel;  // Toggle surgery panel
    }
}

void PhysicalExam::windowResize(int width, int height) {
    mWidth = width;
    mHeight = height;
    glViewport(0, 0, width, height);
}

void PhysicalExam::reset() {
    // Reset camera to preset 0
    if (mCameraPresets[0].isSet) {
        mEye = mCameraPresets[0].eye;
        mUp = mCameraPresets[0].up;
        mTrans = mCameraPresets[0].trans;
        mZoom = mCameraPresets[0].zoom;
        mTrackball.setQuaternion(mCameraPresets[0].quat);
        mCurrentCameraPreset = 0;
    } else {
        mEye << 0.0, 1.0, 3.0;
        mUp << 0.0, 1.0, 0.0;
        mTrans << 0.0, -0.5, 0.0;
        mZoom = 1.0;
        mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
        mCurrentCameraPreset = 0;
    }

    // Reset ground position (ensure it stays at correct height)
    if (mGround) {
        Eigen::Isometry3d ground_tf = Eigen::Isometry3d::Identity();
        ground_tf.translation() = Eigen::Vector3d(0.0, -0.05, 0.0);
        mGround->getJoint(0)->setTransformFromParentBodyNode(ground_tf);
        mGround->setPositions(Eigen::VectorXd::Zero(mGround->getNumDofs()));
        mGround->setVelocities(Eigen::VectorXd::Zero(mGround->getNumDofs()));
    }

    // Reset examination table position
    if (mExamTable) {
        Eigen::Isometry3d table_tf = Eigen::Isometry3d::Identity();
        table_tf.translation() = Eigen::Vector3d(0.0, BED_POSITION_Y, 0.0);
        mExamTable->getJoint(0)->setTransformFromParentBodyNode(table_tf);
        mExamTable->setPositions(Eigen::VectorXd::Zero(mExamTable->getNumDofs()));
        mExamTable->setVelocities(Eigen::VectorXd::Zero(mExamTable->getNumDofs()));
    }

    // Reset character pose and velocities
    if (mCharacter) {
        auto skel = mCharacter->getSkeleton();

        // Zero all velocities (linear and angular)
        skel->setVelocities(Eigen::VectorXd::Zero(skel->getNumDofs()));

        // Reset pose to supine (initial pose for physical examination)
        setPoseSupine();

        // Zero muscle activations
        if (mCharacter->getMuscles().size() > 0) {
            mCharacter->setActivations(mCharacter->getActivations().setZero());
        }
    }

    // Reset force application
    mApplyingForce = false;
    mForceMagnitude = 0.0;

    // Reset recorded data
    mRecordedData.clear();

    // Reset simulation time
    if (mWorld) {
        mWorld->setTime(0.0);
    }
    LOG_INFO("Scene reset");
}

void PhysicalExam::resetSkeleton() {
    if (!mCharacter) {
        LOG_ERROR("[PhysicalExam] Cannot reset skeleton: no character loaded");
        return;
    }

    LOG_INFO("[PhysicalExam] Resetting skeleton by reloading from XML...");

    // Store the actuator type before deletion
    ActuatorType actType = mCharacter->getActuatorType();

    // Get the skeleton before deletion
    auto skel = mCharacter->getSkeleton();

    // FIRST: Ensure GPU has finished all pending rendering operations
    // This prevents GPU from accessing VBOs after we delete them
    glFinish();
    LOG_INFO("[PhysicalExam] Synchronized with GPU (all operations completed)");

    // SECOND: Delete VBOs from GPU (now safe - GPU not using them)
    mShapeRenderer.clearCache();
    LOG_INFO("[PhysicalExam] Cleared ShapeRenderer cache (deleted VBOs)");

    // THIRD: Remove skeleton from world (stops it from being rendered)
    if (mWorld) {
        mWorld->removeSkeleton(skel);
        LOG_INFO("[PhysicalExam] Removed skeleton from world");
    }

    // FOURTH: Delete the character object
    delete mCharacter;
    mCharacter = nullptr;
    LOG_INFO("[PhysicalExam] Deleted character object");

    // Reload character from stored XML paths
    if (mSkeletonPath.empty() || mMusclePath.empty()) {
        LOG_ERROR("[PhysicalExam] Error: skeleton or muscle path not stored");
        return;
    }

    loadCharacter(mSkeletonPath, mMusclePath, actType);

    LOG_INFO("[PhysicalExam] Skeleton reset complete");
}

void PhysicalExam::setPoseStanding() {
    if (!mCharacter) return;

    mCurrentPosePreset = 0;
    auto skel = mCharacter->getSkeleton();

    // Reset to default standing pose
    skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Set pelvis height
    auto root = skel->getRootJoint();
    if (root->getNumDofs() >= 6) {
        Eigen::VectorXd root_pos = root->getPositions();
        root_pos[4] = 0.98;  // Y position (height)
        root->setPositions(root_pos);
    }

    LOG_INFO("Pose: Standing");
}

void PhysicalExam::setPoseSupine() {
    if (!mCharacter) return;

    mCurrentPosePreset = 1;
    auto skel = mCharacter->getSkeleton();

    // Reset all joints
    skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Rotate to lay on back (supine = face up)
    // Root joint: indices 0,1,2 are rotation (roll, pitch, yaw), indices 3,4,5 are translation (x,y,z)
    auto root = skel->getRootJoint();
    if (root->getNumDofs() >= 6) {
        Eigen::VectorXd root_pos = root->getPositions();
        root_pos[0] = -M_PI / 2.0;  // Rotate around X axis (roll) - index 0 is roll rotation
        root_pos[4] = 0.1;  // Table height - index 4 is Y translation
        root->setPositions(root_pos);
    }

    LOG_INFO("Pose: Supine (laying on back)");
}

void PhysicalExam::setPoseProne() {
    if (!mCharacter) return;

    mCurrentPosePreset = 2;
    auto skel = mCharacter->getSkeleton();

    // Reset all joints
    skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Rotate to lay on front (prone = face down)
    // Root joint: indices 0,1,2 are rotation (roll, pitch, yaw), indices 3,4,5 are translation (x,y,z)
    auto root = skel->getRootJoint();
    if (root->getNumDofs() >= 6) {
        Eigen::VectorXd root_pos = root->getPositions();
        root_pos[0] = M_PI / 2.0;  // Rotate around X axis (negative roll) - index 0 is roll rotation
        root_pos[4] = 0.1;  // Table height - index 4 is Y translation
        root->setPositions(root_pos);
    }

    LOG_INFO("Pose: Prone (laying on front)");
}

void PhysicalExam::setPoseSupineKneeFlexed(double knee_angle) {
    if (!mCharacter) return;

    mCurrentPosePreset = 3;
    auto skel = mCharacter->getSkeleton();

    // Start with supine position
    setPoseSupine();

    // Flex both knees
    auto hip_flex_r = skel->getJoint("FemurR");
    auto hip_flex_l = skel->getJoint("FemurL");
    auto knee_r = skel->getJoint("TibiaR");
    auto knee_l = skel->getJoint("TibiaL");

    double angle_rad = knee_angle * M_PI / 180.0;  // Convert to radians
    
    if (hip_flex_r && hip_flex_l) {
        Eigen::VectorXd hip_r_pos = hip_flex_r->getPositions();
        Eigen::VectorXd hip_l_pos = hip_flex_l->getPositions();

        if (hip_r_pos.size() > 0) hip_r_pos[0] = -angle_rad;
        if (hip_l_pos.size() > 0) hip_l_pos[0] = -angle_rad;
        
        hip_flex_r->setPositions(hip_r_pos);
        hip_flex_l->setPositions(hip_l_pos);
    }

    if (knee_r && knee_l) {
        Eigen::VectorXd knee_r_pos = knee_r->getPositions();
        Eigen::VectorXd knee_l_pos = knee_l->getPositions();

        if (knee_r_pos.size() > 0) knee_r_pos[0] = angle_rad;
        if (knee_l_pos.size() > 0) knee_l_pos[0] = angle_rad;

        knee_r->setPositions(knee_r_pos);
        knee_l->setPositions(knee_l_pos);
    }

    LOG_INFO("Pose: Supine with knee flexion (" << knee_angle << " degrees)");
}

void PhysicalExam::printCameraInfo() {
    Eigen::Quaterniond quat = mTrackball.getCurrQuat();
    
    LOG_INFO("\n======================================");
    LOG_INFO("Copy and paste below to CAMERA_PRESET_DEFINITIONS:");
    LOG_INFO("======================================");
    LOG_INFO("PRESET|[Add description]|" 
              << mEye[0] << "," << mEye[1] << "," << mEye[2] << "|"
              << mUp[0] << "," << mUp[1] << "," << mUp[2] << "|"
              << mTrans[0] << "," << mTrans[1] << "," << mTrans[2] << "|"
              << mZoom << "|"
              << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() 
             );
    LOG_INFO("======================================\n");
}

void PhysicalExam::printBodyNodePositions() {
    if (!mCharacter) {
        LOG_INFO("No character loaded");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        LOG_INFO("No skeleton available");
        return;
    }

    LOG_INFO("\n======================================");
    LOG_INFO("Body Node Positions (World Coordinates)");
    LOG_INFO("======================================");

    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        auto bn = skel->getBodyNode(i);
        Eigen::Vector3d pos = bn->getWorldTransform().translation();

        LOG_INFO(std::setw(20) << std::left << bn->getName() << " | "
                  << "X: " << std::setw(8) << std::fixed << std::setprecision(4) << pos.x() << " "
                  << "Y: " << std::setw(8) << pos.y() << " "
                  << "Z: " << std::setw(8) << pos.z()
                 );
    }
    LOG_INFO("======================================\n");
}

void PhysicalExam::parseAndPrintPostureConfig(const std::string& pastedData) {
    // Usage example in main.cpp:
    // std::string pastedData = R"(
    // Pelvis               | X: 0.0000   Y: 1.0809   Z: -0.0116
    // FemurR               | X: -0.0959  Y: 1.0698   Z: 0.2452
    // ...
    // )";
    // exam.parseAndPrintPostureConfig(pastedData);

    LOG_INFO("\n======================================");
    LOG_INFO("Parsed C++ Array Format (Copy to setupPostureTargets)");
    LOG_INFO("======================================");

    std::istringstream stream(pastedData);
    std::string line;

    while (std::getline(stream, line)) {
        // Skip empty lines
        if (line.empty() || line.find('|') == std::string::npos) continue;

        // Parse format: "BodyName | X: value Y: value Z: value"
        size_t pipePos = line.find('|');
        if (pipePos == std::string::npos) continue;

        std::string bodyName = line.substr(0, pipePos);
        std::string coords = line.substr(pipePos + 1);

        // Trim bodyName
        bodyName.erase(0, bodyName.find_first_not_of(" \t"));
        bodyName.erase(bodyName.find_last_not_of(" \t") + 1);

        // Parse X, Y, Z values
        double x = 0.0, y = 0.0, z = 0.0;
        size_t xPos = coords.find("X:");
        size_t yPos = coords.find("Y:");
        size_t zPos = coords.find("Z:");

        if (xPos != std::string::npos && yPos != std::string::npos && zPos != std::string::npos) {
            x = std::stod(coords.substr(xPos + 2));
            y = std::stod(coords.substr(yPos + 2));
            z = std::stod(coords.substr(zPos + 2));

            // Output in C++ array format
            LOG_INFO("{\"" << std::setw(12) << std::left << (bodyName + "\",")
                      << std::setw(9) << std::right << std::fixed << std::setprecision(4) << x << ", "
                      << std::setw(8) << y << ", "
                      << std::setw(8) << z << ", "
                      << "true,  true,  true,  500.0, 50.0},"
                     );
        }
    }
    LOG_INFO("======================================\n");
}

void PhysicalExam::setupPostureTargets() {
    LOG_INFO("\n=== setupPostureTargets() called ===");
    
    if (!mCharacter) {
        LOG_INFO("ERROR: No character loaded for posture control");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) {
        LOG_INFO("ERROR: No skeleton available for posture control");
        return;
    }

    LOG_INFO("Character and skeleton OK, setting up targets...");

    // Clear existing targets
    mPostureTargets.clear();
    mGraphData->clear_all();

    // ========================================================================
    // POSTURE CONTROL CONFIGURATION
    // Define which body nodes to control and which dimensions (X, Y, Z)
    // Format: {bodyNodeName, controlX, controlY, controlZ, Kp, Ki}
    // ========================================================================
    struct PostureControlConfig {
        std::string bodyNodeName;
        bool controlX;
        bool controlY;
        bool controlZ;
        double kp;
        double ki;
    };

    std::vector<PostureControlConfig> controlConfig = {
        {"TibiaR",     true,  false,  false,  500.0, 50.0},  // Control all XYZ for TibiaR
        {"TalusR",     true,  false,  false,  500.0, 50.0},  // Control all XYZ for TalusR
        // Add more body nodes here as needed
        // Example: {"Pelvis", true, true, false, 500.0, 50.0},  // Only control X and Y
    };

    // PASTE YOUR CAPTURED POSITIONS HERE (from "Capture Positions" button output)
    // Just copy-paste the stdout format directly between R"( and )"
    std::string pastedData = R"(
Pelvis               | X: 0.0000   Y: 1.0809   Z: -0.0116
FemurR               | X: -0.0959  Y: 1.0698   Z: 0.2452
TibiaR               | X: -0.0928  Y: 1.0584   Z: 0.6675
TalusR               | X: -0.0926  Y: 1.0683   Z: 0.9290
FootPinkyR           | X: -0.1244  Y: 1.1735   Z: 0.9384
FootThumbR           | X: -0.0765  Y: 1.1863   Z: 0.9385
FemurL               | X: 0.0959   Y: 1.0698   Z: 0.2452
TibiaL               | X: 0.0928   Y: 1.0584   Z: 0.6675
TalusL               | X: 0.0926   Y: 1.0683   Z: 0.9290
FootPinkyL           | X: 0.1244   Y: 1.1735   Z: 0.9384
FootThumbL           | X: 0.0765   Y: 1.1863   Z: 0.9385
Spine                | X: 0.0000   Y: 1.0809   Z: -0.1511
Torso                | X: 0.0000   Y: 1.0809   Z: -0.3539
Neck                 | X: 0.0000   Y: 1.0809   Z: -0.5604
Head                 | X: 0.0000   Y: 1.0839   Z: -0.6834
ShoulderR            | X: -0.0981  Y: 1.0572   Z: -0.4951
ArmR                 | X: -0.3578  Y: 1.0790   Z: -0.4829
ForeArmR             | X: -0.6674  Y: 1.0866   Z: -0.5006
HandR                | X: -0.8813  Y: 1.1240   Z: -0.4947
ShoulderL            | X: 0.0981   Y: 1.0572   Z: -0.4951
ArmL                 | X: 0.3578   Y: 1.0790   Z: -0.4829
ForeArmL             | X: 0.6674   Y: 1.0866   Z: -0.5006
HandL                | X: 0.8813   Y: 1.1240   Z: -0.4947
)";

    // Create a map for quick lookup of control configuration
    std::map<std::string, PostureControlConfig> configMap;
    for (const auto& config : controlConfig) {
        configMap[config.bodyNodeName] = config;
    }

    // Parse the pasted data
    std::istringstream stream(pastedData);
    std::string line;

    while (std::getline(stream, line)) {
        // Skip empty lines
        if (line.empty() || line.find('|') == std::string::npos) continue;

        // Parse format: "BodyName | X: value Y: value Z: value"
        size_t pipePos = line.find('|');
        if (pipePos == std::string::npos) continue;

        std::string bodyName = line.substr(0, pipePos);
        std::string coords = line.substr(pipePos + 1);

        // Trim bodyName
        bodyName.erase(0, bodyName.find_first_not_of(" \t"));
        bodyName.erase(bodyName.find_last_not_of(" \t") + 1);

        // Check if this body node is in the control configuration
        if (configMap.find(bodyName) == configMap.end()) {
            // Skip this body node - not in configuration
            continue;
        }

        // Parse X, Y, Z values
        double x = 0.0, y = 0.0, z = 0.0;
        size_t xPos = coords.find("X:");
        size_t yPos = coords.find("Y:");
        size_t zPos = coords.find("Z:");

        if (xPos != std::string::npos && yPos != std::string::npos && zPos != std::string::npos) {
            x = std::stod(coords.substr(xPos + 2));
            y = std::stod(coords.substr(yPos + 2));
            z = std::stod(coords.substr(zPos + 2));

            // Check if body node exists in skeleton
            auto bn = skel->getBodyNode(bodyName);
            if (!bn) {
                LOG_INFO("Warning: Body node '" << bodyName << "' not found in skeleton");
                continue;
            }

            // Get control configuration for this body node
            const PostureControlConfig& config = configMap[bodyName];

            // Create target
            PostureTarget target;
            target.bodyNodeName = bodyName;
            target.referencePosition = Eigen::Vector3d(x, y, z);
            target.controlDimensions << (config.controlX ? 1 : 0), (config.controlY ? 1 : 0), (config.controlZ ? 1 : 0);
            target.kp = config.kp;
            target.ki = config.ki;
            target.integralError = Eigen::Vector3d::Zero();

            mPostureTargets.push_back(target);

            // Register graph keys for each controlled dimension
            std::string base_key = bodyName;
            if (config.controlX) mGraphData->register_key(base_key + "_X", 500);
            if (config.controlY) mGraphData->register_key(base_key + "_Y", 500);
            if (config.controlZ) mGraphData->register_key(base_key + "_Z", 500);

            LOG_INFO("Added posture target: " << bodyName
                      << " at (" << x << ", " << y << ", " << z << ")"
                      << " [X:" << (config.controlX ? "Y" : "N")
                      << " Y:" << (config.controlY ? "Y" : "N")
                      << " Z:" << (config.controlZ ? "Y" : "N") << "]"
                      << " Kp=" << config.kp << " Ki=" << config.ki);
        }
    }

    // Initialize mPostureForces vector
    size_t num_controlled_dims = 0;
    for (const auto& target : mPostureTargets) {
        num_controlled_dims += target.controlDimensions.count();
    }
    mPostureForces = Eigen::VectorXd::Zero(num_controlled_dims);

    LOG_INFO("Posture control setup complete with " << mPostureTargets.size() << " targets");
}

void PhysicalExam::applyPostureControl() {
    if (!mCharacter || !mApplyPostureControl || mPostureTargets.empty()) {
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    double dt = mWorld->getTimeStep();
    size_t force_idx = 0;

    // Debug output (controlled by UI checkbox)
    // To reduce spam, only print every 100 frames even when enabled
    static int debug_counter = 0;
    bool print_debug = mShowPostureDebug && (debug_counter++ % 100 == 0);
    
    if (print_debug) {
        LOG_INFO("\n=== Posture Control Debug (frame " << debug_counter << ") ===");
        LOG_INFO("Targets: " << mPostureTargets.size() << ", Apply: " << mApplyPostureControl);
    }

    for (auto& target : mPostureTargets) {
        auto bn = skel->getBodyNode(target.bodyNodeName);
        if (!bn) continue;

        // Get current position
        Eigen::Vector3d currentPos = bn->getWorldTransform().translation();

        // Compute error
        Eigen::Vector3d error = target.referencePosition - currentPos;

        // Update integral error (accumulate over time)
        target.integralError += error * dt;

        // PI control: F = Kp * e + Ki * ∫e·dt
        Eigen::Vector3d controlForce = target.kp * error + target.ki * target.integralError;
        
        if (print_debug) {
            LOG_INFO("  " << target.bodyNodeName << ":");
            LOG_INFO("    Current:   [" << currentPos.transpose() << "]");
            LOG_INFO("    Reference: [" << target.referencePosition.transpose() << "]");
            LOG_INFO("    Error:     [" << error.transpose() << "]");
            LOG_INFO("    Force:     [" << controlForce.transpose() << "]");
        }

        // Apply control only to specified dimensions
        Eigen::Vector3d appliedForce = Eigen::Vector3d::Zero();
        for (int dim = 0; dim < 3; ++dim) {
            if (target.controlDimensions[dim]) {
                appliedForce[dim] = controlForce[dim];
            }
        }

        // Apply external force to body node
        Eigen::Vector3d zeroOffset = Eigen::Vector3d::Zero();
        bn->addExtForce(appliedForce, zeroOffset, false, true);

        // Store force magnitudes for each controlled dimension
        for (int dim = 0; dim < 3; ++dim) {
            if (target.controlDimensions[dim]) {
                mPostureForces[force_idx] = appliedForce[dim];

                // Push to graph data
                std::string key = target.bodyNodeName + std::string("_") +
                                 (dim == 0 ? "X" : (dim == 1 ? "Y" : "Z"));
                if (mGraphData->key_exists(key)) {
                    mGraphData->push(key, std::abs(appliedForce[dim]));
                }

                force_idx++;
            }
        }
    }
}

void PhysicalExam::drawPostureForces() {
    if (!mCharacter || !mApplyPostureControl || mPostureTargets.empty()) {
        return;
    }

    auto skel = mCharacter->getSkeleton();
    if (!skel) return;

    glDisable(GL_LIGHTING);
    glLineWidth(6.0f);

    double visualScale = 0.001;  // Scale for visualization
    size_t force_idx = 0;

    for (const auto& target : mPostureTargets) {
        auto bn = skel->getBodyNode(target.bodyNodeName);
        if (!bn) continue;

        Eigen::Vector3d world_pos = bn->getWorldTransform().translation();

        for (int dim = 0; dim < 3; ++dim) {
            if (!target.controlDimensions[dim]) continue;

            // Get force magnitude
            double forceMag = (force_idx < mPostureForces.size()) ? mPostureForces[force_idx] : 0.0;
            force_idx++;

            // Create force vector in correct dimension
            Eigen::Vector3d force_direction = Eigen::Vector3d::Zero();
            force_direction[dim] = (forceMag > 0) ? 1.0 : -1.0;
            Eigen::Vector3d force_vec = force_direction * (std::abs(forceMag) * visualScale);

            // Use purple color for posture force arrows
            glColor3f(0.6f, 0.2f, 0.8f);

            // Draw arrow line
            glBegin(GL_LINES);
            glVertex3f(world_pos[0], world_pos[1], world_pos[2]);
            glVertex3f(world_pos[0] + force_vec[0],
                       world_pos[1] + force_vec[1],
                       world_pos[2] + force_vec[2]);
            glEnd();

            // Draw arrowhead
            Eigen::Vector3d tip = world_pos + force_vec;
            glPushMatrix();
            glTranslatef(tip[0], tip[1], tip[2]);
            glutSolidSphere(0.02, 15, 15);
            glPopMatrix();
        }
    }

    glLineWidth(1.0f);
    glEnable(GL_LIGHTING);
}

void PhysicalExam::drawGraphPanel() {
    if (!mApplyPostureControl || mPostureTargets.empty() || !mGraphData) {
        return;
    }

    // Create a collapsing header for posture control graphs
    if (ImGui::CollapsingHeader("Posture Control Forces")) {
        for (const auto& target : mPostureTargets) {
            // Create sub-header for each target
            std::string header_name = target.bodyNodeName + " Forces";
            if (ImGui::TreeNode(header_name.c_str())) {
                // Plot each controlled dimension
                for (int dim = 0; dim < 3; ++dim) {
                    if (!target.controlDimensions[dim]) continue;

                    std::string axis_name = (dim == 0) ? "X" : (dim == 1 ? "Y" : "Z");
                    std::string key = target.bodyNodeName + "_" + axis_name;

                    if (!mGraphData->key_exists(key)) continue;

                    std::string plot_title = target.bodyNodeName + " " + axis_name + " Force (N)";

                    // Get data from buffer
                    std::vector<double> data = mGraphData->get(key);
                    if (data.empty()) continue;

                    // Create time array
                    std::vector<double> time_data(data.size());
                    double time_step = mWorld->getTimeStep();
                    for (size_t i = 0; i < data.size(); ++i) {
                        time_data[i] = (i - data.size()) * time_step;
                    }

                    // Set axis limits
                    ImPlot::SetNextAxisLimits(ImAxis_X1, time_data.front(), time_data.back(), ImGuiCond_Always);
                    ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 1000);  // 0 to 1000N range

                    // Create plot
                    if (ImPlot::BeginPlot(plot_title.c_str(), ImVec2(-1, 150))) {
                        ImPlot::SetupAxes("Time (s)", "Force (N)");

                        // Plot line
                        ImPlot::PlotLine(axis_name.c_str(), time_data.data(), data.data(), data.size());

                        ImPlot::EndPlot();
                    }
                }

                ImGui::TreePop();
            }
        }
    }
}

void PhysicalExam::saveCameraPreset(int index) {
    // Kept for API compatibility - just prints camera info
    printCameraInfo();
}

void PhysicalExam::loadCameraPreset(int index) {
    if (index < 0 || index >= 3) {
        LOG_ERROR("Invalid camera preset index: " << index);
        return;
    }
    
    if (!mCameraPresets[index].isSet) {
        LOG_INFO("Camera preset " << (index + 1) << " is not set yet");
        return;
    }
    
    mEye = mCameraPresets[index].eye;
    mUp = mCameraPresets[index].up;
    mTrans = mCameraPresets[index].trans;
    mZoom = mCameraPresets[index].zoom;
    mTrackball.setQuaternion(mCameraPresets[index].quat);
    mCurrentCameraPreset = index;
    
    LOG_INFO("Camera preset " << (index + 1) << " loaded: " 
              << mCameraPresets[index].description);
}

// ============================================================================
// CAMERA PRESET DEFINITIONS - Paste new presets below (one per line)
// Format: PRESET|description|eyeX,eyeY,eyeZ|upX,upY,upZ|transX,transY,transZ|zoom|quatW,quatX,quatY,quatZ
// ============================================================================
const char* CAMERA_PRESET_DEFINITIONS[] = {
    "PRESET|Initial view|0,0.992519,2.97756|0,1,0|0.0119052,-0.723115,0.108916|1|0.823427,0.0367708,0.561259,0.0748684",
    "PRESET|Front Thigh view|0,0.431401,1.2942|0,1,0|-0.0612152,-0.655993,-0.0328963|1|0.81416,0.580639,0.0,0.0",
    // "PRESET|Back Thigh view|0,0.256203,0.768607|0,1,0|0.123724,-1.05037,-0.222596|1|0.0,0.0,0.593647,-0.803454",
    "PRESET|Foot side view|0,0.564545,1.69364|0,1,0|-0.382306,-0.960299,-0.632363|1|0.767598,0.0,0.630236,0.11053",
};

void PhysicalExam::initializeCameraPresets() {
    const int numPresets = sizeof(CAMERA_PRESET_DEFINITIONS) / sizeof(CAMERA_PRESET_DEFINITIONS[0]);
    
    for (int i = 0; i < std::min(numPresets, 3); ++i) {
        std::string line = CAMERA_PRESET_DEFINITIONS[i];
        
        // Parse the preset line
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, '|')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 7 && tokens[0] == "PRESET") {
            // Parse eye position
            std::stringstream eyeSS(tokens[2]);
            std::vector<double> eyeVals;
            while (std::getline(eyeSS, token, ',')) {
                eyeVals.push_back(std::stod(token));
            }
            
            // Parse up vector
            std::stringstream upSS(tokens[3]);
            std::vector<double> upVals;
            while (std::getline(upSS, token, ',')) {
                upVals.push_back(std::stod(token));
            }
            
            // Parse translation
            std::stringstream transSS(tokens[4]);
            std::vector<double> transVals;
            while (std::getline(transSS, token, ',')) {
                transVals.push_back(std::stod(token));
            }
            
            // Parse zoom
            double zoom = std::stod(tokens[5]);
            
            // Parse quaternion
            std::stringstream quatSS(tokens[6]);
            std::vector<double> quatVals;
            while (std::getline(quatSS, token, ',')) {
                quatVals.push_back(std::stod(token));
            }
            
            // Set the preset
            if (eyeVals.size() == 3 && upVals.size() == 3 && 
                transVals.size() == 3 && quatVals.size() == 4) {
                mCameraPresets[i].description = tokens[1];
                mCameraPresets[i].eye << eyeVals[0], eyeVals[1], eyeVals[2];
                mCameraPresets[i].up << upVals[0], upVals[1], upVals[2];
                mCameraPresets[i].trans << transVals[0], transVals[1], transVals[2];
                mCameraPresets[i].zoom = zoom;
                mCameraPresets[i].quat = Eigen::Quaterniond(quatVals[0], quatVals[1], 
                                                            quatVals[2], quatVals[3]);
                mCameraPresets[i].isSet = true;
                
                LOG_INFO("Camera preset " << (i + 1) << " parsed: " << tokens[1]);
            }
        }
    }
}

// ============================================================================
// JOINT ANGLE SWEEP SYSTEM
// ============================================================================

void PhysicalExam::setupSweepMuscles() {
    // Store old visibility state to preserve user preferences
    std::map<std::string, bool> oldVisibility = mMuscleVisibility;

    // Check if we have any previously selected muscles (common muscles from previous sweep)
    bool hasCommonMuscles = false;
    for (const auto& entry : oldVisibility) {
        if (entry.second) {  // If any muscle was visible
            hasCommonMuscles = true;
            break;
        }
    }

    mTrackedMuscles.clear();
    if (!mCharacter) return;
    if (!mUseMuscle) return;  // Guard: no muscles loaded

    auto skel = mCharacter->getSkeleton();
    if (mSweepConfig.joint_index >= skel->getNumJoints()) {
        LOG_ERROR("Invalid joint index: " << mSweepConfig.joint_index);
        return;
    }

    auto joint = skel->getJoint(mSweepConfig.joint_index);
    auto muscles = mCharacter->getMuscles();

    for (auto muscle : muscles) {
        auto related_joints = muscle->GetRelatedJoints();
        if (std::find(related_joints.begin(), related_joints.end(), joint)
            != related_joints.end()) {
            std::string muscleName = muscle->GetName();
            mTrackedMuscles.push_back(muscleName);

            // Register graph data keys for this muscle
            mGraphData->register_key(muscleName + "_fp", 500);
            mGraphData->register_key(muscleName + "_lm", 500);
            mGraphData->register_key(muscleName + "_lm_norm", 500);

            // Initialize visibility
            if (oldVisibility.find(muscleName) != oldVisibility.end()) {
                // Preserve old state for previously tracked muscles
                mMuscleVisibility[muscleName] = oldVisibility[muscleName];
            } else {
                // New muscle: if we have common muscles, default to invisible; otherwise visible
                mMuscleVisibility[muscleName] = !hasCommonMuscles;
            }
        }
    }

    // Remove visibility entries for muscles no longer tracked
    std::map<std::string, bool> newVisibility;
    for (const auto& muscleName : mTrackedMuscles) {
        newVisibility[muscleName] = mMuscleVisibility[muscleName];
    }
    mMuscleVisibility = newVisibility;

    LOG_INFO("Detected " << mTrackedMuscles.size()
              << " muscles crossing joint: "
              << joint->getName());
}

void PhysicalExam::runSweep() {
    if (!mCharacter) {
        LOG_ERROR("No character loaded");
        return;
    }

    // Initialize sweep
    clearSweepData();
    setupSweepMuscles();

    if (mTrackedMuscles.empty()) {
        LOG_INFO("No muscles cross this joint");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(mSweepConfig.joint_index);

    LOG_INFO("Starting sweep: " << joint->getName()
              << " from " << mSweepConfig.angle_min
              << " to " << mSweepConfig.angle_max
              << " rad (" << mSweepConfig.num_steps << " steps)");

    // Store original joint position for restoration
    mSweepOriginalPos = joint->getPositions();

    // Start sweep (will execute incrementally in mainLoop)
    mSweepRunning = true;
    mSweepCurrentStep = 0;
}

void PhysicalExam::collectSweepData(double angle) {
    mSweepAngles.push_back(angle);

    if (!mUseMuscle) return;  // Guard: no muscles loaded

    auto muscles = mCharacter->getMuscles();
    for (auto muscle : muscles) {
        std::string name = muscle->GetName();

        // Only collect data for tracked muscles
        if (std::find(mTrackedMuscles.begin(), mTrackedMuscles.end(), name)
            != mTrackedMuscles.end()) {

            double f_p = muscle->Getf_p();       // Passive force
            double l_m = muscle->lm_rel;         // Muscle length
            double l_m_norm = muscle->lm_norm;   // Normalized muscle length

            mGraphData->push(name + "_fp", f_p);
            mGraphData->push(name + "_lm", l_m);
            mGraphData->push(name + "_lm_norm", l_m_norm);
        }
    }
}

void PhysicalExam::renderMusclePlots() {
    if (mSweepAngles.empty()) return;

    // Display options
    ImGui::Checkbox("Show Legend", &mShowSweepLegend);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Toggle legend display in muscle plots");
    }

    // Convert radian angles to degrees for x_data
    std::vector<double> x_data;
    x_data.reserve(mSweepAngles.size());
    for (double angle_rad : mSweepAngles) {
        x_data.push_back(angle_rad * 180.0 / M_PI);
    }

    // Plot 1: Passive Forces vs Joint Angle
    ImPlotFlags plot_flags = mShowSweepLegend ? 0 : ImPlotFlags_NoLegend;
    if (ImPlot::BeginPlot("Passive Forces vs Joint Angle", ImVec2(-1, 400), plot_flags)) {
        ImPlot::SetupAxisLimits(ImAxis_X1, mSweepConfig.angle_min * 180.0 / M_PI,
            mSweepConfig.angle_max * 180.0 / M_PI, ImGuiCond_Always);
        ImPlot::SetupAxis(ImAxis_X1, "Joint Angle (deg)");
        ImPlot::SetupAxis(ImAxis_Y1, "Passive Force (N)");
        
        // Only plot visible muscles
        for (const auto& muscle_name : mTrackedMuscles) {
            // Check if muscle should be visible
            auto vis_it = mMuscleVisibility.find(muscle_name);
            if (vis_it == mMuscleVisibility.end() || !vis_it->second) {
                continue;  // Skip invisible muscles
            }

            auto fp_data = mGraphData->get(muscle_name + "_fp");
            if (!fp_data.empty() && fp_data.size() == x_data.size()) {
                ImPlot::PlotLine(muscle_name.c_str(), x_data.data(),
                    fp_data.data(), fp_data.size());
            }
        }
        ImPlot::EndPlot();
    }

    ImGui::Spacing();

    // Plot 2: lm_norm vs Joint Angle
    if (ImPlot::BeginPlot("Normalized Muscle Length vs Joint Angle", ImVec2(-1, 400), plot_flags)) {
        ImPlot::SetupAxisLimits(ImAxis_X1, mSweepConfig.angle_min * 180.0 / M_PI,
            mSweepConfig.angle_max * 180.0 / M_PI, ImGuiCond_Always);
        ImPlot::SetupAxis(ImAxis_X1, "Joint Angle (deg)");
        ImPlot::SetupAxis(ImAxis_Y1, "lm_norm");

        // Only plot visible muscles
        for (const auto& muscle_name : mTrackedMuscles) {
            // Check if muscle should be visible
            auto vis_it = mMuscleVisibility.find(muscle_name);
            if (vis_it == mMuscleVisibility.end() || !vis_it->second) {
                continue;  // Skip invisible muscles
            }

            auto lm_norm_data = mGraphData->get(muscle_name + "_lm_norm");
            if (!lm_norm_data.empty() && lm_norm_data.size() == x_data.size()) {
                ImPlot::PlotLine(muscle_name.c_str(), x_data.data(),
                    lm_norm_data.data(), lm_norm_data.size());
            }
        }
        ImPlot::EndPlot();
    }
}

void PhysicalExam::clearSweepData() {
    mSweepAngles.clear();
    mTrackedMuscles.clear();
    // DON'T clear mMuscleVisibility - preserve user selections across sweeps
    if (mGraphData) {
        mGraphData->clear_all();
    }
    LOG_INFO("Sweep data cleared");
}

// ============================================================================
// Control Panel Section Methods
// ============================================================================

void PhysicalExam::drawPosePresetsSection() {
    if (ImGui::CollapsingHeader("Pose Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!mCharacter) {
            ImGui::TextDisabled("Load character first");
        } else {
            ImGui::Text("Select pose:");
            if (ImGui::Button("Standing")) setPoseStanding();
            ImGui::SameLine();
            if (ImGui::Button("Supine")) setPoseSupine();
            ImGui::SameLine();
            if (ImGui::Button("Prone")) setPoseProne();
            if (ImGui::Button("Knee Flexion")) setPoseSupineKneeFlexed(mPresetKneeAngle); ImGui::SameLine();
            ImGui::SetNextItemWidth(150);
            ImGui::SliderFloat("Knee Angle (deg)", &mPresetKneeAngle, 0.0f, 135.0f);
            ImGui::Separator();
            const char* poseNames[] = {"Standing", "Supine", "Prone", "Supine+KneeFlex"};
            ImGui::Text("Current Pose: %s", poseNames[mCurrentPosePreset]);
        }
    }
}

void PhysicalExam::drawForceApplicationSection() {
    if (ImGui::CollapsingHeader("Force Application")) {
        if (!mCharacter) {
            ImGui::TextDisabled("Load character first");
        } else {
            // Body node selection
            ImGui::Text("Target Body:"); ImGui::SameLine();
            const char* bodyNodes[] = {"Pelvis", "FemurR", "FemurL", "TibiaR", "TibiaL", "TalusR", "TalusL"};
            ImGui::SetNextItemWidth(100);
            ImGui::Combo("##BodyNode", &mSelectedBodyNode, bodyNodes, IM_ARRAYSIZE(bodyNodes));
            mForceBodyNode = bodyNodes[mSelectedBodyNode];

            // Confinement force toggle
            ImGui::Checkbox("Confinement", &mApplyConfinementForce);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Apply 500N downward (-Y) force to Pelvis, Torso, ShoulderR, ShoulderL");
            }

            // Posture control toggle
            ImGui::SameLine();
            ImGui::Checkbox("Posture", &mApplyPostureControl);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Apply PI controller to maintain body node positions");
            }

            ImGui::Separator();

            // Force direction
            ImGui::Text("Direction:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(75);
            ImGui::DragFloat("X##ForceDir", &mForceX, 0.01f, -1.0f, 1.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(75);
            ImGui::DragFloat("Y##ForceDir", &mForceY, 0.01f, -1.0f, 1.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(75);
            ImGui::DragFloat("Z##ForceDir", &mForceZ, 0.01f, -1.0f, 1.0f);

            // Force magnitude
            ImGui::Text("Magnitude:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            float forceMag = static_cast<float>(mForceMagnitude);
            if (ImGui::DragFloat("N", &forceMag, 1.0f, 0.0f, 2000.0f)) mForceMagnitude = forceMag;

            ImGui::Separator();

            // Force offset
            ImGui::Text("Application Point Offset:");
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("X##Offset", &mOffsetX, 0.01f, -1.0f, 1.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("Y##Offset", &mOffsetY, 0.01f, -1.0f, 1.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("Z##Offset", &mOffsetZ, 0.01f, -1.0f, 1.0f);

            // Apply/Remove force buttons
            if (!mApplyingForce) {
                if (ImGui::Button("Apply Force")) {
                    mApplyingForce = true;
                }
            } else {
                if (ImGui::Button("Remove Force")) {
                    mApplyingForce = false;
                    mForceMagnitude = 0.0;
                }
            }
        }
    }
}

void PhysicalExam::drawPrintInfoSection() {
    if (ImGui::CollapsingHeader("Print Info")) {
        if (!mCharacter) {
            ImGui::TextDisabled("Load character first");
        } else {
            if (ImGui::Button("Capture Positions", ImVec2(180, 30))) {
                printBodyNodePositions();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Print all body node positions to console");
            }

            if (ImGui::Button("Print Camera to stdout", ImVec2(180, 30))) {
                printCameraInfo();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Print current camera parameters to console");
            }
        }
    }
}

void PhysicalExam::drawRecordingSection() {
    if (ImGui::CollapsingHeader("Recording")) {
        ImGui::Text("Recorded Data Points: %zu", mRecordedData.size());

        if (ImGui::Button("Record Current State")) {
            if (mCharacter) {
                ROMDataPoint data;
                data.force_magnitude = mForceMagnitude;
                std::vector<std::string> joints = {"FemurR", "TibiaR", "TalusR"};
                data.joint_angles = recordJointAngles(joints);
                data.passive_force_total = computePassiveForce();
                mRecordedData.push_back(data);
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Clear Data")) {
            mRecordedData.clear();
        }

        if (!mRecordedData.empty() && ImGui::Button("Export to CSV")) {
            saveToCSV("./results/interactive_exam.csv");
            ImGui::OpenPopup("Export Complete");
        }

        if (ImGui::BeginPopupModal("Export Complete", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Data exported to ./results/interactive_exam.csv");
            if (ImGui::Button("OK", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
}

void PhysicalExam::drawRenderOptionsSection() {
    if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show Exam Table", &mShowExamTable);
        ImGui::Checkbox("Show Joint Forces", &mShowJointPassiveForces);
        if (mShowJointPassiveForces) {
            ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat("Force Arrow Scale", &mJointForceScale, 0.001f, 0.1f, "%.4f");
            ImGui::Checkbox("Show Force Labels", &mShowJointForceLabels);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Display numeric force values at arrow tips");
            }
        }

        ImGui::Checkbox("Show Anchor Points", &mShowAnchorPoints);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Display muscle anchor points as dots instead of lines");
        }

        // Muscle Filtering and Selection
        ImGui::Indent();
        if (ImGui::CollapsingHeader("Muscle##RenderOptions")) {
            if (!mCharacter) {
                ImGui::TextDisabled("Load character first");
            } else {
                ImGui::SetNextItemWidth(100);
                ImGui::DragFloat("Passive Force Normalizer", &mPassiveForceNormalizer, 1.0f, 5.0f, 100.0f);
                ImGui::SetNextItemWidth(100);
                ImGui::SliderFloat("Transparency", &mMuscleTransparency, 0.1f, 1.0f);
        
                auto allMuscles = mCharacter->getMuscles();

                // Initialize selection states if needed
                if (mMuscleSelectionStates.size() != allMuscles.size()) {
                    mMuscleSelectionStates.resize(allMuscles.size(), true);
                }

                // Count selected muscles
                int selectedCount = 0;
                for (bool selected : mMuscleSelectionStates) {
                    if (selected) selectedCount++;
                }

                ImGui::Text("Selected: %d / %zu", selectedCount, allMuscles.size());
                ImGui::Separator();

                // Text filter
                ImGui::InputText("Filter", mMuscleFilterText, IM_ARRAYSIZE(mMuscleFilterText));

                // Filter muscles by name
                std::vector<int> filteredIndices;
                std::string filterStr(mMuscleFilterText);
                std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

                for (int i = 0; i < allMuscles.size(); i++) {
                    std::string muscleName = allMuscles[i]->name;
                    std::transform(muscleName.begin(), muscleName.end(), muscleName.begin(), ::tolower);

                    if (filterStr.empty() || muscleName.find(filterStr) != std::string::npos) {
                        filteredIndices.push_back(i);
                    }
                }

                // Select All / Deselect All buttons for filtered muscles
                if (ImGui::Button("Select")) {
                    for (int idx : filteredIndices) {
                        mMuscleSelectionStates[idx] = true;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Deselect")) {
                    for (int idx : filteredIndices) {
                        mMuscleSelectionStates[idx] = false;
                    }
                }

                ImGui::Text("Filtered Muscles: %zu", filteredIndices.size());

                // Display filtered muscles with checkboxes
                ImGui::BeginChild("MuscleList", ImVec2(0, 150), true);
                for (int idx : filteredIndices) {
                    bool selected = mMuscleSelectionStates[idx];
                    if (ImGui::Checkbox(allMuscles[idx]->name.c_str(), &selected)) {
                        mMuscleSelectionStates[idx] = selected;
                    }
                }
                ImGui::EndChild();
            }
        }
        ImGui::Unindent();

        ImGui::Checkbox("Posture Control Debug", &mShowPostureDebug);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Print posture control debug info to console (positions, errors, forces)");
        }
    }
}

void PhysicalExam::drawJointControlSection() {
    if (ImGui::CollapsingHeader("Joint Angle")) {
        if (!mCharacter) {
            ImGui::TextDisabled("Load character first");
        } else {
            auto skel = mCharacter->getSkeleton();

            // PI Controller mode toggle
            ImGui::Checkbox("Enable PI Controller", &mEnableInterpolation);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("When enabled, uses PI controller to apply torques to reach target joint angles");
            }

            if (mEnableInterpolation) {
                ImGui::SetNextItemWidth(150);
                ImGui::SliderFloat("Kp (Proportional)", (float*)&mJointKp, 10.0f, 2000.0f);
                ImGui::SetNextItemWidth(150);
                ImGui::SliderFloat("Ki (Integral)", (float*)&mJointKi, 1.0f, 200.0f);

                // Reset button - clear all marked targets
                ImGui::SameLine();
                if (ImGui::Button("Reset All")) {
                    for (int i = 0; i < mMarkedJointTargets.size(); ++i) {
                        mMarkedJointTargets[i] = std::nullopt;
                    }
                }

                // Show completion status
                Eigen::VectorXd currentPos = skel->getPositions();
                int marked = 0;
                int completed = 0;
                for (int i = 0; i < mMarkedJointTargets.size(); ++i) {
                    if (mMarkedJointTargets[i].has_value()) {
                        marked++;
                        double targetAngle = mMarkedJointTargets[i].value();
                        if (std::abs(targetAngle - currentPos[i]) <= mInterpolationThreshold) {
                            completed++;
                        }
                    }
                }
                if (marked == 0) {
                    ImGui::Text("No joints marked for interpolation");
                } else if (completed == marked) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Interpolation: Complete (%d/%d marked)", completed, marked);
                } else {
                    ImGui::Text("Interpolation: %d/%d marked DOFs", completed, marked);
                }
            }

            ImGui::Separator();

            // Show warning if interpolation is on and simulation is running
            if (mEnableInterpolation && !mSimulationPaused) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Sliders are read-only during simulation");
            }

            // Joint Position Control
            Eigen::VectorXd pos_lower_limit = skel->getPositionLowerLimits();
            Eigen::VectorXd pos_upper_limit = skel->getPositionUpperLimits();

            // When interpolation is enabled and paused, show marked target angles or current positions
            // This allows user to change targets without them being reset by current positions
            Eigen::VectorXd currentPos = skel->getPositions();
            Eigen::VectorXf pos_rad(currentPos.size());
            if (mEnableInterpolation && mSimulationPaused) {
                // Show target angle if marked, otherwise current position
                for (int i = 0; i < pos_rad.size(); ++i) {
                    pos_rad[i] = mMarkedJointTargets[i].has_value() ?
                             mMarkedJointTargets[i].value() : currentPos[i];
                }
            } else {
                pos_rad = currentPos.cast<float>();
            }

            // Convert to degrees for display
            Eigen::VectorXf pos_deg = pos_rad * (180.0f / M_PI);

            // DOF direction labels
            const char* dof_labels[] = {"X", "Y", "Z", "tX", "tY", "tZ"}; // euler xyz and translation xyz

            int dof_idx = 0;
            for (size_t j = 0; j < skel->getNumJoints(); j++) {
                auto joint = skel->getJoint(j);
                std::string joint_name = joint->getName();
                int num_dofs = joint->getNumDofs();

                if (num_dofs == 0) continue;

                // Display joint name as a header
                ImGui::Text("%s:", joint_name.c_str());
                ImGui::Indent();

                for (int d = 0; d < num_dofs; d++) {
                    // Check if this is a translation DOF (root joint DOFs 3-5 are tx, ty, tz)
                    bool is_translation = (dof_idx >= 3 && dof_idx < 6);

                    // Prepare limits and display value
                    float lower_limit, upper_limit;
                    float display_value;

                    if (dof_idx < 6) {
                        // Root joint - expand limits
                        pos_lower_limit[dof_idx] = -2;
                        pos_upper_limit[dof_idx] = 2;

                        if (is_translation) {
                            // Translation: use raw values (meters)
                            lower_limit = -2.0f;
                            upper_limit = 2.0f;
                            display_value = pos_rad[dof_idx];
                        } else {
                            // Rotation: convert to degrees
                            lower_limit = -2.0f * (180.0f / M_PI);
                            upper_limit = 2.0f * (180.0f / M_PI);
                            display_value = pos_deg[dof_idx];
                        }
                    } else {
                        // Non-root joints: always rotation, convert to degrees
                        lower_limit = pos_lower_limit[dof_idx] * (180.0f / M_PI);
                        upper_limit = pos_upper_limit[dof_idx] * (180.0f / M_PI);
                        display_value = pos_deg[dof_idx];
                    }

                    // Create label: "JointName Direction" or just "JointName" for single DOF
                    std::string label;
                    if (num_dofs > 1 && d < 6) {
                        label = std::string(dof_labels[d]);
                    } else if (num_dofs > 1) {
                        label = "DOF " + std::to_string(d);
                    } else {
                        label = "";
                    }

                    // Add asterisk and target value if joint is marked for interpolation
                    if (mEnableInterpolation && mMarkedJointTargets[dof_idx].has_value()) {
                        double target_value = mMarkedJointTargets[dof_idx].value();
                        char targetStr[32];
                        if (is_translation) {
                            snprintf(targetStr, sizeof(targetStr), " * (→%.3fm)", target_value);
                        } else {
                            snprintf(targetStr, sizeof(targetStr), " * (→%.1f°)", target_value * (180.0 / M_PI));
                        }
                        label += targetStr;
                    }

                    // Store previous value to detect changes
                    float prev_value = display_value;

                    // DragFloat with limits
                    std::string drag_label = label + "##drag_" + joint_name + std::to_string(d);
                    ImGui::SetNextItemWidth(200);
                    const char* format = is_translation ? "%.3fm" : "%.1f°";
                    ImGui::SliderFloat(drag_label.c_str(), &display_value, lower_limit, upper_limit, format);

                    // InputFloat on same line
                    ImGui::SameLine();
                    std::string input_label = "##input_" + joint_name + std::to_string(d);
                    ImGui::SetNextItemWidth(50);
                    const char* input_format = is_translation ? "%.3f" : "%.1f";
                    ImGui::InputFloat(input_label.c_str(), &display_value, 0.0f, 0.0f, input_format);

                    // Clamp to limits after input
                    if (display_value < lower_limit) display_value = lower_limit;
                    if (display_value > upper_limit) display_value = upper_limit;

                    // Update internal storage
                    if (is_translation) {
                        // Translation: store directly in meters
                        pos_rad[dof_idx] = display_value;
                    } else {
                        // Rotation: update degree value and convert to radians
                        pos_deg[dof_idx] = display_value;
                        pos_rad[dof_idx] = display_value * (M_PI / 180.0f);
                    }

                    // Mark joint with target value if changed and interpolation is enabled and paused
                    if (mEnableInterpolation && mSimulationPaused && prev_value != display_value) {
                        mMarkedJointTargets[dof_idx] = pos_rad[dof_idx];  // Store in radians/meters
                    }

                    dof_idx++;
                }

                ImGui::Unindent();
            }

            // Update positions if interpolation is disabled (convert degrees back to radians)
            if (!mEnableInterpolation) {
                skel->setPositions(pos_rad.cast<double>());
            }
            // With interpolation enabled, targets are set when sliders change (marked automatically)
        }
    }
}

void PhysicalExam::drawJointAngleSweepSection() {
    if (ImGui::CollapsingHeader("Joint Angle Sweep", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!mCharacter) {
            ImGui::TextDisabled("Load character first");
        } else {
            auto skel = mCharacter->getSkeleton();

            // Joint selection
            ImGui::Text("Sweep Joint:");
            ImGui::SetNextItemWidth(100);
            int joint_idx = mSweepConfig.joint_index;
            if (ImGui::InputInt("##JointIdx", &joint_idx, 1, 1)) {
                if (joint_idx >= 0 && joint_idx < static_cast<int>(skel->getNumJoints())) {
                    mSweepConfig.joint_index = joint_idx;
                    
                    // Auto-update angle range from joint limits
                    auto joint = skel->getJoint(joint_idx);
                    if (joint->getNumDofs() > 0) {
                        Eigen::VectorXd pos_lower = skel->getPositionLowerLimits();
                        Eigen::VectorXd pos_upper = skel->getPositionUpperLimits();
                        
                        // Get DOF index for this joint
                        int dof_idx = 0;
                        for (size_t j = 0; j < joint_idx; ++j) {
                            dof_idx += skel->getJoint(j)->getNumDofs();
                        }
                        
                        mSweepConfig.angle_min = pos_lower[dof_idx];
                        mSweepConfig.angle_max = pos_upper[dof_idx];
                        
                        LOG_INFO("Joint " << joint->getName() 
                                  << " limits: [" << mSweepConfig.angle_min 
                                  << ", " << mSweepConfig.angle_max << "] rad");
                    }
                }
            }
            ImGui::SameLine();
            if (mSweepConfig.joint_index < static_cast<int>(skel->getNumJoints())) {
                ImGui::Text("%s", skel->getJoint(mSweepConfig.joint_index)->getName().c_str());
            }

            ImGui::Separator();

            // Angle range configuration
            ImGui::Text("Angle Range:");
            ImGui::SetNextItemWidth(120);
            float min_deg = mSweepConfig.angle_min * 180.0 / M_PI;
            if (ImGui::DragFloat("Min (deg)##SweepMin", &min_deg, 1.0f, -180.0f, 180.0f)) {
                mSweepConfig.angle_min = min_deg * M_PI / 180.0;
            }
            ImGui::SameLine();
            ImGui::Text("(%.3f rad)", mSweepConfig.angle_min);

            ImGui::SetNextItemWidth(120);
            float max_deg = mSweepConfig.angle_max * 180.0 / M_PI;
            if (ImGui::DragFloat("Max (deg)##SweepMax", &max_deg, 1.0f, -180.0f, 180.0f)) {
                mSweepConfig.angle_max = max_deg * M_PI / 180.0;
            }
            ImGui::SameLine();
            ImGui::Text("(%.3f rad)", mSweepConfig.angle_max);
            
            // Button to reset to joint limits
            if (ImGui::Button("Use Joint Limits", ImVec2(140, 0))) {
                auto joint = skel->getJoint(mSweepConfig.joint_index);
                if (joint->getNumDofs() > 0) {
                    Eigen::VectorXd pos_lower = skel->getPositionLowerLimits();
                    Eigen::VectorXd pos_upper = skel->getPositionUpperLimits();
                    
                    // Get DOF index for this joint
                    int dof_idx = 0;
                    for (size_t j = 0; j < mSweepConfig.joint_index; ++j) {
                        dof_idx += skel->getJoint(j)->getNumDofs();
                    }
                    
                    mSweepConfig.angle_min = pos_lower[dof_idx];
                    mSweepConfig.angle_max = pos_upper[dof_idx];
                    
                    LOG_INFO("Reset to joint " << joint->getName() 
                              << " limits: [" << mSweepConfig.angle_min 
                              << ", " << mSweepConfig.angle_max << "] rad");
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset angle range to joint's position limits");
            }

            // Number of steps
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("Steps##SweepSteps", &mSweepConfig.num_steps, 5, 10);
            if (mSweepConfig.num_steps < 5) mSweepConfig.num_steps = 5;
            if (mSweepConfig.num_steps > 200) mSweepConfig.num_steps = 200;

            // Restore position option
            ImGui::Checkbox("Restore position after sweep", &mSweepRestorePosition);
            ImGui::Separator();

            // Control buttons
            if (ImGui::Button("Run Sweep", ImVec2(120, 30))) {
                runSweep();
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear Data", ImVec2(120, 30))) {
                clearSweepData();
            }

            // Status display
            ImGui::Separator();
            if (!mSweepAngles.empty()) {
                ImGui::Text("Data points: %zu", mSweepAngles.size());
                ImGui::Text("Tracked muscles: %zu", mTrackedMuscles.size());
            } else {
                ImGui::TextDisabled("No sweep data available");
            }
        }
    }
}
// ============================================================================
// Visualization Panel Section Methods
// ============================================================================

void PhysicalExam::drawTrialManagementSection() {
    if (ImGui::CollapsingHeader("Trial Management")) {
        if (mExamSettingLoaded) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Exam: %s", mExamName.c_str());
            if (!mExamDescription.empty()) {
                ImGui::TextWrapped("%s", mExamDescription.c_str());
            }

            ImGui::Separator();
            ImGui::Text("Total Trials: %zu", mTrials.size());

            if (mCurrentTrialIndex >= 0 && mCurrentTrialIndex < static_cast<int>(mTrials.size())) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f),
                                  "Current Trial: %d / %zu",
                                  mCurrentTrialIndex + 1,
                                  mTrials.size());
                ImGui::Text("Name: %s", mTrials[mCurrentTrialIndex].name.c_str());
                if (!mTrials[mCurrentTrialIndex].description.empty()) {
                    ImGui::TextWrapped("  %s", mTrials[mCurrentTrialIndex].description.c_str());
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "No trial selected");
            }

            ImGui::Separator();

            // Remaining trials
            int remaining = mTrials.size() - (mCurrentTrialIndex + 1);
            if (remaining > 0) {
                ImGui::Text("Remaining Trials: %d", remaining);
                ImGui::Indent();
                for (int i = mCurrentTrialIndex + 1; i < static_cast<int>(mTrials.size()); ++i) {
                    ImGui::BulletText("%d. %s", i + 1, mTrials[i].name.c_str());
                }
                ImGui::Unindent();
            } else if (mCurrentTrialIndex >= 0) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "All trials completed!");
            }

            ImGui::Separator();

            // Start trial button
            bool canStartNext = (mCurrentTrialIndex + 1) < static_cast<int>(mTrials.size());
            if (canStartNext) {
                if (ImGui::Button("Start Next Trial", ImVec2(180, 40))) {
                    startNextTrial();
                }
            } else if (mCurrentTrialIndex < 0) {
                if (ImGui::Button("Start First Trial", ImVec2(180, 40))) {
                    startNextTrial();
                }
            } else {
                ImGui::TextDisabled("All trials completed");
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset Trials", ImVec2(120, 40))) {
                mCurrentTrialIndex = -1;
                mTrialRunning = false;
                mRecordedData.clear();
                LOG_INFO("Trials reset");
            }

        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No exam setting loaded");
            ImGui::TextWrapped("Load an exam setting config to start trials");
        }
    }
}

void PhysicalExam::drawCurrentStateSection() {
    if (ImGui::CollapsingHeader("State", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (mCharacter) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Character Loaded");
            ImGui::Text("Skeleton DOFs: %zu", mCharacter->getSkeleton()->getNumDofs());
            ImGui::Text("Body Nodes: %zu", mCharacter->getSkeleton()->getNumBodyNodes());
            if (mUseMuscle) {  // Guard: only show muscle count if muscles loaded
                ImGui::Text("Muscles: %zu", mCharacter->getMuscles().size());
            } else {
                ImGui::TextDisabled("Muscles: Not loaded");
            }
            ImGui::Separator();
            ImGui::Text("Simulation Time: %.3f s", mWorld->getTime());
            ImGui::Separator();

            ImGui::Text("Applied Force:");
            if (mApplyingForce) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "  Magnitude: %.2f N", mForceMagnitude);
                ImGui::Text("  Body: %s", mForceBodyNode.c_str());
                ImGui::Text("  Direction: [%.2f, %.2f, %.2f]", mForceX, mForceY, mForceZ);
                ImGui::Text("  Offset: [%.2f, %.2f, %.2f]", mOffsetX, mOffsetY, mOffsetZ);
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "  No force applied");
            }

            ImGui::Separator();

            // Muscle passive forces (only if muscles loaded)
            if (mUseMuscle) {  // Guard: only show muscle forces if muscles loaded
                ImGui::Text("Muscle Forces:");
                double total_passive = 0.0;
                std::vector<std::pair<double, std::string>> muscle_forces;

                auto muscles = mCharacter->getMuscles();
                for (auto& muscle : muscles) {
                    double f_p = muscle->Getf_p();
                    total_passive += f_p;
                    muscle_forces.push_back({f_p, muscle->GetName()});
                }

                // Sort by passive force (descending)
                std::sort(muscle_forces.begin(), muscle_forces.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

                ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "  Total Passive: %.2f N", total_passive);

                // Show top 3 muscles with highest passive force
                ImGui::Text("  Top 3 Passive Forces:");
                for (int i = 0; i < std::min(3, (int)muscle_forces.size()); i++) {
                    ImGui::Text("    %d. %s: %.2f N", i+1,
                               muscle_forces[i].second.c_str(),
                               muscle_forces[i].first);
                }

                ImGui::Separator();
            }

            // Current joint angles
            ImGui::Text("Joint Angles:");
            std::vector<std::string> joints = {"FemurR", "TibiaR", "TalusR"};
            auto angles = recordJointAngles(joints);
            for (const auto& [joint, angle] : angles) {
                ImGui::Text("  %s: %.3f rad (%.1f deg)", joint.c_str(), angle[0], angle[0] * 180.0 / M_PI);
            }
        } else {
            ImGui::TextDisabled("Load character to see state");
        }
    }
}

void PhysicalExam::drawRecordedDataSection() {
    if (ImGui::CollapsingHeader("Recorded Data")) {
        ImGui::Text("Total data points: %zu", mRecordedData.size());

        if (!mRecordedData.empty()) {
            ImGui::Separator();

            // Table header
            ImGui::Columns(3, "romdata");
            ImGui::Text("Force (N)");
            ImGui::NextColumn();
            ImGui::Text("Passive (N)");
            ImGui::NextColumn();
            ImGui::Text("Joints");
            ImGui::NextColumn();
            ImGui::Separator();

            // Show last 10 data points
            size_t start = mRecordedData.size() > 10 ? mRecordedData.size() - 10 : 0;
            for (size_t i = start; i < mRecordedData.size(); ++i) {
                ImGui::Text("%.2f", mRecordedData[i].force_magnitude);
                ImGui::NextColumn();
                ImGui::Text("%.2f", mRecordedData[i].passive_force_total);
                ImGui::NextColumn();
                ImGui::Text("%zu", mRecordedData[i].joint_angles.size());
                ImGui::NextColumn();
            }

            ImGui::Columns(1);
        } else {
            ImGui::TextDisabled("No data recorded yet");
        }
    }
}

void PhysicalExam::drawROMAnalysisSection() {
    if (ImGui::CollapsingHeader("ROM Analysis")) {
        if (mRecordedData.size() >= 2) {
            ImGui::Text("Data Range:");
            double minForce = mRecordedData[0].force_magnitude;
            double maxForce = mRecordedData.back().force_magnitude;
            ImGui::Text("  Force: %.2f - %.2f N", minForce, maxForce);

            // Find min/max joint angles
            if (!mRecordedData.empty() && !mRecordedData[0].joint_angles.empty()) {
                std::string firstJoint = mRecordedData[0].joint_angles.begin()->first;
                double minAngle = mRecordedData[0].joint_angles.begin()->second[0];
                double maxAngle = minAngle;

                for (const auto& data : mRecordedData) {
                    if (data.joint_angles.count(firstJoint)) {
                        double angle = data.joint_angles.at(firstJoint)[0];
                        minAngle = std::min(minAngle, angle);
                        maxAngle = std::max(maxAngle, angle);
                    }
                }

                ImGui::Text("  %s ROM: %.2f\u00b0", firstJoint.c_str(), (maxAngle - minAngle) * 180.0 / M_PI);
            }
        } else {
            ImGui::TextDisabled("Record at least 2 data points");
        }
    }
}

void PhysicalExam::drawCameraStatusSection() {
    if (ImGui::CollapsingHeader("Camera Status")) {
        // Current preset description
        if (mCurrentCameraPreset >= 0 && mCurrentCameraPreset < 3 &&
            mCameraPresets[mCurrentCameraPreset].isSet) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Current View:");
            ImGui::SameLine();
            ImGui::Text("%s", mCameraPresets[mCurrentCameraPreset].description.c_str());
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Current View: Custom");
        }

        ImGui::Separator();
        ImGui::Text("Camera Settings:");

        ImGui::Text("Eye: [%.3f, %.3f, %.3f]", mEye[0], mEye[1], mEye[2]);
        ImGui::Text("Up:  [%.3f, %.3f, %.3f]", mUp[0], mUp[1], mUp[2]);
        ImGui::Text("Trans: [%.3f, %.3f, %.3f]", mTrans[0], mTrans[1], mTrans[2]);
        ImGui::Text("Zoom: %.3f", mZoom);

        Eigen::Quaterniond quat = mTrackball.getCurrQuat();
        ImGui::Text("Quaternion: [%.3f, %.3f, %.3f, %.3f]",
                    quat.w(), quat.x(), quat.y(), quat.z());

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Camera Presets:");
        for (int i = 0; i < 3; ++i) {
            if (mCameraPresets[i].isSet) {
                ImGui::Text("Press %d: %s", (i == 0 ? 8 : (i == 1 ? 9 : 0)),
                           mCameraPresets[i].description.c_str());
            }
        }
    }
}

void PhysicalExam::drawSweepMusclePlotsSection() {
    if (ImGui::CollapsingHeader("Sweep Muscle Plots", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (mSweepAngles.empty()) {
            ImGui::TextDisabled("No sweep data available");
            ImGui::TextWrapped("Run a joint angle sweep from the control panel to generate plots");
        } else {
            ImGui::Indent();
            // Muscle Selection Sub-header
            if (ImGui::CollapsingHeader("Muscle Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (mTrackedMuscles.empty()) {
                    ImGui::TextDisabled("No muscles tracked");
                } else {
                    // Filter textbox
                    ImGui::Text("Filter:");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(200);
                    ImGui::InputText("##MuscleFilter", mMuscleFilterBuffer, sizeof(mMuscleFilterBuffer));
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Clear")) {
                        mMuscleFilterBuffer[0] = '\0';
                    }
                    
                    // Convert filter to lowercase for case-insensitive matching
                    std::string filter_lower(mMuscleFilterBuffer);
                    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
                    
                    // Build filtered muscle list
                    std::vector<std::string> filteredMuscles;
                    for (const auto& muscle_name : mTrackedMuscles) {
                        // Case-insensitive substring search
                        std::string muscle_lower = muscle_name;
                        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
                        
                        if (filter_lower.empty() || muscle_lower.find(filter_lower) != std::string::npos) {
                            filteredMuscles.push_back(muscle_name);
                        }
                    }
                    
                    // Count visible muscles (from filtered list)
                    int visibleCount = 0;
                    for (const auto& muscle_name : filteredMuscles) {
                        if (mMuscleVisibility[muscle_name]) {
                            visibleCount++;
                        }
                    }

                    // Display visibility count
                    ImGui::Text("Show (%d/%zu)", visibleCount, filteredMuscles.size());
                    if (!filter_lower.empty()) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "(total %zu)", mTrackedMuscles.size());
                    }
                    ImGui::SameLine();

                    // Select All / Deselect All buttons (operate on filtered list)
                    if (ImGui::SmallButton("Select All")) {
                        for (const auto& muscle_name : filteredMuscles) {
                            mMuscleVisibility[muscle_name] = true;
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Deselect All")) {
                        for (const auto& muscle_name : filteredMuscles) {
                            mMuscleVisibility[muscle_name] = false;
                        }
                    }

                    ImGui::Separator();

                    // Individual muscle checkboxes (in a scrollable region if many muscles)
                    if (filteredMuscles.size() > 10) {
                        // Scrollable region for many muscles
                        ImGui::BeginChild("MuscleCheckboxes", ImVec2(0, 100), true);
                    }

                    for (auto& muscle_name : filteredMuscles) {
                        bool isVisible = mMuscleVisibility[muscle_name];
                        if (ImGui::Checkbox(muscle_name.c_str(), &isVisible)) {
                            mMuscleVisibility[muscle_name] = isVisible;
                        }
                    }

                    if (filteredMuscles.size() > 10) {
                        ImGui::EndChild();
                    }
                }
            }
            ImGui::Unindent();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Render the plots with filtered muscles
            renderMusclePlots();
        }
    }
}

void PhysicalExam::drawMuscleInfoSection() {
    if (ImGui::CollapsingHeader("Muscle Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!mCharacter) {
            ImGui::TextDisabled("No character loaded");
            return;
        }

        auto muscles = mCharacter->getMuscles();
        if (muscles.empty()) {
            ImGui::TextDisabled("No muscles available");
            return;
        }

        // Build muscle name list
        std::vector<std::string> muscleNames;
        for (auto m : muscles) {
            muscleNames.push_back(m->name);
        }

        ImGui::Indent();

        // Filter textbox
        ImGui::Text("Filter:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("##MuscleInfoFilter", mMuscleInfoFilterBuffer, sizeof(mMuscleInfoFilterBuffer));
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear##InfoFilter")) {
            mMuscleInfoFilterBuffer[0] = '\0';
        }

        // Convert filter to lowercase for case-insensitive matching
        std::string filter_lower(mMuscleInfoFilterBuffer);
        std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

        // Build filtered muscle list
        std::vector<std::string> filteredMuscles;
        for (const auto& muscle_name : muscleNames) {
            std::string muscle_lower = muscle_name;
            std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);

            if (filter_lower.empty() || muscle_lower.find(filter_lower) != std::string::npos) {
                filteredMuscles.push_back(muscle_name);
            }
        }

        ImGui::Text("Muscles: %zu", filteredMuscles.size());
        if (!filter_lower.empty() && filteredMuscles.size() < muscleNames.size()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "(total %zu)", muscleNames.size());
        }

        ImGui::Separator();

        // Muscle selection (scrollable list)
        ImGui::Text("Select Muscle:");
        ImGui::BeginChild("MuscleInfoList", ImVec2(0, 50), true);
        for (const auto& muscle_name : filteredMuscles) {
            bool isSelected = (muscle_name == mSelectedMuscleInfo);
            if (ImGui::Selectable(muscle_name.c_str(), isSelected)) {
                mSelectedMuscleInfo = muscle_name;
            }
        }
        ImGui::EndChild();

        // Display detailed muscle information
        if (!mSelectedMuscleInfo.empty()) {
            // Find the selected muscle
            Muscle* selectedMuscle = nullptr;
            for (auto m : muscles) {
                if (m->name == mSelectedMuscleInfo) {
                    selectedMuscle = m;
                    break;
                }
            }

            if (selectedMuscle) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Muscle: %s", mSelectedMuscleInfo.c_str());
                ImGui::Separator();

                // Basic Parameters
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Basic Parameters:");
                ImGui::Columns(4, nullptr, false);
                ImGui::Text("f0:"); ImGui::NextColumn(); ImGui::Text("%.3f N", selectedMuscle->f0); ImGui::NextColumn();
                ImGui::Text("f0_base:"); ImGui::NextColumn(); ImGui::Text("%.3f N", selectedMuscle->f0_base); ImGui::NextColumn();
                ImGui::Text("lm_opt:"); ImGui::NextColumn(); ImGui::Text("%.4f", selectedMuscle->lm_opt); ImGui::NextColumn();
                ImGui::Text("lt_rel:"); ImGui::NextColumn(); ImGui::Text("%.4f", selectedMuscle->lt_rel); ImGui::NextColumn();
                ImGui::Text("lt_rel_base:"); ImGui::NextColumn(); ImGui::Text("%.4f", selectedMuscle->lt_rel_base); ImGui::NextColumn();
                ImGui::Text("lmt_ref:"); ImGui::NextColumn(); ImGui::Text("%.4f m", selectedMuscle->lmt_ref); ImGui::NextColumn();
                ImGui::Text("lmt_base:"); ImGui::NextColumn(); ImGui::Text("%.4f m", selectedMuscle->lmt_base); ImGui::NextColumn();
                ImGui::Text("lmt:"); ImGui::NextColumn(); ImGui::Text("%.4f m", selectedMuscle->lmt); ImGui::NextColumn();
                ImGui::Columns(1);

                ImGui::Separator();

                // Current State (cyan color)
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Current State:");
                ImGui::Columns(4, nullptr, false);
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "lm_rel:"); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "%.4f", selectedMuscle->lm_rel); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "lm_norm:"); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "%.4f", selectedMuscle->lm_norm); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "lmt_rel:"); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "%.4f", selectedMuscle->lmt_rel); ImGui::NextColumn();
                ImGui::Columns(1);

                ImGui::Separator();

                // Modification Parameters
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Modification Parameters:");
                ImGui::Columns(4, nullptr, false);
                ImGui::Text("l_ratio:"); ImGui::NextColumn(); ImGui::Text("%.3f", selectedMuscle->l_ratio); ImGui::NextColumn();
                ImGui::Text("f_ratio:"); ImGui::NextColumn(); ImGui::Text("%.3f", selectedMuscle->f_ratio); ImGui::NextColumn();
                ImGui::Text("lt_rel_ofs:"); ImGui::NextColumn(); ImGui::Text("%.4f", selectedMuscle->lt_rel_ofs); ImGui::NextColumn();
                ImGui::Columns(1);

                ImGui::Separator();

                // Force Outputs (computed)
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Force Outputs:");
                ImGui::Columns(4, nullptr, false);
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "F_psv:"); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%.4f", selectedMuscle->F_psv(selectedMuscle->lm_norm)); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Getf_p:"); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%.3f N", selectedMuscle->Getf_p()); ImGui::NextColumn();
                ImGui::Columns(1);

                ImGui::Separator();

                // Other Info
                ImGui::Columns(2, nullptr, false);
                ImGui::Text("Num Related DOFs:"); ImGui::NextColumn(); ImGui::Text("%d", selectedMuscle->num_related_dofs); ImGui::NextColumn();
                ImGui::Columns(1);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Selected muscle not found!");
            }
        } else {
            ImGui::Separator();
            ImGui::TextDisabled("Select a muscle to view details");
        }

        ImGui::Unindent();
    }
}

// ============================================================================
// SURGERY SCRIPT RECORDING AND EXECUTION
// ============================================================================

void PhysicalExam::startRecording() {
    mRecordedOperations.clear();
    mRecordingSurgery = true;
    LOG_INFO("[Surgery Recording] Started recording surgery operations");
}

void PhysicalExam::stopRecording() {
    mRecordingSurgery = false;
    LOG_INFO("[Surgery Recording] Stopped recording. Captured " 
              << mRecordedOperations.size() << " operation(s)");
}

void PhysicalExam::exportRecording(const std::string& filepath) {
    if (mRecordedOperations.empty()) {
        LOG_ERROR("[Surgery Recording] No operations to export!");
        return;
    }
    
    try {
        SurgeryScript::saveToFile(mRecordedOperations, filepath, "Recorded surgery operations");
        LOG_INFO("[Surgery Recording] Exported " << mRecordedOperations.size() 
                  << " operation(s) to " << filepath);
    } catch (const std::exception& e) {
        LOG_ERROR("[Surgery Recording] Export failed: " << e.what());
    }
}

void PhysicalExam::recordOperation(std::unique_ptr<SurgeryOperation> op) {
    if (!mRecordingSurgery) return;
    
    LOG_INFO("[Surgery Recording] Recorded: " << op->getDescription());
    mRecordedOperations.push_back(std::move(op));
}

void PhysicalExam::loadSurgeryScript() {
    try {
        // Construct full filename with .yaml extension
        std::string loadFilenameWithExt = std::string(mLoadPathBuffer);
        // Remove existing extension if present
        size_t loadLastDot = loadFilenameWithExt.find_last_of('.');
        if (loadLastDot != std::string::npos) {
            loadFilenameWithExt = loadFilenameWithExt.substr(0, loadLastDot);
        }
        loadFilenameWithExt += ".yaml";

        // Construct URI path and resolve it
        std::string loadUriPath = std::string("@data/surgery/") + loadFilenameWithExt;
        URIResolver& resolver = URIResolver::getInstance();
        resolver.initialize();
        std::string loadResolvedPath = resolver.resolve(loadUriPath);

        mLoadedScript = SurgeryScript::loadFromFile(loadResolvedPath);

        if (mLoadedScript.empty()) {
            LOG_ERROR("[Surgery Script] No operations loaded from " << loadResolvedPath);
            return;
        }

        LOG_INFO("[Surgery Script] Loaded " << mLoadedScript.size()
                  << " operation(s) from " << loadResolvedPath);

        // Show preview
        mShowScriptPreview = true;

    } catch (const std::exception& e) {
        LOG_ERROR("[Surgery Script] Failed to load: " << e.what());
    }
}

void PhysicalExam::executeSurgeryScript(std::vector<std::unique_ptr<SurgeryOperation>>& ops) {
    if (!mCharacter) {
        LOG_ERROR("[Surgery Script] Error: No character loaded!");
        return;
    }
    
    LOG_INFO("[Surgery Script] Executing " << ops.size() << " operation(s)...");
    
    int successCount = 0;
    int failCount = 0;
    
    for (size_t i = 0; i < ops.size(); ++i) {
        LOG_INFO("[Surgery Script] Operation " << (i + 1) << "/" << ops.size() 
                  << ": " << ops[i]->getDescription());
        
        bool success = ops[i]->execute(this);
        if (success) {
            successCount++;
        } else {
            failCount++;
            LOG_ERROR("[Surgery Script] Operation " << (i + 1) << " FAILED");
        }
    }
    
    LOG_INFO("[Surgery Script] Execution complete. Success: " << successCount 
              << ", Failed: " << failCount);
}

void PhysicalExam::showScriptPreview() {
    if (!mShowScriptPreview || mLoadedScript.empty()) return;
    
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(mWidth / 2 - 300, mHeight / 2 - 200), ImGuiCond_FirstUseEver);
    
    if (ImGui::Begin("Surgery Script Preview", &mShowScriptPreview)) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 1.0f, 1.0f), "Script Preview");
        ImGui::Separator();
        
        ImGui::Text("Total operations: %zu", mLoadedScript.size());
        ImGui::Spacing();
        
        // Display operations in scrollable region
        ImGui::BeginChild("ScriptOperations", ImVec2(0, 250), true);
        for (size_t i = 0; i < mLoadedScript.size(); ++i) {
            ImGui::Text("%zu. %s", i + 1, mLoadedScript[i]->getDescription().c_str());
        }
        ImGui::EndChild();
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Action buttons
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

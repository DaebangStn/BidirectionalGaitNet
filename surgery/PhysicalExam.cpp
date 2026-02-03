#include <glad/glad.h>
#include "PhysicalExam.h"
#include "Log.h"
#include "rm/rm.hpp"
#include "common/GLfunctions.h"
#include "SurgeryScript.h"
#include "optimizer/ContractureOptimizer.h"
#include "common/imgui_common.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <set>
#include <chrono>
#include <ctime>
#include <imgui.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <GL/glu.h>
#define TINYCOLORMAP_WITH_EIGEN
#include <tinycolormap.hpp>

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
    : ViewerAppBase("Muscle Surgery", width, height)  // Base class handles GLFW/ImGui
    , SurgeryExecutor("physical_exam")
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
    , mSelectedTrialFileIndex(-1)
    , mPassiveForceNormalizer(10.0f)  // Default normalization factor
    , mMuscleTransparency(1.0f)        // Default muscle transparency (0.0-1.0)
    , mShowJointPassiveForces(false)   // Show joint passive forces by default
    , mJointForceScale(0.01f)          // Default scale for force arrows
    , mShowJointForceLabels(false)     // Labels off by default
    , mTopPassiveForcesCount(3)       // Show top 3 passive forces by default
    , mShowPostureDebug(false)        // Posture control debug off by default
    , mVerboseTorque(false)           // Verbose torque debug off by default
    , mVisualizeSweep(false)          // Step-by-step sweep visualization off by default
    , mSweepNextPressed(false)
    , mSweepQuitPressed(false)
    , mShowExamTable(false)            // Show examination table by default
    , mShowAnchorPoints(false)         // Anchor point visualization off by default
    , mApplyPostureControl(true)
    , mGraphData(nullptr)
    , mEnableInterpolation(false)
    , mJointKp(500.0)               // Proportional gain
    , mJointKi(50.0)                // Integral gain
    , mInterpolationThreshold(0.01)   // 0.01 radians threshold
    , mSweepRestorePosition(false)   // Restore position after sweep by default
    // mRenderMode inherited from ViewerAppBase (defaults to Wireframe)
    , mStdCharacter(nullptr)  // Initialize to nullptr
    , mRenderMainCharacter(true)  // Default to rendering main character
    , mRenderStdCharacter(false)  // Default to not rendering std character
    , mShowStdCharacterInPlots(true)  // Default to showing std character in plots
    , mPlotWhiteBackground(false)  // Default to dark plot background
    , mShowTrialNameInPlots(true)  // Default to showing trial name
    , mShowCharacterInTitles(true)  // Default to not showing character info in titles
    , mCurrentSweepName("GUI Sweep")  // Default sweep name
{
    mForceBodyNode = "FemurR";  // Default body node
    mMuscleFilterBuffer[0] = '\0';  // Initialize filter buffer as empty string

    // Muscle Selection UI
    std::memset(mMuscleFilterText, 0, sizeof(mMuscleFilterText));

    // Initialize muscle info panel
    mMuscleInfoFilterBuffer[0] = '\0';  // Initialize muscle info filter buffer
    mSelectedMuscleInfo = "";  // No muscle selected initially

    // Initialize camera presets from preset definitions below
    initializeCameraPresets();
    // Initialize graph data for posture control
    mGraphData = new CBufferData<double>();

    // Initialize sweep configuration
    mSweepConfig.joint_index = 1;
    mSweepConfig.dof_index = 0;
    mSweepConfig.angle_min = -1.57;  // -90 degrees
    mSweepConfig.angle_max = 1.57;   // +90 degrees
    mSweepConfig.num_steps = 50;
    mSweepRunning = false;
    mSweepCurrentStep = 0;
    mSelectedPlotJointIndex = 0;  // Will be set to sweep joint when sweep starts

    // Create DART world early (needed for loadExamSetting which is called before startLoop)
    mWorld = dart::simulation::World::create();
    mWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));
    mWorld->setTimeStep(1.0 / mSimulationHz);

    // Scan trial files from directory
    scanTrialFiles();

    // Initialize surgery panel (character will be set later via setCharacter)
    mSurgeryPanel = std::make_unique<SurgeryPanel>(nullptr, &mShapeRenderer);
}

PhysicalExam::~PhysicalExam() {
    // Note: Character has no destructor, so we don't delete it
    // It will be cleaned up when the program exits
    
    // Clean up standard character
    if (mStdCharacter) {
        delete mStdCharacter;
        mStdCharacter = nullptr;
    }

    if (mGraphData) {
        delete mGraphData;
        mGraphData = nullptr;
    }

    // GLFW/ImGui cleanup handled by ViewerAppBase destructor
}

void PhysicalExam::loadClinicalROM(const std::string& pid, const std::string& visit) {
    mClinicalROM.clear();
    mClinicalROMPID = pid;
    mClinicalROMVisit = visit;

    if (pid.empty() || visit.empty()) {
        LOG_INFO("[ClinicalROM] No PID or visit specified, clearing clinical ROM data");
        return;
    }

    // Get PID root directory from resource manager
    namespace fs = std::filesystem;
    fs::path pidRoot = rm::getManager().getPidRoot();
    if (pidRoot.empty()) {
        LOG_WARN("[ClinicalROM] PID root not configured");
        return;
    }

    // Construct path to rom.yaml: {pid_root}/{pid}/{visit}/rom.yaml
    fs::path romPath = pidRoot / pid / visit / "rom.yaml";
    if (!fs::exists(romPath)) {
        LOG_INFO("[ClinicalROM] No rom.yaml found at: " << romPath.string());
        return;
    }

    try {
        YAML::Node romConfig = YAML::LoadFile(romPath.string());

        // Parse ROM data structure: rom.{side}.{joint}.{field} = value
        // Expected format:
        // rom:
        //   left:
        //     hip:
        //       abduction_ext_r2: 35.0
        //   right:
        //     hip:
        //       abduction_ext_r2: 38.0
        if (romConfig["rom"]) {
            YAML::Node romData = romConfig["rom"];
            for (auto sideIt = romData.begin(); sideIt != romData.end(); ++sideIt) {
                std::string side = sideIt->first.as<std::string>();
                YAML::Node sideData = sideIt->second;

                for (auto jointIt = sideData.begin(); jointIt != sideData.end(); ++jointIt) {
                    std::string joint = jointIt->first.as<std::string>();
                    YAML::Node jointData = jointIt->second;

                    for (auto fieldIt = jointData.begin(); fieldIt != jointData.end(); ++fieldIt) {
                        std::string field = fieldIt->first.as<std::string>();
                        if (fieldIt->second.IsScalar()) {
                            try {
                                float value = fieldIt->second.as<float>();
                                std::string key = side + "." + joint + "." + field;
                                mClinicalROM[key] = value;
                            } catch (const YAML::BadConversion&) {
                                // Skip non-numeric values
                            }
                        }
                    }
                }
            }
        }

        LOG_INFO("[ClinicalROM] Loaded " << mClinicalROM.size() << " ROM values from: " << romPath.string());
    } catch (const std::exception& e) {
        LOG_ERROR("[ClinicalROM] Failed to parse rom.yaml: " << e.what());
    }
}

void PhysicalExam::onBrowsePIDChanged(const std::string& pid)
{
    mBrowseCharacterPID = pid;

    if (pid.empty()) {
        mClinicalWeightAvailable = false;
        mClinicalROM.clear();
        mClinicalROMPID.clear();
        mClinicalROMVisit.clear();
        return;
    }

    // Get visit from navigator state
    const auto& pidState = mPIDNavigator->getState();
    std::string visit = pidState.getVisitDir();

    // Rescan files when PID changes
    if (mBrowseSkeletonDataSource == CharacterDataSource::PatientData) {
        scanSkeletonFilesForBrowse();
    }
    if (mBrowseMuscleDataSource == CharacterDataSource::PatientData) {
        scanMuscleFilesForBrowse();
    }

    // Load clinical ROM data for the new PID
    loadClinicalROM(pid, visit);
}

void PhysicalExam::onBrowseVisitChanged(const std::string& pid, const std::string& visit)
{
    if (pid.empty() || visit.empty()) return;

    // Rescan files when visit changes (if using patient data)
    if (mBrowseSkeletonDataSource == CharacterDataSource::PatientData) {
        scanSkeletonFilesForBrowse();
    }
    if (mBrowseMuscleDataSource == CharacterDataSource::PatientData) {
        scanMuscleFilesForBrowse();
    }

    // Reload clinical ROM data for the new visit
    loadClinicalROM(pid, visit);
}

void PhysicalExam::loadRenderConfigImpl() {
    // Common config (geometry, default_open_panels) already loaded by ViewerAppBase
    // Uses inherited mControlPanelWidth and mPlotPanelWidth from geometry.control/plot

    // Load normative ROM values from ROM config files
    // Source: Moon, Seung Jun, et al. "Normative values of physical examinations
    //         commonly used for cerebral palsy." Yonsei medical journal 58.6 (2017): 1170-1176.
    namespace fs = std::filesystem;
    try {
        std::string rom_dir_uri = "@data/config/rom";
        std::filesystem::path resolved_path = rm::getManager().resolveDir(rom_dir_uri);

        if (!resolved_path.empty() && fs::exists(resolved_path) && fs::is_directory(resolved_path)) {
            for (const auto& entry : fs::directory_iterator(resolved_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".yaml") {
                    try {
                        YAML::Node config = YAML::LoadFile(entry.path().string());
                        if (config["exam"] && config["exam"]["alias"] && config["exam"]["normative"]) {
                            std::string alias = config["exam"]["alias"].as<std::string>();
                            double normative = config["exam"]["normative"].as<double>();

                            // Remove the side suffix ("/left" or "/right") from alias
                            // e.g., "hip/abduction_knee0/left" -> "hip/abduction_knee0"
                            size_t lastSlash = alias.rfind('/');
                            if (lastSlash != std::string::npos) {
                                std::string key = alias.substr(0, lastSlash);
                                // Only add if not already present (avoid duplicates from L/R pairs)
                                if (mNormativeROM.find(key) == mNormativeROM.end()) {
                                    mNormativeROM[key] = normative;
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("[PhysicalExam] Error parsing ROM config " << entry.path().string() << ": " << e.what());
                    }
                }
            }
            LOG_INFO("[PhysicalExam] Loaded " << mNormativeROM.size() << " normative ROM values from ROM configs");
        }
    } catch (const std::exception& e) {
        LOG_WARN("[PhysicalExam] Could not load normative ROM from config files: " << e.what());
    }
}

void PhysicalExam::onInitialize() {
    // GLFW/ImGui already initialized by ViewerAppBase constructor

    // Initialize GLUT (needed for glutSolidCube/glutSolidSphere in rendering)
    int argc = 1;
    char* argv[1] = {(char*)"physical_exam"};
    glutInit(&argc, argv);

    // Set window position from config
    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);

    // Initialize trackball (using inherited mCamera)
    mCamera.trackball.setTrackball(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5), mWidth * 0.5);
    mCamera.trackball.setQuaternion(Eigen::Quaterniond::Identity());
    loadCameraPreset(0);

    // mWorld already created in constructor (needed for loadExamSetting)

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

    // Initialize PIDNavigator for patient data browsing
    rm::ResourceManager* resourceManager = &rm::getManager();
    if (resourceManager) {
        mPIDNavigator = std::make_unique<PIDNav::PIDNavigator>(
            resourceManager,
            nullptr  // No file filter - just for PID selection
        );
        mPIDNavigator->setPIDChangeCallback([this](const std::string& pid) {
            onBrowsePIDChanged(pid);
        });
        mPIDNavigator->setVisitChangeCallback([this](const std::string& pid, const std::string& visit) {
            onBrowseVisitChanged(pid, visit);
        });
        mPIDNavigator->scanPIDs();
    }

    // Initialize skeleton/muscle browse lists
    scanSkeletonFilesForBrowse();
    scanMuscleFilesForBrowse();
}

void PhysicalExam::onFrameStart() {
    // Execute one sweep step if sweep is running
    if (mSweepRunning && mCharacter) {
        auto skel = mCharacter->getSkeleton();
        auto joint = skel->getJoint(mSweepConfig.joint_index);

        if (mSweepCurrentStep <= mSweepConfig.num_steps) {
            double angle = mSweepConfig.angle_min +
                (mSweepConfig.angle_max - mSweepConfig.angle_min) *
                mSweepCurrentStep / (double)mSweepConfig.num_steps;

            Eigen::VectorXd pos = joint->getPositions();
            pos[mSweepConfig.dof_index] = angle;
            setCharacterPose(joint->getName(), pos);

            if (!mCharacter->getMuscles().empty()) {
                mCharacter->getMuscleTuple();
            }
            if (mStdCharacter && !mStdCharacter->getMuscles().empty()) {
                mStdCharacter->getMuscleTuple();
            }

            collectAngleSweepData(angle, mSweepConfig.joint_index);
            mSweepCurrentStep++;
        } else {
            if (mSweepRestorePosition) {
                setCharacterPose(joint->getName(), mSweepOriginalPos);
            }
            mSweepRunning = false;
            LOG_INFO("Sweep completed. Collected " << mAngleSweepData.size() << " data points");

            // Add GUI sweep results to trial buffer for visualization
            if (!mAngleSweepData.empty()) {
                TrialDataBuffer buffer;
                buffer.trial_name = mCurrentSweepName;
                buffer.trial_description = "Manual GUI sweep";
                buffer.timestamp = std::chrono::system_clock::now();
                buffer.angle_sweep_data = mAngleSweepData;
                buffer.std_angle_sweep_data = mStdAngleSweepData;
                buffer.tracked_muscles = mAngleSweepTrackedMuscles;
                buffer.std_tracked_muscles = mStdAngleSweepTrackedMuscles;
                // Convert JointSweepConfig to AngleSweepTrialConfig
                buffer.config.joint_name = joint->getName();
                buffer.config.dof_index = mSweepConfig.dof_index;
                buffer.config.angle_min = mSweepConfig.angle_min;
                buffer.config.angle_max = mSweepConfig.angle_max;
                buffer.config.num_steps = mSweepConfig.num_steps;
                // torque_cutoff uses default (15.0) for GUI sweep
                buffer.cutoff_angles = computeCutoffAngles(mAngleSweepData, buffer.torque_cutoff);
                buffer.rom_metrics = computeROMMetrics(mAngleSweepData);
                if (!mStdAngleSweepData.empty()) {
                    buffer.std_rom_metrics = computeROMMetrics(mStdAngleSweepData);
                }
                buffer.base_pose = mCharacter->getSkeleton()->getPositions();  // Store full skeleton pose
                addTrialToBuffer(buffer);
                // Select the newly added buffer
                mSelectedBufferIndex = static_cast<int>(mTrialBuffers.size()) - 1;
            }
        }

        if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            if (mSweepRestorePosition) {
                setCharacterPose(joint->getName(), mSweepOriginalPos);
            }
            mSweepRunning = false;
            LOG_INFO("Sweep interrupted by user");
        }
    }

    // Simulation stepping
    for (int step = 0; step < 5; step++) {
        if (mCharacter) mCharacter->step();

        bool shouldStep = !mSimulationPaused || mSingleStep;
        if (shouldStep) {
            if (mEnableInterpolation && mCharacter) {
                auto skel = mCharacter->getSkeleton();
                Eigen::VectorXd currentPos = skel->getPositions();
                Eigen::VectorXd currentForces = skel->getForces();
                double dt = mWorld->getTimeStep();

                for (size_t i = 0; i < mMarkedJointTargets.size(); ++i) {
                    if (!mMarkedJointTargets[i].has_value()) continue;

                    double targetAngle = mMarkedJointTargets[i].value();
                    double error = targetAngle - currentPos[i];

                    if (std::abs(error) > mInterpolationThreshold) {
                        mJointIntegralError[i] += error * dt;
                        double torque = mJointKp * error + mJointKi * mJointIntegralError[i];
                        currentForces[i] += torque;
                    } else {
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
        if (mCharacter) mCharacter->setZeroForces();
    }
}

void PhysicalExam::drawUI() {
    drawLeftPanel();
    drawRightPanel();
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

    // Resolve URIs
    std::string resolved_skel = rm::resolve(skel_path);

    // Create character
    mCharacter = new Character(resolved_skel, SKEL_COLLIDE_ALL);

    // Load muscles if path is provided
    if (!muscle_path.empty()) {
        std::string resolved_muscle = rm::resolve(muscle_path);
        mCharacter->setMuscles(resolved_muscle);

        // Zero muscle activations
        if (mCharacter->getMuscles().size() > 0) {
            mCharacter->setActivations(mCharacter->getActivations().setZero());
        }
    } else {
        LOG_INFO("Skipping muscle loading (no muscle path provided)");
    }

    // Set actuator type
    mCharacter->setActuatorType(_actType);

    // Add to world
    mWorld->addSkeleton(mCharacter->getSkeleton());

    // Load standard character if paths are set
    if (!mStdSkeletonPath.empty()) {
        std::string resolved_std_skel = rm::resolve(mStdSkeletonPath);
        
        mStdCharacter = new Character(resolved_std_skel, SKEL_COLLIDE_ALL);
        mStdCharacter->getSkeleton()->setName("Human_std");  // Avoid duplicate name warning

        if (!mStdMusclePath.empty()) {
            std::string resolved_std_muscle = rm::resolve(mStdMusclePath);
            mStdCharacter->setMuscles(resolved_std_muscle);
            
            if (mStdCharacter->getMuscles().size() > 0) {
                mStdCharacter->setActivations(mStdCharacter->getActivations().setZero());
            }
        }
        
        LOG_VERBOSE("Standard character loaded successfully");
    }

    // Set initial pose to supine (laying on back on examination bed)
    setPoseSupine();

    // Setup posture control targets (must be called after character is loaded)
    // setupPostureTargets();

    // Initialize marked joint targets and PI controller state (all unmarked initially)
    auto skel = mCharacter->getSkeleton();
    mMarkedJointTargets.resize(skel->getNumDofs(), std::nullopt);
    mJointIntegralError.resize(skel->getNumDofs(), 0.0);
    LOG_VERBOSE("Initialized joint PI controller system (" << skel->getNumDofs() << " DOFs)");

    // Initialize muscle selection states
    auto muscles = mCharacter->getMuscles();
    mMuscleSelectionStates.clear();
    if (muscles.size() > 0) {
        mMuscleSelectionStates.resize(muscles.size(), true);  // All muscles selected by default
    }

    LOG_VERBOSE("Character loaded successfully in supine position" << (muscle_path.empty() ? " (skeleton only)" : ""));

    // Update SurgeryPanel with the new character
    if (mSurgeryPanel) {
        mSurgeryPanel->setCharacter(mCharacter);
    }
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

        // Use synced method to set both main and std character
        setCharacterPose(joint_name, angles);
    }

    // Store as initial pose for reset
    mInitialPose = joint_angles;
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

double PhysicalExam::getPassiveTorqueJoint_forCharacter(Character* character, dart::dynamics::Joint* joint) {
    if (!character || !joint) return 0.0;
    if (character->getMuscles().empty()) return 0.0;

    // Get DOF range for this joint
    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int num_dofs = static_cast<int>(joint->getNumDofs());

    double total_torque = 0.0;
    auto muscles = character->getMuscles();

    for (auto& muscle : muscles) {
        // GetRelatedJtp() returns a reduced vector indexed by related_dof_indices
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        const auto& related_indices = muscle->related_dof_indices;

        // Check each related DOF to see if it's in this joint's range
        for (size_t i = 0; i < related_indices.size(); ++i) {
            int dof_idx = related_indices[i];
            if (dof_idx >= first_dof && dof_idx < first_dof + num_dofs) {
                total_torque += jtp[i];
            }
        }
    }

    return total_torque;
}

double PhysicalExam::getPassiveTorqueJoint(int joint_idx) {
    if (!mCharacter) return 0.0;
    if (mCharacter->getMuscles().empty()) return 0.0;

    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(joint_idx);
    if (!joint) return 0.0;

    // Get DOF range for this joint
    int first_dof = static_cast<int>(joint->getIndexInSkeleton(0));
    int num_dofs = static_cast<int>(joint->getNumDofs());

    double total_torque = 0.0;
    auto muscles = mCharacter->getMuscles();

    for (auto& muscle : muscles) {
        // GetRelatedJtp() returns a reduced vector indexed by related_dof_indices
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        const auto& related_indices = muscle->related_dof_indices;

        // Check each related DOF to see if it's in this joint's range
        for (size_t i = 0; i < related_indices.size(); ++i) {
            int global_dof = related_indices[i];
            if (global_dof >= first_dof && global_dof < first_dof + num_dofs) {
                total_torque += jtp[i];
            }
        }
    }

    return total_torque;
}

double PhysicalExam::getPassiveTorqueJointGlobalY(
    Character* character, dart::dynamics::Joint* joint) {
    // Compute physical passive torque about joint and project onto global Y axis
    // Uses cross product (r × F) to correctly compute physical torque,
    // avoiding exponential map Jacobian issues with BallJoint generalized forces
    if (!character || !joint) return 0.0;
    if (character->getMuscles().empty()) return 0.0;

    int num_dofs = static_cast<int>(joint->getNumDofs());

    // Only applies to 3-DOF joints (BallJoint)
    if (num_dofs != 3) {
        return getPassiveTorqueJoint_forCharacter(character, joint);  // Fallback
    }

    auto* child_body = joint->getChildBodyNode();
    Eigen::Vector3d joint_center = child_body->getTransform().translation();

    if (joint_center.hasNaN()) {
        return 0.0;
    }

    // Build set of descendant bodies (bodies affected by this joint)
    std::set<dart::dynamics::BodyNode*> descendant_bodies;
    std::function<void(dart::dynamics::BodyNode*)> collect_descendants;
    collect_descendants = [&](dart::dynamics::BodyNode* bn) {
        descendant_bodies.insert(bn);
        for (size_t i = 0; i < bn->getNumChildBodyNodes(); ++i) {
            collect_descendants(bn->getChildBodyNode(i));
        }
    };
    collect_descendants(child_body);

    double total_torque = 0.0;
    for (auto& muscle : character->getMuscles()) {
        Eigen::Vector3d torque_world = muscle->GetPassiveTorqueAboutPoint(
            joint_center, &descendant_bodies);
        total_torque += torque_world.y();
    }

    // Debug output for comparison with JTP Y-axis
    if (mVerboseTorque) {
        std::cout << "  [getPassiveTorqueJointGlobalY] joint=" << joint->getName()
                  << " joint_center=[" << joint_center.transpose() << "]"
                  << " descendant_bodies=" << descendant_bodies.size()
                  << " raw_total=" << total_torque
                  << " negated=" << -total_torque << std::endl;
    }

    // Negate to match renderer convention (global Y torque sign was reversed)
    return -total_torque;
}

double PhysicalExam::getPassiveTorqueJointDof(
    Character* character, dart::dynamics::Joint* joint, int dof_index) {
    if (!character || !joint) return 0.0;
    if (character->getMuscles().empty()) return 0.0;

    // Get the skeleton DOF index for this specific joint DOF
    int skel_dof_idx = joint->getIndexInSkeleton(dof_index);

    double total_torque = 0.0;
    for (auto& muscle : character->getMuscles()) {
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        const auto& related_indices = muscle->related_dof_indices;

        for (size_t i = 0; i < related_indices.size(); ++i) {
            if (related_indices[i] == skel_dof_idx) {
                total_torque += jtp[i];
                break;  // Only one contribution per muscle per DOF
            }
        }
    }
    return total_torque;
}

// Pose synchronization methods
void PhysicalExam::setCharacterPose(const Eigen::VectorXd& positions) {
    if (mCharacter) {
        mCharacter->getSkeleton()->setPositions(positions);
    } else {
        LOG_WARN("No main character loaded");
    }
    
    if (mStdCharacter) {
        auto std_skel = mStdCharacter->getSkeleton();
        // Only set if the DOF counts match
        if (std_skel->getNumDofs() == positions.size()) {
            std_skel->setPositions(positions);
        }
    } else {
        LOG_WARN("No standard character loaded");
    }
}

void PhysicalExam::setCharacterPose(const std::string& joint_name, const Eigen::VectorXd& positions) {
    if (mCharacter) {
        auto joint = mCharacter->getSkeleton()->getJoint(joint_name);
        if (joint) {
            joint->setPositions(positions);
        }
    } else {
        LOG_WARN("No main character loaded: " << joint_name);
    }
    
    if (mStdCharacter) {
        auto std_joint = mStdCharacter->getSkeleton()->getJoint(joint_name);
        if (std_joint && std_joint->getNumDofs() == positions.size()) {
            std_joint->setPositions(positions);
        }
    } else {
        LOG_WARN("No standard character loaded: " << joint_name);
    }
}

void PhysicalExam::loadExamSetting(const std::string& config_path) {
    // Store original config path for output naming
    mExamConfigPath = config_path;

    // Resolve URI if needed
    std::string resolved_config_path = rm::resolve(config_path);

    LOG_INFO("Loading exam setting from: " << resolved_config_path);
    
    // Parse YAML configuration
    YAML::Node config = YAML::LoadFile(resolved_config_path);

    mExamName = config["name"].as<std::string>();
    mExamDescription = config["description"] ? config["description"].as<std::string>() : "";
    
    std::string skeleton_path = config["character"]["skeleton"].as<std::string>();
    std::string muscle_path = config["character"]["muscle"] ? config["character"]["muscle"].as<std::string>() : "";
    std::string _actTypeString = config["character"]["actuator"].as<std::string>();
    ActuatorType _actType = getActuatorType(_actTypeString);

    // Set standard character paths from config before loading main character
    if (config["std_character"]) {
        mStdSkeletonPath = config["std_character"]["skeleton"].as<std::string>();
        mStdMusclePath = config["std_character"]["muscle"] 
            ? config["std_character"]["muscle"].as<std::string>() 
            : "";
    } else {
        mStdSkeletonPath = "";
        mStdMusclePath = "";
        mStdCharacter = nullptr;
    }
    
    loadCharacter(skeleton_path, muscle_path, _actType);
    
    // Initialize rendering flags
    mRenderMainCharacter = true;
    mRenderStdCharacter = false;  // Default to only showing main
    mShowStdCharacterInPlots = true;  // Default to showing overlay in plots
    
    mExamSettingLoaded = true;
    mCurrentTrialIndex = -1;
    mTrialRunning = false;
    setPaused(true);

    // Parse trials from config
    mTrials.clear();
    if (config["trials"]) {
        for (const auto& trial_entry : config["trials"]) {
            if (trial_entry["file"]) {
                // Load trial from external file
                std::string trial_file = trial_entry["file"].as<std::string>();
                std::string resolved_trial = rm::resolve(trial_file);
                try {
                    YAML::Node trial_node = YAML::LoadFile(resolved_trial);
                    TrialConfig trial = parseTrialConfig(trial_node);
                    mTrials.push_back(trial);
                    LOG_INFO("Loaded trial from file: " << trial.name);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load trial file " << resolved_trial << ": " << e.what());
                }
            } else {
                // Inline trial definition
                try {
                    TrialConfig trial = parseTrialConfig(trial_entry);
                    mTrials.push_back(trial);
                    LOG_INFO("Loaded inline trial: " << trial.name);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to parse inline trial: " << e.what());
                }
            }
        }
        LOG_VERBOSE("Loaded " << mTrials.size() << " trial(s) from config");
    }
}

void PhysicalExam::scanTrialFiles() {
    namespace fs = std::filesystem;
    
    mAvailableTrialFiles.clear();
    
    // Resolve the ROM config directory URI using resolveDir for directories
    std::string rom_dir_uri = "@data/config/rom";
    std::filesystem::path resolved_path = rm::getManager().resolveDir(rom_dir_uri);
    
    if (resolved_path.empty() || !fs::exists(resolved_path) || !fs::is_directory(resolved_path)) {
        LOG_WARN("ROM config directory not found: " << rom_dir_uri);
        return;
    }
    
    std::string resolved_dir = resolved_path.string();
    
    // Scan for YAML files
    for (const auto& entry : fs::directory_iterator(resolved_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".yaml") {
            try {
                // Load YAML to extract name
                std::string file_path = entry.path().string();
                YAML::Node trial_node = YAML::LoadFile(file_path);
                
                TrialFileInfo info;
                info.file_path = file_path;
                
                // Extract name field
                if (trial_node["name"]) {
                    info.name = trial_node["name"].as<std::string>();
                } else {
                    // Fallback to filename if name not found
                    info.name = entry.path().filename().string();
                }
                
                mAvailableTrialFiles.push_back(info);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load trial file " << entry.path().string() << ": " << e.what());
            }
        }
    }
    
    // Sort alphabetically by name
    std::sort(mAvailableTrialFiles.begin(), mAvailableTrialFiles.end(),
              [](const TrialFileInfo& a, const TrialFileInfo& b) {
                  return a.name < b.name;
              });
    
    LOG_INFO("Scanned " << mAvailableTrialFiles.size() << " ROM config file(s) from " << resolved_dir);
}

TrialConfig PhysicalExam::parseTrialConfig(const YAML::Node& trial_node) {
    TrialConfig trial;

    // Parse trial configuration
    trial.name = trial_node["name"].as<std::string>();
    trial.description = trial_node["description"] ?
        trial_node["description"].as<std::string>() : "";

    // Parse pose (values in degrees, converted to radians)
    YAML::Node pose_node = trial_node["pose"];
    for (YAML::const_iterator it = pose_node.begin(); it != pose_node.end(); ++it) {
        std::string joint_name = it->first.as<std::string>();
        bool is_root = (joint_name == "Pelvis");

        if (it->second.IsSequence()) {
            std::vector<double> values = it->second.as<std::vector<double>>();
            Eigen::VectorXd angles(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                // For root joint: first 3 are rotations (convert), last 3 are translations (keep)
                // For other joints: all are rotations (convert)
                if (is_root && i >= 3) {
                    angles[i] = values[i];  // Translation - keep as meters
                } else {
                    angles[i] = values[i] * M_PI / 180.0;  // Rotation - degrees to radians
                }
            }
            trial.pose[joint_name] = angles;
        } else {
            Eigen::VectorXd angles(1);
            angles[0] = it->second.as<double>() * M_PI / 180.0;  // Degrees to radians
            trial.pose[joint_name] = angles;
        }
    }

    // Detect format: unified ROM format vs old trial format
    // Unified ROM format has top-level "joint" and "exam" section
    // Old format has "mode" and "angle_sweep" or "force" section
    bool is_unified_format = trial_node["joint"] && trial_node["exam"];

    if (is_unified_format) {
        // Parse unified ROM config format
        trial.mode = TrialMode::ANGLE_SWEEP;

        // Parse top-level joint and DOF
        trial.angle_sweep.joint_name = trial_node["joint"].as<std::string>();

        // Handle dof_index or dof (for composite DOF like abd_knee)
        if (trial_node["dof_index"]) {
            trial.angle_sweep.dof_index = trial_node["dof_index"].as<int>();
            trial.angle_sweep.dof_type = "";  // Simple DOF
        } else if (trial_node["dof"]) {
            // Composite DOF (e.g., "abd_knee")
            trial.angle_sweep.dof_type = trial_node["dof"].as<std::string>();
            trial.angle_sweep.dof_index = 0;  // Not used for composite DOF
        } else {
            trial.angle_sweep.dof_index = 0;  // Default
            trial.angle_sweep.dof_type = "";
        }

        // Parse exam section for sweep parameters
        YAML::Node exam_node = trial_node["exam"];
        trial.angle_sweep.angle_min = exam_node["angle_min"].as<double>(-90.0) * M_PI / 180.0;
        trial.angle_sweep.angle_max = exam_node["angle_max"].as<double>(90.0) * M_PI / 180.0;
        trial.angle_sweep.num_steps = exam_node["num_steps"].as<int>(100);
        trial.angle_sweep.angle_step = exam_node["angle_step"].as<double>(1.0);
        trial.angle_sweep.shank_scale = exam_node["shank_scale"].as<double>(0.7);
        if (exam_node["alias"]) {
            trial.angle_sweep.alias = exam_node["alias"].as<std::string>();
        }

        // Parse clinical_data section for ROM display and clinical value lookup
        if (trial_node["clinical_data"]) {
            auto cd = trial_node["clinical_data"];
            if (cd["neg"]) {
                trial.angle_sweep.neg = cd["neg"].as<bool>(false);
            }
            if (cd["side"]) {
                trial.angle_sweep.cd_side = cd["side"].as<std::string>();
            }
            if (cd["joint"]) {
                trial.angle_sweep.cd_joint = cd["joint"].as<std::string>();
            }
            if (cd["field"]) {
                trial.angle_sweep.cd_field = cd["field"].as<std::string>();
            }
            // cd_neg: whether to negate the clinical value when comparing
            trial.angle_sweep.cd_neg = cd["neg"].as<bool>(false);
        }

        // Parse torque_cutoff (with backward compat for old "torque" key)
        trial.torque_cutoff = trial_node["torque_cutoff"].as<double>(
            trial_node["torque"].as<double>(15.0));

        // For composite DOF, num_steps is not used (sweep until IK fails)
        if (!trial.angle_sweep.dof_type.empty()) {
            trial.angle_sweep.num_steps = 0;
            LOG_INFO("Parsed ROM config (composite DOF): " << trial.name
                      << " (joint: " << trial.angle_sweep.joint_name
                      << ", dof: " << trial.angle_sweep.dof_type
                      << ", torque_cutoff: " << trial.torque_cutoff << " Nm)");
        } else {
            LOG_INFO("Parsed ROM config: " << trial.name
                      << " (joint: " << trial.angle_sweep.joint_name
                      << ", dof_index: " << trial.angle_sweep.dof_index
                      << ", torque_cutoff: " << trial.torque_cutoff << " Nm)");
        }
    } else if (trial_node["force"]) {
        // Parse force configuration (legacy format)
        trial.mode = TrialMode::FORCE_SWEEP;
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

        // Parse which joints to record (for data collection)
        if (trial_node["recording"] && trial_node["recording"]["joints"]) {
            trial.record_joints = trial_node["recording"]["joints"].as<std::vector<std::string>>();
        }

        LOG_INFO("Parsed force sweep trial: " << trial.name);
    } else {
        // Unknown format - log warning
        LOG_WARN("Unknown trial config format for: " << trial.name);
    }

    return trial;
}

void PhysicalExam::loadAndRunTrial(const std::string& trial_file_path) {
    if (!mExamSettingLoaded || !mCharacter) {
        LOG_ERROR("Cannot run trial: exam setting not loaded or character not available");
        return;
    }

    if (mTrialRunning) {
        LOG_WARN("Trial already running, please wait for completion");
        return;
    }

    try {
        // Load and parse trial YAML
        YAML::Node trial_node = YAML::LoadFile(trial_file_path);
        TrialConfig trial = parseTrialConfig(trial_node);

        // Create single-element trials vector and set current index
        mTrials.clear();
        mTrials.push_back(trial);
        mCurrentTrialIndex = 0;
        mTrialRunning = true;
        mCurrentForceStep = 0;
        mRecordedData.clear();
        
        // Initialize HDF5 if needed
        if (mExamOutputPath.empty()) {
            initExamHDF5();
        }
        
        // Run the trial
        runCurrentTrial();

        // Buffer the trial results for visualization
        if (!mAngleSweepData.empty() && trial.mode == TrialMode::ANGLE_SWEEP) {
            TrialDataBuffer buffer;
            buffer.trial_name = trial.name;
            buffer.trial_description = trial.description;
            buffer.alias = trial.angle_sweep.alias;
            // abd_knee: don't negate ROM angle for display (neg flag only affects cutoff direction)
            buffer.neg = (trial.angle_sweep.dof_type == "abd_knee") ? false : trial.angle_sweep.neg;
            buffer.timestamp = std::chrono::system_clock::now();
            buffer.angle_sweep_data = mAngleSweepData;
            buffer.std_angle_sweep_data = mStdAngleSweepData;
            buffer.tracked_muscles = mAngleSweepTrackedMuscles;
            buffer.std_tracked_muscles = mStdAngleSweepTrackedMuscles;
            buffer.config = trial.angle_sweep;
            buffer.torque_cutoff = trial.torque_cutoff;
            // Copy clinical data reference for ROM table rendering
            buffer.cd_side = trial.angle_sweep.cd_side;
            buffer.cd_joint = trial.angle_sweep.cd_joint;
            buffer.cd_field = trial.angle_sweep.cd_field;
            buffer.cd_neg = trial.angle_sweep.cd_neg;
            // Negate cutoff based on neg flag (neg:false → negative torque direction)
            double effective_cutoff = trial.angle_sweep.neg ? trial.torque_cutoff : -trial.torque_cutoff;
            // abd_knee has reversed torque direction - flip the cutoff sign
            if (trial.angle_sweep.dof_type == "abd_knee") {
                effective_cutoff = -effective_cutoff;
            }
            buffer.cutoff_angles = computeCutoffAngles(mAngleSweepData, effective_cutoff);
            buffer.rom_metrics = computeROMMetrics(mAngleSweepData);
            buffer.std_rom_metrics = computeROMMetrics(mStdAngleSweepData);
            buffer.base_pose = mCharacter->getSkeleton()->getPositions();

            // Capture normative pose: set skeleton to normative angle and capture
            // Strip side suffix ("/left" or "/right") to match mNormativeROM key format
            std::string normKey = buffer.alias;
            size_t lastSlash = normKey.rfind('/');
            if (lastSlash != std::string::npos) {
                normKey = normKey.substr(0, lastSlash);
            }
            LOG_INFO("[NormativePose] Searching for key: '" << normKey << "' (alias: '" << buffer.alias << "')");
            auto normIt = mNormativeROM.find(normKey);
            if (normIt != mNormativeROM.end()) {
                double normative_deg = normIt->second;
                double normative_rad = normative_deg * M_PI / 180.0;
                LOG_INFO("[NormativePose] Found normative: " << normative_deg << " deg, neg=" << trial.angle_sweep.neg);

                // Apply neg flag: if neg=true, negate the angle for internal representation
                if (trial.angle_sweep.neg) {
                    normative_rad = -normative_rad;
                    LOG_INFO("[NormativePose] After neg applied: " << (normative_rad * 180.0 / M_PI) << " deg");
                }

                auto skel = mCharacter->getSkeleton();
                auto joint = skel->getJoint(trial.angle_sweep.joint_name);
                if (joint) {
                    int dof_idx = joint->getIndexInSkeleton(trial.angle_sweep.dof_index);
                    Eigen::VectorXd pos = skel->getPositions();
                    double before_angle = pos[dof_idx];
                    pos[dof_idx] = normative_rad;
                    skel->setPositions(pos);
                    buffer.normative_pose = skel->getPositions();
                    LOG_INFO("[NormativePose] Joint=" << trial.angle_sweep.joint_name
                             << ", dof_idx=" << dof_idx
                             << ", before=" << (before_angle * 180.0 / M_PI) << " deg"
                             << ", after=" << (normative_rad * 180.0 / M_PI) << " deg");

                    // Restore base pose
                    skel->setPositions(buffer.base_pose);
                }
            } else {
                LOG_WARN("[NormativePose] Key '" << normKey << "' not found in mNormativeROM");
            }

            addTrialToBuffer(buffer);
            LOG_INFO("Trial '" << trial.name << "' buffered for visualization");
        }

        mTrialRunning = false;
        LOG_INFO("Trial completed. Results saved to: " << mExamOutputPath);

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load and run trial from " << trial_file_path << ": " << e.what());
        mTrialRunning = false;
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

    // Buffer the trial results for visualization
    const TrialConfig& trial = mTrials[mCurrentTrialIndex];
    if (!mAngleSweepData.empty() && trial.mode == TrialMode::ANGLE_SWEEP) {
        TrialDataBuffer buffer;
        buffer.trial_name = trial.name;
        buffer.trial_description = trial.description;
        buffer.alias = trial.angle_sweep.alias;
        buffer.timestamp = std::chrono::system_clock::now();
        buffer.angle_sweep_data = mAngleSweepData;
        buffer.std_angle_sweep_data = mStdAngleSweepData;
        buffer.tracked_muscles = mAngleSweepTrackedMuscles;
        buffer.std_tracked_muscles = mStdAngleSweepTrackedMuscles;
        buffer.config = trial.angle_sweep;
        // Copy clinical data reference for ROM table rendering
        buffer.cd_side = trial.angle_sweep.cd_side;
        buffer.cd_joint = trial.angle_sweep.cd_joint;
        buffer.cd_field = trial.angle_sweep.cd_field;
        buffer.cd_neg = trial.angle_sweep.cd_neg;
        buffer.rom_metrics = computeROMMetrics(mAngleSweepData);
        buffer.std_rom_metrics = computeROMMetrics(mStdAngleSweepData);
        buffer.base_pose = mCharacter->getSkeleton()->getPositions();

        // Capture normative pose: set skeleton to normative angle and capture
        // Strip side suffix ("/left" or "/right") to match mNormativeROM key format
        std::string normKey = buffer.alias;
        size_t lastSlash = normKey.rfind('/');
        if (lastSlash != std::string::npos) {
            normKey = normKey.substr(0, lastSlash);
        }
        LOG_INFO("[NormativePose] Searching for key: '" << normKey << "' (alias: '" << buffer.alias << "')");
        auto normIt = mNormativeROM.find(normKey);
        if (normIt != mNormativeROM.end()) {
            double normative_deg = normIt->second;
            double normative_rad = normative_deg * M_PI / 180.0;
            LOG_INFO("[NormativePose] Found normative: " << normative_deg << " deg, neg=" << trial.angle_sweep.neg);

            // Apply neg flag: if neg=true, negate the angle for internal representation
            if (trial.angle_sweep.neg) {
                normative_rad = -normative_rad;
                LOG_INFO("[NormativePose] After neg applied: " << (normative_rad * 180.0 / M_PI) << " deg");
            }

            auto skel = mCharacter->getSkeleton();
            auto joint = skel->getJoint(trial.angle_sweep.joint_name);
            if (joint) {
                int dof_idx = joint->getIndexInSkeleton(trial.angle_sweep.dof_index);
                Eigen::VectorXd pos = skel->getPositions();
                double before_angle = pos[dof_idx];
                pos[dof_idx] = normative_rad;
                skel->setPositions(pos);
                buffer.normative_pose = skel->getPositions();
                LOG_INFO("[NormativePose] Joint=" << trial.angle_sweep.joint_name
                         << ", dof_idx=" << dof_idx
                         << ", before=" << (before_angle * 180.0 / M_PI) << " deg"
                         << ", after=" << (normative_rad * 180.0 / M_PI) << " deg");

                // Restore base pose
                skel->setPositions(buffer.base_pose);
            }
        } else {
            LOG_WARN("[NormativePose] Key '" << normKey << "' not found in mNormativeROM");
        }

        addTrialToBuffer(buffer);
        LOG_INFO("Trial '" << trial.name << "' buffered for visualization");
    }

    mTrialRunning = false;
    LOG_INFO("Trial completed. Results saved to: " << mExamOutputPath);
}

void PhysicalExam::runCurrentTrial() {
    if (!mExamSettingLoaded || mCurrentTrialIndex < 0 ||
        mCurrentTrialIndex >= static_cast<int>(mTrials.size())) {
        LOG_ERROR("Invalid trial index");
        return;
    }

    const TrialConfig& trial = mTrials[mCurrentTrialIndex];

    // Dispatch based on trial mode
    if (trial.mode == TrialMode::ANGLE_SWEEP) {
        runAngleSweepTrial(trial);
        return;
    }

    // Force sweep mode (original logic)
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
        data.passive_force_total = 0.0;  // Deprecated: use getPassiveTorqueJoint() for angle sweep
        mRecordedData.push_back(data);
    }

    // Disabled: use manual export button instead
    // appendTrialToHDF5(trial);
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
        data.passive_force_total = 0.0;  // Deprecated - use per-joint torque instead
        mRecordedData.push_back(data);
    }

    // Save results (deprecated)
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

// ============================================================================
// ANGLE SWEEP TRIAL IMPLEMENTATION
// ============================================================================

void PhysicalExam::setupTrackedMusclesForAngleSweep(const std::string& joint_name) {
    mAngleSweepTrackedMuscles.clear();

    if (!mCharacter || mCharacter->getMuscles().empty()) return;

    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(joint_name);
    if (!joint) {
        LOG_ERROR("Joint not found for angle sweep tracking: " << joint_name);
        return;
    }

    // Find all muscles that cross this joint
    auto muscles = mCharacter->getMuscles();
    for (auto* muscle : muscles) {
        auto related_joints = muscle->GetRelatedJoints();
        for (auto* rj : related_joints) {
            if (rj == joint) {
                mAngleSweepTrackedMuscles.push_back(muscle->GetName());
                break;
            }
        }
    }

    // Setup standard character muscles
    if (mStdCharacter && !mStdCharacter->getMuscles().empty()) {
        mStdAngleSweepTrackedMuscles.clear();
        auto std_muscles = mStdCharacter->getMuscles();
        auto std_skel = mStdCharacter->getSkeleton();
        auto std_joint = std_skel->getJoint(joint_name);

        if (std_joint) {
            for (auto muscle : std_muscles) {
                auto related_joints = muscle->GetRelatedJoints();
                if (std::find(related_joints.begin(), related_joints.end(), std_joint)
                    != related_joints.end()) {
                    mStdAngleSweepTrackedMuscles.push_back(muscle->GetName());
                }
            }
        }
    }
}

void PhysicalExam::collectAngleSweepData(double angle, int joint_index, bool use_global_y) {
    AngleSweepDataPoint data;
    data.joint_angle = angle;

    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(joint_index);

    // Debug: verify joint name
    if (mVerboseTorque) {
        std::cout << "  [collectAngleSweepData] joint_index=" << joint_index
                  << " joint_name=" << (joint ? joint->getName() : "NULL") << std::endl;
    }

    // Use DOF-specific torque for simple sweeps (symmetric), global Y for composite DOFs
    data.passive_torque_total = use_global_y
        ? getPassiveTorqueJointGlobalY(mCharacter, joint)
        : getPassiveTorqueJointDof(mCharacter, joint, mSweepConfig.dof_index);

    // Debug: check for NaN
    if (std::isnan(data.passive_torque_total)) {
        LOG_WARN("NaN detected in passive_torque_total at angle=" << angle << " rad, use_global_y=" << use_global_y);
    }

    // Verbose torque debug output
    if (mVerboseTorque) {
        int skel_dof = joint->getIndexInSkeleton(mSweepConfig.dof_index);
        double jtp_sum = 0.0;
        for (auto& muscle : mCharacter->getMuscles()) {
            Eigen::VectorXd jtp = muscle->GetRelatedJtp();
            const auto& related_indices = muscle->related_dof_indices;
            for (size_t i = 0; i < related_indices.size(); ++i) {
                if (related_indices[i] == skel_dof) {
                    jtp_sum += jtp[i];
                    break;
                }
            }
        }
        std::cout << "[Sweep] angle=" << std::fixed << std::setprecision(1)
                  << angle * 180.0 / M_PI << "° use_global_y=" << use_global_y
                  << " passive_torque_total=" << std::setprecision(2) << data.passive_torque_total
                  << " JTP_sum=" << jtp_sum << std::endl;
    }

    // Compute stiffness (dtau/dtheta) using backward difference
    size_t N = mAngleSweepData.size();
    if (N >= 1) {
        double angle_prev = mAngleSweepData[N-1].joint_angle;
        double torque_prev = mAngleSweepData[N-1].passive_torque_total;
        if (std::abs(angle - angle_prev) > 1e-10) {
            data.passive_torque_stiffness = 
                (data.passive_torque_total - torque_prev) / (angle - angle_prev);
        } else {
            data.passive_torque_stiffness = 0.0;
        }
    } else {
        data.passive_torque_stiffness = 0.0;  // First point
    }

    // Collect per-muscle data (only muscles crossing the swept joint)
    for (const auto& muscle_name : mAngleSweepTrackedMuscles) {
        Muscle* muscle = mCharacter->getMuscleByName(muscle_name);
        if (!muscle) {
            LOG_WARN("Muscle not found during data collection: " << muscle_name);
            continue;
        }

        data.muscle_fp[muscle_name] = muscle->Getf_p();
        data.muscle_lm_norm[muscle_name] = muscle->lm_norm;

        // Per-DOF joint torques from muscle's Jacobian
        Eigen::VectorXd jtp = muscle->GetRelatedJtp();
        std::vector<double> jtp_vec(jtp.data(), jtp.data() + jtp.size());
        data.muscle_jtp[muscle_name] = jtp_vec;

        // Extract jtp at swept DOF specifically for debugging
        int skel_dof_idx = joint->getIndexInSkeleton(mSweepConfig.dof_index);
        double jtp_at_swept_dof = 0.0;
        const auto& related_indices = muscle->related_dof_indices;
        for (size_t j = 0; j < related_indices.size(); ++j) {
            if (related_indices[j] == skel_dof_idx) {
                jtp_at_swept_dof = jtp[j];
                break;
            }
        }
        data.muscle_jtp_dof[muscle_name] = jtp_at_swept_dof;
    }

    mAngleSweepData.push_back(data);

    // Copy second point's stiffness to first point (first point has no backward diff)
    if (mAngleSweepData.size() >= 2) {
        mAngleSweepData[0].passive_torque_stiffness = mAngleSweepData[1].passive_torque_stiffness;
    }

    // Collect standard character data if available
    if (mStdCharacter && !mStdCharacter->getMuscles().empty()) {
        AngleSweepDataPoint std_data;
        std_data.joint_angle = angle;
        
        // Note: std character is already in sync with main character via setCharacterPose
        auto std_skel = mStdCharacter->getSkeleton();
        auto main_skel = mCharacter->getSkeleton();
        auto main_joint = main_skel->getJoint(joint_index);
        auto std_joint = std_skel->getJoint(main_joint->getName());

        // Compute passive torque for std character using same method as main character
        std_data.passive_torque_total = use_global_y
            ? getPassiveTorqueJointGlobalY(mStdCharacter, std_joint)
            : getPassiveTorqueJointDof(mStdCharacter, std_joint, mSweepConfig.dof_index);
        
        // Compute stiffness for std
        size_t N_std = mStdAngleSweepData.size();
        if (N_std >= 1) {
            double angle_prev = mStdAngleSweepData[N_std-1].joint_angle;
            double torque_prev = mStdAngleSweepData[N_std-1].passive_torque_total;
            if (std::abs(angle - angle_prev) > 1e-10) {
                std_data.passive_torque_stiffness = 
                    (std_data.passive_torque_total - torque_prev) / (angle - angle_prev);
            } else {
                std_data.passive_torque_stiffness = 0.0;
            }
        } else {
            std_data.passive_torque_stiffness = 0.0;
        }
        
        // Collect std muscle data
        for (const auto& muscle_name : mStdAngleSweepTrackedMuscles) {
            Muscle* muscle = mStdCharacter->getMuscleByName(muscle_name);
            if (!muscle) continue;

            std_data.muscle_fp[muscle_name] = muscle->Getf_p();
            std_data.muscle_lm_norm[muscle_name] = muscle->lm_norm;

            Eigen::VectorXd jtp = muscle->GetRelatedJtp();
            std::vector<double> jtp_vec(jtp.data(), jtp.data() + jtp.size());
            std_data.muscle_jtp[muscle_name] = jtp_vec;

            // Extract jtp at swept DOF specifically for debugging
            int std_skel_dof_idx = std_joint->getIndexInSkeleton(mSweepConfig.dof_index);
            double jtp_at_swept_dof = 0.0;
            const auto& related_indices = muscle->related_dof_indices;
            for (size_t j = 0; j < related_indices.size(); ++j) {
                if (related_indices[j] == std_skel_dof_idx) {
                    jtp_at_swept_dof = jtp[j];
                    break;
                }
            }
            std_data.muscle_jtp_dof[muscle_name] = jtp_at_swept_dof;
        }

        mStdAngleSweepData.push_back(std_data);

        // Copy second point's stiffness to first point for std character
        if (mStdAngleSweepData.size() == 2) {
            mStdAngleSweepData[0].passive_torque_stiffness = mStdAngleSweepData[1].passive_torque_stiffness;
        }
    }
}

ROMMetrics PhysicalExam::computeROMMetrics(
    const std::vector<AngleSweepDataPoint>& data) const {

    ROMMetrics metrics;

    if (data.empty()) {
        return metrics;  // Return zeros
    }

    // Peak value tracking
    metrics.peak_stiffness = 0.0;
    metrics.peak_torque = 0.0;

    // Find the largest continuous angle range where values stay within thresholds
    double best_rom_start_rad = data.front().joint_angle;
    double best_rom_end_rad = data.front().joint_angle;
    double best_rom_size_rad = 0.0;

    double current_range_start_rad = data.front().joint_angle;
    bool in_valid_range = false;

    for (const auto& pt : data) {
        double abs_stiffness = std::abs(pt.passive_torque_stiffness);
        double abs_torque = std::abs(pt.passive_torque_total);

        // Track peak values
        if (abs_stiffness > metrics.peak_stiffness) {
            metrics.peak_stiffness = abs_stiffness;
            metrics.angle_at_peak_stiffness = pt.joint_angle * 180.0 / M_PI;
        }

        if (abs_torque > metrics.peak_torque) {
            metrics.peak_torque = abs_torque;
            metrics.angle_at_peak_torque = pt.joint_angle * 180.0 / M_PI;
        }

        // Check if current point is within threshold
        bool stiffness_ok = abs_stiffness <= mROMThresholds.max_stiffness;
        bool torque_ok = abs_torque <= mROMThresholds.max_torque;

        bool within_threshold = false;
        switch (mROMMetric) {
            case ROMMetric::STIFFNESS:
                within_threshold = stiffness_ok;
                break;
            case ROMMetric::TORQUE:
                within_threshold = torque_ok;
                break;
            case ROMMetric::EITHER:
                // For EITHER mode: point is valid only if BOTH metrics are ok
                within_threshold = stiffness_ok && torque_ok;
                break;
            case ROMMetric::BOTH:
                // For BOTH mode: point is valid if at least one metric is ok
                within_threshold = stiffness_ok || torque_ok;
                break;
        }

        if (within_threshold) {
            if (!in_valid_range) {
                // Start of new valid range
                current_range_start_rad = pt.joint_angle;
                in_valid_range = true;
            }
            // Continue valid range - check if this makes it the best so far
            double current_size_rad = pt.joint_angle - current_range_start_rad;
            if (current_size_rad > best_rom_size_rad) {
                best_rom_size_rad = current_size_rad;
                best_rom_start_rad = current_range_start_rad;
                best_rom_end_rad = pt.joint_angle;
            }
        } else {
            // Out of threshold - end current range
            in_valid_range = false;
        }
    }

    // Functional ROM = largest continuous range within thresholds
    metrics.rom_min_angle = best_rom_start_rad * 180.0 / M_PI;
    metrics.rom_max_angle = best_rom_end_rad * 180.0 / M_PI;
    metrics.rom_deg = metrics.rom_max_angle - metrics.rom_min_angle;

    return metrics;
}

std::vector<double> PhysicalExam::computeCutoffAngles(
    const std::vector<AngleSweepDataPoint>& data,
    double torque_cutoff) const {

    std::vector<double> crossings;
    if (data.size() < 2) {
        LOG_WARN("[computeCutoffAngles] Data size < 2: " << data.size());
        return crossings;
    }

    // Log torque range for debugging
    double min_tau = std::numeric_limits<double>::max();
    double max_tau = std::numeric_limits<double>::lowest();
    for (const auto& pt : data) {
        min_tau = std::min(min_tau, pt.passive_torque_total);
        max_tau = std::max(max_tau, pt.passive_torque_total);
    }
    LOG_INFO("[computeCutoffAngles] Torque range: [" << min_tau << ", " << max_tau
             << "] Nm, cutoff: " << torque_cutoff << " Nm, data points: " << data.size());

    for (size_t i = 1; i < data.size(); ++i) {
        double tau0 = data[i-1].passive_torque_total;
        double tau1 = data[i].passive_torque_total;
        double ang0 = data[i-1].joint_angle;
        double ang1 = data[i].joint_angle;

        // Check if cutoff is crossed between points (signed)
        // Crossing occurs if (tau0 - cutoff) and (tau1 - cutoff) have opposite signs
        double d0 = tau0 - torque_cutoff;
        double d1 = tau1 - torque_cutoff;

        if (d0 * d1 < 0) {  // Sign change = crossing
            // Linear interpolation: find t where tau = cutoff
            double t = d0 / (d0 - d1);
            double crossing_angle = ang0 + t * (ang1 - ang0);
            double crossing_deg = crossing_angle * 180.0 / M_PI;
            crossings.push_back(crossing_deg);
            LOG_INFO("[computeCutoffAngles] Found crossing at " << crossing_deg << "° (torque "
                     << tau0 << " -> " << tau1 << " Nm)");
        }
    }

    if (crossings.empty()) {
        LOG_WARN("[computeCutoffAngles] No crossings found - cutoff " << torque_cutoff
                 << " Nm not crossed by torque range [" << min_tau << ", " << max_tau << "] Nm");
    }

    return crossings;
}

std::vector<double> PhysicalExam::normalizeXAxis(
    const std::vector<double>& x_data_deg,
    double rom_min_deg, double rom_max_deg) const {

    if (mXAxisMode == XAxisMode::RAW_ANGLE || x_data_deg.empty()) {
        return x_data_deg;  // No transformation
    }

    // Normalize to 0-100% based on computed ROM range
    // Maps [rom_min, rom_max] → [0%, 100%]
    double range = rom_max_deg - rom_min_deg;

    std::vector<double> normalized;
    normalized.reserve(x_data_deg.size());

    if (range > 1e-6) {  // Avoid division by zero
        for (double angle : x_data_deg) {
            // Map angle to percentage of ROM range
            normalized.push_back((angle - rom_min_deg) / range * 100.0);
        }
    } else {
        normalized = x_data_deg;  // Fallback to raw
    }

    return normalized;
}

void PhysicalExam::runAngleSweepTrial(const TrialConfig& trial) {
    LOG_VERBOSE("Running angle sweep trial: " << trial.name);

    // Set current sweep name for plot titles
    mCurrentSweepName = trial.name;

    // 1. Apply initial pose
    applyPosePreset(trial.pose);

    // 2. Get target joint
    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(trial.angle_sweep.joint_name);
    if (!joint) {
        LOG_ERROR("Joint not found: " << trial.angle_sweep.joint_name);
        return;
    }

    // 3. Store joint index and DOF index for passive torque calculation
    mAngleSweepJointIdx = static_cast<int>(joint->getJointIndexInSkeleton());
    mSweepConfig.dof_index = trial.angle_sweep.dof_index;

    // 4. Identify muscles crossing this joint
    setupTrackedMusclesForAngleSweep(trial.angle_sweep.joint_name);

    // 5. Clear previous data (both main and standard character)
    mAngleSweepData.clear();
    mStdAngleSweepData.clear();

    // 6. Kinematic sweep loop
    if (trial.angle_sweep.dof_type == "abd_knee") {
        // Composite DOF: sweep until IK fails (geometry constraint violated)
        bool is_left = (trial.angle_sweep.joint_name.find("L") != std::string::npos);
        int hip_joint_idx = static_cast<int>(joint->getJointIndexInSkeleton());
        double angle_deg = trial.angle_sweep.angle_min * 180.0 / M_PI;  // Convert to degrees for IK

        // Get knee joint for setting knee angle
        std::string knee_name = is_left ? "TibiaL" : "TibiaR";
        auto knee_joint = skel->getJoint(knee_name);
        if (!knee_joint) {
            LOG_ERROR("Knee joint not found: " << knee_name);
            return;
        }
        int knee_dof_idx = static_cast<int>(knee_joint->getIndexInSkeleton(0));
        int hip_dof_start = static_cast<int>(joint->getIndexInSkeleton(0));

        LOG_INFO("Starting abd_knee sweep from " << angle_deg << "° with step "
                  << trial.angle_sweep.angle_step << "°");

        while (true) {
            // Compute IK pose for this abduction angle
            auto ik_result = ContractureOptimizer::computeAbdKneePose(
                skel, hip_joint_idx, angle_deg, is_left, trial.angle_sweep.shank_scale);

            // Stop if IK failed (geometry constraint violated: |x| > d)
            if (!ik_result.success) {
                LOG_INFO("abd_knee sweep stopped at " << angle_deg << "° (IK limit reached)");
                break;
            }

            // Apply hip axis-angle (3 DOFs)
            Eigen::VectorXd pos = skel->getPositions();
            pos.segment<3>(hip_dof_start) = ik_result.hip_positions;

            // Apply knee angle
            pos[knee_dof_idx] = ik_result.knee_angle;

            // Set positions for main character
            skel->setPositions(pos);

            // Compute IK independently for std character if exists
            if (mStdCharacter) {
                auto std_skel = mStdCharacter->getSkeleton();
                auto std_joint = std_skel->getJoint(trial.angle_sweep.joint_name);
                auto std_knee = std_skel->getJoint(knee_name);
                if (std_joint && std_knee) {
                    int std_hip_idx = static_cast<int>(std_skel->getIndexOf(std_joint));
                    int std_hip_start = static_cast<int>(std_joint->getIndexInSkeleton(0));
                    int std_knee_idx = static_cast<int>(std_knee->getIndexInSkeleton(0));

                    // Compute IK independently for std character
                    auto std_ik_result = ContractureOptimizer::computeAbdKneePose(
                        std_skel, std_hip_idx, angle_deg, is_left, trial.angle_sweep.shank_scale);

                    if (std_ik_result.success) {
                        Eigen::VectorXd std_pos = std_skel->getPositions();
                        std_pos.segment<3>(std_hip_start) = std_ik_result.hip_positions;
                        std_pos[std_knee_idx] = std_ik_result.knee_angle;
                        std_skel->setPositions(std_pos);
                    }
                }
            }

            // Update muscle geometry
            if (!mCharacter->getMuscles().empty()) {
                mCharacter->getMuscleTuple();
            }
            if (mStdCharacter && !mStdCharacter->getMuscles().empty()) {
                mStdCharacter->getMuscleTuple();
            }

            // Collect data point with global Y torque projection
            double angle_rad = angle_deg * M_PI / 180.0;
            collectAngleSweepData(angle_rad, mAngleSweepJointIdx, /*use_global_y=*/true);

            // Visualize mode: render 3D only and wait for N key to proceed, Q to exit
            if (mVisualizeSweep) {
                bool exitSweep = false;
                mSweepNextPressed = false;
                mSweepQuitPressed = false;
                // Disable ImGui input capture so camera controls work
                auto& io = ImGui::GetIO();
                io.WantCaptureMouse = false;
                io.WantCaptureKeyboard = false;
                while (true) {
                    glfwPollEvents();
                    // Check flags set by keyPress callback
                    if (glfwWindowShouldClose(mWindow)) { exitSweep = true; break; }
                    if (mSweepQuitPressed) { exitSweep = true; break; }
                    if (mSweepNextPressed) { mSweepNextPressed = false; break; }

                    // Render 3D content only (no ImGui to avoid frame conflicts)
                    GUI::InitGL();
                    GUI::InitLighting();
                    updateCamera();
                    setCamera();
                    if (mRenderGround) GUI::DrawGroundGrid(mGroundMode);
                    drawContent();

                    // Draw simple OpenGL text overlay
                    glMatrixMode(GL_PROJECTION);
                    glPushMatrix();
                    glLoadIdentity();
                    int w, h;
                    glfwGetFramebufferSize(mWindow, &w, &h);
                    glOrtho(0, w, h, 0, -1, 1);
                    glMatrixMode(GL_MODELVIEW);
                    glPushMatrix();
                    glLoadIdentity();
                    glDisable(GL_LIGHTING);
                    glDisable(GL_DEPTH_TEST);
                    glColor3f(1.0f, 1.0f, 0.0f);
                    glRasterPos2i(10, 20);
                    char buf[64];
                    snprintf(buf, sizeof(buf), "Sweep: %.1f deg  [N] Next  [Q] Quit", angle_deg);
                    for (char* c = buf; *c; ++c) {
                        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
                    }
                    glEnable(GL_DEPTH_TEST);
                    glEnable(GL_LIGHTING);
                    glPopMatrix();
                    glMatrixMode(GL_PROJECTION);
                    glPopMatrix();
                    glMatrixMode(GL_MODELVIEW);

                    glfwSwapBuffers(mWindow);
                }
                if (exitSweep) break;
            }

            angle_deg += trial.angle_sweep.angle_step;  // IK requires positive angle input
        }

        LOG_VERBOSE("Collected " << mAngleSweepData.size() << " abd_knee sweep data points");
    } else {
        // Simple DOF: existing for-loop logic with num_steps
        for (int step = 0; step <= trial.angle_sweep.num_steps; ++step) {
            // Calculate target angle
            double angle = trial.angle_sweep.angle_min +
                (trial.angle_sweep.angle_max - trial.angle_sweep.angle_min) *
                step / static_cast<double>(trial.angle_sweep.num_steps);

            // Set swept joint to target angle
            Eigen::VectorXd pos = joint->getPositions();
            pos[trial.angle_sweep.dof_index] = angle;
            setCharacterPose(joint->getName(), pos);

            // Update muscle geometry (kinematic only - no physics step)
            if (!mCharacter->getMuscles().empty()) {
                mCharacter->getMuscleTuple();
            }
            if (mStdCharacter && !mStdCharacter->getMuscles().empty()) {
                mStdCharacter->getMuscleTuple();
            }

            // Collect data point
            collectAngleSweepData(angle, mAngleSweepJointIdx);

            // Visualize mode: render 3D only and wait for N key to proceed, Q to exit
            if (mVisualizeSweep) {
                bool exitSweep = false;
                double angle_deg_display = angle * 180.0 / M_PI;
                mSweepNextPressed = false;
                mSweepQuitPressed = false;
                auto& io = ImGui::GetIO();
                io.WantCaptureMouse = false;
                io.WantCaptureKeyboard = false;
                while (true) {
                    glfwPollEvents();
                    if (glfwWindowShouldClose(mWindow)) { exitSweep = true; break; }
                    if (mSweepQuitPressed) { exitSweep = true; break; }
                    if (mSweepNextPressed) { mSweepNextPressed = false; break; }

                    GUI::InitGL();
                    GUI::InitLighting();
                    updateCamera();
                    setCamera();
                    if (mRenderGround) GUI::DrawGroundGrid(mGroundMode);
                    drawContent();

                    // Draw simple OpenGL text overlay
                    glMatrixMode(GL_PROJECTION);
                    glPushMatrix();
                    glLoadIdentity();
                    int w, h;
                    glfwGetFramebufferSize(mWindow, &w, &h);
                    glOrtho(0, w, h, 0, -1, 1);
                    glMatrixMode(GL_MODELVIEW);
                    glPushMatrix();
                    glLoadIdentity();
                    glDisable(GL_LIGHTING);
                    glDisable(GL_DEPTH_TEST);
                    glColor3f(1.0f, 1.0f, 0.0f);
                    glRasterPos2i(10, 20);
                    char buf[64];
                    snprintf(buf, sizeof(buf), "Sweep: %.1f deg  [N] Next  [Q] Quit", angle_deg_display);
                    for (char* c = buf; *c; ++c) {
                        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
                    }
                    glEnable(GL_DEPTH_TEST);
                    glEnable(GL_LIGHTING);
                    glPopMatrix();
                    glMatrixMode(GL_PROJECTION);
                    glPopMatrix();
                    glMatrixMode(GL_MODELVIEW);

                    glfwSwapBuffers(mWindow);
                }
                if (exitSweep) break;
            }
        }

        LOG_VERBOSE("Collected " << mAngleSweepData.size() << " angle sweep data points");
    }

    // 7. Disabled: use manual export button instead
    // appendTrialToHDF5(trial);
}

// ============================================================================
// HDF5 EXAM EXPORT FUNCTIONS
// ============================================================================

void PhysicalExam::setOutputDir(const std::string& output_dir) {
    mOutputDir = output_dir;
}

std::string PhysicalExam::extractPidFromPath(const std::string& path) const {
    // Extract patient ID from PID backend URI (e.g., "@pid:12964246/gait/pre/..." → "12964246")
    const std::string prefix = "@pid:";
    if (path.find(prefix) != 0) {
        return "";  // Not a PID path
    }

    // Extract PID (everything between "@pid:" and next "/")
    size_t start = prefix.size();
    size_t end = path.find('/', start);
    if (end == std::string::npos) {
        return path.substr(start);  // No slash found, rest is PID
    }
    return path.substr(start, end - start);
}

void PhysicalExam::initExamHDF5() {
    // Generate output path from exam config path
    // e.g., "data/config/angle_sweep_test.yaml" → "{output_dir}/angle_sweep_test.h5"
    std::filesystem::path configPath(mExamConfigPath);
    std::string baseName = configPath.stem().string();

    // Check if skeleton/muscle path uses PID backend
    std::string pid = extractPidFromPath(mSkeletonPath);
    if (pid.empty()) {
        pid = extractPidFromPath(mMusclePath);
    }

    // Use mOutputDir (default "./results" if not set)
    std::string outputDir = mOutputDir.empty() ? "./results" : mOutputDir;

    // Ensure output directory exists
    std::filesystem::create_directories(outputDir);

    // Build output filename with optional PID suffix
    if (!pid.empty()) {
        mExamOutputPath = outputDir + "/" + baseName + "_" + pid + ".h5";
    } else {
        mExamOutputPath = outputDir + "/" + baseName + ".h5";
    }

    // Create HDF5 file with exam metadata
    H5::H5File file(mExamOutputPath, H5F_ACC_TRUNC);

    // Write root-level attributes
    H5::DataSpace scalar(H5S_SCALAR);

    // Variable-length string type
    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);

    // exam_name
    H5::Attribute nameAttr = file.createAttribute("exam_name", strType, scalar);
    nameAttr.write(strType, mExamName);

    // exam_description
    H5::Attribute descAttr = file.createAttribute("exam_description", strType, scalar);
    descAttr.write(strType, mExamDescription);

    // skeleton_path, muscle_path
    H5::Attribute skelAttr = file.createAttribute("skeleton_path", strType, scalar);
    skelAttr.write(strType, mSkeletonPath);

    H5::Attribute muscleAttr = file.createAttribute("muscle_path", strType, scalar);
    muscleAttr.write(strType, mMusclePath);

    // timestamp (ISO 8601)
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::string timestamp = std::ctime(&t);
    if (!timestamp.empty() && timestamp.back() == '\n') {
        timestamp.pop_back();  // Remove newline
    }
    H5::Attribute timeAttr = file.createAttribute("timestamp", strType, scalar);
    timeAttr.write(strType, timestamp);

    // num_trials
    int numTrials = static_cast<int>(mTrials.size());
    H5::Attribute numAttr = file.createAttribute("num_trials", H5::PredType::NATIVE_INT, scalar);
    numAttr.write(H5::PredType::NATIVE_INT, &numTrials);

    file.close();
    LOG_INFO("Created exam HDF5: " << mExamOutputPath);
}

void PhysicalExam::writeAngleSweepDataForCharacter(
    H5::Group& group, 
    const TrialConfig& trial,
    const std::vector<AngleSweepDataPoint>& data,
    const std::vector<std::string>& tracked_muscles) {
    
    size_t N = data.size();
    size_t M = tracked_muscles.size();

    if (N == 0 || M == 0) {
        LOG_WARN("No angle sweep data to write");
        return;
    }

    // Prepare buffers
    std::vector<float> angles(N), passiveTorques(N), passiveTorqueStiffness(N);
    std::vector<float> muscleFp(N * M), muscleLmNorm(N * M), muscleJtpMag(N * M), muscleJtpDof(N * M);

    for (size_t i = 0; i < N; ++i) {
        const auto& d = data[i];
        angles[i] = static_cast<float>(d.joint_angle);
        passiveTorques[i] = static_cast<float>(d.passive_torque_total);
        passiveTorqueStiffness[i] = static_cast<float>(d.passive_torque_stiffness);

        for (size_t m = 0; m < M; ++m) {
            const std::string& name = tracked_muscles[m];
            muscleFp[i*M + m] = static_cast<float>(d.muscle_fp.at(name));
            muscleLmNorm[i*M + m] = static_cast<float>(d.muscle_lm_norm.at(name));

            double jtpMag = 0.0;
            if (d.muscle_jtp.count(name)) {
                for (double v : d.muscle_jtp.at(name)) jtpMag += v*v;
                jtpMag = std::sqrt(jtpMag);
            }
            muscleJtpMag[i*M + m] = static_cast<float>(jtpMag);

            // Per-muscle jtp at swept DOF only
            double jtpDof = d.muscle_jtp_dof.count(name) ? d.muscle_jtp_dof.at(name) : 0.0;
            muscleJtpDof[i*M + m] = static_cast<float>(jtpDof);
        }
    }

    // Write 1D datasets
    hsize_t dims1[1] = {N};
    H5::DataSpace space1(1, dims1);

    H5::DataSet anglesDs = group.createDataSet("joint_angles", H5::PredType::NATIVE_FLOAT, space1);
    anglesDs.write(angles.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet passiveDs = group.createDataSet("passive_torques", H5::PredType::NATIVE_FLOAT, space1);
    passiveDs.write(passiveTorques.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet stiffnessDs = group.createDataSet("passive_torque_stiffness", H5::PredType::NATIVE_FLOAT, space1);
    stiffnessDs.write(passiveTorqueStiffness.data(), H5::PredType::NATIVE_FLOAT);

    // Write 2D datasets (N steps × M muscles)
    hsize_t dims2[2] = {N, M};
    H5::DataSpace space2(2, dims2);

    H5::DataSet fpDs = group.createDataSet("muscle_fp", H5::PredType::NATIVE_FLOAT, space2);
    fpDs.write(muscleFp.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet lmDs = group.createDataSet("muscle_lm_norm", H5::PredType::NATIVE_FLOAT, space2);
    lmDs.write(muscleLmNorm.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet jtpDs = group.createDataSet("muscle_jtp_mag", H5::PredType::NATIVE_FLOAT, space2);
    jtpDs.write(muscleJtpMag.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet jtpDofDs = group.createDataSet("muscle_jtp_dof", H5::PredType::NATIVE_FLOAT, space2);
    jtpDofDs.write(muscleJtpDof.data(), H5::PredType::NATIVE_FLOAT);

    // Write muscle names
    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
    hsize_t dimsM[1] = {M};
    H5::DataSpace spaceM(1, dimsM);
    std::vector<const char*> cstrs(M);
    for (size_t m = 0; m < M; ++m) cstrs[m] = tracked_muscles[m].c_str();
    H5::DataSet namesDs = group.createDataSet("muscle_names", strType, spaceM);
    namesDs.write(cstrs.data(), strType);

    // Trial-specific attributes
    H5::DataSpace scalar(H5S_SCALAR);

    H5::Attribute jointAttr = group.createAttribute("joint_name", strType, scalar);
    jointAttr.write(strType, trial.angle_sweep.joint_name);

    int dofIdx = trial.angle_sweep.dof_index;
    H5::Attribute dofAttr = group.createAttribute("dof_index", H5::PredType::NATIVE_INT, scalar);
    dofAttr.write(H5::PredType::NATIVE_INT, &dofIdx);

    float angleMin = static_cast<float>(trial.angle_sweep.angle_min);
    float angleMax = static_cast<float>(trial.angle_sweep.angle_max);
    H5::Attribute minAttr = group.createAttribute("angle_min_rad", H5::PredType::NATIVE_FLOAT, scalar);
    minAttr.write(H5::PredType::NATIVE_FLOAT, &angleMin);
    H5::Attribute maxAttr = group.createAttribute("angle_max_rad", H5::PredType::NATIVE_FLOAT, scalar);
    maxAttr.write(H5::PredType::NATIVE_FLOAT, &angleMax);
}

void PhysicalExam::writeAngleSweepData(H5::Group& group, const TrialConfig& trial) {
    // Create /main subgroup
    H5::Group mainGroup = group.createGroup("main");
    writeAngleSweepDataForCharacter(mainGroup, trial, 
        mAngleSweepData, mAngleSweepTrackedMuscles);
    
    // Create /std subgroup if data exists
    if (!mStdAngleSweepData.empty() && mStdCharacter) {
        H5::Group stdGroup = group.createGroup("std");
        writeAngleSweepDataForCharacter(stdGroup, trial, 
            mStdAngleSweepData, mStdAngleSweepTrackedMuscles);
    }
}

void PhysicalExam::writeForceSweepData(H5::Group& group, const TrialConfig& trial) {
    size_t N = mRecordedData.size();

    if (N == 0) {
        LOG_WARN("No force sweep data to write");
        return;
    }

    // Prepare buffers
    std::vector<float> forceMags(N), passiveForces(N);
    for (size_t i = 0; i < N; ++i) {
        forceMags[i] = static_cast<float>(mRecordedData[i].force_magnitude);
        passiveForces[i] = static_cast<float>(mRecordedData[i].passive_force_total);
    }

    // Flatten joint angles into 2D array
    std::vector<std::string> jointNames;
    size_t totalDofs = 0;
    if (!mRecordedData.empty()) {
        for (const auto& [jn, angles] : mRecordedData[0].joint_angles) {
            jointNames.push_back(jn);
            totalDofs += angles.size();
        }
    }

    std::vector<float> jointAngles(N * totalDofs);
    for (size_t i = 0; i < N; ++i) {
        size_t offset = 0;
        for (const auto& [jn, angles] : mRecordedData[i].joint_angles) {
            for (int d = 0; d < angles.size(); ++d) {
                jointAngles[i * totalDofs + offset + d] = static_cast<float>(angles[d]);
            }
            offset += angles.size();
        }
    }

    // Write datasets
    hsize_t dims1[1] = {N};
    H5::DataSpace space1(1, dims1);

    H5::DataSet forceDs = group.createDataSet("force_magnitudes", H5::PredType::NATIVE_FLOAT, space1);
    forceDs.write(forceMags.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet passiveDs = group.createDataSet("passive_forces", H5::PredType::NATIVE_FLOAT, space1);
    passiveDs.write(passiveForces.data(), H5::PredType::NATIVE_FLOAT);

    if (totalDofs > 0) {
        hsize_t dims2[2] = {N, totalDofs};
        H5::DataSpace space2(2, dims2);
        H5::DataSet angleDs = group.createDataSet("joint_angles", H5::PredType::NATIVE_FLOAT, space2);
        angleDs.write(jointAngles.data(), H5::PredType::NATIVE_FLOAT);
    }

    // Write joint names
    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
    if (!jointNames.empty()) {
        hsize_t dimsJ[1] = {jointNames.size()};
        H5::DataSpace spaceJ(1, dimsJ);
        std::vector<const char*> cstrs(jointNames.size());
        for (size_t j = 0; j < jointNames.size(); ++j) cstrs[j] = jointNames[j].c_str();
        H5::DataSet namesDs = group.createDataSet("joint_names", strType, spaceJ);
        namesDs.write(cstrs.data(), strType);
    }

    // Trial-specific attributes
    H5::DataSpace scalar(H5S_SCALAR);

    H5::Attribute bodyAttr = group.createAttribute("force_body_node", strType, scalar);
    bodyAttr.write(strType, trial.force_body_node);

    float forceMin = static_cast<float>(trial.force_min);
    float forceMax = static_cast<float>(trial.force_max);
    H5::Attribute minAttr = group.createAttribute("force_min", H5::PredType::NATIVE_FLOAT, scalar);
    minAttr.write(H5::PredType::NATIVE_FLOAT, &forceMin);
    H5::Attribute maxAttr = group.createAttribute("force_max", H5::PredType::NATIVE_FLOAT, scalar);
    maxAttr.write(H5::PredType::NATIVE_FLOAT, &forceMax);
}

void PhysicalExam::appendTrialToHDF5(const TrialConfig& trial) {
    if (mExamOutputPath.empty()) {
        LOG_ERROR("No exam HDF5 file initialized");
        return;
    }

    try {
        // Open existing HDF5 file in read-write mode
        H5::H5File file(mExamOutputPath, H5F_ACC_RDWR);

        // Create trial group
        H5::Group trialGroup = file.createGroup("/" + trial.name);

        // Write common trial attributes
        H5::DataSpace scalar(H5S_SCALAR);
        H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);

        std::string modeStr = (trial.mode == TrialMode::ANGLE_SWEEP) ? "angle_sweep" : "force_sweep";
        H5::Attribute modeAttr = trialGroup.createAttribute("trial_mode", strType, scalar);
        modeAttr.write(strType, modeStr);

        H5::Attribute descAttr = trialGroup.createAttribute("description", strType, scalar);
        descAttr.write(strType, trial.description);

        // Dispatch to mode-specific writer
        if (trial.mode == TrialMode::ANGLE_SWEEP) {
            writeAngleSweepData(trialGroup, trial);
        } else {
            writeForceSweepData(trialGroup, trial);
        }

        file.close();
        LOG_INFO("Appended trial '" << trial.name << "' to HDF5: " << mExamOutputPath);

    } catch (const H5::Exception& e) {
        LOG_ERROR("HDF5 error appending trial: " << e.getCDetailMsg());
    }
}

void PhysicalExam::exportTrialBuffersToHDF5() {
    if (mTrialBuffers.empty()) {
        LOG_WARN("No trial buffers to export");
        return;
    }

    // Generate output path with timestamp
    std::string outputDir = mOutputDir.empty() ? "./results" : mOutputDir;
    std::filesystem::create_directories(outputDir);

    auto now = std::chrono::system_clock::now();
    auto time_t_val = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time_t_val);
    char timeBuf[32];
    std::strftime(timeBuf, sizeof(timeBuf), "%Y%m%d_%H%M%S", tm);

    std::string outputPath = outputDir + "/trial_buffers_" + timeBuf + ".h5";

    try {
        // Create HDF5 file
        H5::H5File file(outputPath, H5F_ACC_TRUNC);
        H5::DataSpace scalar(H5S_SCALAR);
        H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);

        // Write root-level attributes
        std::string timestamp(timeBuf);
        H5::Attribute timeAttr = file.createAttribute("timestamp", strType, scalar);
        timeAttr.write(strType, timestamp);

        int numTrials = static_cast<int>(mTrialBuffers.size());
        H5::Attribute numAttr = file.createAttribute("num_trials", H5::PredType::NATIVE_INT, scalar);
        numAttr.write(H5::PredType::NATIVE_INT, &numTrials);

        // Export each buffer
        for (const auto& buffer : mTrialBuffers) {
            // Create trial group
            H5::Group trialGroup = file.createGroup("/" + buffer.trial_name);

            // Write trial attributes
            H5::Attribute modeAttr = trialGroup.createAttribute("trial_mode", strType, scalar);
            modeAttr.write(strType, std::string("angle_sweep"));

            H5::Attribute descAttr = trialGroup.createAttribute("description", strType, scalar);
            descAttr.write(strType, buffer.trial_description);

            // Create TrialConfig from buffer.config for writeAngleSweepData
            TrialConfig trial;
            trial.name = buffer.trial_name;
            trial.description = buffer.trial_description;
            trial.mode = TrialMode::ANGLE_SWEEP;
            trial.angle_sweep = buffer.config;

            // Write main character data
            if (!buffer.angle_sweep_data.empty()) {
                H5::Group mainGroup = trialGroup.createGroup("main_character");
                writeAngleSweepDataForCharacter(mainGroup, trial,
                    buffer.angle_sweep_data, buffer.tracked_muscles);
            }

            // Write standard character data
            if (!buffer.std_angle_sweep_data.empty()) {
                H5::Group stdGroup = trialGroup.createGroup("std_character");
                writeAngleSweepDataForCharacter(stdGroup, trial,
                    buffer.std_angle_sweep_data, buffer.std_tracked_muscles);
            }
        }

        file.close();
        LOG_INFO("Exported " << mTrialBuffers.size() << " trial buffers to: " << outputPath);

    } catch (const H5::Exception& e) {
        LOG_ERROR("HDF5 error exporting buffers: " << e.getCDetailMsg());
    }
}

void PhysicalExam::runAllTrials() {
    if (!mExamSettingLoaded || mTrials.empty()) {
        LOG_ERROR("No exam setting loaded or no trials available");
        return;
    }

    // Initialize HDF5 output if needed
    if (mExamOutputPath.empty()) {
        initExamHDF5();
    }

    LOG_INFO("Running all " << mTrials.size() << " trials...");

    for (size_t i = 0; i < mTrials.size(); ++i) {
        mCurrentTrialIndex = static_cast<int>(i);
        LOG_INFO("Trial " << (i+1) << "/" << mTrials.size() << ": " << mTrials[i].name);

        mTrialRunning = true;
        mCurrentForceStep = 0;
        mRecordedData.clear();

        runCurrentTrial();

        // Buffer the trial results
        const TrialConfig& trial = mTrials[mCurrentTrialIndex];
        if (!mAngleSweepData.empty() && trial.mode == TrialMode::ANGLE_SWEEP) {
            TrialDataBuffer buffer;
            buffer.trial_name = trial.name;
            buffer.trial_description = trial.description;
            buffer.alias = trial.angle_sweep.alias;
            // abd_knee: don't negate ROM angle for display (neg flag only affects cutoff direction)
            buffer.neg = (trial.angle_sweep.dof_type == "abd_knee") ? false : trial.angle_sweep.neg;
            buffer.timestamp = std::chrono::system_clock::now();
            buffer.angle_sweep_data = mAngleSweepData;
            buffer.std_angle_sweep_data = mStdAngleSweepData;
            buffer.tracked_muscles = mAngleSweepTrackedMuscles;
            buffer.std_tracked_muscles = mStdAngleSweepTrackedMuscles;
            buffer.config = trial.angle_sweep;
            buffer.torque_cutoff = trial.torque_cutoff;
            // Copy clinical data reference for ROM table rendering
            buffer.cd_side = trial.angle_sweep.cd_side;
            buffer.cd_joint = trial.angle_sweep.cd_joint;
            buffer.cd_field = trial.angle_sweep.cd_field;
            buffer.cd_neg = trial.angle_sweep.cd_neg;
            // Negate cutoff based on neg flag (neg:false → negative torque direction)
            double effective_cutoff = trial.angle_sweep.neg ? trial.torque_cutoff : -trial.torque_cutoff;
            // abd_knee has reversed torque direction - flip the cutoff sign
            if (trial.angle_sweep.dof_type == "abd_knee") {
                effective_cutoff = -effective_cutoff;
            }
            buffer.cutoff_angles = computeCutoffAngles(mAngleSweepData, effective_cutoff);
            buffer.rom_metrics = computeROMMetrics(mAngleSweepData);
            if (!mStdAngleSweepData.empty()) {
                buffer.std_rom_metrics = computeROMMetrics(mStdAngleSweepData);
            }
            buffer.base_pose = mCharacter->getSkeleton()->getPositions();

            // Capture normative pose: set skeleton to normative angle and capture
            // Strip side suffix ("/left" or "/right") to match mNormativeROM key format
            std::string normKey = buffer.alias;
            size_t lastSlash = normKey.rfind('/');
            if (lastSlash != std::string::npos) {
                normKey = normKey.substr(0, lastSlash);
            }
            LOG_INFO("[NormativePose] Searching for key: '" << normKey << "' (alias: '" << buffer.alias << "')");
            auto normIt = mNormativeROM.find(normKey);
            if (normIt != mNormativeROM.end()) {
                double normative_deg = normIt->second;
                double normative_rad = normative_deg * M_PI / 180.0;
                LOG_INFO("[NormativePose] Found normative: " << normative_deg << " deg, neg=" << trial.angle_sweep.neg);

                // Apply neg flag: if neg=true, negate the angle for internal representation
                if (trial.angle_sweep.neg) {
                    normative_rad = -normative_rad;
                    LOG_INFO("[NormativePose] After neg applied: " << (normative_rad * 180.0 / M_PI) << " deg");
                }

                auto skel = mCharacter->getSkeleton();
                auto joint = skel->getJoint(trial.angle_sweep.joint_name);
                if (joint) {
                    int dof_idx = joint->getIndexInSkeleton(trial.angle_sweep.dof_index);
                    Eigen::VectorXd pos = skel->getPositions();
                    double before_angle = pos[dof_idx];
                    pos[dof_idx] = normative_rad;
                    skel->setPositions(pos);
                    buffer.normative_pose = skel->getPositions();
                    LOG_INFO("[NormativePose] Joint=" << trial.angle_sweep.joint_name
                             << ", dof_idx=" << dof_idx
                             << ", before=" << (before_angle * 180.0 / M_PI) << " deg"
                             << ", after=" << (normative_rad * 180.0 / M_PI) << " deg");

                    // Restore base pose
                    skel->setPositions(buffer.base_pose);
                }
            } else {
                LOG_WARN("[NormativePose] Key '" << normKey << "' not found in mNormativeROM");
            }

            addTrialToBuffer(buffer);
        }

        mTrialRunning = false;
    }

    // Export buffers to HDF5
    if (!mTrialBuffers.empty()) {
        exportTrialBuffersToHDF5();
    }

    LOG_INFO("All trials complete. Results saved to: " << mExamOutputPath);
}

int PhysicalExam::runTrialsCLI(const std::vector<std::string>& trial_paths,
                                bool verbose,
                                double torque_threshold,
                                double length_threshold,
                                const std::string& sort_by) {
    if (!mCharacter || mCharacter->getMuscles().empty()) {
        LOG_ERROR("No character or muscles loaded");
        return 1;
    }

    // Structure to hold per-muscle data at a pose
    struct MusclePoseData {
        double lm_norm;      // Normalized muscle length
        double jtp_dof;      // Joint torque at target DOF
    };

    // Store trial names and data
    std::vector<std::string> trial_names;
    std::vector<std::map<std::string, MusclePoseData>> zero_data_list;
    std::vector<std::map<std::string, MusclePoseData>> norm_data_list;
    std::vector<std::vector<std::string>> trial_muscles;  // Muscles crossing each trial's joint

    // Process each trial
    for (const auto& trial_path : trial_paths) {
        std::string resolved_path = rm::resolve(trial_path);
        LOG_INFO("Processing trial: " << resolved_path);

        YAML::Node trial_node;
        try {
            trial_node = YAML::LoadFile(resolved_path);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load trial config: " << resolved_path << " - " << e.what());
            continue;
        }

        // Parse trial config
        TrialConfig trial = parseTrialConfig(trial_node);
        trial_names.push_back(trial.name);

        // Get normative angle from exam section
        double normative_deg = 0.0;
        if (trial_node["exam"] && trial_node["exam"]["normative"]) {
            normative_deg = trial_node["exam"]["normative"].as<double>();
        } else {
            LOG_WARN("No normative angle found in trial config: " << trial.name);
        }

        // Apply neg flag: if neg=true, the normative angle is in the negative direction
        // (e.g., extRot has normative=41.5 with neg=true, meaning actual angle is -41.5)
        if (trial.angle_sweep.neg) {
            normative_deg = -normative_deg;
        }
        double normative_rad = normative_deg * M_PI / 180.0;

        // Get joint and DOF info
        std::string joint_name = trial.angle_sweep.joint_name;
        int dof_index = trial.angle_sweep.dof_index;

        auto skel = mCharacter->getSkeleton();
        auto joint = skel->getJoint(joint_name);
        if (!joint) {
            LOG_ERROR("Joint not found: " << joint_name);
            continue;
        }
        int skel_dof_idx = joint->getIndexInSkeleton(dof_index);

        // Find muscles crossing this joint
        std::vector<std::string> tracked_muscles;
        for (auto* muscle : mCharacter->getMuscles()) {
            auto related_joints = muscle->GetRelatedJoints();
            for (auto* rj : related_joints) {
                if (rj == joint) {
                    tracked_muscles.push_back(muscle->GetName());
                    break;
                }
            }
        }
        trial_muscles.push_back(tracked_muscles);

        // Apply base pose from trial config
        for (const auto& [jname, angles] : trial.pose) {
            setCharacterPose(jname, angles);
        }

        // Lambda to collect muscle data at current pose
        auto collectMuscleData = [&]() -> std::map<std::string, MusclePoseData> {
            // Update muscle geometry for current skeleton state
            // NOTE: Don't call SetMuscle() which recalculates lmt_ref
            for (auto* m : mCharacter->getMuscles()) {
                m->UpdateGeometry();
            }

            std::map<std::string, MusclePoseData> data;
            for (const auto& muscle_name : tracked_muscles) {
                Muscle* muscle = mCharacter->getMuscleByName(muscle_name);
                if (!muscle) continue;

                MusclePoseData md;
                md.lm_norm = muscle->GetLmNorm();

                // Get jtp at the swept DOF
                Eigen::VectorXd jtp = muscle->GetRelatedJtp();
                const auto& related_indices = muscle->related_dof_indices;
                md.jtp_dof = 0.0;
                for (size_t j = 0; j < related_indices.size(); ++j) {
                    if (related_indices[j] == skel_dof_idx) {
                        md.jtp_dof = jtp[j];
                        break;
                    }
                }
                data[muscle_name] = md;
            }
            return data;
        };

        // Set joint DOF to 0 (zero angle)
        Eigen::VectorXd joint_pos = joint->getPositions();
        Eigen::VectorXd zero_pos = joint_pos;
        zero_pos[dof_index] = 0.0;
        joint->setPositions(zero_pos);

        // Collect data at zero angle
        auto zero_data = collectMuscleData();
        zero_data_list.push_back(zero_data);

        // Set joint DOF to normative angle
        Eigen::VectorXd norm_pos = joint_pos;
        norm_pos[dof_index] = normative_rad;
        joint->setPositions(norm_pos);

        // Collect data at normative angle
        auto norm_data = collectMuscleData();
        norm_data_list.push_back(norm_data);

        LOG_INFO("Trial " << trial.name << ": joint=" << joint_name
                  << ", dof=" << dof_index << ", normative=" << normative_deg << "°"
                  << ", muscles=" << tracked_muscles.size());
    }

    if (trial_names.empty()) {
        LOG_ERROR("No valid trials processed");
        return 1;
    }

    // Collect all muscle names across trials
    std::set<std::string> all_muscles_set;
    for (const auto& muscles : trial_muscles) {
        all_muscles_set.insert(muscles.begin(), muscles.end());
    }
    std::vector<std::string> all_muscles(all_muscles_set.begin(), all_muscles_set.end());

    // Structure to hold a muscle row for sorting
    struct MuscleRow {
        std::string name;
        std::vector<std::string> torque_cols;
        std::vector<std::string> length_cols;
        double max_torque_diff = 0.0;
        double max_length_diff = 0.0;
        bool has_significant_change = false;
    };

    // Build all muscle rows
    std::vector<MuscleRow> rows;
    for (const auto& muscle_name : all_muscles) {
        MuscleRow row;
        row.name = muscle_name;

        for (size_t t = 0; t < trial_names.size(); ++t) {
            auto& zero_data = zero_data_list[t];
            auto& norm_data = norm_data_list[t];
            auto& muscles = trial_muscles[t];

            // Check if this muscle is tracked for this trial
            bool in_trial = std::find(muscles.begin(), muscles.end(), muscle_name) != muscles.end();

            if (in_trial && zero_data.count(muscle_name) && norm_data.count(muscle_name)) {
                double jtp_zero = zero_data[muscle_name].jtp_dof;
                double jtp_norm = norm_data[muscle_name].jtp_dof;
                double lm_zero = zero_data[muscle_name].lm_norm;
                double lm_norm = norm_data[muscle_name].lm_norm;

                double delta_jtp = std::abs(jtp_norm - jtp_zero);
                double delta_lm = std::abs(lm_norm - lm_zero);

                // Track max diffs across trials
                row.max_torque_diff = std::max(row.max_torque_diff, delta_jtp);
                row.max_length_diff = std::max(row.max_length_diff, delta_lm);

                // Format torque column
                std::ostringstream torque_ss;
                if (delta_jtp > torque_threshold) {
                    row.has_significant_change = true;
                    torque_ss << std::fixed << std::setprecision(2)
                              << jtp_zero << " -> " << jtp_norm;
                } else {
                    torque_ss << "-";
                }
                row.torque_cols.push_back(torque_ss.str());

                // Format length column
                std::ostringstream length_ss;
                if (delta_lm > length_threshold) {
                    row.has_significant_change = true;
                    length_ss << std::fixed << std::setprecision(3)
                              << lm_zero << " -> " << lm_norm;
                } else {
                    length_ss << "-";
                }
                row.length_cols.push_back(length_ss.str());
            } else {
                row.torque_cols.push_back("-");
                row.length_cols.push_back("-");
            }
        }

        rows.push_back(row);
    }

    // Sort rows based on sort_by option
    if (sort_by == "torque") {
        std::sort(rows.begin(), rows.end(), [](const MuscleRow& a, const MuscleRow& b) {
            return a.max_torque_diff > b.max_torque_diff;  // Descending
        });
    } else if (sort_by == "length") {
        std::sort(rows.begin(), rows.end(), [](const MuscleRow& a, const MuscleRow& b) {
            return a.max_length_diff > b.max_length_diff;  // Descending
        });
    } else {
        // Default: sort alphabetically by name
        std::sort(rows.begin(), rows.end(), [](const MuscleRow& a, const MuscleRow& b) {
            return a.name < b.name;
        });
    }

    // Print table header
    std::cout << "\n";
    std::cout << std::left << std::setw(28) << "muscle_name";
    for (const auto& name : trial_names) {
        // Shorten trial name if too long
        std::string short_name = name.length() > 15 ? name.substr(0, 12) + "..." : name;
        std::cout << std::setw(22) << (short_name + " (torque)");
        std::cout << std::setw(22) << (short_name + " (lm_norm)");
    }
    std::cout << "\n";

    // Print separator
    int total_width = 28 + trial_names.size() * 44;
    std::cout << std::string(total_width, '-') << "\n";

    // Print each muscle row
    int printed_count = 0;
    for (const auto& row : rows) {
        // Skip muscle if no significant change (unless verbose)
        if (!verbose && !row.has_significant_change) {
            continue;
        }

        // Print muscle row
        std::cout << std::left << std::setw(28) << row.name;
        for (size_t t = 0; t < trial_names.size(); ++t) {
            std::cout << std::setw(22) << row.torque_cols[t];
            std::cout << std::setw(22) << row.length_cols[t];
        }
        std::cout << "\n";
        printed_count++;
    }

    std::cout << "\n";
    std::cout << "Printed " << printed_count << "/" << all_muscles.size() << " muscles";
    if (!verbose) {
        std::cout << " (use -v to show all)";
    }
    if (!sort_by.empty()) {
        std::cout << " (sorted by " << sort_by << " diff, descending)";
    }
    std::cout << "\n";
    std::cout << "Thresholds: torque=" << torque_threshold << " Nm, length=" << length_threshold << "\n";

    return 0;
}

// setCamera(), render(), mainLoop() removed - handled by ViewerAppBase::startLoop()

void PhysicalExam::drawLeftPanel() {
    // Left panel - matches GLFWApp layout
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::Begin("Controls##1", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    if (ImGui::BeginTabBar("LeftPanelTabs")) {
        if (ImGui::BeginTabItem("Patient")) {
            ImGui::BeginChild("PatientScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
            drawClinicalDataSection();
            drawCharacterLoadSection();
            drawPosePresetsSection();
            drawJointControlSection();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Exam")) {
            ImGui::BeginChild("ExamScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
            drawForceApplicationSection();
            drawJointAngleSweepSection();
            drawTrialManagementSection();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Surgery")) {
            ImGui::BeginChild("SurgeryScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
            // Use SurgeryPanel for surgery operations (replaces drawSurgeryTabContent)
            if (mSurgeryPanel) mSurgeryPanel->drawSurgeryContent();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Render")) {
            ImGui::BeginChild("ViewScroll", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
            drawRenderOptionsSection();
            drawPrintInfoSection();
            drawRecordingSection();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void PhysicalExam::drawRightPanel() {
    // Right panel - matches GLFWApp layout
    ImGui::SetNextWindowSize(ImVec2(mPlotPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::Begin("Visualization", nullptr,
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos(ImVec2(mWidth - ImGui::GetWindowSize().x, 0),
                        ImGuiCond_Always);

    // Scrollable child area for plots
    ImGui::BeginChild("ScrollArea", ImVec2(0, 0), false,
        ImGuiWindowFlags_AlwaysVerticalScrollbar);

    if (ImGui::BeginTabBar("RightPanelTabs")) {
        if (ImGui::BeginTabItem("Basic")) {
            drawBasicTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Sweep")) {
            drawSweepTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Etc")) {
            drawEtcTabContent();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::EndChild();
    ImGui::End();
}

void PhysicalExam::drawBasicTabContent() {
    // Loaded Files Section
    if (collapsingHeaderWithControls("Loaded Files")) {
        ImGui::Indent();

        if (!mSkeletonPath.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Skeleton:");
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::Text("%s", mSkeletonPath.c_str());
        } else {
            ImGui::TextDisabled("Skeleton: Not loaded");
        }

        ImGui::Spacing();

        if (!mMusclePath.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Muscle:");
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::Text("%s", mMusclePath.c_str());
        } else {
            ImGui::TextDisabled("Muscle: Not loaded");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Reference Character:");

        if (!mStdSkeletonPath.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.6f, 0.8f, 1.0f), "Skeleton:");
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::Text("%s", mStdSkeletonPath.c_str());
        } else {
            ImGui::TextDisabled("Skeleton: Not loaded");
        }

        if (!mStdMusclePath.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Muscle:");
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::Text("%s", mStdMusclePath.c_str());
        } else {
            ImGui::TextDisabled("Muscle: Not loaded");
        }

        ImGui::Unindent();
    }
    ImGui::Spacing();

    drawCurrentStateSection();
    drawMuscleInfoSection();
    drawCameraStatusSection();
}

void PhysicalExam::drawSweepTabContent() {
    // ========================================================================
    // Trial Buffer Selection Section
    // ========================================================================
    if (mTrialBuffers.empty()) {
        ImGui::TextDisabled("No trial data buffered");
        ImGui::TextWrapped("Run trials from the Trial Management section to generate data");
        return;
    }

    if (collapsingHeaderWithControls("Trial Buffer Selection")) {
        ImGui::Indent();

        // Filter input
        static char bufferFilterText[256] = "";
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##BufferFilter", "Filter buffers...", bufferFilterText, sizeof(bufferFilterText));

        // Build filtered items for ListBox
        std::vector<std::string> buffer_labels;
        std::vector<int> filteredIndices;  // Map filtered index -> original index
        std::string filterLower = bufferFilterText;
        std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

        for (size_t i = 0; i < mTrialBuffers.size(); ++i) {
            const auto& buf = mTrialBuffers[i];

            // Check filter match (case-insensitive)
            if (!filterLower.empty()) {
                std::string nameLower = buf.trial_name;
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                if (nameLower.find(filterLower) == std::string::npos) {
                    continue;  // Skip non-matching items
                }
            }

            // Format: "[index] trial_name (ROM angle°)"
            std::ostringstream oss;
            oss << "[" << i << "] " << buf.trial_name << " (";
            if (!buf.cutoff_angles.empty()) {
                oss << std::fixed << std::setprecision(1) << buf.cutoff_angles[0] << "°";
            } else {
                oss << "N/A";
            }
            oss << ")";
            buffer_labels.push_back(oss.str());
            filteredIndices.push_back(static_cast<int>(i));
        }

        // Convert to const char* array for ImGui
        std::vector<const char*> items;
        for (const auto& label : buffer_labels) {
            items.push_back(label.c_str());
        }

        // Find current selection in filtered list
        int filteredSelection = -1;
        for (size_t i = 0; i < filteredIndices.size(); ++i) {
            if (filteredIndices[i] == mSelectedBufferIndex) {
                filteredSelection = static_cast<int>(i);
                break;
            }
        }

        // ListBox for buffer selection
        ImGui::Text("Buffered Trials (%zu / %zu):", filteredIndices.size(), mTrialBuffers.size());
        int prev_filtered_selection = filteredSelection;
        if (ImGui::ListBox("##TrialBuffers", &filteredSelection,
                           items.data(), static_cast<int>(items.size()),
                           std::min(7, static_cast<int>(items.size())))) {
            // Selection changed - map back to original index
            if (filteredSelection != prev_filtered_selection && filteredSelection >= 0 &&
                filteredSelection < static_cast<int>(filteredIndices.size())) {
                int newOriginalIndex = filteredIndices[filteredSelection];
                if (newOriginalIndex != mSelectedBufferIndex) {
                    mSelectedBufferIndex = newOriginalIndex;
                    loadBufferForVisualization(mSelectedBufferIndex);
                }
            }
        }

        // Show selected buffer info
        if (mSelectedBufferIndex >= 0 && mSelectedBufferIndex < static_cast<int>(mTrialBuffers.size())) {
            const auto& selected = mTrialBuffers[mSelectedBufferIndex];
            ImGui::Separator();

            // Collapsible tree node for buffer details
            std::string nodeLabel = "Selected: " + selected.trial_name;
            if (ImGui::TreeNodeEx(nodeLabel.c_str())) {
                if (!selected.trial_description.empty()) {
                    ImGui::TextWrapped("Description: %s", selected.trial_description.c_str());
                }
                ImGui::Text("Data Points: %zu (main) / %zu (std)",
                           selected.angle_sweep_data.size(),
                           selected.std_angle_sweep_data.size());
                ImGui::Text("Muscles: %zu (main) / %zu (std)",
                           selected.tracked_muscles.size(),
                           selected.std_tracked_muscles.size());
                ImGui::Text("Torque Cutoff: %.1f Nm", selected.torque_cutoff);

                // Timestamp formatting
                auto time_t = std::chrono::system_clock::to_time_t(selected.timestamp);
                std::tm* tm = std::localtime(&time_t);
                char time_buf[64];
                std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm);
                ImGui::Text("Recorded: %s", time_buf);

                ImGui::TreePop();
            }

            // Remove button (outside tree node)
            ImGui::Spacing();
            if (ImGui::Button("Remove Selected Buffer")) {
                removeTrialBuffer(mSelectedBufferIndex);
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear All Buffers")) {
                clearTrialBuffers();
            }
            ImGui::SameLine();
            if (ImGui::Button("Export to HDF")) {
                exportTrialBuffersToHDF5();
            }
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ========================================================================
    // Check if data is loaded for plotting
    // ========================================================================
    if (mAngleSweepData.empty()) {
        ImGui::TextDisabled("No sweep data loaded for visualization");
        ImGui::TextWrapped("Select a trial buffer above to visualize its data");
        return;
    }

    // Muscle Selection Section
    if (collapsingHeaderWithControls("Muscle Selection")) {
        if (mAngleSweepTrackedMuscles.empty()) {
            ImGui::TextDisabled("No muscles tracked");
        } else {
            ImGui::Indent();
            
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
            for (const auto& muscle_name : mAngleSweepTrackedMuscles) {
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
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "(total %zu)", mAngleSweepTrackedMuscles.size());
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
            
            ImGui::Unindent();
        }
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Render the plots with filtered muscles
    renderMusclePlots();
}

void PhysicalExam::drawEtcTabContent() {
    // ROM summary table
    drawROMSummaryTable();

    ImGui::Separator();

    // Posture control graphs
    drawGraphPanel();
}

void PhysicalExam::drawROMSummaryTable() {
    // Define expected ROM structure in display order: hip -> knee -> ankle
    static const std::vector<std::pair<std::string, std::vector<std::string>>> expectedROMs = {
        {"hip", {"abduction_knee0", "abduction_knee90", "adduction",
                 "external_rotation", "internal_rotation", "staheli_extension"}},
        {"knee", {"popliteal", "popliteal_relax"}},
        {"ankle", {"dorsiflexion_knee0", "dorsiflexion_knee90", "plantarflexion"}}
    };

    // ROMEntry: nullopt = no trial, NaN = trial exists but no ROM, value = ROM data
    struct ROMEntry {
        std::optional<double> left;   // nullopt = no trial ("-"), NaN = no ROM ("N/A")
        std::optional<double> right;
        // Clinical data reference for lookup
        std::string cd_joint_left;
        std::string cd_field_left;
        bool cd_neg_left = false;
        std::string cd_joint_right;
        std::string cd_field_right;
        bool cd_neg_right = false;
    };

    // Initialize with all expected measurements (nullopt = no trial)
    std::map<std::string, std::map<std::string, ROMEntry>> grouped;
    for (const auto& [joint, measurements] : expectedROMs) {
        for (const auto& meas : measurements) {
            grouped[joint][meas] = ROMEntry{};  // Both left and right are nullopt
        }
    }

    // Fill in from trial buffers
    for (const auto& buf : mTrialBuffers) {
        if (buf.alias.empty()) continue;

        // Parse alias: "joint/measurement/side"
        std::vector<std::string> parts;
        std::istringstream iss(buf.alias);
        std::string part;
        while (std::getline(iss, part, '/')) {
            parts.push_back(part);
        }
        if (parts.size() != 3) continue;

        std::string joint = parts[0];
        std::string measurement = parts[1];
        std::string side = parts[2];

        // Trial exists: set value or NaN (if no cutoff angle)
        double value = buf.cutoff_angles.empty() ? std::numeric_limits<double>::quiet_NaN() : buf.cutoff_angles[0];

        // Apply negation if neg flag is set (e.g., dorsiflexion stored as negative)
        if (buf.neg && !std::isnan(value)) {
            value = -value;
        }

        if (side == "left") {
            grouped[joint][measurement].left = value;
            grouped[joint][measurement].cd_joint_left = buf.cd_joint;
            grouped[joint][measurement].cd_field_left = buf.cd_field;
            grouped[joint][measurement].cd_neg_left = buf.cd_neg;
        } else if (side == "right") {
            grouped[joint][measurement].right = value;
            grouped[joint][measurement].cd_joint_right = buf.cd_joint;
            grouped[joint][measurement].cd_field_right = buf.cd_field;
            grouped[joint][measurement].cd_neg_right = buf.cd_neg;
        }
    }

    // Helper lambda to render measured value with clinical comparison
    // Color based on deviation from clinical value (degrees)
    auto renderMeasuredWithClinicalComparison = [](const std::optional<double>& measured,
                                                    const std::optional<float>& clinical) {
        if (!measured.has_value()) {
            ImGui::TextDisabled("-");
            return;
        }
        if (std::isnan(*measured)) {
            ImGui::TextDisabled("N/A");
            return;
        }

        // If no clinical data, show measured value in white
        if (!clinical.has_value()) {
            ImGui::Text("%.1f", *measured);
            return;
        }

        // Color based on deviation from clinical value (direct comparison)
        double diff = std::abs(*measured - *clinical);
        ImVec4 color;
        if (diff <= 5.0) {
            // Within 5°: green (matching)
            color = ImVec4(0.3f, 0.9f, 0.3f, 1.0f);
        } else if (diff <= 15.0) {
            // 5-15°: yellow (mild deviation)
            color = ImVec4(0.9f, 0.9f, 0.3f, 1.0f);
        } else {
            // >15°: red (significant deviation)
            color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
        }
        ImGui::TextColored(color, "%.1f", *measured);
    };

    // Helper lambda to get clinical value from mClinicalROM
    // Clinical values are displayed exactly as stored in rom.yaml (no transformation).
    // The neg flag is only for measured ROM display, not clinical values.
    auto getClinicalValue = [this](const std::string& side, const std::string& cd_joint,
                                    const std::string& cd_field, bool /*cd_neg*/) -> std::optional<float> {
        if (cd_joint.empty() || cd_field.empty()) {
            return std::nullopt;
        }
        std::string key = side + "." + cd_joint + "." + cd_field;
        auto it = mClinicalROM.find(key);
        if (it != mClinicalROM.end() && it->second.has_value()) {
            return *it->second;  // Return raw value as-is from rom.yaml
        }
        return std::nullopt;
    };

    // Check if we have any clinical data loaded
    bool hasClinicalData = !mClinicalROM.empty();

    // Render table (larger font for readability)
    ImGui::SetWindowFontScale(1.3f);

    // Display character config as table title
    {
        std::string title = characterConfig();
        if (!title.empty()) {
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "%s", title.c_str());
        }
    }

    // 6 columns when clinical data available, 4 columns otherwise
    int numColumns = hasClinicalData ? 6 : 4;
    if (ImGui::BeginTable("ROMSummary", numColumns, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Measurement", ImGuiTableColumnFlags_WidthStretch);
        if (hasClinicalData) {
            ImGui::TableSetupColumn("CD L", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("Meas L", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("CD R", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("Meas R", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("Norm", ImGuiTableColumnFlags_WidthFixed, 55.0f);
        } else {
            ImGui::TableSetupColumn("Norm", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Left", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Right", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        }
        ImGui::TableHeadersRow();

        // Iterate in defined order
        for (const auto& [joint, measurements] : expectedROMs) {
            // Joint header row
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", joint.c_str());
            for (int i = 1; i < numColumns; i++) {
                ImGui::TableNextColumn();
            }

            for (const auto& meas : measurements) {
                const auto& entry = grouped[joint][meas];
                std::string normKey = joint + "/" + meas;

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  %s", meas.c_str());

                // Look up normative value
                auto normIt = mNormativeROM.find(normKey);
                double normative = normIt != mNormativeROM.end() ? normIt->second : 0.0;

                if (hasClinicalData) {
                    // Clinical Data Left column
                    ImGui::TableNextColumn();
                    auto clinicalLeft = getClinicalValue("left", entry.cd_joint_left, entry.cd_field_left, entry.cd_neg_left);
                    if (clinicalLeft.has_value()) {
                        ImGui::TextDisabled("%.1f", *clinicalLeft);
                    } else {
                        ImGui::TextDisabled("-");
                    }

                    // Measured Left column (color-coded vs clinical)
                    ImGui::TableNextColumn();
                    renderMeasuredWithClinicalComparison(entry.left, clinicalLeft);

                    // Clinical Data Right column
                    ImGui::TableNextColumn();
                    auto clinicalRight = getClinicalValue("right", entry.cd_joint_right, entry.cd_field_right, entry.cd_neg_right);
                    if (clinicalRight.has_value()) {
                        ImGui::TextDisabled("%.1f", *clinicalRight);
                    } else {
                        ImGui::TextDisabled("-");
                    }

                    // Measured Right column (color-coded vs clinical)
                    ImGui::TableNextColumn();
                    renderMeasuredWithClinicalComparison(entry.right, clinicalRight);

                    // Normative column
                    ImGui::TableNextColumn();
                    if (normIt != mNormativeROM.end()) {
                        ImGui::TextDisabled("%.1f", normative);
                    } else {
                        ImGui::TextDisabled("-");
                    }
                } else {
                    // Original 4-column layout when no clinical data
                    // Normative value column
                    ImGui::TableNextColumn();
                    if (normIt != mNormativeROM.end()) {
                        ImGui::TextDisabled("%.1f", normative);
                    } else {
                        ImGui::TextDisabled("-");
                    }

                    // Left value (color based on normative comparison)
                    ImGui::TableNextColumn();
                    if (!entry.left.has_value()) {
                        ImGui::TextDisabled("-");
                    } else if (std::isnan(*entry.left)) {
                        ImGui::TextDisabled("N/A");
                    } else if (normIt != mNormativeROM.end()) {
                        double diff_pct = std::abs(*entry.left - normative) / normative * 100.0;
                        ImVec4 color = diff_pct <= 10.0 ? ImVec4(0.3f, 0.9f, 0.3f, 1.0f) :
                                       diff_pct <= 25.0 ? ImVec4(0.9f, 0.9f, 0.3f, 1.0f) :
                                                          ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
                        ImGui::TextColored(color, "%.1f", *entry.left);
                    } else {
                        ImGui::Text("%.1f", *entry.left);
                    }

                    // Right value (color based on normative comparison)
                    ImGui::TableNextColumn();
                    if (!entry.right.has_value()) {
                        ImGui::TextDisabled("-");
                    } else if (std::isnan(*entry.right)) {
                        ImGui::TextDisabled("N/A");
                    } else if (normIt != mNormativeROM.end()) {
                        double diff_pct = std::abs(*entry.right - normative) / normative * 100.0;
                        ImVec4 color = diff_pct <= 10.0 ? ImVec4(0.3f, 0.9f, 0.3f, 1.0f) :
                                       diff_pct <= 25.0 ? ImVec4(0.9f, 0.9f, 0.3f, 1.0f) :
                                                          ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
                        ImGui::TextColored(color, "%.1f", *entry.right);
                    } else {
                        ImGui::Text("%.1f", *entry.right);
                    }
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::SetWindowFontScale(1.0f);  // Reset font scale
}

void PhysicalExam::drawContent() {
    glEnable(GL_LIGHTING);

    GUI::DrawGroundGrid(GroundMode::Wireframe, 10, 0.5f);

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

    drawSkeleton();
    drawMuscles();

    if (mCharacter && mRenderMainCharacter) {
        drawJointPassiveForces();
        drawSelectedAnchors();
        drawReferenceAnchor();
    }

    drawForceArrow();
    drawConfinementForces();
    drawPostureForces();

    // Draw origin axis gizmo when camera is moving
    if (mCameraMoving) {
        Eigen::Vector3d center = -mCamera.trans;
        GUI::DrawOriginAxisGizmo(center);
    }
}

void PhysicalExam::drawSkeleton() {
    // Render main character (white)
    if (mCharacter && mRenderMainCharacter) {
        GUI::DrawSkeleton(mCharacter->getSkeleton(),
                          Eigen::Vector4d(1.0, 1.0, 1.0, 0.9),
                          mRenderMode, &mShapeRenderer);
    }

    // Render standard character (gray, semi-transparent)
    if (mStdCharacter && mRenderStdCharacter) {
        GUI::DrawSkeleton(mStdCharacter->getSkeleton(),
                          Eigen::Vector4d(0.5, 0.5, 0.5, 0.5),
                          mRenderMode, &mShapeRenderer);
    }
}

void PhysicalExam::drawMuscles() {
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_DEPTH_TEST);
    glDisableClientState(GL_COLOR_ARRAY);

    // Helper to compute muscle color based on current mode
    auto getMuscleColor = [&](Muscle* muscle) -> Eigen::Vector4f {
        if (mMuscleColorMode == 0) {
            // Passive Force mode (blue gradient)
            double f_p = muscle->Getf_p();
            double normalized = std::min(1.0, f_p / mPassiveForceNormalizer);
            return Eigen::Vector4f(0.1f, 0.1f, 0.1f + 0.9f * normalized, mMuscleTransparency);
        } else {
            // Normalized Length mode (viridis)
            double lm_norm = muscle->GetLmNorm();
            double t = (lm_norm - mLmNormMin) / (mLmNormMax - mLmNormMin);
            Eigen::Vector3d rgb = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis).ConvertToEigen();
            return Eigen::Vector4f(rgb[0], rgb[1], rgb[2], mMuscleTransparency);
        }
    };

    // Render main character muscles
    if (mCharacter && mRenderMainCharacter && !mCharacter->getMuscles().empty()) {
        auto& muscles = mCharacter->getMuscles();

        if (mShowAnchorPoints) {
            glDisable(GL_LIGHTING);
            glLineWidth(mMuscleLineWidth);

            for (size_t i = 0; i < muscles.size(); i++) {
                if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

                auto muscle = muscles[i];
                auto& anchors = muscle->GetAnchors();

                Eigen::Vector4f color = getMuscleColor(muscle);
                glColor4f(color[0], color[1], color[2], color[3]);

                glBegin(GL_LINE_STRIP);
                for (auto& anchor : anchors) {
                    Eigen::Vector3d pos = anchor->GetPoint();
                    glVertex3f(pos[0], pos[1], pos[2]);
                }
                glEnd();

                for (auto& anchor : anchors) {
                    Eigen::Vector3d anchorPos = anchor->GetPoint();

                    if (!anchor->bodynodes.empty()) {
                        glColor4f(0.0f, 0.8f, 0.0f, 0.6f);
                        glBegin(GL_LINES);
                        for (auto& bodynode : anchor->bodynodes) {
                            Eigen::Vector3d bnPos = bodynode->getWorldTransform().translation();
                            glVertex3f(anchorPos[0], anchorPos[1], anchorPos[2]);
                            glVertex3f(bnPos[0], bnPos[1], bnPos[2]);
                        }
                        glEnd();
                    }

                    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
                    glPushMatrix();
                    glTranslatef(anchorPos[0], anchorPos[1], anchorPos[2]);
                    glutSolidSphere(0.004, 12, 12);
                    glPopMatrix();
                }
            }

            glLineWidth(1.0f);
            glEnable(GL_LIGHTING);
        } else {
            for (size_t i = 0; i < muscles.size(); i++) {
                if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

                auto muscle = muscles[i];
                Eigen::Vector4f color = getMuscleColor(muscle);
                glColor4f(color[0], color[1], color[2], color[3]);
                mShapeRenderer.renderMuscleLine(muscle, mMuscleLineWidth);
            }
        }
    }

    // Render standard character muscles (semi-transparent reddish)
    if (mStdCharacter && mRenderStdCharacter && !mStdCharacter->getMuscles().empty()) {
        auto& muscles = mStdCharacter->getMuscles();
        glDisable(GL_LIGHTING);
        glLineWidth(1.5f);

        for (auto muscle : muscles) {
            auto& anchors = muscle->GetAnchors();
            glColor4f(0.6f, 0.4f, 0.4f, 0.4f);
            glBegin(GL_LINE_STRIP);
            for (auto& anchor : anchors) {
                Eigen::Vector3d pos = anchor->GetPoint();
                glVertex3f(pos[0], pos[1], pos[2]);
            }
            glEnd();
        }

        glLineWidth(1.0f);
        glEnable(GL_LIGHTING);
    }
}

void PhysicalExam::drawForceArrow() {
    if (!mCharacter || !mApplyingForce) return;

    // Get body node from UI selection
    const char* bodyNodes[] = {"Pelvis", "FemurR", "FemurL", "TibiaR", "TibiaL", "TalusR", "TalusL"};
    auto bn = mCharacter->getSkeleton()->getBodyNode(bodyNodes[mSelectedBodyNode]);
    if (!bn) return;

    // Get force parameters directly from UI
    Eigen::Vector3d offset(mOffsetX, mOffsetY, mOffsetZ);
    Eigen::Vector3d direction(mForceX, mForceY, mForceZ);
    direction.normalize();

    // Calculate force application point in world coordinates
    Eigen::Vector3d world_pos = bn->getWorldTransform() * offset;
    double length = mForceMagnitude * 0.001;  // Scale for visualization

    Eigen::Vector4d purple(0.6, 0.2, 0.8, 1.0);
    GUI::DrawArrow3D(world_pos, direction, length, 0.01, purple);
}

void PhysicalExam::drawConfinementForces() {
    if (!mCharacter || !mApplyConfinementForce) return;

    const char* confinementBodies[] = {"Pelvis", "Torso", "ShoulderR", "ShoulderL"};
    Eigen::Vector3d forceDirection(0.0, -1.0, 0.0);  // Downward
    double length = 500.0 * 0.001;  // forceMagnitude * visualScale
    Eigen::Vector4d purple(0.6, 0.2, 0.8, 1.0);

    for (const char* bodyName : confinementBodies) {
        auto bn = mCharacter->getSkeleton()->getBodyNode(bodyName);
        if (!bn) continue;

        Eigen::Vector3d world_pos = bn->getWorldTransform().translation();
        GUI::DrawArrow3D(world_pos, forceDirection, length, 0.01, purple);
    }
}

void PhysicalExam::drawSelectedAnchors() {
    if (!mCharacter || !mSurgeryPanel) return;

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) return;

    // Get selection state from SurgeryPanel
    std::string candidateMuscle = mSurgeryPanel->getCandidateMuscle();
    int candidateAnchorIdx = mSurgeryPanel->getCandidateAnchorIndex();
    std::string referenceMuscle = mSurgeryPanel->getReferenceMuscle();
    int referenceAnchorIdx = mSurgeryPanel->getReferenceAnchorIndex();

    glDisable(GL_LIGHTING);

    // Draw selected candidate anchor with green dot
    if (!candidateMuscle.empty() && candidateAnchorIdx >= 0) {
        Muscle* candidateMusclePtr = nullptr;
        for (auto m : muscles) {
            if (m->name == candidateMuscle) {
                candidateMusclePtr = m;
                break;
            }
        }

        if (candidateMusclePtr) {
            auto anchors = candidateMusclePtr->GetAnchors();
            if (candidateAnchorIdx < (int)anchors.size()) {
                auto anchor = anchors[candidateAnchorIdx];
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
    if (!referenceMuscle.empty() && referenceAnchorIdx >= 0) {
        Muscle* referenceMusclePtr = nullptr;
        for (auto m : muscles) {
            if (m->name == referenceMuscle) {
                referenceMusclePtr = m;
                break;
            }
        }

        if (referenceMusclePtr) {
            auto anchors = referenceMusclePtr->GetAnchors();
            if (referenceAnchorIdx < (int)anchors.size()) {
                auto anchor = anchors[referenceAnchorIdx];
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
    if (!mCharacter || !mSurgeryPanel) return;

    auto muscles = mCharacter->getMuscles();
    if (muscles.empty()) return;

    // Get FDO selection state from SurgeryPanel
    std::string rotateAnchorMuscle = mSurgeryPanel->getRotateAnchorMuscle();
    int rotateAnchorIdx = mSurgeryPanel->getRotateAnchorIndex();
    Eigen::Vector3f searchDirF = mSurgeryPanel->getRotateAnchorSearchDir();
    Eigen::Vector3d search_dir(searchDirF[0], searchDirF[1], searchDirF[2]);

    // Draw reference anchor for FDO rotation operations
    if (!rotateAnchorMuscle.empty() && rotateAnchorIdx >= 0) {
        Muscle* refMuscle = nullptr;
        for (auto m : muscles) {
            if (m->name == rotateAnchorMuscle) {
                refMuscle = m;
                break;
            }
        }

        if (refMuscle) {
            auto anchors = refMuscle->GetAnchors();
            if (rotateAnchorIdx < (int)anchors.size()) {
                auto anchor = anchors[rotateAnchorIdx];
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
                if (search_dir.norm() > 1e-6 && !anchor->bodynodes.empty()) {
                    try {
                        // Create AnchorReference for reference anchor
                        AnchorReference ref_anchor(rotateAnchorMuscle, rotateAnchorIdx, 0);

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
    if (mCharacter->getMuscles().empty()) return;

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
                    float scale = base_font_scale * mCamera.zoom;
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

// mouseMove, mousePress, mouseScroll removed - handled by ViewerAppBase

void PhysicalExam::keyPress(int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;

    // Handle visualize sweep keys (always process, even when ImGui has focus)
    if (mVisualizeSweep) {
        if (key == GLFW_KEY_N) { mSweepNextPressed = true; return; }
        if (key == GLFW_KEY_Q) { mSweepQuitPressed = true; return; }
    }

    // Ctrl+1/2/3 for camera planes (delegate to base)
    if ((mods & GLFW_MOD_CONTROL) && key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
        alignCameraToPlane(key - GLFW_KEY_1 + 1);
        return;
    }

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_SPACE) {
        setPaused(!mSimulationPaused);
    }
    else if (key == GLFW_KEY_S) {
        if (mSimulationPaused) mSingleStep = true;
    }
    else if (key == GLFW_KEY_C && (mods & GLFW_MOD_CONTROL)) {
        // Ctrl+C: Toggle between main and standard character rendering
        if (mStdCharacter) {
            if (mRenderMainCharacter && !mRenderStdCharacter) {
                // Currently showing main -> switch to std
                mRenderMainCharacter = false;
                mRenderStdCharacter = true;
                LOG_INFO("Rendering: Standard Character");
            } else if (!mRenderMainCharacter && mRenderStdCharacter) {
                // Currently showing std -> switch to main
                mRenderMainCharacter = true;
                mRenderStdCharacter = false;
                LOG_INFO("Rendering: Main Character");
            } else {
                // Currently showing both or neither -> switch to main
                mRenderMainCharacter = true;
                mRenderStdCharacter = false;
                LOG_INFO("Rendering: Main Character");
            }
        }
    }
    else if (key == GLFW_KEY_R) {
        reset();
    }
    else if (key == GLFW_KEY_O) {
        // Cycle through render modes: Primitive -> Mesh -> Wireframe -> Primitive
        if (mRenderMode == RenderMode::Primitive) {
            mRenderMode = RenderMode::Mesh;
            LOG_INFO("Rendering mode: Mesh");
        } else if (mRenderMode == RenderMode::Mesh) {
            mRenderMode = RenderMode::Wireframe;
            LOG_INFO("Rendering mode: Wireframe");
        } else {
            mRenderMode = RenderMode::Primitive;
            LOG_INFO("Rendering mode: Primitive");
        }
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
    else if (key == GLFW_KEY_9) {
        selectCameraPresetInteractive();
    }
    else if (key == GLFW_KEY_0) {
        loadCameraPreset(0);  // Top view
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
        mCamera = mCameraPresets[0].state;  // Copy entire camera state
        mCurrentCameraPreset = 0;
    } else {
        mCamera.eye << 0.0, 1.0, 3.0;
        mCamera.up << 0.0, 1.0, 0.0;
        mCamera.trans << 0.0, -0.5, 0.0;
        mCamera.zoom = 1.0;
        mCamera.trackball.setQuaternion(Eigen::Quaterniond::Identity());
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

    // Store the actuator type before deletion
    ActuatorType actType = mCharacter->getActuatorType();

    // Get the skeleton before deletion
    auto skel = mCharacter->getSkeleton();

    // FIRST: Ensure GPU has finished all pending rendering operations
    // This prevents GPU from accessing VBOs after we delete them
    glFinish();
    // SECOND: Delete VBOs from GPU (now safe - GPU not using them)
    mShapeRenderer.clearCache();
    // THIRD: Remove skeleton from world (stops it from being rendered)
    if (mWorld) {
        mWorld->removeSkeleton(skel);
        LOG_INFO("[PhysicalExam] Removed skeleton from world");
    }

    // FOURTH: Delete the character object
    delete mCharacter;
    mCharacter = nullptr;

    // Reload character from stored XML paths
    if (mSkeletonPath.empty() || mMusclePath.empty()) {
        LOG_ERROR("[PhysicalExam] Error: skeleton or muscle path not stored");
        return;
    }

    loadCharacter(mSkeletonPath, mMusclePath, actType);
}

void PhysicalExam::setPoseStanding() {
    if (!mCharacter) return;

    mCurrentPosePreset = 0;
    auto skel = mCharacter->getSkeleton();

    // Reset to default standing pose
    setCharacterPose(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Set pelvis height
    auto root = skel->getRootJoint();
    if (root->getNumDofs() >= 6) {
        Eigen::VectorXd root_pos = root->getPositions();
        root_pos[4] = 0.98;  // Y position (height)
        setCharacterPose(root->getName(), root_pos);
    }
}

void PhysicalExam::setPoseSupine() {
    if (!mCharacter) return;

    mCurrentPosePreset = 1;
    auto skel = mCharacter->getSkeleton();

    // Reset all joints
    setCharacterPose(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Rotate to lay on back (supine = face up)
    // Root joint: indices 0,1,2 are rotation (roll, pitch, yaw), indices 3,4,5 are translation (x,y,z)
    auto root = skel->getRootJoint();
    if (root->getNumDofs() >= 6) {
        Eigen::VectorXd root_pos = root->getPositions();
        root_pos[0] = -M_PI / 2.0;  // Rotate around X axis (roll) - index 0 is roll rotation
        root_pos[4] = 0.1;  // Table height - index 4 is Y translation
        setCharacterPose(root->getName(), root_pos);
    }
}

void PhysicalExam::setPoseProne() {
    if (!mCharacter) return;

    mCurrentPosePreset = 2;
    auto skel = mCharacter->getSkeleton();

    // Reset all joints
    setCharacterPose(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Rotate to lay on front (prone = face down)
    // Root joint: indices 0,1,2 are rotation (roll, pitch, yaw), indices 3,4,5 are translation (x,y,z)
    auto root = skel->getRootJoint();
    if (root->getNumDofs() >= 6) {
        Eigen::VectorXd root_pos = root->getPositions();
        root_pos[0] = M_PI / 2.0;  // Rotate around X axis (negative roll) - index 0 is roll rotation
        root_pos[4] = 0.1;  // Table height - index 4 is Y translation
        setCharacterPose(root->getName(), root_pos);
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
        
        setCharacterPose(hip_flex_r->getName(), hip_r_pos);
        setCharacterPose(hip_flex_l->getName(), hip_l_pos);
    }

    if (knee_r && knee_l) {
        Eigen::VectorXd knee_r_pos = knee_r->getPositions();
        Eigen::VectorXd knee_l_pos = knee_l->getPositions();

        if (knee_r_pos.size() > 0) knee_r_pos[0] = angle_rad;
        if (knee_l_pos.size() > 0) knee_l_pos[0] = angle_rad;

        setCharacterPose(knee_r->getName(), knee_r_pos);
        setCharacterPose(knee_l->getName(), knee_l_pos);
    }

    LOG_INFO("Pose: Supine with knee flexion (" << knee_angle << " degrees)");
}

void PhysicalExam::printCameraInfo() {
    Eigen::Quaterniond quat = mCamera.trackball.getCurrQuat();

    LOG_INFO("\n======================================");
    LOG_INFO("Copy and paste below to CAMERA_PRESET_DEFINITIONS:");
    LOG_INFO("======================================");
    LOG_INFO("PRESET|[Add description]|"
              << mCamera.eye[0] << "," << mCamera.eye[1] << "," << mCamera.eye[2] << "|"
              << mCamera.up[0] << "," << mCamera.up[1] << "," << mCamera.up[2] << "|"
              << mCamera.trans[0] << "," << mCamera.trans[1] << "," << mCamera.trans[2] << "|"
              << mCamera.zoom << "|"
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
    if (collapsingHeaderWithControls("Posture Control Forces")) {
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
    if (index < 0 || index >= 10) {
        LOG_ERROR("Invalid camera preset index: " << index);
        return;
    }

    if (!mCameraPresets[index].isSet) {
        LOG_INFO("Camera preset " << (index + 1) << " is not set yet");
        return;
    }

    mCamera = mCameraPresets[index].state;  // Copy entire camera state
    mCurrentCameraPreset = index;
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
    "PRESET|Surgery overview|0,0.813471,2.44041|0,1,0|-0.289215,-0.570655,-0.280608|1|0.81416,0.580639,0,0"
};

void PhysicalExam::initializeCameraPresets() {
    const int numPresets = sizeof(CAMERA_PRESET_DEFINITIONS) / sizeof(CAMERA_PRESET_DEFINITIONS[0]);
    
    for (int i = 0; i < numPresets; ++i) {
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
            
            // Set the preset (using CameraState)
            if (eyeVals.size() == 3 && upVals.size() == 3 &&
                transVals.size() == 3 && quatVals.size() == 4) {
                mCameraPresets[i].description = tokens[1];
                mCameraPresets[i].state.eye << eyeVals[0], eyeVals[1], eyeVals[2];
                mCameraPresets[i].state.up << upVals[0], upVals[1], upVals[2];
                mCameraPresets[i].state.trans << transVals[0], transVals[1], transVals[2];
                mCameraPresets[i].state.zoom = zoom;
                mCameraPresets[i].state.trackball.setQuaternion(
                    Eigen::Quaterniond(quatVals[0], quatVals[1], quatVals[2], quatVals[3]));
                mCameraPresets[i].isSet = true;
            }
        }
    }
}

std::string PhysicalExam::characterConfig() const {
    namespace fs = std::filesystem;
    std::string skel_stem = mSkeletonPath.empty() ? "" : fs::path(mSkeletonPath).stem().string();
    std::string musc_stem = mMusclePath.empty() ? "" : fs::path(mMusclePath).stem().string();
    std::string result = skel_stem;
    if (!musc_stem.empty()) result += " | " + musc_stem;
    if (!mBrowseCharacterPID.empty()) result = "pid:" + mBrowseCharacterPID + " " + result;
    return result;
}

void PhysicalExam::selectCameraPresetInteractive() {
    // Collect available preset indices
    std::vector<int> availableIndices;
    for (int i = 0; i < 10; ++i) {
        if (mCameraPresets[i].isSet) {
            availableIndices.push_back(i);
        }
    }

    if (availableIndices.empty()) {
        std::cout << "No camera presets available." << std::endl;
        return;
    }

    // Display available presets
    std::cout << "\n=== Available Camera Presets ===" << std::endl;
    for (int idx : availableIndices) {
        std::cout << "  [" << idx << "] " << mCameraPresets[idx].description << std::endl;
    }
    std::cout << "================================" << std::endl;
    std::cout << "Enter preset number: ";
    std::cout.flush();

    // Read user input
    int selection = -1;
    std::cin >> selection;

    // Validate and load
    if (selection >= 0 && selection < 10 && mCameraPresets[selection].isSet) {
        loadCameraPreset(selection);
        std::cout << "Loaded preset: " << mCameraPresets[selection].description << std::endl;
    } else {
        std::cout << "Invalid selection: " << selection << std::endl;
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

    mAngleSweepTrackedMuscles.clear();
    if (!mCharacter) return;
    if (mCharacter->getMuscles().empty()) return;

    auto skel = mCharacter->getSkeleton();
    if (mSweepConfig.joint_index >= skel->getNumJoints()) {
        LOG_ERROR("Invalid joint index: " << mSweepConfig.joint_index);
        return;
    }

    auto joint = skel->getJoint(mSweepConfig.joint_index);
    auto muscles = mCharacter->getMuscles();

    int root_dofs = skel->getRootJoint()->getNumDofs();
    int total_dofs = skel->getNumDofs() - root_dofs;

    for (auto muscle : muscles) {
        auto related_joints = muscle->GetRelatedJoints();
        if (std::find(related_joints.begin(), related_joints.end(), joint)
            != related_joints.end()) {
            std::string muscleName = muscle->GetName();
            mAngleSweepTrackedMuscles.push_back(muscleName);

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

    // Set default plot joint to sweep joint
    mSelectedPlotJointIndex = mSweepConfig.joint_index;

    // Remove visibility entries for muscles no longer tracked
    std::map<std::string, bool> newVisibility;
    for (const auto& muscleName : mAngleSweepTrackedMuscles) {
        newVisibility[muscleName] = mMuscleVisibility[muscleName];
    }
    mMuscleVisibility = newVisibility;

    LOG_INFO("Detected " << mAngleSweepTrackedMuscles.size()
              << " muscles crossing joint: "
              << joint->getName());
    
    // Setup standard character muscles
    if (mStdCharacter && !mStdCharacter->getMuscles().empty()) {
        mStdAngleSweepTrackedMuscles.clear();
        auto std_muscles = mStdCharacter->getMuscles();
        auto std_skel = mStdCharacter->getSkeleton();
        auto std_joint = std_skel->getJoint(joint->getName());
        
        if (std_joint) {
            for (auto muscle : std_muscles) {
                auto related_joints = muscle->GetRelatedJoints();
                if (std::find(related_joints.begin(), related_joints.end(), std_joint)
                    != related_joints.end()) {
                    mStdAngleSweepTrackedMuscles.push_back(muscle->GetName());
                }
            }
            LOG_INFO("Standard character: " << mStdAngleSweepTrackedMuscles.size()
                      << " muscles crossing joint");
        }
    }
}

void PhysicalExam::runSweep() {
    if (!mCharacter) {
        LOG_ERROR("No character loaded");
        return;
    }

    // Initialize sweep
    clearSweepData();
    setupSweepMuscles();

    if (mAngleSweepTrackedMuscles.empty()) {
        LOG_INFO("No muscles cross this joint");
        return;
    }

    auto skel = mCharacter->getSkeleton();
    auto joint = skel->getJoint(mSweepConfig.joint_index);

    // Set current sweep name for plot titles (with joint name and DOF index)
    mCurrentSweepName = "GUI Sweep: " + joint->getName() + " DOF" + std::to_string(mSweepConfig.dof_index);

    LOG_INFO("Starting sweep: " << joint->getName()
              << " from " << mSweepConfig.angle_min
              << " to " << mSweepConfig.angle_max
              << " rad (" << mSweepConfig.num_steps << " steps)");

    // Clear previous sweep data
    mAngleSweepData.clear();
    mStdAngleSweepData.clear();
    
    // Set joint index for passive torque calculation
    mAngleSweepJointIdx = mSweepConfig.joint_index;

    // Store original joint position for restoration
    mSweepOriginalPos = joint->getPositions();

    // Start sweep (will execute incrementally in mainLoop)
    mSweepRunning = true;
    mSweepCurrentStep = 0;
}


void PhysicalExam::renderMusclePlots() {
    if (mAngleSweepData.empty()) return;

    // Apply plot background color based on checkbox
    ImPlotStyle& style = ImPlot::GetStyle();
    if (mPlotWhiteBackground) {
        style.Colors[ImPlotCol_PlotBg] = ImVec4(1, 1, 1, 1);
        style.Colors[ImPlotCol_AxisGrid] = ImVec4(0, 0, 0, 1);  // Black grid lines on white background
    } else {
        // Restore default (dark background)
        style.Colors[ImPlotCol_PlotBg] = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
        style.Colors[ImPlotCol_AxisGrid] = ImVec4(0.5f, 0.5f, 0.5f, 0.25f);  // Default light gray grid lines
    }

    static bool sShowSweepLegend = true;

    ImGui::Checkbox("Show Legend", &sShowSweepLegend);
    ImGui::SameLine();
    ImGui::Checkbox("Title (character)", &mShowCharacterInTitles);

    // NEW: Checkbox to toggle standard character overlay
    if (mStdCharacter && !mStdAngleSweepData.empty()) {
        ImGui::SameLine();
        ImGui::Checkbox("Show Standard Character", &mShowStdCharacterInPlots);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Overlay standard character data (dashed lines) for comparison");
        }
    }

    // X-axis normalization control
    ImGui::SeparatorText("X-Axis Mode");
    int mode_idx = static_cast<int>(mXAxisMode);
    if (ImGui::RadioButton("Angle (deg)", &mode_idx, 0)) {
        mXAxisMode = XAxisMode::RAW_ANGLE;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Normalized (%)", &mode_idx, 1)) {
        mXAxisMode = XAxisMode::NORMALIZED;
    }

    // Require a valid buffer for visualization
    if (mSelectedBufferIndex < 0 || mSelectedBufferIndex >= static_cast<int>(mTrialBuffers.size())) {
        ImGui::TextDisabled("No trial buffer selected");
        return;
    }

    // Get angle range from selected buffer's config
    const auto& buffer = mTrialBuffers[mSelectedBufferIndex];
    double angle_min = buffer.config.angle_min;
    double angle_max = buffer.config.angle_max;

    // For composite DOF (e.g., abd_knee), angle range may not be set in config
    // Use actual data range instead
    if (std::abs(angle_max - angle_min) < 1e-6 && !mAngleSweepData.empty()) {
        angle_min = mAngleSweepData.front().joint_angle;
        angle_max = mAngleSweepData.back().joint_angle;
    }

    // Compute ROM metrics for both characters (needed for normalization)
    ROMMetrics main_rom_metrics = computeROMMetrics(mAngleSweepData);
    ROMMetrics std_rom_metrics;
    bool has_std_for_plots = mShowStdCharacterInPlots && !mStdAngleSweepData.empty();
    if (has_std_for_plots) {
        std_rom_metrics = computeROMMetrics(mStdAngleSweepData);
    }

    // Build x_data_raw (convert from radians to degrees)
    std::vector<double> x_data_raw;
    x_data_raw.reserve(mAngleSweepData.size());
    for (const auto& pt : mAngleSweepData) {
        x_data_raw.push_back(pt.joint_angle * 180.0 / M_PI);
    }

    // Apply normalization using main character's ROM range
    std::vector<double> x_data = normalizeXAxis(x_data_raw,
                                                 main_rom_metrics.rom_min_angle,
                                                 main_rom_metrics.rom_max_angle);

    // Build std character x_data_raw
    std::vector<double> std_x_data_raw;
    if (has_std_for_plots) {
        std_x_data_raw.reserve(mStdAngleSweepData.size());
        for (const auto& pt : mStdAngleSweepData) {
            std_x_data_raw.push_back(pt.joint_angle * 180.0 / M_PI);
        }
    }

    // Apply normalization to std character using its own ROM range
    std::vector<double> std_x_data = normalizeXAxis(std_x_data_raw,
                                                     std_rom_metrics.rom_min_angle,
                                                     std_rom_metrics.rom_max_angle);

    // Update x-axis label and limits based on mode
    const char* x_axis_label;
    double x_min, x_max;

    if (mXAxisMode == XAxisMode::RAW_ANGLE) {
        x_axis_label = "Joint Angle (deg)";
        x_min = angle_min * 180.0 / M_PI;
        x_max = angle_max * 180.0 / M_PI;
    } else {
        x_axis_label = "Normalized Position (%)";
        x_min = 0.0;   // Fixed 0-100 range
        x_max = 100.0;
    }

    ImPlotFlags plot_flags = sShowSweepLegend ? 0 : ImPlotFlags_NoLegend;

    // Build character info prefix if enabled: [skeleton | muscle | sweep] or pid:(pid) [...]
    std::string char_prefix;
    if (mShowCharacterInTitles) {
        char_prefix = characterConfig() + " | " + mCurrentSweepName;
    }

    // Use trial name for plot titles if enabled, with ## suffixes for uniqueness
    std::string passive_plot_title = mShowCharacterInTitles ? (char_prefix + "##passive") :
                                     (mShowTrialNameInPlots ? (mCurrentSweepName + "##passive") : "Passive Forces vs Joint Angle");
    std::string length_plot_title = mShowCharacterInTitles ? (char_prefix + "##length") :
                                    (mShowTrialNameInPlots ? (mCurrentSweepName + "##length") : "Normalized Muscle Length vs Joint Angle");
    std::string torque_plot_title = mShowCharacterInTitles ? (char_prefix + "##torque") :
                                    (mShowTrialNameInPlots ? (mCurrentSweepName + "##torque") : "Total Passive Torque vs Joint Angle");
    std::string stiffness_plot_title = mShowCharacterInTitles ? (char_prefix + "##stiffness") :
                                       (mShowTrialNameInPlots ? (mCurrentSweepName + "##stiffness") : "Passive Stiffness (dtau/dtheta) vs Joint Angle");

    // ====================================================================
    // RANGE OF MOTION ANALYSIS
    // ====================================================================
    if (collapsingHeaderWithControls("Range of Motion Analysis")) {
        if (mAngleSweepData.empty()) {
            ImGui::TextDisabled("No sweep data available. Run a sweep first.");
        } else {
            // ROM metric selector
            ImGui::SeparatorText("ROM Metric Selection");
            ImGui::Text("Compute ROM using:");
            int metric_idx = static_cast<int>(mROMMetric);
            if (ImGui::RadioButton("Stiffness only", &metric_idx, 0)) {
                mROMMetric = ROMMetric::STIFFNESS;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Torque only", &metric_idx, 1)) {
                mROMMetric = ROMMetric::TORQUE;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Either (OR)", &metric_idx, 2)) {
                mROMMetric = ROMMetric::EITHER;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Both (AND)", &metric_idx, 3)) {
                mROMMetric = ROMMetric::BOTH;
            }

            // Threshold configuration
            ImGui::SeparatorText("Threshold Criteria");
            ImGui::Text("Maximum |Stiffness|:");
            ImGui::SameLine(200);
            ImGui::PushItemWidth(120);
            ImGui::InputFloat("##max_stiffness", &mROMThresholds.max_stiffness, 1.0f, 10.0f, "%.1f Nm/rad");
            ImGui::PopItemWidth();

            ImGui::Text("Maximum |Torque|:");
            ImGui::SameLine(200);
            ImGui::PushItemWidth(120);
            ImGui::InputFloat("##max_torque", &mROMThresholds.max_torque, 1.0f, 10.0f, "%.1f Nm");
            ImGui::PopItemWidth();

            // Use already-computed ROM metrics
            const ROMMetrics& main_metrics = main_rom_metrics;
            const ROMMetrics& std_metrics = std_rom_metrics;
            bool has_std = !mStdAngleSweepData.empty() && mStdCharacter;

            // Display table
            ImGui::SeparatorText("ROM Metrics");
            if (ImGui::BeginTable("ROM_Table", has_std ? 4 : 2,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {

                // Header
                ImGui::TableSetupColumn("Metric");
                ImGui::TableSetupColumn("Main Character");
                if (has_std) {
                    ImGui::TableSetupColumn("Standard Character");
                    ImGui::TableSetupColumn("Difference");
                }
                ImGui::TableHeadersRow();

                // Sweep Range row (angle_min/angle_max from config)
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Sweep Range");
                ImGui::TableNextColumn();
                ImGui::Text("%.1f° to %.1f°", angle_min * 180.0 / M_PI, angle_max * 180.0 / M_PI);
                if (has_std) {
                    ImGui::TableNextColumn();
                    ImGui::Text("(same)");
                    ImGui::TableNextColumn();
                    ImGui::Text("-");
                }

                // Joint ROM row (angle where torque meets cutoff)
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Joint ROM");
                ImGui::TableNextColumn();
                ImGui::Text("%.2f° [%.1f° to %.1f°]", main_metrics.rom_deg,
                    main_metrics.rom_min_angle, main_metrics.rom_max_angle);
                if (has_std) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f° [%.1f° to %.1f°]", std_metrics.rom_deg,
                        std_metrics.rom_min_angle, std_metrics.rom_max_angle);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f°", main_metrics.rom_deg - std_metrics.rom_deg);
                }

                // Peak Stiffness row
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Peak Stiffness");
                ImGui::TableNextColumn();
                bool stiff_flag = main_metrics.peak_stiffness > mROMThresholds.max_stiffness;
                if (stiff_flag) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
                ImGui::Text("%.2f Nm/rad @ %.1f°", main_metrics.peak_stiffness,
                    main_metrics.angle_at_peak_stiffness);
                if (stiff_flag) ImGui::PopStyleColor();
                if (has_std) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f Nm/rad @ %.1f°", std_metrics.peak_stiffness,
                        std_metrics.angle_at_peak_stiffness);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f Nm/rad", main_metrics.peak_stiffness - std_metrics.peak_stiffness);
                }

                // Peak Torque row
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Peak Torque");
                ImGui::TableNextColumn();
                bool torque_flag = main_metrics.peak_torque > mROMThresholds.max_torque;
                if (torque_flag) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
                ImGui::Text("%.2f Nm @ %.1f°", main_metrics.peak_torque,
                    main_metrics.angle_at_peak_torque);
                if (torque_flag) ImGui::PopStyleColor();
                if (has_std) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f Nm @ %.1f°", std_metrics.peak_torque,
                        std_metrics.angle_at_peak_torque);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f Nm", main_metrics.peak_torque - std_metrics.peak_torque);
                }

                ImGui::EndTable();
            }
        }
    }

    // Total Passive Torque & Stiffness Plot
    if (collapsingHeaderWithControls("Total Passive Torque")) {
        // Build torque and stiffness data
        std::vector<double> torque_data, stiffness_data;
        torque_data.reserve(mAngleSweepData.size());
        stiffness_data.reserve(mAngleSweepData.size());
        for (const auto& pt : mAngleSweepData) {
            torque_data.push_back(pt.passive_torque_total);
            stiffness_data.push_back(pt.passive_torque_stiffness);
        }

        if (torque_data.empty()) {
            ImGui::TextDisabled("No total passive torque data available");
        } else {
            // Torque plot
            if (ImPlot::BeginPlot(torque_plot_title.c_str(), ImVec2(-1, 300), plot_flags)) {
                ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);
                ImPlot::SetupAxis(ImAxis_X1, x_axis_label);
                ImPlot::SetupAxis(ImAxis_Y1, "Passive Torque (Nm)");
                ImPlot::PlotLine("Total Torque", x_data.data(), torque_data.data(), torque_data.size());

                // Plot standard character data
                if (mShowStdCharacterInPlots && !mStdAngleSweepData.empty() && mStdCharacter) {
                    std::vector<double> std_torque_data;
                    std_torque_data.reserve(mStdAngleSweepData.size());
                    for (const auto& pt : mStdAngleSweepData) {
                        std_torque_data.push_back(pt.passive_torque_total);
                    }
                    if (!std_torque_data.empty() && std_torque_data.size() == std_x_data.size()) {
                        ImPlot::SetNextLineStyle(ImVec4(0.5, 0.5, 0.5, 1.0), 1.0f);
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_None);
                        ImPlot::PlotLine("Total Torque (std)", std_x_data.data(),
                            std_torque_data.data(), std_torque_data.size());
                    }
                }

                ImPlot::EndPlot();
            }

            // Stiffness plot
            // if (!stiffness_data.empty()) {
            //     if (ImPlot::BeginPlot(stiffness_plot_title.c_str(), ImVec2(-1, 300), plot_flags)) {
            //         ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);
            //         ImPlot::SetupAxis(ImAxis_X1, x_axis_label);
            //         ImPlot::SetupAxis(ImAxis_Y1, "Stiffness (Nm/rad)");
            //         ImPlot::PlotLine("Stiffness", x_data.data(), stiffness_data.data(), stiffness_data.size());

            //         // Plot standard character data
            //         if (mShowStdCharacterInPlots && !mStdAngleSweepData.empty() && mStdCharacter) {
            //             std::vector<double> std_stiffness_data;
            //             std_stiffness_data.reserve(mStdAngleSweepData.size());
            //             for (const auto& pt : mStdAngleSweepData) {
            //                 std_stiffness_data.push_back(pt.passive_torque_stiffness);
            //             }
            //             if (!std_stiffness_data.empty() && std_stiffness_data.size() == std_x_data.size()) {
            //                 ImPlot::SetNextLineStyle(ImVec4(0.5, 0.5, 0.5, 1.0), 1.0f);
            //                 ImPlot::SetNextMarkerStyle(ImPlotMarker_None);
            //                 ImPlot::PlotLine("Stiffness (std)", std_x_data.data(),
            //                     std_stiffness_data.data(), std_stiffness_data.size());
            //             }
            //         }

            //         ImPlot::EndPlot();
            //     }
            // }
        }
    }

    // Normalized Muscle Length Plot
    if (collapsingHeaderWithControls("Normalized Length (l_m_norm)")) {
        if (ImPlot::BeginPlot(length_plot_title.c_str(), ImVec2(-1, 400), plot_flags)) {
            ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);
            ImPlot::SetupAxis(ImAxis_X1, x_axis_label);
            ImPlot::SetupAxis(ImAxis_Y1, "lm_norm");

            for (const auto& muscle_name : mAngleSweepTrackedMuscles) {
                auto vis_it = mMuscleVisibility.find(muscle_name);
                if (vis_it == mMuscleVisibility.end() || !vis_it->second) {
                    continue;
                }

                // Build y_data for this muscle
                std::vector<double> lm_norm_data;
                lm_norm_data.reserve(mAngleSweepData.size());
                for (const auto& pt : mAngleSweepData) {
                    auto it = pt.muscle_lm_norm.find(muscle_name);
                    if (it != pt.muscle_lm_norm.end()) {
                        lm_norm_data.push_back(it->second);
                    }
                }

                if (!lm_norm_data.empty() && lm_norm_data.size() == x_data.size()) {
                    ImPlot::PlotLine(muscle_name.c_str(), x_data.data(),
                        lm_norm_data.data(), lm_norm_data.size());
                }
            }

            // Plot standard character data (dashed lines) - only if checkbox enabled
            if (mShowStdCharacterInPlots && !mStdAngleSweepData.empty() && mStdCharacter) {
                // Prepare filter for case-insensitive matching
                std::string filter_lower(mMuscleFilterBuffer);
                std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

                for (const auto& muscle_name : mStdAngleSweepTrackedMuscles) {
                    // Apply text filter (case-insensitive)
                    if (!filter_lower.empty()) {
                        std::string muscle_lower = muscle_name;
                        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
                        if (muscle_lower.find(filter_lower) == std::string::npos) {
                            continue;
                        }
                    }

                    // Check visibility - same logic as main muscles (hide if not in map or unchecked)
                    auto vis_it = mMuscleVisibility.find(muscle_name);
                    if (vis_it == mMuscleVisibility.end() || !vis_it->second) {
                        continue;
                    }

                    // Build y_data for this muscle
                    std::vector<double> std_lm_norm_data;
                    std_lm_norm_data.reserve(mStdAngleSweepData.size());
                    for (const auto& pt : mStdAngleSweepData) {
                        auto it = pt.muscle_lm_norm.find(muscle_name);
                        if (it != pt.muscle_lm_norm.end()) {
                            std_lm_norm_data.push_back(it->second);
                        }
                    }

                    if (!std_lm_norm_data.empty() && std_lm_norm_data.size() == std_x_data.size()) {
                        std::string label = muscle_name + " (std)";
                        ImPlot::SetNextLineStyle(ImVec4(0.5, 0.5, 0.5, 1.0), 1.0f);
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_None);
                        ImPlot::PlotLine(label.c_str(), std_x_data.data(),
                            std_lm_norm_data.data(), std_lm_norm_data.size());
                    }
                }
            }

            ImPlot::EndPlot();
        }
    }

    // Passive Forces Plot
    if (collapsingHeaderWithControls("Passive Forces")) {
        if (ImPlot::BeginPlot(passive_plot_title.c_str(), ImVec2(-1, 400), plot_flags)) {
            ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);
            ImPlot::SetupAxis(ImAxis_X1, x_axis_label);
            ImPlot::SetupAxis(ImAxis_Y1, "Passive Force (N)");

            for (const auto& muscle_name : mAngleSweepTrackedMuscles) {
                auto vis_it = mMuscleVisibility.find(muscle_name);
                if (vis_it == mMuscleVisibility.end() || !vis_it->second) {
                    continue;
                }

                // Build y_data for this muscle
                std::vector<double> fp_data;
                fp_data.reserve(mAngleSweepData.size());
                for (const auto& pt : mAngleSweepData) {
                    auto it = pt.muscle_fp.find(muscle_name);
                    if (it != pt.muscle_fp.end()) {
                        fp_data.push_back(it->second);
                    }
                }

                if (!fp_data.empty() && fp_data.size() == x_data.size()) {
                    ImPlot::PlotLine(muscle_name.c_str(), x_data.data(),
                        fp_data.data(), fp_data.size());
                }
            }

            // Plot standard character data (dashed lines) - only if checkbox enabled
            if (mShowStdCharacterInPlots && !mStdAngleSweepData.empty() && mStdCharacter) {
                // Prepare filter for case-insensitive matching
                std::string filter_lower(mMuscleFilterBuffer);
                std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

                for (const auto& muscle_name : mStdAngleSweepTrackedMuscles) {
                    // Apply text filter (case-insensitive)
                    if (!filter_lower.empty()) {
                        std::string muscle_lower = muscle_name;
                        std::transform(muscle_lower.begin(), muscle_lower.end(), muscle_lower.begin(), ::tolower);
                        if (muscle_lower.find(filter_lower) == std::string::npos) {
                            continue;
                        }
                    }

                    // Check visibility - same logic as main muscles (hide if not in map or unchecked)
                    auto vis_it = mMuscleVisibility.find(muscle_name);
                    if (vis_it == mMuscleVisibility.end() || !vis_it->second) {
                        continue;
                    }

                    // Build y_data for this muscle
                    std::vector<double> std_fp_data;
                    std_fp_data.reserve(mStdAngleSweepData.size());
                    for (const auto& pt : mStdAngleSweepData) {
                        auto it = pt.muscle_fp.find(muscle_name);
                        if (it != pt.muscle_fp.end()) {
                            std_fp_data.push_back(it->second);
                        }
                    }

                    if (!std_fp_data.empty() && std_fp_data.size() == std_x_data.size()) {
                        std::string label = muscle_name + " (std)";
                        ImPlot::SetNextLineStyle(ImVec4(0.5, 0.5, 0.5, 1.0), 1.0f);
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_None);
                        ImPlot::PlotLine(label.c_str(), std_x_data.data(),
                            std_fp_data.data(), std_fp_data.size());
                    }
                }
            }

            ImPlot::EndPlot();
        }
    }

    // Per-Muscle Joint Torque Plot with joint selection
    if (collapsingHeaderWithControls("Per-Muscle Joint Torque")) {
        if (!mCharacter) {
            ImGui::TextDisabled("No character loaded");
        } else {
            auto skel = mCharacter->getSkeleton();
            int root_dofs = skel->getRootJoint()->getNumDofs();

            // Joint selection listbox
            ImGui::Text("Select Joint:");
            if (ImGui::BeginListBox("##JointSelect", ImVec2(-1, 100))) {
                for (size_t i = 1; i < skel->getNumJoints(); ++i) {  // Skip root
                    auto joint = skel->getJoint(i);
                    bool is_selected = (mSelectedPlotJointIndex == static_cast<int>(i));
                    if (ImGui::Selectable(joint->getName().c_str(), is_selected)) {
                        mSelectedPlotJointIndex = static_cast<int>(i);
                    }
                }
                ImGui::EndListBox();
            }

            // Get selected joint info
            auto selected_joint = skel->getJoint(mSelectedPlotJointIndex);
            std::string joint_name = selected_joint->getName();
            int joint_dof_start = selected_joint->getIndexInSkeleton(0);  // Full skeleton index
            int num_dofs = selected_joint->getNumDofs();
            int joint_dof_end = joint_dof_start + num_dofs;  // One past last DOF

            // Plot each muscle's contribution to this joint (magnitude across joint DOFs only)
            std::string plot_title = "Per-Muscle Passive Torque: " + joint_name;
            if (ImPlot::BeginPlot(plot_title.c_str(), ImVec2(-1, 400), plot_flags)) {
                ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Always);
                ImPlot::SetupAxis(ImAxis_X1, x_axis_label);
                ImPlot::SetupAxis(ImAxis_Y1, "Passive Torque (Nm)");

                for (const auto& muscle_name : mAngleSweepTrackedMuscles) {
                    auto vis_it = mMuscleVisibility.find(muscle_name);
                    if (vis_it == mMuscleVisibility.end() || !vis_it->second) continue;

                    // Get muscle to access its DOF mapping
                    Muscle* muscle = mCharacter->getMuscleByName(muscle_name);
                    if (!muscle) continue;
                    const std::vector<int>& related_dof_indices = muscle->related_dof_indices;

                    // Build magnitude data from mAngleSweepData
                    std::vector<double> magnitude_data;
                    magnitude_data.reserve(mAngleSweepData.size());

                    for (const auto& pt : mAngleSweepData) {
                        auto it = pt.muscle_jtp.find(muscle_name);
                        if (it != pt.muscle_jtp.end()) {
                            const std::vector<double>& jtp_vec = it->second;
                            // Compute magnitude across DOFs for this joint only
                            double mag_squared = 0.0;
                            for (size_t i = 0; i < jtp_vec.size() && i < related_dof_indices.size(); ++i) {
                                int dof_idx = related_dof_indices[i];
                                // Only include torques for DOFs belonging to the selected joint
                                if (dof_idx >= joint_dof_start && dof_idx < joint_dof_end) {
                                    mag_squared += jtp_vec[i] * jtp_vec[i];
                                }
                            }
                            magnitude_data.push_back(std::sqrt(mag_squared));
                        } else {
                            magnitude_data.push_back(0.0);
                        }
                    }

                    if (!magnitude_data.empty() && magnitude_data.size() == x_data.size()) {
                        ImPlot::PlotLine(muscle_name.c_str(), x_data.data(),
                            magnitude_data.data(), magnitude_data.size());
                    }
                }
                ImPlot::EndPlot();
            }
        }
    }
}

void PhysicalExam::clearSweepData() {
    mAngleSweepData.clear();
    mAngleSweepTrackedMuscles.clear();
    mStdAngleSweepData.clear();
    mStdAngleSweepTrackedMuscles.clear();
    // DON'T clear mMuscleVisibility - preserve user selections across sweeps
    // Note: mGraphData is now only used for posture control, not sweep data
    LOG_INFO("Sweep data cleared");
}

// ============================================================================
// Multi-Trial Buffer Management
// ============================================================================

void PhysicalExam::addTrialToBuffer(const TrialDataBuffer& buffer) {
    // FIFO eviction if at mMaxTrialBuffers limit
    while (static_cast<int>(mTrialBuffers.size()) >= mMaxTrialBuffers) {
        LOG_WARN("Buffer limit reached, removing oldest trial: "
                  << mTrialBuffers.front().trial_name);
        mTrialBuffers.erase(mTrialBuffers.begin());

        // Adjust selected index if needed
        if (mSelectedBufferIndex > 0) {
            mSelectedBufferIndex--;
        } else if (mSelectedBufferIndex == 0) {
            mSelectedBufferIndex = -1;
        }
    }

    mTrialBuffers.push_back(buffer);
    LOG_INFO("Added trial to buffer: " << buffer.trial_name
              << " (total: " << mTrialBuffers.size() << ")");
}

void PhysicalExam::loadBufferForVisualization(int buffer_index) {
    if (buffer_index < 0 || buffer_index >= static_cast<int>(mTrialBuffers.size())) {
        LOG_WARN("Invalid buffer index: " << buffer_index);
        return;
    }

    const auto& buffer = mTrialBuffers[buffer_index];

    // Load buffer data into current working data for plotting
    mAngleSweepData = buffer.angle_sweep_data;
    mStdAngleSweepData = buffer.std_angle_sweep_data;
    mAngleSweepTrackedMuscles = buffer.tracked_muscles;
    mStdAngleSweepTrackedMuscles = buffer.std_tracked_muscles;
    mCurrentSweepName = buffer.trial_name;

    // Apply normative pose to character and std character (only if DOF count matches)
    // Falls back to base_pose if normative_pose is not available
    const Eigen::VectorXd& pose_to_apply = (buffer.normative_pose.size() > 0) ? buffer.normative_pose : buffer.base_pose;
    if (pose_to_apply.size() > 0 && mCharacter) {
        auto skel = mCharacter->getSkeleton();
        if (pose_to_apply.size() == skel->getNumDofs()) {
            skel->setPositions(pose_to_apply);
            if (mStdCharacter) {
                auto stdSkel = mStdCharacter->getSkeleton();
                if (pose_to_apply.size() == stdSkel->getNumDofs()) {
                    stdSkel->setPositions(pose_to_apply);
                }
            }
        } else {
            LOG_WARN("Buffer pose DOF count (" << pose_to_apply.size()
                     << ") doesn't match skeleton (" << skel->getNumDofs() << "), skipping pose restore");
        }
    }

    // Initialize muscle visibility for any new muscles
    for (const auto& muscle : mAngleSweepTrackedMuscles) {
        if (mMuscleVisibility.find(muscle) == mMuscleVisibility.end()) {
            mMuscleVisibility[muscle] = true;
        }
    }

    mSelectedBufferIndex = buffer_index;
    LOG_INFO("Loaded buffer for visualization: " << buffer.trial_name);
}

void PhysicalExam::removeTrialBuffer(int buffer_index) {
    if (buffer_index < 0 || buffer_index >= static_cast<int>(mTrialBuffers.size())) {
        return;
    }

    std::string name = mTrialBuffers[buffer_index].trial_name;
    mTrialBuffers.erase(mTrialBuffers.begin() + buffer_index);

    // Adjust selected index
    if (mSelectedBufferIndex >= static_cast<int>(mTrialBuffers.size())) {
        mSelectedBufferIndex = static_cast<int>(mTrialBuffers.size()) - 1;
    }

    // If removed the selected one, load new selection or clear
    if (mSelectedBufferIndex >= 0) {
        loadBufferForVisualization(mSelectedBufferIndex);
    } else {
        mAngleSweepData.clear();
        mStdAngleSweepData.clear();
    }

    LOG_INFO("Removed trial buffer: " << name);
}

void PhysicalExam::clearTrialBuffers() {
    mTrialBuffers.clear();
    mSelectedBufferIndex = -1;
    mAngleSweepData.clear();
    mStdAngleSweepData.clear();
    mAngleSweepTrackedMuscles.clear();
    mStdAngleSweepTrackedMuscles.clear();
    LOG_INFO("All trial buffers cleared");
}

void PhysicalExam::runSelectedTrials() {
    if (!mExamSettingLoaded || !mCharacter) {
        LOG_ERROR("Cannot run trials: exam setting not loaded or character not available");
        return;
    }

    if (mTrialRunning) {
        LOG_WARN("Trials already running");
        return;
    }

    // Collect selected trial file paths
    std::vector<std::string> selected_paths;
    for (size_t i = 0; i < mTrialMultiSelectStates.size(); ++i) {
        if (mTrialMultiSelectStates[i] && i < mAvailableTrialFiles.size()) {
            selected_paths.push_back(mAvailableTrialFiles[i].file_path);
        }
    }

    if (selected_paths.empty()) {
        LOG_WARN("No trials selected");
        return;
    }

    LOG_INFO("Running " << selected_paths.size() << " selected trials");
    mTrialRunning = true;

    for (const auto& path : selected_paths) {
        try {
            std::string resolved_path = rm::resolve(path);
            YAML::Node trial_node = YAML::LoadFile(resolved_path);
            TrialConfig trial = parseTrialConfig(trial_node);

            // Clear current working data
            mTrials.clear();
            mTrials.push_back(trial);
            mCurrentTrialIndex = 0;
            mRecordedData.clear();

            // Run the trial (populates mAngleSweepData, mStdAngleSweepData)
            runCurrentTrial();

            // Create buffer from completed trial
            TrialDataBuffer buffer;
            buffer.trial_name = trial.name;
            buffer.trial_description = trial.description;
            buffer.alias = trial.angle_sweep.alias;
            // abd_knee: don't negate ROM angle for display (neg flag only affects cutoff direction)
            buffer.neg = (trial.angle_sweep.dof_type == "abd_knee") ? false : trial.angle_sweep.neg;
            buffer.timestamp = std::chrono::system_clock::now();
            buffer.angle_sweep_data = mAngleSweepData;
            buffer.std_angle_sweep_data = mStdAngleSweepData;
            buffer.tracked_muscles = mAngleSweepTrackedMuscles;
            buffer.std_tracked_muscles = mStdAngleSweepTrackedMuscles;
            buffer.config = trial.angle_sweep;
            buffer.torque_cutoff = trial.torque_cutoff;
            // Copy clinical data reference for ROM table rendering
            buffer.cd_side = trial.angle_sweep.cd_side;
            buffer.cd_joint = trial.angle_sweep.cd_joint;
            buffer.cd_field = trial.angle_sweep.cd_field;
            buffer.cd_neg = trial.angle_sweep.cd_neg;
            // Negate cutoff based on neg flag (neg:false → negative torque direction)
            double effective_cutoff = trial.angle_sweep.neg ? trial.torque_cutoff : -trial.torque_cutoff;
            // abd_knee has reversed torque direction - flip the cutoff sign
            if (trial.angle_sweep.dof_type == "abd_knee") {
                effective_cutoff = -effective_cutoff;
            }
            buffer.cutoff_angles = computeCutoffAngles(mAngleSweepData, effective_cutoff);
            buffer.rom_metrics = computeROMMetrics(mAngleSweepData);
            if (!mStdAngleSweepData.empty()) {
                buffer.std_rom_metrics = computeROMMetrics(mStdAngleSweepData);
            }
            buffer.base_pose = mCharacter->getSkeleton()->getPositions();

            // Capture normative pose: set skeleton to normative angle and capture
            // Strip side suffix ("/left" or "/right") to match mNormativeROM key format
            std::string normKey = buffer.alias;
            size_t lastSlash = normKey.rfind('/');
            if (lastSlash != std::string::npos) {
                normKey = normKey.substr(0, lastSlash);
            }
            LOG_INFO("[NormativePose] Searching for key: '" << normKey << "' (alias: '" << buffer.alias << "')");
            auto normIt = mNormativeROM.find(normKey);
            if (normIt != mNormativeROM.end()) {
                double normative_deg = normIt->second;
                double normative_rad = normative_deg * M_PI / 180.0;
                LOG_INFO("[NormativePose] Found normative: " << normative_deg << " deg, neg=" << trial.angle_sweep.neg);

                // Apply neg flag: if neg=true, negate the angle for internal representation
                if (trial.angle_sweep.neg) {
                    normative_rad = -normative_rad;
                    LOG_INFO("[NormativePose] After neg applied: " << (normative_rad * 180.0 / M_PI) << " deg");
                }

                auto skel = mCharacter->getSkeleton();
                auto joint = skel->getJoint(trial.angle_sweep.joint_name);
                if (joint) {
                    int dof_idx = joint->getIndexInSkeleton(trial.angle_sweep.dof_index);
                    Eigen::VectorXd pos = skel->getPositions();
                    double before_angle = pos[dof_idx];
                    pos[dof_idx] = normative_rad;
                    skel->setPositions(pos);
                    buffer.normative_pose = skel->getPositions();
                    LOG_INFO("[NormativePose] Joint=" << trial.angle_sweep.joint_name
                             << ", dof_idx=" << dof_idx
                             << ", before=" << (before_angle * 180.0 / M_PI) << " deg"
                             << ", after=" << (normative_rad * 180.0 / M_PI) << " deg");

                    // Restore base pose
                    skel->setPositions(buffer.base_pose);
                }
            } else {
                LOG_WARN("[NormativePose] Key '" << normKey << "' not found in mNormativeROM");
            }

            // Add to buffer with limit enforcement
            addTrialToBuffer(buffer);

            LOG_VERBOSE("Trial '" << trial.name << "' completed and buffered");

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to run trial from " << path << ": " << e.what());
        }
    }

    mTrialRunning = false;

    // Auto-select the last run buffer if enabled
    if (mAutoSelectNewBuffer && !mTrialBuffers.empty()) {
        mSelectedBufferIndex = static_cast<int>(mTrialBuffers.size()) - 1;
        loadBufferForVisualization(mSelectedBufferIndex);
    }

    LOG_INFO("All selected trials completed. " << mTrialBuffers.size() << " buffers available");
}

// ============================================================================
// Character Loading (Browse & Rebuild) Section Methods
// ============================================================================

void PhysicalExam::drawClinicalDataSection() {
    if (collapsingHeaderWithControls("Clinical Data")) {
        if (!mPIDNavigator) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "PID Navigator not available");
            return;
        }

        // PID Navigator selector (PID list only, no file sections)
        // PID changes are handled by onBrowsePIDChanged callback
        mPIDNavigator->renderUI(nullptr, 120, 0);

        // Handle deselection (e.g., after PID list refresh)
        const auto& pidState = mPIDNavigator->getState();
        if (pidState.selectedPID < 0 && !mBrowseCharacterPID.empty()) {
            mBrowseCharacterPID.clear();
            mClinicalWeightAvailable = false;
            mClinicalROM.clear();
            mClinicalROMPID.clear();
            mClinicalROMVisit.clear();
        }
    }
}

void PhysicalExam::drawCharacterLoadSection() {
    if (collapsingHeaderWithControls("Character Loading")) {
        // ============ SKELETON SECTION ============
        bool skelChanged = false;
        ImGui::Text("Skeleton Source:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Default##skel", mBrowseSkeletonDataSource == CharacterDataSource::DefaultData)) {
            if (mBrowseSkeletonDataSource != CharacterDataSource::DefaultData) {
                mBrowseSkeletonDataSource = CharacterDataSource::DefaultData;
                skelChanged = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Patient##skel", mBrowseSkeletonDataSource == CharacterDataSource::PatientData)) {
            if (mBrowseSkeletonDataSource != CharacterDataSource::PatientData) {
                mBrowseSkeletonDataSource = CharacterDataSource::PatientData;
                skelChanged = true;
            }
        }

        if (skelChanged) {
            scanSkeletonFilesForBrowse();
            mBrowseSkeletonPath.clear();
        }

        // Skeleton file list
        std::string skelPrefix = "@data/skeleton/";
        if (mBrowseSkeletonDataSource == CharacterDataSource::PatientData && !mBrowseCharacterPID.empty()) {
            std::string visit = mPIDNavigator->getState().getVisitDir();
            skelPrefix = "@pid:" + mBrowseCharacterPID + "/" + visit + "/skeleton/";
        }

        if (ImGui::BeginListBox("##SkeletonList", ImVec2(-1, 60))) {
            for (size_t i = 0; i < mBrowseSkeletonCandidates.size(); ++i) {
                const auto& filename = mBrowseSkeletonCandidates[i];
                bool isSelected = (mBrowseSkeletonPath.find(filename) != std::string::npos);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
                    mBrowseSkeletonPath = skelPrefix + filename;
                }
            }
            ImGui::EndListBox();
        }

        ImGui::Spacing();

        // ============ MUSCLE SECTION ============
        bool muscleChanged = false;
        ImGui::Text("Muscle Source:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Default##muscle", mBrowseMuscleDataSource == CharacterDataSource::DefaultData)) {
            if (mBrowseMuscleDataSource != CharacterDataSource::DefaultData) {
                mBrowseMuscleDataSource = CharacterDataSource::DefaultData;
                muscleChanged = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Patient##muscle", mBrowseMuscleDataSource == CharacterDataSource::PatientData)) {
            if (mBrowseMuscleDataSource != CharacterDataSource::PatientData) {
                mBrowseMuscleDataSource = CharacterDataSource::PatientData;
                muscleChanged = true;
            }
        }

        if (muscleChanged) {
            scanMuscleFilesForBrowse();
            mBrowseMusclePath.clear();
        }

        // Muscle file list
        std::string musclePrefix = "@data/muscle/";
        if (mBrowseMuscleDataSource == CharacterDataSource::PatientData && !mBrowseCharacterPID.empty()) {
            std::string visit = mPIDNavigator->getState().getVisitDir();
            musclePrefix = "@pid:" + mBrowseCharacterPID + "/" + visit + "/muscle/";
        }

        if (ImGui::BeginListBox("##MuscleList", ImVec2(-1, 60))) {
            for (size_t i = 0; i < mBrowseMuscleCandidates.size(); ++i) {
                const auto& filename = mBrowseMuscleCandidates[i];
                bool isSelected = (mBrowseMusclePath.find(filename) != std::string::npos);
                if (ImGui::Selectable(filename.c_str(), isSelected)) {
                    mBrowseMusclePath = musclePrefix + filename;
                }
            }
            ImGui::EndListBox();
        }

        ImGui::Separator();

        // Rebuild button
        bool canRebuild = !mBrowseSkeletonPath.empty() && !mBrowseMusclePath.empty();
        if (!canRebuild) ImGui::BeginDisabled();
        if (ImGui::Button("Rebuild", ImVec2(-1, 0))) {
            reloadCharacterFromBrowse();
        }
        if (!canRebuild) ImGui::EndDisabled();
    }
}

void PhysicalExam::scanSkeletonFilesForBrowse() {
    namespace fs = std::filesystem;
    mBrowseSkeletonCandidates.clear();

    try {
        fs::path skelDir;
        if (mBrowseSkeletonDataSource == CharacterDataSource::PatientData && !mBrowseCharacterPID.empty()) {
            // Use patient data directory: {pid_root}/{pid}/{visit}/skeleton/
            fs::path pidRoot = rm::getManager().getPidRoot();
            if (!pidRoot.empty()) {
                std::string visit = mPIDNavigator->getState().getVisitDir();
                skelDir = pidRoot / mBrowseCharacterPID / visit / "skeleton";

                // Create directory if it doesn't exist
                if (!fs::exists(skelDir)) {
                    LOG_INFO("[PhysicalExam] Creating skeleton directory: " << skelDir);
                    fs::create_directories(skelDir);
                }
            } else {
                LOG_WARN("[PhysicalExam] PID root not available");
                return;
            }
        } else {
            // Use default data directory
            skelDir = rm::getManager().resolveDir("@data/skeleton");
            if (!fs::exists(skelDir)) return;
        }

        for (const auto& entry : fs::directory_iterator(skelDir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".yaml" || ext == ".xml") {
                mBrowseSkeletonCandidates.push_back(entry.path().filename().string());
            }
        }
        std::sort(mBrowseSkeletonCandidates.begin(), mBrowseSkeletonCandidates.end());
    }
    catch (const std::exception& e) {
        LOG_WARN("[PhysicalExam] Error scanning skeleton files: " << e.what());
    }
}

void PhysicalExam::scanMuscleFilesForBrowse() {
    namespace fs = std::filesystem;
    mBrowseMuscleCandidates.clear();

    try {
        fs::path muscleDir;
        if (mBrowseMuscleDataSource == CharacterDataSource::PatientData && !mBrowseCharacterPID.empty()) {
            // Use patient data directory: {pid_root}/{pid}/{visit}/muscle/
            fs::path pidRoot = rm::getManager().getPidRoot();
            if (!pidRoot.empty()) {
                std::string visit = mPIDNavigator->getState().getVisitDir();
                muscleDir = pidRoot / mBrowseCharacterPID / visit / "muscle";

                // Create directory if it doesn't exist
                if (!fs::exists(muscleDir)) {
                    LOG_INFO("[PhysicalExam] Creating muscle directory: " << muscleDir);
                    fs::create_directories(muscleDir);
                }
            } else {
                LOG_WARN("[PhysicalExam] PID root not available");
                return;
            }
        } else {
            // Use default data directory
            muscleDir = rm::getManager().resolveDir("@data/muscle");
            if (!fs::exists(muscleDir)) return;
        }

        for (const auto& entry : fs::directory_iterator(muscleDir)) {
            auto ext = entry.path().extension().string();
            if (ext == ".yaml" || ext == ".xml") {
                mBrowseMuscleCandidates.push_back(entry.path().filename().string());
            }
        }
        std::sort(mBrowseMuscleCandidates.begin(), mBrowseMuscleCandidates.end());
    }
    catch (const std::exception& e) {
        LOG_WARN("[PhysicalExam] Error scanning muscle files: " << e.what());
    }
}

void PhysicalExam::reloadCharacterFromBrowse() {
    if (mBrowseSkeletonPath.empty() || mBrowseMusclePath.empty()) {
        LOG_WARN("[PhysicalExam] Cannot reload: skeleton or muscle path not selected");
        return;
    }

    LOG_INFO("[PhysicalExam] Reloading character:");
    LOG_INFO("  Skeleton: " << mBrowseSkeletonPath);
    LOG_INFO("  Muscle: " << mBrowseMusclePath);

    // Call the existing loadCharacter method
    loadCharacter(mBrowseSkeletonPath, mBrowseMusclePath, ActuatorType::tor);

    // Reinitialize sweep muscles for the new character
    setupSweepMuscles();

    LOG_INFO("[PhysicalExam] Character reloaded successfully");
}

// ============================================================================
// Control Panel Section Methods
// ============================================================================

void PhysicalExam::drawPosePresetsSection() {
    if (collapsingHeaderWithControls("Pose Presets")) {
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
    if (collapsingHeaderWithControls("Force Application")) {
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
    if (collapsingHeaderWithControls("Print Info")) {
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
    if (collapsingHeaderWithControls("Recording")) {
        ImGui::Text("Recorded Data Points: %zu", mRecordedData.size());

        if (ImGui::Button("Record Current State")) {
            if (mCharacter) {
                ROMDataPoint data;
                data.force_magnitude = mForceMagnitude;
                std::vector<std::string> joints = {"FemurR", "TibiaR", "TalusR"};
                data.joint_angles = recordJointAngles(joints);
                data.passive_force_total = 0.0;  // Deprecated: use getPassiveTorqueJoint() for angle sweep
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
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Render");
    ImGui::Separator();
    // Character rendering selection
        if (mStdCharacter) {
            ImGui::Text("Character Display:");
            int render_option = 0;
            if (mRenderMainCharacter && mRenderStdCharacter) render_option = 2;
            else if (mRenderStdCharacter) render_option = 1;
            else render_option = 0;
            
            if (ImGui::RadioButton("Main##Render", &render_option, 0)) {
                mRenderMainCharacter = true;
                mRenderStdCharacter = false;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Std##Render", &render_option, 1)) {
                mRenderMainCharacter = false;
                mRenderStdCharacter = true;
            }
        } else {
            // No standard character loaded - just show status
            ImGui::Text("Rendering: Main Character");
        }
        
        ImGui::Separator();

        // Plot background color and trial name display
        ImGui::Checkbox("White Plot", &mPlotWhiteBackground);
        ImGui::SameLine(200);
        ImGui::Checkbox("Show Trial Name", &mShowTrialNameInPlots);

        ImGui::Separator();
        
        // Render Mode section
        ImGui::Text("Skeleton Render Mode (O):");
        int mode = static_cast<int>(mRenderMode);
        if (ImGui::RadioButton("Primitive", &mode, 0)) mRenderMode = RenderMode::Primitive;
        ImGui::SameLine();
        if (ImGui::RadioButton("Mesh", &mode, 1)) mRenderMode = RenderMode::Mesh;
        ImGui::SameLine();
        if (ImGui::RadioButton("Wire", &mode, 2)) mRenderMode = RenderMode::Wireframe;
        ImGui::Separator();

        // Muscle Render Mode section
        ImGui::Text("Muscle Color Mode:");
        ImGui::RadioButton("Passive Force", &mMuscleColorMode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Norm. Length", &mMuscleColorMode, 1);

        // Mode-specific controls
        if (mMuscleColorMode == 0) {
            ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("##ForceNorm", &mPassiveForceNormalizer, 0, 0, "%.0f");
            ImGui::SameLine();
            ImGui::Text("Force Norm");
        } else {
            ImGui::SetNextItemWidth(50);
            ImGui::InputFloat("##LmMin", &mLmNormMin, 0, 0, "%.2f");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            ImGui::InputFloat("##LmMax", &mLmNormMax, 0, 0, "%.2f");
            ImGui::SameLine();
            ImGui::Text("Lm Range");

            // Viridis colorbar
            auto viridisColor = [](float t) -> ImU32 {
                auto c = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis);
                return IM_COL32((int)(c.r()*255), (int)(c.g()*255), (int)(c.b()*255), 255);
            };

            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            float barWidth = 150.0f;
            float barHeight = 12.0f;
            int segments = 32;

            for (int i = 0; i < segments; i++) {
                float t0 = (float)i / segments;
                float t1 = (float)(i + 1) / segments;
                float x0 = pos.x + t0 * barWidth;
                float x1 = pos.x + t1 * barWidth;
                drawList->AddRectFilledMultiColor(
                    ImVec2(x0, pos.y), ImVec2(x1, pos.y + barHeight),
                    viridisColor(t0), viridisColor(t1), viridisColor(t1), viridisColor(t0));
            }
            drawList->AddRect(ImVec2(pos.x, pos.y), ImVec2(pos.x + barWidth, pos.y + barHeight), IM_COL32(128, 128, 128, 255));

            // Labels
            ImGui::Dummy(ImVec2(barWidth, barHeight + 2));
            ImGui::Text("%.2f", mLmNormMin);
            ImGui::SameLine(barWidth - 25);
            ImGui::Text("%.2f", mLmNormMax);
        }
        // Common controls on same line
        ImGui::SetNextItemWidth(40);
        ImGui::InputFloat("##LineWidth", &mMuscleLineWidth, 0, 0, "%.1f");
        ImGui::SameLine();
        ImGui::Text("Width");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("##Alpha", &mMuscleTransparency, 0.1f, 1.0f, "%.2f");
        ImGui::SameLine();
        ImGui::Text("Alpha");

        ImGui::Checkbox("Show Anchor Points", &mShowAnchorPoints);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Display muscle anchor points as dots instead of lines");
        }
        ImGui::Separator();

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
        ImGui::Separator();

        // Muscle Filtering and Selection
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Muscle Filter");
        ImGui::Separator();
        ImGui::Indent();
        if (!mCharacter) {
                ImGui::TextDisabled("Load character first");
            } else {
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
        ImGui::Unindent();

        ImGui::Checkbox("Posture Control Debug", &mShowPostureDebug);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Print posture control debug info to console (positions, errors, forces)");
        }
}

void PhysicalExam::drawJointControlSection() {
    if (collapsingHeaderWithControls("Joint Angle")) {
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

            // abd_knee composite DOF control box
            if (ImGui::BeginChild("AbdKneeBox", ImVec2(0, 100), true)) {
                ImGui::Text("Abd Knee IK");

                static float abd_knee_shank_ratio = 0.5f;
                static float abd_knee_angle = 0.0f;  // degrees
                static int abd_knee_side = 0;  // 0=Left, 1=Right

                // Row 1: Shank ratio and Knee angle
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Shank Ratio", &abd_knee_shank_ratio, 0.0f, 0.0f, "%.2f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Angle (deg)", &abd_knee_angle, 0.0f, 0.0f, "%.1f");

                // Row 2: Left/Right radio buttons and Apply
                ImGui::RadioButton("Left", &abd_knee_side, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Right", &abd_knee_side, 1);
                ImGui::SameLine();

                if (ImGui::Button("Apply")) {
                    bool is_left = (abd_knee_side == 0);
                    std::string hip_name = is_left ? "FemurL" : "FemurR";
                    std::string knee_name = is_left ? "TibiaL" : "TibiaR";

                    auto hip_joint = skel->getJoint(hip_name);
                    auto knee_joint = skel->getJoint(knee_name);

                    if (hip_joint && knee_joint) {
                        int hip_idx = static_cast<int>(skel->getIndexOf(hip_joint));
                        auto ik_result = PMuscle::ContractureOptimizer::computeAbdKneePose(
                            skel, hip_idx, abd_knee_angle, is_left, abd_knee_shank_ratio);

                        if (ik_result.success) {
                            Eigen::VectorXd pos = skel->getPositions();
                            int hip_dof_start = static_cast<int>(hip_joint->getIndexInSkeleton(0));
                            int knee_dof_idx = static_cast<int>(knee_joint->getIndexInSkeleton(0));

                            pos.segment<3>(hip_dof_start) = ik_result.hip_positions;
                            pos[knee_dof_idx] = ik_result.knee_angle;
                            skel->setPositions(pos);

                            // Update muscle geometry
                            if (!mCharacter->getMuscles().empty()) {
                                mCharacter->getMuscleTuple();
                            }
                        } else {
                            LOG_WARN("abd_knee IK failed at angle " << abd_knee_angle << "°");
                        }
                    }
                }
            }
            ImGui::EndChild();

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
                setCharacterPose(pos_rad.cast<double>());
            }
            // With interpolation enabled, targets are set when sliders change (marked automatically)
        }
    }
}

void PhysicalExam::drawJointAngleSweepSection() {
    if (collapsingHeaderWithControls("Joint Angle Sweep")) {
        if (!mCharacter) {
            ImGui::TextDisabled("Load character first");
        } else {
            auto skel = mCharacter->getSkeleton();

            // Joint selection
            ImGui::Text("Sweep Joint:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            int joint_idx = mSweepConfig.joint_index;
            if (ImGui::InputInt("##JointIdx", &joint_idx, 1, 1)) {
                if (joint_idx >= 0 && joint_idx < static_cast<int>(skel->getNumJoints())) {
                    mSweepConfig.joint_index = joint_idx;
                    mSweepConfig.dof_index = 0;  // Reset DOF index when joint changes

                    // Auto-update angle range from joint limits
                    auto joint = skel->getJoint(joint_idx);
                    if (joint->getNumDofs() > 0) {
                        Eigen::VectorXd pos_lower = skel->getPositionLowerLimits();
                        Eigen::VectorXd pos_upper = skel->getPositionUpperLimits();

                        int global_dof_idx = 0;
                        for (size_t j = 0; j < joint_idx; ++j) {
                            global_dof_idx += skel->getJoint(j)->getNumDofs();
                        }
                        global_dof_idx += mSweepConfig.dof_index;

                        mSweepConfig.angle_min = pos_lower[global_dof_idx];
                        mSweepConfig.angle_max = pos_upper[global_dof_idx];
                    }
                }
            }
            ImGui::SameLine();
            auto current_joint = skel->getJoint(mSweepConfig.joint_index);
            int num_dofs = current_joint->getNumDofs();
            ImGui::Text("%s", current_joint->getName().c_str());
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::SliderInt("##SweepDof", &mSweepConfig.dof_index, 0, num_dofs - 1)) {
                // Update limits when DOF changes
                Eigen::VectorXd pos_lower = skel->getPositionLowerLimits();
                Eigen::VectorXd pos_upper = skel->getPositionUpperLimits();

                int global_dof_idx = 0;
                for (size_t j = 0; j < mSweepConfig.joint_index; ++j) {
                    global_dof_idx += skel->getJoint(j)->getNumDofs();
                }
                global_dof_idx += mSweepConfig.dof_index;

                mSweepConfig.angle_min = pos_lower[global_dof_idx];
                mSweepConfig.angle_max = pos_upper[global_dof_idx];
            }
            ImGui::SameLine();
            ImGui::Text("/%d", num_dofs);

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

                    // Get global DOF index for this joint's selected DOF
                    int global_dof_idx = 0;
                    for (size_t j = 0; j < mSweepConfig.joint_index; ++j) {
                        global_dof_idx += skel->getJoint(j)->getNumDofs();
                    }
                    global_dof_idx += mSweepConfig.dof_index;

                    mSweepConfig.angle_min = pos_lower[global_dof_idx];
                    mSweepConfig.angle_max = pos_upper[global_dof_idx];
                }
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset angle range to joint's position limits");
            }
            
            ImGui::SameLine();
            // Number of steps
            ImGui::SetNextItemWidth(50);
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
            if (!mAngleSweepData.empty()) {
                ImGui::Text("Data points: %zu", mAngleSweepData.size());
                ImGui::Text("Tracked muscles: %zu", mAngleSweepTrackedMuscles.size());
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
    if (collapsingHeaderWithControls("Trial Management")) {
        if (mExamSettingLoaded) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Exam: %s", mExamName.c_str());
            if (!mExamDescription.empty()) {
                ImGui::TextWrapped("%s", mExamDescription.c_str());
            }
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No exam setting loaded");
            ImGui::TextWrapped("Load an exam setting config to run trials");
        }

        ImGui::Separator();

        // Build trial names list for FilterableChecklist
        if (mAvailableTrialFiles.empty()) {
            ImGui::TextDisabled("No trial files found in data/config/rom");
        } else {
            std::vector<std::string> trialNames;
            trialNames.reserve(mAvailableTrialFiles.size());
            for (const auto& file : mAvailableTrialFiles) {
                trialNames.push_back(file.name);
            }

            // Use FilterableChecklist for trial selection
            static char trialFilterBuffer[256] = "";
            ImGuiCommon::FilterableChecklist(
                "##TrialSelect",
                trialNames,
                mTrialMultiSelectStates,
                trialFilterBuffer,
                sizeof(trialFilterBuffer),
                150.0f
            );

            ImGui::Separator();

            // Run Selected Trials button
            int selectedCount = static_cast<int>(std::count(mTrialMultiSelectStates.begin(),
                                          mTrialMultiSelectStates.end(), true));

            bool canRunTrials = (selectedCount > 0 && mExamSettingLoaded && !mTrialRunning);

            if (!canRunTrials) ImGui::BeginDisabled();

            std::string btnLabel = "Run " + std::to_string(selectedCount) + " Trial(s)";
            if (ImGui::Button(btnLabel.c_str(), ImVec2(120, 0))) {
                clearTrialBuffers();
                runSelectedTrials();
            }

            if (!canRunTrials) {
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered()) {
                    if (selectedCount == 0) {
                        ImGui::SetTooltip("Select at least one trial");
                    } else if (!mExamSettingLoaded) {
                        ImGui::SetTooltip("Load an exam setting first");
                    } else if (mTrialRunning) {
                        ImGui::SetTooltip("Trials are currently running");
                    }
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset Trials", ImVec2(120, 0))) {
                mCurrentTrialIndex = -1;
                mTrialRunning = false;
                mRecordedData.clear();
                mTrials.clear();
                LOG_INFO("Trials reset");
            }

            // Verbose torque checkbox
            ImGui::SameLine();
            ImGui::Checkbox("Verbose", &mVerboseTorque);
            ImGui::SameLine();
            ImGui::Checkbox("Visualize", &mVisualizeSweep);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Press N to advance to next angle during sweep");
            }

            // Show current trial status if running
            if (mTrialRunning && mCurrentTrialIndex >= 0 &&
                mCurrentTrialIndex < static_cast<int>(mTrials.size())) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Running: %s",
                                  mTrials[mCurrentTrialIndex].name.c_str());
            }
        }

        // Buffer management section
        ImGui::Separator();
        ImGui::Text("Trial Buffers: %zu / %d", mTrialBuffers.size(), mMaxTrialBuffers);

        if (!mTrialBuffers.empty()) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear Buffers")) {
                clearTrialBuffers();
            }
        }

        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("Max Buffers", &mMaxTrialBuffers);
        mMaxTrialBuffers = std::clamp(mMaxTrialBuffers, 1, 100);
    }
}

void PhysicalExam::drawCurrentStateSection() {
    if (collapsingHeaderWithControls("State")) {
        if (mCharacter) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Character Loaded");
            ImGui::Text("Skeleton DOFs: %zu", mCharacter->getSkeleton()->getNumDofs());
            ImGui::Text("Body Nodes: %zu", mCharacter->getSkeleton()->getNumBodyNodes());
            if (!mCharacter->getMuscles().empty()) {
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
            if (!mCharacter->getMuscles().empty()) {
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

                // Configurable top N passive forces
                ImGui::SetNextItemWidth(30);
                ImGui::InputInt("## of Top Passive Forces", &mTopPassiveForcesCount, 0, 0, ImGuiInputTextFlags_CharsDecimal);
                if (mTopPassiveForcesCount < 1) mTopPassiveForcesCount = 1;
                if (mTopPassiveForcesCount > (int)muscle_forces.size()) mTopPassiveForcesCount = (int)muscle_forces.size();
                ImGui::SameLine();
                ImGui::Text("  Top %d Passive Forces:", mTopPassiveForcesCount);
                for (int i = 0; i < std::min(mTopPassiveForcesCount, (int)muscle_forces.size()); i++) {
                    ImGui::Text("    %d. %s: %.2f N", i+1,
                               muscle_forces[i].second.c_str(),
                               muscle_forces[i].first);
                }

                ImGui::Separator();

                // JTP listbox for debugging - click DOF to print JTP
                ImGui::Text("JTP Debug (click to print):");
                {
                    auto skel = mCharacter->getSkeleton();
                    ImGui::BeginChild("JTPList", ImVec2(0, 150), true);

                    // Femur Y-axis projection entries (for abd_knee composite DOF)
                    for (const char* femur_name : {"FemurL", "FemurR"}) {
                        auto* femur_joint = skel->getJoint(femur_name);
                        if (femur_joint && femur_joint->getNumDofs() == 3) {
                            std::string label = std::string(femur_name) + " (Y-axis)";
                            if (ImGui::Selectable(label.c_str())) {
                                auto* child_body = femur_joint->getChildBodyNode();
                                Eigen::Vector3d joint_center = child_body->getTransform().translation();

                                // Build set of descendant bodies
                                std::set<dart::dynamics::BodyNode*> descendant_bodies;
                                std::function<void(dart::dynamics::BodyNode*)> collect_descendants;
                                collect_descendants = [&](dart::dynamics::BodyNode* bn) {
                                    descendant_bodies.insert(bn);
                                    for (size_t i = 0; i < bn->getNumChildBodyNodes(); ++i) {
                                        collect_descendants(bn->getChildBodyNode(i));
                                    }
                                };
                                collect_descendants(child_body);

                                double total_torque = 0.0;
                                std::cout << "\n[JTP Y-axis] " << femur_name
                                          << " joint_center=[" << joint_center.transpose() << "]"
                                          << " descendant_bodies=" << descendant_bodies.size() << std::endl;
                                for (auto& muscle : mCharacter->getMuscles()) {
                                    Eigen::Vector3d torque_world = muscle->GetPassiveTorqueAboutPoint(
                                        joint_center, &descendant_bodies);
                                    double contribution = torque_world.y();
                                    if (std::abs(contribution) > 1e-6) {
                                        std::cout << "    " << muscle->GetName()
                                                  << " Y-torque=" << std::fixed << std::setprecision(2)
                                                  << contribution << " Nm" << std::endl;
                                    }
                                    total_torque += contribution;
                                }
                                std::cout << "  Total Y-axis JTP: " << -total_torque << " Nm" << std::endl;
                            }
                        }
                    }
                    ImGui::Separator();

                    for (size_t j = 0; j < skel->getNumJoints(); ++j) {
                        auto joint = skel->getJoint(j);
                        // Skip pelvis (root joint)
                        if (joint->getName().find("Pelvis") != std::string::npos ||
                            joint->getName().find("pelvis") != std::string::npos) {
                            continue;
                        }
                        for (size_t d = 0; d < joint->getNumDofs(); ++d) {
                            int dof_idx = joint->getIndexInSkeleton(d);
                            std::string label = joint->getName() + " [" + std::to_string(dof_idx) + "]";
                            if (ImGui::Selectable(label.c_str())) {
                                double total_jtp = 0.0;
                                std::cout << "\n[JTP] " << joint->getName() << " DOF " << dof_idx << ":" << std::endl;
                                for (auto& muscle : muscles) {
                                    Eigen::VectorXd jtp = muscle->GetRelatedJtp();
                                    const auto& related_indices = muscle->related_dof_indices;
                                    for (size_t i = 0; i < related_indices.size(); ++i) {
                                        if (related_indices[i] == dof_idx) {
                                            if (std::abs(jtp[i]) > 1e-6) {
                                                std::cout << "    " << muscle->GetName()
                                                          << " jtp=" << std::fixed << std::setprecision(2)
                                                          << jtp[i] << " Nm" << std::endl;
                                            }
                                            total_jtp += jtp[i];
                                            break;
                                        }
                                    }
                                }
                                std::cout << "  Total JTP: " << total_jtp << " Nm" << std::endl;
                            }
                        }
                    }
                    ImGui::EndChild();
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

void PhysicalExam::drawCameraStatusSection() {
    if (collapsingHeaderWithControls("Camera Status")) {
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

        ImGui::Text("Eye: [%.3f, %.3f, %.3f]", mCamera.eye[0], mCamera.eye[1], mCamera.eye[2]);
        ImGui::Text("Up:  [%.3f, %.3f, %.3f]", mCamera.up[0], mCamera.up[1], mCamera.up[2]);
        ImGui::Text("Trans: [%.3f, %.3f, %.3f]", mCamera.trans[0], mCamera.trans[1], mCamera.trans[2]);
        ImGui::Text("Zoom: %.3f", mCamera.zoom);

        Eigen::Quaterniond quat = mCamera.trackball.getCurrQuat();
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

void PhysicalExam::drawMuscleInfoSection() {
    if (collapsingHeaderWithControls("Muscle Information")) {
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
                ImGui::Text("lm_opt:"); ImGui::NextColumn(); ImGui::Text("%.4f", selectedMuscle->lm_contract); ImGui::NextColumn();
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

bool PhysicalExam::collapsingHeaderWithControls(const std::string& title)
{
    bool isDefaultOpen = isPanelDefaultOpen(title);  // inherited from ViewerAppBase
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen * isDefaultOpen;
    return ImGui::CollapsingHeader(title.c_str(), flags);
}

// isPanelDefaultOpen() is inherited from ViewerAppBase

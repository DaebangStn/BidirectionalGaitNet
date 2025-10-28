#include "GLFWApp.h"
#include "UriResolver.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <filesystem>
#include <algorithm>
#include "DARTHelper.h"
#include <tinyxml2.h>
#include <sstream>
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "Motion.h"
#include "NPZ.h"
#include "HDF.h"
#include "Log.h"

namespace fs = std::filesystem;


const std::vector<std::string> CHANNELS =
    {
        "Xposition",
        "Yposition",
        "Zposition",
        "Xrotation",
        "Yrotation",
        "Zrotation",
};

const char* CAMERA_PRESET_DEFINITIONS[] = {
    "PRESET|Frontal view|0,0,3.0|0,1,0|0,0,0|1|1,0,0,0",
    "PRESET|Sagittal view|0,0,3.0|0,1,0|0,0,0|1|0.707,0.0,0.707,0.0",
    "PRESET|Foot view|0,0,1.26824|0,1,0|0,900,15|1|0.707,0.0,0.707,0.0",
};

GLFWApp::GLFWApp(int argc, char **argv)
{
    mRenderEnv = nullptr;
    mMotionCharacter = nullptr;
    mGVAELoaded = false;
    mRenderConditions = false;

    // Set default values before loading config
    mWidth = 2560;
    mHeight = 1440;
    mWindowXPos = 0;
    mWindowYPos = 0;
    mControlPanelWidth = 450;
    mPlotPanelWidth = 450;
    mDefaultRolloutCount = 10;
    mXmin = 0.0;
    mXminResizablePlotPane = 0.0;
    mYminResizablePlotPane = 0.0;
    mYmaxResizablePlotPane = 1.0;
    mPlotTitle = false;

    // Initialize viewer time management with default cycle duration
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mViewerPlaybackSpeed = 1.5;
    mLastPlaybackSpeed = 1.5;
    mViewerCycleDuration = 2.0 / 1.1;  // Default cycle duration (~1.818s)
    mLastRealTime = 0.0;
    mSimulationStepDuration = 0.0;
    mSimStepDurationAvg = -1.0;
    mRealDeltaTimeAvg = 0.0;
    mIsPlaybackTooFast = false;
    mShowTimingPane = false;
    mShowResizablePlotPane = false;
    mResetPhase = -1.0;  // Default to randomized reset
    mResizablePlots.resize(1);
    strcpy(mResizePlotKeys, "");
    mResizePlotPane = true;
    mSetResizablePlotPane = false;
    mPlotTitleResizablePlotPane = false;

    // Initialize motion navigation control
    mMotionNavigationMode = NAVIGATION_VIEWER_TIME;
    mManualFrameIndex = 0;
    mMaxFrameIndex = 0;

    // Load configuration from render.yaml (will override defaults if file exists)
    loadRenderConfig();
    updateResizablePlotsFromKeys();

    mZoom = 1.0;
    mPersp = 45.0;
    mMouseDown = false;
    mRotate = false;
    mTranslate = false;
    mZooming = false;
    mTrans = Eigen::Vector3d(0.0, 0.0, 0.0);
    mRelTrans = Eigen::Vector3d(0.0, 0.0, 0.0);  // Initialize user translation offset
    mEye = Eigen::Vector3d(0.0, 0.0, 1.0);
    mUp = Eigen::Vector3d(0.0, 1.0, 0.0);
    mDrawOBJ = true;

    // Rendering Option
    mDrawReferenceSkeleton = false;
    mDrawCharacter = true;
    mDrawPDTarget = false;
    mDrawJointSphere = false;
    mStochasticPolicy = false;
    mDrawFootStep = false;
    mDrawEOE = false;

    mMuscleRenderType = activationLevel;
    mMuscleRenderTypeInt = 2;
    mMuscleResolution = 0.0;

    // Muscle Selection UI
    std::memset(mMuscleFilterText, 0, sizeof(mMuscleFilterText));
    // Note: mMuscleSelectionStates will be initialized in initEnv when muscles are available

    // Initialize Graph Data Buffer
    mGraphData = new CBufferData<double>();

    // Register reward keys (for both deepmimic and gaitnet reward types)
    mGraphData->register_key("r", 500);
    mGraphData->register_key("r_p", 500);
    mGraphData->register_key("r_v", 500);
    mGraphData->register_key("r_com", 500);
    mGraphData->register_key("r_ee", 500);
    mGraphData->register_key("r_metabolic", 500);
    mGraphData->register_key("r_torque", 500);
    mGraphData->register_key("r_energy", 500);
    mGraphData->register_key("r_knee_pain", 500);
    mGraphData->register_key("r_loco", 500);
    mGraphData->register_key("r_avg", 500);
    mGraphData->register_key("r_step", 500);

    // Register contact keys
    mGraphData->register_key("contact_left", 500);
    mGraphData->register_key("contact_right", 500);
    mGraphData->register_key("contact_phaseR", 1000);
    mGraphData->register_key("grf_left", 500);
    mGraphData->register_key("grf_right", 500);

    // Register kinematic keys
    // mGraphData->register_key("sway_Torso_X", 500);
    mGraphData->register_key("angle_HipR", 1000);
    mGraphData->register_key("angle_HipIRR", 1000);
    mGraphData->register_key("angle_HipAbR", 1000);
    mGraphData->register_key("angle_KneeR", 1000);
    mGraphData->register_key("angle_AnkleR", 1000);
    mGraphData->register_key("angle_Rotation", 1000);
    mGraphData->register_key("angle_Obliquity", 1000);
    mGraphData->register_key("angle_Tilt", 1000);
    
    mGraphData->register_key("energy_metabolic_step", 1000);
    mGraphData->register_key("energy_metabolic", 1000);
    mGraphData->register_key("energy_torque_step", 1000);
    mGraphData->register_key("energy_torque", 1000);
    mGraphData->register_key("energy_combined", 1000);

    // Register knee loading key (max value only)
    mGraphData->register_key("knee_loading_max", 1000);

    // Register joint loading keys (hip, knee, ankle)
    std::vector<std::string> joints = {"hip", "knee", "ankle"};
    for (const auto& joint : joints) {
        mGraphData->register_key(joint + "_force_x", 1000);
        mGraphData->register_key(joint + "_force_y", 1000);
        mGraphData->register_key(joint + "_force_z", 1000);
        mGraphData->register_key(joint + "_force_mag", 1000);
        mGraphData->register_key(joint + "_torque_x", 1000);
        mGraphData->register_key(joint + "_torque_y", 1000);
        mGraphData->register_key(joint + "_torque_z", 1000);
        mGraphData->register_key(joint + "_torque_mag", 1000);
    }

    // Forward GaitNEt
    selected_fgn = 0;
    mDrawFGNSkeleton = false;

    // Backward GaitNEt
    selected_bgn = 0;

    // C3D
    selected_c3d = 0;

    mFocus = 1;
    mRenderC3D = false;

    mTrackball.setTrackball(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5), mWidth * 0.5);
    mTrackball.setQuaternion(Eigen::Quaterniond::Identity());

    // Initialize camera presets
    initializeCameraPresets();
    loadCameraPreset(0);

    // Initialize motion load mode with default value
    mMotionLoadMode = "hdf5";  // Default: load both NPZ and HDF5

    if (argc > 1)
    {
        std::string path = std::string(argv[1]);
        mNetworkPaths.push_back(path);
    }

    // GLFW Initialization
    glfwInit();
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    glfwWindowHintString(GLFW_X11_CLASS_NAME, "MuscleSim");
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    mWindow = glfwCreateWindow(mWidth, mHeight, "MuscleSim", nullptr, nullptr);
    if (mWindow == NULL)
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Set window position from config
    glfwSetWindowPos(mWindow, mWindowXPos, mWindowYPos);
    glfwMakeContextCurrent(mWindow);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }
    glViewport(0, 0, mWidth, mHeight);
    glfwSetWindowUserPointer(mWindow, this); // 창 사이즈 변경

    auto framebufferSizeCallback = [](GLFWwindow *window, int width, int height)
    {
        GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
        app->mWidth = width;
        app->mHeight = height;
        glViewport(0, 0, width, height);
    };
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);

    auto keyCallback = [](GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, true);
        }
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
    };
    glfwSetKeyCallback(mWindow, keyCallback);

    auto cursorPosCallback = [](GLFWwindow *window, double xpos, double ypos)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
    };
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);

    auto mouseButtonCallback = [](GLFWwindow *window, int button, int action, int mods)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
    };
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);

    auto scrollCallback = [](GLFWwindow *window, double xoffset, double yoffset)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
        }
    };
    glfwSetScrollCallback(mWindow, scrollCallback);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ImPlot::CreateContext();

    mns = py::module::import("__main__").attr("__dict__");
    py::module::import("sys").attr("path").attr("insert")(1, "python");

    mSelectedMuscles.clear();
    mRelatedDofs.clear();

    mMotionFrameIdx = 0;
    mLastFrameIdx = 0;
    mCycleAccumulation = Eigen::Vector3d::Zero();

    // Initialize HDF5 selection
    mSelectedHDF5FileIdx = -1;
    mSelectedHDF5ParamIdx = 0;
    mSelectedHDF5CycleIdx = 0;
    mMaxHDF5ParamIdx = 0;
    mMaxHDF5CycleIdx = 0;
    mCurrentHDF5FilePath = "";
    mMotionLoadError = "";
    mParamFailureMessage = "";
    mLastLoadedHDF5ParamsFile = "";

    py::gil_scoped_acquire gil;
    
    // Import Python modules
    try {
        loading_network = py::module::import("python.ray_model").attr("loading_network");
    } catch (const py::error_already_set& e) {
        LOG_WARN("Warning: Failed to import python.ray_model: " << e.what());
        loading_network = py::none();
    }

    // Determine metadata path
    if (!mNetworkPaths.empty()) {
        std::string path = mNetworkPaths.back();
        if (path.substr(path.length() - 4) == ".xml") {
            mCachedMetadata = path;
            mNetworkPaths.pop_back();
        } else {
            try {
                py::object py_metadata = py::module::import("python.ray_model").attr("loading_metadata")(path);
                if (!py_metadata.is_none()) {
                    // Handle both Ray 2.0.1 (string) and Ray 2.12.0 (dict) metadata formats
                    if (py::isinstance<py::str>(py_metadata)) {
                        // Ray 2.0.1: metadata is XML string
                        mCachedMetadata = py_metadata.cast<std::string>();
                    } else if (py::isinstance<py::dict>(py_metadata)) {
                        // Ray 2.12.0: metadata is dict (usually empty)
                        // For now, skip dict metadata as it doesn't contain XML config
                        LOG_INFO("Checkpoint uses Ray 2.12.0 format with dict metadata (skipping)");
                    }
                }
            } catch (const py::error_already_set& e) {
                LOG_WARN("Warning: Failed to load metadata from network path: " << e.what());
            }
        }
    }
    initEnv(mCachedMetadata);
}

void GLFWApp::loadRenderConfig()
{
    try {
        // Use URIResolver to resolve the config path
        PMuscle::URIResolver& resolver = PMuscle::URIResolver::getInstance();
        resolver.initialize();
        std::string resolved_path = resolver.resolve("render.yaml");

        LOG_VERBOSE("[Config] Loading render config from: " << resolved_path);

        YAML::Node config = YAML::LoadFile(resolved_path);

        // Load geometry settings
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

        // Load glfwapp settings
        if (config["glfwapp"]) {
            if (config["glfwapp"]["rollout"] && config["glfwapp"]["rollout"]["count"])
                mDefaultRolloutCount = config["glfwapp"]["rollout"]["count"].as<int>();

            if (config["glfwapp"]["plot"]) {
                if (config["glfwapp"]["plot"]["title"])
                    mPlotTitle = config["glfwapp"]["plot"]["title"].as<bool>();

                if (config["glfwapp"]["plot"]["x_min"])
                    mXmin = config["glfwapp"]["plot"]["x_min"].as<double>();
            }

            if (config["glfwapp"]["playback_speed"]) {
                mViewerPlaybackSpeed = config["glfwapp"]["playback_speed"].as<float>();
                mLastPlaybackSpeed = mViewerPlaybackSpeed;
            }

            if (config["glfwapp"]["resetPhase"]) {
                mResetPhase = config["glfwapp"]["resetPhase"].as<double>();
                LOG_VERBOSE("[Config] Reset phase set to: " << mResetPhase
                          << (mResetPhase < 0.0 ? " (randomized)" : ""));
            }

            if (config["glfwapp"]["motion_load_mode"]) {
                mMotionLoadMode = config["glfwapp"]["motion_load_mode"].as<std::string>();
            }

            if (config["glfwapp"]["resizable_plot"]) {
                if (config["glfwapp"]["resizable_plot"]["x_min"])
                    mXminResizablePlotPane = config["glfwapp"]["resizable_plot"]["x_min"].as<double>();
                if (config["glfwapp"]["resizable_plot"]["y_min"])
                    mYminResizablePlotPane = config["glfwapp"]["resizable_plot"]["y_min"].as<double>();
                if (config["glfwapp"]["resizable_plot"]["y_max"])
                    mYmaxResizablePlotPane = config["glfwapp"]["resizable_plot"]["y_max"].as<double>();
                if (config["glfwapp"]["resizable_plot"]["keys"]) {
                    std::string keys = config["glfwapp"]["resizable_plot"]["keys"].as<std::string>();
                    strncpy(mResizePlotKeys, keys.c_str(), sizeof(mResizePlotKeys) - 1);
                    mResizePlotKeys[sizeof(mResizePlotKeys) - 1] = '\0';
                }
                if (config["glfwapp"]["resizable_plot"]["title"])
                    mPlotTitleResizablePlotPane = config["glfwapp"]["resizable_plot"]["title"].as<bool>();
            }
        }

        LOG_VERBOSE("[Config] Loaded - Window: " << mWidth << "x" << mHeight
                     << ", Control: " << mControlPanelWidth
                     << ", Plot: " << mPlotPanelWidth
                     << ", Rollout: " << mDefaultRolloutCount
                     << ", Playback Speed: " << mViewerPlaybackSpeed
                     << ", Motion Load Mode: " << mMotionLoadMode); 

    } catch (const std::exception& e) {
        std::cerr << "[Config] Warning: Could not load render.yaml: " << e.what() << std::endl;
        std::cerr << "[Config] Using default values." << std::endl;
    }
}

GLFWApp::~GLFWApp()
{
    // Clean up Motion* pointers in new architecture
    for (Motion* motion : mMotionsNew) {
        delete motion;
    }
    mMotionsNew.clear();

    delete mRenderEnv;
    delete mMotionCharacter;
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void GLFWApp::setWindowIcon(const char* icon_path)
{
    GLFWimage images[1];
    images[0].pixels = stbi_load(icon_path, &images[0].width, &images[0].height, 0, 4); // RGBA channels
    if (images[0].pixels) {
        glfwSetWindowIcon(mWindow, 1, images);
        stbi_image_free(images[0].pixels);
    } else {
        std::cerr << "Failed to load icon" << std::endl;
    }
}

void GLFWApp::writeBVH(const dart::dynamics::Joint *jn, std::ofstream &_f, const bool isPos)
{
    auto bn = jn->getParentBodyNode();

    if (!isPos) // HIERARCHY
    {

        _f << "JOINT\tCharacter_" << jn->getName() << std::endl;
        _f << "{"
           << std::endl;
        Eigen::Vector3d current_joint = jn->getParentBodyNode()->getTransform() * (jn->getTransformFromParentBodyNode() * Eigen::Vector3d::Zero());
        Eigen::Vector3d parent_joint = jn->getParentBodyNode()->getTransform() * ((jn->getParentBodyNode()->getParentJoint())->getTransformFromChildBodyNode() * Eigen::Vector3d::Zero());

        Eigen::Vector3d offset = current_joint - parent_joint; // jn->getTransformFromParentBodyNode() * ((bn->getParentJoint()->getTransformFromChildBodyNode()).inverse() * Eigen::Vector3d::Zero());
        offset *= 100.0;
        _f << "OFFSET\t" << offset.transpose() << std::endl;
        _f << "CHANNELS\t" << 3 << "\t" << CHANNELS[5] << "\t" << CHANNELS[3] << "\t" << CHANNELS[4] << std::endl;

        if (jn->getChildBodyNode()->getNumChildBodyNodes() == 0)
        {
            _f << "End Site" << std::endl;
            _f << "{"
               << std::endl;
            _f << "OFFSET\t" << (jn->getChildBodyNode()->getCOM() - current_joint).transpose() * 100.0 << std::endl;
            _f << "}" << std::endl;
        }

        else
            for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
                writeBVH(jn->getChildBodyNode()->getChildJoint(idx), _f, false);

        _f << "}" << std::endl;
    }
    else
    {
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        int idx = jn->getJointIndexInSkeleton();

        if (jn->getNumDofs() == 1)
            pos = (mJointCalibration[idx].transpose() * Eigen::AngleAxisd(jn->getPositions()[0], ((RevoluteJoint *)(jn))->getAxis()).toRotationMatrix() * mJointCalibration[idx]).eulerAngles(2, 0, 1) * 180.0 / M_PI;
        else if (jn->getNumDofs() == 3)
            pos = (mJointCalibration[idx].transpose() * BallJoint::convertToRotation(jn->getPositions()) * mJointCalibration[idx]).eulerAngles(2, 0, 1) * 180.0 / M_PI;

        _f << pos.transpose() << " ";

        for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
            writeBVH(jn->getChildBodyNode()->getChildJoint(idx), _f, true);
    }
}

void GLFWApp::exportBVH(const std::vector<Eigen::VectorXd> &motion, const dart::dynamics::SkeletonPtr &skel)
{
    if (!mRenderEnv)
        return;

    std::ofstream bvh;
    bvh.open("motion1.bvh");
    // HIERARCHY WRITING
    Eigen::VectorXd pos_bkup = skel->getPositions();
    skel->setPositions(Eigen::VectorXd::Zero(pos_bkup.rows()));
    bvh << "HIERARCHY" << std::endl;
    dart::dynamics::Joint *jn = mRenderEnv->getCharacter()->getSkeleton()->getRootJoint();
    dart::dynamics::BodyNode *bn = jn->getChildBodyNode();
    Eigen::Vector3d offset = bn->getTransform().translation();
    bvh << "ROOT\tCharacter_" << jn->getName() << std::endl;
    bvh << "{"
       << std::endl;
    bvh << "OFFSET\t" << offset.transpose() << std::endl;
    bvh << "CHANNELS\t" << 6 << "\t"
        << CHANNELS[0] << "\t" << CHANNELS[1] << "\t" << CHANNELS[2] << "\t"
        << CHANNELS[5] << "\t" << CHANNELS[3] << "\t" << CHANNELS[4] << std::endl;

    for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
        writeBVH(jn->getChildBodyNode()->getChildJoint(idx), bvh, false);

    bvh << "}" << std::endl;
    bvh << "MOTION" << std::endl;
    bvh << "Frames:  " << mMotionBuffer.size() << std::endl;
    bvh << "Frame Time: " << 1.0 / 120 << std::endl;

    bvh.precision(4);
    for (Eigen::VectorXd p : mMotionBuffer)
    {
        skel->setPositions(p);
        Eigen::Vector6d root_pos;
        root_pos.head(3) = skel->getRootBodyNode()->getCOM() * 100.0;
        root_pos.tail(3) = skel->getRootBodyNode()->getTransform().linear().eulerAngles(2, 0, 1) * 180.0 / M_PI;

        // root_pos.setZero();

        bvh << root_pos.transpose() << " ";
        for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
            writeBVH(jn->getChildBodyNode()->getChildJoint(idx), bvh, true);
        bvh << std::endl;
    }
    bvh << std::endl;
    bvh.close();
    // BVH Head Write
}

void GLFWApp::update(bool _isSave)
{
    if (!mRenderEnv) return;
    Eigen::VectorXf action = (mNetworks.size() > 0 ? mNetworks[0].joint.attr("get_action")(mRenderEnv->getState(), mStochasticPolicy).cast<Eigen::VectorXf>() : mRenderEnv->getAction().cast<float>());

    mRenderEnv->setAction(action.cast<double>());

    if (_isSave)
    {
        mRenderEnv->step();
        mMotionBuffer.push_back(mRenderEnv->getCharacter()->getSkeleton()->getPositions());
    }
    else mRenderEnv->step();

    // Check for gait cycle completion AFTER step (when phase counters are updated)
    if (mRenderEnv->isGaitCycleComplete())
    {
        mRolloutStatus.step();
        if (mRolloutStatus.cycle == 0)
        {
            mRolloutStatus.pause = true; // Pause simulation when rollout completes
            return;
        }
    }

    if (mC3dMotion.size() > 0)
    {
        if (mC3DCount + (mC3DReader->getFrameRate() / 60) >= mC3dMotion.size())
        {
            mC3DCOM += mC3dMotion.back().segment(3, 3); // mC3DReader->getBVHSkeleton()->getPositions().segment(3,3);
        }
        mC3DCount += (mC3DReader->getFrameRate() / 60);
        mC3DCount %= mC3dMotion.size();
    }
}

void GLFWApp::plotGraphData(const std::vector<std::string>& keys, ImAxis y_axis,
                            bool show_phase, bool plot_avg_copy, std::string postfix)
{
    if (keys.empty() || !mGraphData) return;

    ImPlot::SetAxis(y_axis);

    for (const auto &key : keys)
    {
        if (!mGraphData->key_exists(key))
        {
            std::cerr << "Key " << key << " not found in mGraphData" << std::endl;
            continue;
        }

        // Get buffer data
        std::vector<double> values = mGraphData->get(key);
        if (values.empty())
            continue;

        int bufferSize = static_cast<int>(values.size());

        // Create x-axis data
        std::vector<float> x(bufferSize);
        for (int i = 0; i < bufferSize; ++i)
        {
            x[i] = -(bufferSize - 1 - i);  // Most recent at 0, oldest at -N
            if (show_phase && mRenderEnv)
                x[i] *= mRenderEnv->getWorld()->getTimeStep();
        }

        // Create y-axis data
        std::vector<float> y(bufferSize);
        for (int i = 0; i < bufferSize; ++i)
        {
            y[i] = static_cast<float>(values[i]);
        }

        // Format key name
        std::string selected_key = key;
        size_t underscore_pos = key.find('_');
        if (underscore_pos != std::string::npos)
        {
            selected_key = key.substr(underscore_pos + 1);
        }
        selected_key = selected_key + postfix;

        // Plot the line
        ImPlot::PlotLine(selected_key.c_str(), x.data(), y.data(), bufferSize);
    }
}

void GLFWApp::plotPhaseBar(double x_min, double x_max, double y_min, double y_max)
{
    if (!mGraphData || !mRenderEnv)
        return;

    if (!mGraphData->key_exists("contact_phaseR"))
    {
        std::cerr << "[GUI] Key contact_phaseR not found in mGraphData" << std::endl;
        return;
    }

    // Get phase buffer data
    std::vector<double> phase_values = mGraphData->get("contact_phaseR");

    // Ensure there are at least two points to compare
    if (phase_values.size() < 2)
        return;

    bool prev_phase = static_cast<bool>(phase_values[0]);
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f); // Thicker line

    for (size_t i = 1; i < phase_values.size(); ++i)
    {
        bool current_phase = static_cast<bool>(phase_values[i]);
        bool phase_change = (prev_phase != current_phase);
        prev_phase = current_phase;

        if (phase_change)
        {
            const double x_val = -(static_cast<int>(phase_values.size()) - 1 - static_cast<int>(i)) * mRenderEnv->getWorld()->getTimeStep();

            // Red: Heel strike (stance phase), Blue: Toe off (swing phase)
            const auto color = current_phase ? IM_COL32(127, 0, 0, 255) : IM_COL32(0, 0, 127, 255);
            ImPlot::PushStyleColor(ImPlotCol_Line, color);

            const double x_vals[2] = {x_val, x_val};
            const double y_vals[2] = {y_min, y_max};
            ImPlot::PlotLine("##phase", x_vals, y_vals, 2);

            ImPlot::PopStyleColor();
        }
    }

    ImPlot::PopStyleVar();
}

float GLFWApp::getHeelStrikeTime()
{
    if (!mGraphData->key_exists("contact_phaseR"))
    {
        std::cout << "[HeelStrike] contact_phaseR key not found in graph data" << std::endl;
        return 0.0;
    }

    if (!mRenderEnv) return 0.0;

    std::vector<double> contact_phase_buffer = mGraphData->get("contact_phaseR");

    // Ensure there are at least two points to compare for transitions
    if (contact_phase_buffer.size() < 2)
    {
        std::cout << "[HeelStrike] Not enough data points for heel strike detection" << std::endl;
        return 0.0;
    }

    bool prev_phase = static_cast<bool>(contact_phase_buffer[0]);
    double heel_strike_time = 0.0;
    bool found_heel_strike = false;

    // Search for the most recent heel strike (swing to stance transition)
    for (size_t i = 1; i < contact_phase_buffer.size(); ++i)
    {
        bool current_phase = static_cast<bool>(contact_phase_buffer[i]);

        // Check for heel strike: transition from swing (false) to stance (true)
        if (!prev_phase && current_phase)
        {
            // Calculate time based on buffer index and time step
            double heel_strike_time_candidate = (-(static_cast<int>(contact_phase_buffer.size())) + static_cast<int>(i)) * mRenderEnv->getWorld()->getTimeStep();
            if (heel_strike_time_candidate < -0.3)
            {
                heel_strike_time = heel_strike_time_candidate;
                found_heel_strike = true;
            } // Don't break - we want the most recent (last) heel strike
        }
        prev_phase = current_phase;
    }

    if (found_heel_strike)
    {
        std::cout << "[HeelStrike] Found heel strike at time: " << heel_strike_time << std::endl;
    }
    else
    {
        std::cout << "[HeelStrike] No heel strike found in current data" << std::endl;
    }
    return heel_strike_time;
}

void GLFWApp::startLoop()
{
    mLastRealTime = glfwGetTime();

    while (!glfwWindowShouldClose(mWindow))
    {
        // Measure real-time delta
        double currentRealTime = glfwGetTime();
        double realDeltaTime = currentRealTime - mLastRealTime;
        mLastRealTime = currentRealTime;

        // Update moving average of frame delta time (alpha = 0.005)
        const double alpha = 0.005;
        if (mRealDeltaTimeAvg == 0.0) {
            mRealDeltaTimeAvg = realDeltaTime;
        } else {
            mRealDeltaTimeAvg = alpha * realDeltaTime + (1.0 - alpha) * mRealDeltaTimeAvg;
        }

        // Detect playback speed change and resync viewer time to simulation time
        if (mRenderEnv && mViewerPlaybackSpeed != mLastPlaybackSpeed)
        {
            mViewerTime = mRenderEnv->getWorld()->getTime();
            mLastPlaybackSpeed = mViewerPlaybackSpeed;
        }

        // Update viewer time (master clock for playback)
        // In manual frame mode when paused, update pose but don't advance time
        double dt = realDeltaTime * mViewerPlaybackSpeed;
        if (!mRolloutStatus.pause || mRolloutStatus.cycle > 0)
        {
            // Playing: normal time progression
            updateViewerTime(dt);
        }
        else if (mMotionNavigationMode == NAVIGATION_MANUAL_FRAME)
        {
            // Paused + manual frame mode: compute pose but don't advance time
            updateViewerTime(0.0);
        }

        // Simulation Step with performance monitoring and sync control
        if (!mRolloutStatus.pause || mRolloutStatus.cycle > 0)
        {
            double simStartTime = glfwGetTime();
            double simStartWorldTime = mRenderEnv ? mRenderEnv->getWorld()->getTime() : 0.0;

            // Sync control: only step simulation if viewer time has caught up with sim time
            // This prevents viewer time from running ahead of simulation time
            bool shouldStepSimulation = (mViewerTime >= simStartWorldTime);

            if (shouldStepSimulation)
            {
                update();
            }
            // else: idle rendering - render current state without stepping simulation

            double simEndTime = glfwGetTime();
            double simEndWorldTime = mRenderEnv ? mRenderEnv->getWorld()->getTime() : 0.0;

            // Measure actual simulation step duration
            mSimulationStepDuration = simEndTime - simStartTime;

            // Update moving average with exponential smoothing (alpha = 0.005)
            const double alpha = 0.01;
            if (mSimStepDurationAvg < 0.0) {
                // Initialize on first measurement
                mSimStepDurationAvg = mSimulationStepDuration;
            } else {
                // Exponential moving average: new_avg = alpha * new_value + (1 - alpha) * old_avg
                mSimStepDurationAvg = alpha * mSimulationStepDuration + (1.0 - alpha) * mSimStepDurationAvg;
            }

            double simTimeAdvanced = simEndWorldTime - simStartWorldTime;

            // Check if playback is too fast for simulation to keep up
            // If simulation advances by dt, it should take at most dt/playback_speed in real time
            if (mRenderEnv && simTimeAdvanced > 0.0)
            {
                double expectedRealTime = simTimeAdvanced / mViewerPlaybackSpeed;
                mIsPlaybackTooFast = (mSimulationStepDuration > expectedRealTime * 1.2); // 20% tolerance
            }
            else
            {
                mIsPlaybackTooFast = false;
            }
        }

        // Rendering
        drawSimFrame();
        drawUIFrame();
        glfwPollEvents();
        glfwSwapBuffers(mWindow);
    }
}

void GLFWApp::initGL()
{
    static float ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float diffuse[] = {0.6, 0.6, 0.6, 1.0};
    static float front_mat_shininess[] = {60.0};
    static float front_mat_specular[] = {0.2, 0.2, 0.2, 1.0};
    static float front_mat_diffuse[] = {0.5, 0.28, 0.38, 1.0};
    static float lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float lmodel_twoside[] = {GL_FALSE};
    GLfloat position[] = {1.0, 0.0, 0.0, 0.0};
    GLfloat position1[] = {-1.0, 0.0, 0.0, 0.0};
    GLfloat position2[] = {0.0, 3.0, 0.0, 0.0};

    if (mRenderConditions) glClearColor(0.0, 0.0, 0.0, 1.0);
    else glClearColor(1.0, 1.0, 1.0, 1.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT, GL_FILL);

    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, position1);

    glEnable(GL_LIGHT2);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT2, GL_POSITION, position2);

    glEnable(GL_LIGHTING);

    glEnable(GL_COLOR_MATERIAL);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_MULTISAMPLE);
}

void GLFWApp::initEnv(std::string metadata)
{
    if (mRenderEnv)
    {
        delete mRenderEnv;
        mRenderEnv = nullptr;
    }
    // Create RenderEnvironment wrapper
    mRenderEnv = new RenderEnvironment(metadata, mGraphData);
    if (!mMotionCharacter)
    {
        for (const auto& muscle: mRenderEnv->getCharacter()->getMuscles()) {
            const auto& muscle_name = muscle->GetName();
            if(muscle_name.find("R_") != std::string::npos) {
                std::string key = "act_" + muscle_name;
                mGraphData->register_key(key, 1000);
            }
        }

        // Detect format by examining first non-whitespace character
        size_t start = metadata.find_first_not_of(" \t\n\r");
        if (start != std::string::npos && metadata[start] == '<') {
            // XML format - extract skeleton path
            TiXmlDocument doc;
            doc.Parse(metadata.c_str());
            TiXmlElement* skel_elem = doc.FirstChildElement("skeleton");
            if (skel_elem) {
                std::string skeletonPath = Trim(std::string(skel_elem->GetText()));
                mMotionCharacter = new Character(skeletonPath, 0, 0, 0);
            } else {
                std::cerr << "No skeleton path found in XML metadata" << std::endl;
                exit(-1);
            }
        } else {
            // YAML format - extract skeleton path
            YAML::Node config = YAML::Load(metadata);
            if (config["environment"] && config["environment"]["skeleton"]) {
                std::string skelPath = config["environment"]["skeleton"]["file"].as<std::string>();
                std::string resolved = PMuscle::URIResolver::getInstance().resolve(skelPath);
                mMotionCharacter = new Character(resolved, 0, 0, 0);
            } else {
                std::cerr << "No skeleton path found in YAML metadata" << std::endl;
                exit(-1);
            }
        }
    }
    
    // Set window title
    if (!mRolloutStatus.name.empty()) {
        mCheckpointName = mRolloutStatus.name;
    } else if (!mNetworkPaths.empty()) {
        mCheckpointName = std::filesystem::path(mNetworkPaths.back()).stem().string();
    } else {
        mCheckpointName = std::filesystem::path(mCachedMetadata).parent_path().filename().string();
    }
    glfwSetWindowTitle(mWindow, mCheckpointName.c_str());

    // Initialize motion skeleton
    initializeMotionSkeleton();

    // // Hardcoded: Load Sim_Healthy.npz as reference motion
    // // Use mMotionCharacter which matches the NPZ data format (not render character)
    // try {
    //     std::string npz_path = "data/npz_motions/Sim_Healthy.npz";
    //     if (fs::exists(npz_path)) {
    //         std::cout << "[GLFWApp] Loading hardcoded reference motion: " << npz_path << std::endl;
    //         NPZ* npz = new NPZ(npz_path);
    //         // CRITICAL: Use mMotionCharacter (NPZ-compatible) not mRenderEnv->getCharacter()
    //         npz->setRefMotion(mMotionCharacter, mRenderEnv->getWorld());
    //         mRenderEnv->setMotion(npz);
    //         std::cout << "[GLFWApp] Successfully loaded hardcoded NPZ reference motion" << std::endl;
    //     } else {
    //         std::cerr << "[GLFWApp] Hardcoded NPZ file not found: " << npz_path << std::endl;
    //     }
    // } catch (const std::exception& e) {
    //     std::cerr << "[GLFWApp] Error loading hardcoded NPZ: " << e.what() << std::endl;
    // }

    // Load networks
    auto character = mRenderEnv->getCharacter();
    mNetworks.clear();
    for (const auto& path : mNetworkPaths) {
        loadNetworkFromPath(path);
    }
    
    if (!mNetworks.empty()) {
        mRenderEnv->setMuscleNetwork(mNetworks.back().muscle);
    }

    // Initialize DOF tracking
    mRelatedDofs.clear();
    mRelatedDofs.resize(mRenderEnv->getCharacter()->getSkeleton()->getNumDofs() * 2, false);

    // Initialize muscle selection states
    if (mRenderEnv->getUseMuscle()) {
        auto muscles = character->getMuscles();
        mMuscleSelectionStates.clear();
        mMuscleSelectionStates.resize(muscles.size(), true);
    }

    // Load GaitNet lists (FGN, BGN, C3D)
    std::string path = "distillation";
    mFGNList.clear();
    mBGNList.clear();
    if (fs::exists(path) && fs::is_directory(path)) {
        for (const auto &entry : fs::recursive_directory_iterator(path)) {
            if (fs::is_regular_file(entry)) {
                std::string filename = entry.path().filename().string();
                if (filename.size() > 7) {
                    if (filename.substr(filename.size() - 7) == ".fgn.pt") {
                        mFGNList.push_back(entry.path().string());
                    } else if (filename.substr(filename.size() - 7) == ".bgn.pt") {
                        mBGNList.push_back(entry.path().string());
                    }
                }
            }
        }
    }

    // C3D files
    path = "c3d";
    mC3DList.clear();
    if (fs::exists(path) && fs::is_directory(path)) {
        for (const auto &entry : fs::directory_iterator(path)) {
            mC3DList.push_back(entry.path().string());
        }
    }

    // Load motion files (includes BVH and HDF5 scanning)
    loadMotionFiles();

    mRenderEnv->setParamDefault();
    reset();
    reset();
}

void GLFWApp::drawAxis()
{
    GUI::DrawLine(Eigen::Vector3d(0, 2E-3, 0), Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0));
    GUI::DrawLine(Eigen::Vector3d(0, 2E-3, 0), Eigen::Vector3d(0.0, 0.5, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0));
    GUI::DrawLine(Eigen::Vector3d(0, 2E-3, 0), Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d(0.0, 0.0, 1.0));
}

void GLFWApp::drawSingleBodyNode(const BodyNode *bn, const Eigen::Vector4d &color)
{
    if (!bn) return;

    glPushMatrix();
    glMultMatrixd(bn->getTransform().data());

    bn->eachShapeNodeWith<VisualAspect>([this, &color](const dart::dynamics::ShapeNode* sn) {
        if (!sn) return true;

        const auto &va = sn->getVisualAspect();

        if (!va || va->isHidden()) return true;

        glPushMatrix();
        Eigen::Affine3d tmp = sn->getRelativeTransform();
        glMultMatrixd(tmp.data());
        Eigen::Vector4d c = va->getRGBA();

        drawShape(sn->getShape().get(), color);

        glPopMatrix();
        return true;
    });
    glPopMatrix();
}

void GLFWApp::drawKinematicsControlPanel()
{
    ImGui::SetNextWindowSize(ImVec2(400, mHeight - 80), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(mControlPanelWidth + 10, 10), ImGuiCond_Once);
    ImGui::Begin("Kinematics Control");

    // FGN
    ImGui::Checkbox("Draw FGN Result\t", &mDrawFGNSkeleton);
    if (ImGui::CollapsingHeader("FGN"))
    {
        int idx = 0;
        for (const auto &ns : mFGNList)
        {
            std::string filename = fs::path(ns).filename().string();
            if (ImGui::Selectable(filename.c_str(), selected_fgn == idx))
                selected_fgn = idx;
            if (selected_fgn)
                ImGui::SetItemDefaultFocus();
            idx++;
        }
    }

    if (mRenderEnv)
    {
        if (ImGui::Button("Load FGN"))
        {
            mDrawFGNSkeleton = true;
            py::tuple res = py::module::import("forward_gaitnet").attr("load_FGN")(mFGNList[selected_fgn], mRenderEnv->getNumParamState(), mRenderEnv->getCharacter()->posToSixDof(mRenderEnv->getCharacter()->getSkeleton()->getPositions()).rows());
            mFGN = res[0];
            mFGNmetadata = res[1].cast<std::string>();

            mNetworkPaths.clear();
            mNetworks.clear();
            std::cout << "METADATA " << std::endl
                      << mFGNmetadata << std::endl;
            initEnv(mFGNmetadata);
        }
    }

    // BGN
    if (ImGui::CollapsingHeader("BGN"))
    {
        int idx = 0;
        for (const auto &ns : mBGNList)
        {
            std::string filename = fs::path(ns).filename().string();
            if (ImGui::Selectable(filename.c_str(), selected_bgn == idx))
                selected_bgn = idx;
            if (selected_bgn)
                ImGui::SetItemDefaultFocus();
            idx++;
        }
    }
    if (mRenderEnv && ImGui::Button("Load BGN"))
    {
        mGVAELoaded = true;
        py::object load_gaitvae = py::module::import("advanced_vae").attr("load_gaitvae");
        int rows = mRenderEnv->getCharacter()->posToSixDof(mRenderEnv->getCharacter()->getSkeleton()->getPositions()).rows();
        mGVAE = load_gaitvae(mBGNList[selected_fgn], rows, 60, mRenderEnv->getNumKnownParam(), mRenderEnv->getNumParamState());

        // TODO: Update for Motion* interface
        // mPredictedMotion.motion = mMotions[mMotionIdx].motion;
        // mPredictedMotion.param = mMotions[mMotionIdx].param;
        // mPredictedMotion.name = "Unpredicted";
    }

    if (ImGui::CollapsingHeader("Motions", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int mMotionPhaseOffset = 0;

        // Display motion status
        bool has_motions = !mMotionsNew.empty();
        if (has_motions) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Motion Loaded (%zu)", mMotionsNew.size());
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "No Motion Loaded");
        }

        // Load/Unload motion buttons
        if (!has_motions) {
            if (ImGui::Button("Load Motion")) {
                loadMotionFiles();
            }
        } else {
            if (ImGui::Button("Unload Motion")) {
                unloadMotion();
            }
        }

        // BVH motions are now integrated into mMotions (loaded via loadMotionFiles())
        // They appear in the NPZ/HDF5 motion lists automatically with source_type="bvh"

        // C3D Motion Files
        if (ImGui::CollapsingHeader("C3D"))
        {
            if (mC3DList.empty())
            {
                ImGui::Text("No C3D files found in c3d directory");
            }
            else
            {
                int idx = 0;
                for (auto ns : mC3DList)
                {
                    if (ImGui::Selectable(ns.c_str(), selected_c3d == idx))
                        selected_c3d = idx;
                    if (selected_c3d)
                        ImGui::SetItemDefaultFocus();
                    idx++;
                }
            }
            static float femur_torsion_l = 0.0;
            static float femur_torsion_r = 0.0;
            static float c3d_scale = 1.0;
            static float height_offset = 0.0;
            ImGui::SliderFloat("Femur Torsion L", &femur_torsion_l, -0.55, 0.55);
            ImGui::SliderFloat("Femur Torsion R", &femur_torsion_r, -0.55, 0.55);
            ImGui::SliderFloat("C3D Scale", &c3d_scale, 0.5, 2.0);
            ImGui::SliderFloat("Height Offset", &height_offset, -0.5, 0.5);

            if (mRenderEnv && ImGui::Button("Load C3D"))
            {
                if (selected_c3d < mC3DList.size() && !mC3DList.empty())
                {
                    mRenderC3D = true;
                    mC3DReader = new C3D_Reader("data/skeleton_gaitnet_narrow_model.xml", "data/marker_set.xml", mRenderEnv->GetEnvironment());
                    std::cout << "Loading C3D: " << mC3DList[selected_c3d] << std::endl;
                    mC3dMotion = mC3DReader->loadC3D(mC3DList[selected_c3d], femur_torsion_l, femur_torsion_r, c3d_scale, height_offset);
                    mC3DCOM = Eigen::Vector3d::Zero();
                }
                else
                {
                    std::cout << "Error: No C3D files available or invalid selection (selected: " << selected_c3d << ", available: " << mC3DList.size() << ")" << std::endl;
                }
            }

            if (mRenderEnv && ImGui::Button("Convert C3D to Motion"))
            {
                // TODO: Implement C3D to Motion* conversion
            }
        }

        // Check currently selected motion type
        bool npz_selected = false;
        bool hdf_selected = false;
        bool bvh_selected = false;
        if (!mMotionsNew.empty() && mMotionIdx >= 0 && mMotionIdx < mMotionsNew.size()) {
            if (mMotionsNew[mMotionIdx]->getSourceType() == "npz") {
                npz_selected = true;
            } else if (mMotionsNew[mMotionIdx]->getSourceType() == "hdfRollout" ||
                       mMotionsNew[mMotionIdx]->getSourceType() == "hdfSingle") {
                hdf_selected = true;
            } else if (mMotionsNew[mMotionIdx]->getSourceType() == "bvh") {
                bvh_selected = true;
            }
        }

        // Unified Motion Files collapsing header with color based on selection
        bool any_motion_selected = npz_selected || hdf_selected || bvh_selected;
        if (any_motion_selected) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));  // Green when selected
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.4f, 0.8f, 0.4f, 1.0f));
        }

        if (ImGui::CollapsingHeader("Motion Files", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // Loaded motions list (NPZ, HDF Single, BVH)
            // Note: hdfRollout motions are loaded via HDF5 Loading Controls, not shown here
            ImGui::Text("Loaded Motions:");
            if (ImGui::BeginListBox("##AllMotions_List", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing())))
            {
                for (int i = 0; i < mMotionsNew.size(); i++)
                {
                    // Skip hdfRollout motions (loaded via HDF5 Loading Controls)
                    if (mMotionsNew[i]->getSourceType() == "hdfRollout") continue;

                    // Add type prefix for displayed motion types
                    std::string prefix;
                    if (mMotionsNew[i]->getSourceType() == "npz") prefix = "[NPZ] ";
                    else if (mMotionsNew[i]->getSourceType() == "hdfSingle") prefix = "[HDF] ";
                    else if (mMotionsNew[i]->getSourceType() == "bvh") prefix = "[BVH] ";
                    std::string display_name = prefix + mMotionsNew[i]->getName();

                    if (ImGui::Selectable(display_name.c_str(), mMotionIdx == i)) {
                        mMotionIdx = i;
                        // Update max frame index for manual navigation
                        if (mMotionsNew[i]->getSourceType() == "bvh" || mMotionsNew[i]->getSourceType() == "hdfSingle") {
                            // TODO: Motion* interface doesn't have hdf5_total_timesteps - need to add getTotalTimesteps() method
                            mMaxFrameIndex = mMotionsNew[i]->getNumFrames() - 1;  // Temporary: using getNumFrames()
                        } else {
                            mMaxFrameIndex = mMotionsNew[i]->getNumFrames() - 1;
                        }
                        // Clamp manual frame index to valid range
                        if (mManualFrameIndex > mMaxFrameIndex) {
                            mManualFrameIndex = mMaxFrameIndex;
                        }

                        if (mRenderEnv) {
                            // Apply parameters from motion file (supports HDF, NPZ, HDFRollout)
                            if (mMotionsNew[i]->hasParameters()) {
                                bool success = mMotionsNew[i]->applyParametersToEnvironment(mRenderEnv->GetEnvironment());
                                if (!success) {
                                    // Count mismatch or error - use defaults
                                    Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
                                    mRenderEnv->setParamState(default_params, false, true);
                                    std::cout << "[" << mMotionsNew[i]->getName() << "] Using default parameters due to mismatch" << std::endl;
                                }
                            } else {
                                // No parameters - use defaults
                                Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
                                mRenderEnv->setParamState(default_params, false, true);
                                std::cout << "[" << mMotionsNew[i]->getName() << "] Warning: No parameters in motion file, using defaults" << std::endl;
                            }
                            mLastLoadedHDF5ParamsFile = "";  // Clear HDF5 parameter tracking
                        }

                        // Align motion with simulated character
                        alignMotionToSimulation();
                    }

                    if (mMotionIdx == i)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }

            // HDF5 Loading Controls (collapsible sub-section)
            ImGui::Separator();
            if (ImGui::TreeNode("Load HDF rollout data"))
        {
            // Static tracking variables (moved outside to be accessible from file selection)
            static int last_param_idx = -1;
            static int last_cycle_idx = -1;
            static int last_file_idx = -1;
            static bool loading_success = false;

            // HDF5 Files listbox
            if (ImGui::BeginListBox("##HDF5_Files", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing())))
        {
            for (int i = 0; i < mHDF5Files.size(); i++)
            {
                if (ImGui::Selectable(mHDF5Files[i].c_str(), mSelectedHDF5FileIdx == i))
                {
                    mSelectedHDF5FileIdx = i;
                    mSelectedHDF5ParamIdx = 0;
                    mSelectedHDF5CycleIdx = 0;

                    // Scan params in selected file to get max indices
                    try {
                        H5::H5File h5file(mHDF5Files[i], H5F_ACC_RDONLY);
                        hsize_t num_params = h5file.getNumObjs();
                        mMaxHDF5ParamIdx = 0;
                        for (hsize_t j = 0; j < num_params; j++) {
                            std::string param_name = h5file.getObjnameByIdx(j);
                            if (param_name.find("param_") == 0) {
                                int param_idx = std::stoi(param_name.substr(6));
                                if (param_idx > mMaxHDF5ParamIdx) {
                                    mMaxHDF5ParamIdx = param_idx;
                                }
                            }
                        }
                        h5file.close();
                        std::cout << "Selected file with max param_idx: " << mMaxHDF5ParamIdx << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Error reading params: " << e.what() << std::endl;
                    }

                    // Load parameters using Motion* interface
                    if (mRenderEnv) {
                        // Find HDFRollout motion matching this file
                        Motion* rollout_motion = nullptr;
                        std::string selected_file = mHDF5Files[mSelectedHDF5FileIdx];
                        for (auto* motion : mMotionsNew) {
                            if (motion->getSourceType() == "hdfRollout" && motion->getName().find(selected_file) != std::string::npos) {
                                rollout_motion = motion;
                                break;
                            }
                        }

                        if (rollout_motion && rollout_motion->hasParameters()) {
                            bool success = rollout_motion->applyParametersToEnvironment(mRenderEnv->GetEnvironment());
                            if (!success) {
                                Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
                                mRenderEnv->setParamState(default_params, false, true);
                                std::cout << "[" << rollout_motion->getName() << "] Using default parameters due to mismatch" << std::endl;
                            }
                        } else if (rollout_motion) {
                            std::cout << "[" << rollout_motion->getName() << "] Warning: No parameters in motion file" << std::endl;
                        }
                    }
                    // TODO: Update for Motion* interface
                    // loadSelectedHDF5Motion();
                    // alignMotionToSimulation();

                    // Update tracking variables to reflect the load
                    last_file_idx = mSelectedHDF5FileIdx;
                    last_param_idx = mSelectedHDF5ParamIdx;
                    last_cycle_idx = mSelectedHDF5CycleIdx;
                    loading_success = true;
                }
                if (mSelectedHDF5FileIdx == i)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndListBox();
        }

        // Param and Cycle sliders (only show if file is selected)
        if (mSelectedHDF5FileIdx >= 0) {

            // Param slider with status
            ImGui::SliderInt("Param", &mSelectedHDF5ParamIdx, 0, mMaxHDF5ParamIdx);
            ImGui::SameLine();
            ImGui::Text("%d / %d", mSelectedHDF5ParamIdx, mMaxHDF5ParamIdx);

            // When param changes, update max cycle index and build available cycle list
            if (mSelectedHDF5ParamIdx != last_param_idx) {
                try {
                    H5::H5File h5file(mHDF5Files[mSelectedHDF5FileIdx], H5F_ACC_RDONLY);
                    std::string param_name = "param_" + std::to_string(mSelectedHDF5ParamIdx);
                    if (h5file.nameExists(param_name)) {
                        H5::Group param_group = h5file.openGroup(param_name);
                        hsize_t num_cycles = param_group.getNumObjs();
                        mMaxHDF5CycleIdx = 0;

                        // Find max cycle index by checking all cycle groups
                        for (hsize_t j = 0; j < num_cycles; j++) {
                            std::string cycle_name = param_group.getObjnameByIdx(j);
                            if (cycle_name.find("cycle_") == 0) {
                                int cycle_idx = std::stoi(cycle_name.substr(6));
                                if (cycle_idx > mMaxHDF5CycleIdx) {
                                    mMaxHDF5CycleIdx = cycle_idx;
                                }
                            }
                        }
                        param_group.close();

                        // Clamp cycle index to valid range
                        mSelectedHDF5CycleIdx = std::min(mSelectedHDF5CycleIdx, mMaxHDF5CycleIdx);

                        std::cout << "Param " << mSelectedHDF5ParamIdx << " has max cycle index: " << mMaxHDF5CycleIdx << std::endl;
                    } else {
                        std::cerr << "Param " << param_name << " does not exist in file" << std::endl;
                    }
                    h5file.close();
                } catch (const std::exception& e) {
                    std::cerr << "Error reading cycles: " << e.what() << std::endl;
                }
                // NOTE: Do NOT update last_param_idx here - let auto-load section handle it
            }

            // Cycle slider with status
            ImGui::SliderInt("Cycle", &mSelectedHDF5CycleIdx, 0, mMaxHDF5CycleIdx);
            ImGui::SameLine();
            ImGui::Text("%d / %d", mSelectedHDF5CycleIdx, mMaxHDF5CycleIdx);

            // Show current file
            ImGui::Text("File: %s", mHDF5Files[mSelectedHDF5FileIdx].c_str());

            // Verify if selected param/cycle exists before loading
            bool can_load = false;
            std::string param_name = "param_" + std::to_string(mSelectedHDF5ParamIdx);
            std::string cycle_name = "cycle_" + std::to_string(mSelectedHDF5CycleIdx);

            try {
                H5::H5File h5file(mHDF5Files[mSelectedHDF5FileIdx], H5F_ACC_RDONLY);
                if (h5file.nameExists(param_name)) {
                    H5::Group param_group = h5file.openGroup(param_name);
                    if (param_group.nameExists(cycle_name)) {
                        can_load = true;
                    }
                    param_group.close();
                }
                h5file.close();
            } catch (const std::exception& e) {
                // Silently ignore, will show "does not exist" status
            }

            // Auto-load when indices change (only if param/cycle exists)
            bool param_changed = (mSelectedHDF5ParamIdx != last_param_idx);
            bool cycle_changed = (mSelectedHDF5CycleIdx != last_cycle_idx);

            if ((param_changed || cycle_changed) && can_load) {
                // Load parameters using Motion* interface
                if (param_changed && mRenderEnv) {
                    // Find HDFRollout motion matching this file
                    Motion* rollout_motion = nullptr;
                    std::string selected_file = mHDF5Files[mSelectedHDF5FileIdx];
                    for (auto* motion : mMotionsNew) {
                        if (motion->getSourceType() == "hdfRollout" && motion->getName().find(selected_file) != std::string::npos) {
                            rollout_motion = motion;
                            break;
                        }
                    }

                    if (rollout_motion && rollout_motion->hasParameters()) {
                        bool success = rollout_motion->applyParametersToEnvironment(mRenderEnv->GetEnvironment());
                        if (!success) {
                            Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
                            mRenderEnv->setParamState(default_params, false, true);
                            std::cout << "[" << rollout_motion->getName() << "] Using default parameters due to mismatch" << std::endl;
                        }
                    } else if (rollout_motion) {
                        std::cout << "[" << rollout_motion->getName() << "] Warning: No parameters in motion file" << std::endl;
                    }
                }

                // Then load motion data
                // TODO: Update for Motion* interface
                // loadSelectedHDF5Motion();
                // alignMotionToSimulation();
                loading_success = true;
                last_param_idx = mSelectedHDF5ParamIdx;
                last_cycle_idx = mSelectedHDF5CycleIdx;
            }

            // Show loading status
            if (!mParamFailureMessage.empty()) {
                // Show parameter failure error (from rollout)
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", mParamFailureMessage.c_str());
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Previous motion still displayed");
            } else if (!mMotionLoadError.empty()) {
                // Show motion load error (e.g., too short)
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", mMotionLoadError.c_str());
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Previous motion still displayed");
            } else if (can_load) {
                if (loading_success) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Status: Loaded %s / %s", param_name.c_str(), cycle_name.c_str());
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Status: Ready to load %s / %s", param_name.c_str(), cycle_name.c_str());
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Status: %s / %s does not exist", param_name.c_str(), cycle_name.c_str());
            }
        }
            ImGui::TreePop();  // End HDF5 Loading Controls TreeNode
        }
        }  // End Motion Files collapsing header

        if (any_motion_selected) {
            ImGui::PopStyleColor(3);
        }

        // Motion Navigation Control
        ImGui::Separator();
        ImGui::Text("Motion Navigation:");
        int nav_mode = static_cast<int>(mMotionNavigationMode);
        ImGui::RadioButton("Viewer Time", &nav_mode, NAVIGATION_VIEWER_TIME);
        ImGui::SameLine();
        ImGui::RadioButton("Manual Frame", &nav_mode, NAVIGATION_MANUAL_FRAME);
        mMotionNavigationMode = static_cast<MotionNavigationMode>(nav_mode);

        if (mMotionNavigationMode == NAVIGATION_MANUAL_FRAME) {
            // Manual frame slider
            if (!mMotionsNew.empty() && mMotionIdx >= 0 && mMotionIdx < mMotionsNew.size()) {
                static int last_manual_frame = -1;

                ImGui::SliderInt("Frame Index", &mManualFrameIndex, 0, mMaxFrameIndex);
                ImGui::Text("Frame: %d / %d", mManualFrameIndex, mMaxFrameIndex);

                // Show frame time for HDF5/BVH motions with timestamps
                std::vector<double> timestamps = mMotionsNew[mMotionIdx]->getTimestamps();
                if ((mMotionsNew[mMotionIdx]->getSourceType() == "hdfRollout" || mMotionsNew[mMotionIdx]->getSourceType() == "hdfSingle" || mMotionsNew[mMotionIdx]->getSourceType() == "bvh") &&
                    !timestamps.empty() &&
                    mManualFrameIndex < timestamps.size()) {
                    double frame_time = timestamps[mManualFrameIndex];
                    ImGui::Text("Time: %.3f s", frame_time);
                }
                last_manual_frame = mManualFrameIndex;
            }
        } else {
            // Show current auto-computed frame in viewer time mode
            if (!mMotionsNew.empty() && mMotionIdx >= 0 && mMotionIdx < mMotionsNew.size()) {
                double phase = mViewerPhase;
                if (mRenderEnv) {
                    phase = mViewerTime / (mRenderEnv->getMotion()->getMaxTime() / (mRenderEnv->getCadence() / sqrt(mRenderEnv->getCharacter()->getGlobalRatio())));
                    phase = fmod(phase, 1.0);
                }
                double frame_float = computeFrameFloat(mMotionsNew[mMotionIdx], phase);
                int current_frame = (int)frame_float;
                ImGui::Text("Auto Frame: %d / %d", current_frame, mMaxFrameIndex);
            }
        }
        ImGui::Separator();

        if (!mMotionsNew.empty() && mMotionIdx >= 0 && mMotionIdx < mMotionsNew.size()) {
            ImGui::SliderInt("Motion Phase Offset", &mMotionPhaseOffset, 0, std::max(0, mMotionsNew[mMotionIdx]->getNumFrames() - 1));
        }
        // TODO: Update for Motion* interface
        // if (ImGui::Button("Convert Motion"))
        // {
        //     int frames_per_cycle = mMotions[mMotionIdx].values_per_frame;
        //     int num_cycles = mMotions[mMotionIdx].num_frames;
        //     Eigen::VectorXd m = mMotions[mMotionIdx].motion;
        //     mMotions[mMotionIdx].motion << m.tail((num_cycles - mMotionPhaseOffset) * frames_per_cycle), m.head(mMotionPhaseOffset * frames_per_cycle);
        // }
        
        // TODO: Update for Motion* interface
        // if (ImGui::Button("Add Current Simulation motion to motion "))
        // {
        //     if(mRenderEnv) addSimulationMotion();
        // }

        if (mRenderEnv && !mMotionsNew.empty() && mMotionIdx >= 0 && mMotionIdx < mMotionsNew.size() && ImGui::Button("Set to Param of motion")) {
            // Apply motion parameters using Motion* interface
            if (mMotionsNew[mMotionIdx]->hasParameters()) {
                bool success = mMotionsNew[mMotionIdx]->applyParametersToEnvironment(mRenderEnv->GetEnvironment());
                if (!success) {
                    Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
                    mRenderEnv->setParamState(default_params, false, true);
                    std::cout << "[" << mMotionsNew[mMotionIdx]->getName() << "] Using default parameters due to mismatch" << std::endl;
                }
            } else {
                std::cout << "[" << mMotionsNew[mMotionIdx]->getName() << "] Warning: No parameters in motion file" << std::endl;
            }
        }

    }

    // TODO: Update for Motion* interface
    // if (mGVAELoaded)
    // {
    //     if (ImGui::CollapsingHeader("GVAE"))
    //     {
    //         if (ImGui::Button("predict new motion"))
    //         {
    //             Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
    //             input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
    //             py::tuple res = mGVAE.attr("render_forward")(input.cast<float>());
    //             Eigen::VectorXd motion = res[0].cast<Eigen::VectorXd>();
    //             Eigen::VectorXd param = res[1].cast<Eigen::VectorXd>();
    //
    //             mPredictedMotion.motion = motion;
    //             mPredictedMotion.param = mRenderEnv->getParamStateFromNormalized(param);
    //         }
    //
    //         if (ImGui::Button("Sampling 1000 params"))
    //         {
    //             Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
    //             input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
    //             mGVAE.attr("sampling")(input.cast<float>(), mMotions[mMotionIdx].param);
    //         }
    //
    //         if (ImGui::Button("Set to predicted param"))
    //             mRenderEnv->setParamState(mPredictedMotion.param, false, true);
    //
    //         if (ImGui::Button("Predict and set param"))
    //         {
    //             Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
    //             input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
    //             py::tuple res = mGVAE.attr("render_forward")(input.cast<float>());
    //             Eigen::VectorXd motion = res[0].cast<Eigen::VectorXd>();
    //             Eigen::VectorXd param = res[1].cast<Eigen::VectorXd>();
    //
    //             mPredictedMotion.motion = motion;
    //             mPredictedMotion.param = mRenderEnv->getParamStateFromNormalized(param);
    //             mRenderEnv->setParamState(mPredictedMotion.param, false, true);
    //         }
    //     }
    // }
    if (ImGui::Button("Save added motion"))
    {
        py::list motions;
        py::list params;

        // TODO: Update for Motion* interface
        // for (auto m : mAddedMotions)
        // {
        //     motions.append(m.motion);
        //     params.append(m.param);
        // }
        //
        // py::object save_motions = py::module::import("converter_to_gvae_set").attr("save_motions");
        // save_motions(motions, params);
    }

    // TODO: Update for Motion* interface
    // TODO: Implement "Save Selected Motion" feature for Motion* interface

    if (mGVAELoaded)
        if (ImGui::CollapsingHeader("Predicted Parameters"))
        {
            Eigen::VectorXf ParamState = mPredictedMotion.param.cast<float>();
            Eigen::VectorXf ParamMin = mRenderEnv->getParamMin().cast<float>();
            Eigen::VectorXf ParamMax = mRenderEnv->getParamMax().cast<float>();
            int idx = 0;
            for (auto c : mRenderEnv->getParamName())
            {
                ImGui::SliderFloat(c.c_str(), &ParamState[idx], ParamMin[idx], ParamMax[idx] + 1E-10);
                idx++;
            }
        }
    ImGui::End();
    if(mRenderEnv)
        mRenderEnv->getCharacter()->updateRefSkelParam(mMotionSkeleton);
}

void GLFWApp::drawSimVisualizationPanel()
{
    
    ImGui::SetNextWindowPos(ImVec2(mWidth - mPlotPanelWidth - 10, 10), ImGuiCond_Once);
    if (!mRenderEnv)
    {
        ImGui::SetNextWindowSize(ImVec2(mPlotPanelWidth, 200), ImGuiCond_Always);
        ImGui::Begin("Sim visualization##1", nullptr, ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Environment not loaded.");



        ImGui::End();
        return;
    }
    ImGui::SetNextWindowSize(ImVec2(mPlotPanelWidth, mHeight - 80), ImGuiCond_Appearing);
    ImGui::Begin("Sim visualization##2");

    // Status
    if (ImGui::CollapsingHeader("Status", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Target Vel      : %.3f m/s", mRenderEnv->getTargetCOMVelocity());
        ImGui::Text("Average Vel     : %.3f m/s", mRenderEnv->getAvgVelocity()[2]);

        // Character position
        Eigen::Vector3d char_pos = mRenderEnv->getCharacter()->getSkeleton()->getRootBodyNode()->getCOM();
        ImGui::Text("Character Pos   : (%.3f, %.3f, %.3f)", char_pos[0], char_pos[1], char_pos[2]);

        ImGui::Separator();
        
        ImGui::Indent();
        if (ImGui::CollapsingHeader("Metadata"))
        {
            if (ImGui::Button("Print")) std::cout << mRenderEnv->getMetadata() << std::endl;
            ImGui::TextUnformatted(mRenderEnv->getMetadata().c_str());
        }
        ImGui::Unindent();

        // Rollout status display
        if (mRolloutStatus.cycle == -1)
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Rollout: Infinite Mode");
        else if (mRolloutStatus.cycle == 0)
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Rollout: Completed");
        else
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Rollout: %d cycles remaining", mRolloutStatus.cycle);
    }

    // Plot X-axis range control
    if (ImGui::Button("HS")) mXmin = getHeelStrikeTime();
    ImGui::SameLine();
    if (ImGui::Button("1.1")) mXmin = -1.1;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(30);
    ImGui::InputDouble("X(min)", &mXmin);

    // Plot title control
    ImGui::Checkbox("Title##PlotTitleCheckbox", &mPlotTitle);
    ImGui::SameLine();
    ImGui::TextDisabled("(Show checkpoint name as plot titles)");

    // Rewards
    if (ImGui::CollapsingHeader("Rewards"))
    {
        std::string title_str = mPlotTitle ? mCheckpointName : "Reward";
        if (ImPlot::BeginPlot((title_str + "##Reward").c_str()))
        {
            ImPlot::SetupAxes("Time (s)", "Reward");

            // Plot reward data using common plotting function
            std::vector<std::string> rewardKeys = {"r", "r_p", "r_v", "r_com", "r_ee", "r_energy", "r_knee_pain", "r_loco", "r_avg", "r_step"};
            if (mRenderEnv->getSeparateTorqueEnergy()) {
                rewardKeys.push_back("r_torque");
                rewardKeys.push_back("r_metabolic");
            }
            plotGraphData(rewardKeys, ImAxis_Y1, true, false, "");
            ImPlot::EndPlot();
        }
    }

    // Metabolic Energy
    if (ImGui::CollapsingHeader("Energy"))
    {
        // Display current metabolic type
        MetabolicType currentType = mRenderEnv->getCharacter()->getMetabolicType();
        const char* typeNames[] = {"LEGACY (Disabled)", "A", "A2", "MA"};
        if (currentType == MetabolicType::LEGACY) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Mode: %s", typeNames[currentType]);
        } else {
            ImGui::Text("Mode: %s", typeNames[currentType]);

            // Display current metabolic energy value
            ImGui::Text("Current: %.2f", mRenderEnv->getCharacter()->getMetabolicStepEnergy());

            // Display torque energy if coefficient is non-zero
            double torqueCoeff = mRenderEnv->getCharacter()->getTorqueEnergyCoeff();
            if (torqueCoeff > 0.0) {
                ImGui::Text("Torque Step: %.2f, Total: %.2f",
                    mRenderEnv->getCharacter()->getTorqueStepEnergy(),
                    mRenderEnv->getCharacter()->getTorqueEnergy()
                );
            }

            // Display combined energy
            ImGui::Text("Combined: %.2f", mRenderEnv->getCharacter()->getEnergy());

            ImGui::Separator();

            // Checkboxes for energy plotting
            static bool plot_mean_energy = false;
            ImGui::Checkbox("Plot Mean Energy", &plot_mean_energy);

            std::string title_energy = mPlotTitle ? mCheckpointName : "Energy";
            if (ImPlot::BeginPlot((title_energy + "##Energy").c_str()))
            {
                ImPlot::SetupAxes("Time (s)", "Energy");

                // Plot energy data based on checkboxes
                std::vector<std::string> metabolicKeys;
                metabolicKeys.push_back("energy_combined");
                if (plot_mean_energy) {
                    metabolicKeys.push_back("energy_metabolic");
                } else {
                    metabolicKeys.push_back("energy_metabolic_step");
                }
                if (torqueCoeff > 0.0) {
                    if (plot_mean_energy) {
                        metabolicKeys.push_back("energy_torque");
                    } else {
                        metabolicKeys.push_back("energy_torque_step");
                    }
                }
                plotGraphData(metabolicKeys, ImAxis_Y1, true, false, "");

                ImPlot::EndPlot();
            }
        }
    }

    // Knee Loading
    if (ImGui::CollapsingHeader("Knee Loading", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Display current knee loading max value
        ImGui::Text("Max Knee Loading: %.2f kN", mRenderEnv->getCharacter()->getKneeLoadingMax());

        ImGui::Separator();

        std::string title_knee = mPlotTitle ? mCheckpointName : "Max Knee Loading";
        if (ImPlot::BeginPlot((title_knee + "##KneeLoading").c_str()))
        {
            ImPlot::SetupAxes("Time (s)", "Knee Loading (kN)");

            // Plot max knee loading
            std::vector<std::string> kneeKeys = {"knee_loading_max"};
            plotGraphData(kneeKeys, ImAxis_Y1, true, false, "");

            ImPlot::EndPlot();
        }
    }

    // Joint Loading
    if (ImGui::CollapsingHeader("Joint Loading", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Joint selection dropdown
        static int selected_joint = 1; // Default to knee (0=hip, 1=knee, 2=ankle)
        const char* joint_names[] = {"Hip", "Knee", "Ankle"};
        const char* joint_prefixes[] = {"hip", "knee", "ankle"};

        ImGui::Text("Select Joint:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::Combo("##JointSelector", &selected_joint, joint_names, IM_ARRAYSIZE(joint_names));

        static bool plot_component = false;
        ImGui::Checkbox("Component", &plot_component);

        std::string selected_prefix = joint_prefixes[selected_joint];
        std::string selected_name = joint_names[selected_joint];

        // Force plot
        std::string title_force = mPlotTitle ? mCheckpointName : (selected_name + " Forces (N)");
        if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(0, mXmin, 0, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -1, 6);
        if (ImPlot::BeginPlot((title_force + "##JointForces").c_str()))
        {
            ImPlot::SetupAxes("Time (s)", "Force (N)");
            std::vector<std::string> forceKeys;
            if (plot_component) {
                forceKeys = {
                    selected_prefix + "_force_x",
                    selected_prefix + "_force_y",
                    selected_prefix + "_force_z",
                };
            } else {
                forceKeys = {
                    selected_prefix + "_force_mag"
                };
            }
            plotGraphData(forceKeys, ImAxis_Y1, true, false, "");
            ImPlotRect limits = ImPlot::GetPlotLimits();
            plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);
            ImPlot::EndPlot();
        }

        // Torque plot
        std::string title_torque = mPlotTitle ? mCheckpointName : (selected_name + " Torques (Nm)");
        if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(0, mXmin, 0, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -150, 150);
        if (ImPlot::BeginPlot((title_torque + "##JointTorques").c_str()))
        {
            ImPlot::SetupAxes("Time (s)", "Torque (Nm)");
            std::vector<std::string> torqueKeys;
            if (plot_component) {
                torqueKeys = {
                    selected_prefix + "_torque_x",
                    selected_prefix + "_torque_y",
                    selected_prefix + "_torque_z",
                };
            } else {
                torqueKeys = {
                    selected_prefix + "_torque_mag"
                };
            }
            plotGraphData(torqueKeys, ImAxis_Y1, true, false, "");
            ImPlotRect limits = ImPlot::GetPlotLimits();
            plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);
            ImPlot::EndPlot();
        }
    }

    // Kinematics
    if (ImGui::CollapsingHeader("Kinematics", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(0, mXmin, 0, ImGuiCond_Always);
        else ImPlot::SetNextAxisLimits(0, -1.5, 0);
        ImPlot::SetNextAxisLimits(3, -45, 60);

        std::string title_major_joints = mPlotTitle ? mCheckpointName : "Major Joint Angles (deg)";
        if (ImPlot::BeginPlot((title_major_joints + "##MajorJoints").c_str()))
        {
            ImPlot::SetupAxes("Time (s)", "Angle (deg)");

            std::vector<std::string> jointKeys = {"angle_HipR", "angle_KneeR", "angle_AnkleR"};
            plotGraphData(jointKeys, ImAxis_Y1, true, false, "");

            // Overlay phase bars
            ImPlotRect limits = ImPlot::GetPlotLimits();
            plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

            ImPlot::EndPlot();
        }

        ImGui::Separator();

        if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(0, mXmin, 0, ImGuiCond_Always);
        else ImPlot::SetNextAxisLimits(0, -1.5, 0);
        ImPlot::SetNextAxisLimits(3, -10, 15);

        std::string title_minor_joints = mPlotTitle ? mCheckpointName : "Minor Joint Angles (deg)";
        if (ImPlot::BeginPlot((title_minor_joints + "##MinorJoints").c_str()))
        {

            ImPlot::SetupAxes("Time (s)", "Angle (deg)");

            std::vector<std::string> jointKeys = {"angle_HipIRR", "angle_HipAbR"};
            plotGraphData(jointKeys, ImAxis_Y1, true, false, "");

            // Overlay phase bars
            ImPlotRect limits = ImPlot::GetPlotLimits();
            plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

            ImPlot::EndPlot();
        }

        // // Pelvis Angles Plot
        // ImPlot::SetNextAxisLimits(0, -3, 0);
        // ImPlot::SetNextAxisLimits(3, -20, 20);
        // if (ImPlot::BeginPlot("Pelvis Angles (deg)"))
        // {
        //     ImPlot::SetupAxes("Time (s)", "Angle (deg)");

        //     std::vector<std::string> pelvisKeys = {"angle_Rotation", "angle_Obliquity", "angle_Tilt"};
        //     plotGraphData(pelvisKeys, ImAxis_Y1, true, false, "");

        //     // Overlay phase bars
        //     ImPlotRect limits = ImPlot::GetPlotLimits();
        //     plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

        //     ImPlot::EndPlot();
        // }

        // ImGui::Separator();

        // // Torso Sway Plot
        // ImPlot::SetNextAxisLimits(0, -3, 0);
        // ImPlot::SetNextAxisLimits(3, -0.2, 0.2);
        // if (ImPlot::BeginPlot("Torso Sway (m)"))
        // {
        //     ImPlot::SetupAxes("Time (s)", "Sway (m)");

        //     std::vector<std::string> swayKeys = {"sway_Torso_X"};
        //     plotGraphData(swayKeys, ImAxis_Y1, true, false, "");

        //     // Overlay phase bars
        //     ImPlotRect limits = ImPlot::GetPlotLimits();
        //     plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

        //     ImPlot::EndPlot();
        // }
    }

    // State
    if (ImGui::CollapsingHeader("State"))
    {
        auto state = mRenderEnv->getState();
        ImPlot::SetNextAxisLimits(0, -0.5, state.rows() + 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -5, 5);

        double *x = new double[state.rows()]();
        double *y = new double[state.rows()]();
        for (int i = 0; i < state.rows(); i++)
        {
            x[i] = i;
            y[i] = state[i];
        }
        if (ImPlot::BeginPlot("state"))
        {
            ImPlot::PlotBars("", x, y, state.rows(), 1.0);
            ImPlot::EndPlot();
        }

        ImGui::Separator();

        // Constraint Force
        Eigen::VectorXd cf = mRenderEnv->getCharacter()->getSkeleton()->getConstraintForces();
        ImPlot::SetNextAxisLimits(0, -0.5, cf.rows() + 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -5, 5);
        double *x_cf = new double[cf.rows()]();
        double *y_cf = cf.data();

        for (int i = 0; i < cf.rows(); i++)
            x_cf[i] = i;

        if (ImPlot::BeginPlot("Constraint Force"))
        {
            ImPlot::PlotBars("dt", x_cf, y_cf, cf.rows(), 1.0);
            ImPlot::EndPlot();
        }
    }

    // Torques
    static int joint_selected = 0;
    if (ImGui::CollapsingHeader("Torques"))
    {
        if (ImGui::BeginListBox("Joint", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing())))
        {
            int idx = 0;

            for (int i = 0; i < mRenderEnv->getCharacter()->getSkeleton()->getNumDofs(); i++)
            {
                if (ImGui::Selectable((std::to_string(i) + "_force").c_str(), joint_selected == i))
                    joint_selected = i;
                ImGui::SetItemDefaultFocus();
            }
            ImGui::EndListBox();
        }
        ImPlot::SetNextAxisLimits(3, 0, 1.5);
        ImPlot::SetNextAxisLimits(0, 0, 2.5, ImGuiCond_Always);
        if (ImPlot::BeginPlot((std::to_string(joint_selected) + "_torque_graph").c_str(), ImVec2(-1, 250)))
        {
            std::vector<std::vector<double>> p;
            std::vector<double> px;
            std::vector<double> py;
            p.clear();
            px.clear();
            py.clear();

            for (int i = 0; i < mRenderEnv->getDesiredTorqueLogs().size(); i++)
            {
                px.push_back(0.01 * i - mRenderEnv->getDesiredTorqueLogs().size() * 0.01 + 2.5);
                py.push_back(mRenderEnv->getDesiredTorqueLogs()[i][joint_selected]);
            }

            p.push_back(px);
            p.push_back(py);

            ImPlot::PlotLine("##activation_graph", p[0].data(), p[1].data(), p[0].size());
            ImPlot::EndPlot();
        }

        ImGui::Separator();

        // Torque bars
        if (mRenderEnv->getUseMuscle())
        {
            MuscleTuple tp = mRenderEnv->getCharacter()->getMuscleTuple(false);

            Eigen::VectorXd fullJtp = Eigen::VectorXd::Zero(mRenderEnv->getCharacter()->getSkeleton()->getNumDofs());
            if (mRenderEnv->getCharacter()->getIncludeJtPinSPD())
                fullJtp.tail(fullJtp.rows() - mRenderEnv->getCharacter()->getSkeleton()->getRootJoint()->getNumDofs()) = tp.JtP;
            Eigen::VectorXd dt = mRenderEnv->getCharacter()->getSPDForces(mRenderEnv->getCharacter()->getPDTarget(), fullJtp).tail(tp.JtP.rows());

            auto mtl = mRenderEnv->getCharacter()->getMuscleTorqueLogs();

            Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(tp.JtP.rows());
            Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(tp.JtP.rows());

            for (int i = 0; i < tp.JtA.rows(); i++)
            {
                for (int j = 0; j < tp.JtA.cols(); j++)
                {
                    if (tp.JtA(i, j) > 0)
                        max_tau[i] += tp.JtA(i, j);
                    else
                        min_tau[i] += tp.JtA(i, j);
                }
            }

            ImPlot::SetNextAxisLimits(0, -0.5, dt.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -5, 5);
            double *x_tau = new double[dt.rows()]();
            double *y_tau = dt.data();
            double *y_min = min_tau.data();
            double *y_max = max_tau.data();
            double *y_passive = tp.JtP.data();

            for (int i = 0; i < dt.rows(); i++)
                x_tau[i] = i;

            if (ImPlot::BeginPlot("torque"))
            {
                ImPlot::PlotBars("min", x_tau, y_min, dt.rows(), 1.0);
                ImPlot::PlotBars("max", x_tau, y_max, dt.rows(), 1.0);
                ImPlot::PlotBars("dt", x_tau, y_tau, dt.rows(), 1.0);
                ImPlot::PlotBars("passive", x_tau, y_passive, dt.rows(), 1.0);
                if (mtl.size() > 0)
                    ImPlot::PlotBars("exact", x_tau, mtl.back().tail(mtl.back().rows() - 6).data(), dt.rows(), 1.0);

                ImPlot::EndPlot();
            }
        }
        else
        {
            Eigen::VectorXd dt = mRenderEnv->getCharacter()->getTorque();
            ImPlot::SetNextAxisLimits(0, -0.5, dt.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -5, 5);
            double *x_tau = new double[dt.rows()]();
            for (int i = 0; i < dt.rows(); i++)
                x_tau[i] = i;
            if (ImPlot::BeginPlot("torque"))
            {
                ImPlot::PlotBars("dt", x_tau, dt.data(), dt.rows(), 1.0);
                ImPlot::EndPlot();
            }
        }
    }

    // Muscles
    static int selected = 0;
    if (ImGui::CollapsingHeader("Muscles"))
    {

        auto m = mRenderEnv->getCharacter()->getMuscles()[selected];

        ImPlot::SetNextAxisLimits(3, 500, 0);
        ImPlot::SetNextAxisLimits(0, 0, 1.5, ImGuiCond_Always);
        if (ImPlot::BeginPlot((m->name + "_force_graph").c_str(), ImVec2(-1, 250)))
        {
            ImPlot::SetupAxes("length", "force");
            std::vector<std::vector<double>> p = m->GetGraphData();

            ImPlot::PlotLine("##active", p[1].data(), p[2].data(), 250);
            ImPlot::PlotLine("##active_with_activation", p[1].data(), p[3].data(), 250);
            ImPlot::PlotLine("##passive", p[1].data(), p[4].data(), 250);

            ImPlot::PlotInfLines("current", p[0].data(), 1);
            ImPlot::EndPlot();
        }

        ImPlot::SetNextAxisLimits(3, 0, 1.5);
        ImPlot::SetNextAxisLimits(0, 0, 2.5, ImGuiCond_Always);
        if (ImPlot::BeginPlot((m->name + "_activation_graph").c_str(), ImVec2(-1, 250)))
        {
            // Only plot if this is a right-side muscle (has data in mGraphData)
            std::string key = "act_" + m->name;
            if (mGraphData->key_exists(key))
            {
                std::vector<double> activation_data = mGraphData->get(key);
                std::vector<double> px;
                std::vector<double> py;

                for (int i = 0; i < activation_data.size(); i++)
                {
                    px.push_back(0.01 * i - activation_data.size() * 0.01 + 2.5);
                    py.push_back(activation_data[i]);
                }

                ImPlot::PlotLine("##activation_graph", px.data(), py.data(), px.size());
            }
            ImPlot::EndPlot();
        }

        ImGui::Separator();

        // Activation bars
        if (mRenderEnv->getUseMuscle())
        {
            Eigen::VectorXd activation = mRenderEnv->getCharacter()->getActivations();

            ImPlot::SetNextAxisLimits(0, -0.5, activation.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, 0, 1);
            double *x_act = new double[activation.rows()]();
            double *y_act = new double[activation.rows()]();

            for (int i = 0; i < activation.rows(); i++)
            {
                x_act[i] = i;
                y_act[i] = activation[i];
            }
            if (ImPlot::BeginPlot("activation"))
            {
                ImPlot::PlotBars("activation_level", x_act, y_act, activation.rows(), 1.0);
                ImPlot::EndPlot();
            }
        }

        ImGui::Separator();

        ImGui::Text("Muscle Name");
        if (ImGui::BeginListBox("Muscle", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing())))
        {
            int idx = 0;
            for (auto m : mRenderEnv->getCharacter()->getMuscles())
            {
                if (ImGui::Selectable((m->name + "_force").c_str(), selected == idx))
                    selected = idx;
                if (selected)
                    ImGui::SetItemDefaultFocus();
                idx++;
            }
            ImGui::EndListBox();
        }
    }

    // Network Weights
    if (ImGui::CollapsingHeader("Network Weights"))
    {
        if (mRenderEnv->getWeights().size() > 0)
        {
            auto weight = mRenderEnv->getWeights().data();
            ImPlot::SetNextAxisLimits(0, -0.5, mRenderEnv->getWeights().size() - 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, 0.0, 1.0);

            double *x_w = new double[mRenderEnv->getWeights().size()]();
            for (int i = 0; i < mRenderEnv->getWeights().size(); i++)
                x_w[i] = i;

            if (ImPlot::BeginPlot("weight"))
            {
                ImPlot::PlotBars("", x_w, weight, mRenderEnv->getWeights().size(), 0.6);
                ImPlot::EndPlot();
            }
        }

        ImGui::Separator();

        if (mRenderEnv->getDmins().size() > 0)
        {
            auto dmins = mRenderEnv->getDmins().data();
            auto betas = mRenderEnv->getBetas().data();

            ImPlot::SetNextAxisLimits(0, -0.5, mRenderEnv->getDmins().size() - 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, 0.0, 1.0);

            double *x_dmin = new double[mRenderEnv->getDmins().size()]();
            double *x_beta = new double[mRenderEnv->getBetas().size()]();

            for (int i = 0; i < mRenderEnv->getDmins().size(); i++)
            {
                x_dmin[i] = (i - 0.15);
                x_beta[i] = (i + 0.15);
            }
            if (ImPlot::BeginPlot("dmins_and_betas"))
            {
                ImPlot::PlotBars("dmin", x_dmin, dmins, mRenderEnv->getDmins().size(), 0.3);
                ImPlot::PlotBars("beta", x_beta, betas, mRenderEnv->getBetas().size(), 0.3);

                ImPlot::EndPlot();
            }
        }
    }

    // Camera Status
    drawCameraStatusSection();

    ImGui::End();
}

void GLFWApp::drawCameraStatusSection() {
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
        ImGui::Text("RelTrans: [%.3f, %.3f, %.3f]", mRelTrans[0], mRelTrans[1], mRelTrans[2]);
        ImGui::Text("Zoom: %.3f", mZoom);

        Eigen::Quaterniond quat = mTrackball.getCurrQuat();
        ImGui::Text("Quaternion: [%.3f, %.3f, %.3f, %.3f]",
                    quat.w(), quat.x(), quat.y(), quat.z());

        // Camera presets section
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Camera Presets:");
        for (int i = 0; i < 3; i++) {
            if (mCameraPresets[i].isSet) {
                if (ImGui::Button(("Load " + std::to_string(i)).c_str())) {
                    loadCameraPreset(i);
                }
                ImGui::SameLine();
                ImGui::Text("%s", mCameraPresets[i].description.c_str());
            }
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Keyboard Shortcuts:");
        ImGui::Text("Press C: Print camera info to console");
        ImGui::Text("Press 0/1/2: Load camera presets");
    }
}

void GLFWApp::drawJointControlSection() {
    if (ImGui::CollapsingHeader("Joint")) {
        if (!mRenderEnv || !mRenderEnv->getCharacter()) {
            ImGui::TextDisabled("Load environment first");
        } else {
            auto skel = mRenderEnv->getCharacter()->getSkeleton();

            // Joint Position Control
            Eigen::VectorXd pos_lower_limit = skel->getPositionLowerLimits();
            Eigen::VectorXd pos_upper_limit = skel->getPositionUpperLimits();
            Eigen::VectorXd currentPos = skel->getPositions();
            Eigen::VectorXf pos_rad = currentPos.cast<float>();

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

                    // Create label: "JointName Direction" or just "JointName" for single DOF
                    std::string label;
                    if (num_dofs > 1 && d < 6) {
                        label = std::string(dof_labels[d]);
                    } else if (num_dofs > 1) {
                        label = "DOF " + std::to_string(d);
                    } else {
                        label = "";
                    }
                    
                    if (is_translation) {
                        // Root joint - expand limits
                        // Translation: use raw values (meters), use InputFloat instead of SliderFloat
                        display_value = pos_rad[dof_idx];

                        std::string drag_label = label + "##drag_" + joint_name + std::to_string(d);
                        ImGui::SetNextItemWidth(200);
                        ImGui::InputFloat(drag_label.c_str(), &display_value, 0.0f, 0.0f, "%.3fm");
                    } else {
                        // Non-root joints: always rotation, convert to degrees
                        lower_limit = pos_lower_limit[dof_idx] * (180.0f / M_PI);
                        upper_limit = pos_upper_limit[dof_idx] * (180.0f / M_PI);
                        display_value = pos_deg[dof_idx];

                        // Store previous value to detect changes
                        float prev_value = display_value;

                        std::string drag_label = label + "##drag_" + joint_name + std::to_string(d);
                        ImGui::SetNextItemWidth(200);
                        const char* format = is_translation ? "%.3fm" : "%.1f°";

                        // Check if limits are valid for SliderFloat (must be finite and within ImGui's range)
                        const float max_slider_range = 1e37f; // ImGui's acceptable range is roughly ±FLT_MAX/2
                        bool valid_limits = std::isfinite(lower_limit) && std::isfinite(upper_limit) &&
                                          std::abs(lower_limit) < max_slider_range &&
                                          std::abs(upper_limit) < max_slider_range;

                        if (valid_limits) {
                            // Use SliderFloat with valid limits
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
                        } else {
                            // Use InputFloat without limits (similar to dof_idx < 3 case)
                            ImGui::InputFloat(drag_label.c_str(), &display_value, 0.0f, 0.0f, format);
                        }
                    }
                   
                    
                    // Update internal storage
                    if (is_translation) {
                        // Translation: store directly in meters
                        pos_rad[dof_idx] = display_value;
                    } else {
                        // Rotation: update degree value and convert to radians
                        pos_deg[dof_idx] = display_value;
                        pos_rad[dof_idx] = display_value * (M_PI / 180.0f);
                    }
                    
                    dof_idx++;
                }
                
                ImGui::Unindent();
            }
            
            // Update positions
            skel->setPositions(pos_rad.cast<double>());
        }
    }
}

void GLFWApp::printCameraInfo() {
    Eigen::Quaterniond quat = mTrackball.getCurrQuat();

    std::cout << "\n======================================" << std::endl;
    std::cout << "Copy and paste below to CAMERA_PRESET_DEFINITIONS:" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "PRESET|[Add description]|"
              << mEye[0] << "," << mEye[1] << "," << mEye[2] << "|"
              << mUp[0] << "," << mUp[1] << "," << mUp[2] << "|"
              << mRelTrans[0] << "," << mRelTrans[1] << "," << mRelTrans[2] << "|"
              << mZoom << "|"
              << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z()
              << std::endl;
    std::cout << "======================================\n" << std::endl;
}

void GLFWApp::initializeCameraPresets() {
    mCurrentCameraPreset = -1;

    for (int i = 0; i < 3; i++) {
        mCameraPresets[i].isSet = false;

        std::string definition(CAMERA_PRESET_DEFINITIONS[i]);
        std::stringstream ss(definition);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, '|')) {
            tokens.push_back(token);
        }

        if (tokens.size() == 7 && tokens[0] == "PRESET") {
            mCameraPresets[i].description = tokens[1];

            std::stringstream ss_eye(tokens[2]);
            std::string val;
            std::getline(ss_eye, val, ','); mCameraPresets[i].eye[0] = std::stod(val);
            std::getline(ss_eye, val, ','); mCameraPresets[i].eye[1] = std::stod(val);
            std::getline(ss_eye, val, ','); mCameraPresets[i].eye[2] = std::stod(val);

            std::stringstream ss_up(tokens[3]);
            std::getline(ss_up, val, ','); mCameraPresets[i].up[0] = std::stod(val);
            std::getline(ss_up, val, ','); mCameraPresets[i].up[1] = std::stod(val);
            std::getline(ss_up, val, ','); mCameraPresets[i].up[2] = std::stod(val);

            // Note: 'trans' in preset definitions represents mRelTrans (user manual offset)
            std::stringstream ss_trans(tokens[4]);
            std::getline(ss_trans, val, ','); mCameraPresets[i].trans[0] = std::stod(val);
            std::getline(ss_trans, val, ','); mCameraPresets[i].trans[1] = std::stod(val);
            std::getline(ss_trans, val, ','); mCameraPresets[i].trans[2] = std::stod(val);

            mCameraPresets[i].zoom = std::stod(tokens[5]);

            if (tokens.size() > 6) {
                std::stringstream ss_quat(tokens[6]);
                std::getline(ss_quat, val, ','); double w = std::stod(val);
                std::getline(ss_quat, val, ','); double x = std::stod(val);
                std::getline(ss_quat, val, ','); double y = std::stod(val);
                std::getline(ss_quat, val, ','); double z = std::stod(val);
                mCameraPresets[i].quat = Eigen::Quaterniond(w, x, y, z);
            }

            mCameraPresets[i].isSet = true;
        } else {
            if (tokens.size() != 7) {
                std::cout << "Camera preset " << i << " is not valid. Because it have " << tokens.size() << " tokens. It should have 7 tokens" << std::endl;
            } else {
                std::cout << "Camera preset " << i << " is not valid. Because the first token is not PRESET. Got " << tokens[0] << std::endl;
            }
        }
    }
}

void GLFWApp::runRollout() {
    mRolloutStatus.cycle = mRolloutCycles;
    mRolloutStatus.pause = false;
}


void GLFWApp::loadCameraPreset(int index) {
    if (index < 0 || index >= 3 || !mCameraPresets[index].isSet)
    {
        LOG_WARN("[Camera] Preset " << index << " is not valid");
        return;
    }
    LOG_VERBOSE("[Camera] Loading camera preset " << index << ": " << mCameraPresets[index].description);

    mEye = mCameraPresets[index].eye;
    mUp = mCameraPresets[index].up;
    mRelTrans = mCameraPresets[index].trans;  // Restore user manual translation offset
    mZoom = mCameraPresets[index].zoom;
    mTrackball.setQuaternion(mCameraPresets[index].quat);
    mCurrentCameraPreset = index;
}

void GLFWApp::drawSimControlPanel()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    if (!mRenderEnv) {
        ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, 60), ImGuiCond_Always);
        ImGui::Begin("Sim Control##1", nullptr, ImGuiWindowFlags_NoCollapse);
        if (ImGui::Button("Load Environment")) initEnv(mCachedMetadata);
        ImGui::End();
        return;
    }
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight - 80), ImGuiCond_Appearing);
    ImGui::Begin("Sim Control##2");
    
    if (ImGui::Button("Unload Environment"))
    {
        delete mRenderEnv;
        mRenderEnv = nullptr;
    }

    if (!mRenderEnv) {
        ImGui::End();
        return;
    }

    // Metabolic Energy Control
    if (ImGui::CollapsingHeader("Metabolic Energy", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Get current metabolic type
        MetabolicType currentType = mRenderEnv->getCharacter()->getMetabolicType();
        int currentTypeInt = static_cast<int>(currentType);

        // Dropdown for metabolic type selection
        const char* metabolicTypes[] = {"LEGACY", "A", "A2", "MA", "MA2"};
        ImGui::SetNextItemWidth(50);
        if (ImGui::Combo("Type", &currentTypeInt, metabolicTypes, IM_ARRAYSIZE(metabolicTypes)))
        {
            // Update metabolic type when selection changes
            mRenderEnv->getCharacter()->setMetabolicType(static_cast<MetabolicType>(currentTypeInt));
        }
        ImGui::SameLine();
        // Button to reset step-based metrics (metabolic, torque, knee loading)
        if (ImGui::Button("Reset"))
        {
            mRenderEnv->getCharacter()->resetStep();
        }

        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::Text("LEGACY: No metabolic computation");
            ImGui::Text("A: Sum of absolute activations");
            ImGui::Text("A2: Sum of squared activations");
            ImGui::Text("MA: Mass-weighted absolute activations");
            ImGui::Text("MA2: Mass-weighted squared activations");
            ImGui::Separator();
            ImGui::Text("Note: Call cacheMuscleMass() before using MA/MA2 modes");
            ImGui::EndTooltip();
        }

        // Metabolic weight slider
        float metabolicWeight = static_cast<float>(mRenderEnv->getMetabolicWeight());
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("Weight", &metabolicWeight, 0.0f, 0.0f, "%.3f"))
        {
            mRenderEnv->setMetabolicWeight(static_cast<double>(metabolicWeight));
        }

        // Torque energy coefficient input
        float torqueCoeff = static_cast<float>(mRenderEnv->getCharacter()->getTorqueEnergyCoeff());
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("Torque Coeff", &torqueCoeff, 0.0f, 0.0f, "%.3f"))
        {
            mRenderEnv->getCharacter()->setTorqueEnergyCoeff(static_cast<double>(torqueCoeff));
        }

        // Separate torque energy checkbox
        bool separateTorque = mRenderEnv->getSeparateTorqueEnergy();
        if (ImGui::Checkbox("Separate Torque", &separateTorque))
        {
            mRenderEnv->setSeparateTorqueEnergy(separateTorque);
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::Text("When enabled:");
            ImGui::Text("- r_energy uses only metabolic energy");
            ImGui::Text("- r_torque calculated separately");
            ImGui::Text("- Logs: r_energy, r_metabolic, r_torque");
            ImGui::Separator();
            ImGui::Text("When disabled:");
            ImGui::Text("- r_energy uses combined energy");
            ImGui::Text("- Logs: r_energy only");
            ImGui::EndTooltip();
        }

        ImGui::Separator();
        ImGui::Text("Knee Pain Penalty");

        // Knee pain weight slider
        float kneePainWeight = static_cast<float>(mRenderEnv->getKneePainWeight());
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("Knee Weight", &kneePainWeight, 0.0f, 0.0f, "%.3f"))
        {
            mRenderEnv->setKneePainWeight(static_cast<double>(kneePainWeight));
        }

        // Scale knee pain slider
        float scaleKneePain = static_cast<float>(mRenderEnv->getScaleKneePain());
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("Knee Scale", &scaleKneePain, 0.0f, 0.0f, "%.3f"))
        {
            mRenderEnv->setScaleKneePain(static_cast<double>(scaleKneePain));
        }

        // Multiplicative knee pain checkbox
        bool useMultiplicative = mRenderEnv->getUseMultiplicativeKneePain();
        if (ImGui::Checkbox("Multiplicative Mode", &useMultiplicative))
        {
            mRenderEnv->setUseMultiplicativeKneePain(useMultiplicative);
        }
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("When enabled, knee pain multiplies with main gait term (scale not used)");
        }
    }

    // Rollout Control
    if (ImGui::CollapsingHeader("Rollout", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (mRolloutCycles == -1) mRolloutCycles = mDefaultRolloutCount;
        ImGui::SetNextItemWidth(70);
        ImGui::InputInt("Cycles", &mRolloutCycles);
        if (mRolloutCycles < 1) mRolloutCycles = 1;

        ImGui::SameLine();

        // Run button
        if (ImGui::Button("Run##Rollout")) runRollout();
    }

    // Muscle Control
    if (ImGui::CollapsingHeader("Muscle"))
    {
        Eigen::VectorXf activation = mRenderEnv->getCharacter()->getActivations().cast<float>(); // * mRenderEnv->getActionScale();
        int idx = 0;
        for (auto m : mRenderEnv->getCharacter()->getMuscles())
        {
            ImGui::SliderFloat((m->GetName().c_str()), &activation[idx], 0.0, 1.0);
            idx++;
        }
        mRenderEnv->getCharacter()->setActivations((activation.cast<double>()));
    }

    // Joint Control - use new detailed control method
    drawJointControlSection();

    // Body Parameters
    if (ImGui::CollapsingHeader("Sim Parameters"))
    {
        Eigen::VectorXf group_v = Eigen::VectorXf::Ones(mRenderEnv->getGroupParam().size());
        int idx = 0;

        for (auto p_g : mRenderEnv->getGroupParam())
            group_v[idx++] = p_g.v;

        idx = 0;
        for (auto p_g : mRenderEnv->getGroupParam())
        {
            ImGui::SliderFloat(p_g.name.c_str(), &group_v[idx], 0.0, 1.0);
            idx++;
        }
        mRenderEnv->setGroupParam(group_v.cast<double>());
        mRenderEnv->getCharacter()->updateRefSkelParam(mMotionSkeleton);
    }

    // Rendering
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Load Ref Motion..."))
        {
            IGFD::FileDialogConfig config;
            config.path = "data/motion";
            ImGuiFileDialog::Instance()->OpenDialog("ChooseRefMotionDlgKey", "Choose Reference Motion File",
                ".*", config);
        }
        ImGui::Checkbox("Draw Reference Motion", &mDrawReferenceSkeleton);
            
        // File dialog display and handling
        if (ImGuiFileDialog::Instance()->Display("ChooseRefMotionDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                std::string fileName = ImGuiFileDialog::Instance()->GetCurrentFileName();

                // Load the selected motion file
                try {
                    std::cout << "[RefMotion] Loading reference motion from: " << filePathName << std::endl;

                    // Determine file type and create appropriate Motion object
                    Motion* newMotion = nullptr;

                    if (filePathName.find(".bvh") != std::string::npos) {
                        // Load BVH file
                        BVH* bvh = new BVH(filePathName);
                        bvh->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());
                        newMotion = bvh;
                        LOG_INFO("[RefMotion] Loaded BVH file with " << bvh->getNumFrames() << " frames");
                    }
                    else if (filePathName.find(".npz") != std::string::npos) {
                        // Load NPZ file
                        NPZ* npz = new NPZ(filePathName);
                        npz->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());
                        newMotion = npz;
                        LOG_INFO("[RefMotion] Loaded NPZ file with " << npz->getNumFrames() << " frames");
                    }
                    else if (filePathName.find(".h5") != std::string::npos ||
                             filePathName.find(".hdf5") != std::string::npos) {
                        // Load HDF5 file (single-cycle extracted format)
                        HDF* hdf = new HDF(filePathName);
                        hdf->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());
                        newMotion = hdf;
                        LOG_INFO("[RefMotion] Loaded HDF file with " << hdf->getNumFrames() << " frames");
                    }
                    else {
                        std::cerr << "[RefMotion] Unsupported file format: " << filePathName << std::endl;
                    }

                    // Update environment with new motion
                    if (newMotion) {
                        mRenderEnv->setMotion(newMotion);
                        std::cout << "[RefMotion] Successfully updated reference motion" << std::endl;
                    }

                } catch (const std::exception& e) {
                    std::cerr << "[RefMotion] Error loading motion file: " << e.what() << std::endl;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::Checkbox("Draw PD Target Motion", &mDrawPDTarget);
        ImGui::Checkbox("Draw Joint Sphere", &mDrawJointSphere);
        ImGui::Checkbox("Stochastic Policy", &mStochasticPolicy);
        ImGui::Checkbox("Draw Foot Step", &mDrawFootStep);
        ImGui::Checkbox("Draw EOE", &mDrawEOE);
        ImGui::Checkbox("Draw C3D", &mRenderC3D);

        ImGui::Separator();
        // Muscle Filtering and Selection
        if (ImGui::CollapsingHeader("Muscle##Rendering"))
        {
            ImGui::Indent();
            ImGui::SetNextItemWidth(125);
            ImGui::SliderFloat("Resolution", &mMuscleResolution, 0.0, 1000.0);
            ImGui::SetNextItemWidth(125);
            ImGui::SliderFloat("Transparency", &mMuscleTransparency, 0.1, 1.0);

            ImGui::Separator();

            // Get all muscles
            auto allMuscles = mRenderEnv->getCharacter()->getMuscles();

            // Initialize selection states if needed
            if (mMuscleSelectionStates.size() != allMuscles.size())
            {
                mMuscleSelectionStates.resize(allMuscles.size(), true);
            }

            // Count selected muscles
            int selectedCount = 0;
            for (bool selected : mMuscleSelectionStates)
            {
                if (selected) selectedCount++;
            }

            ImGui::Text("Selected: %d / %zu", selectedCount, allMuscles.size());

            // Text filter
            ImGui::InputText("Filter", mMuscleFilterText, IM_ARRAYSIZE(mMuscleFilterText));

            // Filter muscles by name
            std::vector<int> filteredIndices;
            std::string filterStr(mMuscleFilterText);
            std::transform(filterStr.begin(), filterStr.end(), filterStr.begin(), ::tolower);

            for (int i = 0; i < allMuscles.size(); i++)
            {
                std::string muscleName = allMuscles[i]->name;
                std::transform(muscleName.begin(), muscleName.end(), muscleName.begin(), ::tolower);

                if (filterStr.empty() || muscleName.find(filterStr) != std::string::npos)
                {
                    filteredIndices.push_back(i);
                }
            }

            // Select All / Deselect All buttons for filtered muscles
            if (ImGui::Button("Select"))
            {
                for (int idx : filteredIndices)
                {
                    mMuscleSelectionStates[idx] = true;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Deselect"))
            {
                for (int idx : filteredIndices)
                {
                    mMuscleSelectionStates[idx] = false;
                }
            }

            ImGui::Text("Filtered Muscles: %zu", filteredIndices.size());

            // Display filtered muscles with checkboxes
            ImGui::BeginChild("MuscleList", ImVec2(0, 300), true);
            for (int idx : filteredIndices)
            {
                bool selected = mMuscleSelectionStates[idx];
                if (ImGui::Checkbox(allMuscles[idx]->name.c_str(), &selected))
                {
                    mMuscleSelectionStates[idx] = selected;
                }
            }
            ImGui::EndChild();
        }

        if (mRenderEnv->getUseMuscle()) mRenderEnv->getCharacter()->getMuscleTuple(false);

        
        // If no muscles are manually selected, show none (empty list)
        // The rendering code will use mSelectedMuscles if it has content
        
        ImGui::RadioButton("PassiveForce", &mMuscleRenderTypeInt, 0);
        ImGui::RadioButton("ContractileForce", &mMuscleRenderTypeInt, 1);
        ImGui::RadioButton("ActivatonLevel", &mMuscleRenderTypeInt, 2);
        ImGui::RadioButton("Contracture", &mMuscleRenderTypeInt, 3);
        ImGui::RadioButton("Weakness", &mMuscleRenderTypeInt, 4);
        mMuscleRenderType = MuscleRenderingType(mMuscleRenderTypeInt);
        ImGui::Unindent();
    }
    


    // Network
    if (ImGui::CollapsingHeader("Network"))
    {
        if (mRenderEnv->getWeights().size() > 0)
        {
            for (int i = 0; i < mUseWeights.size(); i++)
            {
                bool uw = mUseWeights[i];

                if (mRenderEnv->getUseMuscle())
                    ImGui::Checkbox((std::to_string(i / 2) + "_th network" + (i % 2 == 0 ? "_joint" : "_muscle")).c_str(), &uw);
                else
                    ImGui::Checkbox((std::to_string(i) + "_th network").c_str(), &uw);

                mUseWeights[i] = uw;
                ImGui::SameLine();
            }
            mRenderEnv->setUseWeights(mUseWeights);
            ImGui::NewLine();
        }
    }

    ImGui::End();
}

void GLFWApp::updateUnifiedKeys()
{
    std::string unified_str = "";
    for (int i = 0; i < mResizablePlots.size(); ++i) {
        for (int j = 0; j < mResizablePlots[i].keys.size(); ++j) {
            unified_str += mResizablePlots[i].keys[j];
            if (j < mResizablePlots[i].keys.size() - 1) {
                unified_str += ",";
            }
        }
        if (i < mResizablePlots.size() - 1) {
            unified_str += " | ";
        }
    }
    strncpy(mResizePlotKeys, unified_str.c_str(), sizeof(mResizePlotKeys) - 1);
    mResizePlotKeys[sizeof(mResizePlotKeys) - 1] = '\0';
}

void GLFWApp::updateResizablePlotsFromKeys()
{
    std::string unified_keys_str(mResizePlotKeys);
    std::vector<std::string> plot_key_groups;
    std::string current_group;
    std::stringstream ss(unified_keys_str);

    char c;
    while(ss.get(c)) {
        if (c == '|') {
            plot_key_groups.push_back(current_group);
            current_group.clear();
        } else {
            current_group += c;
        }
    }
    plot_key_groups.push_back(current_group);

    mResizablePlots.resize(plot_key_groups.size());
    for (int i = 0; i < mResizablePlots.size(); ++i) {
        mResizablePlots[i].keys.clear();
        std::stringstream key_ss(plot_key_groups[i]);
        std::string key;
        while(std::getline(key_ss, key, ',')) {
            // Trim whitespace
            key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            key.erase(std::find_if(key.rbegin(), key.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), key.end());
            if (!key.empty()) {
                mResizablePlots[i].keys.push_back(key);
            }
        }
    }
    updateUnifiedKeys();
    mResizePlotPane = true;
}

void GLFWApp::drawResizablePlotPane()
{
    if (!mShowResizablePlotPane) return;

    const float RESIZABLE_PLOT_WIDTH = 400;
    const float RESIZABLE_PLOT_HEIGHT = 500;
    const float RESIZABLE_PLOT_TEXT_HEIGHT = 20;
    const float KEY_CONFIG_HEIGHT = 70;
    
    if (mResizePlotPane) {
        ImGui::SetNextWindowSize(ImVec2(RESIZABLE_PLOT_WIDTH * mResizablePlots.size(), RESIZABLE_PLOT_HEIGHT), ImGuiCond_Always);
        mResizePlotPane = false;
    } else {
        ImGui::SetNextWindowSize(ImVec2(RESIZABLE_PLOT_WIDTH * mResizablePlots.size(), RESIZABLE_PLOT_HEIGHT), ImGuiCond_Once);
    }

    ImGui::Begin("Resizable Plot Pane (A to toggle)", &mShowResizablePlotPane);

    // Controls for number of plots
    ImGui::Text("%zu plots", mResizablePlots.size());
    ImGui::SameLine();
    if (ImGui::Button("Add")) {
        mResizablePlots.emplace_back();
        updateUnifiedKeys();
        mResizePlotPane = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove") && mResizablePlots.size() > 1) {
        mResizablePlots.pop_back();
        updateUnifiedKeys();
        mResizePlotPane = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit")) {
        mResizePlotPane = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("HS")) mXminResizablePlotPane = getHeelStrikeTime();
    ImGui::SameLine();
    if (ImGui::Button("1.1")) mXminResizablePlotPane = -1.1;
    ImGui::SetNextItemWidth(30);
    ImGui::SameLine();
    if (ImGui::InputDouble("X(min)", &mXminResizablePlotPane)) {
        mSetResizablePlotPane = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(30);
    if (ImGui::InputDouble("Y(min)", &mYminResizablePlotPane)) {
        mSetResizablePlotPane = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(30);
    if (ImGui::InputDouble("Y(max)", &mYmaxResizablePlotPane)) {
        mSetResizablePlotPane = true;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Title", &mPlotTitleResizablePlotPane);

    // height: ImGui::GetWindowSize().y - RESIZABLE_PLOT_TEXT_HEIGHT

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::InputText("Key config", mResizePlotKeys, sizeof(mResizePlotKeys), ImGuiInputTextFlags_EnterReturnsTrue)) {
        updateResizablePlotsFromKeys();
    }

    // height: ImGui::GetWindowSize().y - RESIZABLE_PLOT_TEXT_HEIGHT * 2 

    ImGui::Separator();

    // Draw each plot horizontally
    float plot_width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * (mResizablePlots.size() - 1)) / mResizablePlots.size();
    if (plot_width <= 0) plot_width = 1;

    for (int i = 0; i < mResizablePlots.size(); ++i) {
        ImGui::BeginChild(std::string("PlotChild" + std::to_string(i)).c_str(), ImVec2(plot_width, RESIZABLE_PLOT_HEIGHT - 5 * RESIZABLE_PLOT_TEXT_HEIGHT), true);
        ImGui::Text("Plot %d", i + 1);

        ImGui::Columns(2, "##key_management_columns");

        // Left column: Added keys
        ImGui::Text("Added Keys");
        if (ImGui::BeginListBox(("Keys##" + std::to_string(i)).c_str(), ImVec2(-1, KEY_CONFIG_HEIGHT))) {
            for (int j = 0; j < mResizablePlots[i].keys.size(); ++j) {
                if (ImGui::Selectable(mResizablePlots[i].keys[j].c_str(), mResizablePlots[i].selectedKey == j)) {
                    mResizablePlots[i].selectedKey = j;
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    mResizablePlots[i].keys.erase(mResizablePlots[i].keys.begin() + j);
                    if (mResizablePlots[i].selectedKey == j) mResizablePlots[i].selectedKey = -1;
                    else if (mResizablePlots[i].selectedKey > j) mResizablePlots[i].selectedKey--;
                    updateUnifiedKeys();
                    break; 
                }
            }
            ImGui::EndListBox();
        }

        ImGui::NextColumn();

        // Right column: Key search and candidates
        ImGui::Text("Search Key");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        static std::vector<std::string> all_keys;
        bool enterPressed = ImGui::InputText(("##NewKey" + std::to_string(i)).c_str(),
                                             mResizablePlots[i].newKeyInput,
                                             sizeof(mResizablePlots[i].newKeyInput),
                                             ImGuiInputTextFlags_EnterReturnsTrue);

        if (ImGui::IsItemActivated()) { // Update keys when input is activated
            all_keys = mGraphData->get_keys();
        }

        std::vector<std::string> candidates;
        if (strlen(mResizablePlots[i].newKeyInput) > 0) {
            std::string search_str = mResizablePlots[i].newKeyInput;
            std::transform(search_str.begin(), search_str.end(), search_str.begin(), ::tolower);
            for (const auto& key : all_keys) {
                std::string lower_key = key;
                std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
                if (lower_key.find(search_str) != std::string::npos) {
                    candidates.push_back(key);
                }
            }
        }

        if (enterPressed && strlen(mResizablePlots[i].newKeyInput) > 0) {
            if (!candidates.empty()) {
                for (const auto& candidate : candidates) {
                    mResizablePlots[i].keys.push_back(candidate);
                }
            } else {
                mResizablePlots[i].keys.push_back(mResizablePlots[i].newKeyInput);
            }
            memset(mResizablePlots[i].newKeyInput, 0, sizeof(mResizablePlots[i].newKeyInput));
            updateUnifiedKeys();
        }

        if (!candidates.empty()) {
            if (ImGui::BeginListBox(("Candidates##" + std::to_string(i)).c_str(), ImVec2(-1, KEY_CONFIG_HEIGHT - RESIZABLE_PLOT_TEXT_HEIGHT))) {
                for (const auto& candidate : candidates) {
                    if (ImGui::Selectable(candidate.c_str())) {
                        mResizablePlots[i].keys.push_back(candidate);
                        memset(mResizablePlots[i].newKeyInput, 0, sizeof(mResizablePlots[i].newKeyInput));
                        updateUnifiedKeys();
                    }
                }
                ImGui::EndListBox();
            }
        }

        // height: ImGui::GetWindowSize().y - RESIZABLE_PLOT_TEXT_HEIGHT * 2 - KEY_CONFIG_HEIGHT

        ImGui::Columns(1);
        ImGui::Separator();

        // The actual plot
        if (std::abs(mXminResizablePlotPane) > 1e-6) ImPlot::SetNextAxisLimits(ImAxis_X1, mXminResizablePlotPane, 0, ImGuiCond_Always);
        else ImPlot::SetNextAxisLimits(ImAxis_X1, -1.5, 0, ImGuiCond_Once);
        if (mSetResizablePlotPane){
            ImPlot::SetNextAxisLimits(ImAxis_Y1, mYminResizablePlotPane, mYmaxResizablePlotPane, ImGuiCond_Always);
        } else {
            ImPlot::SetNextAxisLimits(ImAxis_Y1, mYminResizablePlotPane, mYmaxResizablePlotPane, ImGuiCond_Once);
        }
        if (ImPlot::BeginPlot(("##Plot" + std::to_string(i)).c_str(), ImVec2(-1, -1))) {
            ImPlot::SetupAxes("Time (s)", "Value");
            plotGraphData(mResizablePlots[i].keys, ImAxis_Y1, true, false, "");

            ImPlotRect limits = ImPlot::GetPlotLimits();
            plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);
            ImPlot::EndPlot();
        }

        ImGui::EndChild();
        if (i < mResizablePlots.size() - 1) {
            ImGui::SameLine();
        }
    }
    if (mPlotTitleResizablePlotPane) {
        ImGui::SetCursorPosX((ImGui::GetWindowWidth() - ImGui::CalcTextSize(mCheckpointName.c_str()).x) * 0.5f);
        ImGui::Text("%s", mCheckpointName.c_str());
    }
    mSetResizablePlotPane = false;
    ImGui::End();
}

void GLFWApp::drawUIFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    drawSimControlPanel();
    drawSimVisualizationPanel();
    drawKinematicsControlPanel();
    drawTimingPane();
    drawResizablePlotPane();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLFWApp::drawTimingPane()
{
    if (!mShowTimingPane)
        return;

    // Create a compact floating window
    ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize;
    if (!ImGui::Begin("Timing Info (T to toggle)", &mShowTimingPane, window_flags))
    {
        ImGui::End();
        return;
    }

    // Use fixed-width font for better alignment
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);

    // Display timing metrics in a compact table format
    ImGui::Text("Viewer Time    : %.3f s", mViewerTime);
    ImGui::Text("Phase          : %.3f", mViewerPhase);
    if (mRenderEnv) {
        ImGui::Text("Simulation Time: %.3f s", mRenderEnv->getWorld()->getTime());
    }
    ImGui::Text("Sim Step Count : %d", mRenderEnv->GetEnvironment()->getSimulationStep());
    
    ImGui::Separator();

    ImGui::SetNextItemWidth(100);
    ImGui::InputDouble("Cycle Duration", &mViewerCycleDuration, 0.1, 0.5, "%.3f");
    if (mViewerCycleDuration < 0.1) mViewerCycleDuration = 0.1;

    if(ImGui::Button("0.5x")) mViewerPlaybackSpeed = 0.5;
    ImGui::SameLine();
    if(ImGui::Button("1x")) mViewerPlaybackSpeed = 1.0;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("Playback Speed", &mViewerPlaybackSpeed, 0.1, 2.5, "%.2f");

    if (mRenderEnv)
    {
        ImGui::Separator();
        ImGui::Text("Frame Delta    : %.1f ms", mRealDeltaTimeAvg * 1000.0);
        ImGui::Text("Sim Step Time  : %.1f ms", mSimulationStepDuration * 1000.0);
        ImGui::Text("Sim Step Avg   : %.1f ms", mSimStepDurationAvg * 1000.0);
    }

    if (mIsPlaybackTooFast)
    {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "WARNING: Playback too fast!");
        if (!mRenderEnv) {
             ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Simulation cannot keep up");
        }
    }

    ImGui::PopFont();
    ImGui::End();
}

void GLFWApp::drawPhase(double phase, double normalized_phase)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glViewport(0, 0, mWidth, mHeight);
    gluOrtho2D(0.0, (GLdouble)mWidth, 0.0, (GLdouble)mHeight);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glLineWidth(1.0);
    glColor3f(0.0f, 0.0f, 0.0f);
    glTranslatef(mWidth * 0.5, mHeight * 0.05, 0.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 360; i++)
    {
        double theta = i / 180.0 * M_PI;
        double x = mHeight * 0.04 * cos(theta);
        double y = mHeight * 0.04 * sin(theta);
        glVertex2d(x, y);
    }
    glEnd();

    glColor3f(1, 0, 0);
    glBegin(GL_LINES);
    glVertex2d(0, 0);
    glVertex2d(mHeight * 0.04 * sin(normalized_phase * 2 * M_PI), mHeight * 0.04 * cos(normalized_phase * 2 * M_PI));
    glEnd();

    glColor3f(0, 0, 0);
    glLineWidth(2.0);

    glBegin(GL_LINES);
    glVertex2d(0, 0);
    glVertex2d(mHeight * 0.04 * sin(phase * 2 * M_PI), mHeight * 0.04 * cos(phase * 2 * M_PI));
    glEnd();

    glPointSize(2.0);
    glBegin(GL_POINTS);
    glVertex2d(mHeight * 0.04 * sin(phase * 2 * M_PI), mHeight * 0.04 * cos(phase * 2 * M_PI));
    glEnd();

    // Draw foot contact indicators
    if (mRenderEnv) {
        const double indicator_size = mHeight * 0.03;  // Size of the indicator squares
        const double spacing = indicator_size * 2.8;  // Space between indicators

        bool isLeftLegStance = mRenderEnv->getIsLeftLegStance();

        // Left foot indicator (slightly to the left of center)
        // Active (stance) when isLeftLegStance is true
        glColor4f(1.0f, 0.0f, 0.0f, isLeftLegStance ? 1.0f : 0.2f);
        glBegin(GL_QUADS);
        glVertex2d(-spacing, -indicator_size * 0.5);
        glVertex2d(-spacing + indicator_size, -indicator_size * 0.5);
        glVertex2d(-spacing + indicator_size, indicator_size * 0.5);
        glVertex2d(-spacing, indicator_size * 0.5);
        glEnd();

        // Right foot indicator (slightly to the right of center)
        // Active (stance) when isLeftLegStance is false
        glColor4f(1.0f, 0.0f, 0.0f, !isLeftLegStance ? 1.0f : 0.2f);
        glBegin(GL_QUADS);
        glVertex2d(spacing - indicator_size, -indicator_size * 0.5);
        glVertex2d(spacing, -indicator_size * 0.5);
        glVertex2d(spacing, indicator_size * 0.5);
        glVertex2d(spacing - indicator_size, indicator_size * 0.5);
        glEnd();
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}

void GLFWApp::drawPlayableMotion()
{
    // Motion pose is computed in updateViewerTime(), this function only renders
    if (mMotionsNew.empty() || mMotionIdx < 0 || mMotionIdx >= mMotionsNew.size()) {
        return;
    }
    if (mMotionStates[mMotionIdx].currentPose.size() == 0) {
        return;
    }
    drawSkeleton(mMotionStates[mMotionIdx].currentPose, Eigen::Vector4d(0.8, 0.8, 0.2, 0.7));
}


void GLFWApp::drawSimFrame()
{
    initGL();
    setCamera();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glViewport(0, 0, mWidth, mHeight);
    gluPerspective(mPersp, mWidth / mHeight, 0.1, 100.0);
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, 0.0, mUp[0], mUp[1], mUp[2]);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    mTrackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
    mTrackball.setRadius(std::min(mWidth, mHeight) * 0.4);
    mTrackball.applyGLRotation();

    glScalef(mZoom, mZoom, mZoom);
    // Apply combined translation: automatic focus tracking + user manual offset
    Eigen::Vector3d totalTrans = mTrans + mRelTrans;
    glTranslatef(totalTrans[0] * 0.001, totalTrans[1] * 0.001, totalTrans[2] * 0.001);
    glEnable(GL_DEPTH_TEST);

    if (!mRenderConditions) drawGround();

    // Simulated Character
    if (mRenderEnv){
        // Draw phase using viewer time
        drawPhase(mViewerPhase, mViewerPhase);
        if (mDrawCharacter)
        {
            drawSkeleton(mRenderEnv->getCharacter()->getSkeleton()->getPositions(), Eigen::Vector4d(0.65, 0.65, 0.65, 1.0));
            if (!mRenderConditions) drawShadow();
            if (mMuscleSelectionStates.size() > 0) drawMuscles(mMuscleRenderType);
        }
        if ((mRenderEnv->getRewardType() == gaitnet) && mDrawFootStep) drawFootStep();
        if (mDrawJointSphere)
        {
            for (auto jn : mRenderEnv->getCharacter()->getSkeleton()->getJoints())
            {
                Eigen::Vector3d jn_pos = jn->getChildBodyNode()->getTransform() * jn->getTransformFromChildBodyNode() * Eigen::Vector3d::Zero();
                glColor4f(0.0, 0.0, 0.0, 1.0);
                GUI::DrawSphere(jn_pos, 0.01);
                glColor4f(0.5, 0.5, 0.5, 0.2);
                GUI::DrawSphere(jn_pos, 0.1);
            }
        }
        if (mDrawReferenceSkeleton && !mRenderConditions)
        {
            Eigen::VectorXd pos = (mDrawPDTarget ? mRenderEnv->getCharacter()->getPDTarget() : mRenderEnv->getTargetPositions());
            drawSkeleton(pos, Eigen::Vector4d(1.0, 0.35, 0.35, 1.0));
        }
        if (mDrawEOE)
        {
            glColor4f(1.0, 0.0, 0.0, 1.0);
            GUI::DrawSphere(mRenderEnv->getCharacter()->getSkeleton()->getCOM(), 0.01);
            glColor4f(0.5, 0.5, 0.8, 0.2);
            glBegin(GL_QUADS);
            glVertex3f(-10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter()->getGlobalRatio(), -10);
            glVertex3f(10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter()->getGlobalRatio(), -10);
            glVertex3f(10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter()->getGlobalRatio(), 10);
            glVertex3f(-10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter()->getGlobalRatio(), 10);
            glEnd();
        }

        // FGN - use viewer phase for playback
        if (mDrawFGNSkeleton)
        {
            Eigen::VectorXd FGN_in = Eigen::VectorXd::Zero(mRenderEnv->getNumParamState() + 2);
            Eigen::VectorXd phase = Eigen::VectorXd::Zero(2);

            // Use viewer phase instead of simulation phase
            phase[0] = sin(2 * M_PI * mViewerPhase);
            phase[1] = cos(2 * M_PI * mViewerPhase);

            FGN_in << mRenderEnv->getNormalizedParamStateFromParam(mRenderEnv->getParamState()), phase;

            Eigen::VectorXd res = mFGN.attr("get_action")(FGN_in).cast<Eigen::VectorXd>();
            if (!mRolloutStatus.pause || mRolloutStatus.cycle > 0)
            {
                // Because of display Hz
                mFGNRootOffset[0] += res[6] * 0.5;
                mFGNRootOffset[2] += res[8] * 0.5;
            }
            res[6] = mFGNRootOffset[0];
            res[8] = mFGNRootOffset[2];

            Eigen::VectorXd pos = mRenderEnv->getCharacter()->sixDofToPos(res);
            drawSkeleton(pos, Eigen::Vector4d(0.35, 0.35, 1.0, 1.0));
        }
    }

    drawPlayableMotion();

    // FGN playback using viewer time (independent of mRenderEnv)
    if (!mRenderEnv && mDrawFGNSkeleton && !mFGN.is_none())
    {
        drawPhase(mViewerPhase, mViewerPhase);  // Draw phase bar using viewer phase

        // FGN network forward pass
        // Note: Full implementation requires mRenderEnv for parameter state
        Eigen::VectorXd phase = Eigen::VectorXd::Zero(2);
        phase[0] = sin(2 * M_PI * mViewerPhase);
        phase[1] = cos(2 * M_PI * mViewerPhase);

        // Would need: FGN_in << normalized_param_state, phase
        // Then: res = mFGN.attr("get_action")(FGN_in).cast<Eigen::VectorXd>();
        // Then: drawSkeleton with converted position
    }

    // Draw Marker Network
    if (mC3dMotion.size() > 0 && !mRenderConditions && mRenderC3D)
    {
        auto skel = mC3DReader->getBVHSkeleton();
        Eigen::VectorXd pos = mC3dMotion[mC3DCount];

        pos[3] += mC3DCOM[0];
        pos[5] += mC3DCOM[2];

        skel->setPositions(pos);

        // Draw Joint Origin and Axis
        glColor4f(0.0, 0.0, 1.0, 1.0);
        for (auto jn : skel->getJoints())
        {
            if (jn->getParentBodyNode() == nullptr)
                continue;

            Eigen::Vector3d p = jn->getParentBodyNode()->getTransform() * jn->getTransformFromParentBodyNode() * Eigen::Vector3d::Zero();
            Eigen::Vector3d axis_x = jn->getParentBodyNode()->getTransform() * jn->getTransformFromParentBodyNode() * Eigen::Vector3d(0.1, 0.0, 0.0);
            Eigen::Vector3d axis_y = jn->getParentBodyNode()->getTransform() * jn->getTransformFromParentBodyNode() * Eigen::Vector3d(0.0, 0.1, 0.0);
            Eigen::Vector3d axis_z = jn->getParentBodyNode()->getTransform() * jn->getTransformFromParentBodyNode() * Eigen::Vector3d(0.0, 0.0, 0.1);

            // GUI::DrawSphere(p, 0.01);
            // GUI::DrawLine(p, axis_x, Eigen::Vector3d(1.0, 0.0, 0.0));
            // GUI::DrawLine(p, axis_y, Eigen::Vector3d(0.0, 1.0, 0.0));
            // GUI::DrawLine(p, axis_z, Eigen::Vector3d(0.0, 0.0, 1.0));
        }

        for (auto bn : skel->getBodyNodes())
            drawSingleBodyNode(bn, Eigen::Vector4d(0.1, 0.75, 0.1, 0.25));

        // glColor4f(1.0, 0.0, 0.0, 1.0);

        // for (auto p : mC3DReader->getMarkerPos(mC3DCount))
        //     GUI::DrawSphere(p, 0.01);

        // Draw Attached Marker
        glColor4f(1.0, 0.0, 0.0, 1.0);
        auto ms = mC3DReader->getMarkerSet();
        for (auto m : ms)
            GUI::DrawSphere(m.getGlobalPos(), 0.015);
        // drawThinSkeleton(skel);
        // drawSkeleton(mTestMotion[mC3DCount % mTestMotion.size()], Eigen::Vector4d(1.0, 0.0, 0.0, 0.5));
        // drawSkeleton(mRenderEnv->getCharacter()->sixDofToPos(mC3DReader->mConvertedPos[mC3DCount % mC3DReader->mConvertedPos.size()]), Eigen::Vector4d(1.0, 0.0, 0.0, 0.5));
    }

    // BVH motions now drawn via drawPlayableMotion() (integrated into mMotions)

    if (mMouseDown) drawAxis();

}

// void GLFWApp::drawGround(double height)
// {
//     glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//     glDisable(GL_LIGHTING);
//     double width = 0.005;
//     int count = 0;
//     glBegin(GL_QUADS);
//     for (double x = -100.0; x < 100.01; x += 1.0)
//     {
//         for (double z = -100.0; z < 100.01; z += 1.0)
//         {
//             if (count % 2 == 0)
//                 glColor3f(216.0 / 255.0, 211.0 / 255.0, 204.0 / 255.0);
//             else
//                 glColor3f(216.0 / 255.0 - 0.1, 211.0 / 255.0 - 0.1, 204.0 / 255.0 - 0.1);
//             count++;
//             glVertex3f(x, height, z);
//             glVertex3f(x + 1.0, height, z);
//             glVertex3f(x + 1.0, height, z + 1.0);
//             glVertex3f(x, height, z + 1.0);
//         }
//     }
//     glEnd();
//     glEnable(GL_LIGHTING);
// }

void GLFWApp::drawGround()
{
    // Get ground dimensions from the world
    // Ground is typically the last skeleton added to the world
    Eigen::Vector3d size(100.0, 1.0, 250.0); // Default size
    const double mWidth = size[0];
    const double mHeight = size[1];
    const double depth = size[2];

    glDisable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Draw checkerboard pattern
    glBegin(GL_QUADS);

    Eigen::Vector3d color1,color2;
    color1 << 216.0/255.0, 211.0/255.0, 204.0/255.0;
    color2 << 216.0/255.0-0.1, 211.0/255.0-0.1, 204.0/255.0-0.1;
    const double grid_size = 1.0;

    for(double x=-mWidth/2.0; x<mWidth/2.0 + 0.01; x+=grid_size)
    {
        for(double z=-5.0; z<depth + 0.01 - 5.0; z+=grid_size)
        {
            bool isEven = (int(x/grid_size) + int(z/grid_size)) % 2 == 0;
            if(isEven) glColor4f(color1[0], color1[1], color1[2] ,1.0);
            else glColor4f(color2[0], color2[1], color2[2] ,1.0);
            glVertex3f(x,           0.0, z);
            glVertex3f(x+grid_size, 0.0, z);
            glVertex3f(x+grid_size, 0.0, z+grid_size);
            glVertex3f(x,           0.0, z+grid_size);
        }
    }
    glEnd();

    glEnable(GL_LIGHTING);
}

void GLFWApp::mouseScroll(double xoffset, double yoffset)
{
    if (yoffset < 0)
        mEye *= 1.05;
    else if ((yoffset > 0) && (mEye.norm() > 0.5))
        mEye *= 0.95;
}

void GLFWApp::mouseMove(double xpos, double ypos)
{
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;
    mMouseX = xpos;
    mMouseY = ypos;
    if (mRotate)
    {
        if (deltaX != 0 || deltaY != 0)
            mTrackball.updateBall(xpos, mHeight - ypos);
    }
    if (mTranslate)
    {
        Eigen::Matrix3d rot;
        rot = mTrackball.getRotationMatrix();
        mRelTrans += (1 / mZoom) * rot.transpose() * Eigen::Vector3d(deltaX, -deltaY, 0.0);
    }
    if (mZooming)
        mZoom = std::max(0.01, mZoom + deltaY * 0.01);
}

void GLFWApp::mousePress(int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        mMouseDown = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mRotate = true;
            mTrackball.startBall(mMouseX, mHeight - mMouseY);
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            mTranslate = true;
    }
    else if (action == GLFW_RELEASE)
    {
        mMouseDown = false;
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mRotate = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            mTranslate = false;
    }
}

void GLFWApp::reset()
{
    mSimStepDurationAvg = -1.0;
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mC3DCount = 0;
    mGraphData->clear_all();
    mC3DCOM = Eigen::Vector3d::Zero();

    // Reset motion playback tracking for cycle accumulation
    mLastFrameIdx = 0;
    mCycleAccumulation = Eigen::Vector3d::Zero();
    mCycleAccumulation[0] = 1.0;  // Initial x offset for visualization

    if (mRenderEnv) {
        mRenderEnv->reset(mResetPhase);
        
        mFGNRootOffset = mRenderEnv->getCharacter()->getSkeleton()->getRootJoint()->getPositions().tail(3);
        mUseWeights = mRenderEnv->getUseWeights();
        mViewerTime = mRenderEnv->getWorld()->getTime();
        mViewerPhase = mRenderEnv->getCharacter()->getLocalTime() / (mRenderEnv->getMotion()->getMaxTime() / mRenderEnv->getCadence());

        // Align motion with simulated character position
    }
    alignMotionToSimulation();
}

double GLFWApp::computeFrameFloat(Motion* motion, double phase)
{
    // phase: [0, 1)
    double frame_float;

    std::vector<double> timestamps = motion->getTimestamps();

    if (!timestamps.empty()) {
        // Motion with timestamps: Use actual simulation time for accurate interpolation
        double t_start = timestamps.front();
        double t_end = timestamps.back();
        double total_duration = t_end - t_start;

        // Map phase [0, 1) to one gait cycle worth of time
        double motion_time = t_start + phase * total_duration;

        // Handle wrapping (keep motion_time within valid range)
        motion_time = std::fmod(motion_time - t_start, total_duration) + t_start;

        // Binary search for the frame at or after motion_time
        auto it = std::lower_bound(timestamps.begin(),
                                  timestamps.end(),
                                  motion_time);

        // Calculate frame indices and interpolation weight
        int frame_idx_right = std::distance(timestamps.begin(), it);

        // Clamp to valid range
        if (frame_idx_right >= timestamps.size()) {
            frame_idx_right = timestamps.size() - 1;
        }

        int frame_idx_left = (frame_idx_right > 0) ? frame_idx_right - 1 : 0;

        // Calculate interpolation weight based on timestamps
        double t_left = timestamps[frame_idx_left];
        double t_right = timestamps[frame_idx_right];
        double weight = (frame_idx_left == frame_idx_right) ? 0.0 :
                       (motion_time - t_left) / (t_right - t_left);

        // Set frame_float to maintain compatibility with existing code
        frame_float = frame_idx_left + weight;
    } else {
        // No timestamps: direct frame mapping using timesteps per cycle
        frame_float = phase * motion->getTimestepsPerCycle();
    }

    return frame_float;
}

void GLFWApp::motionPoseEval(Motion* motion, int motionIdx, double frame_float)
{
    if (mMotionsNew.empty() || motionIdx >= mMotionsNew.size()) {
        std::cerr << "[motionPoseEval] Warning: No motions loaded or invalid index" << std::endl;
        return;
    }
    if (!mMotionCharacter) {
        std::cerr << "[motionPoseEval] Warning: No motion character loaded" << std::endl;
        return;
    }

    int frames_per_cycle = motion->getValuesPerFrame();
    int total_frames = motion->getTotalTimesteps();

    // Clamp frame_float to valid range for this motion
    if (frame_float < 0) frame_float = 0;
    if (frame_float >= total_frames) frame_float = fmod(frame_float, total_frames);

    int current_frame_idx = (int)frame_float;
    current_frame_idx = std::max(0, std::min(current_frame_idx, total_frames - 1));

    // 1. Extract and interpolate frame data
    Eigen::VectorXd interpolated_frame;
    double weight_1 = frame_float - floor(frame_float);

    Eigen::VectorXd raw_motion = motion->getRawMotionData();

    // Safety check: ensure motion data is large enough
    int required_size = total_frames * frames_per_cycle;
    if (raw_motion.size() < required_size) {
        std::cerr << "[motionPoseEval] Warning: Motion data too small! Expected " << required_size
                  << " but got " << raw_motion.size() << std::endl;
        mMotionStates[motionIdx].currentPose = Eigen::VectorXd::Zero(frames_per_cycle);
        return;
    }

    if (weight_1 > 1e-6) {
        int next_frame_idx = (current_frame_idx + 1) % total_frames;
        Eigen::VectorXd p1 = raw_motion.segment(current_frame_idx * frames_per_cycle, frames_per_cycle);
        Eigen::VectorXd p2 = raw_motion.segment(next_frame_idx * frames_per_cycle, frames_per_cycle);

        if (motion->getSourceType() == "npz") {
            // NPZ motion: simple linear interpolation (no DOF-wise slerp)
            interpolated_frame = p1 * (1.0 - weight_1) + p2 * weight_1;
        } else {
            // HDF5/BVH motion: use Character's skeleton-aware interpolation
            bool phase_overflow = (next_frame_idx < current_frame_idx);  // Detect cycle wraparound
            interpolated_frame = mMotionCharacter->interpolatePose(p1, p2, weight_1, phase_overflow);
        }
    } else {
        // No interpolation needed, use exact frame
        interpolated_frame = raw_motion.segment(current_frame_idx * frames_per_cycle, frames_per_cycle);
    }

    // 2. Convert to full skeleton pose if NPZ
    Eigen::VectorXd motion_pos;
    if (motion->getSourceType() == "npz") {
        // NPZ: convert from 6D rotation format to angles
        motion_pos = mMotionCharacter->sixDofToPos(interpolated_frame);
    } else {
        // HDF/BVH: already in angle format
        motion_pos = interpolated_frame;
    }

    // 3. Apply position offset (different handling for NPZ vs HDF/BVH)
    if (motion->getSourceType() == "npz") {
        // NPZ: Use mCycleAccumulation only (already accumulated in updateViewerTime)
        motion_pos[3] = mCycleAccumulation[0] + mMotionStates[motionIdx].displayOffset[0];
        motion_pos[4] = mCycleAccumulation[1];
        motion_pos[5] = mCycleAccumulation[2] + mMotionStates[motionIdx].displayOffset[2];
    } else {
        // HDF/BVH: Use delta from initial position plus cycle accumulation
        Eigen::Vector3d current_root_pos(motion_pos[3], motion_pos[4], motion_pos[5]);
        Eigen::Vector3d delta = current_root_pos - mMotionStates[motionIdx].initialRootPosition;

        motion_pos[3] = delta[0] + mCycleAccumulation[0] + mMotionStates[motionIdx].displayOffset[0];
        motion_pos[4] = current_root_pos[1] + mMotionStates[motionIdx].displayOffset[1];
        motion_pos[5] = delta[2] + mCycleAccumulation[2] + mMotionStates[motionIdx].displayOffset[2];
    }

    // Store the computed pose
    mMotionStates[motionIdx].currentPose = motion_pos;
}

void GLFWApp::alignMotionToSimulation()
{
    // Safety check: Skip if no motions loaded or invalid index
    if (mMotionsNew.empty() || mMotionIdx < 0 || mMotionIdx >= mMotionsNew.size()) {
        return;
    }

    // Temporarily clear displayOffset to evaluate raw motion pose
    Eigen::Vector3d saved_offset = mMotionStates[mMotionIdx].displayOffset;
    mMotionStates[mMotionIdx].displayOffset = Eigen::Vector3d::Zero();

    // Calculate the correct frame based on current phase
    double phase = mViewerPhase;
    double frame_float = computeFrameFloat(mMotionsNew[mMotionIdx], phase);

    // Evaluate motion pose at the current time/phase (without displayOffset)
    motionPoseEval(mMotionsNew[mMotionIdx], mMotionIdx, frame_float);

    if (!mRenderEnv || mMotionsNew.empty()) return;

    // Get simulated character's current position (from root body node)
    Eigen::Vector3d sim_pos = mRenderEnv->getCharacter()->getSkeleton()->getRootBodyNode()->getCOM();

    // Calculate displayOffset to align motion with simulation
    if (mMotionStates[mMotionIdx].currentPose.size() > 0) {
        double motion_x = mMotionStates[mMotionIdx].currentPose[3];
        double motion_z = mMotionStates[mMotionIdx].currentPose[5];

        // Set displayOffset: shift X to separate, align Z to coincide
        mMotionStates[mMotionIdx].displayOffset[0] = sim_pos[0] - motion_x + 1.0;  // X: shift apart for visualization
        mMotionStates[mMotionIdx].displayOffset[2] = sim_pos[2] - motion_z;        // Z: align perfectly
    } else {
        // Fallback to default offset if pose evaluation failed
        mMotionStates[mMotionIdx].displayOffset[0] = 1.0;
    }
    motionPoseEval(mMotionsNew[mMotionIdx], mMotionIdx, frame_float);
}

void GLFWApp::updateViewerTime(double dt)
{
    // Update viewer time and phase
    mViewerTime += dt;
    mViewerPhase = fmod(mViewerTime / mViewerCycleDuration, 1.0);

    // Early return if no motions loaded or invalid index
    if (mMotionsNew.empty() || mMotionIdx < 0 || mMotionIdx >= mMotionsNew.size()) {
        if (!mMotionsNew.empty()) {
            std::cout << "[updateViewerTime] Warning: Invalid motion index" << std::endl;
        }
        return;
    }

    // Calculate phase [0, 1) for motion playback
    double phase;
    if (mRenderEnv) {
        phase = mViewerTime / (mRenderEnv->getMotion()->getMaxTime() / (mRenderEnv->getCadence() / sqrt(mRenderEnv->getCharacter()->getGlobalRatio())));
        phase = fmod(phase, 1.0);
    } else {
        if (!mMotionSkeleton) return;
        phase = mViewerPhase;
    }

    // Get current motion
    Motion* current_motion = mMotionsNew[mMotionIdx];
    int value_per_frame = current_motion->getValuesPerFrame();

    // Calculate frame indices based on navigation mode
    double frame_float;
    if (mMotionNavigationMode == NAVIGATION_VIEWER_TIME) {
        // Automatic playback: compute frame from phase/time
        frame_float = computeFrameFloat(current_motion, phase);
    } else {
        // Manual mode: use manually set frame index
        frame_float = (double)mManualFrameIndex;
    }

    int current_frame_idx = (int)frame_float;
    int total_frames = current_motion->getTotalTimesteps();
    current_frame_idx = current_frame_idx % total_frames;

    // Update motion state (different handling for NPZ vs HDF5/BVH)
    Character* character = mRenderEnv ? mRenderEnv->getCharacter() : mMotionCharacter;
    if (!character) return;

    std::string source_type = current_motion->getSourceType();
    if (source_type == "hdfRollout" || source_type == "hdfSingle" || source_type == "bvh") {
        // HDF/BVH: Absolute positions, accumulate only on cycle wrap in automatic mode
        Eigen::VectorXd raw_motion = current_motion->getRawMotionData();
        Eigen::VectorXd current_frame = raw_motion.segment(
            current_frame_idx * value_per_frame, value_per_frame);

        // Detect cycle wrap and accumulate forward progress (only in automatic playback mode)
        // In manual frame mode, user controls frame directly - no accumulation needed
        if (mMotionNavigationMode == NAVIGATION_VIEWER_TIME && current_frame_idx < mLastFrameIdx) {
            Eigen::VectorXd last_frame = raw_motion.segment(
                mLastFrameIdx * value_per_frame, value_per_frame);
            Eigen::Vector3d last_root_pos(last_frame[3], last_frame[4], last_frame[5]);
            const int frame_per_cycle = current_motion->getTimestepsPerCycle();

            mCycleAccumulation[0] += (last_root_pos[0] - mMotionStates[mMotionIdx].initialRootPosition[0]) * (frame_per_cycle) / (frame_per_cycle - 1);
            mCycleAccumulation[2] += (last_root_pos[2] - mMotionStates[mMotionIdx].initialRootPosition[2]) * (frame_per_cycle) / (frame_per_cycle - 1);
        }
        mLastFrameIdx = current_frame_idx;
    } else {
        // NPZ: Incremental data, accumulate every frame in automatic mode only
        Eigen::VectorXd raw_motion = current_motion->getRawMotionData();
        Eigen::VectorXd current_frame = raw_motion.segment(
            current_frame_idx * value_per_frame, value_per_frame);

        Eigen::VectorXd motion_pos = character->sixDofToPos(current_frame);

        // NPZ root positions are deltas - accumulate them directly (only in automatic playback)
        // In manual frame mode, accumulation causes unwanted jumps when moving backward
        if (mMotionNavigationMode == NAVIGATION_VIEWER_TIME) {
            mCycleAccumulation[0] += motion_pos[3] * 0.5;
            mCycleAccumulation[1] = motion_pos[4];  // Y is absolute
            mCycleAccumulation[2] += motion_pos[5] * 0.5;
        }

        mLastFrameIdx = current_frame_idx;
    }

    motionPoseEval(mMotionsNew[mMotionIdx], mMotionIdx, frame_float);
}

void GLFWApp::keyboardPress(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_S:
            if (mRenderEnv) {
                update();  // Advance simulation by one control step
                double dt = 1.0 / mRenderEnv->getControlHz();
                updateViewerTime(dt);  // Advance viewer time and motion state
            } else if (!mMotionsNew.empty() && mMotionIdx >= 0 && mMotionIdx < mMotionsNew.size()) {
                // Step viewer time by single frame duration when no simulation environment
                Motion* current_motion = mMotionsNew[mMotionIdx];
                double frame_duration = 0.0;

                // Use timesteps per cycle for all motion types
                int num_frames = current_motion->getTimestepsPerCycle();
                frame_duration = mViewerCycleDuration / num_frames;

                updateViewerTime(frame_duration);  // Advance by single frame
            }
            break;
        case GLFW_KEY_R:
            if (mods == GLFW_MOD_CONTROL) runRollout();
            else reset();
            break;
        // case GLFW_KEY_O:
            // mDrawOBJ = !mDrawOBJ;
            // break;
        case GLFW_KEY_SPACE:
            mRolloutStatus.pause = !mRolloutStatus.pause;
            mRolloutStatus.cycle = -1;
            break;
        case GLFW_KEY_A:
            mShowResizablePlotPane = !mShowResizablePlotPane;
            break;
        case GLFW_KEY_T:
            mShowTimingPane = !mShowTimingPane;
            break;
        // Camera Setting
        case GLFW_KEY_C:
            printCameraInfo();
            break;
        // case GLFW_KEY_F:
            // mFocus += 1;
            // mFocus %= 5;
            // break;

        case GLFW_KEY_0:
        case GLFW_KEY_KP_0:
            loadCameraPreset(0);
            break;
        case GLFW_KEY_1:
        case GLFW_KEY_KP_1:
            loadCameraPreset(1);
            break;
        case GLFW_KEY_2:
        case GLFW_KEY_KP_2:
            loadCameraPreset(2);
            break;


        default:
            break;
        }
    }
}
void GLFWApp::drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr)
{
    glColor3f(0.5, 0.5, 0.5);
    // Just Connect the joint position
    for (auto jn : skelptr->getJoints())
    {
        // jn position
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        if (jn->getParentBodyNode() == nullptr)
        {
            pos = jn->getTransformFromParentBodyNode().translation();
            pos += jn->getPositions().tail(3);
        }
        // continue;
        else
            pos = jn->getParentBodyNode()->getTransform() * jn->getTransformFromParentBodyNode().translation();

        GUI::DrawSphere(pos, 0.015);

        int j = 0;
        while (true)
        {
            Eigen::Vector3d child_pos;
            if (jn->getChildBodyNode()->getNumChildJoints() > 0)
                child_pos = jn->getChildBodyNode()->getTransform() * jn->getChildBodyNode()->getChildJoint(j)->getTransformFromParentBodyNode().translation();
            else
            {
                child_pos = jn->getChildBodyNode()->getCOM();
                GUI::DrawSphere(child_pos, 0.015);
            }
            double length = (pos - child_pos).norm();

            glPushMatrix();

            // get Angle Axis vector which transform from (0,0,1) to (pos - child_pos) using atan2
            Eigen::Vector3d line = pos - child_pos;
            Eigen::Vector3d axis = Eigen::Vector3d::UnitZ().cross(line.normalized());

            double sin_angle = axis.norm();
            double cos_angle = Eigen::Vector3d::UnitZ().dot(line.normalized());
            double angle = atan2(sin_angle, cos_angle);

            glTranslatef((pos[0] + child_pos[0]) * 0.5, (pos[1] + child_pos[1]) * 0.5, (pos[2] + child_pos[2]) * 0.5);
            glRotatef(angle * 180 / M_PI, axis[0], axis[1], axis[2]);
            GUI::DrawCylinder(0.01, length);
            glPopMatrix();
            j++;

            if (jn->getChildBodyNode()->getNumChildJoints() == j || jn->getChildBodyNode()->getNumChildJoints() == 0)
                break;
        }
    }
}

void GLFWApp::drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color, bool isLineSkeleton)
{
    mMotionSkeleton->setPositions(pos);
    if (!isLineSkeleton)
    {
        for (const auto bn : mMotionSkeleton->getBodyNodes()) drawSingleBodyNode(bn, color);
    }
}

void GLFWApp::drawShape(const Shape *shape, const Eigen::Vector4d &color)
{
    if (!shape) return;

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glColor4d(color[0], color[1], color[2], color[3]);
    if (!mDrawOBJ)
    {

        // glColor4dv(color.data());
        if (shape->is<SphereShape>())
        {
            const auto *sphere = dynamic_cast<const SphereShape *>(shape);
            GUI::DrawSphere(sphere->getRadius());
        }
        else if (shape->is<BoxShape>()) 
        {
            const auto *box = dynamic_cast<const BoxShape *>(shape);
            GUI::DrawCube(box->getSize());
        }
        else if (shape->is<CapsuleShape>()) 
        {
            const auto *capsule = dynamic_cast<const CapsuleShape *>(shape);
            GUI::DrawCapsule(capsule->getRadius(), capsule->getHeight());
        }
        else if (shape->is<CylinderShape>()) 
        {
            const auto *cylinder = dynamic_cast<const CylinderShape *>(shape);
            GUI::DrawCylinder(cylinder->getRadius(), cylinder->getHeight());
        }
    }
    else
    {
        if (shape->is<MeshShape>()) 
        {
            const auto &mesh = dynamic_cast<const MeshShape *>(shape);
            mShapeRenderer.renderMesh(mesh, false, 0.0, color);
        }
    }
}

void GLFWApp::setCamera()
{
    if (mRenderEnv)
    {
        if (mFocus == 1)
        {
            mTrans = -mRenderEnv->getCharacter()->getSkeleton()->getCOM();
            mTrans[1] = -1;
            mTrans *= 1000;
        }
        else if (mFocus == 2)
        {
            mTrans = -mRenderEnv->getTargetPositions().segment(3, 3); //-mRenderEnv->getCharacter()->getSkeleton()->getCOM();
            mTrans[1] = -1;
            mTrans *= 1000;
        }
        else if (mFocus == 3)
        {
            if (mC3dMotion.size() == 0)
                mFocus++;
            else
            {
                mTrans = -(mC3DReader->getBVHSkeleton()->getCOM());
                mTrans[1] = -1;
                mTrans *= 1000;
            }
        }
        else if (mFocus == 4)
        {
            mTrans[0] = -mFGNRootOffset[0];
            mTrans[1] = -1;
            mTrans[2] = -mFGNRootOffset[2];
            mTrans *= 1000;
        }
    }
    else
    {
        // Motion-only viewing mode: focus on current motion position
        if (!mMotionsNew.empty() && mMotionSkeleton) {
            // Calculate current position based on cycle accumulation
            double phase = mViewerPhase;
            Motion* current_motion = mMotionsNew[mMotionIdx];

            double frame_float;
            if (mMotionNavigationMode == NAVIGATION_VIEWER_TIME) {
                frame_float = computeFrameFloat(current_motion, phase);
            } else {
                frame_float = (double)mManualFrameIndex;
            }

            int current_frame_idx = (int)frame_float;
            int total_frames = current_motion->getTotalTimesteps();
            current_frame_idx = current_frame_idx % total_frames;

            Eigen::VectorXd raw_motion = current_motion->getRawMotionData();
            int value_per_frame = current_motion->getValuesPerFrame();
            Eigen::VectorXd current_frame = raw_motion.segment(
                current_frame_idx * value_per_frame, value_per_frame);

            Eigen::VectorXd current_pos;
            if (current_motion->getSourceType() == "npz") {
                current_pos = mMotionCharacter->sixDofToPos(current_frame);
            } else {
                current_pos = current_frame;
            }

            if (current_motion->getSourceType() == "npz") {
                // NPZ: Use mCycleAccumulation only
                mTrans[0] = -(mCycleAccumulation[0] + mMotionStates[mMotionIdx].displayOffset[0]);
                mTrans[1] = -(mCycleAccumulation[1] + mMotionStates[mMotionIdx].displayOffset[1]) - 1;
                mTrans[2] = -(mCycleAccumulation[2] + mMotionStates[mMotionIdx].displayOffset[2]);
            } else {
                // HDF: Use delta from initial position plus cycle accumulation
                Eigen::Vector3d root_pos(current_pos[3], current_pos[4], current_pos[5]);
                Eigen::Vector3d delta = root_pos - mMotionStates[mMotionIdx].initialRootPosition;

                mTrans[0] = -(delta[0] + mCycleAccumulation[0] + mMotionStates[mMotionIdx].displayOffset[0]);
                mTrans[1] = -(root_pos[1] + mMotionStates[mMotionIdx].displayOffset[1]) - 1;
                mTrans[2] = -(delta[2] + mCycleAccumulation[2] + mMotionStates[mMotionIdx].displayOffset[2]);
            }
        } else {
            mTrans = Eigen::Vector3d::Zero();
            mTrans[1] = -1;
        }
        mTrans *= 1000;
    }
}

void GLFWApp::drawCollision()
{
    const auto result = mRenderEnv->getWorld()->getConstraintSolver()->getLastCollisionResult();
    for (const auto &contact : result.getContacts())
    {
        Eigen::Vector3d v = contact.point;
        Eigen::Vector3d f = contact.force / 1000.0;
        glLineWidth(2.0);
        glColor3f(0.8, 0.8, 0.2);
        glBegin(GL_LINES);
        glVertex3f(v[0], v[1], v[2]);
        glVertex3f(v[0] + f[0], v[1] + f[1], v[2] + f[2]);
        glEnd();
        glColor3f(0.8, 0.8, 0.2);
        glPushMatrix();
        glTranslated(v[0], v[1], v[2]);
        GUI::DrawSphere(0.01);
        glPopMatrix();
    }
}

void GLFWApp::drawMuscles(MuscleRenderingType renderingType)
{
    int count = 0;
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    auto muscles = mRenderEnv->getCharacter()->getMuscles();
    for (int i = 0; i < muscles.size(); i++)
    {
        // Skip if muscle is not selected (using same order as environment)
        if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

        auto muscle = muscles[i];
        muscle->UpdateGeometry();
        double a = muscle->activation;
        Eigen::Vector4d color;
        switch (renderingType)
        {
        case activationLevel:
            color = Eigen::Vector4d(0.2 + 1.6 * a, 0.2, 0.2, mMuscleTransparency + 0.9 * a);
            break;
        case passiveForce:
        {
            double f_p = muscle->Getf_p() / mMuscleResolution;
            color = Eigen::Vector4d(0.1, 0.1, 0.1 + 0.9 * f_p, mMuscleTransparency + f_p);
            break;
        }
        case contractileForce:
        {
            double f_c = muscle->Getf_A() * a / mMuscleResolution;
            color = Eigen::Vector4d(0.1, 0.1 + 0.9 * f_c, 0.1, mMuscleTransparency + f_c);
            break;
        }
        case weakness:
        {
            color = Eigen::Vector4d(0.1, 0.1 + 2.0 * (1.0 - muscle->f0 / muscle->f0_base), 0.1 + 2.0 * (1.0 - muscle->f0 / muscle->f0_base), mMuscleTransparency + 2.0 * (1.0 - muscle->f0 / muscle->f0_base));
            break;
        }
        case contracture:
        {
            color = Eigen::Vector4d(0.05 + 10.0 * (1.0 - muscle->lmt_ref / muscle->lmt_base), 0.05, 0.05 + 10.0 * (1.0 - muscle->lmt_ref / muscle->lmt_base), mMuscleTransparency + 5.0 * (1.0 - muscle->lmt_ref / muscle->lmt_base));
            break;
        }
        default:
            color.setOnes();
            break;
        }
        glColor4dv(color.data());
        mShapeRenderer.renderMuscle(muscle, -1.0);
    }
    glEnable(GL_LIGHTING);
}

void GLFWApp::drawFootStep()
{
    Eigen::Vector3d current_foot = mRenderEnv->getCurrentFootStep();
    glColor4d(0.2, 0.2, 0.8, 0.5);
    glPushMatrix();
    glTranslated(0, current_foot[1], current_foot[2]);
    GUI::DrawCube(Eigen::Vector3d(1.0, 0.15, 0.15));
    glPopMatrix();

    Eigen::Vector3d target_foot = mRenderEnv->getCurrentTargetFootStep();
    glColor4d(0.2, 0.8, 0.2, 0.5);
    glPushMatrix();
    glTranslated(0, target_foot[1], target_foot[2]);
    GUI::DrawCube(Eigen::Vector3d(1.0, 0.15, 0.15));
    glPopMatrix();

    Eigen::Vector3d next_foot = mRenderEnv->getNextTargetFootStep();
    glColor4d(0.8, 0.2, 0.2, 0.5);
    glPushMatrix();
    glTranslated(0, next_foot[1], next_foot[2]);
    GUI::DrawCube(Eigen::Vector3d(1.0, 0.15, 0.15));
    glPopMatrix();
}

void GLFWApp::drawShadow()
{
    Eigen::VectorXd pos = mRenderEnv->getCharacter()->getSkeleton()->getPositions();

    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glPushMatrix();
    glTranslatef(pos[3], 2E-3, pos[5]);
    glScalef(1.0, 1E-4, 1.0);
    glRotatef(30.0, 1.0, 0.0, 1.0);
    glTranslatef(-pos[3], 0.0, -pos[5]);
    drawSkeleton(pos, Eigen::Vector4d(0.1, 0.1, 0.1, 1.0));
    glPopMatrix();
    glEnable(GL_LIGHTING);
}

void GLFWApp::loadNetworkFromPath(const std::string& path)
{
    if (loading_network.is_none()) {
        std::cerr << "Warning: loading_network not available, skipping: " << path << std::endl;
        return;
    }

    try {
        auto character = mRenderEnv->getCharacter();
        Network new_elem;
        new_elem.name = path;
        
        py::tuple res = loading_network(
            path.c_str(),
            mRenderEnv->getState().rows(),
            mRenderEnv->getAction().rows(),
            (character->getActuatorType() == mass || character->getActuatorType() == mass_lower)
        );
        
        new_elem.joint = res[0];
        new_elem.muscle = res[1];
        mNetworks.push_back(new_elem);
    } catch (const std::exception& e) {
        std::cerr << "Error loading network from " << path << ": " << e.what() << std::endl;
    }
}

void GLFWApp::initializeMotionSkeleton()
{
    mMotionSkeleton = mRenderEnv->getCharacter()->getSkeleton()->cloneSkeleton();
    
    // Setup BVH joint calibration
    mJointCalibration.clear();
    for (auto jn : mRenderEnv->getCharacter()->getSkeleton()->getJoints()) {
        if (jn == mRenderEnv->getCharacter()->getSkeleton()->getRootJoint()) {
            mJointCalibration.push_back(Eigen::Matrix3d::Identity());
        } else {
            mJointCalibration.push_back(
                (jn->getTransformFromParentBodyNode() * jn->getParentBodyNode()->getTransform()).linear().transpose()
            );
        }
    }

    // Setup skeleton info for motions
    mSkelInfosForMotions.clear();
    for (auto bn : mMotionSkeleton->getBodyNodes()) {
        ModifyInfo skelInfo;
        mSkelInfosForMotions.push_back(std::make_pair(bn->getName(), skelInfo));
    }
}

void GLFWApp::loadNPZMotion()
{
	std::string motion_path = "data/npz_motions";
	if (!fs::exists(motion_path) || !fs::is_directory(motion_path)) {
		std::cerr << "Motion directory not found: " << motion_path << std::endl;
		return;
	}

	for (const auto &entry : fs::directory_iterator(motion_path)) {
		std::string file_path = entry.path().string();
		std::string ext = entry.path().extension().string();
		if (ext != ".npz")
			continue;

		try {
			// NEW ARCHITECTURE: Create NPZ* and keep it alive
			NPZ* npz = new NPZ(file_path);
			npz->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());

			// Store in new motion architecture
			mMotionsNew.push_back(npz);

			// Create viewer state
			MotionViewerState state;
			state.displayOffset = Eigen::Vector3d(1.0, 0, 0);  // Default offset
			state.initialRootPosition = Eigen::Vector3d::Zero();  // NPZ doesn't use this
			mMotionStates.push_back(state);

			if (npz->hasParameters()) {
				LOG_VERBOSE(npz->getLogHeader() << " Loaded " << npz->getName() << " with " << npz->getNumFrames() << " frames (" << npz->getParameterValues().size() << " parameters)");
			} else {
				LOG_VERBOSE(npz->getLogHeader() << " Loaded " << npz->getName() << " with " << npz->getNumFrames() << " frames");
			}
			if (!npz->hasParameters()) {
				LOG_WARN("[" << npz->getName() << "] Warning: No parameters in motion file");
			}


		} catch (const std::exception& e) {
			std::cerr << "[NPZ] Error loading " << file_path << ": " << e.what() << std::endl;
			continue;
		}
	}
}

void GLFWApp::loadHDFRolloutMotion()
{
	scanHDF5Structure();

	// NEW ARCHITECTURE: Create HDFRollout instances
	std::string rollout_dir = "data/trained_nn";
	std::vector<std::string> rollout_files;

	if (fs::exists(rollout_dir) && fs::is_directory(rollout_dir)) {
		for (const auto &entry : fs::directory_iterator(rollout_dir)) {
			if (fs::is_regular_file(entry)) {
				std::string ext = entry.path().extension().string();
				std::string filename = entry.path().filename().string();

				// Look for rollout files (typically have _rollout in name or are larger HDF5 files)
				if ((ext == ".h5" || ext == ".hdf5") &&
				    (filename.find("rollout") != std::string::npos || filename.find("trained") != std::string::npos)) {
					rollout_files.push_back(entry.path().string());
				}
			}
		}
	}

	if (!rollout_files.empty() && mRenderEnv) {
		std::cout << "[HDF Rollout] Loading " << rollout_files.size() << " rollout files..." << std::endl;  // Static header, no motion object yet

		for (const auto& rollout_path : rollout_files) {
			try {
				// Create HDFRollout instance
				HDFRollout* rollout = new HDFRollout(rollout_path);

				// Load first param/cycle combination by default
				if (rollout->getNumParams() > 0) {
					rollout->loadParamCycle(0, 0);
					rollout->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());

					// Store in new motion architecture
					mMotionsNew.push_back(rollout);

					// Create viewer state
					MotionViewerState state;
					Eigen::VectorXd first_pose = rollout->getPose(0);
					state.displayOffset = Eigen::Vector3d(1.0, 0, 0);  // Default offset
					if (first_pose.size() >= 6) {
						state.initialRootPosition = Eigen::Vector3d(first_pose[3], first_pose[4], first_pose[5]);
					}
					mMotionStates.push_back(state);

					if (rollout->hasParameters()) {
						LOG_VERBOSE(rollout->getLogHeader() << " Loaded " << rollout->getName()
							  << " with " << rollout->getNumParams() << " params, "
							  << rollout->getNumCycles() << " cycles (" << rollout->getParameterNames().size() << " parameter names)");
					} else {
						LOG_VERBOSE(rollout->getLogHeader() << " Loaded " << rollout->getName()
							  << " with " << rollout->getNumParams() << " params, "
							  << rollout->getNumCycles() << " cycles");
					}
				} else {
					std::cerr << rollout->getLogHeader() << " Warning: " << rollout_path << " has no param groups" << std::endl;
					delete rollout;  // No params, don't keep
				}

			} catch (const std::exception& e) {
				std::cerr << "[HDF Rollout] Error loading " << rollout_path << ": " << e.what() << std::endl;  // Static header, rollout may not exist
			}
		}
	}
}


void GLFWApp::loadBVHMotion()
{
	// Scan for BVH files in data/motion directory
	std::vector<std::string> bvh_files;
	std::string motion_dir = "data/motion";
	if (fs::exists(motion_dir) && fs::is_directory(motion_dir)) {
		for (const auto &entry : fs::directory_iterator(motion_dir)) {
			if (fs::is_regular_file(entry)) {
				std::string filename = entry.path().filename().string();
				if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".bvh") {
					bvh_files.push_back(entry.path().string());
				}
			}
		}
	}

	if (!bvh_files.empty() && mRenderEnv) {
		LOG_INFO("[BVH] Loading " << bvh_files.size() << " BVH files into mMotions...");

		for (const auto& bvh_path : bvh_files) {
			try {
				// NEW ARCHITECTURE: Create BVH* and keep it alive
				BVH* bvh = new BVH(bvh_path);
				bvh->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());

				// Store in new motion architecture
				mMotionsNew.push_back(bvh);

				// Create viewer state
				MotionViewerState state;
				Eigen::VectorXd first_pose = bvh->getPose(0);
				state.displayOffset = Eigen::Vector3d(1.0, 0, 0);  // Default offset
				if (first_pose.size() >= 6) {
					state.initialRootPosition = Eigen::Vector3d(first_pose[3], first_pose[4], first_pose[5]);
				}
				mMotionStates.push_back(state);

				LOG_VERBOSE(bvh->getLogHeader() << " Loaded " << bvh->getName() << " with " << bvh->getNumFrames() << " frames");
			// Note: BVH files don't have parameters by design, so no warning needed

			} catch (const std::exception& e) {
				std::cerr << "[BVH] Error loading " << bvh_path << ": " << e.what() << std::endl;
			}
		}
	}
}

void GLFWApp::loadHDFSingleMotion()
{
	std::vector<std::string> hdf_single_files;
	std::string motion_dir = "data/motion";

	if (fs::exists(motion_dir) && fs::is_directory(motion_dir)) {
		for (const auto &entry : fs::directory_iterator(motion_dir)) {
			if (fs::is_regular_file(entry)) {
				std::string ext = entry.path().extension().string();
				if (ext == ".h5" || ext == ".hdf5") {
					hdf_single_files.push_back(entry.path().string());
				}
			}
		}
	}

	if (!hdf_single_files.empty() && mRenderEnv) {
		LOG_INFO("[HDF Single] Loading " << hdf_single_files.size() << " extracted HDF5 files...");

		for (const auto& hdf_path : hdf_single_files) {
			try {
				// NEW ARCHITECTURE: Create HDF* and keep it alive
				HDF* hdf = new HDF(hdf_path);
				hdf->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());

				// Store in new motion architecture
				mMotionsNew.push_back(hdf);

				// Create viewer state
				MotionViewerState state;
				Eigen::VectorXd first_pose = hdf->getPose(0);
				if (first_pose.size() >= 6) {
					state.displayOffset = Eigen::Vector3d(1.0, 0, 0);  // Default offset
					state.initialRootPosition = Eigen::Vector3d(first_pose[3], first_pose[4], first_pose[5]);
				}
				mMotionStates.push_back(state);

				if (hdf->hasParameters()) {
					LOG_VERBOSE(hdf->getLogHeader() << " Loaded " << hdf->getName() << " with " << hdf->getNumFrames() << " frames (" << hdf->getParameterNames().size() << " parameters)");
				} else {
					LOG_VERBOSE(hdf->getLogHeader() << " Loaded " << hdf->getName() << " with " << hdf->getNumFrames() << " frames");
				}
				if (!hdf->hasParameters()) {
					LOG_WARN("[" << hdf->getName() << "] Warning: No parameters in motion file");
				}


			} catch (const std::exception& e) {
				std::cerr << "[HDF Single] Error loading " << hdf_path << ": " << e.what() << std::endl;
			}
		}
	}
}

void GLFWApp::loadMotionFiles()
{
	py::gil_scoped_acquire gil;

	// Clear motions (mMotionsNew now used)
	mMotionIdx = 0;

	// Check motion load mode from config
	if (mMotionLoadMode == "no") {
		LOG_INFO("[Motion] Motion loading disabled");
		return;
	}

	loadHDFSingleMotion();
	loadBVHMotion();
	loadNPZMotion();
	loadHDFRolloutMotion();
}

void GLFWApp::scanHDF5Structure()
{
    mHDF5Files.clear();
    mHDF5Params.clear();
    mHDF5Cycles.clear();
    mSelectedHDF5FileIdx = -1;
    mSelectedHDF5ParamIdx = -1;
    mSelectedHDF5CycleIdx = -1;

    // Only scan sampled/ directory for hdfRollout files (nested structure)
    // Note: hdfSingle files (flat structure from data/motion/) are loaded separately
    std::string search_path = "sampled";

    if (!fs::exists(search_path) || !fs::is_directory(search_path)) {
        std::cout << "[Motion] HDF5 rollout directory not found: " << search_path << std::endl;
        return;
    }

    try {
        for (const auto &entry : fs::recursive_directory_iterator(search_path)) {
            std::string file_name = entry.path().string();
            if (file_name.find("rollout_data.h5") != std::string::npos) {
                mHDF5Files.push_back(file_name);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning " << search_path << " for HDF5 rollout files: " << e.what() << std::endl;
    }

    std::cout << "[Motion] Found " << mHDF5Files.size() << " HDF5 rollout files" << std::endl;
}

// Removed: loadHDF5Parameters() - now using Motion::applyParametersToEnvironment() instead

void GLFWApp::loadParametersFromCurrentMotion()
{
    if (!mRenderEnv) {
        std::cerr << "Render environment not initialized" << std::endl;
        return;
    }

    if (mMotionIdx < 0 || mMotionIdx >= mMotionsNew.size()) {
        std::cerr << "No motion selected or invalid motion index" << std::endl;
        return;
    }

    Motion* motion = mMotionsNew[mMotionIdx];

    if (!motion->hasParameters()) {
        std::cerr << "Current motion (" << motion->getName() << ") has no parameters" << std::endl;
        return;
    }

    std::cout << "Loading parameters from motion: " << motion->getName() << std::endl;

    try {
        // Get parameters from motion
        std::vector<std::string> hdf5_param_names = motion->getParameterNames();
        std::vector<float> hdf5_param_values = motion->getParameterValues();

        if (hdf5_param_names.size() != hdf5_param_values.size()) {
            std::cerr << "Error: Parameter names count (" << hdf5_param_names.size()
                      << ") != values count (" << hdf5_param_values.size() << ")" << std::endl;
            return;
        }

        // Get simulation parameters
        const std::vector<std::string>& sim_param_names = mRenderEnv->getParamName();
        Eigen::VectorXd current_params = mRenderEnv->getParamState();

        std::cout << "  Motion has " << hdf5_param_names.size() << " parameters" << std::endl;
        std::cout << "  Simulation has " << sim_param_names.size() << " parameters" << std::endl;

        // Match and rebuild parameter vector
        Eigen::VectorXd new_params = current_params;  // Start with current values
        int matched_count = 0;

        for (size_t i = 0; i < hdf5_param_names.size(); i++) {
            for (size_t j = 0; j < sim_param_names.size(); j++) {
                if (hdf5_param_names[i] == sim_param_names[j]) {
                    new_params[j] = static_cast<double>(hdf5_param_values[i]);
                    matched_count++;
                    break;
                }
            }
        }

        std::cout << "  Matched " << matched_count << " parameters by name" << std::endl;

        // Apply parameters to simulation environment
        mRenderEnv->setParamState(new_params, false, true);

        std::cout << "✓ Successfully loaded parameters from " << motion->getSourceType()
                  << " motion: " << motion->getName() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading parameters from motion: " << e.what() << std::endl;
    }
}

#if 0  // TODO: Update for Motion* interface
void GLFWApp::loadSelectedHDF5Motion()
{
    if (mSelectedHDF5FileIdx < 0) {
        std::cerr << "No file selected" << std::endl;
        return;
    }

    std::string file_path = mHDF5Files[mSelectedHDF5FileIdx];
    std::string param_name = "param_" + std::to_string(mSelectedHDF5ParamIdx);
    std::string cycle_name = "cycle_" + std::to_string(mSelectedHDF5CycleIdx);

    std::cout << "Loading: " << file_path << " / " << param_name << " / " << cycle_name << std::endl;

    try {
        H5::H5File h5file(file_path, H5F_ACC_RDONLY);
        H5::Group param_group = h5file.openGroup(param_name);

        // Check parameter-level failure first
        if (param_group.attrExists("success")) {
            H5::Attribute success_attr = param_group.openAttribute("success");
            bool success = true;
            success_attr.read(H5::PredType::NATIVE_HBOOL, &success);
            success_attr.close();

            if (!success) {
                std::string error_msg = "Parameter " + std::to_string(mSelectedHDF5ParamIdx) + " failed during rollout";
                mParamFailureMessage = error_msg;
                std::cerr << "Error: " << error_msg << std::endl;
                std::cerr << "Keeping previous motion." << std::endl;

                // Close HDF5 resources and return without loading
                param_group.close();
                h5file.close();
                return;
            }
        }

        // Clear parameter error on successful validation
        mParamFailureMessage = "";

        H5::Group cycle_group = param_group.openGroup(cycle_name);

        // Read motions dataset
        if (!cycle_group.nameExists("motions")) {
            std::cerr << "No motions dataset in " << cycle_name << std::endl;
            return;
        }

        H5::DataSet motions_dataset = cycle_group.openDataSet("motions");
        H5::DataSpace dataspace = motions_dataset.getSpace();

        // Get dimensions
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        int num_steps = dims[0];
        int motion_dim = dims[1];

        // Validation: check if motion has enough timesteps for interpolation
        const int MIN_TIMESTEPS = 20;
        if (num_steps < MIN_TIMESTEPS) {
            std::string error_msg = "Motion too short: " + std::to_string(num_steps) + " timesteps (minimum " + std::to_string(MIN_TIMESTEPS) + " required)";
            mMotionLoadError = error_msg;
            std::cerr << "Error: " << error_msg << std::endl;
            std::cerr << "Keeping previous motion." << std::endl;

            // Close HDF5 resources and return without loading
            dataspace.close();
            motions_dataset.close();
            cycle_group.close();
            param_group.close();
            h5file.close();
            return;
        }

        // Clear error on successful validation
        mMotionLoadError = "";

        // Read motion data
        std::vector<double> motion_data(num_steps * motion_dim);
        motions_dataset.read(motion_data.data(), H5::PredType::NATIVE_DOUBLE);

        // Convert to Eigen matrix
        Eigen::MatrixXd cycle_motion(num_steps, motion_dim);
        for (int i = 0; i < num_steps; i++) {
            for (int j = 0; j < motion_dim; j++) {
                cycle_motion(i, j) = motion_data[i * motion_dim + j];
            }
        }

        // Load timestamps
        std::vector<double> cycle_timestamps;
        if (cycle_group.nameExists("time")) {
            H5::DataSet time_ds = cycle_group.openDataSet("time");
            cycle_timestamps.resize(num_steps);
            time_ds.read(cycle_timestamps.data(), H5::PredType::NATIVE_DOUBLE);
            time_ds.close();
        }

        // Create ViewerMotion
        ViewerMotion motion_elem;
        motion_elem.name = file_path + "_" + param_name + "_" + cycle_name;
        motion_elem.source_type = "hdfRollout";
        motion_elem.values_per_frame = motion_dim;
        motion_elem.num_frames = 1;  // Single cycle
        motion_elem.hdf5_total_timesteps = num_steps;
        motion_elem.hdf5_timesteps_per_cycle = num_steps;

        // Flatten motion data
        motion_elem.motion = Eigen::VectorXd::Zero(num_steps * motion_dim);
        int offset = 0;
        for (int i = 0; i < num_steps; i++) {
            for (int j = 0; j < motion_dim; j++) {
                motion_elem.motion[offset++] = cycle_motion(i, j);
            }
        }

        // Store timestamps
        motion_elem.timestamps = cycle_timestamps;

        // Set parameters
        motion_elem.param = Eigen::VectorXd::Zero(mRenderEnv ? mRenderEnv->getNumKnownParam() : 10);

        // Extract first frame root position
        Eigen::VectorXd first_frame = motion_elem.motion.segment(0, motion_dim);
        motion_elem.initialRootPosition = Eigen::Vector3d(first_frame[3], first_frame[4], first_frame[5]);

        // Calculate display offset
        if (mRenderEnv && mRenderEnv->getCharacter()) {
            Eigen::VectorXd char_positions = mRenderEnv->getCharacter()->getSkeleton()->getPositions();
            Eigen::Vector3d char_root(char_positions[3], char_positions[4], char_positions[5]);
            motion_elem.displayOffset = char_root - motion_elem.initialRootPosition + Eigen::Vector3d(1.0, 0, 0);
        } else {
            motion_elem.displayOffset = Eigen::Vector3d(1.0, 0, 0);
        }

        // Add to motions list
        mMotions.push_back(motion_elem);
        mMotionIdx = mMotions.size() - 1;  // Select the newly loaded motion

        // Update max frame index for manual navigation
        mMaxFrameIndex = motion_elem.hdf5_total_timesteps - 1;
        mManualFrameIndex = 0;  // Reset to first frame

        if (!cycle_timestamps.empty()) {
            LOG_VERBOSE("  Loaded: " << num_steps << " timesteps, " << motion_dim << " DOF, time range: [" << cycle_timestamps.front() << ", " << cycle_timestamps.back() << "] seconds");
        } else {
            LOG_VERBOSE("  Loaded: " << num_steps << " timesteps, " << motion_dim << " DOF");
        }

        dataspace.close();
        motions_dataset.close();
        cycle_group.close();
        param_group.close();
        h5file.close();

    } catch (const std::exception& e) {
        mMotionLoadError = std::string("Load failed: ") + e.what();
        std::cerr << "Error loading motion: " << e.what() << std::endl;
    }
}
#endif  // loadSelectedHDF5Motion

#if 0  // TODO: Update for Motion* interface
void GLFWApp::addSimulationMotion()
{
    ViewerMotion current_motion;
    current_motion.name = "New Motion " + std::to_string(mMotions.size());
    current_motion.param = mRenderEnv->getParamState();
    current_motion.source_type = "simulation";
    current_motion.num_frames = 60;  // Default: 2 seconds at 30Hz
    current_motion.values_per_frame = 101;  // Default frame count
    current_motion.motion = Eigen::VectorXd::Zero(current_motion.num_frames * current_motion.values_per_frame);

    std::vector<double> phis;
    // phis list of 1/60 for 2 seconds
    for (int i = 0; i < 60; i++)
        phis.push_back(((double)i) / mRenderEnv->getControlHz());

    // rollout
    std::vector<Eigen::VectorXd> current_trajectory;
    std::vector<double> current_phi;
    std::vector<Eigen::VectorXd> refined_trajectory;
    reset();

    double prev_phi = -1.0;
    int phi_offset = -1.0;
    while (!mRenderEnv->isEOE())
    {
        for (int i = 0; i < 60 / mRenderEnv->getControlHz(); i++)
            update();

        current_trajectory.push_back(mRenderEnv->getCharacter()->posToSixDof(mRenderEnv->getCharacter()->getSkeleton()->getPositions()));

        if (prev_phi > mRenderEnv->getNormalizedPhase())
            phi_offset += 1;
        prev_phi = mRenderEnv->getNormalizedPhase();

        current_phi.push_back(mRenderEnv->getNormalizedPhase() + phi_offset);
    }

    int phi_idx = 0;
    int current_idx = 0;
    refined_trajectory.clear();
    while (phi_idx < phis.size() && current_idx < current_trajectory.size() - 1)
    {
        // if phi is smaller than current phi, then add current trajectory to refined trajectory
        if (current_phi[current_idx] <= phis[phi_idx] && phis[phi_idx] < current_phi[current_idx + 1])
        {
            // Interpolate between current_idx and current_idx+1
            double t = (phis[phi_idx] - current_phi[current_idx]) / (current_phi[current_idx + 1] - current_phi[current_idx]);
            // calculate v
            Eigen::Vector3d v1 = current_trajectory[current_idx].segment(6, 3) - current_trajectory[current_idx - 1].segment(6, 3);
            Eigen::Vector3d v2 = current_trajectory[current_idx + 1].segment(6, 3) - current_trajectory[current_idx].segment(6, 3);

            Eigen::VectorXd interpolated = (1 - t) * current_trajectory[current_idx] + t * current_trajectory[current_idx + 1];
            Eigen::Vector3d v = (1 - t) * v1 + t * v2;

            interpolated[6] = v[0];
            interpolated[8] = v[2];
            int start_idx = interpolated.rows() * refined_trajectory.size();
            current_motion.motion.segment(start_idx, interpolated.rows()) = interpolated;
            refined_trajectory.push_back(interpolated);

            phi_idx++;
        }
        else
            current_idx++;
    }
    mMotions.push_back(current_motion);
    mAddedMotions.push_back(current_motion);
}
#endif  // addSimulationMotion

void GLFWApp::unloadMotion()
{
    // Delete all loaded Motion* instances
    for (Motion* motion : mMotionsNew) {
        delete motion;
    }
    mMotionsNew.clear();
    mMotionStates.clear();

    // Reset motion playback indices (-1 indicates no motion selected)
    mMotionIdx = -1;
    mMotionFrameIdx = 0;
    mLastFrameIdx = 0;
    mCycleAccumulation = Eigen::Vector3d::Zero();

    // Reset HDF5 rollout selection indices
    mSelectedHDF5FileIdx = -1;
    mSelectedHDF5ParamIdx = 0;
    mSelectedHDF5CycleIdx = 0;
    mMaxHDF5ParamIdx = 0;
    mMaxHDF5CycleIdx = 0;

    // Clear HDF5 file lists
    mHDF5Files.clear();
    mHDF5Params.clear();
    mHDF5Cycles.clear();

    // Clear error messages
    mMotionLoadError.clear();
    mParamFailureMessage.clear();
    mLastLoadedHDF5ParamsFile.clear();

    // Reset manual navigation state
    mManualFrameIndex = 0;
    mMaxFrameIndex = 0;

    // Reset simulation parameters to default from XML metadata
    if (mRenderEnv) {
        Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
        mRenderEnv->setParamState(default_params, false, true);
        mRenderEnv->getCharacter()->updateRefSkelParam(mMotionSkeleton);
        std::cout << "[Motion] All motions unloaded, parameters reset to defaults" << std::endl;
    }
}

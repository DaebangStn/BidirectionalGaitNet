#include "GLFWApp.h"
#include "UriResolver.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <filesystem>
#include <algorithm>


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

GLFWApp::GLFWApp(int argc, char **argv, bool rendermode)
{
    mRenderEnv = nullptr;
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
    mPlotTitle = false;

    // Load configuration from render.yaml (will override defaults if file exists)
    loadRenderConfig();

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

    mDrawMotion = false;

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
    mMotionRootOffset = Eigen::Vector3d::Zero();


    py::gil_scoped_acquire gil;
    
    // Import Python modules
    try {
        loading_network = py::module::import("python.ray_model").attr("loading_network");
    } catch (const py::error_already_set& e) {
        std::cerr << "Warning: Failed to import python.ray_model: " << e.what() << std::endl;
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
                    mCachedMetadata = py_metadata.cast<std::string>();
                }
            } catch (const py::error_already_set& e) {
                std::cerr << "Warning: Failed to load metadata from network path: " << e.what() << std::endl;
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

        std::cout << "[Config] Loading render config from: " << resolved_path << std::endl;

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
        }

        std::cout << "[Config] Loaded - Window: " << mWidth << "x" << mHeight
                  << ", Control: " << mControlPanelWidth
                  << ", Plot: " << mPlotPanelWidth
                  << ", Rollout: " << mDefaultRolloutCount << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Config] Warning: Could not load render.yaml: " << e.what() << std::endl;
        std::cerr << "[Config] Using default values." << std::endl;
    }
}

GLFWApp::~GLFWApp()
{
    delete mRenderEnv;
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
        _f << "{" << std::endl;
        Eigen::Vector3d current_joint = jn->getParentBodyNode()->getTransform() * (jn->getTransformFromParentBodyNode() * Eigen::Vector3d::Zero());
        Eigen::Vector3d parent_joint = jn->getParentBodyNode()->getTransform() * ((jn->getParentBodyNode()->getParentJoint())->getTransformFromChildBodyNode() * Eigen::Vector3d::Zero());

        Eigen::Vector3d offset = current_joint - parent_joint; // jn->getTransformFromParentBodyNode() * ((bn->getParentJoint()->getTransformFromChildBodyNode()).inverse() * Eigen::Vector3d::Zero());
        offset *= 100.0;
        _f << "OFFSET\t" << offset.transpose() << std::endl;
        _f << "CHANNELS\t" << 3 << "\t" << CHANNELS[5] << "\t" << CHANNELS[3] << "\t" << CHANNELS[4] << std::endl;

        if (jn->getChildBodyNode()->getNumChildBodyNodes() == 0)
        {
            _f << "End Site" << std::endl;
            _f << "{" << std::endl;
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
    dart::dynamics::Joint *jn = mRenderEnv->getCharacter(0)->getSkeleton()->getRootJoint();
    dart::dynamics::BodyNode *bn = jn->getChildBodyNode();
    Eigen::Vector3d offset = bn->getTransform().translation();
    bvh << "ROOT\tCharacter_" << jn->getName() << std::endl;
    bvh << "{" << std::endl;
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
        mMotionBuffer.push_back(mRenderEnv->getCharacter(0)->getSkeleton()->getPositions());
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

void GLFWApp::_setXminToHeelStrike()
{
    if (!mGraphData->key_exists("contact_phaseR"))
    {
        std::cout << "[HeelStrike] contact_phaseR key not found in graph data" << std::endl;
        return;
    }

    if (!mRenderEnv) return;

    std::vector<double> contact_phase_buffer = mGraphData->get("contact_phaseR");

    // Ensure there are at least two points to compare for transitions
    if (contact_phase_buffer.size() < 2)
    {
        std::cout << "[HeelStrike] Not enough data points for heel strike detection" << std::endl;
        return;
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
        mXmin = heel_strike_time;
        std::cout << "[HeelStrike] Found heel strike at time: " << heel_strike_time << std::endl;
    }
    else
    {
        std::cout << "[HeelStrike] No heel strike found in current data" << std::endl;
    }
}

void GLFWApp::startLoop()
{
    while (!glfwWindowShouldClose(mWindow))
    {
        // Simulation Step
        if (!mRolloutStatus.pause || mRolloutStatus.cycle > 0) update();

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
    
    // Load networks
    auto character = mRenderEnv->getCharacter(0);
    mNetworks.clear();
    for (const auto& path : mNetworkPaths) {
        loadNetworkFromPath(path);
    }
    
    if (!mNetworks.empty()) {
        mRenderEnv->setMuscleNetwork(mNetworks.back().muscle);
    }

    // Initialize DOF tracking
    mRelatedDofs.clear();
    mRelatedDofs.resize(mRenderEnv->getCharacter(0)->getSkeleton()->getNumDofs() * 2, false);

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

    // Load motion files
    loadMotionFiles();

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

    // Todo: motion control panel here

    if (!mRenderEnv)
    {
        ImGui::Separator();
        ImGui::Text("Environment not loaded.");
        ImGui::End();
        return;
    }
    // mFGNList
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

    if (ImGui::Button("Load FGN"))
    {
        mDrawFGNSkeleton = true;
        py::tuple res = py::module::import("forward_gaitnet").attr("load_FGN")(mFGNList[selected_fgn], mRenderEnv->getNumParamState(), mRenderEnv->getCharacter(0)->posToSixDof(mRenderEnv->getCharacter(0)->getSkeleton()->getPositions()).rows());
        mFGN = res[0];
        mFGNmetadata = res[1].cast<std::string>();

        mNetworkPaths.clear();
        mNetworks.clear();
        std::cout << "METADATA " << std::endl
                  << mFGNmetadata << std::endl;
        initEnv(mFGNmetadata);
    }
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
    if (ImGui::Button("Load BGN"))
    {
        mGVAELoaded = true;
        py::object load_gaitvae = py::module::import("advanced_vae").attr("load_gaitvae");
        int rows = mRenderEnv->getCharacter(0)->posToSixDof(mRenderEnv->getCharacter(0)->getSkeleton()->getPositions()).rows();
        mGVAE = load_gaitvae(mBGNList[selected_fgn], rows, 60, mRenderEnv->getNumKnownParam(), mRenderEnv->getNumParamState());

        mPredictedMotion.motion = mMotions[mMotionIdx].motion;
        mPredictedMotion.param = mMotions[mMotionIdx].param;
        mPredictedMotion.name = "Unpredicted";
    }
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
        // ImGui Slider
        ImGui::SliderFloat("Femur Torsion L", &femur_torsion_l, -0.55, 0.55);
        ImGui::SliderFloat("Femur Torsion R", &femur_torsion_r, -0.55, 0.55);
        ImGui::SliderFloat("C3D Scale", &c3d_scale, 0.5, 2.0);
        ImGui::SliderFloat("Height Offset", &height_offset, -0.5, 0.5);
    
        if (ImGui::Button("Load C3D"))
        {
            if (selected_c3d < mC3DList.size() && !mC3DList.empty())
            {
                mRenderC3D = true;
                mC3DReader = new C3D_Reader("data/skeleton_gaitnet_narrow_model.xml", "data/marker_set.xml", mRenderEnv->GetEnvironment());
                std::cout << "Loading C3D: " << mC3DList[selected_c3d] << std::endl;
                mC3dMotion = mC3DReader->loadC3D(mC3DList[selected_c3d], femur_torsion_l, femur_torsion_r, c3d_scale, height_offset); // /* ,torsionL, torsionR*/);
                mC3DCOM = Eigen::Vector3d::Zero();
            }
            else
            {
                std::cout << "Error: No C3D files available or invalid selection (selected: " << selected_c3d << ", available: " << mC3DList.size() << ")" << std::endl;
            }
        }
    
        if (ImGui::Button("Convert C3D to Motion"))
        {
            auto m = mC3DReader->convertToMotion();
            m.name = "C3D Motion" + std::to_string(mMotions.size());
            mMotions.push_back(m);
            mAddedMotions.push_back(m);
        }
    }

    static int mMotionPhaseOffset = 0;

    if (ImGui::CollapsingHeader("Motions"))
    {

        ImGui::Checkbox("Draw Motion\t", &mDrawMotion);
        if (ImGui::ListBoxHeader("motion list", ImVec2(-FLT_MIN, 20 * ImGui::GetTextLineHeightWithSpacing())))
        {
            for (int i = 0; i < mMotions.size(); i++)
            {
                if (ImGui::Selectable(mMotions[i].name.c_str(), mMotionIdx == i))
                    mMotionIdx = i;

                if (mMotionIdx == i)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::ListBoxFooter();
        }
    }
    ImGui::SliderInt("Motion Phase Offset", &mMotionPhaseOffset, 0, 59);
    if (ImGui::Button("Convert Motion"))
    {
        int size = 101;
        Eigen::VectorXd m = mMotions[mMotionIdx].motion;
        mMotions[mMotionIdx].motion << m.tail((60 - mMotionPhaseOffset) * size), m.head(mMotionPhaseOffset * size);
    }

    // Button
    if (ImGui::Button("Add Current Simulation motion to motion "))
    {
        Motion current_motion;
        current_motion.name = "New Motion " + std::to_string(mMotions.size());
        current_motion.param = mRenderEnv->getParamState();
        current_motion.motion = Eigen::VectorXd::Zero(6060);

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

            current_trajectory.push_back(mRenderEnv->getCharacter(0)->posToSixDof(mRenderEnv->getCharacter(0)->getSkeleton()->getPositions()));

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

    if (ImGui::Button("Set to Param of reference"))
        mRenderEnv->setParamState(mMotions[mMotionIdx].param, false, true);

    if (mGVAELoaded)
    {
        if (ImGui::Button("predict new motion"))
        {
            Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
            input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
            py::tuple res = mGVAE.attr("render_forward")(input.cast<float>());
            Eigen::VectorXd motion = res[0].cast<Eigen::VectorXd>();
            Eigen::VectorXd param = res[1].cast<Eigen::VectorXd>();

            mPredictedMotion.motion = motion;
            mPredictedMotion.param = mRenderEnv->getParamStateFromNormalized(param);
        }

        if (ImGui::Button("Sampling 1000 params"))
        {
            Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
            input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
            mGVAE.attr("sampling")(input.cast<float>(), mMotions[mMotionIdx].param);
        }

        if (ImGui::Button("Set to predicted param"))
            mRenderEnv->setParamState(mPredictedMotion.param, false, true);

        if (ImGui::Button("Predict and set param"))
        {
            Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
            input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
            py::tuple res = mGVAE.attr("render_forward")(input.cast<float>());
            Eigen::VectorXd motion = res[0].cast<Eigen::VectorXd>();
            Eigen::VectorXd param = res[1].cast<Eigen::VectorXd>();

            mPredictedMotion.motion = motion;
            mPredictedMotion.param = mRenderEnv->getParamStateFromNormalized(param);
            mRenderEnv->setParamState(mPredictedMotion.param, false, true);
        }
    }
    if (ImGui::Button("Save added motion"))
    {
        py::list motions;
        py::list params;

        for (auto m : mAddedMotions)
        {
            motions.append(m.motion);
            params.append(m.param);
        }

        py::object save_motions = py::module::import("converter_to_gvae_set").attr("save_motions");
        save_motions(motions, params);
    }

    if (ImGui::Button("Save Selected Motion"))
    {
        py::list motions;
        py::list params;
        Motion motion = mMotions[mMotionIdx];

        motions.append(motion.motion);
        params.append(motion.param);

        py::object save_motions = py::module::import("converter_to_gvae_set").attr("save_motions");
        save_motions(motions, params);
    }

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
    mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
}

void GLFWApp::drawSimVisualizationPanel()
{
    
    ImGui::SetNextWindowPos(ImVec2(mWidth - mPlotPanelWidth - 10, 10), ImGuiCond_Once);
    if (!mRenderEnv)
    {
        ImGui::SetNextWindowSize(ImVec2(mPlotPanelWidth, 60), ImGuiCond_Always);
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
        ImGui::Text("Elapsed Time    : %.3f s", mRenderEnv->getWorld()->getTime());
        ImGui::Text("Phase           : %.3f", std::fmod(mRenderEnv->getCharacter(0)->getLocalTime(), (mRenderEnv->getBVH(0)->getMaxTime() / mRenderEnv->getCadence())) / (mRenderEnv->getBVH(0)->getMaxTime() / mRenderEnv->getCadence()));
        ImGui::Text("Target Vel      : %.3f m/s", mRenderEnv->getTargetCOMVelocity());
        ImGui::Text("Average Vel     : %.3f m/s", mRenderEnv->getAvgVelocity()[2]);
        ImGui::Text("Current Vel     : %.3f m/s", mRenderEnv->getCharacter(0)->getSkeleton()->getCOMLinearVelocity()[2]);

        ImGui::Separator();
        
        ImGui::Indent();
        if (ImGui::CollapsingHeader("Metadata"))
        {
            if (ImGui::Button("Print"))
                std::cout << mRenderEnv->getMetadata() << std::endl;
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
    ImGui::SetNextItemWidth(50);
    ImGui::InputDouble("X-axis Min", &mXmin);
    ImGui::SameLine();
    if (ImGui::Button("HS")) _setXminToHeelStrike();
    ImGui::SameLine();
    if (ImGui::Button("1.1")) mXmin = -1.1;

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
            std::vector<std::string> rewardKeys = {"r", "r_p", "r_v", "r_com", "r_ee", "r_metabolic", "r_loco", "r_avg", "r_step"};
            plotGraphData(rewardKeys, ImAxis_Y1, true, false, "");
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
        Eigen::VectorXd cf = mRenderEnv->getCharacter(0)->getSkeleton()->getConstraintForces();
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
        if (ImGui::ListBoxHeader("Joint", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing())))
        {
            int idx = 0;

            for (int i = 0; i < mRenderEnv->getCharacter(0)->getSkeleton()->getNumDofs(); i++)
            {
                if (ImGui::Selectable((std::to_string(i) + "_force").c_str(), joint_selected == i))
                    joint_selected = i;
                ImGui::SetItemDefaultFocus();
            }
            ImGui::ListBoxFooter();
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
            MuscleTuple tp = mRenderEnv->getCharacter(0)->getMuscleTuple(false);

            Eigen::VectorXd fullJtp = Eigen::VectorXd::Zero(mRenderEnv->getCharacter(0)->getSkeleton()->getNumDofs());
            if (mRenderEnv->getCharacter(0)->getIncludeJtPinSPD())
                fullJtp.tail(fullJtp.rows() - mRenderEnv->getCharacter(0)->getSkeleton()->getRootJoint()->getNumDofs()) = tp.JtP;
            Eigen::VectorXd dt = mRenderEnv->getCharacter(0)->getSPDForces(mRenderEnv->getCharacter(0)->getPDTarget(), fullJtp).tail(tp.JtP.rows());

            auto mtl = mRenderEnv->getCharacter(0)->getMuscleTorqueLogs();

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
            Eigen::VectorXd dt = mRenderEnv->getCharacter(0)->getTorque();
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

        auto m = mRenderEnv->getCharacter(0)->getMuscles()[selected];

        ImPlot::SetNextAxisLimits(3, 500, 0);
        ImPlot::SetNextAxisLimits(0, 0, 1.5, ImGuiCond_Always);
        if (ImPlot::BeginPlot((m->name + "_force_graph").c_str(), ImVec2(-1, 250)))
        {
            ImPlot::SetupAxes("length", "force");
            std::vector<std::vector<double>> p = m->GetGraphData();

            ImPlot::PlotLine("##active", p[1].data(), p[2].data(), 250);
            ImPlot::PlotLine("##active_with_activation", p[1].data(), p[3].data(), 250);
            ImPlot::PlotLine("##passive", p[1].data(), p[4].data(), 250);

            ImPlot::PlotVLines("current", p[0].data(), 1);
            ImPlot::EndPlot();
        }

        ImPlot::SetNextAxisLimits(3, 0, 1.5);
        ImPlot::SetNextAxisLimits(0, 0, 2.5, ImGuiCond_Always);
        if (ImPlot::BeginPlot((m->name + "_activation_graph").c_str(), ImVec2(-1, 250)))
        {
            std::vector<std::vector<double>> p; // = m->GetGraphData();
            std::vector<double> px;
            std::vector<double> py;
            p.clear();
            px.clear();
            py.clear();

            for (int i = 0; i < mRenderEnv->getCharacter(0)->getActivationLogs().size(); i++)
            {
                px.push_back(0.01 * i - mRenderEnv->getCharacter(0)->getActivationLogs().size() * 0.01 + 2.5);
                py.push_back(mRenderEnv->getCharacter(0)->getActivationLogs()[i][selected]);
            }

            p.push_back(px);
            p.push_back(py);

            ImPlot::PlotLine("##activation_graph", p[0].data(), p[1].data(), p[0].size());
            ImPlot::EndPlot();
        }

        ImGui::Separator();

        // Activation bars
        if (mRenderEnv->getUseMuscle())
        {
            Eigen::VectorXd acitvation = mRenderEnv->getCharacter(0)->getActivations();

            ImPlot::SetNextAxisLimits(0, -0.5, acitvation.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, 0, 1);
            double *x_act = new double[acitvation.rows()]();
            double *y_act = new double[acitvation.rows()]();

            for (int i = 0; i < acitvation.rows(); i++)
            {
                x_act[i] = i;
                y_act[i] = acitvation[i];
            }
            if (ImPlot::BeginPlot("activation"))
            {
                ImPlot::PlotBars("activation_level", x_act, y_act, acitvation.rows(), 1.0);
                ImPlot::EndPlot();
            }
        }

        ImGui::Separator();

        ImGui::Text("Muscle Name");
        if (ImGui::ListBoxHeader("Muscle", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing())))
        {
            int idx = 0;
            for (auto m : mRenderEnv->getCharacter(0)->getMuscles())
            {
                if (ImGui::Selectable((m->name + "_force").c_str(), selected == idx))
                    selected = idx;
                if (selected)
                    ImGui::SetItemDefaultFocus();
                idx++;
            }
            ImGui::ListBoxFooter();
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
        if (!mRenderEnv || !mRenderEnv->getCharacter(0)) {
            ImGui::TextDisabled("Load environment first");
        } else {
            auto skel = mRenderEnv->getCharacter(0)->getSkeleton();
            
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
                    
                    if (dof_idx < 6) {
                        // Root joint - expand limits
                        if (is_translation) {
                            // Translation: use raw values (meters)
                            lower_limit = -2.0f;
                            upper_limit = 2.0f;
                            display_value = pos_rad[dof_idx];
                        } else {
                            // Rotation: convert to degrees
                            lower_limit = -360.0f;
                            upper_limit = 360.0f;
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

void GLFWApp::loadCameraPreset(int index) {
    if (index < 0 || index >= 3 || !mCameraPresets[index].isSet)
    {
        std::cout << "[Camera] Preset " << index << " is not valid" << std::endl;
        return;
    }
    std::cout << "[Camera] Loading camera preset " << index << ": " << mCameraPresets[index].description << std::endl;

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

    // Rollout Control
    if (ImGui::CollapsingHeader("Rollout", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int rolloutInput = -1; // Will be initialized from config
        if (rolloutInput == -1) {
            rolloutInput = mDefaultRolloutCount; // Initialize from config on first use
        }

        ImGui::SetNextItemWidth(70);
        ImGui::InputInt("Cycles", &rolloutInput);
        if (rolloutInput < 1) rolloutInput = 1;

        ImGui::SameLine();

        // Run button
        if (ImGui::Button("Run"))
        {
            mRolloutStatus.cycle = rolloutInput;
            mRolloutStatus.pause = false;
        }
    }

    // Muscle Control
    if (ImGui::CollapsingHeader("Muscle"))
    {
        Eigen::VectorXf activation = mRenderEnv->getCharacter(0)->getActivations().cast<float>(); // * mRenderEnv->getActionScale();
        int idx = 0;
        for (auto m : mRenderEnv->getCharacter(0)->getMuscles())
        {
            ImGui::SliderFloat((m->GetName().c_str()), &activation[idx], 0.0, 1.0);
            idx++;
        }
        mRenderEnv->getCharacter(0)->setActivations((activation.cast<double>()));
    }

    // Joint Control - use new detailed control method
    drawJointControlSection();

    // Gait Parameters
    if (ImGui::CollapsingHeader("Gait Parameters"))
    {
        Eigen::VectorXf ParamState = mRenderEnv->getParamState().cast<float>();
        Eigen::VectorXf ParamMin = mRenderEnv->getParamMin().cast<float>();
        Eigen::VectorXf ParamMax = mRenderEnv->getParamMax().cast<float>();

        int idx = 0;
        for (auto c : mRenderEnv->getParamName())
        {
            ImGui::SliderFloat(c.c_str(), &ParamState[idx], ParamMin[idx], ParamMax[idx] + 1E-10);
            idx++;
        }
        mRenderEnv->setParamState(ParamState.cast<double>(), false, true);
        mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
    }

    // Body Parameters
    if (ImGui::CollapsingHeader("Body Parameters"))
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
        mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
    }

    // Rendering
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Draw Reference Motion", &mDrawReferenceSkeleton);
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
            auto allMuscles = mRenderEnv->getCharacter(0)->getMuscles();

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

        if (mRenderEnv->getUseMuscle()) mRenderEnv->getCharacter(0)->getMuscleTuple(false);

        
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
    
    // // Build selected muscles list based on selection states
    // auto allMuscles = mRenderEnv->getCharacter(0)->getMuscles();
    // for (int i = 0; i < mMuscleSelectionStates.size() && i < allMuscles.size(); i++)
    // {
    //     if (mMuscleSelectionStates[i])
    //     {
    //         mSelectedMuscles.push_back(allMuscles[i]);
    //     }
    // }
    // Related Dof Muscle Rendering
    // mSelectedMuscles.clear();
    // Muscle Selection
    // if (ImGui::CollapsingHeader("Muscle Selection"))
    // {
    //     for (int i = 0; i < mRelatedDofs.size(); i += 2)
    //     {
    //         bool dof_plus, dof_minus;
    //         dof_plus = mRelatedDofs[i];
    //         dof_minus = mRelatedDofs[i + 1];
    //         ImGui::Checkbox((std::to_string(i / 2) + " +").c_str(), &dof_plus);
    //         ImGui::SameLine();
    //         ImGui::Checkbox((std::to_string(i / 2) + " -").c_str(), &dof_minus);
    //         mRelatedDofs[i] = dof_plus;
    //         mRelatedDofs[i + 1] = dof_minus;
    //     }

    //     // Check related dof
    //     for (auto m : mRenderEnv->getCharacter(0)->getMuscles())
    //     {
    //         Eigen::VectorXd related_vec = m->GetRelatedVec();
    //         for (int i = 0; i < related_vec.rows(); i++)
    //         {
    //             if (related_vec[i] > 0 && mRelatedDofs[i * 2])
    //             {
    //                 mSelectedMuscles.push_back(m);
    //                 break;
    //             }
    //             else if (related_vec[i] < 0 && mRelatedDofs[i * 2 + 1])
    //             {
    //                 mSelectedMuscles.push_back(m);
    //                 break;
    //             }
    //         }
    //     }
    // }

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

void GLFWApp::drawUIFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    drawSimControlPanel();
    drawSimVisualizationPanel();
    drawKinematicsControlPanel();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}

void GLFWApp::drawSimFrame()
{
    initGL();
    if (mRenderEnv) setCamera();

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

    if (!mRenderConditions) drawGround(1E-3);

    // Simulated Character
    if (mRenderEnv){
        drawPhase(mRenderEnv->getLocalPhase(true), mRenderEnv->getNormalizedPhase());    
        if (mDrawCharacter)
        {
            drawSkeleton(mRenderEnv->getCharacter(0)->getSkeleton()->getPositions(), Eigen::Vector4d(0.65, 0.65, 0.65, 1.0));
            if (!mRenderConditions) drawShadow();
            if (mMuscleSelectionStates.size() > 0) drawMuscles(mMuscleRenderType);
        }
        if ((mRenderEnv->getRewardType() == gaitnet) && mDrawFootStep) drawFootStep();
        if (mDrawJointSphere)
        {
            for (auto jn : mRenderEnv->getCharacter(0)->getSkeleton()->getJoints())
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
            Eigen::VectorXd pos = (mDrawPDTarget ? mRenderEnv->getCharacter(0)->getPDTarget() : mRenderEnv->getTargetPositions());
            drawSkeleton(pos, Eigen::Vector4d(1.0, 0.35, 0.35, 1.0));
        }
        if (mDrawEOE)
        {
            glColor4f(1.0, 0.0, 0.0, 1.0);
            GUI::DrawSphere(mRenderEnv->getCharacter(0)->getSkeleton()->getCOM(), 0.01);
            glColor4f(0.5, 0.5, 0.8, 0.2);
            glBegin(GL_QUADS);
            glVertex3f(-10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter(0)->getGlobalRatio(), -10);
            glVertex3f(10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter(0)->getGlobalRatio(), -10);
            glVertex3f(10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter(0)->getGlobalRatio(), 10);
            glVertex3f(-10, mRenderEnv->getLimitY() * mRenderEnv->getCharacter(0)->getGlobalRatio(), 10);
            glEnd();
        }
        if (mMotions.size() > 0)
        {
            // GVAE
            // For Debugging
            if (mDrawMotion)
            {
                Eigen::VectorXd motion_pos; // Eigen::VectorXd::Zero(101);
                // mMotionFrsameIdx %= 60;
    
                double phase = mRenderEnv->getGlobalTime() / (mRenderEnv->getBVH(0)->getMaxTime() / (mRenderEnv->getCadence() / sqrt(mRenderEnv->getCharacter(0)->getGlobalRatio())));
                phase = fmod(phase, 2.0);
    
                int idx_0 = (int)(phase * 30);
                int idx_1 = (idx_0 + 1);
    
                // Interpolation between idx_0 and idx_1
                motion_pos = mRenderEnv->getCharacter(0)->sixDofToPos(mMotions[mMotionIdx].motion.segment((idx_0 % 60) * 101, 101) * (1.0 - (phase * 30 - (idx_0 % 60))) + mMotions[mMotionIdx].motion.segment((idx_1 % 60) * 101, 101) * (phase * 30 - (idx_0 % 60)));
    
                // Root Offset
                if (!mRolloutStatus.pause || mRolloutStatus.cycle > 0)
                {
                    mMotionRootOffset[0] += motion_pos[3] * 0.5;
                    mMotionRootOffset[1] = motion_pos[4];
                    mMotionRootOffset[2] += motion_pos[5] * 0.5;
                }
                motion_pos.segment(3, 3) = mMotionRootOffset;
    
                drawSkeleton(motion_pos, Eigen::Vector4d(0.8, 0.8, 0.2, 0.7));
            }
    
            // Draw Output Motion
            // Eigen::VectorXd input = Eigen::VectorXd::Zero(mMotions[mMotionIdx].motion.rows() + mRenderEnv->getNumKnownParam());
            // input << mMotions[mMotionIdx].motion, mRenderEnv->getNormalizedParamStateFromParam(mMotions[mMotionIdx].param.head(mRenderEnv->getNumKnownParam()));
            // Eigen::VectorXd output = mGVAE.attr("render_forward")(input.cast<float>()).cast<Eigen::VectorXd>();
            // std::cout << "[DEBUG] Out put " << output.rows() << std::endl;
            // drawMotions(output, mMotions[mMotionIdx].param, Eigen::Vector4d(0.8, 0.8, 0.2, 0.7));
            // drawMotions(mPredictedMotion.motion, mPredictedMotion.param, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector4d(0.8, 0.8, 0.2, 0.7));
        }
        // FGN
        if (mDrawFGNSkeleton)
        {
            Eigen::VectorXd FGN_in = Eigen::VectorXd::Zero(mRenderEnv->getNumParamState() + 2);
            Eigen::VectorXd phase = Eigen::VectorXd::Zero(2);
    
            phase[0] = sin(2 * M_PI * mRenderEnv->getNormalizedPhase());
            phase[1] = cos(2 * M_PI * mRenderEnv->getNormalizedPhase());
    
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
    
            Eigen::VectorXd pos = mRenderEnv->getCharacter(0)->sixDofToPos(res);
            drawSkeleton(pos, Eigen::Vector4d(0.35, 0.35, 1.0, 1.0));
        }    
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
        // drawSkeleton(mRenderEnv->getCharacter(0)->sixDofToPos(mC3DReader->mConvertedPos[mC3DCount % mC3DReader->mConvertedPos.size()]), Eigen::Vector4d(1.0, 0.0, 0.0, 0.5));
    }

    if (mMouseDown) drawAxis();

}

void GLFWApp::drawGround(double height)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    double width = 0.005;
    int count = 0;
    glBegin(GL_QUADS);
    for (double x = -100.0; x < 100.01; x += 1.0)
    {
        for (double z = -100.0; z < 100.01; z += 1.0)
        {
            if (count % 2 == 0)
                glColor3f(216.0 / 255.0, 211.0 / 255.0, 204.0 / 255.0);
            else
                glColor3f(216.0 / 255.0 - 0.1, 211.0 / 255.0 - 0.1, 204.0 / 255.0 - 0.1);
            count++;
            glVertex3f(x, height, z);
            glVertex3f(x + 1.0, height, z);
            glVertex3f(x + 1.0, height, z + 1.0);
            glVertex3f(x, height, z + 1.0);
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
    mC3DCount = 0;
    mGraphData->clear_all();
    mMotionRootOffset = Eigen::Vector3d::Zero();
    mMotionRootOffset[0] = 1.0;
    mC3DCOM = Eigen::Vector3d::Zero();

    if (mRenderEnv) {
        mRenderEnv->reset();
        mFGNRootOffset = mRenderEnv->getCharacter(0)->getSkeleton()->getRootJoint()->getPositions().tail(3);
        mUseWeights = mRenderEnv->getUseWeights();
    }
}

void GLFWApp::keyboardPress(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        // case GLFW_KEY_U:
        //     mRenderEnv->updateParamState();
        //     mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
        //     reset();
        //     break;
        // case GLFW_KEY_COMMA:
        //     mRenderEnv->setParamState(mRenderEnv->getParamDefault(), false, true);
        //     mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
        //     reset();
        //     break;
        // case GLFW_KEY_N:
        //     mRenderEnv->setParamState(mRenderEnv->getParamMin(), false, true);
        //     mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
        //     reset();
        //     break;
        // case GLFW_KEY_M:
        //     mRenderEnv->setParamState(mRenderEnv->getParamMax(), false, true);
        //     mRenderEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
        //     reset();
        //     break;

        // case GLFW_KEY_Z:
        // {
        //     Eigen::VectorXd pos = mRenderEnv->getCharacter(0)->getSkeleton()->getPositions().setZero();
        //     Eigen::VectorXd vel = mRenderEnv->getCharacter(0)->getSkeleton()->getVelocities().setZero();
        //     pos[41] = 1.5;
        //     pos[51] = -1.5;
        //     mRenderEnv->getCharacter(0)->getSkeleton()->setPositions(pos);
        //     mRenderEnv->getCharacter(0)->getSkeleton()->setVelocities(vel);
        // }
        // break;
        // // Rendering Key
        // case GLFW_KEY_T:
        //     mDrawReferenceSkeleton = !mDrawReferenceSkeleton;
        //     break;
        // case GLFW_KEY_P:
        //     mDrawCharacter = !mDrawCharacter;
        //     break;
        case GLFW_KEY_S:
            update();
            break;
        case GLFW_KEY_R:
            reset();
            break;
        // case GLFW_KEY_O:
            // mDrawOBJ = !mDrawOBJ;
            // break;
        case GLFW_KEY_SPACE:
            mRolloutStatus.pause = !mRolloutStatus.pause;
            mRolloutStatus.cycle = -1;
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
        // case GLFW_KEY_B:
            // reset();
// 
            // {
                // mMotionBuffer.clear();
                // while (mRenderEnv->isEOE() == 0)
                    // update(true);
            // }
            // exportBVH(mMotionBuffer, mRenderEnv->getCharacter(0)->getSkeleton());
            // break;

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
    if (mFocus == 1)
    {
        mTrans = -mRenderEnv->getCharacter(0)->getSkeleton()->getCOM();
        mTrans[1] = -1;
        mTrans *= 1000;
    }
    else if (mFocus == 2)
    {
        mTrans = -mRenderEnv->getTargetPositions().segment(3, 3); //-mRenderEnv->getCharacter(0)->getSkeleton()->getCOM();
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

    auto muscles = mRenderEnv->getCharacter(0)->getMuscles();
    for (int i = 0; i < muscles.size(); i++)
    {
        // Skip if muscle is not selected (using same order as environment)
        if (i < mMuscleSelectionStates.size() && !mMuscleSelectionStates[i]) continue;

        auto muscle = muscles[i];
        muscle->Update();
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
    Eigen::VectorXd pos = mRenderEnv->getCharacter(0)->getSkeleton()->getPositions();

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
        auto character = mRenderEnv->getCharacter(0);
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
    mMotionSkeleton = mRenderEnv->getCharacter(0)->getSkeleton()->cloneSkeleton();
    
    // Setup BVH joint calibration
    mJointCalibration.clear();
    for (auto jn : mRenderEnv->getCharacter(0)->getSkeleton()->getJoints()) {
        if (jn == mRenderEnv->getCharacter(0)->getSkeleton()->getRootJoint()) {
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

void GLFWApp::loadMotionFiles()
{
    py::gil_scoped_acquire gil;
    
    mMotions.clear();
    mMotionIdx = 0;
    
    std::string motion_path = "motions";
    if (!fs::exists(motion_path) || !fs::is_directory(motion_path)) {
        std::cerr << "Motion directory not found: " << motion_path << std::endl;
        return;
    }

    try {
        py::object load_motions_from_file = py::module::import("forward_gaitnet").attr("load_motions_from_file");
        
        for (const auto &entry : fs::directory_iterator(motion_path)) {
            std::string file_name = entry.path().string();
            if (file_name.find(".npz") == std::string::npos)
                continue;

            try {
                py::tuple results = load_motions_from_file(file_name, mRenderEnv->getNumKnownParam());
                
                // Handle potential type conversion issues
                py::object params_obj = results[0];
                py::object motions_obj = results[1];
                
                // Unwrap nested tuples if needed (debug build issue)
                if (py::isinstance<py::tuple>(params_obj)) {
                    py::tuple params_tuple = params_obj.cast<py::tuple>();
                    if (params_tuple.size() > 0) {
                        params_obj = params_tuple[0];
                    }
                }
                if (py::isinstance<py::tuple>(motions_obj)) {
                    py::tuple motions_tuple = motions_obj.cast<py::tuple>();
                    if (motions_tuple.size() > 0) {
                        motions_obj = motions_tuple[0];
                    }
                }
                
                Eigen::MatrixXd params = params_obj.cast<Eigen::MatrixXd>();
                Eigen::MatrixXd motions = motions_obj.cast<Eigen::MatrixXd>();

                for (int i = 0; i < params.rows(); i++) {
                    Motion motion_elem;
                    motion_elem.name = file_name + "_" + std::to_string(i);
                    motion_elem.param = params.row(i);
                    motion_elem.motion = motions.row(i);
                    mMotions.push_back(motion_elem);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error loading motion file " << file_name << ": " << e.what() << std::endl;
                continue;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error importing forward_gaitnet module: " << e.what() << std::endl;
    }
}
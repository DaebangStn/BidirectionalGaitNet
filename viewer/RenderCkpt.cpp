#include "RenderCkpt.h"
#include "motion/PlaybackUtils.h"
#include "common/imgui_common.h"
#include "rm/rm.hpp"
#include <rm/global.hpp>
#include "stb_image.h"
#include "common/stb_image_write.h"
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <ctime>
#include "DARTHelper.h"
#include <tinyxml2.h>
#include <sstream>
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "Motion.h"
#include "HDF.h"
#include "C3D.h"
#include "Log.h"
#include <boost/program_options.hpp>

namespace fs = std::filesystem;

// Note: computeMarkerCentroid moved to C3D::computeCentroid() in sim/C3D.cpp

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

RenderCkpt::RenderCkpt(int argc, char **argv)
    : ViewerAppBase("MuscleSim", 2560, 1440)  // Base class handles GLFW/ImGui init
{
    mRenderEnv = nullptr;
    mMotionCharacter = nullptr;
    mGVAELoaded = false;
    mRenderConditions = false;

    // App-specific defaults (base class already loaded render.yaml for panel sizes, etc.)
    mDefaultRolloutCount = 10;
    mXmin = 0.0;
    mXminResizablePlotPane = 0.0;
    mYminResizablePlotPane = 0.0;
    mYmaxResizablePlotPane = 1.0;
    mPlotTitle = false;
    mPlotHideLegend = false;
    mProgressForward = true;

    // Initialize viewer time management with default cycle duration
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mViewerPlaybackSpeed = 2.5;
    mLastPlaybackSpeed = mViewerPlaybackSpeed;
    mViewerCycleDuration = 2.0 / 1.1;  // Default cycle duration (~1.818s)
    mLastRealTime = 0.0;
    mSimulationStepDuration = 0.0;
    mSimStepDurationAvg = -1.0;
    mRealDeltaTimeAvg = 0.0;
    mIsPlaybackTooFast = false;
    mShowResizablePlotPane = false;
    mShowTitlePanel = false;
    mResetPhase = -1.0;  // Default to randomized reset
    mResizablePlots.resize(1);
    strcpy(mResizePlotKeys, "");
    mResizePlotPane = true;
    mSetResizablePlotPane = false;
    mPlotTitleResizablePlotPane = true;
    mMuscleTransparency = 1.0;
    mMuscleResolution = 1.0;
    mGroundMode = GroundMode::Solid;

    // Initialize single motion architecture
    mMotion = nullptr;
    mMotionProcessor = std::make_unique<MotionProcessor>();

    // Note: Base class already calls loadRenderConfig() and resetCamera()
    // RenderCkpt-specific config loaded via loadRenderConfigImpl() override
    updateResizablePlotsFromKeys();

    // RenderCkpt-specific input state
    mZooming = false;

    // Set default camera focus mode (follow character)
    mCamera.focus = CameraFocusMode::FOLLOW_CHARACTER;

    // Rendering Options (mDrawFlags initialized with struct defaults)
    mStochasticPolicy = false;

    mMuscleRenderType = activationLevel;
    mMuscleRenderTypeInt = 2;
    mMuscleResolution = 0.0;

    // Noise Injector UI initialization
    mNoiseMode = 0;  // Default to no noise
    mPlotActivationNoise = false;  // Default: don't plot noise

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
    mGraphData->register_key("r_head_linear_acc", 500);
    mGraphData->register_key("r_head_rot_diff", 500);
    mGraphData->register_key("r_loco", 500);
    mGraphData->register_key("r_avg", 500);
    mGraphData->register_key("r_step", 500);
    mGraphData->register_key("r_drag_x", 500);
    mGraphData->register_key("r_phase", 500);

    // Register contact keys
    mGraphData->register_key("contact_left", 500);
    mGraphData->register_key("contact_right", 500);
    mGraphData->register_key("contact_phaseR", 1000);
    mGraphData->register_key("grf_left", 500);
    mGraphData->register_key("grf_right", 500);

    // Register kinematic keys
    // mGraphData->register_key("sway_Torso_X", 500);
    mGraphData->register_key("local_phase", 1000);
    mGraphData->register_key("sway_Foot_Rx", 1000);
    mGraphData->register_key("sway_Foot_Lx", 1000);
    mGraphData->register_key("sway_Toe_Ry", 1000);
    mGraphData->register_key("sway_Toe_Ly", 1000);
    mGraphData->register_key("sway_FPAr", 1000);
    mGraphData->register_key("sway_FPAl", 1000);
    mGraphData->register_key("sway_AnteversionR", 1000);
    // mGraphData->register_key("sway_AnteversionL", 1000);
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

    // Register COM keys
    mGraphData->register_key("com_x", 1000);
    mGraphData->register_key("com_z", 1000);

    // Register COM velocity keys
    mGraphData->register_key("com_vel_x", 400);
    mGraphData->register_key("com_vel_z", 5000);

    // Register COM regression error key
    mGraphData->register_key("com_deviation", 1000);

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

    // Gait phase metrics (unified keys for both feet)
    mGraphData->register_key("stride_length", 1000);
    mGraphData->register_key("phase_total", 1000);

    // Marker error metrics (C3D IK fitting error)
    mGraphData->register_key("marker_error_mean", 500);
    mGraphData->register_key("marker_error_max", 500);

    // Forward GaitNEt
    selected_fgn = 0;

    // Backward GaitNEt
    selected_bgn = 0;

    // Motion list
    mSelectedMotion = -1;

    // Note: mCamera.focus already set above (1 = follow character)
    // C3D marker rendering moved to c3d_processor

    // Initialize trackball using base class camera
    mCamera.trackball.setTrackball(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5), mWidth * 0.5);
    mCamera.trackball.setQuaternion(Eigen::Quaterniond::Identity());

    // Initialize camera presets
    initializeCameraPresets();
    loadCameraPreset(0);

    // Parse command-line arguments using Boost.Program_options
    namespace po = boost::program_options;

    po::options_description desc("Viewer Options");
    desc.add_options()
        ("checkpoint", po::value<std::string>(), "Checkpoint or metadata path");

    po::variables_map vm;
    try {
        // Parse known args, allow positional for checkpoint
        po::positional_options_description p;
        p.add("checkpoint", 1);

        po::store(po::command_line_parser(argc, argv)
                     .options(desc)
                     .positional(p)
                     .allow_unregistered()
                     .run(), vm);
        po::notify(vm);

        // Store checkpoint/metadata path
        if (vm.count("checkpoint")) {
            std::string path = vm["checkpoint"].as<std::string>();
            mNetworkPaths.push_back(path);
        } else if (argc > 1) {
            // Fallback: treat first arg as checkpoint if not parsed
            std::string path = std::string(argv[1]);
            mNetworkPaths.push_back(path);
        }
    } catch (const po::error& e) {
        LOG_ERROR("Argument parsing error: " << e.what());
        std::cerr << desc << std::endl;
        exit(EXIT_FAILURE);
    }

    // Note: GLFW/ImGui initialization handled by ViewerAppBase constructor
    // Base class registers callbacks that dispatch to virtual methods (keyPress, mouseMove, etc.)

    mns = py::module::import("__main__").attr("__dict__");
    py::module::import("sys").attr("path").attr("insert")(1, "python");

    mSelectedMuscles.clear();
    mRelatedDofs.clear();

    // Initialize muscle plot UI
    memset(mPlotMuscleFilterText, 0, sizeof(mPlotMuscleFilterText));

    py::gil_scoped_acquire gil;
    
    // Import checkpoint loader
    try {
        loading_network = py::module::import("ppo.model").attr("loading_network");
    } catch (const py::error_already_set& e) {
        LOG_WARN("[Checkpoint] Failed to import checkpoint loader: " << e.what());
        loading_network = py::none();
    }

    // Determine metadata path (loading_metadata returns a filepath)
    if (!mNetworkPaths.empty()) {
        std::string path = mNetworkPaths.back();
        try {
            py::object py_metadata = py::module::import("ppo.model").attr("loading_metadata")(path);
            if (!py_metadata.is_none()) {
                mCachedMetadata = py_metadata.cast<std::string>();
            }
            // Note: Keep path in mNetworkPaths - it's used later for checkpoint name and network loading
        } catch (const py::error_already_set& e) {
            LOG_ERROR("[Checkpoint] Error: Failed to load checkpoint from path: " << path);
            LOG_ERROR("[Checkpoint] Reason: " << e.what());
            LOG_ERROR("[Checkpoint] Please check that the checkpoint path exists and is in a valid format.");
            std::exit(1);
        }
    }

    // Initialize motion character for standalone motion playback (independent of simulation)
    initializeMotionCharacter(mCachedMetadata);

    // NOTE: C3D processing moved to c3d_processor executable

    // Scan motion files (HDF only - C3D moved to c3d_processor)
    scanMotionFiles();

    // Initialize Resource Manager for PID-based access (use singleton)
    try {
        mResourceManager = &rm::getManager();

        // Initialize PID Navigator with HDF filter
        mPIDNavigator = std::make_unique<PIDNav::PIDNavigator>(
            mResourceManager,
            std::make_unique<PIDNav::HDFFileFilter>()
        );

        // Register file selection callback
        mPIDNavigator->setFileSelectionCallback(
            [this](const std::string& path, const std::string& filename) {
                onPIDFileSelected(path, filename);
            }
        );

        // Initial PID scan
        mPIDNavigator->scanPIDs();
    } catch (const rm::RMError& e) {
        LOG_WARN("[RenderCkpt] Resource manager init failed: " << e.what());
    } catch (...) {
        LOG_WARN("[RenderCkpt] Resource manager init failed");
    }

    // Load normative kinematics (always available, independent of simulation)
    loadNormativeKinematics();

    // Load simulation environment on startup if enabled (default: true)
    if (mLoadSimulationOnStartup) {
        initEnv(mCachedMetadata);
    } else {
        LOG_INFO("[Config] Simulation environment loading disabled by config");
    }
}

Eigen::Vector3d RenderCkpt::computeMotionCycleDistance(Motion* motion)
{
    Eigen::Vector3d cycleDistance = Eigen::Vector3d::Zero();

    if (!motion || motion->getNumFrames() == 0)
        return cycleDistance;

    // Get raw motion data
    Eigen::VectorXd raw_motion = motion->getRawMotionData();
    int value_per_frame = motion->getValuesPerFrame();
    int frame_per_cycle = motion->getTimestepsPerCycle();

    if (frame_per_cycle <= 1)
        return cycleDistance;

    // Get first and last frame root positions
    Eigen::VectorXd first_frame = raw_motion.segment(0, value_per_frame);
    Eigen::VectorXd last_frame = raw_motion.segment((frame_per_cycle - 1) * value_per_frame, value_per_frame);

    Eigen::Vector3d first_root_pos(first_frame[3], first_frame[4], first_frame[5]);
    Eigen::Vector3d last_root_pos(last_frame[3], last_frame[4], last_frame[5]);

    // Compute cycle distance with the same formula as runtime calculation
    cycleDistance[0] = (last_root_pos[0] - first_root_pos[0]) * frame_per_cycle / (frame_per_cycle - 1);
    cycleDistance[2] = (last_root_pos[2] - first_root_pos[2]) * frame_per_cycle / (frame_per_cycle - 1);

    return cycleDistance;
}

// computeMarkerCycleDistance moved to c3d_processor

// ViewerAppBase override: Load RenderCkpt-specific config from render.yaml
// (Base class already loads geometry, panel widths, default_open_panels)
void RenderCkpt::loadRenderConfigImpl()
{
    try {
        std::string resolved_path = rm::resolve("render.yaml");
        YAML::Node config = YAML::LoadFile(resolved_path);

        // Load glfwapp-specific settings
        if (config["render_ckpt"]) {
            if (config["render_ckpt"]["rollout"] && config["render_ckpt"]["rollout"]["count"])
                mDefaultRolloutCount = config["render_ckpt"]["rollout"]["count"].as<int>();

            if (config["render_ckpt"]["plot"]) {
                if (config["render_ckpt"]["plot"]["title"])
                    mPlotTitle = config["render_ckpt"]["plot"]["title"].as<bool>();

                if (config["render_ckpt"]["plot"]["x_min"])
                    mXmin = config["render_ckpt"]["plot"]["x_min"].as<double>();
            }

            if (config["render_ckpt"]["playback_speed"]) {
                mViewerPlaybackSpeed = config["render_ckpt"]["playback_speed"].as<float>();
                mLastPlaybackSpeed = mViewerPlaybackSpeed;
            }

            if (config["render_ckpt"]["resetPhase"]) {
                mResetPhase = config["render_ckpt"]["resetPhase"].as<double>();
                LOG_VERBOSE("[Config] Reset phase set to: " << mResetPhase
                          << (mResetPhase < 0.0 ? " (randomized)" : ""));
            }

            if (config["render_ckpt"]["load_simulation"]) {
                mLoadSimulationOnStartup = config["render_ckpt"]["load_simulation"].as<bool>();
            }

            if (config["render_ckpt"]["default_pid_motion"]) {
                mDefaultPIDMotion = config["render_ckpt"]["default_pid_motion"].as<std::string>();
            }

            if (config["render_ckpt"]["double_plot_size"]) {
                mDefaultDoublePlotSize = config["render_ckpt"]["double_plot_size"].as<bool>();
            }

            if (config["render_ckpt"]["resizable_plot"]) {
                if (config["render_ckpt"]["resizable_plot"]["x_min"])
                    mXminResizablePlotPane = config["render_ckpt"]["resizable_plot"]["x_min"].as<double>();
                if (config["render_ckpt"]["resizable_plot"]["y_min"])
                    mYminResizablePlotPane = config["render_ckpt"]["resizable_plot"]["y_min"].as<double>();
                if (config["render_ckpt"]["resizable_plot"]["y_max"])
                    mYmaxResizablePlotPane = config["render_ckpt"]["resizable_plot"]["y_max"].as<double>();
                if (config["render_ckpt"]["resizable_plot"]["keys"]) {
                    std::string keys = config["render_ckpt"]["resizable_plot"]["keys"].as<std::string>();
                    strncpy(mResizePlotKeys, keys.c_str(), sizeof(mResizePlotKeys) - 1);
                    mResizePlotKeys[sizeof(mResizePlotKeys) - 1] = '\0';
                }
                if (config["render_ckpt"]["resizable_plot"]["title"])
                    mPlotTitleResizablePlotPane = config["render_ckpt"]["resizable_plot"]["title"].as<bool>();
            }
        }

        LOG_VERBOSE("[RenderCkpt Config] Rollout: " << mDefaultRolloutCount
                     << ", Playback Speed: " << mViewerPlaybackSpeed);

    } catch (const std::exception& e) {
        // Base class already logs config loading issues
    }
}

void RenderCkpt::onInitialize()
{
    // App-specific initialization after GLFW/ImGui is ready
    // Called after loadRenderConfigImpl(), so mDefaultPIDMotion is available

    // Auto-load default PID motion if configured and environment has a PID set
    if (mRenderEnv && mResourceManager && !mDefaultPIDMotion.empty()) {
        const std::string& globalPid = mRenderEnv->getGlobalPid();
        if (!globalPid.empty()) {
            try {
                std::string motionUri = "@pid:" + globalPid + "/motion/" + mDefaultPIDMotion;
                auto handle = mResourceManager->fetch(motionUri);
                std::string motionPath = handle.local_path().string();
                LOG_INFO("[RenderCkpt] Auto-loading default PID motion: " << motionPath);
                onPIDFileSelected(motionPath, mDefaultPIDMotion);
            } catch (const rm::RMError&) {
                // Motion not found, that's OK - user can select manually
                LOG_VERBOSE("[RenderCkpt] Default PID motion not found: " << mDefaultPIDMotion);
            }
        }
    }
}

RenderCkpt::~RenderCkpt()
{
    // Clean up RenderCkpt-specific resources
    // (ImGui/GLFW cleanup handled by ViewerAppBase::~ViewerAppBase)
    delete mMotion;
    mMotion = nullptr;

    delete mRenderEnv;
    delete mMotionCharacter;
}

void RenderCkpt::setWindowIcon(const char* icon_path)
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

// REMOVED: writeBVH() and exportBVH() - BVH format no longer supported

void RenderCkpt::update(bool _isSave)
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

    // Check for gait cycle completion AFTER step (PD-level check)
    if (mRenderEnv->isGaitCycleComplete())
    {
        mRolloutStatus.step();
        mRenderEnv->clearGaitCycleComplete();  // Clear flag after consuming
        if (mRolloutStatus.cycle == 0)
        {
            mRolloutStatus.pause = true; // Pause simulation when rollout completes
            return;
        }
    }

}

// Generate 3-character alphanumeric hash from string (for table UID)
static std::string hashTo3Char(const std::string& str)
{
    static const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    size_t hash = std::hash<std::string>{}(str);
    std::string result(3, '0');
    for (int i = 0; i < 3; ++i) {
        result[i] = charset[hash % 36];
        hash /= 36;
    }
    return result;
}

// Shared static map to track double-size state for each plot title
static std::map<std::string, bool>& getDoublePlotSizeMap()
{
    static std::map<std::string, bool> doublePlotSizeMap;
    return doublePlotSizeMap;
}

// Get plot height based on double height setting
static float getPlotHeight(const std::string& title, float baseHeight = 300.0f)
{
    auto& doublePlotSizeMap = getDoublePlotSizeMap();

    if (doublePlotSizeMap.find(title) == doublePlotSizeMap.end()) {
        return baseHeight;
    }

    return doublePlotSizeMap[title] ? baseHeight * 2.0f : baseHeight;
}

// Member function that uses config system for default open state
bool RenderCkpt::collapsingHeaderWithControls(const std::string& title)
{
    // Check config for default open state
    bool defaultOpen = isPanelDefaultOpen(title);

    // Access shared static map to track double-size state
    auto& doublePlotSizeMap = getDoublePlotSizeMap();

    // Initialize the map entry if it doesn't exist (use config default)
    if (doublePlotSizeMap.find(title) == doublePlotSizeMap.end()) {
        doublePlotSizeMap[title] = mDefaultDoublePlotSize;
    }

    // Calculate position for the 2x checkbox
    float windowWidth = ImGui::GetWindowWidth();
    float checkboxWidth = 45.0f;
    float checkboxPosX = windowWidth - checkboxWidth - 10; // 10px padding

    // Save cursor position
    ImVec2 cursorPos = ImGui::GetCursorPos();

    // Position the 2x checkbox
    ImGui::SetCursorPos(ImVec2(checkboxPosX, cursorPos.y));
    ImGui::Checkbox(("2x##" + title).c_str(), &doublePlotSizeMap[title]);

    // Restore cursor position for the header
    ImGui::SetCursorPos(cursorPos);

    // Draw the collapsing header
    bool headerOpen = false;
    if (defaultOpen) {
        headerOpen = ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
    } else {
        headerOpen = ImGui::CollapsingHeader(title.c_str());
    }

    return headerOpen;
}

void RenderCkpt::plotGraphData(const std::vector<std::string>& keys, ImAxis y_axis,
                            std::string postfix, bool show_stat, int color_ofs)
{
    if (keys.empty() || !mGraphData) return;

    ImPlot::SetAxis(y_axis);

    // Compute statistics for current plot range if show_stat is enabled
    std::map<std::string, std::map<std::string, double>> stats;
    if (show_stat)
    {
        ImPlotRect limits = ImPlot::GetPlotLimits();
        stats = statGraphData(keys, limits.X.Min, limits.X.Max);
    }

    // Get colormap size for stable color assignment
    int colormapSize = ImPlot::GetColormapSize();
    int keyIndex = 0;
    const double timeStep = (mRenderEnv ? mRenderEnv->getWorld()->getTimeStep() : 1.0);

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
            x[i] = static_cast<float>(-(bufferSize - 1 - i) * timeStep);  // Most recent at 0, oldest at -N
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

        // Build plot label with stable ID to prevent color flickering
        // Format: "display_label##stable_id"
        // ImPlot uses the part after ## as the stable identifier for color assignment
        std::string plot_label = selected_key;

        // Append statistics to legend if enabled
        if (show_stat && stats.count(key) > 0)
        {
            char stat_str[128];
            snprintf(stat_str, sizeof(stat_str), " (%.2f|%.2f|%.2f)",
                     stats[key]["min"], stats[key]["mean"], stats[key]["max"]);
            plot_label += stat_str;
        }

        // Add stable ID to prevent color changes when stats update
        plot_label += "##" + key;

        // Assign stable color based on key index in the vector
        // This ensures each key always gets the same color regardless of stats changes
        int colorIndex = keyIndex + color_ofs;
        colorIndex %= colormapSize;
        if (colorIndex < 0)
            colorIndex += colormapSize;
        ImVec4 lineColor = ImPlot::GetColormapColor(colorIndex);
        ImPlot::PushStyleColor(ImPlotCol_Line, lineColor);
        if (mPlotHideLegend) ImPlot::HideNextItem(true, ImPlotCond_Always);
        // Plot the line
        ImPlot::PlotLine(plot_label.c_str(), x.data(), y.data(), bufferSize);

        // Pop the color after plotting
        ImPlot::PopStyleColor();

        // Increment key index for next iteration
        keyIndex++;
    }
}

std::map<std::string, std::map<std::string, double>>
RenderCkpt::statGraphData(const std::vector<std::string>& keys, double xMin, double xMax)
{
    std::map<std::string, std::map<std::string, double>> result;

    if (!mGraphData) {
        LOG_WARN("[StatGraphData] mGraphData not found");
        return result;
    }

    const double timeStep = (mRenderEnv ? mRenderEnv->getWorld()->getTimeStep() : 1.0);

    for (const auto& key : keys)
    {
        if (!mGraphData->key_exists(key)) continue;
        std::vector<double> values = mGraphData->get(key);
        if (values.empty()) continue;

        int bufferSize = static_cast<int>(values.size());
        std::vector<double> filteredValues;

        // Filter values within [xMin, xMax] range
        // X-axis mapping: most recent data at x=0, older data at negative x values
        for (int i = 0; i < bufferSize; ++i)
        {
            double x = -(bufferSize - 1 - i) * timeStep;
            if (x >= xMin && x < xMax)
            {
                filteredValues.push_back(values[i]);
            }
        }

        if (filteredValues.empty())
            continue;

        // Compute statistics
        double minVal = *std::min_element(filteredValues.begin(), filteredValues.end());
        double maxVal = *std::max_element(filteredValues.begin(), filteredValues.end());
        double sum = std::accumulate(filteredValues.begin(), filteredValues.end(), 0.0);
        double meanVal = sum / filteredValues.size();

        result[key]["min"] = minVal;
        result[key]["mean"] = meanVal;
        result[key]["max"] = maxVal;
    }

    return result;
}

void RenderCkpt::plotPhaseBar(double x_min, double x_max, double y_min, double y_max)
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

void RenderCkpt::plotReferenceKinematics(const std::vector<std::string>& keys)
{
    const KinematicsExportData* kin = getActiveKinematics();
    if (!kin || !mShowReferenceKinematics)
        return;

    // Get current plot limits for phase mapping
    ImPlotRect limits = ImPlot::GetPlotLimits();
    double x_min = limits.X.Min;
    double x_max = limits.X.Max;

    int colormapSize = ImPlot::GetColormapSize();
    int keyIndex = 0;

    for (const auto& key : keys)
    {
        // Check if this key exists in reference kinematics
        // Try exact match first, then try without "angle_" prefix for backward compat
        std::string lookupKey = key;
        auto itMean = kin->mean.find(lookupKey);
        auto itStd = kin->std.find(lookupKey);

        if ((itMean == kin->mean.end() || itStd == kin->std.end())
            && key.rfind("angle_", 0) == 0)
        {
            // Try without "angle_" prefix
            lookupKey = key.substr(6);
            itMean = kin->mean.find(lookupKey);
            itStd = kin->std.find(lookupKey);
        }

        if (itMean == kin->mean.end() ||
            itStd == kin->std.end())
        {
            keyIndex++;
            continue;
        }

        const std::vector<double>& meanVec = itMean->second;
        const std::vector<double>& stdVec = itStd->second;
        int numSamples = static_cast<int>(meanVec.size());  // Typically 100

        if (numSamples == 0)
        {
            keyIndex++;
            continue;
        }

        // Map reference kinematics (phase 0-100%) to current x-axis range
        // Reference kinematics is phase-based, so map to visible x range
        std::vector<float> x_data(numSamples);
        std::vector<float> mean_data(numSamples);
        std::vector<float> upper_band(numSamples);
        std::vector<float> lower_band(numSamples);

        double visible_range = x_max - x_min;
        for (int i = 0; i < numSamples; ++i)
        {
            // Distribute samples across the visible x range
            double phase = static_cast<double>(i) / (numSamples - 1);
            x_data[i] = static_cast<float>(x_min + phase * visible_range);
            mean_data[i] = static_cast<float>(meanVec[i]);
            upper_band[i] = static_cast<float>(meanVec[i] + stdVec[i]);
            lower_band[i] = static_cast<float>(meanVec[i] - stdVec[i]);
        }

        // Get color for this key (same index as plotGraphData to match)
        int colorIndex = keyIndex % colormapSize;
        ImVec4 baseColor = ImPlot::GetColormapColor(colorIndex);

        // Plot std band as shaded region (transparent fill)
        ImVec4 bandColor = ImVec4(baseColor.x, baseColor.y, baseColor.z, 0.2f);
        ImPlot::PushStyleColor(ImPlotCol_Fill, bandColor);
        ImPlot::PlotShaded(("##ref_band_" + key).c_str(),
                           x_data.data(), lower_band.data(), upper_band.data(), numSamples);
        ImPlot::PopStyleColor();

        // Plot mean line as dashed line (lighter color, hidden from legend with ##)
        ImVec4 lineColor = ImVec4(baseColor.x * 0.8f, baseColor.y * 0.8f, baseColor.z * 0.8f, 0.7f);
        ImPlot::PushStyleColor(ImPlotCol_Line, lineColor);
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
        ImPlot::PlotLine(("##ref_mean_" + key).c_str(),
                         x_data.data(), mean_data.data(), numSamples);
        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();

        keyIndex++;
    }
}

void RenderCkpt::loadNormativeKinematics()
{
    std::string path = rm::resolve("data/normative_kinematics.h5");
    if (path.empty() || !fs::exists(path)) {
        LOG_WARN("[RenderCkpt] Normative kinematics not found: data/normative_kinematics.h5");
        mHasNormativeKinematics = false;
        return;
    }

    try {
        H5::H5File file(path, H5F_ACC_RDONLY);

        // Check if /kinematics group exists
        if (H5Lexists(file.getId(), "/kinematics", H5P_DEFAULT) <= 0) {
            LOG_WARN("[RenderCkpt] No /kinematics group in normative file");
            mHasNormativeKinematics = false;
            return;
        }

        H5::Group kinGroup = file.openGroup("/kinematics");

        // Read joint_names attribute (fixed-length string, matching C++ HDF export format)
        if (kinGroup.attrExists("joint_names")) {
            H5::Attribute namesAttr = kinGroup.openAttribute("joint_names");
            H5::StrType strType = namesAttr.getStrType();
            size_t strSize = strType.getSize();
            std::vector<char> buffer(strSize + 1, '\0');
            namesAttr.read(strType, buffer.data());
            std::string namesStr(buffer.data());

            // Parse comma-separated joint names
            mNormativeKinematics.jointKeys.clear();
            std::stringstream ss(namesStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    mNormativeKinematics.jointKeys.push_back(token);
                }
            }
        }

        // Read num_cycles attribute
        if (kinGroup.attrExists("num_cycles")) {
            H5::Attribute cyclesAttr = kinGroup.openAttribute("num_cycles");
            cyclesAttr.read(H5::PredType::NATIVE_INT, &mNormativeKinematics.numCycles);
        }

        // Read mean and std datasets for each joint
        mNormativeKinematics.mean.clear();
        mNormativeKinematics.std.clear();

        for (const auto& jointKey : mNormativeKinematics.jointKeys) {
            std::string meanPath = jointKey + "_mean";
            std::string stdPath = jointKey + "_std";

            if (H5Lexists(kinGroup.getId(), meanPath.c_str(), H5P_DEFAULT) > 0 &&
                H5Lexists(kinGroup.getId(), stdPath.c_str(), H5P_DEFAULT) > 0) {
                H5::DataSet meanDs = kinGroup.openDataSet(meanPath);
                H5::DataSet stdDs = kinGroup.openDataSet(stdPath);

                H5::DataSpace meanSpace = meanDs.getSpace();
                hsize_t dims[1];
                meanSpace.getSimpleExtentDims(dims);

                std::vector<double> meanVec(dims[0]);
                std::vector<double> stdVec(dims[0]);

                meanDs.read(meanVec.data(), H5::PredType::NATIVE_DOUBLE);
                stdDs.read(stdVec.data(), H5::PredType::NATIVE_DOUBLE);

                mNormativeKinematics.mean[jointKey] = meanVec;
                mNormativeKinematics.std[jointKey] = stdVec;
            }
        }

        mHasNormativeKinematics = !mNormativeKinematics.mean.empty();
        if (mHasNormativeKinematics) {
            LOG_INFO("[RenderCkpt] Loaded normative kinematics: " << mNormativeKinematics.numCycles
                     << " cycles, " << mNormativeKinematics.mean.size() << " joints");
        }

    } catch (const H5::Exception& e) {
        LOG_ERROR("[RenderCkpt] Failed to load normative kinematics: " << e.getCDetailMsg());
        mHasNormativeKinematics = false;
    }
}

const KinematicsExportData* RenderCkpt::getActiveKinematics() const
{
    if (mKinematicsSource == KinematicsSource::FromNormative && mHasNormativeKinematics)
        return &mNormativeKinematics;
    if (mKinematicsSource == KinematicsSource::FromMotion && mHasReferenceKinematics)
        return &mReferenceKinematics;
    return nullptr;
}

std::string RenderCkpt::getActiveKinematicsLabel() const
{
    if (mKinematicsSource == KinematicsSource::FromNormative && mHasNormativeKinematics) {
        return "Normative (gait120)";
    }
    if (mKinematicsSource == KinematicsSource::FromMotion && mHasReferenceKinematics && mMotion) {
        std::string fullPath = mMotion->getName();
        // Try to get PID from navigator state
        if (mPIDNavigator) {
            std::string pid = mPIDNavigator->getState().getSelectedPID();
            if (!pid.empty()) {
                // Extract relative path after PID directory
                // Path format: /base/path/{pid}/{visit}/{data_type}/{filename}
                size_t pidPos = fullPath.find("/" + pid + "/");
                if (pidPos != std::string::npos) {
                    std::string relPath = fullPath.substr(pidPos + 1);  // {pid}/{visit}/...
                    return "@pid:" + relPath;
                }
            }
        }
        // Fallback: just filename
        return fs::path(fullPath).filename().string();
    }
    return "";
}

float RenderCkpt::getHeelStrikeTime()
{
    if (!mGraphData->key_exists("contact_phaseR"))
    {
        LOG_WARN("[HeelStrike] contact_phaseR key not found in graph data");
        return 0.0;
    }

    if (!mRenderEnv) return 0.0;

    std::vector<double> contact_phase_buffer = mGraphData->get("contact_phaseR");

    // Ensure there are at least two points to compare for transitions
    if (contact_phase_buffer.size() < 2)
    {
        LOG_WARN("[HeelStrike] Not enough data points for heel strike detection");
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
        LOG_INFO("[HeelStrike] Found heel strike at time: " << heel_strike_time);
    }
    else
    {
        LOG_WARN("[HeelStrike] No heel strike found in current data");
    }
    return heel_strike_time;
}

void RenderCkpt::onFrameStart()
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
    else
    {
        // Paused: check if either motion OR marker is in manual mode
        PlaybackNavigationMode navMode = PLAYBACK_SYNC;
        if (mMotion != nullptr) {
            navMode = mMotionState.navigationMode;
        }

        bool needUpdate = (navMode == PLAYBACK_MANUAL_FRAME);

        if (needUpdate) {
            // Paused + manual mode: compute pose but don't advance time
            updateViewerTime(0.0);
        }
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
}

void RenderCkpt::onMaxRecordingTimeReached()
{
    // Pause simulation when video recording reaches max time
    mRolloutStatus.pause = true;
    // Restore camera mode if orbit was enabled
    if (mVideoOrbitEnabled) {
        mCamera.focus = mPreRecordingFocusMode;
    }
    // Restore playback speed
    mViewerPlaybackSpeed = mPreRecordingPlaybackSpeed;
}

double RenderCkpt::getSimulationTime() const
{
    // Return actual simulation world time for accurate video recording timing
    if (mRenderEnv && mRenderEnv->getWorld()) {
        return mRenderEnv->getWorld()->getTime();
    }
    return 0.0;
}

void RenderCkpt::initGL()
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

void RenderCkpt::initializeMotionCharacter(const std::string& metadata)
{
    // Skip if already initialized
    if (mMotionCharacter) return;

    // Extract skeleton path from metadata file and create motion character
    // This allows motion playback without full simulation environment
    std::string skelPath;

    try {
        YAML::Node config = YAML::LoadFile(metadata);
        if (config["environment"] && config["environment"]["skeleton"] && config["environment"]["skeleton"]["file"]) {
            skelPath = config["environment"]["skeleton"]["file"].as<std::string>();

            // Get global pid for @pid:/ URI expansion
            std::string globalPid;
            if (config["environment"]["pid"]) {
                globalPid = config["environment"]["pid"].as<std::string>();
            }
            skelPath = rm::resolve(rm::expand_pid(skelPath, globalPid));
        }
    } catch (const std::exception& e) {
        LOG_WARN("[Motion] Failed to parse metadata for skeleton path: " << e.what());
        return;
    }

    if (skelPath.empty()) {
        LOG_WARN("[Motion] No skeleton path found in metadata, motion rendering unavailable");
        return;
    }

    try {
        mSkeletonPath = skelPath;
        mMotionCharacter = new RenderCharacter(mSkeletonPath);
        mMotionCharacter->loadMarkers("data/marker/default.xml");
        LOG_INFO("[Motion] Initialized standalone motion character from: " << mSkeletonPath);
    } catch (const std::exception& e) {
        LOG_ERROR("[Motion] Failed to create motion character: " << e.what());
    }
}

void RenderCkpt::initEnv(std::string metadata)
{
    if (mRenderEnv)
    {
        delete mRenderEnv;
        mRenderEnv = nullptr;
    }

    // Create RenderEnvironment from metadata path
    mRenderEnv = new RenderEnvironment(metadata, mGraphData);

    // Register muscle activation keys for graphing
    for (const auto& muscle: mRenderEnv->getCharacter()->getMuscles()) {
        const auto& muscle_name = muscle->GetName();
        mGraphData->register_key("act_" + muscle_name, 1000);
        mGraphData->register_key("noise_" + muscle_name, 1000);
        mGraphData->register_key("fp_" + muscle_name, 1000);
        mGraphData->register_key("fa_" + muscle_name, 1000);
        mGraphData->register_key("ft_" + muscle_name, 1000);
        mGraphData->register_key("lm_" + muscle_name, 1000);
    }

    // Register torque keys for each DOF (excluding root joint)
    mPlotJointDofNames.clear();
    auto skel = mRenderEnv->getCharacter()->getSkeleton();
    std::vector<std::string> axisSuffixes = {"_x", "_y", "_z"};
    for (size_t i = 1; i < skel->getNumJoints(); ++i) {  // Skip root (i=0)
        auto joint = skel->getJoint(i);
        std::string jointName = joint->getName();
        int numDofs = joint->getNumDofs();

        for (int d = 0; d < numDofs; ++d) {
            std::string suffix = (d < 3) ? axisSuffixes[d] : "_" + std::to_string(d);
            std::string dofName = jointName + suffix;
            mGraphData->register_key("torque_" + dofName, 1000);
            mPlotJointDofNames.push_back(dofName);
        }
    }
    mPlotJointSelected.resize(mPlotJointDofNames.size(), false);

    // Set window title
    if (!mRolloutStatus.name.empty()) {
        mCheckpointName = mRolloutStatus.name;
    } else if (!mNetworkPaths.empty()) {
        mCheckpointName = std::filesystem::path(mNetworkPaths.back()).stem().string();
    } else {
        mCheckpointName = std::filesystem::path(mCachedMetadata).parent_path().filename().string();
    }
    glfwSetWindowTitle(mWindow, mCheckpointName.c_str());

    // Navigate PID Navigator to the environment's global PID if specified
    const std::string& globalPid = mRenderEnv->getGlobalPid();
    if (mPIDNavigator && !globalPid.empty()) {
        // Parse "PID/visit" format (e.g., "29792292/pre")
        std::string pid = globalPid;
        std::string visit = "pre";
        size_t slashPos = globalPid.find('/');
        if (slashPos != std::string::npos) {
            pid = globalPid.substr(0, slashPos);
            visit = globalPid.substr(slashPos + 1);
        }

        if (mPIDNavigator->navigateTo(pid, visit)) {
            LOG_INFO("[RenderCkpt] PID Navigator initialized to " << pid << "/" << visit);
        }
    }

    // Initialize motion skeleton first (needed for motion loading)
    initializeMotionSkeleton();

    // // Hardcoded: Load Sim_Healthy.npz as reference motion
    // // Use mMotionCharacter which matches the NPZ data format (not render character)
    // try {
    //     std::string npz_path = "data/npz_motions/Sim_Healthy.npz";
    //     if (fs::exists(npz_path)) {
    //         std::cout << "[RenderCkpt] Loading hardcoded reference motion: " << npz_path << std::endl;
    //         NPZ* npz = new NPZ(npz_path);
    //         // CRITICAL: Use mMotionCharacter (NPZ-compatible) not mRenderEnv->getCharacter()
    //         npz->setRefMotion(mMotionCharacter, mRenderEnv->getWorld());
    //         mRenderEnv->setMotion(npz);
    //         std::cout << "[RenderCkpt] Successfully loaded hardcoded NPZ reference motion" << std::endl;
    //     } else {
    //         std::cerr << "[RenderCkpt] Hardcoded NPZ file not found: " << npz_path << std::endl;
    //     }
    // } catch (const std::exception& e) {
    //     std::cerr << "[RenderCkpt] Error loading hardcoded NPZ: " << e.what() << std::endl;
    // }

    // Load networks
    auto character = mRenderEnv->getCharacter();
    mNetworks.clear();
    for (const auto& path : mNetworkPaths) {
        loadNetworkFromPath(path);
    }
    
    // Load muscle network weights into the Environment's MuscleNN
    // (The C++ MuscleNN is created automatically in initialize())
    if (!mNetworks.empty() && mNetworks.back().muscle && !mMuscleStateDict.is_none()) {
        // Transfer the stored Python state_dict to Environment's MuscleNN
        mRenderEnv->setMuscleNetworkWeight(mMuscleStateDict);
    }

    // Initialize DOF tracking
    mRelatedDofs.clear();
    mRelatedDofs.resize(mRenderEnv->getCharacter()->getSkeleton()->getNumDofs() * 2, false);

    // Initialize muscle selection states
    if (mRenderEnv->getUseMuscle()) {
        auto muscles = character->getMuscles();
        mMuscleSelectionStates.clear();
        mMuscleSelectionStates.resize(muscles.size(), true);

        // Initialize plot muscle names and selection
        mPlotMuscleNames.clear();
        for (const auto& m : muscles) {
            mPlotMuscleNames.push_back(m->name);
        }
        mPlotMuscleSelected.resize(muscles.size(), false);
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

    mRenderEnv->setParamDefault();
    reset();
}

void RenderCkpt::drawAxis()
{
    // Get character root position if available
    Eigen::Vector3d origin(0, 2E-3, 0);
    if (mRenderEnv && mRenderEnv->getCharacter()) {
        auto skel = mRenderEnv->getCharacter()->getSkeleton();
        origin = skel->getRootBodyNode()->getTransform().translation();
        origin[1] = 2E-3; // Keep the y at ground level
    }

    GUI::DrawLine(origin, origin + Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0));
    GUI::DrawLine(origin, origin + Eigen::Vector3d(0.0, 0.5, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0));
    GUI::DrawLine(origin, origin + Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d(0.0, 0.0, 1.0));
}

void RenderCkpt::drawJointAxis(dart::dynamics::Joint* joint)
{
    if (!joint) return;

    auto parentBodyNode = joint->getParentBodyNode();
    auto childBodyNode = joint->getChildBodyNode();
    if (!parentBodyNode || !childBodyNode) return;

    // Get the complete transform from parent body to joint location
    Eigen::Isometry3d parentToJoint = parentBodyNode->getTransform() *
                                      joint->getTransformFromParentBodyNode();

    // Joint position in world space
    Eigen::Vector3d p = parentToJoint.translation();

    // Child body's orientation in world space (reflects actual rotation state)
    Eigen::Matrix3d rotation = childBodyNode->getTransform().linear();

    // Axes directions from child body's orientation
    const double axis_length = 0.15;
    Eigen::Vector3d axis_x = p + rotation.col(0) * axis_length;
    Eigen::Vector3d axis_y = p + rotation.col(1) * axis_length;
    Eigen::Vector3d axis_z = p + rotation.col(2) * axis_length;

    // Disable depth test so axes render on top of everything (including muscles)
    glDisable(GL_DEPTH_TEST);

    // Make axes thicker for better visibility
    glLineWidth(4.0);

    // Draw joint axes with RGB colors
    GUI::DrawLine(p, axis_x, Eigen::Vector3d(1.0, 0.0, 0.0)); // Red for X
    GUI::DrawLine(p, axis_y, Eigen::Vector3d(0.0, 1.0, 0.0)); // Green for Y
    GUI::DrawLine(p, axis_z, Eigen::Vector3d(0.0, 0.0, 1.0)); // Blue for Z

    // Reset line width to default
    glLineWidth(1.0);

    // Re-enable depth test
    glEnable(GL_DEPTH_TEST);
}

void RenderCkpt::drawKinematicsControlPanelContent()
{
    // // FGN
    // ImGui::Checkbox("Draw FGN Result\t", &mDrawFlags.fgnSkeleton);
    // if (ImGui::CollapsingHeader("FGN"))
    // {
    //     int idx = 0;
    //     for (const auto &ns : mFGNList)
    //     {
    //         std::string filename = fs::path(ns).filename().string();
    //         if (ImGui::Selectable(filename.c_str(), selected_fgn == idx))
    //             selected_fgn = idx;
    //         if (selected_fgn)
    //             ImGui::SetItemDefaultFocus();
    //         idx++;
    //     }
    // }

    // if (mRenderEnv)
    // {
    //     if (ImGui::Button("Load FGN"))
    //     {
    //         mDrawFlags.fgnSkeleton = true;
    //         py::tuple res = py::module::import("forward_gaitnet").attr("load_FGN")(mFGNList[selected_fgn], mRenderEnv->getNumParamState(), mRenderEnv->getCharacter()->posToSixDof(mRenderEnv->getCharacter()->getSkeleton()->getPositions()).rows());
    //         mFGN = res[0];
    //         mFGNmetadata = res[1].cast<std::string>();

    //         mNetworkPaths.clear();
    //         mNetworks.clear();
    //         std::cout << "METADATA " << std::endl
    //                   << mFGNmetadata << std::endl;
    //         initEnv(mFGNmetadata);
    //     }
    // }

    // // BGN
    // if (ImGui::CollapsingHeader("BGN"))
    // {
    //     int idx = 0;
    //     for (const auto &ns : mBGNList)
    //     {
    //         std::string filename = fs::path(ns).filename().string();
    //         if (ImGui::Selectable(filename.c_str(), selected_bgn == idx))
    //             selected_bgn = idx;
    //         if (selected_bgn)
    //             ImGui::SetItemDefaultFocus();
    //         idx++;
    //     }
    // }
    // if (mRenderEnv && ImGui::Button("Load BGN"))
    // {
    //     mGVAELoaded = true;
    //     py::object load_gaitvae = py::module::import("advanced_vae").attr("load_gaitvae");
    //     int rows = mRenderEnv->getCharacter()->posToSixDof(mRenderEnv->getCharacter()->getSkeleton()->getPositions()).rows();
    //     mGVAE = load_gaitvae(mBGNList[selected_fgn], rows, 60, mRenderEnv->getNumKnownParam(), mRenderEnv->getNumParamState());

    //     // TODO: Update for Motion* interface
    //     // mPredictedMotion.motion = mMotion.motion;
    //     // mPredictedMotion.param = mMotion.param;
    //     // mPredictedMotion.name = "Unpredicted";
    // }

    if (ImGui::CollapsingHeader("Motions", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int mMotionPhaseOffset = 0;

        // Display motion status
        bool has_motions = mMotion != nullptr;
        if (has_motions) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Motion Loaded");
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "No Motion Loaded");
        }

        // Unload motion button (loading is done by clicking in the file list)
        if (has_motions) {
            if (ImGui::Button("Unload Motion")) {
                unloadMotion();
            }
        }

        // Unified motion file list (HDF + C3D)
        bool motion_loaded = (mMotion != nullptr);
        if (motion_loaded) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));  // Green when selected
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.4f, 0.8f, 0.4f, 1.0f));
        }

        if (ImGui::TreeNodeEx("Motion Files", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (mMotionList.empty()) {
                ImGui::Text("No motion files found");
            } else {
                int idx = 0;
                for (const auto& file : mMotionList) {
                    std::filesystem::path filepath(file);
                    // HDF only (C3D moved to c3d_processor)
                    std::string label = "[HDF] " + filepath.filename().string();

                    if (ImGui::Selectable(label.c_str(), mSelectedMotion == idx)) {
                        mSelectedMotion = idx;
                        loadMotionFile(mMotionList[mSelectedMotion]);
                    }
                    if (mSelectedMotion == idx) ImGui::SetItemDefaultFocus();
                    idx++;
                }
            }

            // C3D marker controls moved to c3d_processor executable

            ImGui::TreePop();
        }

        if (motion_loaded) {
            ImGui::PopStyleColor(3);
        }

        // Motion Navigation Control
        ImGui::Separator();
        PlaybackViewerState* motionStatePtr = nullptr;
        if (mMotion != nullptr) {
            motionStatePtr = &mMotionState;
        }

        // Use unified navigation UI for motion playback
        if (motionStatePtr) {
            PlaybackUtils::drawPlaybackNavigationUI("Motion Frame Nav", *motionStatePtr, motionStatePtr->maxFrameIndex);

            // Progress forward toggle
            ImGui::Checkbox("Progress Forward", &mProgressForward);

            // Show additional motion-specific info
            if (motionStatePtr->navigationMode == PLAYBACK_MANUAL_FRAME) {
                // Show frame time for HDF/C3D motions with timestamps in manual mode
                if (mMotion != nullptr) {
                    std::vector<double> timestamps = mMotion->getTimestamps();
                    int manualIndex = std::clamp(motionStatePtr->manualFrameIndex, 0, motionStatePtr->maxFrameIndex);
                    if ((mMotion->getSourceType() == "hdf" ||
                         mMotion->getSourceType() == "c3d") &&
                        !timestamps.empty() &&
                        manualIndex < static_cast<int>(timestamps.size())) {
                        double frame_time = timestamps[manualIndex];
                        ImGui::Text("Time: %.3f s", frame_time);
                    }
                }
            } else {
                // Show current auto-computed frame in sync mode
                if (mMotion != nullptr) {
                    double phase = mViewerPhase;
                    if (mRenderEnv) {
                        phase = mViewerTime / (mRenderEnv->getMotion()->getMaxTime() / (mRenderEnv->getCadence() / sqrt(mRenderEnv->getCharacter()->getGlobalRatio())));
                        phase = fmod(phase, 1.0);
                    }
                    double frame_float = computeFrameFloat(mMotion, phase);
                    int current_frame = (int)frame_float;
                    ImGui::Text("Auto Frame: %d / %d", current_frame, motionStatePtr->maxFrameIndex);
                }
            }
        }

        // Marker Navigation Control (only for C3D motions)
        ImGui::Separator();
        if (mMotion && mMotion->getSourceType() == "c3d") {
            PlaybackUtils::drawPlaybackNavigationUI("Marker Frame Nav", mMotionState, mMotionState.maxFrameIndex);
        } else {
            ImGui::TextDisabled("No C3D motion loaded");
        }
        ImGui::Separator();
        ImGui::Spacing();
    }

    // Clinical Data Section (PID-based HDF access)
    drawClinicalDataSection();
}

void RenderCkpt::drawRightPanel()
{   
    ImGui::SetNextWindowSize(ImVec2(mPlotPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::Begin("Visualization", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowPos(ImVec2(mWidth - ImGui::GetWindowSize().x, 0), ImGuiCond_Always);

    // Plot X-axis range control (available without mRenderEnv)
    if (ImGui::Button("HS")) mXmin = getHeelStrikeTime();
    ImGui::SameLine();
    if (ImGui::Button("1.1")) mXmin = -1.1;
    ImGui::SameLine();
    if (ImGui::Button("0")) mXmin = 0.0;
    ImGui::SameLine();
    mPlotHideLegend = ImGui::Button("Hide"); ImGui::SameLine();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(30);
    ImGui::InputDouble("X(min)", &mXmin);
    ImGui::SameLine();

    // Reference kinematics source selector and overlay toggle
    bool hasAnyKinematics = mHasReferenceKinematics || mHasNormativeKinematics;
    if (hasAnyKinematics) {
        ImGui::Checkbox("Ref Overlay", &mShowReferenceKinematics);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (ImGui::BeginCombo("##RefSrc",
            mKinematicsSource == KinematicsSource::FromMotion ? "Motion" : "Normative"))
        {
            if (mHasReferenceKinematics) {
                if (ImGui::Selectable("Motion", mKinematicsSource == KinematicsSource::FromMotion)) {
                    mKinematicsSource = KinematicsSource::FromMotion;
                    mKinematicsSourceInt = 0;
                }
            }
            if (mHasNormativeKinematics) {
                if (ImGui::Selectable("Normative", mKinematicsSource == KinematicsSource::FromNormative)) {
                    mKinematicsSource = KinematicsSource::FromNormative;
                    mKinematicsSourceInt = 1;
                }
            }
            ImGui::EndCombo();
        }
    }

    ImGui::SameLine();
    // Plot title control
    ImGui::Checkbox("Title##PlotTitleCheckbox", &mPlotTitle);
    ImGui::SameLine();
    ImGui::TextDisabled("(Show checkpoint name as plot titles)");

    // C3D marker sections moved to c3d_processor executable

    if (!mRenderEnv)
    {
        ImGui::Text("Environment not loaded.");

        ImGui::End();
        return;
    }

    // Tab bar for visualization panels
    if (ImGui::BeginTabBar("VisualizationTabs")) {
        if (ImGui::BeginTabItem("Gait")) {
            drawGaitTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Kinematics")) {
            drawKinematicsTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Kinetics")) {
            drawKineticsTabContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Muscle")) {
            drawMuscleTabContent();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

// ============================================================
// Gait Tab Content
// ============================================================
void RenderCkpt::drawGaitTabContent()
{
    // Status
    if (ImGui::CollapsingHeader("Status", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Target / Average Vel      : %.3f / %.3f m/s", mRenderEnv->getTargetCOMVelocity(), mRenderEnv->getAvgVelocity()[2]);

        // Character position
        Eigen::Vector3d char_pos = mRenderEnv->getCharacter()->getSkeleton()->getRootBodyNode()->getCOM();
        ImGui::Text("Character Pos   : (%.3f, %.3f, %.3f)", char_pos[0], char_pos[1], char_pos[2]);

        ImGui::Separator();

        // Gait Phase State
        auto gaitPhase = mRenderEnv->getGaitPhase();

        // Display current state with color coding
        const char* stateNames[] = {"RIGHT_STANCE", "LEFT_TAKEOFF", "LEFT_STANCE", "RIGHT_TAKEOFF"};
        ImVec4 stateColors[] = {
            ImVec4(0.0f, 1.0f, 0.0f, 1.0f),  // RIGHT_STANCE - green
            ImVec4(1.0f, 1.0f, 0.0f, 1.0f),  // LEFT_TAKEOFF - yellow
            ImVec4(0.0f, 0.5f, 1.0f, 1.0f),  // LEFT_STANCE - blue
            ImVec4(1.0f, 0.5f, 0.0f, 1.0f)   // RIGHT_TAKEOFF - orange
        };

        int currentState = static_cast<int>(gaitPhase->getCurrentState());
        ImGui::Text("Gait State      : ");
        ImGui::SameLine();
        ImGui::TextColored(stateColors[currentState], "%s", stateNames[currentState]);

        // Display stance ratios
        double stanceRatioR = gaitPhase->getStanceRatioR();
        double stanceRatioL = gaitPhase->getStanceRatioL();

        ImGui::Text("Stance Ratio R  : %.3f", stanceRatioR);
        ImGui::SameLine();
        ImGui::ProgressBar(stanceRatioR, ImVec2(100, 0));

        ImGui::Text("Stance Ratio L  : %.3f", stanceRatioL);
        ImGui::SameLine();
        ImGui::ProgressBar(stanceRatioL, ImVec2(100, 0));

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

    // Gait Phase Metrics
    if (collapsingHeaderWithControls("Gait Metrics"))
    {
        // Plot selection controls
        static int gaitMetricSelection = 0;  // 0 = Stride Length, 1 = Phase Total, 2 = Local Phase
        static bool showGaitStats = true;

        // Checkbox for stats
        ImGui::Checkbox("Stats##GaitStats", &showGaitStats);
        // Radio buttons for metric selection
        ImGuiCommon::RadioButtonGroup("GaitMetric",
            {"Stride Length", "Step time", "Local Phase"}, &gaitMetricSelection);
        ImGui::Separator();

        // Stride Length plot (gaitMetricSelection == 0)
        if (gaitMetricSelection == 0) {
            std::string title_stride = mPlotTitle ? mCheckpointName : "Stride Length (m)";
            if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(ImAxis_X1, mXmin, 0, ImGuiCond_Always);
            else ImPlot::SetNextAxisLimits(ImAxis_X1, -0.5, 0, ImGuiCond_Once);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, 1.25, 1.4, ImPlotCond_Once);
            if (ImPlot::BeginPlot((title_stride + "##StrideLength").c_str(), ImVec2(-1, getPlotHeight("Gait Metrics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Stride Length (m)");

                // Plot stride length data (unified key for both feet)
                plotGraphData({"stride_length"}, ImAxis_Y1, "", showGaitStats);

                // Plot target stride as horizontal line
                double targetStride = mRenderEnv->getStride() * mRenderEnv->getRefStride() * mRenderEnv->getCharacter()->getGlobalRatio();
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 0.8f)); // Red color
                ImPlot::PlotInfLines("target##stride", &targetStride, 1, ImPlotInfLinesFlags_Horizontal);
                ImPlot::PopStyleColor();

                ImPlot::EndPlot();
            }
        }

        // Phase Total plot (gaitMetricSelection == 1)
        if (gaitMetricSelection == 1) {
            std::string title_phase = mPlotTitle ? mCheckpointName : "Phase Total (s)";
            if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(ImAxis_X1, mXmin, 0, ImGuiCond_Always);
            else ImPlot::SetNextAxisLimits(ImAxis_X1, -0.5, 0, ImGuiCond_Once);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, 1.05, 1.15, ImPlotCond_Once);
            if (ImPlot::BeginPlot((title_phase + "##PhaseTotal").c_str(), ImVec2(-1, getPlotHeight("Gait Metrics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Phase Total (s)");

                // Plot phase total data (unified key for both feet)
                plotGraphData({"phase_total"}, ImAxis_Y1, "", showGaitStats);

                // Plot target phase total as horizontal line
                // Target phase total = motion cycle time / cadence
                double motionCycleTime = mRenderEnv->getMotion()->getMaxTime() / sqrt(mRenderEnv->getCharacter()->getGlobalRatio());
                double targetPhaseTotal = motionCycleTime / mRenderEnv->getCadence();
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 0.8f)); // Red color
                ImPlot::PlotInfLines("target##phase", &targetPhaseTotal, 1, ImPlotInfLinesFlags_Horizontal);
                ImPlot::PopStyleColor();

                ImPlot::EndPlot();
            }
        }

        // Local Phase plot (gaitMetricSelection == 2)
        if (gaitMetricSelection == 2) {
            std::string title_local_phase = mPlotTitle ? mCheckpointName : "Local Phase";
            if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(ImAxis_X1, mXmin, 0, ImGuiCond_Always);
            else ImPlot::SetNextAxisLimits(ImAxis_X1, -1.5, 0, ImGuiCond_Once);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, -0.05, 1.05, ImPlotCond_Once);
            if (ImPlot::BeginPlot((title_local_phase + "##LocalPhase").c_str(), ImVec2(-1, getPlotHeight("Gait Metrics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Local Phase");

                // Plot local phase data
                plotGraphData({"local_phase"}, ImAxis_Y1, "", showGaitStats);

                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);    

                ImPlot::EndPlot();
            }
        }
    }

    // Rewards
    if (collapsingHeaderWithControls("Rewards"))
    {
        std::string title_str = mPlotTitle ? mCheckpointName : "Reward";
        if (ImPlot::BeginPlot((title_str + "##Reward").c_str(), ImVec2(-1, getPlotHeight("Rewards"))))
        {
            ImPlot::SetupAxes("Time (s)", "Reward");

            // Plot reward data using common plotting function
            std::vector<std::string> rewardKeys = {"r", "r_p", "r_v", "r_com", "r_ee", "r_energy", "r_knee_pain", "r_loco", "r_head_linear_acc", "r_head_rot_diff", "r_avg", "r_step", "r_drag_x", "r_phase"};
            if (mRenderEnv->getSeparateTorqueEnergy()) {
                rewardKeys.push_back("r_torque");
                rewardKeys.push_back("r_metabolic");
            }
            plotGraphData(rewardKeys, ImAxis_Y1);
            ImPlot::EndPlot();
        }
    }

    // Metabolic Energy
    if (collapsingHeaderWithControls("Energy"))
    {
        // Display current metabolic type
        MetabolicType currentType = mRenderEnv->getCharacter()->getMetabolicType();
        const char* typeNames[] = {"A", "A2", "MA", "MA2"};
        if (currentType == MetabolicType::LEGACY) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Mode: LEGACY (Disabled)");
        } else {
            const int typeIndex = static_cast<int>(currentType) - static_cast<int>(MetabolicType::A);
            const char* displayName = (typeIndex >= 0 && typeIndex < IM_ARRAYSIZE(typeNames))
                ? typeNames[typeIndex]
                : "Unknown";
            ImGui::Text("Mode: %s", displayName);

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
            if (ImPlot::BeginPlot((title_energy + "##Energy").c_str(), ImVec2(-1, getPlotHeight("Energy"))))
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
                plotGraphData(metabolicKeys, ImAxis_Y1);

                ImPlot::EndPlot();
            }
        }
    }
}

// ============================================================
// Kinematics Tab Content
// ============================================================
void RenderCkpt::drawKinematicsTabContent()
{
    // Center of Mass Trajectory
    if (collapsingHeaderWithControls("COM Trajectory"))
    {
        // Plot selection controls
        static int plotSelection = 1;  // 0 = Trajectory, 1 = Velocity, 2 = Deviation
        static bool showStats = true;

        // Checkbox for stats
        ImGui::Checkbox("Stats", &showStats);

        // Radio buttons for plot selection
        ImGuiCommon::RadioButtonGroup("COMPlotSelection",
            {"Trajectory", "Velocity", "Deviation"}, &plotSelection);

        ImGui::Separator();

        // Static variables for width and height
        static double comPlotWidth = 4.0;   // Z-axis range (forward/backward)
        static double comPlotHeight = 0.2;  // X-axis range (left/right)

        // Width and height controls (only for trajectory plot)
        if (plotSelection == 0) {
            ImGui::SetNextItemWidth(70);
            ImGui::InputDouble("Width", &comPlotWidth, 0.1, 1.0, "%.2f");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(70);
            ImGui::InputDouble("Height", &comPlotHeight, 0.1, 1.0, "%.2f");

            // Minimum values
            if (comPlotWidth < 0.1) comPlotWidth = 0.1;
            if (comPlotHeight < 0.1) comPlotHeight = 0.1;
        }
        // Trajectory plot (plotSelection == 0)
        if (plotSelection == 0) {
            double half_width = comPlotWidth / 2.0;
            double half_height = comPlotHeight / 2.0;

            // Get COM data from GraphData first to calculate limits
            double mean_x = 0.0, mean_z = 0.0;

            if (mGraphData->key_exists("com_x") && mGraphData->key_exists("com_z"))
            {
                auto com_x_data = mGraphData->get("com_x");
                auto com_z_data = mGraphData->get("com_z");

                if (!com_x_data.empty() && !com_z_data.empty())
                {
                    // Calculate mean of X and Z
                    size_t count = std::min(com_x_data.size(), com_z_data.size());
                    for (size_t i = 0; i < count; i++) {
                        mean_x += com_x_data[i];
                        mean_z += com_z_data[i];
                    }
                    mean_x /= count;
                    mean_z /= count;
                }
            }
            ImPlot::SetNextAxisLimits(ImAxis_X1, mean_z - half_width, mean_z + half_width, ImPlotCond_Always);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, mean_x - half_height, mean_x + half_height, ImPlotCond_Always);
            std::string title_com = mPlotTitle ? mCheckpointName : "COM Trajectory";
            if (ImPlot::BeginPlot((title_com + "##COMTraj").c_str(), ImVec2(-1, getPlotHeight("COM Trajectory"))))
            {
                // Setup axes with limits BEFORE plotting
                ImPlot::SetupAxes("Z Position (m)", "X Position (m)");

                // Now plot the data
                if (mGraphData->key_exists("com_x") && mGraphData->key_exists("com_z"))
                {
                    auto com_x_data = mGraphData->get("com_x");
                    auto com_z_data = mGraphData->get("com_z");
                    size_t count = std::min(com_x_data.size(), com_z_data.size());

                    ImPlot::PlotLine("COM Path",
                                   com_z_data.data(),  // X-axis: Z position
                                   com_x_data.data(),  // Y-axis: X position
                                   count);
                }

                ImPlot::EndPlot();
            }
        }

        // Velocity plot (plotSelection == 1)
        if (plotSelection == 1) {
            // Velocity method selection
            static int velocityMethod = 1;  // 0 = Least Squares, 1 = Avg Horizon
            ImGuiCommon::RadioButtonGroup("VelMethod",
                {"Least Squares", "Avg Horizon"}, &velocityMethod);

            // Update RenderEnvironment with selected method
            mRenderEnv->setVelocityMethod(velocityMethod);

            // Velocity plot
            std::string title_vel = mPlotTitle ? mCheckpointName : "COM Velocity (m/s)";
            if (std::abs(mXmin) > 1e-6) ImPlot::SetNextAxisLimits(ImAxis_X1, mXmin, 0, ImGuiCond_Always);
            else ImPlot::SetNextAxisLimits(ImAxis_X1, -10.0, 0, ImGuiCond_Once);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, 1.15, 1.3, ImPlotCond_Once);
            if (ImPlot::BeginPlot((title_vel + "##COMVel").c_str(), ImVec2(-1, getPlotHeight("COM Trajectory"))))
            {
                ImPlot::SetupAxes("Time (s)", "Z Velocity (m/s)");

                // Plot velocity data using common plotting function
                plotGraphData({"com_vel_z"}, ImAxis_Y1, "", showStats);

                // Plot target velocity as horizontal line
                double targetVel = mRenderEnv->getTargetCOMVelocity();
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 0.8f)); // Red color
                ImPlot::PlotInfLines("target_vel##target", &targetVel, 1, ImPlotInfLinesFlags_Horizontal);
                ImPlot::PopStyleColor();

                ImPlot::EndPlot();
            }
        }

        // Deviation plot (plotSelection == 2)
        if (plotSelection == 2) {
            std::string title_err = mPlotTitle ? mCheckpointName : "Lateral Deviation Error (m)";
            ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 10, ImPlotCond_Once);
            if (ImPlot::BeginPlot((title_err + "##COMRegErr").c_str(), ImVec2(-1, getPlotHeight("COM Trajectory"))))
            {
                ImPlot::SetupAxes("Time (s)", "Mean Error (mm)");

                // Plot regression error
                std::vector<std::string> errKeys = {"com_deviation"};
                plotGraphData(errKeys, ImAxis_Y1, "", showStats);

                ImPlot::EndPlot();
            }
        }
    }

    // Kinematics
    if (collapsingHeaderWithControls("Kinematics"))
    {
        static int angle_selection = 0; // 0=Major, 1=Minor, 2=Pelvis, 3=Sway, 4=Anteversion
        static bool stats = true;
        ImGui::Checkbox("Stats##KinematicsStats", &stats);

        ImGuiCommon::RadioButtonGroup("KinematicsAngle",
            {"Major", "Minor", "Pelvis", "Sway", "Anteversion"}, &angle_selection);

        // Helper lambda for computing MSE against active kinematics source
        // Computes MSE only for data within visible plot range [x_min, x_max]
        auto computeMSE = [this](const std::string& key, double x_min, double x_max) -> double {
            const KinematicsExportData* kin = getActiveKinematics();
            if (!kin || !mGraphData || !mRenderEnv) return -1.0;

            // Find reference data
            std::string lookupKey = key;
            auto itMean = kin->mean.find(lookupKey);
            if (itMean == kin->mean.end() && key.rfind("angle_", 0) == 0) {
                lookupKey = key.substr(6);
                itMean = kin->mean.find(lookupKey);
            }
            if (itMean == kin->mean.end()) return -1.0;
            if (!mGraphData->key_exists(key)) return -1.0;

            const std::vector<double>& refMean = itMean->second;
            std::vector<double> allSimData = mGraphData->get(key);
            if (allSimData.empty() || refMean.empty()) return -1.0;

            // Extract simulation data within visible plot range
            const double timeStep = mRenderEnv->getWorld()->getTimeStep();
            int bufferSize = static_cast<int>(allSimData.size());
            std::vector<double> simData;
            std::vector<double> simTime;

            for (int i = 0; i < bufferSize; ++i) {
                double t = -(bufferSize - 1 - i) * timeStep;
                if (t >= x_min && t <= x_max) {
                    simData.push_back(allSimData[i]);
                    simTime.push_back(t);
                }
            }

            if (simData.size() < 2) return -1.0;

            // Resample simulation data to match reference (100 samples)
            int numRefSamples = static_cast<int>(refMean.size());  // 100
            double mse = 0.0;

            for (int i = 0; i < numRefSamples; ++i) {
                // Map reference phase [0, 1] to simulation time range [x_min, x_max]
                double phase = static_cast<double>(i) / (numRefSamples - 1);
                double targetTime = x_min + phase * (x_max - x_min);

                // Linear interpolation to get simulation value at targetTime
                double simValue = 0.0;
                bool found = false;
                for (size_t j = 0; j < simTime.size() - 1; ++j) {
                    if (targetTime >= simTime[j] && targetTime <= simTime[j + 1]) {
                        double alpha = (targetTime - simTime[j]) / (simTime[j + 1] - simTime[j]);
                        simValue = simData[j] + alpha * (simData[j + 1] - simData[j]);
                        found = true;
                        break;
                    }
                }
                // Handle edge case: targetTime at or beyond last sample
                if (!found && targetTime >= simTime.back()) {
                    simValue = simData.back();
                } else if (!found && targetTime <= simTime.front()) {
                    simValue = simData.front();
                }

                double diff = simValue - refMean[i];
                mse += diff * diff;
            }

            return mse / numRefSamples;
        };

        if (angle_selection == 0) { // Major joints
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(3, -45, 60);

            std::string title_major_joints = mPlotTitle ? mCheckpointName : "Major Joint Angles (deg)";
            float plotHeight = getPlotHeight("Kinematics");
            double plotXMin = 0.0, plotXMax = 0.0;
            if (ImPlot::BeginPlot((title_major_joints + "##MajorJoints").c_str(), ImVec2(-1, getPlotHeight("Kinematics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Angle (deg)");

                std::vector<std::string> jointKeys = {"angle_HipR", "angle_KneeR", "angle_AnkleR"};
                plotGraphData(jointKeys, ImAxis_Y1, "", stats);
                plotReferenceKinematics(jointKeys);

                // Get plot limits before EndPlot for MSE computation
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotXMin = limits.X.Min;
                plotXMax = limits.X.Max;
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

                ImPlot::EndPlot();
            }
            // MSE for major joints (using stored plot limits)
            if (getActiveKinematics()) {
                double mseHip = computeMSE("angle_HipR", plotXMin, plotXMax);
                double mseKnee = computeMSE("angle_KneeR", plotXMin, plotXMax);
                double mseAnkle = computeMSE("angle_AnkleR", plotXMin, plotXMax);

                double mseSum = mseHip + mseKnee + mseAnkle;
                ImGui::TextDisabled("ref: %s", getActiveKinematicsLabel().c_str());
                ImGui::SetWindowFontScale(2.0f);
                ImGui::Text("MSE: Hip:%.1f Knee:%.1f Ankle:%.1f (%.1f)", mseHip, mseKnee, mseAnkle, mseSum);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::SameLine();
                if (ImGui::SmallButton("Hdr##MajorMSE")) {
                    const char* hdr = "| Hash | Ckpt | Ref | Hip | Knee | Ankle | Sum |\n|------|------|-----|-----|------|-------|-----|";
                    std::cout << hdr << std::endl;
                    glfwSetClipboardString(mWindow, hdr);
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Row##MajorMSE")) {
                    char content[256];
                    snprintf(content, sizeof(content), "%s|%s|%.1f|%.1f|%.1f|%.1f",
                             mCheckpointName.c_str(), getActiveKinematicsLabel().c_str(),
                             mseHip, mseKnee, mseAnkle, mseSum);
                    std::string hash = hashTo3Char(content);
                    char buf[256];
                    snprintf(buf, sizeof(buf), "| %s | %s | %s | %.1f | %.1f | %.1f | %.1f |",
                             hash.c_str(), mCheckpointName.c_str(), getActiveKinematicsLabel().c_str(),
                             mseHip, mseKnee, mseAnkle, mseSum);
                    std::cout << buf << std::endl;
                    glfwSetClipboardString(mWindow, buf);
                }
            }
            ImGui::Separator();
        }

        if (angle_selection == 1) { // Minor joints
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(3, -10, 15);

            std::string title_minor_joints = mPlotTitle ? mCheckpointName : "Minor Joint Angles (deg)";
            double plotXMin = 0.0, plotXMax = 0.0;
            if (ImPlot::BeginPlot((title_minor_joints + "##MinorJoints").c_str(), ImVec2(-1, getPlotHeight("Kinematics"))))
            {

                ImPlot::SetupAxes("Time (s)", "Angle (deg)");

                std::vector<std::string> jointKeys = {"angle_HipIRR", "angle_HipAbR"};
                plotGraphData(jointKeys, ImAxis_Y1, "", stats);
                plotReferenceKinematics(jointKeys);

                // Get plot limits before EndPlot for MSE computation
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotXMin = limits.X.Min;
                plotXMax = limits.X.Max;
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

                ImPlot::EndPlot();
            }
            // MSE for minor joints (using stored plot limits)
            if (getActiveKinematics()) {
                double mseHipIR = computeMSE("angle_HipIRR", plotXMin, plotXMax);
                double mseHipAb = computeMSE("angle_HipAbR", plotXMin, plotXMax);
                ImGui::TextDisabled("ref: %s", getActiveKinematicsLabel().c_str());
                ImGui::SetWindowFontScale(2.0f);
                ImGui::Text("MSE: HipIR:%.1f HipAb:%.1f", mseHipIR, mseHipAb);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::SameLine();
                if (ImGui::SmallButton("Hdr##MinorMSE")) {
                    const char* hdr = "| Hash | Ckpt | Ref | HipIR | HipAb |\n|------|------|-----|-------|-------|";
                    std::cout << hdr << std::endl;
                    glfwSetClipboardString(mWindow, hdr);
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Row##MinorMSE")) {
                    char content[256];
                    snprintf(content, sizeof(content), "%s|%s|%.1f|%.1f",
                             mCheckpointName.c_str(), getActiveKinematicsLabel().c_str(),
                             mseHipIR, mseHipAb);
                    std::string hash = hashTo3Char(content);
                    char buf[256];
                    snprintf(buf, sizeof(buf), "| %s | %s | %s | %.1f | %.1f |",
                             hash.c_str(), mCheckpointName.c_str(), getActiveKinematicsLabel().c_str(),
                             mseHipIR, mseHipAb);
                    std::cout << buf << std::endl;
                    glfwSetClipboardString(mWindow, buf);
                }
            }
            ImGui::Separator();
        }
        if (angle_selection == 2) { // Pelvis joints
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(3, -20, 20);

            std::string title_pelvis_joints = mPlotTitle ? mCheckpointName : "Pelvis Angles (deg)";
            double plotXMin = 0.0, plotXMax = 0.0;
            if (ImPlot::BeginPlot((title_pelvis_joints + "##PelvisJoints").c_str(), ImVec2(-1, getPlotHeight("Kinematics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Angle (deg)");

                std::vector<std::string> pelvisKeys = {"angle_Rotation", "angle_Obliquity", "angle_Tilt"};
                plotGraphData(pelvisKeys, ImAxis_Y1, "", stats);
                plotReferenceKinematics(pelvisKeys);

                // Get plot limits before EndPlot for MSE computation
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotXMin = limits.X.Min;
                plotXMax = limits.X.Max;
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

                ImPlot::EndPlot();
            }
            // MSE for pelvis joints (using stored plot limits)
            if (getActiveKinematics()) {
                double mseRot = computeMSE("angle_Rotation", plotXMin, plotXMax);
                double mseObl = computeMSE("angle_Obliquity", plotXMin, plotXMax);
                double mseTilt = computeMSE("angle_Tilt", plotXMin, plotXMax);
                ImGui::TextDisabled("ref: %s", getActiveKinematicsLabel().c_str());
                ImGui::SetWindowFontScale(2.0f);
                ImGui::Text("MSE: Rot:%.1f Obl:%.1f Tilt:%.1f", mseRot, mseObl, mseTilt);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::SameLine();
                if (ImGui::SmallButton("Hdr##PelvisMSE")) {
                    const char* hdr = "| Hash | Ckpt | Ref | Rot | Obl | Tilt |\n|------|------|-----|-----|-----|------|";
                    std::cout << hdr << std::endl;
                    glfwSetClipboardString(mWindow, hdr);
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Row##PelvisMSE")) {
                    char content[256];
                    snprintf(content, sizeof(content), "%s|%s|%.1f|%.1f|%.1f",
                             mCheckpointName.c_str(), getActiveKinematicsLabel().c_str(),
                             mseRot, mseObl, mseTilt);
                    std::string hash = hashTo3Char(content);
                    char buf[256];
                    snprintf(buf, sizeof(buf), "| %s | %s | %s | %.1f | %.1f | %.1f |",
                             hash.c_str(), mCheckpointName.c_str(), getActiveKinematicsLabel().c_str(),
                             mseRot, mseObl, mseTilt);
                    std::cout << buf << std::endl;
                    glfwSetClipboardString(mWindow, buf);
                }
            }
        }

        if (angle_selection == 3) { // Foot sway
            static int sway_side_selection = 0; // 0=Right, 1=Left, 2=Both
            ImGuiCommon::RadioButtonGroup("SwaySide",
                {"Right", "Left", "Both"}, &sway_side_selection);

            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, -0.2, 0.2);
            ImPlot::SetNextAxisLimits(ImAxis_Y2, -60.0, 60.0, ImGuiCond_Once);

            std::string title_sway = mPlotTitle ? mCheckpointName : "Foot Sway (m)";
            if (ImPlot::BeginPlot((title_sway + "##FootSway").c_str(), ImVec2(-1, getPlotHeight("Kinematics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Sway (m)");
                ImPlot::SetupAxis(ImAxis_Y2, "out - FPA (°) - in", ImPlotAxisFlags_AuxDefault);

                std::vector<std::string> swayKeys;
                if (sway_side_selection == 0) swayKeys = {"sway_Foot_Rx", "sway_Toe_Ry"};
                else if (sway_side_selection == 1) swayKeys = {"sway_Foot_Lx", "sway_Toe_Ly"};
                else swayKeys = {"sway_Foot_Rx", "sway_Toe_Ry", "sway_Foot_Lx", "sway_Toe_Ly"};
                plotGraphData(swayKeys, ImAxis_Y1, "", stats);

                std::vector<std::string> swayKeys1;
                if (sway_side_selection == 0) swayKeys1 = {"sway_FPAr"};
                else if (sway_side_selection == 1) swayKeys1 = {"sway_FPAl"};
                else swayKeys1 = {"sway_FPAr", "sway_FPAl"};
                int colorOffset = static_cast<int>(swayKeys.size());
                plotGraphData(swayKeys1, ImAxis_Y2, "", stats, colorOffset);

                // Overlay phase bars
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

                ImPlot::EndPlot();
            }
        }

        if (angle_selection == 4) { // Anteversion
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(3, -10, 10);

            std::string title_anteversion = mPlotTitle ? mCheckpointName : "Anteversion (deg)";
            if (ImPlot::BeginPlot((title_anteversion + "##Anteversion").c_str(), ImVec2(-1, getPlotHeight("Kinematics"))))
            {
                ImPlot::SetupAxes("Time (s)", "Anteversion (deg)");

                std::vector<std::string> anteversionKeys = {"sway_AnteversionR"};
                plotGraphData(anteversionKeys, ImAxis_Y1, "", stats);

                // Overlay phase bars
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

                ImPlot::EndPlot();
            }
        }
    }
}

// ============================================================
// Kinetics Tab Content
// ============================================================
void RenderCkpt::drawKineticsTabContent()
{
    // Use FilterableChecklist for joint DOF selection
    ImGuiCommon::FilterableChecklist("##PlotJointList", mPlotJointDofNames,
                                        mPlotJointSelected, mPlotJointFilterText,
                                        sizeof(mPlotJointFilterText), 100.0f);

    // Build keys to plot
    std::vector<std::string> torqueKeysToPlot;
    for (size_t i = 0; i < mPlotJointDofNames.size(); i++) {
        if (mPlotJointSelected[i]) {
            torqueKeysToPlot.push_back("torque_" + mPlotJointDofNames[i]);
        }
    }

    ImGui::Separator();

    // Plot torques using mGraphData
    if (!torqueKeysToPlot.empty()) {
        ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);

        if (ImPlot::BeginPlot("Joint Torques##TorquePlots", ImVec2(-1, getPlotHeight("Torque Plots")))) {
            ImPlot::SetupAxes("Time (s)", "Torque (Nm)", 0, ImPlotAxisFlags_AutoFit);
            plotGraphData(torqueKeysToPlot, ImAxis_Y1);
            ImPlot::EndPlot();
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No joints selected.");
    }

    // Knee Loading
    if (collapsingHeaderWithControls("Knee Loading"))
    {
        // Display current knee loading max value
        ImGui::Text("Max Knee Loading: %.2f kN", mRenderEnv->getCharacter()->getKneeLoadingMax());

        ImGui::Separator();

        // Checkbox to toggle statistics in legend
        static bool show_knee_stats = false;
        ImGui::Checkbox("Stats##KneeLoadingStats", &show_knee_stats);

        std::string title_knee = mPlotTitle ? mCheckpointName : "Max Knee Loading";
        ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
        ImPlot::SetNextAxisLimits(3, 0, 5);
        if (ImPlot::BeginPlot((title_knee + "##KneeLoading").c_str(), ImVec2(-1, getPlotHeight("Knee Loading"))))
        {
            ImPlot::SetupAxes("Time (s)", "Knee Loading (kN)");

            // Plot max knee loading
            std::vector<std::string> kneeKeys = {"knee_loading_max"};
            plotGraphData(kneeKeys, ImAxis_Y1, "", show_knee_stats);

            ImPlot::EndPlot();
        }
    }

    // Ground Reaction Force (GRF)
    if (collapsingHeaderWithControls("Ground Reaction Force"))
    {
        // Checkbox for statistics
        static bool show_grf_stats = false;
        ImGui::Checkbox("Stats##GRFStats", &show_grf_stats);

        std::string title_grf = mPlotTitle ? mCheckpointName : "Ground Reaction Force";
        ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
        ImPlot::SetNextAxisLimits(3, 0, 3);  // 0-3x body weight typical range

        if (ImPlot::BeginPlot((title_grf + "##GRF").c_str(), ImVec2(-1, getPlotHeight("Ground Reaction Force"))))
        {
            ImPlot::SetupAxes("Time (s)", "GRF (Body Weight)");

            // Plot both feet
            std::vector<std::string> grfKeys = {"grf_left", "grf_right"};
            plotGraphData(grfKeys, ImAxis_Y1, "", show_grf_stats);

            // Optional: Overlay phase bars for gait cycle visualization
            ImPlotRect limits = ImPlot::GetPlotLimits();
            plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);

            ImPlot::EndPlot();
        }
    }

    // Joint Loading
    if (collapsingHeaderWithControls("Joint Loading"))
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
        static bool plot_torque = false;
        ImGui::Checkbox("Component", &plot_component);
        ImGui::SameLine();
        ImGui::Checkbox("Torque", &plot_torque);

        std::string selected_prefix = joint_prefixes[selected_joint];
        std::string selected_name = joint_names[selected_joint];

        // Force plot
        if (!plot_torque) {
            std::string title_force = mPlotTitle ? mCheckpointName : (selected_name + " Forces (N)");
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(3, -1, 6);
            if (ImPlot::BeginPlot((title_force + "##JointForces").c_str(), ImVec2(-1, getPlotHeight("Joint Loading"))))
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
                plotGraphData(forceKeys, ImAxis_Y1);
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);
                ImPlot::EndPlot();
            }
        } else {
            // Torque plot
            std::string title_torque = mPlotTitle ? mCheckpointName : (selected_name + " Torques (Nm)");
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);
            ImPlot::SetNextAxisLimits(3, -150, 150);
            if (ImPlot::BeginPlot((title_torque + "##JointTorques").c_str(), ImVec2(-1, getPlotHeight("Joint Loading"))))
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
                plotGraphData(torqueKeys, ImAxis_Y1);
                ImPlotRect limits = ImPlot::GetPlotLimits();
                plotPhaseBar(limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max);
                ImPlot::EndPlot();
            }
        }
    }
}

void RenderCkpt::drawMuscleTabContent()
{
    // Mode selection radio buttons
    ImGuiCommon::RadioButtonGroup("MuscleMetricMode",
        {"Activation", "Fp", "Fa*a", "Total", "Lm Norm", "F-L Curve"}, &mMuscleMetricMode);

    if (mMuscleMetricMode == 0) {
        ImGui::SameLine();
        ImGui::Checkbox("Plot NI", &mPlotActivationNoise);
    }
    
    // Use FilterableChecklist for muscle selection
    ImGuiCommon::FilterableChecklist("##PlotMuscleList", mPlotMuscleNames,
                                     mPlotMuscleSelected, mPlotMuscleFilterText,
                                     sizeof(mPlotMuscleFilterText), 100.0f);

    // Get first selected muscle index
    int firstSelectedIdx = -1;
    for (size_t i = 0; i < mPlotMuscleSelected.size(); i++) {
        if (mPlotMuscleSelected[i]) {
            firstSelectedIdx = static_cast<int>(i);
            break;
        }
    }

    ImGui::Separator();

    // Mode 5: Force-Length Curve
    if (mMuscleMetricMode == 5) {
        if (firstSelectedIdx >= 0) {
            auto muscles = mRenderEnv->getCharacter()->getMuscles();
            for (auto m : muscles) {
                if (m->name == mPlotMuscleNames[firstSelectedIdx]) {
                    ImPlot::SetNextAxisLimits(ImAxis_X1, 0, 1.5, ImGuiCond_Always);
                    ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 1000.0, ImGuiCond_Once);
                    if (ImPlot::BeginPlot((m->name + "_force_length").c_str(), ImVec2(-1, getPlotHeight("Muscle Plots"))))
                    {
                        ImPlot::SetupAxes("length", "force");
                        std::vector<std::vector<double>> p = m->GetGraphData();

                        ImPlot::PlotLine("fa##active", p[1].data(), p[2].data(), 250);
                        ImPlot::PlotLine("fa*a##active_with_activation", p[1].data(), p[3].data(), 250);
                        ImPlot::PlotLine("fp##passive", p[1].data(), p[4].data(), 250);

                        // Draw vertical line at current length (instead of infinite line)
                        double currentX = p[0][0];
                        double vlineX[] = {currentX, currentX};
                        double vlineY[] = {-1000.0, 2000.0};
                        ImPlot::PlotLine("current", vlineX, vlineY, 2);
                        ImPlot::EndPlot();
                    }
                    break;
                }
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No muscles selected.");
        }
    }
    // Modes 0-4: Time-series plots
    else {
        // Build keys with prefix based on mode
        std::string metricPrefix;
        switch (mMuscleMetricMode) {
            case 0: metricPrefix = "act_"; break;
            case 1: metricPrefix = "fp_"; break;
            case 2: metricPrefix = "fa_"; break;
            case 3: metricPrefix = "ft_"; break;
            case 4: metricPrefix = "lm_"; break;
        }
        std::vector<std::string> keysToPlot;
        for (size_t i = 0; i < mPlotMuscleNames.size(); i++) {
            if (mPlotMuscleSelected[i]) {
                keysToPlot.push_back(metricPrefix + mPlotMuscleNames[i]);
            }
        }

        // Add noise keys if enabled (only for activation mode)
        if (mMuscleMetricMode == 0 && mPlotActivationNoise) {
            auto all_keys = mGraphData->get_keys();
            for (size_t i = 0; i < mPlotMuscleNames.size(); i++) {
                if (mPlotMuscleSelected[i]) {
                    std::string noiseKey = "noise_" + mPlotMuscleNames[i];
                    if (std::find(all_keys.begin(), all_keys.end(), noiseKey) != all_keys.end()) {
                        keysToPlot.push_back(noiseKey);
                    }
                }
            }
        }

        if (!keysToPlot.empty()) {
            ImGuiCommon::SetupPlotXAxis(mXmin, -1.5);

            // Set Y-axis label and title based on mode
            std::string yAxisLabel;
            std::string plotTitle;
            switch (mMuscleMetricMode) {
                case 0:  yAxisLabel = "Activation"; plotTitle = "Muscle Activations"; break;
                case 1:  yAxisLabel = "Passive Force"; plotTitle = "Passive Force"; break;
                case 2:  yAxisLabel = "Active Force"; plotTitle = "Active Force"; break;
                case 3:  yAxisLabel = "Total Force"; plotTitle = "Total Force"; break;
                case 4:  yAxisLabel = "Muscle Length"; plotTitle = "Muscle Length"; break;
            }
            std::string title_activations = mPlotTitle ? mCheckpointName : plotTitle;
            if (ImPlot::BeginPlot((title_activations + "##MusclePlots").c_str(), ImVec2(-1, getPlotHeight("Muscle Plots"))))
            {
                ImPlot::SetupAxes("Time (s)", yAxisLabel.c_str(), 0, ImPlotAxisFlags_AutoFit);
                plotGraphData(keysToPlot, ImAxis_Y1);
                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No muscles selected.");
        }
    }

    // Activation bars for all muscles (shown in all modes)
    if (mRenderEnv->getUseMuscle())
    {
        Eigen::VectorXd activation = mRenderEnv->getCharacter()->getActivations();

        ImPlot::SetNextAxisLimits(0, -0.5, activation.rows() + 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, 0, 1);
        std::vector<double> x_act(activation.rows());
        std::vector<double> y_act(activation.rows());

        for (int i = 0; i < activation.rows(); i++)
        {
            x_act[i] = i;
            y_act[i] = activation[i];
        }
        if (ImPlot::BeginPlot("activation##bars", ImVec2(-1, 150)))
        {
            ImPlot::PlotBars("activation_level", x_act.data(), y_act.data(), activation.rows(), 1.0);
            ImPlot::EndPlot();
        }
    }
}

void RenderCkpt::drawCaptureSection() {
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Capture & Recording");
    ImGui::Separator();

    // Capture Region Section
    ImGui::Text("Capture Region");
        ImGui::SameLine();
        ImGui::Checkbox("Show##capture", &mCaptureShowRect);
        ImGui::SameLine();
        int regionWidth = mCaptureX1 - mCaptureX0;
        int regionHeight = mCaptureY1 - mCaptureY0;
        ImGui::Text("Size: %d x %d", regionWidth, regionHeight);

        // Preset combo (if presets exist)
        if (!mCapturePresets.empty()) {
            static int selectedPreset = 0;
            if (ImGui::BeginCombo("Preset##capture", mCapturePresets[selectedPreset].name.c_str())) {
                for (int i = 0; i < (int)mCapturePresets.size(); i++) {
                    if (ImGui::Selectable(mCapturePresets[i].name.c_str(), selectedPreset == i)) {
                        selectedPreset = i;
                        mCaptureX0 = mCapturePresets[i].x0;
                        mCaptureY0 = mCapturePresets[i].y0;
                        mCaptureX1 = mCapturePresets[i].x1;
                        mCaptureY1 = mCapturePresets[i].y1;
                    }
                }
                ImGui::EndCombo();
            }
        }

        // Manual region input
        const int step = 5;
        ImGui::PushItemWidth(80);
        ImGui::InputInt("x0##capture", &mCaptureX0, step);
        ImGui::SameLine();
        ImGui::InputInt("y0##capture", &mCaptureY0, step);
        ImGui::InputInt("x1##capture", &mCaptureX1, step);
        ImGui::SameLine();
        ImGui::InputInt("y1##capture", &mCaptureY1, step);
        ImGui::PopItemWidth();

        ImGui::Separator();

        // Screenshot button with timestamped filename
        if (ImGui::Button("Capture PNG")) {
            // Generate timestamped filename
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm = *std::localtime(&time_t);
            char timestamp[64];
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm);
            std::string filename = std::string("capture_") + timestamp + ".png";

            // Convert relative coords to absolute (center-based)
            int x0 = (int)(mWidth * 0.5) + mCaptureX0;
            int y0 = mCaptureY0;
            int x1 = (int)(mWidth * 0.5) + mCaptureX1;
            int y1 = mCaptureY1;

            // Clamp to window bounds
            x0 = std::max(0, std::min(x0, mWidth));
            y0 = std::max(0, std::min(y0, mHeight));
            x1 = std::max(0, std::min(x1, mWidth));
            y1 = std::max(0, std::min(y1, mHeight));

            if (captureRegionPNG(filename.c_str(), x0, y0, x1, y1)) {
                std::cout << "[Capture] Saved: capture/" << filename << std::endl;
            }
        }

        ImGui::Separator();

        // Video Recording Section
        ImGui::Text("Video Recording (30fps)");
        ImGui::SameLine();
        if (mVideoRecording) {
            if (ImGui::Button("Stop Recording")) {
                stopVideoRecording();
                mRolloutStatus.pause = true;  // Pause simulation
                // Restore camera mode
                if (mVideoOrbitEnabled) {
                    mCamera.focus = mPreRecordingFocusMode;
                }
                // Restore playback speed
                mViewerPlaybackSpeed = mPreRecordingPlaybackSpeed;
            }
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "REC");
            ImGui::SameLine();
            ImGui::Text("%.1fs  Frames: %d", mVideoElapsedTime, mVideoFrameCounter);
        } else {
            if (ImGui::Button("Start Recording")) {
                // Save current focus mode and switch to orbit if enabled
                if (mVideoOrbitEnabled) {
                    mPreRecordingFocusMode = mCamera.focus;
                    mCamera.focus = CameraFocusMode::VIDEO_ORBIT;
                    mVideoOrbitAngle = 0.0;
                }

                // Generate timestamped filename
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::tm tm = *std::localtime(&time_t);
                char timestamp[64];
                std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm);
                std::string filename = std::string("video_") + timestamp + ".mp4";

                // Start simulation (unpause)
                mRolloutStatus.pause = false;
                mRolloutStatus.cycle = -1;  // Run indefinitely

                // Force 1.0x playback speed for accurate video timing
                mPreRecordingPlaybackSpeed = mViewerPlaybackSpeed;
                mViewerPlaybackSpeed = 1.0f;

                startVideoRecording(filename, 30);
            }
            ImGui::SameLine();
            ImGui::Checkbox("Orbit##video", &mVideoOrbitEnabled);
            ImGui::SameLine();
            ImGui::PushItemWidth(60);
            ImGui::InputDouble("Max(s)##video", &mVideoMaxTime, 0, 0, "%.0f");
            ImGui::PopItemWidth();
            // Orbit speed (only show when orbit enabled)
            if (mVideoOrbitEnabled) {
                ImGui::PushItemWidth(80);
                ImGui::InputFloat("deg/s##videoorbit", &mVideoOrbitSpeed, 1.0f, 10.0f, "%.1f");
                ImGui::PopItemWidth();
            }
        }
}

void RenderCkpt::onPostRender() {
    // Record video frame if recording is active
    if (mVideoRecording) {
        recordVideoFrame();
    }

    // Draw capture region rectangle overlay if enabled
    if (mCaptureShowRect) {
        // Convert relative coords to absolute (center-based)
        int x0 = (int)(mWidth * 0.5) + mCaptureX0;
        int y0 = mCaptureY0;
        int x1 = (int)(mWidth * 0.5) + mCaptureX1;
        int y1 = mCaptureY1;

        // Clamp to window bounds
        x0 = std::max(0, std::min(x0, mWidth));
        y0 = std::max(0, std::min(y0, mHeight));
        x1 = std::max(0, std::min(x1, mWidth));
        y1 = std::max(0, std::min(y1, mHeight));

        // Draw rectangle using OpenGL
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, mWidth, mHeight, 0, -1, 1);  // Top-left origin
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(2.0f);
        glColor4f(1.0f, 0.3f, 0.3f, 1.0f);  // Red color
        glBegin(GL_LINE_LOOP);
        glVertex2i(x0, y0);
        glVertex2i(x1, y0);
        glVertex2i(x1, y1);
        glVertex2i(x0, y1);
        glEnd();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        // Draw size text using ImGui foreground draw list
        int width = x1 - x0;
        int height = y1 - y0;
        char sizeText[32];
        snprintf(sizeText, sizeof(sizeText), "%d x %d", width, height);
        ImDrawList* drawList = ImGui::GetForegroundDrawList();
        drawList->AddText(ImVec2((float)x0 + 5, (float)y0 + 5),
                          IM_COL32(255, 100, 100, 255), sizeText);
    }
}

void RenderCkpt::drawCameraStatusSection() {
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

        ImGui::Text("Eye: [%.3f, %.3f, %.3f]", mCamera.eye[0], mCamera.eye[1], mCamera.eye[2]);
        ImGui::Text("Up:  [%.3f, %.3f, %.3f]", mCamera.up[0], mCamera.up[1], mCamera.up[2]);
        ImGui::Text("RelTrans: [%.3f, %.3f, %.3f]", mCamera.relTrans[0], mCamera.relTrans[1], mCamera.relTrans[2]);
        ImGui::Text("Zoom: %.3f", mCamera.zoom);

        Eigen::Quaterniond quat = mCamera.trackball.getCurrQuat();
        ImGui::Text("Quaternion: [%.3f, %.3f, %.3f, %.3f]",
                    quat.w(), quat.x(), quat.y(), quat.z());

        // Focus Mode Selection
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Focus Mode:");
        int focusInt = static_cast<int>(mCamera.focus);
        // Only show Free and Follow; Orbit is controlled via recording checkbox
        if (focusInt > 1) focusInt = 1;  // Clamp to valid range for radio buttons
        ImGui::RadioButton("Free", &focusInt, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Follow", &focusInt, 1);
        mCamera.focus = static_cast<CameraFocusMode>(focusInt);

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

void RenderCkpt::drawJointControlSection() {
    if (ImGui::CollapsingHeader("Joint##control")) {
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

                            // Draw joint axis when slider is being dragged
                            if (ImGui::IsItemActive()) {
                                drawJointAxis(joint);
                            }

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

void RenderCkpt::printCameraInfo() {
    Eigen::Quaterniond quat = mCamera.trackball.getCurrQuat();

    std::cout << "\n======================================" << std::endl;
    std::cout << "Copy and paste below to CAMERA_PRESET_DEFINITIONS:" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "PRESET|[Add description]|"
              << mCamera.eye[0] << "," << mCamera.eye[1] << "," << mCamera.eye[2] << "|"
              << mCamera.up[0] << "," << mCamera.up[1] << "," << mCamera.up[2] << "|"
              << mCamera.relTrans[0] << "," << mCamera.relTrans[1] << "," << mCamera.relTrans[2] << "|"
              << mCamera.zoom << "|"
              << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z()
              << std::endl;
    std::cout << "======================================\n" << std::endl;
}

void RenderCkpt::initializeCameraPresets() {
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

            // Note: 'trans' in preset definitions represents mCamera.relTrans (user manual offset)
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

void RenderCkpt::runRollout() {
    mRolloutStatus.cycle = mRolloutCycles;
    mRolloutStatus.pause = false;
}


void RenderCkpt::loadCameraPreset(int index) {
    if (index < 0 || index >= 3 || !mCameraPresets[index].isSet)
    {
        LOG_WARN("[Camera] Preset " << index << " is not valid");
        return;
    }
    LOG_VERBOSE("[Camera] Loading camera preset " << index << ": " << mCameraPresets[index].description);

    mCamera.eye = mCameraPresets[index].eye;
    mCamera.up = mCameraPresets[index].up;
    mCamera.relTrans = mCameraPresets[index].trans;  // Restore user manual translation offset
    mCamera.zoom = mCameraPresets[index].zoom;
    mCamera.trackball.setQuaternion(mCameraPresets[index].quat);
    mCurrentCameraPreset = index;
}

void RenderCkpt::alignCameraToPlane(int plane) {
    // Align camera to view a specific plane
    // 1 = XY plane (view from +Z, looking at -Z, up is +Y)
    // 2 = YZ plane (view from +X, looking at -X, up is +Z)
    // 3 = ZX plane (view from +Y, looking at -Y, up is +Z)

    Eigen::Quaterniond quat;
    const char* planeName = "";

    switch (plane) {
        case 1: // XY plane - view from +Z axis
            // Looking along -Z, up is +Y → identity rotation (default OpenGL view)
            quat = Eigen::Quaterniond::Identity();
            planeName = "XY";
            break;
        case 2: // YZ plane - view from +X axis
            // Rotate -90° around Y axis to look along -X
            quat = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY());
            planeName = "YZ";
            break;
        case 3: // ZX plane - view from +Y axis (top-down)
            // Rotate +90° around X axis to look along -Y
            quat = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
            planeName = "ZX";
            break;
        default:
            LOG_WARN("[Camera] Invalid plane index: " << plane);
            return;
    }

    mCamera.trackball.setQuaternion(quat);
    LOG_INFO("[Camera] Aligned to " << planeName << " plane");
}

void RenderCkpt::drawSimControlPanelContent()
{
    if (!mRenderEnv) {
        if (ImGui::Button("Load Environment")) initEnv(mCachedMetadata);
        return;
    }

    if (ImGui::Button("Unload Environment"))
    {
        delete mRenderEnv;
        mRenderEnv = nullptr;
    }

    if (!mRenderEnv) {
        return;
    }
    
    // Rollout Control
    if (mRolloutCycles == -1) mRolloutCycles = mDefaultRolloutCount;
    ImGui::SetNextItemWidth(70);
    ImGui::InputInt("Cycles", &mRolloutCycles);
    if (mRolloutCycles < 1) mRolloutCycles = 1;

    ImGui::SameLine();

    if (ImGui::Button("Run##Rollout")) runRollout();

    // Body Mass Control
    double currentMass = mRenderEnv->getCharacter()->getSkeleton()->getMass();
    ImGui::Text("Current Mass: %.2f kg", currentMass);
    ImGui::SameLine();

    static float targetMass = 0.0f;
    if (targetMass == 0.0f) targetMass = static_cast<float>(currentMass);

    ImGui::SetNextItemWidth(100);
    ImGui::InputFloat("Target Mass (kg)", &targetMass, 1.0f, 5.0f, "%.1f");
    ImGui::SameLine();
    if (ImGui::Button("Set Mass")) mRenderEnv->getCharacter()->setBodyMass(static_cast<double>(targetMass));

    // Reward Control with TreeNode categories
    if (ImGui::CollapsingHeader("Reward##control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Metabolic Energy Category
        if (ImGui::TreeNodeEx("Metabolic Energy##metabolic", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent();

            // Get current metabolic type
            MetabolicType currentType = mRenderEnv->getCharacter()->getMetabolicType();
            constexpr int typeOffset = static_cast<int>(MetabolicType::A);
            const char* metabolicTypes[] = {"A", "A2", "MA", "MA2"};

            // Map enum to combo index, default to first entry when legacy is active
            int currentTypeInt = 0;
            if (currentType >= MetabolicType::A && currentType <= MetabolicType::MA2) {
                currentTypeInt = static_cast<int>(currentType) - typeOffset;
            }

            // Dropdown for metabolic type selection (legacy is read-only in combo)
            ImGui::SetNextItemWidth(50);
            if (ImGui::Combo("Type", &currentTypeInt, metabolicTypes, IM_ARRAYSIZE(metabolicTypes)))
            {
                // Clamp index and convert back to enum before applying
                int maxIndex = static_cast<int>(IM_ARRAYSIZE(metabolicTypes)) - 1;
                int clampedIdx = std::max(0, std::min(currentTypeInt, maxIndex));
                MetabolicType newType = static_cast<MetabolicType>(clampedIdx + typeOffset);
                mRenderEnv->getCharacter()->setMetabolicType(newType);
                currentType = newType;
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

            ImGui::Unindent();
            ImGui::TreePop();
        }

        // Imitation Reward Coefficients (deepmimic/scadiver)
        if (ImGui::TreeNodeEx("Imitation Reward##imitation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent();

            // End-effector weight
            float eeWeight = static_cast<float>(mRenderEnv->getEEWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("EE Weight", &eeWeight, 0.0f, 0.0f, "%.1f"))
            {
                mRenderEnv->setEEWeight(static_cast<double>(eeWeight));
            }

            // Position weight
            float posWeight = static_cast<float>(mRenderEnv->getPosWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Pos Weight", &posWeight, 0.0f, 0.0f, "%.1f"))
            {
                mRenderEnv->setPosWeight(static_cast<double>(posWeight));
            }

            // Velocity weight
            float velWeight = static_cast<float>(mRenderEnv->getVelWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Vel Weight", &velWeight, 0.0f, 0.0f, "%.1f"))
            {
                mRenderEnv->setVelWeight(static_cast<double>(velWeight));
            }

            // COM weight
            float comWeight = static_cast<float>(mRenderEnv->getCOMWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("COM Weight", &comWeight, 0.0f, 0.0f, "%.1f"))
            {
                mRenderEnv->setCOMWeight(static_cast<double>(comWeight));
            }

            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::Text("Exponential penalty coefficients for imitation:");
                ImGui::Text("  r = exp(-weight * error^2 / dim)");
                ImGui::Separator();
                ImGui::Text("Higher values = stricter penalty");
                ImGui::Text("EE: End-effector position (default: 40)");
                ImGui::Text("Pos: Joint positions (default: 20)");
                ImGui::Text("Vel: Joint velocities (default: 10)");
                ImGui::Text("COM: Center of mass (default: 10)");
                ImGui::EndTooltip();
            }

            ImGui::Unindent();
            ImGui::TreePop();
        }

        // Knee Pain Penalty Category
        if (ImGui::TreeNodeEx("Knee Pain Penalty##kneepain"))
        {
            ImGui::Indent();

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

            ImGui::Unindent();
            ImGui::TreePop();
        }

        // Locomotion Terms Category
        if (ImGui::TreeNodeEx("Locomotion Terms##locomotion"))
        {
            ImGui::Indent();

            float stepWeight = static_cast<float>(mRenderEnv->getStepWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Step Weight", &stepWeight, 0.0f, 0.0f, "%.3f"))
            {
                stepWeight = std::max(0.0f, stepWeight);
                mRenderEnv->setStepWeight(static_cast<double>(stepWeight));
            }

            float stepClip = static_cast<float>(mRenderEnv->getStepClip());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Step Clip", &stepClip, 0.0f, 0.0f, "%.3f"))
            {
                stepClip = std::max(0.0f, stepClip);
                mRenderEnv->setStepClip(static_cast<double>(stepClip));
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Z-axis step clipping value (default: 0.075)");
            }

            float avgVelWeight = static_cast<float>(mRenderEnv->getAvgVelWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Avg Vel Weight", &avgVelWeight, 0.0f, 0.0f, "%.3f"))
            {
                avgVelWeight = std::max(0.0f, avgVelWeight);
                mRenderEnv->setAvgVelWeight(static_cast<double>(avgVelWeight));
            }

            float avgVelClip = static_cast<float>(mRenderEnv->getAvgVelClip());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Avg Vel Clip", &avgVelClip, 0.0f, 0.0f, "%.3f"))
            {
                mRenderEnv->setAvgVelClip(static_cast<double>(avgVelClip));
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Average velocity clipping (-1 = no clipping)");
            }

            float avgWindowMult = static_cast<float>(mRenderEnv->getAvgVelWindowMult());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Avg Vel Window Mult", &avgWindowMult, 0.0f, 0.0f, "%.3f"))
            {
                avgWindowMult = std::max(0.0f, avgWindowMult);
                mRenderEnv->setAvgVelWindowMult(static_cast<double>(avgWindowMult));
            }

            bool considerX = mRenderEnv->getAvgVelConsiderX();
            if (ImGui::Checkbox("Avg Vel Consider X", &considerX))
            {
                mRenderEnv->setAvgVelConsiderX(considerX);
            }

            bool dragX = mRenderEnv->getDragX();
            if (ImGui::Checkbox("Enable Drag X", &dragX))
            {
                mRenderEnv->setDragX(dragX);
            }
            dragX = mRenderEnv->getDragX();

            bool disableDragControls = !dragX;
            if (disableDragControls)
            {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }

            float dragWeight = static_cast<float>(mRenderEnv->getDragWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Drag Weight", &dragWeight, 0.0f, 0.0f, "%.3f"))
            {
                dragWeight = std::max(0.0f, dragWeight);
                mRenderEnv->setDragWeight(static_cast<double>(dragWeight));
            }

            float dragThreshold = static_cast<float>(mRenderEnv->getDragXThreshold());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Drag Threshold", &dragThreshold, 0.0f, 0.0f, "%.3f"))
            {
                dragThreshold = std::max(0.0f, dragThreshold);
                mRenderEnv->setDragXThreshold(static_cast<double>(dragThreshold));
            }

            if (disableDragControls)
            {
                ImGui::PopStyleVar();
                ImGui::PopItemFlag();
            }

            bool phaseReward = mRenderEnv->getPhaseRewardEnabled();
            if (ImGui::Checkbox("Enable Phase Reward", &phaseReward))
            {
                mRenderEnv->setPhaseRewardEnabled(phaseReward);
            }
            phaseReward = mRenderEnv->getPhaseRewardEnabled();

            bool disablePhaseControls = !phaseReward;
            if (disablePhaseControls)
            {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }

            float phaseWeight = static_cast<float>(mRenderEnv->getPhaseWeight());
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputFloat("Phase Weight", &phaseWeight, 0.0f, 0.0f, "%.3f"))
            {
                phaseWeight = std::max(0.0f, phaseWeight);
                mRenderEnv->setPhaseWeight(static_cast<double>(phaseWeight));
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Phase reward weight (default: 1.0)");
            }

            if (disablePhaseControls)
            {
                ImGui::PopStyleVar();
                ImGui::PopItemFlag();
            }

            ImGui::Unindent();
            ImGui::TreePop();
        }
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
        if (mMotionCharacter)
            mRenderEnv->getCharacter()->updateRefSkelParam(mMotionCharacter->getSkeleton());
    }

    // Gait Phase Control
    if (mRenderEnv && ImGui::CollapsingHeader("Gait Phase##control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        auto gaitPhase = mRenderEnv->getGaitPhase();

        // Update Mode selection
        int updateMode = static_cast<int>(gaitPhase->getUpdateMode());
        ImGui::Text("Update Mode: ");
        ImGui::SameLine();
        if (ImGui::RadioButton("CONTACT", &updateMode, 0)) {
            gaitPhase->setUpdateMode(GaitPhase::CONTACT);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("PHASE", &updateMode, 1)) {
            gaitPhase->setUpdateMode(GaitPhase::PHASE);
        }

        ImGui::Separator();

        // Step Min Ratio control
        static float stepMinRatio = 0.5f;
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("Step Min Ratio", &stepMinRatio, 0.0f, 0.0f, "%.2f")) {
            stepMinRatio = std::max(0.1f, std::min(stepMinRatio, 1.0f));
            gaitPhase->setStepMinRatio(stepMinRatio);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Minimum step progression ratio for heel strike detection (0.1-1.0)");
        }

        // GRF Threshold control
        static float grfThreshold = 0.2f;
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("GRF Threshold", &grfThreshold, 0.05f, 0.5f, "%.2f")) {
            gaitPhase->setGRFThreshold(grfThreshold);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Normalized GRF threshold for contact confirmation");
        }
    }

    // Noise Injection Control Panel
    drawNoiseControlPanel();
}

void RenderCkpt::updateUnifiedKeys()
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

void RenderCkpt::updateResizablePlotsFromKeys()
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

void RenderCkpt::drawResizablePlotPane()
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
    static bool resizable_plot_show_stat = true;
    ImGui::SameLine();
    ImGui::Checkbox("Stat", &resizable_plot_show_stat);

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
            plotGraphData(mResizablePlots[i].keys, ImAxis_Y1, "", resizable_plot_show_stat);

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

void RenderCkpt::drawRenderingContent()
{
    if (!mRenderEnv) {
        ImGui::TextDisabled("No environment loaded");
        return;
    }

    if (ImGui::Button("Load Ref Motion..."))
    {
        IGFD::FileDialogConfig config;
        config.path = "data/motion";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseRefMotionDlgKey", "Choose Reference Motion File",
            ".*", config);
    }
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

                if (filePathName.find(".h5") != std::string::npos ||
                         filePathName.find(".hdf5") != std::string::npos) {
                    // Load HDF5 file (single-cycle extracted format)
                    HDF* hdf = new HDF(filePathName);

                    // Validate DOF match between skeleton and motion
                    int skelDof = mRenderEnv->getCharacter()->getSkeleton()->getNumDofs();
                    int motionDof = hdf->getValuesPerFrame();
                    if (skelDof != motionDof) {
                        LOG_WARN("[RefMotion] DOF mismatch: skeleton has " << skelDof << " DOFs, motion has " << motionDof << " DOFs. Skipping load.");
                        delete hdf;
                    } else {
                        hdf->setRefMotion(mRenderEnv->getCharacter(), mRenderEnv->getWorld());
                        newMotion = hdf;
                        LOG_INFO("[RefMotion] Loaded HDF file with " << hdf->getNumFrames() << " frames");
                    }
                }
                else if (filePathName.find(".c3d") != std::string::npos) {
                    // C3D processing moved to c3d_processor executable
                    std::cerr << "[RefMotion] C3D files not supported. Use c3d_processor executable." << std::endl;
                }
                else {
                    std::cerr << "[RefMotion] Unsupported file format: " << filePathName << std::endl;
                    std::cerr << "[RefMotion] Supported formats: .h5, .hdf5 (HDF)" << std::endl;
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

    ImGui::Checkbox("Draw PD Target Motion", &mDrawFlags.pdTarget);
    ImGui::Checkbox("Draw Ref Motion", &mDrawFlags.refMotion);
    ImGui::Checkbox("Draw Joint Sphere", &mDrawFlags.jointSphere);
    ImGui::Checkbox("Stochastic Policy", &mStochasticPolicy);
    ImGui::Checkbox("Draw Foot Step", &mDrawFlags.footStep);
    ImGui::Checkbox("Draw EOE", &mDrawFlags.eoe);
    ImGui::Checkbox("Draw Collision", &mDrawFlags.collision);

    // Skeleton Render Mode (matches RenderMode enum: Primitive, Mesh, Wireframe)
    const char* renderModes[] = {"Solid", "Mesh", "Wireframe"};
    int currentMode = static_cast<int>(mDrawFlags.skeletonRenderMode);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("Render Mode", &currentMode, renderModes, IM_ARRAYSIZE(renderModes)))
    {
        mDrawFlags.skeletonRenderMode = static_cast<RenderMode>(currentMode);
    }

    ImGui::Separator();
    // Muscle Filtering and Selection
    if (ImGui::CollapsingHeader("Muscle##Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        auto allMuscles = mRenderEnv->getCharacter()->getMuscles();
        ImGuiCommon::MuscleSelector("MuscleList", allMuscles, mMuscleSelectionStates,
                                    mMuscleFilterText, sizeof(mMuscleFilterText), 300.0f);
    }

    if (mRenderEnv->getUseMuscle()) mRenderEnv->getCharacter()->getMuscleTuple(false);

    // If no muscles are manually selected, show none (empty list)
    // The rendering code will use mSelectedMuscles if it has content
    ImGui::SetNextItemWidth(125);
    ImGui::SliderFloat("Resolution", &mMuscleResolution, 0.0, 10.0);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    ImGui::InputFloat("##MuscleResInput", &mMuscleResolution, 0.0f, 0.0f, "%.2f");
    ImGui::SetNextItemWidth(125);
    ImGui::SliderFloat("Transparency", &mMuscleTransparency, 0.1, 1.0);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    ImGui::InputFloat("##MuscleTransInput", &mMuscleTransparency, 0.0f, 0.0f, "%.2f");

    ImGui::Separator();

    ImGui::RadioButton("PassiveForce", &mMuscleRenderTypeInt, 0);
    ImGui::RadioButton("ContractileForce", &mMuscleRenderTypeInt, 1);
    ImGui::RadioButton("Activation", &mMuscleRenderTypeInt, 2);
    ImGui::RadioButton("Contracture", &mMuscleRenderTypeInt, 3);
    mMuscleRenderType = MuscleRenderingType(mMuscleRenderTypeInt);

    drawCaptureSection();
    drawCameraStatusSection();

    // === TIMING SECTION ===
    ImGui::Separator();

    ImVec4 labelColor(0.7f, 0.7f, 0.7f, 1.0f);
    ImVec4 valueColor(0.95f, 0.95f, 0.95f, 1.0f);
    ImVec4 accentColor(0.3f, 0.9f, 0.5f, 1.0f);

    // Viewer Timing
    if (ImGui::BeginTable("ViewerTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Time");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%.3f s", mViewerTime);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Phase");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(accentColor, "%.3f", mViewerPhase);

        ImGui::EndTable();
    }

    // Simulation Timing
    ImGui::Spacing();
    if (ImGui::BeginTable("SimTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Sim Time");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%.3f s", mRenderEnv->getWorld()->getTime());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Cycle Count");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%d", mRenderEnv->getGaitPhase()->getAdaptiveCycleCount());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Step Count");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%d", mRenderEnv->GetEnvironment()->getSimulationStep());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Adaptive Phase");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(accentColor, "%.3f", mRenderEnv->getGaitPhase()->getAdaptivePhase());

        ImGui::EndTable();
    }

    // Playback Controls
    ImGui::Spacing();
    ImGui::TextColored(labelColor, "Cycle Duration");
    ImGui::SetNextItemWidth(120);
    ImGui::InputDouble("##CycleDuration", &mViewerCycleDuration, 0.1, 0.5, "%.3f s");
    if (mViewerCycleDuration < 0.1) mViewerCycleDuration = 0.1;

    ImGui::Spacing();
    if(ImGui::SmallButton("0.5x")) mViewerPlaybackSpeed = 0.5;
    ImGui::SameLine();
    if(ImGui::SmallButton("1.0x")) mViewerPlaybackSpeed = 1.0;
    ImGui::SameLine();
    if(ImGui::SmallButton("2.0x")) mViewerPlaybackSpeed = 2.0;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("##PlaybackSpeed", &mViewerPlaybackSpeed, 0.1f, 2.5f, "%.2fx");

    // Performance Metrics
    ImGui::Spacing();
    if (ImGui::BeginTable("PerfTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Frame Delta");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%.1f ms", mRealDeltaTimeAvg * 1000.0);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Sim Step");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%.1f ms", mSimulationStepDuration * 1000.0);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Sim Avg");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%.1f ms", mSimStepDurationAvg * 1000.0);

        ImGui::EndTable();
    }

    // Warning
    if (mIsPlaybackTooFast)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Warning: Playback too fast!");
    }
}

void RenderCkpt::drawLeftPanel()
{
    ImGui::SetNextWindowSize(ImVec2(mControlPanelWidth, mHeight), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::Begin("Control##LeftPanel", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    if (ImGui::BeginTabBar("LeftPanelTabs")) {
        if (ImGui::BeginTabItem("Sim")) {
            drawSimControlPanelContent();

            // Bone Scale Control - uses RenderCharacter's cached scale info
            if (mMotionCharacter && ImGui::CollapsingHeader("Bone Scale"))
            {
                auto skel = mMotionCharacter->getSkeleton();
                auto& skelInfos = mMotionCharacter->getSkelInfos();

                if (skel && !skelInfos.empty())
                {
                    bool anyChanged = false;
                    for (size_t i = 0; i < skelInfos.size(); ++i)
                    {
                        auto& [boneName, modInfo] = skelInfos[i];

                        auto* bn = skel->getBodyNode(boneName);
                        if (!bn) continue;

                        // Get current shape size for display
                        Eigen::Vector3d currentSize = Eigen::Vector3d::Zero();
                        auto* shapeNode = bn->getShapeNodeWith<dart::dynamics::VisualAspect>(0);
                        if (shapeNode) {
                            const auto* boxShape = dynamic_cast<const dart::dynamics::BoxShape*>(shapeNode->getShape().get());
                            if (boxShape) {
                                currentSize = boxShape->getSize();
                            }
                        }

                        // Create sliders for scale ratios (lx, ly, lz)
                        ImGui::PushID(static_cast<int>(i));
                        if (ImGui::TreeNode(boneName.c_str()))
                        {
                            // Display current size
                            ImGui::Text("Size: [%.3f, %.3f, %.3f]", currentSize[0], currentSize[1], currentSize[2]);

                            float scaleX = static_cast<float>(modInfo.value[0]);
                            float scaleY = static_cast<float>(modInfo.value[1]);
                            float scaleZ = static_cast<float>(modInfo.value[2]);
                            float scale = static_cast<float>(modInfo.value[3]);

                            bool changed = false;
                            changed |= ImGui::SliderFloat("Scale X", &scaleX, 0.5f, 2.0f);
                            changed |= ImGui::SliderFloat("Scale Y", &scaleY, 0.5f, 2.0f);
                            changed |= ImGui::SliderFloat("Scale Z", &scaleZ, 0.5f, 2.0f);
                            changed |= ImGui::SliderFloat("Uniform", &scale, 0.5f, 2.0f);

                            if (changed)
                            {
                                modInfo.value[0] = scaleX;
                                modInfo.value[1] = scaleY;
                                modInfo.value[2] = scaleZ;
                                modInfo.value[3] = scale;
                                anyChanged = true;
                            }

                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }

                    // Apply all bone scales when any changed
                    if (anyChanged)
                    {
                        mMotionCharacter->applySkeletonBodyNode(skelInfos, skel);
                    }
                }
            }

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Kinematics")) {
            drawKinematicsControlPanelContent();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Rendering")) {
            drawRenderingContent();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void RenderCkpt::drawUI()
{
    drawLeftPanel();
    drawRightPanel();
    drawTitlePanel();
    drawResizablePlotPane();
}

void RenderCkpt::drawTimingPaneContent()
{
    // Use fixed-width font for better alignment
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);

    // Define colors for visual hierarchy
    ImVec4 headerColor(0.4f, 0.7f, 1.0f, 1.0f);      // Light blue for headers
    ImVec4 labelColor(0.7f, 0.7f, 0.7f, 1.0f);       // Gray for labels
    ImVec4 valueColor(0.95f, 0.95f, 0.95f, 1.0f);    // White for values
    ImVec4 accentColor(0.3f, 0.9f, 0.5f, 1.0f);      // Green for accents

    // Viewer Timing Section
    ImGui::PushStyleColor(ImGuiCol_Text, headerColor);
    ImGui::Text("VIEWER");
    ImGui::PopStyleColor();
    ImGui::Spacing();

    if (ImGui::BeginTable("ViewerTable", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Time");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(valueColor, "%.3f s", mViewerTime);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(labelColor, "Phase");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(accentColor, "%.3f", mViewerPhase);

        ImGui::EndTable();
    }

    // Simulation Timing Section
    if (mRenderEnv) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, headerColor);
        ImGui::Text("SIMULATION");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        if (ImGui::BeginTable("SimTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Sim Time");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(valueColor, "%.3f s", mRenderEnv->getWorld()->getTime());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Cycle Count");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(valueColor, "%d", mRenderEnv->getGaitPhase()->getAdaptiveCycleCount());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Step Count");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(valueColor, "%d", mRenderEnv->GetEnvironment()->getSimulationStep());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Adaptive Phase");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(accentColor, "%.3f", mRenderEnv->getGaitPhase()->getAdaptivePhase());

            ImGui::EndTable();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Playback Controls Section
    ImGui::PushStyleColor(ImGuiCol_Text, headerColor);
    ImGui::Text("PLAYBACK CONTROLS");
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::TextColored(labelColor, "Cycle Duration");
    ImGui::SetNextItemWidth(120);
    ImGui::InputDouble("##CycleDuration", &mViewerCycleDuration, 0.1, 0.5, "%.3f s");
    if (mViewerCycleDuration < 0.1) mViewerCycleDuration = 0.1;

    ImGui::Spacing();
    ImGui::TextColored(labelColor, "Speed Presets");

    // Styled speed preset buttons
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.4f, 0.6f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.5f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.3f, 0.5f, 1.0f));

    if(ImGui::Button("0.5x", ImVec2(50, 0))) mViewerPlaybackSpeed = 0.5;
    ImGui::SameLine();
    if(ImGui::Button("1.0x", ImVec2(50, 0))) mViewerPlaybackSpeed = 1.0;
    ImGui::SameLine();
    if(ImGui::Button("2.0x", ImVec2(50, 0))) mViewerPlaybackSpeed = 2.0;

    ImGui::PopStyleColor(3);

    ImGui::Spacing();
    ImGui::SetNextItemWidth(180);
    ImGui::SliderFloat("##PlaybackSpeed", &mViewerPlaybackSpeed, 0.1, 2.5, "%.2fx");

    // Performance Metrics Section
    if (mRenderEnv)
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, headerColor);
        ImGui::Text("PERFORMANCE");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        if (ImGui::BeginTable("PerfTable", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Frame Delta");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(valueColor, "%.1f ms", mRealDeltaTimeAvg * 1000.0);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Sim Step");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(valueColor, "%.1f ms", mSimulationStepDuration * 1000.0);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(labelColor, "Sim Avg");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(valueColor, "%.1f ms", mSimStepDurationAvg * 1000.0);

            ImGui::EndTable();
        }
    }

    // Warning Section
    if (mIsPlaybackTooFast)
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.7f, 0.0f, 1.0f));
        ImGui::Text("WARNING");
        ImGui::PopStyleColor();

        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Playback too fast!");
        if (!mRenderEnv) ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Simulation cannot keep up");
    }

    ImGui::PopFont();
}

void RenderCkpt::drawTitlePanel()
{
    if (!mShowTitlePanel) return;

    // Create a compact floating window
    ImGui::SetNextWindowSize(ImVec2(400, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 50), ImGuiCond_FirstUseEver);
    ImGui::Begin("Title Panel (Ctrl+T to toggle)", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::SetWindowFontScale(1.5f);
    ImGui::Text("%s", mCheckpointName.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::End();
}

void RenderCkpt::drawPhase(double phase, double normalized_phase)
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
    glColor3f(1.0f, 1.0f, 1.0f);
    glTranslatef(mWidth * 0.5, mHeight * 0.95, 0.0f);  // Move to top (95% of height from bottom)
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

    glColor3f(1.0f, 1.0f, 1.0f);
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

        const Eigen::Vector2i isContact = mRenderEnv->getIsContact();
        const bool isLeftLegStance = isContact[0] == 1;
        const bool isRightLegStance = isContact[1] == 1;

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
        glColor4f(1.0f, 0.0f, 0.0f, isRightLegStance ? 1.0f : 0.2f);
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

void RenderCkpt::drawPlayableMotion()
{
    // Motion pose is computed in updateViewerTime(), this function only renders
    if (mMotion == nullptr || mMotionState.currentPose.size() == 0 || mMotionState.render == false) return;
    // Draw skeleton
    drawSkeleton(mMotionState.currentPose, Eigen::Vector4d(0.8, 0.8, 0.2, 0.7));
}

bool RenderCkpt::isCurrentMotionFromSource(const std::string& sourceType, const std::string& sourceFile)
{
    Motion* motion = mMotion;
    if (!motion) return false;

    // Check source type
    if (motion->getSourceType() != sourceType)
        return false;

    // C3D check removed - C3D processing moved to c3d_processor

    // For other types, check if getName() contains the file
    return motion->getName().find(sourceFile) != std::string::npos;
}

void RenderCkpt::drawContent()
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Simulated Character
    if (mRenderEnv){
        // Draw phase using viewer time
        drawPhase(mViewerPhase, mViewerPhase);
        if (mDrawFlags.character)
        {
            drawSkeleton(mRenderEnv->getCharacter()->getSkeleton()->getPositions(), Eigen::Vector4d(1.0, 1.0, 1.0, 0.7));
            if (!mRenderConditions) drawShadow();
            if (mMuscleSelectionStates.size() > 0) drawMuscles(mMuscleRenderType);
        }

        // Draw noise visualizations
        drawNoiseVisualizations();

        if ((mRenderEnv->getRewardType() == gaitnet) && mDrawFlags.footStep) drawFootStep();
        if (mDrawFlags.jointSphere)
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
        if (!mRenderConditions && mDrawFlags.pdTarget)
        {
            const auto& character = mRenderEnv->getCharacter();
            Eigen::VectorXd pos = character->getPDTarget();
            pos.head(6) = character->getSkeleton()->getPositions().head(6);
            pos[5] += 1.0;
            drawSkeleton(pos, Eigen::Vector4d(1.0, 0.35, 0.35, 1.0));
        }
        if (!mRenderConditions && mDrawFlags.refMotion)
        {
            const auto& character = mRenderEnv->getCharacter();
            Eigen::VectorXd pos = mRenderEnv->getRefPose();
            pos[5] -= 1.0;
            drawSkeleton(pos, Eigen::Vector4d(0.35, 0.35, 1.0, 1.0));
        }
        if (mDrawFlags.eoe)
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
        if (mDrawFlags.collision) drawCollision();

        // FGN - use viewer phase for playback
        if (mDrawFlags.fgnSkeleton)
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
    if (!mRenderEnv && mDrawFlags.fgnSkeleton && !mFGN.is_none())
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


    if (mMouseDown) drawAxis();

}

void RenderCkpt::reset()
{
    mSimStepDurationAvg = -1.0;
    mViewerTime = 0.0;
    mViewerPhase = 0.0;
    mGraphData->clear_all();

    // Reset motion playback tracking for cycle accumulation
    if (mMotion) {
        mMotionState.displayOffset.setZero();
        mMotionState.displayOffset[0] = 1.0;  // Initial x offset for visualization
        mMotionState.currentPose.setZero();
        mMotionState.cycleAccumulation.setZero();
        mMotionState.lastFrameIdx = 0;
        mMotionState.manualFrameIndex = 0;
    }

    if (mRenderEnv) {
        mRenderEnv->reset(mResetPhase);
        mFGNRootOffset = mRenderEnv->getCharacter()->getSkeleton()->getRootJoint()->getPositions().tail(3);
        mUseWeights = mRenderEnv->getUseWeights();
        mViewerTime = mRenderEnv->getWorld()->getTime();
        mViewerPhase = mRenderEnv->getGaitPhase()->getAdaptiveTime() / (mRenderEnv->getMotion()->getMaxTime() / mRenderEnv->getCadence());
    }
    updateViewerTime(0);
    alignMotionToSimulation();
}

double RenderCkpt::computeFrameFloat(Motion* motion, double phase)
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

// motionPoseEval now delegates to the unified MotionProcessor
// This eliminates ~60 lines of duplicated code
void RenderCkpt::motionPoseEval(Motion* motion, int motionIdx, double frame_float)
{
    (void)motionIdx;  // Unused parameter kept for API compatibility

    if (mMotion == nullptr) {
        std::cerr << "[motionPoseEval] Warning: No motion loaded" << std::endl;
        return;
    }
    if (!mMotionCharacter) {
        std::cerr << "[motionPoseEval] Warning: No motion character loaded" << std::endl;
        return;
    }

    PlaybackViewerState& state = mMotionState;

    // Use unified MotionProcessor for pose evaluation
    state.currentPose = mMotionProcessor->evaluatePoseAtFrame(motion, frame_float, mMotionCharacter, state);

    // Update markers for C3D using processor
    if (motion->getSourceType() == "c3d") {
        int current_frame_idx = static_cast<int>(frame_float);
        state.currentMarkers = mMotionProcessor->getMarkersAtFrameWithOffsets(motion, current_frame_idx, state);
    }
}

RenderCkpt::ViewerClock RenderCkpt::updateViewerClock(double dt)
{
    ViewerClock clock;

    mViewerTime += dt;
    clock.time = mViewerTime;

    if (mViewerCycleDuration != 0.0)
        mViewerPhase = std::fmod(mViewerTime / mViewerCycleDuration, 1.0);
    else
        mViewerPhase = 0.0;

    clock.phase = mViewerPhase;
    return clock;
}

bool RenderCkpt::computeMotionPlayback(MotionPlaybackContext& context)
{
    if (mMotion == nullptr)
    {
        return false;
    }

    context.motion = mMotion;
    context.state = &mMotionState;
    // Always use mMotionCharacter (RenderCharacter) for motion playback interpolation
    // mRenderEnv->getCharacter() is only for simulation-specific operations
    context.character = mMotionCharacter;

    if (!context.motion || !context.state || !context.character)
        return false;

    context.totalFrames = context.motion->getTotalTimesteps();
    context.valuesPerFrame = context.motion->getValuesPerFrame();
    if (context.totalFrames <= 0 || context.valuesPerFrame <= 0)
        return false;

    context.phase = computeMotionPhase();
    context.frameFloat = determineMotionFrame(context.motion, *context.state, context.phase);

    context.wrappedFrameFloat = context.frameFloat;
    if (context.wrappedFrameFloat < 0.0 || context.wrappedFrameFloat >= context.totalFrames) {
        context.wrappedFrameFloat = std::fmod(context.wrappedFrameFloat, static_cast<double>(context.totalFrames));
        if (context.wrappedFrameFloat < 0.0)
            context.wrappedFrameFloat += context.totalFrames;
    }

    context.frameIndex = PlaybackUtils::computeFrameIndex(context.wrappedFrameFloat, context.totalFrames);

    return true;
}

// computeMarkerPlayback and evaluateMarkerPlayback moved to c3d_processor

void RenderCkpt::evaluateMotionPlayback(const MotionPlaybackContext& context)
{
    if (!context.motion || !context.state || !context.character)
        return;

    updateMotionCycleAccumulation(context.motion,
                                  *context.state,
                                  context.frameIndex,
                                  context.character,
                                  context.valuesPerFrame);

    motionPoseEval(context.motion, 0, context.wrappedFrameFloat);  // motionIdx parameter kept for signature compatibility
}

double RenderCkpt::computeMotionPhase()
{
    if (mRenderEnv && mRenderEnv->getMotion() && mRenderEnv->getCadence() > 0.0)
    {
        double cadence_term = mRenderEnv->getCadence() / std::sqrt(mRenderEnv->getCharacter()->getGlobalRatio());
        if (cadence_term != 0.0)
        {
            double denom = mRenderEnv->getMotion()->getMaxTime() / cadence_term;
            if (denom != 0.0)
            {
                double phase = mViewerTime / denom;
                return std::fmod(phase, 1.0);
            }
        }
    }
    return mViewerPhase;
}

double RenderCkpt::determineMotionFrame(Motion* motion, PlaybackViewerState& state, double phase)
{
    if (!motion)
        return 0.0;

    if (state.navigationMode == PLAYBACK_SYNC)
        return computeFrameFloat(motion, phase);

    int total_frames = motion->getTotalTimesteps();
    if (total_frames <= 0)
        return 0.0;

    state.manualFrameIndex = std::clamp(state.manualFrameIndex, 0, total_frames - 1);
    return static_cast<double>(state.manualFrameIndex);
}

void RenderCkpt::updateMotionCycleAccumulation(Motion* current_motion,
                                            PlaybackViewerState& state,
                                            int current_frame_idx,
                                            RenderCharacter* character,
                                            int value_per_frame)
{
    if (!current_motion || value_per_frame <= 0)
        return;

    int total_frames = current_motion->getTotalTimesteps();
    if (total_frames <= 0)
        return;

    Eigen::VectorXd raw_motion = current_motion->getRawMotionData();
    if (raw_motion.size() < static_cast<Eigen::Index>((current_frame_idx + 1) * value_per_frame))
        return;

    std::string source_type = current_motion->getSourceType();

    if (state.navigationMode != PLAYBACK_SYNC) return;

    // Standard cycle accumulation for HDF/C3D (frame-based)
    if (current_frame_idx < state.lastFrameIdx && mProgressForward) {
        state.cycleAccumulation += state.cycleDistance;
    }
    state.lastFrameIdx = current_frame_idx;
}

double RenderCkpt::computeMotionHeightCalibration(const Eigen::VectorXd& motion_pose)
{
    if (!mMotionCharacter || !mMotionCharacter->getSkeleton()) {
        return 0.0;
    }

    // Safety check: validate DOF match to prevent Eigen assertion failure
    auto skel = mMotionCharacter->getSkeleton();
    if (motion_pose.size() != skel->getNumDofs()) {
        LOG_WARN("[computeMotionHeightCalibration] DOF mismatch: motion_pose has " << motion_pose.size() << " values, skeleton has " << skel->getNumDofs() << " DOFs. Returning 0.");
        return 0.0;
    }

    // Temporarily set the motion character to the pose we want to calibrate
    Eigen::VectorXd original_pose = skel->getPositions();
    skel->setPositions(motion_pose);

    // Phase 1: Find the lowest body node Y position
    double lowest_y = std::numeric_limits<double>::max();
    for (auto bn : mMotionCharacter->getSkeleton()->getBodyNodes())
    {
        double bn_y = bn->getCOM()[1];
        if (bn_y < lowest_y) {
            lowest_y = bn_y;
        }
    }

    // Restore original pose
    mMotionCharacter->getSkeleton()->setPositions(original_pose);

    // Calculate offset to raise lowest point to ground level (y=0) with safety margin
    const double SAFETY_MARGIN = 1E-3;  // 1mm above ground
    double height_offset = -lowest_y + SAFETY_MARGIN;

    return height_offset;
}
 
void RenderCkpt::setMotion(Motion* motion)
{
    // Delete old motion and assign new one
    delete mMotion;
    mMotion = motion;

    // Initialize viewer state
    if (motion) {
        mMotionState.cycleDistance = computeMotionCycleDistance(motion);
        mMotionState.maxFrameIndex = std::max(0, motion->getNumFrames() - 1);
        mMotionState.currentPose.resize(0);
        mMotionState.displayOffset.setZero();
        mMotionState.displayOffset[0] = 1.0;
        mMotionState.manualFrameIndex = 0;
    }
}

void RenderCkpt::alignMotionToSimulation()
{
    // Silently skip if no motion loaded (valid state, not an error)
    if (mMotion == nullptr) {
        return;
    }
    
    PlaybackViewerState& state = mMotionState;
    if (state.currentPose.size() > 0) {
        state.displayOffset[1] = computeMotionHeightCalibration(state.currentPose);
    }

    // Silently skip alignment when no render environment (valid state)
    if (!mRenderEnv) {
        return;
    }
    
    // Calculate the correct frame based on current phase
    double phase = mViewerPhase;
    double frame_float = computeFrameFloat(mMotion, phase);

    // Evaluate motion pose at the current time/phase (without displayOffset)
    motionPoseEval(mMotion, 0, frame_float);  // motionIdx parameter kept for signature compatibility

    // Get simulated character's current position (from root body node)
    Eigen::Vector3d sim_pos = mRenderEnv->getCharacter()->getSkeleton()->getRootBodyNode()->getCOM();

    // Calculate displayOffset to align motion with simulation
    if (state.currentPose.size() > 0) {
        double motion_x = state.currentPose[3];
        double motion_z = state.currentPose[5];

        // Set displayOffset: shift X to separate, align Z to coincide
        state.displayOffset[0] = sim_pos[0] - motion_x + 1.0;  // X: shift apart for visualization
        state.displayOffset[2] = sim_pos[2] - motion_z;        // Z: align perfectly
    } else {
        // Fallback to default offset if pose evaluation failed
        state.displayOffset[0] = 1.0;
    }

    // C3D marker alignment moved to c3d_processor

    motionPoseEval(mMotion, 0, frame_float);  // motionIdx parameter kept for signature compatibility
}

// computeMarkerHeightCalibration and computeViewerMetric moved to c3d_processor

void RenderCkpt::updateViewerTime(double dt)
{
    ViewerClock clock = updateViewerClock(dt);

    MotionPlaybackContext motionContext;
    bool haveMotion = computeMotionPlayback(motionContext);

    // C3D marker playback moved to c3d_processor

    // Silently skip when motion context unavailable (e.g., no character loaded)
    if (!haveMotion) {
        return;
    }

    evaluateMotionPlayback(motionContext);
}

void RenderCkpt::keyPress(int key, int scancode, int action, int mods)
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
            } else if (mMotion != nullptr) {
                // Step viewer time by single frame duration when no simulation environment
                Motion* current_motion = mMotion;
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
        case GLFW_KEY_O:
            // Cycle through render modes: Primitive -> Mesh -> Wireframe -> Primitive
            mDrawFlags.skeletonRenderMode = static_cast<RenderMode>(
                (static_cast<int>(mDrawFlags.skeletonRenderMode) + 1) % 3);
            break;
        case GLFW_KEY_SPACE:
            mRolloutStatus.pause = !mRolloutStatus.pause;
            mRolloutStatus.cycle = -1;
            break;
        case GLFW_KEY_A:
            mShowResizablePlotPane = !mShowResizablePlotPane;
            break;
        case GLFW_KEY_T:
            mShowTitlePanel = !mShowTitlePanel;
            break;
        // Camera Setting
        case GLFW_KEY_C:
            printCameraInfo();
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
            break;
        case GLFW_KEY_H:
            mXminResizablePlotPane = getHeelStrikeTime();
            mXmin = mXminResizablePlotPane;
            break;
        case GLFW_KEY_0:
        case GLFW_KEY_KP_0:
            loadCameraPreset(0);
            break;
        case GLFW_KEY_1:
        case GLFW_KEY_KP_1:
            loadCameraPreset(1);
            // alignCameraToPlane(1);  // XY plane
            break;
        case GLFW_KEY_2:
        case GLFW_KEY_KP_2:
            loadCameraPreset(2);
            // alignCameraToPlane(2);  // YZ plane
            break;

        default:
            break;
        }
    }
}
void RenderCkpt::drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr)
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

void RenderCkpt::drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color)
{
    if (!mMotionCharacter) return;
    auto skel = mMotionCharacter->getSkeleton();
    skel->setPositions(pos);
    // glDepthMask(GL_FALSE);
    GUI::DrawSkeleton(skel, color, mDrawFlags.skeletonRenderMode, &mShapeRenderer);
    // glDepthMask(GL_TRUE);
}


void RenderCkpt::updateCamera()
{
    if (mRenderEnv)
    {
        if (mCamera.focus == CameraFocusMode::FOLLOW_CHARACTER)
        {
            mCamera.trans = -mRenderEnv->getCharacter()->getSkeleton()->getCOM();
            mCamera.trans[1] = -1;
        }
        else if (mCamera.focus == CameraFocusMode::VIDEO_ORBIT)
        {
            // Follow character position (same as FOLLOW_CHARACTER)
            mCamera.trans = -mRenderEnv->getCharacter()->getSkeleton()->getCOM();
            mCamera.trans[1] = -1;

            // Update orbit angle (based on ~60fps)
            double dt = 1.0 / 60.0;
            mVideoOrbitAngle += mVideoOrbitSpeed * dt;
            if (mVideoOrbitAngle >= 360.0) mVideoOrbitAngle -= 360.0;

            // Apply rotation around Y-axis
            double radians = mVideoOrbitAngle * M_PI / 180.0;
            Eigen::Quaterniond orbitQuat(Eigen::AngleAxisd(radians, Eigen::Vector3d::UnitY()));
            mCamera.trackball.setQuaternion(orbitQuat);
        }
    }
    else
    {
        // Motion-only viewing mode: focus on current motion position
        if (mMotion != nullptr && mMotionCharacter) {
            if (mMotion == nullptr) {
                mCamera.trans = Eigen::Vector3d::Zero();
                mCamera.trans[1] = -1;
                return;
            }
            // Calculate current position based on cycle accumulation
            double phase = mViewerPhase;
            Motion* current_motion = mMotion;
            PlaybackViewerState& state = mMotionState;

            double frame_float;
            if (state.navigationMode == PLAYBACK_SYNC) {
                frame_float = computeFrameFloat(current_motion, phase);
            } else {
                frame_float = static_cast<double>(state.manualFrameIndex);
            }

            int current_frame_idx = (int)frame_float;
            int total_frames = current_motion->getTotalTimesteps();
            current_frame_idx = current_frame_idx % total_frames;

            Eigen::VectorXd raw_motion = current_motion->getRawMotionData();
            int value_per_frame = current_motion->getValuesPerFrame();
            Eigen::VectorXd current_frame = raw_motion.segment(
                current_frame_idx * value_per_frame, value_per_frame);

            // HDF/C3D: motion data is already in angle format
            Eigen::VectorXd current_pos = current_frame;

            // Standard position offset handling for HDF/C3D
            mCamera.trans[0] = -(current_pos[3] + state.cycleAccumulation[0] + state.displayOffset[0]);
            mCamera.trans[1] = -(current_pos[4] + state.displayOffset[1]) - 1;
            mCamera.trans[2] = -(current_pos[5] + state.cycleAccumulation[2] + state.displayOffset[2]);
        } else {
            mCamera.trans = Eigen::Vector3d::Zero();
            mCamera.trans[1] = -1;
        }
    }
}

void RenderCkpt::drawCollision()
{
    glDisable(GL_LIGHTING);
    const auto result = mRenderEnv->getWorld()->getConstraintSolver()->getLastCollisionResult();
    for (const auto& contact : result.getContacts()) {
        Eigen::Vector3d v = contact.point;
        Eigen::Vector3d f = contact.force / 1500.0;

        // Draw arrow shaft
        glLineWidth(20.0);
        glColor3f(1.0, 0.4, 0.4);

        glBegin(GL_LINES);
        glVertex3f(v[0], v[1], v[2]);
        glVertex3f(v[0]+f[0], v[1]+f[1], v[2]+f[2]);
        glEnd();

        // Draw arrow head
        const float head_size = 0.025;

        Eigen::Vector3d dir = f.normalized();
        Eigen::Vector3d up = Eigen::Vector3d::UnitY();
        Eigen::Vector3d right = dir.cross(up).normalized();
        if(right.norm() < 0.1) right = Eigen::Vector3d::UnitX();
        Eigen::Vector3d up2 = right.cross(dir).normalized();

        Eigen::Vector3d base = v + f;
        Eigen::Vector3d tip = base + dir * head_size;

        glBegin(GL_TRIANGLES);
        glVertex3f(tip[0], tip[1], tip[2]);
        glVertex3f(base[0] + right[0]*head_size, base[1] + right[1]*head_size, base[2] + right[2]*head_size);
        glVertex3f(base[0] - right[0]*head_size, base[1] - right[1]*head_size, base[2] - right[2]*head_size);

        glVertex3f(tip[0], tip[1], tip[2]);
        glVertex3f(base[0] + up2[0]*head_size, base[1] + up2[1]*head_size, base[2] + up2[2]*head_size);
        glVertex3f(base[0] - up2[0]*head_size, base[1] - up2[1]*head_size, base[2] - up2[2]*head_size);
        glEnd();

        glPushMatrix();
        glTranslated(v[0], v[1], v[2]);
        glColor3f(0.5, 0.2, 0.5);
        GUI::DrawSphere(0.01);
        glPopMatrix();
    }

    glEnable(GL_LIGHTING);
}

void RenderCkpt::drawMuscles(MuscleRenderingType renderingType)
{
    int count = 0;
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DEPTH_TEST);

    // Check if activation noise is active
    bool activationNoiseActive = false;
    const std::vector<double>* noises = nullptr;

    if (mRenderEnv && mRenderEnv->getNoiseInjector()) {
        auto* ni = mRenderEnv->getNoiseInjector();
        activationNoiseActive = ni->isEnabled() && ni->isActivationEnabled();
        if (activationNoiseActive) {
            noises = &(ni->getVisualization().activationNoises);
        }
    }

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
            color = Eigen::Vector4d(0.2 + 1.6 * a, 0.2, 0.2, mMuscleTransparency * (1.2 * a));
            break;
        case passiveForce:
        {
            double f_p = muscle->Getf_p() / mMuscleResolution;
            color = Eigen::Vector4d(0.1, 0.1, 0.1 + 0.9 * f_p, mMuscleTransparency * (0.9 * f_p));
            break;
        }
        case contractileForce:
        {
            double f_c = muscle->GetActiveForce() / mMuscleResolution;
            color = Eigen::Vector4d(0.1, 0.1 + 0.9 * f_c, 0.1, mMuscleTransparency * (0.9 * f_c));
            break;
        }
        case contracture:
        {
            color = Eigen::Vector4d(0.05 + 10.0 * (1.0 - muscle->lmt_ref / muscle->lmt_base), 0.05, 0.05 + 10.0 * (1.0 - muscle->lmt_ref / muscle->lmt_base), mMuscleTransparency * (0.05 + 5.0 * (1.0 - muscle->lmt_ref / muscle->lmt_base)));
            break;
        }
        default:
            color.setOnes();
            break;
        }

        // Override color to green with intensity proportional to noise magnitude
        if (activationNoiseActive && noises && i < noises->size()) {
            double noise_magnitude = std::abs((*noises)[i]);  // Absolute value of noise
            double max_amp = mRenderEnv->getNoiseInjector()->getActivationAmplitude();
            double intensity = std::clamp(noise_magnitude / max_amp, 0.0, 1.0);

            // Green color: darker (0.1) to bright (1.0) based on noise intensity
            double green_val = 0.1 + 0.9 * intensity;
            color = Eigen::Vector4d(0.1, green_val, 0.1, mMuscleTransparency + 0.5);
        }

        glColor4dv(color.data());
        mShapeRenderer.renderMuscle(muscle, -1.0);
    }
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

void RenderCkpt::drawFootStep()
{
    Eigen::Vector3d current_foot = mRenderEnv->getCurrentFootStep();
    glColor4d(0.2, 0.2, 0.8, 0.5);
    glPushMatrix();
    glTranslated(0, current_foot[1], current_foot[2]);
    GUI::DrawCube(Eigen::Vector3d(0.75, 0.25, 0.15));
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

void RenderCkpt::drawNoiseControlPanel()
{
    if (!mRenderEnv) return;

    if (ImGui::CollapsingHeader("Noise Injection", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto* ni = mRenderEnv->getNoiseInjector();

        // Noise type selection with radio buttons
        ImGui::Text("Noise Type:");
        if (ImGui::RadioButton("None (Disabled)", &mNoiseMode, 0)) {}
        if (ImGui::RadioButton("Position", &mNoiseMode, 1)) {}
        if (ImGui::RadioButton("Force", &mNoiseMode, 2)) {}
        if (ImGui::RadioButton("Activation", &mNoiseMode, 3)) {}
        if (ImGui::RadioButton("All Types", &mNoiseMode, 4)) {}

        // Apply noise mode selection - enable NoiseInjector if any mode is active
        bool anyNoiseEnabled = (mNoiseMode > 0);
        ni->setEnabled(anyNoiseEnabled);
        ni->setPositionNoiseEnabled(mNoiseMode == 1 || mNoiseMode == 4);
        ni->setForceNoiseEnabled(mNoiseMode == 2 || mNoiseMode == 4);
        ni->setActivationNoiseEnabled(mNoiseMode == 3 || mNoiseMode == 4);

        ImGui::Separator();

        // Position Noise Controls
        if (ImGui::TreeNode("Position Noise")) {
            float pos_amp = static_cast<float>(ni->getPositionAmplitude());
            ImGui::SetNextItemWidth(170);
            if (ImGui::SliderFloat("Amp##PosAmp", &pos_amp, 0.0f, 0.015f, "%.4f")) {
                ni->setPositionAmplitude(pos_amp);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputFloat("##PosAmpInput", &pos_amp, 0.0f, 0.0f, "%.4f")) {
                ni->setPositionAmplitude(pos_amp);
            }

            float pos_freq = static_cast<float>(ni->getPositionFrequency());
            ImGui::SetNextItemWidth(170);
            if (ImGui::SliderFloat("Freq##PosFreq", &pos_freq, 0.1f, 1.0f, "%.2f")) {
                ni->setPositionFrequency(pos_freq);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputFloat("##PosFreqInput", &pos_freq, 0.0f, 0.0f, "%.2f")) {
                ni->setPositionFrequency(pos_freq);
            }

            ImGui::Separator();

            // Target nodes management for position
            ImGui::Text("Affected Body Nodes:");
            auto posTargetNodes = ni->getPositionTargetNodes();
            std::vector<std::string> posNodesToRemove;

            for (const auto& node : posTargetNodes) {
                ImGui::BulletText("%s", node.c_str());
                ImGui::SameLine();
                std::string buttonLabel = "Remove##Pos" + node;
                if (ImGui::SmallButton(buttonLabel.c_str())) {
                    posNodesToRemove.push_back(node);
                }
            }

            // Remove nodes
            if (!posNodesToRemove.empty()) {
                std::vector<std::string> newNodes;
                for (const auto& node : posTargetNodes) {
                    if (std::find(posNodesToRemove.begin(), posNodesToRemove.end(), node) == posNodesToRemove.end()) {
                        newNodes.push_back(node);
                    }
                }
                ni->setPositionTargetNodes(newNodes);
            }

            // Add new node with search
            ImGui::Spacing();
            static char posSearchBuffer[64] = "";
            ImGui::SetNextItemWidth(150);
            bool posEnterPressed = ImGui::InputText("##PosSearch", posSearchBuffer, sizeof(posSearchBuffer), ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::SameLine();
            ImGui::Text("(Search, Enter to add all)");

            // Show filtered body nodes
            if (strlen(posSearchBuffer) > 0) {
                auto skeleton = mRenderEnv->getCharacter()->getSkeleton();
                std::string searchTerm(posSearchBuffer);
                std::transform(searchTerm.begin(), searchTerm.end(), searchTerm.begin(), ::tolower);

                // Collect matching nodes
                std::vector<std::string> matchingNodes;
                for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i) {
                    std::string nodeName = skeleton->getBodyNode(i)->getName();
                    std::string lowerName = nodeName;
                    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

                    if (lowerName.find(searchTerm) != std::string::npos) {
                        matchingNodes.push_back(nodeName);
                    }
                }

                // If Enter pressed, add all matching nodes
                if (posEnterPressed && !matchingNodes.empty()) {
                    auto currentNodes = ni->getPositionTargetNodes();
                    int addedCount = 0;
                    for (const auto& nodeName : matchingNodes) {
                        if (std::find(currentNodes.begin(), currentNodes.end(), nodeName) == currentNodes.end()) {
                            currentNodes.push_back(nodeName);
                            addedCount++;
                        }
                    }
                    ni->setPositionTargetNodes(currentNodes);
                    LOG_INFO("[NoiseInjector] Added " << addedCount << " position target nodes");
                    posSearchBuffer[0] = '\0';  // Clear search
                } else {
                    // Show selectable list
                    ImGui::BeginChild("##PosNodeList", ImVec2(0, 150), true);
                    for (const auto& nodeName : matchingNodes) {
                        if (ImGui::Selectable(nodeName.c_str())) {
                            // Add to target nodes if not already present
                            auto currentNodes = ni->getPositionTargetNodes();
                            if (std::find(currentNodes.begin(), currentNodes.end(), nodeName) == currentNodes.end()) {
                                currentNodes.push_back(nodeName);
                                ni->setPositionTargetNodes(currentNodes);
                                LOG_INFO("[NoiseInjector] Added position target node: " << nodeName);
                            }
                            posSearchBuffer[0] = '\0';  // Clear search
                        }
                    }
                    ImGui::EndChild();
                }
            }

            ImGui::TreePop();
        }

        // Force Noise Controls
        if (ImGui::TreeNode("Force Noise")) {
            float force_amp = static_cast<float>(ni->getForceAmplitude());
            ImGui::SetNextItemWidth(170);
            if (ImGui::SliderFloat("Amp##ForceAmp", &force_amp, 0.0f, 100.0f, "%.1f")) {
                ni->setForceAmplitude(force_amp);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputFloat("##ForceAmpInput", &force_amp, 0.0f, 0.0f, "%.1f")) {
                ni->setForceAmplitude(force_amp);
            }

            float force_freq = static_cast<float>(ni->getForceFrequency());
            ImGui::SetNextItemWidth(170);
            if (ImGui::SliderFloat("Freq##ForceFreq", &force_freq, 0.1f, 5.0f, "%.2f")) {
                ni->setForceFrequency(force_freq);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputFloat("##ForceFreqInput", &force_freq, 0.0f, 0.0f, "%.2f")) {
                ni->setForceFrequency(force_freq);
            }

            ImGui::Separator();

            // Target nodes management for force
            ImGui::Text("Affected Body Nodes:");
            auto forceTargetNodes = ni->getForceTargetNodes();
            std::vector<std::string> forceNodesToRemove;

            for (const auto& node : forceTargetNodes) {
                ImGui::BulletText("%s", node.c_str());
                ImGui::SameLine();
                std::string buttonLabel = "Remove##Force" + node;
                if (ImGui::SmallButton(buttonLabel.c_str())) {
                    forceNodesToRemove.push_back(node);
                }
            }

            // Remove nodes
            if (!forceNodesToRemove.empty()) {
                std::vector<std::string> newNodes;
                for (const auto& node : forceTargetNodes) {
                    if (std::find(forceNodesToRemove.begin(), forceNodesToRemove.end(), node) == forceNodesToRemove.end()) {
                        newNodes.push_back(node);
                    }
                }
                ni->setForceTargetNodes(newNodes);
            }

            // Add new node with search
            ImGui::Spacing();
            static char forceSearchBuffer[64] = "";
            ImGui::SetNextItemWidth(150);
            bool forceEnterPressed = ImGui::InputText("##ForceSearch", forceSearchBuffer, sizeof(forceSearchBuffer), ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::SameLine();
            ImGui::Text("(Search, Enter to add all)");

            // Show filtered body nodes
            if (strlen(forceSearchBuffer) > 0) {
                auto skeleton = mRenderEnv->getCharacter()->getSkeleton();
                std::string searchTerm(forceSearchBuffer);
                std::transform(searchTerm.begin(), searchTerm.end(), searchTerm.begin(), ::tolower);

                // Collect matching nodes
                std::vector<std::string> matchingNodes;
                for (size_t i = 0; i < skeleton->getNumBodyNodes(); ++i) {
                    std::string nodeName = skeleton->getBodyNode(i)->getName();
                    std::string lowerName = nodeName;
                    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

                    if (lowerName.find(searchTerm) != std::string::npos) {
                        matchingNodes.push_back(nodeName);
                    }
                }

                // If Enter pressed, add all matching nodes
                if (forceEnterPressed && !matchingNodes.empty()) {
                    auto currentNodes = ni->getForceTargetNodes();
                    int addedCount = 0;
                    for (const auto& nodeName : matchingNodes) {
                        if (std::find(currentNodes.begin(), currentNodes.end(), nodeName) == currentNodes.end()) {
                            currentNodes.push_back(nodeName);
                            addedCount++;
                        }
                    }
                    ni->setForceTargetNodes(currentNodes);
                    LOG_INFO("[NoiseInjector] Added " << addedCount << " force target nodes");
                    forceSearchBuffer[0] = '\0';  // Clear search
                } else {
                    // Show selectable list
                    ImGui::BeginChild("##ForceNodeList", ImVec2(0, 150), true);
                    for (const auto& nodeName : matchingNodes) {
                        if (ImGui::Selectable(nodeName.c_str())) {
                            // Add to target nodes if not already present
                            auto currentNodes = ni->getForceTargetNodes();
                            if (std::find(currentNodes.begin(), currentNodes.end(), nodeName) == currentNodes.end()) {
                                currentNodes.push_back(nodeName);
                                ni->setForceTargetNodes(currentNodes);
                                LOG_INFO("[NoiseInjector] Added force target node: " << nodeName);
                            }
                            forceSearchBuffer[0] = '\0';  // Clear search
                        }
                    }
                    ImGui::EndChild();
                }
            }

            ImGui::TreePop();
        }

        // Activation Noise Controls
        if (ImGui::TreeNode("Activation Noise")) {
            float act_amp = static_cast<float>(ni->getActivationAmplitude());
            ImGui::SetNextItemWidth(170);
            if (ImGui::SliderFloat("Amp##ActAmp", &act_amp, 0.0f, 0.2f, "%.3f")) {
                ni->setActivationAmplitude(act_amp);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputFloat("##ActAmpInput", &act_amp, 0.0f, 0.0f, "%.3f")) {
                ni->setActivationAmplitude(act_amp);
            }

            float act_freq = static_cast<float>(ni->getActivationFrequency());
            ImGui::SetNextItemWidth(170);
            if (ImGui::SliderFloat("Freq##ActFreq", &act_freq, 0.0f, 1.0f, "%.2f")) {
                ni->setActivationFrequency(act_freq);
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputFloat("##ActFreqInput", &act_freq, 0.0f, 0.0f, "%.2f")) {
                ni->setActivationFrequency(act_freq);
            }
            ImGui::TreePop();
        }

        ImGui::Separator();

        // Visualization toggle
        ImGui::Checkbox("Draw Noise Arrows", &mDrawFlags.noiseArrows);
    }
}

void RenderCkpt::drawNoiseVisualizations()
{
    if (!mDrawFlags.noiseArrows || !mRenderEnv) return;

    auto* ni = mRenderEnv->getNoiseInjector();
    if (!ni || !ni->isEnabled()) return;

    const auto& viz = ni->getVisualization();

    glPushMatrix();

    // Draw force arrows (red)
    if (ni->isForceEnabled()) {
        for (const auto& [position, force] : viz.forceArrows) {
            Eigen::Vector3d direction = force.normalized();
            double magnitude = force.norm();
            double length = magnitude / 50.0;  // Scale for visibility

            // Red arrows for external forces
            Eigen::Vector4d color(1.0, 0.0, 0.0, 0.8);
            GUI::DrawArrow3D(position, direction, length, 0.01, color);
        }
    }

    // Draw position noise indicators (blue arrows at joints)
    if (ni->isPositionEnabled()) {
        auto skeleton = mRenderEnv->getCharacter()->getSkeleton();
        for (const auto& [nodeName, offset] : viz.positionNoises) {
            // Draw at the actual body node position
            auto* bn = skeleton->getBodyNode(nodeName);
            if (bn && offset.norm() > 1e-6) {
                Eigen::Vector4d color(0.0, 0.0, 1.0, 0.8);
                Eigen::Vector3d nodePos = bn->getWorldTransform().translation();
                GUI::DrawArrow3D(nodePos, offset.normalized(), offset.norm() * 250.0, 0.008, color);
            }
        }
    }

    glPopMatrix();
}

void RenderCkpt::drawShadow()
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

void RenderCkpt::loadNetworkFromPath(const std::string& path)
{
    if (loading_network.is_none()) {
        std::cerr << "Warning: loading_network not available, skipping: " << path << std::endl;
        return;
    }

    try {
        auto character = mRenderEnv->getCharacter();
        Network new_elem;
        new_elem.name = path;

        bool use_muscle = (character->getActuatorType() == mass || character->getActuatorType() == mass_lower);

        // Prepare arguments for loading_network
        py::tuple res;
        if (use_muscle) {
            // Pass muscle dimensions for CleanRL checkpoint compatibility
            int num_muscles = character->getNumMuscles();
            int num_muscle_dofs = character->getNumMuscleRelatedDof();
            int num_actuator_action = mRenderEnv->getNumActuatorAction();

            res = loading_network(
                path.c_str(),
                mRenderEnv->getState().rows(),
                mRenderEnv->getAction().rows(),
                use_muscle,
                "cpu",  // device
                num_muscles,
                num_muscle_dofs,
                num_actuator_action
            );
        } else {
            // No muscle network needed
            res = loading_network(
                path.c_str(),
                mRenderEnv->getState().rows(),
                mRenderEnv->getAction().rows(),
                use_muscle
            );
        }

        new_elem.joint = res[0];

        // Convert Python muscle state_dict to C++ MuscleNN
        if (use_muscle && !res[1].is_none()) {
            int num_muscles = character->getNumMuscles();
            int num_muscle_dofs = character->getNumMuscleRelatedDof();
            int num_actuator_action = mRenderEnv->getNumActuatorAction();
            bool is_cascaded = false;  // TODO: detect from network structure if needed

            // Create C++ MuscleNN
            // Force CPU to avoid CUDA context allocation issues in multi-process scenarios
            new_elem.muscle = make_muscle_nn(num_muscle_dofs, num_actuator_action, num_muscles, is_cascaded, true);
            py::dict state_dict = res[1].cast<py::dict>();

            // Store the Python state_dict for transfer to Environment
            mMuscleStateDict = res[1];

            // Convert Python state_dict to C++ format
            std::unordered_map<std::string, torch::Tensor> cpp_state_dict;
            for (auto item : state_dict) {
                std::string key = item.first.cast<std::string>();
                py::array_t<float> np_array = item.second.cast<py::array_t<float>>();

                auto buf = np_array.request();
                std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());

                torch::Tensor tensor = torch::from_blob(
                    buf.ptr,
                    shape,
                    torch::TensorOptions().dtype(torch::kFloat32)
                ).clone();

                cpp_state_dict[key] = tensor;
            }

            new_elem.muscle->load_state_dict(cpp_state_dict);
        }

        mNetworks.push_back(new_elem);
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading network from " << path << ": " << e.what());
    }
}

void RenderCkpt::initializeMotionSkeleton()
{
    // Now uses mMotionCharacter->getSkeleton() instead of creating mMotionSkeleton
    if (!mMotionCharacter) {
        LOG_WARN("[initializeMotionSkeleton] mMotionCharacter not initialized");
        return;
    }
    auto skel = mMotionCharacter->getSkeleton();

    // Setup BVH joint calibration
    mJointCalibration.clear();
    for (auto jn : skel->getJoints()) {
        if (jn == skel->getRootJoint()) {
            mJointCalibration.push_back(Eigen::Matrix3d::Identity());
        } else {
            mJointCalibration.push_back(
                (jn->getTransformFromParentBodyNode() * jn->getParentBodyNode()->getTransform()).linear().transpose()
            );
        }
    }

    // Setup skeleton info for motions
    mSkelInfosForMotions.clear();
    for (auto bn : skel->getBodyNodes()) {
        ModifyInfo skelInfo;
        mSkelInfosForMotions.push_back(std::make_pair(bn->getName(), skelInfo));
    }
}

// REMOVED: loadNPZMotion(), loadHDFRolloutMotion(), loadBVHMotion(), loadHDFMotion(), loadMotionFiles()
// Motion loading is now on-demand via scanMotionFiles() + loadMotionFile()

void RenderCkpt::scanMotionFiles()
{
    mMotionList.clear();
    namespace fs = std::filesystem;

    // Scan data/motion/ for HDF files (.h5, .hdf5)
    std::string hdf_path = "data/motion";
    if (fs::exists(hdf_path) && fs::is_directory(hdf_path)) {
        for (const auto& entry : fs::directory_iterator(hdf_path)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (ext == ".h5" || ext == ".hdf5") {
                mMotionList.push_back(entry.path().string());
            }
        }
    }

    // C3D scanning moved to c3d_processor

    LOG_INFO("[Motion] Scanned " << mMotionList.size() << " motion files (HDF only, C3D moved to c3d_processor)");
}

void RenderCkpt::onPIDFileSelected(const std::string& path,
                                 const std::string& filename)
{
    // Load the HDF motion file using existing unified loader
    loadMotionFile(path);

    // Cache reference kinematics from the loaded motion if available
    HDF* hdf = dynamic_cast<HDF*>(mMotion);
    if (hdf && hdf->hasReferenceKinematics()) {
        mReferenceKinematics = hdf->getReferenceKinematics();
        mHasReferenceKinematics = true;
        LOG_INFO("[RenderCkpt] Loaded reference kinematics for "
                 << mReferenceKinematics.jointKeys.size() << " joints");
    } else {
        mHasReferenceKinematics = false;
    }

    LOG_INFO("[RenderCkpt] Loaded PID HDF file: " << filename);
}

void RenderCkpt::loadMotionFile(const std::string& path)
{
    namespace fs = std::filesystem;
    std::string ext = fs::path(path).extension().string();
    Motion* motion = nullptr;
    LOG_INFO("[Motion] Loading motion file: " << path);

    if (ext == ".c3d") {
        // C3D processing moved to c3d_processor executable
        LOG_ERROR("[Motion] C3D files not supported in viewer. Use c3d_processor executable.");
        return;
    } else if (ext == ".h5" || ext == ".hdf5") {
        motion = new HDF(path);
    } else {
        LOG_ERROR("[Motion] Unknown file extension: " << ext);
        return;
    }

    if (motion) {
        // Validate DOF match between skeleton and motion
        if (!mMotionCharacter || !mMotionCharacter->getSkeleton()) {
            LOG_WARN("[Motion] No motion character loaded. Skipping motion load.");
            delete motion;
            return;
        }

        int skelDof = mMotionCharacter->getSkeleton()->getNumDofs();
        int motionDof = motion->getValuesPerFrame();
        if (skelDof != motionDof) {
            LOG_WARN("[Motion] DOF mismatch: skeleton has " << skelDof << " DOFs, motion has " << motionDof << " DOFs. Skipping load.");
            delete motion;
            return;
        }

        setMotion(motion);
        mMotionPath = path;  // Store path for reloading
        mMotionState.cycleDistance = computeMotionCycleDistance(motion);
        mMotionState.maxFrameIndex = std::max(0, motion->getNumFrames() - 1);
        updateViewerTime(0);  // Initialize motion context
        alignMotionToSimulation();
        LOG_INFO("[Motion] Loaded: " << path);
    } else {
        LOG_ERROR("[Motion] Failed to load: " << path);
    }
}

// =============================================================================
// Clinical Data (PID-based HDF access) Methods
// =============================================================================

void RenderCkpt::drawClinicalDataSection()
{
    // Render PID Navigator UI with collapsing header
    if (mPIDNavigator) {
        mPIDNavigator->renderUI("Clinical Data", 150.0f, 150.0f, false);
    } else {
        if (ImGui::CollapsingHeader("Clinical Data", 0)) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                             "PID Navigator not initialized");
        }
    }
}

// REMOVED: scanHDF5Structure() - HDF5 rollout format no longer used

void RenderCkpt::loadParametersFromCurrentMotion()
{
    if (!mRenderEnv) {
        LOG_ERROR("Render environment not initialized");
        return;
    }

    Motion* motion = mMotion;

    if (!motion) {
        LOG_ERROR("No motion loaded");
        return;
    }

    if (!motion->hasParameters()) {
        LOG_WARN("Current motion (" + motion->getName() + ") has no parameters");
        return;
    }

    LOG_INFO("Loading parameters from motion: " << motion->getName());

    try {
        // Get parameters from motion
        std::vector<std::string> hdf5_param_names = motion->getParameterNames();
        std::vector<float> hdf5_param_values = motion->getParameterValues();

        if (hdf5_param_names.size() != hdf5_param_values.size()) {
            LOG_WARN("Parameter names count (" + std::to_string(hdf5_param_names.size())
                      << ") != values count (" + std::to_string(hdf5_param_values.size()) + ")");
            return;
        }

        // Get simulation parameters
        const std::vector<std::string>& sim_param_names = mRenderEnv->getParamName();
        Eigen::VectorXd current_params = mRenderEnv->getParamState();

        LOG_INFO("  Motion has " + std::to_string(hdf5_param_names.size()) + " parameters");
        LOG_INFO("  Simulation has " + std::to_string(sim_param_names.size()) + " parameters");

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

        LOG_INFO("  Matched " + std::to_string(matched_count) + " parameters by name");

        // Apply parameters to simulation environment
        mRenderEnv->setParamState(new_params);

        LOG_INFO("✓ Successfully loaded parameters from " + motion->getSourceType()
                  + " motion: " + motion->getName());

    } catch (const std::exception& e) {
        LOG_ERROR("Error loading parameters from motion: " + std::string(e.what()));
    }
}

void RenderCkpt::unloadMotion()
{
    // Delete single motion
    delete mMotion;
    mMotion = nullptr;
    mMotionState = PlaybackViewerState();  // Reset to default
    mSelectedMotion = -1;  // Reset selection

    // Reset motion context
    updateViewerTime(0);

    // Reset simulation parameters to default from XML metadata
    if (mRenderEnv && mMotionCharacter) {
        Eigen::VectorXd default_params = mRenderEnv->getParamDefault();
        mRenderEnv->setParamState(default_params);
        mRenderEnv->getCharacter()->updateRefSkelParam(mMotionCharacter->getSkeleton());
        LOG_INFO("[Motion] Motion unloaded, parameters reset to defaults");
    }
}

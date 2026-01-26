/**
 * @file c3d_calibration_cli_main.cpp
 * @brief CLI tool for automated skeleton calibration from C3D motion capture data
 *
 * This tool automates the skeleton retrieval pipeline that is done manually
 * through the GUI in C3DProcessorApp. It enables agents and scripts to
 * programmatically extract calibrated skeletons from C3D files.
 *
 * Modes:
 *   static  - Static calibration only (from medial marker trial)
 *   dynamic - Dynamic calibration + HDF export (requires existing calibration)
 *   full    - Static then dynamic (complete pipeline)
 *
 * Usage:
 *   # Process all PIDs, all visits, all motions
 *   c3d_calibration_cli --mode full -v
 *
 *   # Single patient, all visits
 *   c3d_calibration_cli --mode full --pid 12964246 -v
 *
 *   # Specific motion for a patient
 *   c3d_calibration_cli --mode dynamic --pid 12964246 --visit pre --motion Walk01 -v
 */

#include "C3D_Reader.h"
#include "RenderCharacter.h"
#include "C3D.h"
#include "DARTHelper.h"
#include "rm/global.hpp"
#include "Log.h"

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <regex>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// ============================================================================
// Default Paths (matching C3DProcessorApp)
// ============================================================================
const std::string DEFAULT_SKELETON = "@data/skeleton/base.yaml";
const std::string DEFAULT_MARKER = "@data/marker/static.xml";
const std::string DEFAULT_DYNAMIC_CONFIG = "@data/config/skeleton_fitting.yaml";
const std::string DEFAULT_STATIC_CONFIG = "@data/config/static_fitting.yaml";

// ============================================================================
// Progress Bar
// ============================================================================
class ProgressBar {
public:
    ProgressBar(int total, bool enabled = true)
        : mTotal(total), mCurrent(0), mEnabled(enabled) {}

    void update(int current, const std::string& pid = "",
                const std::string& visit = "", const std::string& motion = "") {
        if (!mEnabled) return;
        mCurrent = current;
        int percent = mTotal > 0 ? (100 * mCurrent) / mTotal : 0;
        int filled = mTotal > 0 ? (20 * mCurrent) / mTotal : 0;

        std::cout << "\rProcessing: [";
        for (int i = 0; i < 20; ++i) {
            std::cout << (i < filled ? "=" : " ");
        }
        std::cout << "] " << std::setw(3) << percent << "% "
                  << "(" << mCurrent << "/" << mTotal << ")";
        if (!pid.empty()) {
            std::cout << " PID:" << pid;
            if (!visit.empty()) std::cout << " visit:" << visit;
            if (!motion.empty()) std::cout << " motion:" << motion;
        }
        std::cout << "     " << std::flush;
    }

    void finish() {
        if (mEnabled) std::cout << std::endl;
    }

private:
    int mTotal;
    int mCurrent;
    bool mEnabled;
};

// ============================================================================
// Auto-Discovery Functions
// ============================================================================

/**
 * Discover all available PIDs from the resource manager
 */
std::vector<std::string> discoverPIDs(rm::ResourceManager& mgr) {
    std::vector<std::string> pids;
    try {
        auto items = mgr.list("@pid");
        for (const auto& item : items) {
            // Filter to numeric PIDs only
            if (!item.empty() && std::all_of(item.begin(), item.end(), ::isdigit)) {
                pids.push_back(item);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[Discovery] Failed to list PIDs: " << e.what());
    }
    std::sort(pids.begin(), pids.end());
    return pids;
}

/**
 * Discover available visits for a PID
 */
std::vector<std::string> discoverVisits(rm::ResourceManager& mgr, const std::string& pid) {
    std::vector<std::string> visits;
    for (const auto& v : {"pre", "op1", "op2"}) {
        std::string pattern = "@pid:" + pid + "/" + v;
        if (mgr.exists(pattern)) {
            visits.push_back(v);
        }
    }
    return visits;
}

/**
 * Find static C3D file in gait directory
 */
std::string findStaticC3D(const std::string& gaitDir) {
    // Try common patterns
    std::vector<std::string> patterns = {
        "Static_edit-Static.c3d",
        "Static_edit.c3d",
        "Static.c3d"
    };

    for (const auto& pattern : patterns) {
        std::string path = gaitDir + "/" + pattern;
        if (fs::exists(path)) {
            return path;
        }
    }

    // Fall back to glob for *Static*.c3d
    for (const auto& entry : fs::directory_iterator(gaitDir)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (name.find("Static") != std::string::npos &&
                entry.path().extension() == ".c3d") {
                return entry.path().string();
            }
        }
    }

    return "";  // Not found
}

/**
 * Discover motion C3D files (non-static) in gait directory
 */
std::vector<std::string> discoverMotions(const std::string& gaitDir,
                                         const std::string& motionFilter = "") {
    std::vector<std::string> motions;

    if (!fs::exists(gaitDir)) return motions;

    for (const auto& entry : fs::directory_iterator(gaitDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".c3d") continue;

        std::string stem = entry.path().stem().string();

        // Skip static files
        if (stem.find("Static") != std::string::npos) continue;

        // Apply filter if specified
        if (!motionFilter.empty() && stem.find(motionFilter) == std::string::npos) {
            continue;
        }

        motions.push_back(stem);
    }

    std::sort(motions.begin(), motions.end());
    return motions;
}

// ============================================================================
// HDF5 Export (extracted from C3DProcessorApp::exportMotionToHDF5)
// ============================================================================

void exportMotionHDF5(
    const DynamicCalibrationResult& result,
    const std::string& outputPath,
    const std::string& pid,
    const std::string& visit,
    const std::string& c3dFilename,
    int frameRate,
    C3D* c3dMotion,
    RenderCharacter* freeCharacter,
    RenderCharacter* motionCharacter,
    const SkeletonFittingConfig& config)
{
    try {
        H5::H5File file(outputPath, H5F_ACC_TRUNC);

        const auto& poses = result.motionPoses;
        int numFrames = static_cast<int>(poses.size());
        int dofPerFrame = poses.empty() ? 56 : static_cast<int>(poses[0].size());

        // 1. Write /motions (numFrames x DOF)
        hsize_t dims_motions[2] = {(hsize_t)numFrames, (hsize_t)dofPerFrame};
        H5::DataSpace space_motions(2, dims_motions);
        H5::DataSet ds_motions = file.createDataSet("/motions", H5::PredType::NATIVE_FLOAT, space_motions);

        std::vector<float> motionBuffer(numFrames * dofPerFrame);
        for (int f = 0; f < numFrames; ++f) {
            for (int d = 0; d < dofPerFrame; ++d) {
                motionBuffer[f * dofPerFrame + d] = static_cast<float>(poses[f][d]);
            }
        }
        ds_motions.write(motionBuffer.data(), H5::PredType::NATIVE_FLOAT);

        // 2. Write /phase (normalized 0-1)
        hsize_t dims_1d[1] = {(hsize_t)numFrames};
        H5::DataSpace space_1d(1, dims_1d);
        H5::DataSet ds_phase = file.createDataSet("/phase", H5::PredType::NATIVE_FLOAT, space_1d);

        std::vector<float> phaseBuffer(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            phaseBuffer[f] = static_cast<float>(f) / static_cast<float>(numFrames - 1);
        }
        ds_phase.write(phaseBuffer.data(), H5::PredType::NATIVE_FLOAT);

        // 3. Write /time
        H5::DataSet ds_time = file.createDataSet("/time", H5::PredType::NATIVE_FLOAT, space_1d);
        double dt = 1.0 / frameRate;
        std::vector<float> timeBuffer(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            timeBuffer[f] = static_cast<float>(f * dt);
        }
        ds_time.write(timeBuffer.data(), H5::PredType::NATIVE_FLOAT);

        // 4. Write metadata as HDF5 attributes
        H5::Group root = file.openGroup("/");
        H5::DataSpace scalarSpace(H5S_SCALAR);

        auto writeStrAttr = [&](const char* name, const std::string& value) {
            H5::StrType strType(H5::PredType::C_S1, value.size() + 1);
            H5::Attribute attr = root.createAttribute(name, strType, scalarSpace);
            attr.write(strType, value.c_str());
        };

        auto writeIntAttr = [&](const char* name, int value) {
            H5::Attribute attr = root.createAttribute(name, H5::PredType::NATIVE_INT, scalarSpace);
            attr.write(H5::PredType::NATIVE_INT, &value);
        };

        writeStrAttr("source_type", "c3d_dynamic_calibration");
        writeStrAttr("c3d_file", c3dFilename);
        writeIntAttr("frame_rate", frameRate);
        writeIntAttr("num_frames", numFrames);
        writeIntAttr("dof_per_frame", dofPerFrame);
        writeStrAttr("pid", pid);
        writeStrAttr("visit", visit);

        // 5. Write marker tracking errors
        const auto& mappings = config.markerMappings;
        int numMarkers = static_cast<int>(mappings.size());

        if (numMarkers > 0 && c3dMotion && freeCharacter && motionCharacter) {
            H5::Group errorGroup = file.createGroup("/marker_error");

            auto writeCharacterErrors = [&](RenderCharacter* character,
                                           const std::vector<Eigen::VectorXd>& charPoses,
                                           const std::string& groupName) {
                if (!character || charPoses.empty()) return;

                H5::Group charGroup = errorGroup.createGroup(groupName);
                auto skel = character->getSkeleton();

                std::vector<float> errorBuffer(numFrames * numMarkers, 0.0f);
                std::vector<float> meanBuffer(numFrames, 0.0f);

                for (int f = 0; f < numFrames; ++f) {
                    skel->setPositions(charPoses[f]);
                    auto expectedMarkers = character->getExpectedMarkerPositions();
                    const auto& c3dMarkers = c3dMotion->getMarkers(f);
                    const auto& skelMarkers = character->getMarkers();

                    double frameErrorSum = 0.0;
                    int validCount = 0;

                    for (int m = 0; m < numMarkers; ++m) {
                        const auto& mapping = mappings[m];
                        int dataIdx = mapping.dataIndex;

                        int skelIdx = -1;
                        for (size_t j = 0; j < skelMarkers.size(); ++j) {
                            if (skelMarkers[j].name == mapping.name) {
                                skelIdx = static_cast<int>(j);
                                break;
                            }
                        }

                        bool hasData = dataIdx >= 0 && dataIdx < (int)c3dMarkers.size();
                        bool hasSkel = skelIdx >= 0 && skelIdx < (int)expectedMarkers.size();

                        if (hasData && hasSkel) {
                            Eigen::Vector3d dataM = c3dMarkers[dataIdx];
                            Eigen::Vector3d skelM = expectedMarkers[skelIdx];
                            if (dataM.array().isFinite().all() && skelM.array().isFinite().all()) {
                                double err = (dataM - skelM).norm() * 1000.0;
                                errorBuffer[f * numMarkers + m] = static_cast<float>(err);
                                frameErrorSum += err;
                                validCount++;
                            }
                        }
                    }
                    meanBuffer[f] = validCount > 0 ? static_cast<float>(frameErrorSum / validCount) : 0.0f;
                }

                hsize_t dims_error[2] = {(hsize_t)numFrames, (hsize_t)numMarkers};
                H5::DataSpace space_error(2, dims_error);
                H5::DataSet ds_data = charGroup.createDataSet("data", H5::PredType::NATIVE_FLOAT, space_error);
                ds_data.write(errorBuffer.data(), H5::PredType::NATIVE_FLOAT);

                H5::DataSet ds_mean = charGroup.createDataSet("mean", H5::PredType::NATIVE_FLOAT, space_1d);
                ds_mean.write(meanBuffer.data(), H5::PredType::NATIVE_FLOAT);
            };

            writeCharacterErrors(freeCharacter, result.freePoses, "free");
            writeCharacterErrors(motionCharacter, result.motionPoses, "motion");

            std::string namesStr;
            for (size_t i = 0; i < mappings.size(); ++i) {
                if (i > 0) namesStr += ",";
                namesStr += mappings[i].name;
            }
            writeStrAttr("marker_names", namesStr);
            writeIntAttr("num_markers", numMarkers);
        }

        file.close();
        LOG_INFO("[Export] HDF5 written: " << outputPath);

    } catch (const H5::Exception& e) {
        LOG_ERROR("[Export] HDF5 error: " << e.getDetailMsg());
        throw;
    }
}

// ============================================================================
// Calibration Functions
// ============================================================================

/**
 * Run static calibration (from medial marker trial)
 * Returns 0 on success, non-zero on error
 */
int runStaticCalibration(
    C3D_Reader* reader,
    RenderCharacter* freeCharacter,
    const std::string& staticC3DPath,
    const std::string& staticConfigPath,
    const std::string& outputDir,
    bool verbose)
{
    if (verbose) {
        LOG_INFO("[Static] Loading C3D: " << staticC3DPath);
    }

    // Load static C3D markers only
    C3D* c3dData = reader->loadC3DMarkersOnly(staticC3DPath);
    if (!c3dData) {
        LOG_ERROR("[Static] Failed to load C3D file: " << staticC3DPath);
        return 1;
    }

    // Check for medial markers
    if (!C3D_Reader::hasMedialMarkers(c3dData->getLabels())) {
        LOG_ERROR("[Static] No medial markers found in: " << staticC3DPath);
        LOG_ERROR("[Static] Static calibration requires L/R.Knee.Medial and L/R.Ankle.Medial markers");
        delete c3dData;
        return 1;
    }

    if (verbose) {
        LOG_INFO("[Static] Medial markers detected, running calibration...");
    }

    // Run static calibration
    StaticCalibrationResult result = reader->calibrateStatic(c3dData, staticConfigPath);
    if (!result.success) {
        LOG_ERROR("[Static] Calibration failed: " << result.errorMessage);
        delete c3dData;
        return 1;
    }

    if (verbose) {
        LOG_INFO("[Static] Calibration succeeded");
        LOG_INFO("[Static] Bone scales: " << result.boneScales.size());
        LOG_INFO("[Static] Personalized offsets: " << result.personalizedOffsets.size());
    }

    // Apply personalized offsets to freeCharacter
    if (freeCharacter) {
        auto& markers = freeCharacter->getMarkersForEdit();
        for (auto& marker : markers) {
            auto it = result.personalizedOffsets.find(marker.name);
            if (it != result.personalizedOffsets.end()) {
                marker.offset = it->second;
            }
        }
    }

    // Create output directory
    std::string calibDir = outputDir + "/calibration";
    if (!fs::exists(calibDir)) {
        fs::create_directories(calibDir);
    }

    // Export calibration files
    reader->exportPersonalizedCalibration(result, calibDir);

    if (verbose) {
        LOG_INFO("[Static] Exported to: " << calibDir);
    }

    delete c3dData;
    return 0;
}

/**
 * Run dynamic calibration (motion tracking + export)
 * Returns 0 on success, non-zero on error
 */
int runDynamicCalibration(
    C3D_Reader* reader,
    RenderCharacter* freeCharacter,
    RenderCharacter* motionCharacter,
    const std::string& motionC3DPath,
    const std::string& motionOutputDir,
    const std::string& skeletonOutputDir,
    const std::string& skeletonName,
    const std::string& pid,
    const std::string& visit,
    bool verbose)
{
    if (verbose) {
        LOG_INFO("[Dynamic] Loading C3D: " << motionC3DPath);
    }

    // Load motion C3D
    C3D* c3dData = reader->loadC3DMarkersOnly(motionC3DPath);
    if (!c3dData) {
        LOG_ERROR("[Dynamic] Failed to load C3D file: " << motionC3DPath);
        return 1;
    }

    // Run dynamic calibration
    DynamicCalibrationResult result = reader->calibrateDynamic(c3dData);
    if (!result.success) {
        LOG_ERROR("[Dynamic] Calibration failed: " << result.errorMessage);
        delete c3dData;
        return 1;
    }

    if (verbose) {
        LOG_INFO("[Dynamic] Calibration succeeded: " << result.motionPoses.size() << " frames");
    }

    // Create output directories
    if (!fs::exists(motionOutputDir)) {
        fs::create_directories(motionOutputDir);
    }
    if (!fs::exists(skeletonOutputDir)) {
        fs::create_directories(skeletonOutputDir);
    }

    // Export HDF5
    std::string hdfPath = motionOutputDir + "/" + skeletonName + ".h5";
    std::string c3dFilename = fs::path(motionC3DPath).filename().string();

    exportMotionHDF5(
        result,
        hdfPath,
        pid,
        visit,
        c3dFilename,
        reader->getFrameRate(),
        c3dData,
        freeCharacter,
        motionCharacter,
        reader->getFittingConfig()
    );

    // Export skeleton YAML
    std::string skelPath = skeletonOutputDir + "/" + skeletonName + ".yaml";
    if (motionCharacter) {
        motionCharacter->exportSkeletonYAML(skelPath);
        if (verbose) {
            LOG_INFO("[Dynamic] Skeleton exported: " << skelPath);
        }
    }

    delete c3dData;
    return 0;
}

/**
 * Check if calibration files exist for a PID/visit
 */
bool hasCalibrationFiles(rm::ResourceManager& mgr, const std::string& pid, const std::string& visit) {
    std::string markerPath = "@pid:" + pid + "/" + visit + "/skeleton/calibration/static_calibrated_marker.xml";
    std::string scalePath = "@pid:" + pid + "/" + visit + "/skeleton/calibration/static_calibrated_body_scale.yaml";
    return mgr.exists(markerPath) && mgr.exists(scalePath);
}

/**
 * Load calibration files into characters
 */
bool loadCalibrationFiles(
    RenderCharacter* freeCharacter,
    RenderCharacter* motionCharacter,
    rm::ResourceManager& mgr,
    const std::string& pid,
    const std::string& visit,
    bool verbose)
{
    std::string markerPath = mgr.resolve("@pid:" + pid + "/" + visit + "/skeleton/calibration/static_calibrated_marker.xml");
    std::string scalePath = mgr.resolve("@pid:" + pid + "/" + visit + "/skeleton/calibration/static_calibrated_body_scale.yaml");

    // Load body scales
    if (freeCharacter && !freeCharacter->loadBodyScaleYAML(scalePath)) {
        LOG_ERROR("[Calibration] Failed to load body scales for freeCharacter: " << scalePath);
        return false;
    }
    if (motionCharacter && !motionCharacter->loadBodyScaleYAML(scalePath)) {
        LOG_ERROR("[Calibration] Failed to load body scales for motionCharacter: " << scalePath);
        return false;
    }

    // Load markers
    if (freeCharacter) freeCharacter->loadMarkers(markerPath);
    if (motionCharacter) motionCharacter->loadMarkers(markerPath);

    if (verbose) {
        LOG_INFO("[Calibration] Loaded calibration from: " << markerPath);
    }

    return true;
}

// ============================================================================
// Processing Job Structure
// ============================================================================
struct ProcessingJob {
    std::string pid;
    std::string visit;
    std::string motionName;
    std::string staticC3DPath;
    std::string motionC3DPath;
    bool needsStatic;
    bool needsDynamic;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::string mode;
    std::string pid;
    std::string visit;
    std::string motion;
    std::string skeletonName;
    std::string configPath;
    bool verbose = false;
    bool dryRun = false;

    po::options_description desc("C3D Calibration CLI");
    desc.add_options()
        ("help,h", "Show help message")
        ("mode,m", po::value<std::string>(&mode)->default_value("full"),
         "Mode: static | dynamic | full (default: full)")
        ("pid,p", po::value<std::string>(&pid),
         "Patient ID (omit for all PIDs)")
        ("visit", po::value<std::string>(&visit),
         "Visit type: pre | op1 | op2 (omit for all visits)")
        ("motion,n", po::value<std::string>(&motion),
         "Motion name filter (omit for all motions)")
        ("skeleton-name", po::value<std::string>(&skeletonName),
         "Output skeleton name (default: motion filename)")
        ("config,c", po::value<std::string>(&configPath),
         "Custom config YAML for skeleton/marker paths (optional)")
        ("verbose,v", po::bool_switch(&verbose),
         "Verbose output")
        ("dry-run", po::bool_switch(&dryRun),
         "Show what would be processed without executing");

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .run(), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        LOG_ERROR("Command line error: " << e.what());
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << "C3D Skeleton Calibration CLI\n\n";
        std::cout << "Automates skeleton retrieval pipeline from C3D motion capture data.\n\n";
        std::cout << "Usage: " << argv[0] << " [options]\n\n";
        std::cout << desc << std::endl;
        std::cout << "\nModes:\n";
        std::cout << "  static  - Static calibration only (from medial marker trial)\n";
        std::cout << "  dynamic - Dynamic calibration + HDF export (requires existing calibration)\n";
        std::cout << "  full    - Static then dynamic (complete pipeline)\n";
        std::cout << "\nExamples:\n";
        std::cout << "  # Process all PIDs, all visits, all motions\n";
        std::cout << "  " << argv[0] << " --mode full -v\n\n";
        std::cout << "  # All visits and motions for a patient\n";
        std::cout << "  " << argv[0] << " --mode full --pid 12964246 -v\n\n";
        std::cout << "  # Specific motion\n";
        std::cout << "  " << argv[0] << " --mode dynamic --pid 12964246 --visit pre --motion Walk01 -v\n";
        return 0;
    }

    // Validate mode
    if (mode != "static" && mode != "dynamic" && mode != "full") {
        LOG_ERROR("Invalid mode: " << mode << " (must be static, dynamic, or full)");
        return 1;
    }

    if (verbose) {
        LOG_INFO("C3D Calibration CLI");
        LOG_INFO("Mode: " << mode);
        LOG_INFO("PID: " << (pid.empty() ? "(all)" : pid));
        LOG_INFO("Visit: " << (visit.empty() ? "(all)" : visit));
        LOG_INFO("Motion: " << (motion.empty() ? "(all)" : motion));
    }

    // Get resource manager
    rm::ResourceManager& mgr = rm::getManager();

    // Load config paths
    std::string skeletonPath = DEFAULT_SKELETON;
    std::string markerPath = DEFAULT_MARKER;
    std::string dynamicConfigPath = DEFAULT_DYNAMIC_CONFIG;
    std::string staticConfigPath = DEFAULT_STATIC_CONFIG;

    if (!configPath.empty()) {
        try {
            YAML::Node config = YAML::LoadFile(rm::resolve(configPath));
            if (config["skeleton"]) skeletonPath = config["skeleton"].as<std::string>();
            if (config["marker"]) markerPath = config["marker"].as<std::string>();
            if (config["dynamic_config"]) dynamicConfigPath = config["dynamic_config"].as<std::string>();
            if (config["static_config"]) staticConfigPath = config["static_config"].as<std::string>();
            if (verbose) {
                LOG_INFO("Loaded config from: " << configPath);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load config: " << e.what());
            return 1;
        }
    }

    // Resolve paths
    std::string resolvedSkeletonPath = rm::resolve(skeletonPath);
    std::string resolvedMarkerPath = rm::resolve(markerPath);
    std::string resolvedDynamicConfig = rm::resolve(dynamicConfigPath);
    std::string resolvedStaticConfig = rm::resolve(staticConfigPath);

    if (verbose) {
        LOG_INFO("Skeleton: " << skeletonPath);
        LOG_INFO("Marker: " << markerPath);
        LOG_INFO("Dynamic config: " << dynamicConfigPath);
        LOG_INFO("Static config: " << staticConfigPath);
    }

    // Build job list
    std::vector<ProcessingJob> jobs;

    // Discover PIDs
    std::vector<std::string> pids;
    if (!pid.empty()) {
        pids.push_back(pid);
    } else {
        pids = discoverPIDs(mgr);
        if (verbose) {
            LOG_INFO("Discovered " << pids.size() << " PIDs");
        }
    }

    // Build jobs for each PID/visit/motion combination
    for (const auto& p : pids) {
        std::vector<std::string> visits_to_process;
        if (!visit.empty()) {
            visits_to_process.push_back(visit);
        } else {
            visits_to_process = discoverVisits(mgr, p);
        }

        for (const auto& v : visits_to_process) {
            // Resolve gait directory
            std::string gaitPattern = "@pid:" + p + "/" + v + "/gait";
            std::string gaitDir;
            try {
                gaitDir = mgr.resolveDir(gaitPattern);
            } catch (...) {
                if (verbose) {
                    LOG_WARN("Gait directory not found: " << gaitPattern);
                }
                continue;
            }

            if (gaitDir.empty() || !fs::exists(gaitDir)) {
                if (verbose) {
                    LOG_WARN("Gait directory does not exist: " << gaitDir);
                }
                continue;
            }

            // Find static C3D
            std::string staticPath = findStaticC3D(gaitDir);

            // Discover motions
            std::vector<std::string> motions = discoverMotions(gaitDir, motion);

            bool hasCalib = hasCalibrationFiles(mgr, p, v);

            if (mode == "static") {
                // Static mode: just calibrate
                if (!staticPath.empty()) {
                    ProcessingJob job;
                    job.pid = p;
                    job.visit = v;
                    job.staticC3DPath = staticPath;
                    job.needsStatic = true;
                    job.needsDynamic = false;
                    jobs.push_back(job);
                }
            } else if (mode == "dynamic") {
                // Dynamic mode: requires existing calibration
                if (!hasCalib) {
                    LOG_ERROR("No calibration found at @pid:" << p << "/" << v << "/skeleton/calibration/");
                    LOG_ERROR("Hint: Use --mode full to create calibration first");
                    continue;
                }

                for (const auto& m : motions) {
                    ProcessingJob job;
                    job.pid = p;
                    job.visit = v;
                    job.motionName = m;
                    job.motionC3DPath = gaitDir + "/" + m + ".c3d";
                    job.needsStatic = false;
                    job.needsDynamic = true;
                    jobs.push_back(job);
                }
            } else {  // full mode
                // First job: static calibration (if static file exists and no calibration)
                if (!staticPath.empty() && !hasCalib) {
                    ProcessingJob staticJob;
                    staticJob.pid = p;
                    staticJob.visit = v;
                    staticJob.staticC3DPath = staticPath;
                    staticJob.needsStatic = true;
                    staticJob.needsDynamic = false;
                    jobs.push_back(staticJob);
                }

                // Then: dynamic calibration for each motion
                for (const auto& m : motions) {
                    ProcessingJob job;
                    job.pid = p;
                    job.visit = v;
                    job.motionName = m;
                    job.motionC3DPath = gaitDir + "/" + m + ".c3d";
                    job.needsStatic = false;
                    job.needsDynamic = true;
                    jobs.push_back(job);
                }
            }
        }
    }

    if (jobs.empty()) {
        LOG_WARN("No jobs to process");
        return 0;
    }

    if (verbose || dryRun) {
        LOG_INFO("Jobs to process: " << jobs.size());
        if (dryRun) {
            for (const auto& job : jobs) {
                std::cout << "  [" << (job.needsStatic ? "S" : "D") << "] "
                          << "PID:" << job.pid << " visit:" << job.visit;
                if (!job.motionName.empty()) {
                    std::cout << " motion:" << job.motionName;
                }
                std::cout << std::endl;
            }
            return 0;
        }
    }

    // Create characters
    auto freeCharacter = std::make_unique<RenderCharacter>(
        resolvedSkeletonPath, SKEL_COLLIDE_ALL | SKEL_FREE_JOINTS);
    auto motionCharacter = std::make_unique<RenderCharacter>(
        resolvedSkeletonPath, SKEL_COLLIDE_ALL);

    // Load default markers
    freeCharacter->loadMarkers(resolvedMarkerPath);
    motionCharacter->loadMarkers(resolvedMarkerPath);

    // Create C3D Reader
    auto reader = std::make_unique<C3D_Reader>(
        resolvedDynamicConfig,
        resolvedMarkerPath,
        freeCharacter.get(),
        motionCharacter.get()
    );

    // Process jobs
    ProgressBar progress(static_cast<int>(jobs.size()), !verbose);
    int successCount = 0;
    int failCount = 0;
    std::string lastPid, lastVisit;

    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto& job = jobs[i];
        progress.update(static_cast<int>(i + 1), job.pid, job.visit, job.motionName);

        // Reload calibration when PID/visit changes
        if (job.needsDynamic && (job.pid != lastPid || job.visit != lastVisit)) {
            // Reset to default and reload calibration
            freeCharacter = std::make_unique<RenderCharacter>(
                resolvedSkeletonPath, SKEL_COLLIDE_ALL | SKEL_FREE_JOINTS);
            motionCharacter = std::make_unique<RenderCharacter>(
                resolvedSkeletonPath, SKEL_COLLIDE_ALL);

            if (hasCalibrationFiles(mgr, job.pid, job.visit)) {
                if (!loadCalibrationFiles(freeCharacter.get(), motionCharacter.get(),
                                          mgr, job.pid, job.visit, verbose)) {
                    LOG_ERROR("Failed to load calibration for PID:" << job.pid << " visit:" << job.visit);
                    failCount++;
                    continue;
                }
            } else {
                // Load default markers
                freeCharacter->loadMarkers(resolvedMarkerPath);
                motionCharacter->loadMarkers(resolvedMarkerPath);
            }

            // Recreate reader with new characters
            std::string markerConfigForReader = resolvedMarkerPath;
            if (hasCalibrationFiles(mgr, job.pid, job.visit)) {
                markerConfigForReader = mgr.resolve("@pid:" + job.pid + "/" + job.visit + "/skeleton/calibration/static_calibrated_marker.xml");
            }
            reader = std::make_unique<C3D_Reader>(
                resolvedDynamicConfig,
                markerConfigForReader,
                freeCharacter.get(),
                motionCharacter.get()
            );

            lastPid = job.pid;
            lastVisit = job.visit;
        }

        int result = 0;

        if (job.needsStatic) {
            // Run static calibration
            std::string skelOutputDir = mgr.resolveDirCreate("@pid:" + job.pid + "/" + job.visit + "/skeleton");
            result = runStaticCalibration(
                reader.get(),
                freeCharacter.get(),
                job.staticC3DPath,
                resolvedStaticConfig,
                skelOutputDir,
                verbose
            );

            // If static succeeded, update lastPid/lastVisit to trigger reload next time
            if (result == 0) {
                lastPid = "";
                lastVisit = "";
            }
        }

        if (job.needsDynamic && result == 0) {
            // Determine skeleton name
            std::string skelName = skeletonName.empty() ? job.motionName : skeletonName;

            std::string motionOutputDir = mgr.resolveDirCreate("@pid:" + job.pid + "/" + job.visit + "/motion");
            std::string skelOutputDir = mgr.resolveDirCreate("@pid:" + job.pid + "/" + job.visit + "/skeleton");

            result = runDynamicCalibration(
                reader.get(),
                freeCharacter.get(),
                motionCharacter.get(),
                job.motionC3DPath,
                motionOutputDir,
                skelOutputDir,
                skelName,
                job.pid,
                job.visit,
                verbose
            );
        }

        if (result == 0) {
            successCount++;
        } else {
            failCount++;
        }
    }

    progress.finish();

    // Summary
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "Success: " << successCount << std::endl;
    std::cout << "Failed:  " << failCount << std::endl;

    return failCount > 0 ? 1 : 0;
}

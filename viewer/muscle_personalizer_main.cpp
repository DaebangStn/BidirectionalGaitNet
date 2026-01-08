#include "MusclePersonalizerApp.h"
#include "SurgeryExecutor.h"
#include "WaypointOptimizer.h"
#include "ContractureOptimizer.h"
#include "Character.h"
#include "rm/global.hpp"
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <regex>
#include <cmath>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// Convert glob pattern to regex (e.g., "*Sol*" -> ".*Sol.*")
static std::regex globToRegex(const std::string& pattern) {
    std::string regexStr;
    for (char c : pattern) {
        switch (c) {
            case '*': regexStr += ".*"; break;
            case '?': regexStr += "."; break;
            case '.': regexStr += "\\."; break;
            default: regexStr += c;
        }
    }
    return std::regex(regexStr, std::regex::icase);
}

// Print LengthCurveCharacteristics
static void printCurveChars(const std::string& label, const PMuscle::LengthCurveCharacteristics& chars) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "    " << std::setw(10) << label << ": "
              << "min_phase=" << std::setw(6) << chars.min_phase
              << "  max_phase=" << std::setw(6) << chars.max_phase
              << "  delta=" << std::setw(8) << chars.delta
              << "  range=[" << chars.min_length << ", " << chars.max_length << "]"
              << std::endl;
}

// Compute distance between two curve characteristics
static double curveDistance(const PMuscle::LengthCurveCharacteristics& a,
                            const PMuscle::LengthCurveCharacteristics& b) {
    // Same weights as in WaypointOptimizer (kPhaseWeight=0.007, kDeltaWeight=0.5)
    double phase_diff = 0.007 * std::pow(a.min_phase - b.min_phase, 2)
                      + 0.007 * std::pow(a.max_phase - b.max_phase, 2);
    double delta_diff = 0.5 * std::pow(a.delta - b.delta, 2);
    return std::sqrt(phase_diff + delta_diff);
}

// Command-line waypoint optimization for debugging
int runWaypointOptimizationCLI(const std::string& configPath,
                                const std::string& hdfPath,
                                const std::vector<std::string>& muscleNames,
                                bool verbose)
{
    // Load config
    std::string resolvedConfig = rm::resolve(configPath);
    YAML::Node config = YAML::LoadFile(resolvedConfig);

    std::string skeletonPath = config["paths"]["skeleton_default"].as<std::string>();
    std::string musclePath = config["paths"]["muscle_default"].as<std::string>();

    // Load reference character paths (for waypoint optimization target)
    std::string refSkeletonPath = config["paths"]["reference_skeleton"].as<std::string>(skeletonPath);
    std::string refMusclePath = config["paths"]["reference_muscle"].as<std::string>(musclePath);

    std::cout << "[CLI] Loading subject character..." << std::endl;
    std::cout << "[CLI]   Skeleton: " << skeletonPath << std::endl;
    std::cout << "[CLI]   Muscle: " << musclePath << std::endl;

    // Create surgery executor and load subject character
    auto executor = std::make_unique<PMuscle::SurgeryExecutor>("cli-waypoint-opt");
    executor->loadCharacter(skeletonPath, musclePath);

    Character* character = executor->getCharacter();
    if (!character) {
        std::cerr << "[CLI] Failed to load subject character" << std::endl;
        return 1;
    }

    std::cout << "[CLI] Subject loaded with " << character->getMuscles().size() << " muscles" << std::endl;

    // Load reference character
    std::cout << "[CLI] Loading reference character..." << std::endl;
    std::cout << "[CLI]   Skeleton: " << refSkeletonPath << std::endl;
    std::cout << "[CLI]   Muscle: " << refMusclePath << std::endl;

    auto refExecutor = std::make_unique<PMuscle::SurgeryExecutor>("cli-reference");
    refExecutor->loadCharacter(refSkeletonPath, refMusclePath);

    Character* refCharacter = refExecutor->getCharacter();
    if (!refCharacter) {
        std::cerr << "[CLI] Failed to load reference character" << std::endl;
        return 1;
    }

    std::cout << "[CLI] Reference loaded with " << refCharacter->getMuscles().size() << " muscles" << std::endl;

    // Use provided muscle names or all muscles
    std::vector<std::string> targetMuscles = muscleNames;
    if (targetMuscles.empty()) {
        for (const auto& m : character->getMuscles()) {
            targetMuscles.push_back(m->name);
        }
    }

    if (targetMuscles.empty()) {
        std::cerr << "[CLI] No muscles to optimize" << std::endl;
        return 1;
    }

    // Build Config from YAML
    PMuscle::WaypointOptimizer::Config optConfig;
    optConfig.maxIterations = config["waypoint_optimization"]["max_iterations"].as<int>(100);
    optConfig.numSampling = config["waypoint_optimization"]["num_sampling"].as<int>(50);
    optConfig.lambdaShape = config["waypoint_optimization"]["lambda_shape"].as<double>(1.0);
    optConfig.lambdaLengthCurve = config["waypoint_optimization"]["lambda_length_curve"].as<double>(1.0);
    optConfig.fixOriginInsertion = config["waypoint_optimization"]["fix_origin_insertion"].as<bool>(true);
    optConfig.analyticalGradient = config["waypoint_optimization"]["analytical_gradient"].as<bool>(false);
    optConfig.weightPhase = config["waypoint_optimization"]["weight_phase"].as<double>(1.0);
    optConfig.weightDelta = config["waypoint_optimization"]["weight_delta"].as<double>(50.0);
    optConfig.numParallel = config["waypoint_optimization"]["parallelism"].as<int>(1);
    optConfig.verbose = verbose;
    // lengthType defaults to MTU_LENGTH in Config constructor

    std::cout << "[CLI] Running waypoint optimization..." << std::endl;
    std::cout << "[CLI]   HDF: " << hdfPath << std::endl;
    std::cout << "[CLI]   Muscles: " << targetMuscles.size() << std::endl;
    std::cout << "[CLI]   Config: maxIter=" << optConfig.maxIterations << " numSampling=" << optConfig.numSampling
              << " lambdaShape=" << optConfig.lambdaShape << " lambdaLengthCurve=" << optConfig.lambdaLengthCurve
              << std::endl;
    std::cout << "[CLI]   Gradient: " << (optConfig.analyticalGradient ? "analytical" : "numeric")
              << " weightPhase=" << optConfig.weightPhase << " weightDelta=" << optConfig.weightDelta
              << " numParallel=" << optConfig.numParallel << std::endl;

    // Use optimizeWaypointsWithResults with separate reference character
    auto results = executor->optimizeWaypointsWithResults(
        targetMuscles,
        hdfPath,
        optConfig,
        refCharacter,  // Use separate reference character
        nullptr,       // No mutex needed for CLI
        nullptr        // No callback needed
    );

    // Print detailed results and verify improvement
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "OPTIMIZATION RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    int improved = 0, unchanged = 0, worsened = 0;
    double total_before_dist = 0.0, total_after_dist = 0.0;

    for (const auto& result : results) {
        double dist_before = curveDistance(result.subject_before_chars, result.reference_chars);
        double dist_after = curveDistance(result.subject_after_chars, result.reference_chars);
        double improvement = dist_before - dist_after;

        total_before_dist += dist_before;
        total_after_dist += dist_after;

        if (improvement > 1e-6) improved++;
        else if (improvement < -1e-6) worsened++;
        else unchanged++;

        std::cout << "\n[" << result.muscle_name << "] " << (result.success ? "SUCCESS" : "FAILED") << std::endl;

        // Print energy breakdown
        double shape_change = result.initial_shape_energy - result.final_shape_energy;
        double length_change = result.initial_length_energy - result.final_length_energy;
        double total_change = result.initial_total_cost - result.final_total_cost;

        std::cout << "    Energy: shape=" << std::fixed << std::setprecision(4)
                  << result.initial_shape_energy << "→" << result.final_shape_energy;
        if (shape_change > 1e-6) std::cout << " \033[32m↓" << shape_change << "\033[0m";
        else if (shape_change < -1e-6) std::cout << " \033[31m↑" << (-shape_change) << "\033[0m";

        std::cout << " | length=" << result.initial_length_energy << "→" << result.final_length_energy;
        if (length_change > 1e-6) std::cout << " \033[32m↓" << length_change << "\033[0m";
        else if (length_change < -1e-6) std::cout << " \033[31m↑" << (-length_change) << "\033[0m";

        std::cout << " | total=" << result.initial_total_cost << "→" << result.final_total_cost;
        if (total_change > 1e-6) std::cout << " \033[32m↓" << total_change << "\033[0m";
        else if (total_change < -1e-6) std::cout << " \033[31m↑" << (-total_change) << "\033[0m";
        std::cout << std::endl;

        if (verbose || !result.success || worsened > 0) {
            printCurveChars("Reference", result.reference_chars);
            printCurveChars("Before", result.subject_before_chars);
            printCurveChars("After", result.subject_after_chars);
        }

        std::cout << "    Distance: before=" << std::fixed << std::setprecision(6) << dist_before
                  << " after=" << dist_after;
        if (improvement > 1e-6) {
            std::cout << " \033[32m↓ improved by " << improvement << "\033[0m";
        } else if (improvement < -1e-6) {
            std::cout << " \033[31m↑ WORSENED by " << (-improvement) << "\033[0m";
        } else {
            std::cout << " unchanged";
        }
        std::cout << std::endl;
    }

    // Summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Total muscles: " << results.size() << std::endl;
    std::cout << "  Improved:  " << improved << std::endl;
    std::cout << "  Unchanged: " << unchanged << std::endl;
    std::cout << "  Worsened:  " << worsened << std::endl;
    std::cout << "Total distance: before=" << std::fixed << std::setprecision(6) << total_before_dist
              << " after=" << total_after_dist;
    if (total_before_dist > total_after_dist) {
        std::cout << " \033[32m(overall improved by " << (total_before_dist - total_after_dist) << ")\033[0m";
    } else if (total_before_dist < total_after_dist) {
        std::cout << " \033[31m(overall WORSENED by " << (total_after_dist - total_before_dist) << ")\033[0m";
    }
    std::cout << std::endl;

    if (worsened > 0) {
        std::cerr << "\n[WARNING] " << worsened << " muscle(s) worsened after optimization!" << std::endl;
    }

    return 0;
}

// Command-line export for testing cyclic consistency
int runExportCLI(const std::string& skeletonPath,
                  const std::string& musclePath,
                  const std::string& outputName)
{
    std::cout << "[CLI-Export] Loading character..." << std::endl;
    std::cout << "[CLI-Export]   Skeleton: " << skeletonPath << std::endl;
    std::cout << "[CLI-Export]   Muscle: " << musclePath << std::endl;

    // Create surgery executor and load character
    auto executor = std::make_unique<PMuscle::SurgeryExecutor>("cli-export");
    executor->loadCharacter(skeletonPath, musclePath);

    Character* character = executor->getCharacter();
    if (!character) {
        std::cerr << "[CLI-Export] Failed to load character" << std::endl;
        return 1;
    }

    std::cout << "[CLI-Export] Character loaded with " << character->getMuscles().size() << " muscles" << std::endl;

    // Build output paths - always use resolveDir since file may not exist yet
    fs::path skelDir = rm::getManager().resolveDir("@data/skeleton");
    fs::path muscleDir = rm::getManager().resolveDir("@data/muscle");

    std::string skelResolved = (skelDir / (outputName + ".yaml")).string();
    std::string muscleResolved = (muscleDir / (outputName + ".yaml")).string();

    std::cout << "[CLI-Export] Exporting..." << std::endl;
    std::cout << "[CLI-Export]   Skeleton -> " << skelResolved << std::endl;
    std::cout << "[CLI-Export]   Muscle -> " << muscleResolved << std::endl;

    try {
        executor->exportSkeleton(skelResolved);
        std::cout << "[CLI-Export] Skeleton exported successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[CLI-Export] Error exporting skeleton: " << e.what() << std::endl;
        return 1;
    }

    try {
        executor->exportMuscles(muscleResolved);
        std::cout << "[CLI-Export] Muscle exported successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[CLI-Export] Error exporting muscle: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[CLI-Export] Export completed successfully" << std::endl;
    return 0;
}

// Command-line ROM config test for debugging IK computation
int runTestROMCLI(const std::string& configPath,
                   const std::string& romPath,
                   double romAngle)
{
    // Load config
    std::string resolvedConfig = rm::resolve(configPath);
    YAML::Node config = YAML::LoadFile(resolvedConfig);

    std::string skeletonPath = config["paths"]["skeleton_default"].as<std::string>();
    std::string musclePath = config["paths"]["muscle_default"].as<std::string>();

    std::cout << "[ROM-Test] Loading character..." << std::endl;
    std::cout << "[ROM-Test]   Skeleton: " << skeletonPath << std::endl;
    std::cout << "[ROM-Test]   Muscle: " << musclePath << std::endl;

    // Create surgery executor and load character
    auto executor = std::make_unique<PMuscle::SurgeryExecutor>("cli-rom-test");
    executor->loadCharacter(skeletonPath, musclePath);

    Character* character = executor->getCharacter();
    if (!character) {
        std::cerr << "[ROM-Test] Failed to load character" << std::endl;
        return 1;
    }

    auto skeleton = character->getSkeleton();
    std::cout << "[ROM-Test] Character loaded with " << skeleton->getNumDofs() << " DOFs" << std::endl;

    // Load ROM config
    std::cout << "[ROM-Test] Loading ROM config: " << romPath << std::endl;
    PMuscle::ROMTrialConfig romConfig = PMuscle::ContractureOptimizer::loadROMConfig(
        romPath, skeleton);

    // Override rom_angle with CLI argument
    romConfig.rom_angle = romAngle;

    std::cout << "[ROM-Test] ROM Config:" << std::endl;
    std::cout << "  Name: " << romConfig.name << std::endl;
    std::cout << "  Joint: " << romConfig.joint << std::endl;
    std::cout << "  DOF type: " << romConfig.dof_type << std::endl;
    std::cout << "  ROM angle: " << romConfig.rom_angle << "°" << std::endl;
    std::cout << "  Composite DOF: " << (romConfig.is_composite_dof ? "yes" : "no") << std::endl;

    // Build pose data (this triggers IK computation for abd_knee)
    PMuscle::ContractureOptimizer optimizer;
    std::vector<PMuscle::ROMTrialConfig> configs = { romConfig };
    auto poseData = optimizer.buildPoseData(character, configs);

    if (poseData.empty()) {
        std::cerr << "[ROM-Test] Failed to build pose data" << std::endl;
        return 1;
    }

    std::cout << "[ROM-Test] Pose data built successfully" << std::endl;
    std::cout << "[ROM-Test] Final pose (first 12 DOFs): " << poseData[0].q.head(12).transpose() << std::endl;

    // Print all joints with their DOF indices
    std::cout << "[ROM-Test] Skeleton joint structure:" << std::endl;
    for (size_t i = 0; i < skeleton->getNumJoints(); ++i) {
        auto* j = skeleton->getJoint(i);
        int dof_start = (j->getNumDofs() > 0) ? static_cast<int>(j->getIndexInSkeleton(0)) : -1;
        std::cout << "  Joint " << i << ": " << j->getName()
                  << " DOFs=" << j->getNumDofs()
                  << " start=" << dof_start << std::endl;
        if (i > 10) {
            std::cout << "  ..." << std::endl;
            break;
        }
    }

    // Print FemurL DOF indices
    auto* femurL = skeleton->getJoint("FemurL");
    if (femurL) {
        int hip_start = static_cast<int>(femurL->getIndexInSkeleton(0));
        std::cout << "[ROM-Test] FemurL DOF start: " << hip_start << std::endl;
        std::cout << "[ROM-Test] FemurL values: ["
                  << poseData[0].q[hip_start] << ", "
                  << poseData[0].q[hip_start+1] << ", "
                  << poseData[0].q[hip_start+2] << "]" << std::endl;
    }
    auto* tibiaL = skeleton->getJoint("TibiaL");
    if (tibiaL) {
        int knee_idx = static_cast<int>(tibiaL->getIndexInSkeleton(0));
        std::cout << "[ROM-Test] TibiaL DOF idx: " << knee_idx
                  << " value: " << poseData[0].q[knee_idx] << std::endl;
    }

    return 0;
}

int main(int argc, char** argv)
{
    std::string configPath = "@data/config/muscle_personalizer.yaml";
    std::string hdfPath;
    std::string muscleFilter;
    std::string skeletonPath;
    std::string muscleFilePath;
    std::string outputName;
    std::string romPath;
    double romAngle = 45.0;
    bool verbose = false;

    // Parse command line arguments
    po::options_description desc("Muscle Personalizer - Interactive muscle configuration tool");
    desc.add_options()
        ("help,h", "Show this help message")
        ("config,c", po::value<std::string>(&configPath)->default_value(configPath),
         "Path to configuration YAML file")
        ("waypoint-opt", "Run waypoint optimization in CLI mode (for debugging)")
        ("hdf", po::value<std::string>(&hdfPath),
         "Path to HDF motion file (required for --waypoint-opt)")
        ("muscle-filter,m", po::value<std::string>(&muscleFilter),
         "Muscle filter pattern with wildcards (e.g., '*Sol*' for Soleus muscles)")
        ("verbose,v", po::bool_switch(&verbose),
         "Show detailed curve characteristics for all muscles")
        ("export", "Export skeleton and muscle configs in CLI mode")
        ("skeleton", po::value<std::string>(&skeletonPath),
         "Input skeleton path (for --export)")
        ("muscle", po::value<std::string>(&muscleFilePath),
         "Input muscle path (for --export)")
        ("output,o", po::value<std::string>(&outputName),
         "Output config name without extension (for --export)")
        ("test-rom", "Test ROM config IK computation in CLI mode")
        ("rom", po::value<std::string>(&romPath),
         "Path to ROM config YAML (for --test-rom)")
        ("rom-angle", po::value<double>(&romAngle)->default_value(45.0),
         "ROM angle in degrees (for --test-rom or --contracture)")
        ("contracture", "Run contracture optimization in CLI mode")
        ("rom-configs", po::value<std::vector<std::string>>()->multitoken(),
         "Paths to ROM config YAML files (for --contracture)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (const po::error& e) {
        std::cerr << "Error parsing command line: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        std::cout << "\nFeatures:" << std::endl;
        std::cout << "  1. Weight Application - Scale muscle f0 parameters based on target body mass" << std::endl;
        std::cout << "  2. Waypoint Optimization - Optimize muscle paths from HDF motion files" << std::endl;
        std::cout << "  3. Contracture Estimation - Fit lm_contract parameters from ROM trials" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  # Optimize all muscles" << std::endl;
        std::cout << "  ./muscle_personalizer --waypoint-opt --hdf data/motion/walk.h5" << std::endl;
        std::cout << "\n  # Optimize only Soleus muscles with verbose output" << std::endl;
        std::cout << "  ./muscle_personalizer --waypoint-opt --hdf data/motion/walk.h5 -m '*Sol*' -v" << std::endl;
        std::cout << "\n  # Filter patterns: *Sol* (Soleus), L_* (left side), *Rect* (Rectus)" << std::endl;
        return 0;
    }

    // CLI mode for waypoint optimization debugging
    if (vm.count("waypoint-opt")) {
        if (hdfPath.empty()) {
            std::cerr << "Error: --hdf is required for --waypoint-opt mode" << std::endl;
            return 1;
        }

        // Filter muscle names if pattern provided
        std::vector<std::string> filteredMuscles;
        if (!muscleFilter.empty()) {
            // Load character to get muscle names for filtering
            std::string resolvedConfig = rm::resolve(configPath);
            YAML::Node config = YAML::LoadFile(resolvedConfig);
            std::string skeletonPath = config["paths"]["skeleton_default"].as<std::string>();
            std::string musclePath = config["paths"]["muscle_default"].as<std::string>();

            auto tempExecutor = std::make_unique<PMuscle::SurgeryExecutor>("temp-filter");
            tempExecutor->loadCharacter(skeletonPath, musclePath);
            Character* character = tempExecutor->getCharacter();

            if (character) {
                std::regex pattern = globToRegex(muscleFilter);
                for (const auto& m : character->getMuscles()) {
                    if (std::regex_match(m->name, pattern)) {
                        filteredMuscles.push_back(m->name);
                    }
                }
                std::cout << "[CLI] Filter '" << muscleFilter << "' matched " << filteredMuscles.size() << " muscles:" << std::endl;
                for (const auto& name : filteredMuscles) {
                    std::cout << "[CLI]   - " << name << std::endl;
                }
            }

            if (filteredMuscles.empty()) {
                std::cerr << "Error: No muscles matched filter '" << muscleFilter << "'" << std::endl;
                return 1;
            }
        }

        return runWaypointOptimizationCLI(configPath, hdfPath, filteredMuscles, verbose);
    }

    // CLI mode for export
    if (vm.count("export")) {
        if (skeletonPath.empty() || muscleFilePath.empty() || outputName.empty()) {
            std::cerr << "Error: --skeleton, --muscle, and --output are required for --export mode" << std::endl;
            return 1;
        }
        return runExportCLI(skeletonPath, muscleFilePath, outputName);
    }

    // CLI mode for ROM config testing
    if (vm.count("test-rom")) {
        if (romPath.empty()) {
            std::cerr << "Error: --rom is required for --test-rom mode" << std::endl;
            return 1;
        }
        return runTestROMCLI(configPath, romPath, romAngle);
    }

    // CLI mode for contracture optimization (L/R symmetry verification)
    if (vm.count("contracture")) {
        if (!vm.count("rom-configs") || vm["rom-configs"].as<std::vector<std::string>>().empty()) {
            std::cerr << "Error: --rom-configs is required for --contracture mode" << std::endl;
            return 1;
        }

        auto romConfigs = vm["rom-configs"].as<std::vector<std::string>>();
        std::cout << "[Contracture CLI] Loading " << romConfigs.size() << " ROM configs" << std::endl;

        // Load config
        std::string resolvedConfig = rm::resolve(configPath);
        YAML::Node config = YAML::LoadFile(resolvedConfig);
        std::string skelPath = config["paths"]["skeleton_default"].as<std::string>();
        std::string muscPath = config["paths"]["muscle_default"].as<std::string>();
        std::string muscleGroupsPath = config["paths"]["muscle_groups"].as<std::string>("@data/config/muscle_groups.yaml");

        // Load character
        auto executor = std::make_unique<PMuscle::SurgeryExecutor>("cli-contracture");
        executor->loadCharacter(skelPath, muscPath);
        Character* character = executor->getCharacter();
        if (!character) {
            std::cerr << "[Contracture CLI] Failed to load character" << std::endl;
            return 1;
        }
        auto skeleton = character->getSkeleton();
        std::cout << "[Contracture CLI] Character loaded: " << character->getMuscles().size() << " muscles" << std::endl;

        // Load ROM trial configs
        std::vector<PMuscle::ROMTrialConfig> trials;
        for (const auto& romConfigPath : romConfigs) {
            auto trial = PMuscle::ContractureOptimizer::loadROMConfig(romConfigPath, skeleton);
            // Use rom_angle from YAML (typically set via clinical_data or manual)
            // For testing, set a fixed ROM angle
            trial.rom_angle = romAngle;
            trials.push_back(trial);
            std::cout << "[Contracture CLI] Loaded: " << trial.name
                      << " (joint=" << trial.joint << ", dof=" << trial.dof_index
                      << ", rom=" << trial.rom_angle << "°)" << std::endl;
        }

        // Load muscle groups
        PMuscle::ContractureOptimizer optimizer;
        int numGroups = optimizer.loadMuscleGroups(muscleGroupsPath, character);
        std::cout << "[Contracture CLI] Loaded " << numGroups << " muscle groups" << std::endl;

        // Load grid search mapping from config
        auto ce = config["contracture_estimation"];
        if (ce && ce["grid_search_mapping"]) {
            std::vector<PMuscle::GridSearchMapping> gridSearchMapping;
            for (const auto& entry : ce["grid_search_mapping"]) {
                PMuscle::GridSearchMapping mapping;
                if (entry["trials"]) {
                    for (const auto& t : entry["trials"]) {
                        mapping.trials.push_back(t.as<std::string>());
                    }
                }
                if (entry["groups"]) {
                    for (const auto& g : entry["groups"]) {
                        mapping.groups.push_back(g.as<std::string>());
                    }
                }
                if (!mapping.trials.empty() && !mapping.groups.empty()) {
                    gridSearchMapping.push_back(mapping);
                }
            }
            optimizer.setGridSearchMapping(gridSearchMapping);
            std::cout << "[Contracture CLI] Loaded " << gridSearchMapping.size() << " grid search mappings" << std::endl;
        }

        // Configure optimizer
        PMuscle::ContractureOptimizer::Config optConfig;
        optConfig.maxIterations = 50;
        optConfig.minRatio = 0.7;
        optConfig.maxRatio = 1.3;
        optConfig.verbose = verbose;
        optConfig.outerIterations = 1;

        // Run optimization
        std::cout << "\n[Contracture CLI] Running optimization..." << std::endl;
        auto results = optimizer.optimize(character, trials, optConfig);

        // Print results
        std::cout << "\n========== CONTRACTURE OPTIMIZATION RESULTS ==========" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        for (const auto& gr : results) {
            std::cout << "  " << std::setw(25) << std::left << gr.group_name
                      << " ratio=" << gr.ratio << std::endl;
        }
        std::cout << "======================================================" << std::endl;

        return 0;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "    Muscle Personalizer Application" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Config: " << configPath << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        MusclePersonalizerApp app(configPath);
        app.startLoop();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

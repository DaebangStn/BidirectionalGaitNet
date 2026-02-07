/**
 * @file contracture_cli_main.cpp
 * @brief CLI tool for contracture optimization (tiered grid+Ceres by default)
 *
 * Usage:
 *   contracture_cli -v                                          # default ROM trials, tiered optimization
 *   contracture_cli --rom-configs @data/config/rom/intRot_R.yaml @data/config/rom/extRot_R.yaml -v
 *   contracture_cli --grid-only --groups hip_lateral_rotator_r --rom-configs @data/config/rom/intRot_R.yaml -v
 *   contracture_cli --max-iterations 50 --lambda-line-reg 0.2 -v
 */

#include "optimizer/ContractureOptimizer.h"
#include "SurgeryExecutor.h"
#include "Character.h"
#include "Log.h"

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <regex>
#include <cmath>

namespace po = boost::program_options;
using namespace PMuscle;

// Resolve @-prefixed paths relative to data directory
std::string resolvePath(const std::string& path, const std::string& base_dir = "") {
    if (path.empty()) return path;
    if (path[0] == '@') {
        // Get executable directory
        std::string data_dir;
        if (!base_dir.empty()) {
            data_dir = base_dir;
        } else {
            // Default to searching relative to CWD
            data_dir = ".";
        }
        return data_dir + "/" + path.substr(1);
    }
    return path;
}

void printSeedSearchResults(
    const std::vector<SeedSearchResult>& results,
    bool verbose)
{
    if (results.empty()) {
        std::cout << "No results.\n";
        return;
    }

    // Collect all trial names from the first result
    const auto& trial_names = results[0].trial_names;
    const size_t num_trials = trial_names.size();

    // Calculate column widths
    size_t group_col_width = 24;
    size_t ratio_col_width = 8;
    size_t trial_col_width = 20;

    for (const auto& res : results) {
        group_col_width = std::max(group_col_width, res.group_name.size() + 2);
    }
    for (const auto& name : trial_names) {
        trial_col_width = std::max(trial_col_width, name.size() + 2);
    }

    // Print header
    std::cout << "\n========== SEED SEARCH RESULTS ==========\n";
    std::cout << std::left << std::setw(group_col_width) << "Group"
              << std::setw(ratio_col_width) << "Ratio";
    for (const auto& name : trial_names) {
        std::cout << std::setw(trial_col_width) << name;
    }
    std::cout << "\n";

    // Print separator
    size_t total_width = group_col_width + ratio_col_width + trial_col_width * num_trials;
    std::cout << std::string(total_width, '-') << "\n";

    // Print each group's results
    for (const auto& res : results) {
        std::cout << std::left << std::setw(group_col_width) << res.group_name
                  << std::fixed << std::setprecision(2)
                  << std::setw(ratio_col_width) << res.best_ratio;

        // Print per-trial torque changes
        for (size_t ti = 0; ti < num_trials; ++ti) {
            double before = res.torques_before[ti];
            double after = res.torques_after[ti];

            std::ostringstream cell;
            if (std::abs(before) < 0.01 && std::abs(after) < 0.01) {
                cell << "-";
            } else {
                cell << std::fixed << std::setprecision(1)
                     << before << " -> " << after;
            }
            std::cout << std::setw(trial_col_width) << cell.str();
        }
        std::cout << "\n";
    }

    // Print footer
    std::cout << std::string(total_width, '=') << "\n";
    std::cout << "Best error: " << std::fixed << std::setprecision(4)
              << results[0].best_error << "\n\n";
}

// Print fiber consistency summary (shared between Ceres and tiered output)
void printFiberConsistency(const std::vector<MuscleGroupResult>& group_results) {
    std::regex fiber_re("^(.+?)(\\d+)_(l|r)$");
    std::map<std::string, std::vector<double>> fiber_groups;
    for (const auto& grp : group_results) {
        std::smatch match;
        if (std::regex_match(grp.group_name, match, fiber_re)) {
            std::string key = match[1].str() + "_" + match[3].str();
            fiber_groups[key].push_back(grp.ratio);
        }
    }

    // Remove single-fiber groups
    for (auto it = fiber_groups.begin(); it != fiber_groups.end(); ) {
        if (it->second.size() < 2)
            it = fiber_groups.erase(it);
        else
            ++it;
    }

    if (!fiber_groups.empty()) {
        std::cout << "\n--- Fiber Consistency ---\n";
        std::cout << std::left << std::setw(30) << "Base Muscle"
                  << std::setw(8) << "N"
                  << std::setw(10) << "Mean"
                  << std::setw(10) << "Std" << "\n";
        std::cout << std::string(58, '-') << "\n";

        for (const auto& [base_name, ratios] : fiber_groups) {
            double sum = 0.0;
            for (double r : ratios) sum += r;
            double mean = sum / ratios.size();
            double var = 0.0;
            for (double r : ratios) var += (r - mean) * (r - mean);
            double std_dev = std::sqrt(var / ratios.size());

            std::cout << std::left << std::setw(30) << base_name
                      << std::setw(8) << ratios.size()
                      << std::fixed << std::setprecision(4)
                      << std::setw(10) << mean
                      << std::setw(10) << std_dev << "\n";
        }
    }
}

// Print per-trial torque summary (shared between Ceres and tiered output)
void printTrialTorques(const std::vector<TrialTorqueResult>& trial_results) {
    if (trial_results.empty()) return;
    std::cout << "\n--- Per-Trial Torque ---\n";
    for (const auto& trial : trial_results) {
        std::cout << trial.trial_name << " (" << trial.joint << " DOF " << trial.dof_index << "): "
                  << "obs=" << std::fixed << std::setprecision(2) << trial.observed_torque
                  << " before=" << trial.computed_torque_before
                  << " after=" << trial.computed_torque_after << "\n";
    }
}

void printResults(const ContractureOptResult& result, bool verbose)
{
    std::cout << "\n========== OPTIMIZATION RESULTS ==========\n";
    std::cout << "Converged: " << (result.converged ? "yes" : "no")
              << " | Iterations: " << result.iterations
              << " | Final cost: " << std::fixed << std::setprecision(6) << result.final_cost << "\n";

    // Search group results (grid search phase)
    if (!result.search_group_results.empty()) {
        std::cout << "\n--- Search Groups (Grid Search) ---\n";
        size_t name_width = 30;
        for (const auto& sg : result.search_group_results) {
            name_width = std::max(name_width, sg.search_group_name.size() + 2);
        }
        std::cout << std::left << std::setw(name_width) << "Search Group"
                  << std::setw(10) << "Ratio"
                  << std::setw(12) << "Error" << "\n";
        std::cout << std::string(name_width + 22, '-') << "\n";
        for (const auto& sg : result.search_group_results) {
            std::cout << std::left << std::setw(name_width) << sg.search_group_name
                      << std::fixed << std::setprecision(4) << std::setw(10) << sg.ratio
                      << std::setw(12) << sg.best_error << "\n";
        }
    }

    // Optimization group results (Ceres phase)
    if (!result.group_results.empty()) {
        std::cout << "\n--- Optimization Groups (Ceres) ---\n";
        size_t name_width = 30;
        for (const auto& grp : result.group_results) {
            name_width = std::max(name_width, grp.group_name.size() + 2);
        }
        std::cout << std::left << std::setw(name_width) << "Group"
                  << std::setw(10) << "Ratio" << "\n";
        std::cout << std::string(name_width + 10, '-') << "\n";
        for (const auto& grp : result.group_results) {
            std::cout << std::left << std::setw(name_width) << grp.group_name
                      << std::fixed << std::setprecision(4) << std::setw(10) << grp.ratio << "\n";
        }

        printFiberConsistency(result.group_results);
    }

    if (verbose) {
        printTrialTorques(result.trial_results);
    }

    std::cout << std::string(50, '=') << "\n\n";
}

int main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> rom_config_paths;
    std::vector<std::string> group_names;
    bool verbose = false;
    bool grid_only = false;

    // Defaults (overridden by YAML, then by CLI)
    double grid_begin = 0.5;
    double grid_end = 1.3;
    double grid_interval = 0.025;
    double lambda_line_reg = 0.1;
    double lambda_ratio_reg = 0.0;
    double lambda_torque_reg = 0.0;
    int max_iterations = 100;
    double min_ratio = 0.5;
    double max_ratio = 1.2;
    int outer_iterations = 1;

    po::options_description desc("Contracture CLI");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(&config_path)
            ->default_value("@data/config/muscle_personalizer.yaml"),
         "Path to muscle_personalizer.yaml config file")
        ("rom-configs", po::value<std::vector<std::string>>(&rom_config_paths)->multitoken(),
         "List of ROM trial YAML paths (e.g., @data/config/rom/intRot_R.yaml)")
        ("groups", po::value<std::vector<std::string>>(&group_names)->multitoken(),
         "List of muscle group names (required for --grid-only)")
        ("verbose,v", po::bool_switch(&verbose),
         "Show verbose output")
        ("grid-only", po::bool_switch(&grid_only),
         "Run grid search only (default: tiered grid+Ceres)")
        ("grid-begin", po::value<double>(&grid_begin),
         "Grid search start ratio")
        ("grid-end", po::value<double>(&grid_end),
         "Grid search end ratio")
        ("grid-interval", po::value<double>(&grid_interval),
         "Grid search step interval")
        ("lambda-line-reg", po::value<double>(&lambda_line_reg),
         "Line consistency regularization lambda")
        ("lambda-ratio-reg", po::value<double>(&lambda_ratio_reg),
         "Ratio regularization lambda")
        ("lambda-torque-reg", po::value<double>(&lambda_torque_reg),
         "Torque regularization lambda")
        ("max-iterations", po::value<int>(&max_iterations),
         "Max Ceres iterations")
        ("min-ratio", po::value<double>(&min_ratio),
         "Minimum ratio bound for optimization")
        ("max-ratio", po::value<double>(&max_ratio),
         "Maximum ratio bound for optimization")
        ("outer-iterations", po::value<int>(&outer_iterations),
         "Outer iterations for biarticular convergence");

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
        std::cout << "Usage: " << argv[0] << " [options]\n\n";
        std::cout << desc << std::endl;
        std::cout << "\nExamples:\n";
        std::cout << "  # Default: tiered optimization with default ROM trials\n";
        std::cout << "  " << argv[0] << " -v\n\n";
        std::cout << "  # Explicit ROM trials\n";
        std::cout << "  " << argv[0] << " \\\n"
                  << "    --rom-configs @data/config/rom/intRot_R.yaml @data/config/rom/extRot_R.yaml \\\n"
                  << "    -v\n\n";
        std::cout << "  # Grid search only (debugging)\n";
        std::cout << "  " << argv[0] << " \\\n"
                  << "    --grid-only --groups hip_lateral_rotator_r \\\n"
                  << "    --rom-configs @data/config/rom/intRot_R.yaml -v\n\n";
        std::cout << "  # Override optimization parameters\n";
        std::cout << "  " << argv[0] << " --max-iterations 50 --lambda-line-reg 0.2 -v\n";
        return 0;
    }

    // Resolve paths
    config_path = resolvePath(config_path);
    for (auto& path : rom_config_paths) {
        path = resolvePath(path);
    }

    // Load configuration
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load config: " << e.what());
        return 1;
    }

    // Read defaults from YAML contracture_estimation section
    auto ce = config["contracture_estimation"];
    if (ce) {
        // Only apply YAML defaults when CLI arg was not explicitly provided
        if (!vm.count("max-iterations") && ce["max_iterations"])
            max_iterations = ce["max_iterations"].as<int>();
        if (!vm.count("min-ratio") && ce["min_ratio"])
            min_ratio = ce["min_ratio"].as<double>();
        if (!vm.count("max-ratio") && ce["max_ratio"])
            max_ratio = ce["max_ratio"].as<double>();
        if (!vm.count("outer-iterations") && ce["outer_iterations"])
            outer_iterations = ce["outer_iterations"].as<int>();
        if (!vm.count("lambda-ratio-reg") && ce["lambda_ratio_reg"])
            lambda_ratio_reg = ce["lambda_ratio_reg"].as<double>();
        if (!vm.count("lambda-torque-reg") && ce["lambda_torque_reg"])
            lambda_torque_reg = ce["lambda_torque_reg"].as<double>();
        if (!vm.count("lambda-line-reg") && ce["lambda_line_reg"])
            lambda_line_reg = ce["lambda_line_reg"].as<double>();

        // Grid search parameters
        if (ce["grid_search"]) {
            auto gs = ce["grid_search"];
            if (!vm.count("grid-begin") && gs["begin"])
                grid_begin = gs["begin"].as<double>();
            if (!vm.count("grid-end") && gs["end"])
                grid_end = gs["end"].as<double>();
            if (!vm.count("grid-interval") && gs["interval"])
                grid_interval = gs["interval"].as<double>();
        }
    }

    // Get paths from config
    std::string skeleton_path = "@data/skeleton/base.yaml";
    std::string muscle_path = "@data/muscle/base.yaml";
    std::string muscle_groups_path = "@data/config/muscle_groups.yaml";
    std::string rom_config_dir = "@data/config/rom";

    if (config["paths"]) {
        if (config["paths"]["skeleton_default"])
            skeleton_path = config["paths"]["skeleton_default"].as<std::string>();
        if (config["paths"]["muscle_default"])
            muscle_path = config["paths"]["muscle_default"].as<std::string>();
        if (config["paths"]["muscle_groups"])
            muscle_groups_path = config["paths"]["muscle_groups"].as<std::string>();
        if (config["paths"]["rom_config_dir"])
            rom_config_dir = config["paths"]["rom_config_dir"].as<std::string>();
    }

    // If no --rom-configs provided, use default_rom_trials from config
    if (rom_config_paths.empty() && ce && ce["default_rom_trials"]) {
        std::string dir = resolvePath(rom_config_dir);
        for (const auto& trial : ce["default_rom_trials"]) {
            std::string name = trial.as<std::string>();
            rom_config_paths.push_back(dir + "/" + name + ".yaml");
        }
        if (verbose) {
            LOG_INFO("Using " << rom_config_paths.size() << " default ROM trials from config");
        }
    }

    // Validate required arguments
    if (rom_config_paths.empty()) {
        LOG_ERROR("--rom-configs is required (or set default_rom_trials in config)");
        std::cout << desc << std::endl;
        return 1;
    }

    if (group_names.empty() && grid_only) {
        LOG_ERROR("--groups is required for --grid-only mode");
        std::cout << desc << std::endl;
        return 1;
    }

    skeleton_path = resolvePath(skeleton_path);
    muscle_path = resolvePath(muscle_path);
    muscle_groups_path = resolvePath(muscle_groups_path);

    if (verbose) {
        LOG_INFO("Config: " << config_path);
        LOG_INFO("Skeleton: " << skeleton_path);
        LOG_INFO("Muscle: " << muscle_path);
        LOG_INFO("Muscle groups: " << muscle_groups_path);
        LOG_INFO("ROM configs: " << rom_config_paths.size() << " files");
        for (const auto& p : rom_config_paths) {
            LOG_INFO("  - " << p);
        }
        LOG_INFO("Grid: " << grid_begin << " to " << grid_end << " by " << grid_interval);
        LOG_INFO("Bounds: min_ratio=" << min_ratio << " max_ratio=" << max_ratio);
        LOG_INFO("Ceres: max_iter=" << max_iterations << " outer_iter=" << outer_iterations);
        LOG_INFO("Lambda: line=" << lambda_line_reg << " ratio=" << lambda_ratio_reg
                 << " torque=" << lambda_torque_reg);
        if (grid_only) {
            LOG_INFO("Mode: grid-only");
            LOG_INFO("Groups: " << group_names.size());
            for (const auto& g : group_names) {
                LOG_INFO("  - " << g);
            }
        } else {
            LOG_INFO("Mode: tiered (grid search -> Ceres)");
        }
    }

    // Create character using SurgeryExecutor (handles skeleton + muscle loading)
    std::unique_ptr<SurgeryExecutor> executor;
    Character* character = nullptr;
    try {
        executor = std::make_unique<SurgeryExecutor>("contracture-cli");
        executor->loadCharacter(skeleton_path, muscle_path);
        character = executor->getCharacter();
        if (!character) {
            LOG_ERROR("Failed to create character");
            return 1;
        }
        if (verbose) {
            LOG_INFO("Character loaded: " << character->getMuscles().size() << " muscles");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create character: " << e.what());
        return 1;
    }

    // Create optimizer and load muscle groups
    ContractureOptimizer optimizer;
    int num_groups = optimizer.loadMuscleGroups(muscle_groups_path, character);
    if (num_groups <= 0) {
        LOG_ERROR("Failed to load muscle groups from: " << muscle_groups_path);
        return 1;
    }
    if (verbose) {
        LOG_INFO("Loaded " << num_groups << " muscle groups");
    }

    // Load grid search mapping from config
    if (ce && ce["grid_search_mapping"]) {
        std::vector<GridSearchMapping> gridSearchMapping;
        for (const auto& entry : ce["grid_search_mapping"]) {
            GridSearchMapping mapping;
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
        if (verbose) {
            LOG_INFO("Loaded " << gridSearchMapping.size() << " grid search mappings");
        }
    }

    // Load ROM trial configs
    std::vector<ROMTrialConfig> rom_configs;
    auto skeleton = character->getSkeleton();
    for (const auto& path : rom_config_paths) {
        try {
            auto rom_cfg = ContractureOptimizer::loadROMConfig(path, skeleton);
            rom_configs.push_back(rom_cfg);
            if (verbose) {
                LOG_INFO("Loaded ROM config: " << rom_cfg.name);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load ROM config " << path << ": " << e.what());
            return 1;
        }
    }

    // Configure optimization
    ContractureOptimizer::Config opt_config;
    opt_config.gridSearchBegin = grid_begin;
    opt_config.gridSearchEnd = grid_end;
    opt_config.gridSearchInterval = grid_interval;
    opt_config.verbose = verbose;
    opt_config.lambdaLineReg = lambda_line_reg;
    opt_config.lambdaRatioReg = lambda_ratio_reg;
    opt_config.lambdaTorqueReg = lambda_torque_reg;
    opt_config.maxIterations = max_iterations;
    opt_config.minRatio = min_ratio;
    opt_config.maxRatio = max_ratio;
    opt_config.outerIterations = outer_iterations;

    if (grid_only) {
        // Grid search only mode (debugging)
        LOG_INFO("Running grid search over " << group_names.size() << " groups and "
                 << rom_configs.size() << " trials...");

        auto results = optimizer.seedSearch(
            character,
            rom_configs,
            group_names,
            opt_config
        );

        if (results.empty()) {
            LOG_ERROR("Grid search returned no results");
            return 1;
        }

        printSeedSearchResults(results, verbose);
    } else {
        // Default: tiered optimization (grid search -> Ceres)
        LOG_INFO("Running tiered optimization over "
                 << rom_configs.size() << " trials...");

        auto result = optimizer.optimize(character, rom_configs, opt_config);

        printResults(result, verbose);
    }

    return 0;
}

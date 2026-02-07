/**
 * @file contracture_cli_main.cpp
 * @brief CLI tool for standalone contracture grid search optimization
 *
 * Usage:
 *   contracture_cli --config @data/config/muscle_personalizer.yaml \
 *                   --rom-configs @data/config/rom/intRot_R.yaml @data/config/rom/extRot_R.yaml \
 *                   --groups hip_lateral_rotator_r glt_minimus_r \
 *                   -v
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

void printResults(
    const std::vector<SimpleSearchResult>& results,
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
    std::cout << "\n========== GRID SEARCH RESULTS ==========\n";
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

void printCeresResults(const ContractureOptResult& result, bool verbose)
{
    std::cout << "\n========== CERES OPTIMIZATION RESULTS ==========\n";
    std::cout << "Converged: " << (result.converged ? "yes" : "no")
              << " | Iterations: " << result.iterations
              << " | Final cost: " << std::fixed << std::setprecision(6) << result.final_cost << "\n\n";

    // Group results table
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

    // Fiber consistency summary
    std::regex fiber_re("^(.+?)(\\d+)_(l|r)$");
    std::map<std::string, std::vector<double>> fiber_groups;
    for (const auto& grp : result.group_results) {
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

    // Per-trial torque summary
    if (verbose && !result.trial_results.empty()) {
        std::cout << "\n--- Per-Trial Torque ---\n";
        for (const auto& trial : result.trial_results) {
            std::cout << trial.trial_name << " (" << trial.joint << " DOF " << trial.dof_index << "): "
                      << "obs=" << std::fixed << std::setprecision(2) << trial.observed_torque
                      << " before=" << trial.computed_torque_before
                      << " after=" << trial.computed_torque_after << "\n";
        }
    }

    std::cout << std::string(50, '=') << "\n\n";
}

int main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> rom_config_paths;
    std::vector<std::string> group_names;
    bool verbose = false;
    bool use_ceres = false;

    // Grid search parameters
    double grid_begin = 0.5;
    double grid_end = 1.3;
    double grid_interval = 0.025;

    // Ceres parameters
    double lambda_line_reg = 0.1;
    double lambda_ratio_reg = 0.1;
    double lambda_torque_reg = 0.01;
    int max_iterations = 100;

    po::options_description desc("Contracture CLI");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(&config_path)
            ->default_value("@data/config/muscle_personalizer.yaml"),
         "Path to muscle_personalizer.yaml config file")
        ("rom-configs", po::value<std::vector<std::string>>(&rom_config_paths)->multitoken(),
         "List of ROM trial YAML paths (e.g., @data/config/rom/intRot_R.yaml)")
        ("groups", po::value<std::vector<std::string>>(&group_names)->multitoken(),
         "List of muscle group names to search")
        ("verbose,v", po::bool_switch(&verbose),
         "Show verbose output")
        ("grid-begin", po::value<double>(&grid_begin)->default_value(0.5),
         "Grid search start ratio")
        ("grid-end", po::value<double>(&grid_end)->default_value(1.3),
         "Grid search end ratio")
        ("grid-interval", po::value<double>(&grid_interval)->default_value(0.025),
         "Grid search step interval")
        ("ceres", po::bool_switch(&use_ceres),
         "Use Ceres optimization instead of grid search")
        ("lambda-line-reg", po::value<double>(&lambda_line_reg)->default_value(0.1),
         "Line consistency regularization lambda (Ceres mode only)")
        ("lambda-ratio-reg", po::value<double>(&lambda_ratio_reg)->default_value(0.1),
         "Ratio regularization lambda (Ceres mode only)")
        ("lambda-torque-reg", po::value<double>(&lambda_torque_reg)->default_value(0.01),
         "Torque regularization lambda (Ceres mode only)")
        ("max-iterations", po::value<int>(&max_iterations)->default_value(100),
         "Max Ceres iterations (Ceres mode only)");

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
        std::cout << "  " << argv[0] << " \\\n"
                  << "    --config @data/config/muscle_personalizer.yaml \\\n"
                  << "    --rom-configs @data/config/rom/intRot_R.yaml @data/config/rom/extRot_R.yaml \\\n"
                  << "    --groups hip_lateral_rotator_r glt_minimus_r \\\n"
                  << "    -v\n";
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
    if (rom_config_paths.empty() && config["contracture_estimation"]["default_rom_trials"]) {
        std::string dir = resolvePath(rom_config_dir);
        for (const auto& trial : config["contracture_estimation"]["default_rom_trials"]) {
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

    if (group_names.empty() && !use_ceres) {
        LOG_ERROR("--groups is required for grid search mode (not needed for --ceres)");
        std::cout << desc << std::endl;
        return 1;
    }

    if (verbose) {
        LOG_INFO("Config: " << config_path);
        LOG_INFO("ROM configs: " << rom_config_paths.size() << " files");
        for (const auto& p : rom_config_paths) {
            LOG_INFO("  - " << p);
        }
        LOG_INFO("Groups: " << group_names.size());
        for (const auto& g : group_names) {
            LOG_INFO("  - " << g);
        }
        LOG_INFO("Grid: " << grid_begin << " to " << grid_end << " by " << grid_interval);
    }

    skeleton_path = resolvePath(skeleton_path);
    muscle_path = resolvePath(muscle_path);
    muscle_groups_path = resolvePath(muscle_groups_path);

    if (verbose) {
        LOG_INFO("Skeleton: " << skeleton_path);
        LOG_INFO("Muscle: " << muscle_path);
        LOG_INFO("Muscle groups: " << muscle_groups_path);
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

    if (use_ceres) {
        // Ceres optimization mode
        opt_config.lambdaLineReg = lambda_line_reg;
        opt_config.lambdaRatioReg = lambda_ratio_reg;
        opt_config.lambdaTorqueReg = lambda_torque_reg;
        opt_config.maxIterations = max_iterations;

        LOG_INFO("Running Ceres optimization over " << group_names.size() << " groups and "
                 << rom_configs.size() << " trials...");
        if (verbose) {
            LOG_INFO("  lambda_line_reg=" << lambda_line_reg
                     << " lambda_ratio_reg=" << lambda_ratio_reg
                     << " lambda_torque_reg=" << lambda_torque_reg
                     << " max_iterations=" << max_iterations);
        }

        auto result = optimizer.optimizeWithResults(character, rom_configs, opt_config);

        printCeresResults(result, verbose);
    } else {
        // Grid search mode
        LOG_INFO("Running grid search over " << group_names.size() << " groups and "
                 << rom_configs.size() << " trials...");

        auto results = optimizer.simpleGridSearch(
            character,
            rom_configs,
            group_names,
            opt_config
        );

        if (results.empty()) {
            LOG_ERROR("Grid search returned no results");
            return 1;
        }

        printResults(results, verbose);
    }

    return 0;
}

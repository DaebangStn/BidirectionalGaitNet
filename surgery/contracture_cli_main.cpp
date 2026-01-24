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

int main(int argc, char** argv) {
    std::string config_path;
    std::vector<std::string> rom_config_paths;
    std::vector<std::string> group_names;
    bool verbose = false;

    // Grid search parameters
    double grid_begin = 0.5;
    double grid_end = 1.3;
    double grid_interval = 0.025;

    po::options_description desc("Contracture Grid Search CLI");
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
         "Grid search step interval");

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

    // Validate required arguments
    if (rom_config_paths.empty()) {
        LOG_ERROR("--rom-configs is required");
        std::cout << desc << std::endl;
        return 1;
    }

    if (group_names.empty()) {
        LOG_ERROR("--groups is required");
        std::cout << desc << std::endl;
        return 1;
    }

    // Resolve paths
    config_path = resolvePath(config_path);
    for (auto& path : rom_config_paths) {
        path = resolvePath(path);
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

    if (config["paths"]) {
        if (config["paths"]["skeleton_default"]) {
            skeleton_path = config["paths"]["skeleton_default"].as<std::string>();
        }
        if (config["paths"]["muscle_default"]) {
            muscle_path = config["paths"]["muscle_default"].as<std::string>();
        }
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

    // Configure grid search
    ContractureOptimizer::Config opt_config;
    opt_config.gridSearchBegin = grid_begin;
    opt_config.gridSearchEnd = grid_end;
    opt_config.gridSearchInterval = grid_interval;
    opt_config.verbose = verbose;

    // Run simple grid search
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

    // Print results
    printResults(results, verbose);

    return 0;
}

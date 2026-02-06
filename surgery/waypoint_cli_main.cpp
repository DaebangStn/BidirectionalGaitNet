/**
 * @file waypoint_cli_main.cpp
 * @brief CLI tool for standalone waypoint optimization debugging
 *
 * Usage:
 *   waypoint_cli --skeleton @pid:12964246/pre/skeleton/trimmed_unified.yaml \
 *                --muscle @data/muscle/base.yaml \
 *                --muscles R_Peroneus_Tertius1 \
 *                -v
 */

#include "optimizer/WaypointOptimizer.h"
#include "SurgeryExecutor.h"
#include "Character.h"
#include "Log.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>

namespace po = boost::program_options;
using namespace PMuscle;

// Resolve @-prefixed paths relative to data directory
std::string resolvePath(const std::string& path) {
    if (path.empty()) return path;
    if (path[0] == '@') {
        return "./" + path.substr(1);
    }
    return path;
}

void printResult(const WaypointOptResult& r) {
    std::cout << "\n========== " << r.muscle_name << " ==========\n";
    std::cout << "Success: " << (r.success ? "YES" : "NO") << "\n";
    std::cout << "DOF: " << r.dof_name << " (idx=" << r.dof_idx << ")\n";
    std::cout << "Iterations: " << r.num_iterations << "\n";
    std::cout << "Bound hits: " << r.num_bound_hits << "\n";
    std::cout << "\nEnergies:\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Shape:  " << r.initial_shape_energy << " -> " << r.final_shape_energy
              << " (delta: " << (r.final_shape_energy - r.initial_shape_energy) << ")\n";
    std::cout << "  Length: " << r.initial_length_energy << " -> " << r.final_length_energy
              << " (delta: " << (r.final_length_energy - r.initial_length_energy) << ")\n";
    std::cout << "  Total:  " << r.initial_total_cost << " -> " << r.final_total_cost
              << " (delta: " << (r.final_total_cost - r.initial_total_cost) << ")\n";
    std::cout << "==========================================\n";
}

int main(int argc, char** argv) {
    std::string skeleton_path;
    std::string muscle_path;
    std::string reference_skeleton_path;
    std::string reference_muscle_path;
    std::string hdf_motion_path;
    std::vector<std::string> muscle_names;
    bool verbose = false;

    // Optimizer parameters
    int max_iterations = 10000;
    int num_sampling = 10;
    double lambda_shape = 0.1;
    double lambda_length = 0.1;
    double max_displacement = 0.2;
    double max_displacement_oi = 0.03;
    bool fix_origin_insertion = true;
    bool use_normalized_length = false;

    // Length curve weights
    double weight_phase = 1.0;
    double weight_delta = 50.0;
    double weight_samples = 1.0;
    int num_phase_samples = 3;
    int loss_power = 2;
    bool adaptive_sample_weight = false;

    po::options_description desc("Waypoint Optimization CLI");
    desc.add_options()
        ("help,h", "Show help message")
        ("skeleton,s", po::value<std::string>(&skeleton_path)->required(),
         "Path to subject skeleton YAML")
        ("muscle,m", po::value<std::string>(&muscle_path)->required(),
         "Path to subject muscle YAML")
        ("ref-skeleton", po::value<std::string>(&reference_skeleton_path),
         "Path to reference skeleton YAML (default: same as subject)")
        ("ref-muscle", po::value<std::string>(&reference_muscle_path),
         "Path to reference muscle YAML (default: same as subject)")
        ("hdf-motion", po::value<std::string>(&hdf_motion_path),
         "Path to HDF motion file (sets base pose from first frame)")
        ("muscles", po::value<std::vector<std::string>>(&muscle_names)->multitoken()->required(),
         "List of muscle names to optimize")
        ("verbose,v", po::bool_switch(&verbose),
         "Show verbose output")
        ("max-iter", po::value<int>(&max_iterations)->default_value(10000),
         "Maximum iterations")
        ("num-sampling", po::value<int>(&num_sampling)->default_value(10),
         "Number of DOF samples")
        ("lambda-shape", po::value<double>(&lambda_shape)->default_value(0.1),
         "Shape energy weight")
        ("lambda-length", po::value<double>(&lambda_length)->default_value(0.1),
         "Length curve energy weight")
        ("max-disp", po::value<double>(&max_displacement)->default_value(0.2),
         "Max displacement for waypoints (m)")
        ("max-disp-oi", po::value<double>(&max_displacement_oi)->default_value(0.03),
         "Max displacement for origin/insertion (m)")
        ("no-fix-oi", po::bool_switch(),
         "Don't fix origin/insertion positions")
        ("normalized", po::bool_switch(&use_normalized_length),
         "Use normalized muscle length (lm_norm) instead of MTU length")
        ("weight-phase", po::value<double>(&weight_phase)->default_value(1.0),
         "Phase matching weight in length curve energy")
        ("weight-delta", po::value<double>(&weight_delta)->default_value(50.0),
         "Delta matching weight in length curve energy")
        ("weight-samples", po::value<double>(&weight_samples)->default_value(1.0),
         "Sample matching weight in length curve energy")
        ("num-phase-samples", po::value<int>(&num_phase_samples)->default_value(3),
         "Number of phase sample points")
        ("loss-power", po::value<int>(&loss_power)->default_value(2),
         "Loss power exponent (2=squared, 3=cube)")
        ("adaptive-sample-weight", po::bool_switch(&adaptive_sample_weight),
         "Use adaptive weighting for sample matching");

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << desc << std::endl;
            std::cout << "\nExamples:\n";
            std::cout << "  " << argv[0] << " \\\n"
                      << "    --skeleton @pid:12964246/pre/skeleton/trimmed_unified.yaml \\\n"
                      << "    --muscle @data/muscle/base.yaml \\\n"
                      << "    --muscles R_Peroneus_Tertius1 L_Gastrocnemius \\\n"
                      << "    -v\n";
            std::cout << "\n  Reproduce GUI settings (normalized, delta=5):\n";
            std::cout << "  " << argv[0] << " \\\n"
                      << "    --skeleton ... --muscle ... \\\n"
                      << "    --muscles R_Peroneus_Tertius1 \\\n"
                      << "    --lambda-shape 1.0 --lambda-length 0.2 \\\n"
                      << "    --weight-delta 5 --no-fix-oi --normalized -v\n";
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        LOG_ERROR("Command line error: " << e.what());
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("no-fix-oi")) {
        fix_origin_insertion = false;
    }

    // Use subject as reference if not specified
    if (reference_skeleton_path.empty()) {
        reference_skeleton_path = skeleton_path;
    }
    if (reference_muscle_path.empty()) {
        reference_muscle_path = muscle_path;
    }

    // Resolve paths
    skeleton_path = resolvePath(skeleton_path);
    muscle_path = resolvePath(muscle_path);
    reference_skeleton_path = resolvePath(reference_skeleton_path);
    reference_muscle_path = resolvePath(reference_muscle_path);
    hdf_motion_path = resolvePath(hdf_motion_path);

    if (verbose) {
        LOG_INFO("Subject skeleton: " << skeleton_path);
        LOG_INFO("Subject muscle: " << muscle_path);
        LOG_INFO("Reference skeleton: " << reference_skeleton_path);
        LOG_INFO("Reference muscle: " << reference_muscle_path);
        LOG_INFO("HDF motion: " << (hdf_motion_path.empty() ? "(none - zero pose)" : hdf_motion_path));
        LOG_INFO("Muscles to optimize: " << muscle_names.size());
        for (const auto& m : muscle_names) {
            LOG_INFO("  - " << m);
        }
        LOG_INFO("Config: maxIter=" << max_iterations << " numSamp=" << num_sampling
                 << " λS=" << lambda_shape << " λL=" << lambda_length
                 << " maxDisp=" << max_displacement << " maxDispOI=" << max_displacement_oi
                 << " fixOI=" << fix_origin_insertion);
        LOG_INFO("Length weights: phase=" << weight_phase << " delta=" << weight_delta
                 << " samples=" << weight_samples
                 << " phaseSamples=" << num_phase_samples
                 << " lossPower=" << loss_power
                 << " adaptive=" << adaptive_sample_weight);
    }

    // Create subject character
    std::unique_ptr<SurgeryExecutor> subject_executor;
    Character* subject_character = nullptr;
    try {
        subject_executor = std::make_unique<SurgeryExecutor>("waypoint-cli-subject");
        subject_executor->loadCharacter(skeleton_path, muscle_path);
        subject_character = subject_executor->getCharacter();
        if (!subject_character) {
            LOG_ERROR("Failed to create subject character");
            return 1;
        }
        if (verbose) {
            LOG_INFO("Subject character loaded: " << subject_character->getMuscles().size() << " muscles");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create subject character: " << e.what());
        return 1;
    }

    // Create reference character
    std::unique_ptr<SurgeryExecutor> reference_executor;
    Character* reference_character = nullptr;
    try {
        reference_executor = std::make_unique<SurgeryExecutor>("waypoint-cli-reference");
        reference_executor->loadCharacter(reference_skeleton_path, reference_muscle_path);
        reference_character = reference_executor->getCharacter();
        if (!reference_character) {
            LOG_ERROR("Failed to create reference character");
            return 1;
        }
        if (verbose) {
            LOG_INFO("Reference character loaded: " << reference_character->getMuscles().size() << " muscles");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create reference character: " << e.what());
        return 1;
    }

    // Configure optimizer
    WaypointOptimizer::Config config;
    config.maxIterations = max_iterations;
    config.numSampling = num_sampling;
    config.lambdaShape = lambda_shape;
    config.lambdaLengthCurve = lambda_length;
    config.maxDisplacement = max_displacement;
    config.maxDisplacementOriginInsertion = max_displacement_oi;
    config.fixOriginInsertion = fix_origin_insertion;
    config.verbose = verbose;
    config.lengthType = use_normalized_length ? LengthCurveType::NORMALIZED : LengthCurveType::MTU_LENGTH;
    config.weightPhase = weight_phase;
    config.weightDelta = weight_delta;
    config.weightSamples = weight_samples;
    config.numPhaseSamples = num_phase_samples;
    config.lossPower = loss_power;
    config.adaptiveSampleWeight = adaptive_sample_weight;

    // Run optimization
    LOG_INFO("Running waypoint optimization for " << muscle_names.size() << " muscle(s)...");

    auto results = subject_executor->optimizeWaypointsWithResults(
        muscle_names,
        hdf_motion_path,
        config,
        reference_character,
        nullptr,  // characterMutex
        nullptr   // resultCallback
    );

    // Print results
    int success_count = 0;
    for (const auto& r : results) {
        printResult(r);
        if (r.success) success_count++;
    }

    std::cout << "\n========== SUMMARY ==========\n";
    std::cout << "Total: " << results.size() << " muscles\n";
    std::cout << "Success: " << success_count << "\n";
    std::cout << "Failed: " << (results.size() - success_count) << "\n";
    std::cout << "=============================\n";

    return (success_count == static_cast<int>(results.size())) ? 0 : 1;
}

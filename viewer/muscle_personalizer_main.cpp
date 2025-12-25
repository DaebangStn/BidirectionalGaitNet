#include "MusclePersonalizerApp.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    std::string configPath = "@data/config/muscle_personalizer.yaml";

    // Parse command line arguments
    po::options_description desc("Muscle Personalizer - Interactive muscle configuration tool");
    desc.add_options()
        ("help,h", "Show this help message")
        ("config,c", po::value<std::string>(&configPath)->default_value(configPath),
         "Path to configuration YAML file");

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

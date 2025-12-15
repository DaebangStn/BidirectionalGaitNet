#include "MotionEditorApp.h"
#include "Log.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    std::string configPath = "data/rm_config.yaml";

    // Parse command line arguments
    po::options_description desc("Motion Editor - H5 Motion Trimming Tool");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(&configPath)->default_value(configPath),
            "Resource manager config file path")
    ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        std::cout << "\nKey bindings:" << std::endl;
        std::cout << "  SPACE    - Play/Pause toggle" << std::endl;
        std::cout << "  S        - Step single frame forward" << std::endl;
        std::cout << "  R        - Reset playback to start" << std::endl;
        std::cout << "  [        - Set trim start to current frame" << std::endl;
        std::cout << "  ]        - Set trim end to current frame" << std::endl;
        std::cout << "  1/2/3    - Align camera to XY/YZ/ZX plane" << std::endl;
        std::cout << "  F        - Toggle camera follow skeleton" << std::endl;
        std::cout << "  O        - Cycle render mode (Primitive/Wire)" << std::endl;
        std::cout << "  ESC      - Close application" << std::endl;
        return 0;
    }

    LOG_INFO("[MotionEditor] Starting Motion Editor");
    LOG_INFO("[MotionEditor] Config: " << configPath);

    try {
        MotionEditorApp app(configPath);
        app.startLoop();
    } catch (const std::exception& e) {
        LOG_ERROR("[MotionEditor] Fatal error: " << e.what());
        return 1;
    }

    return 0;
}

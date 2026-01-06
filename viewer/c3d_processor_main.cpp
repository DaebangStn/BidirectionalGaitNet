#include "C3DProcessorApp.h"
#include "Log.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    po::options_description desc("C3D Processor - Standalone C3D motion capture processing application\n\nOptions");
    desc.add_options()
        ("help,h", "Show this help message")
        ("config", po::value<std::string>()->default_value("@data/config/c3d_processor.yaml"),
            "Configuration YAML path (supports @data/ URI scheme)");

    // Allow positional argument for config path
    po::positional_options_description positional;
    positional.add("config", 1);

    po::variables_map vm;

    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .positional(positional)
                      .run(), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            std::cout << "\nUsage examples:\n"
                      << "  c3d_processor                                    # Uses default config\n"
                      << "  c3d_processor @data/config/c3d_processor.yaml    # Positional argument\n"
                      << "  c3d_processor @pid:12345/gait/pre/c3d_config.yaml\n"
                      << "\nControls:\n"
                      << "  Space         Play/Pause\n"
                      << "  R             Reset to frame 0\n"
                      << "  L             Toggle marker labels\n"
                      << "  F             Toggle camera follow\n"
                      << "  Left drag     Rotate camera\n"
                      << "  Right drag    Pan camera\n"
                      << "  Scroll        Zoom\n"
                      << "  1/2/3         Align camera to XY/YZ/ZX plane\n"
                      << "  C             Reset camera position\n"
                      << "  ESC           Quit\n"
                      << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const po::required_option& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        std::cerr << desc << std::endl;
        return 1;
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        std::cerr << desc << std::endl;
        return 1;
    }

    std::string configPath = vm["config"].as<std::string>();

    LOG_INFO("[C3DProcessor] Starting with config: " << configPath);

    try {
        C3DProcessorApp app(configPath);
        app.startLoop();
    } catch (const std::exception& e) {
        LOG_ERROR("[C3DProcessor] Fatal error: " << e.what());
        return 1;
    }

    return 0;
}

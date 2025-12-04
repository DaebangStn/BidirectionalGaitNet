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
        ("skeleton,s", po::value<std::string>()->default_value("@data/skeleton/base_v2.xml"),
            "Skeleton XML path (supports @data/ URI scheme)")
        ("marker,m", po::value<std::string>()->default_value("@data/marker/default.xml"),
            "Marker configuration XML/YAML path (supports @data/ URI scheme)")
        ("config,c", po::value<std::string>()->default_value("@data/config/skeleton_fitting.yaml"),
            "Skeleton fitting config YAML path (supports @data/ URI scheme)");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            std::cout << "\nUsage examples:\n"
                      << "  c3d_processor                    # Uses default skeleton, marker, and fitting config\n"
                      << "  c3d_processor -m data/marker/default.yaml\n"
                      << "  c3d_processor -c @data/config/skeleton_fitting.yaml\n"
                      << "  c3d_processor -s @data/skeleton/base_v2.xml\n"
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

    std::string skeletonPath = vm["skeleton"].as<std::string>();
    std::string markerPath = vm["marker"].as<std::string>();
    std::string configPath = vm["config"].as<std::string>();

    LOG_INFO("[C3DProcessor] Starting with skeleton: " << skeletonPath << ", marker: " << markerPath << ", config: " << configPath);

    try {
        C3DProcessorApp app(skeletonPath, markerPath, configPath);
        app.startLoop();
    } catch (const std::exception& e) {
        LOG_ERROR("[C3DProcessor] Fatal error: " << e.what());
        return 1;
    }

    return 0;
}

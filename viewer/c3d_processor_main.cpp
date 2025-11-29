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
        ("skeleton,s", po::value<std::string>()->default_value("@data/skeleton/base.xml"),
            "Skeleton XML path (supports @data/ URI scheme)")
        ("marker,m", po::value<std::string>()->required(),
            "Marker configuration XML/YAML path (required, supports @data/ URI scheme)");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            std::cout << "\nUsage examples:\n"
                      << "  c3d_processor -m data/marker/skeleton_fitting.yaml\n"
                      << "  c3d_processor -s @data/skeleton/custom.xml -m @data/marker/config.xml\n"
                      << "\nControls:\n"
                      << "  Space         Play/Pause\n"
                      << "  R             Reset to frame 0\n"
                      << "  L             Toggle marker labels\n"
                      << "  Left drag     Rotate camera\n"
                      << "  Right drag    Pan camera\n"
                      << "  Scroll        Zoom\n"
                      << "  1             Align camera to XY plane\n"
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

    LOG_INFO("[C3DProcessor] Starting with skeleton: " << skeletonPath << ", marker: " << markerPath);

    try {
        C3DProcessorApp app(skeletonPath, markerPath);
        app.startLoop();
    } catch (const std::exception& e) {
        LOG_ERROR("[C3DProcessor] Fatal error: " << e.what());
        return 1;
    }

    return 0;
}

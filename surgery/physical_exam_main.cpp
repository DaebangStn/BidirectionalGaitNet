#include "PhysicalExam.h"
#include "Log.h"
#include <pybind11/embed.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

namespace py = pybind11;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string config_path;

    po::options_description desc("Physical Exam Options");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(&config_path)->default_value("@data/config/base.yaml"),
         "Exam setting config file");

    po::positional_options_description pos;
    pos.add("config", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .positional(pos)
                      .run(), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        LOG_ERROR("Command line error: " << e.what());
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << "Usage: " << argv[0] << " [config_file]\n\n";
        std::cout << desc << std::endl;
        std::cout << "\nExamples:\n";
        std::cout << "  " << argv[0] << "                                    # Uses default @data/config/base.yaml\n";
        std::cout << "  " << argv[0] << " @data/config/knee_extension_exam.yaml\n";
        return 0;
    }

    // Initialize Python interpreter
    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    py::module_::import("numpy");

    // Create and initialize physical examination
    PMuscle::PhysicalExam exam(1920, 1080);
    exam.initialize();

    LOG_INFO("Loading exam setting from config: " << config_path);

    try {
        exam.loadExamSetting(config_path);
        LOG_INFO("Exam setting loaded. Starting in paused state.");
        LOG_INFO("Use 'Start Next Trial' button to begin trials.");
        exam.mainLoop();
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading exam setting: " << e.what());
        return 1;
    }

    return 0;
}

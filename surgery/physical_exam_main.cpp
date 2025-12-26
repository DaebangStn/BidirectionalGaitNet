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
    std::string output_dir;
    bool headless_mode = false;

    po::options_description desc("Physical Exam Options");
    desc.add_options()
        ("help,h", "Show help message")
        ("config,c", po::value<std::string>(&config_path)->default_value("@data/config/phys_exam.yaml"),
         "Exam setting config file")
        ("output-dir,o", po::value<std::string>(&output_dir)->default_value("./results"),
         "Output directory for HDF5 results")
        ("headless", po::bool_switch(&headless_mode),
         "Run all trials without GUI and exit");

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
        std::cout << "  " << argv[0] << "                                    # Uses default config\n";
        std::cout << "  " << argv[0] << " @data/config/phys_exam.yaml\n";
        std::cout << "  " << argv[0] << " @data/config/phys_exam.yaml -o ./my_results\n";
        std::cout << "  " << argv[0] << " --headless @data/config/phys_exam.yaml\n";
        return 0;
    }

    // Initialize Python interpreter
    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    py::module_::import("numpy");

    // Create physical examination (ViewerAppBase handles GLFW/ImGui init)
    PMuscle::PhysicalExam exam(1920, 1080);

    LOG_INFO("Loading exam setting from config: " << config_path);
    LOG_INFO("Output directory: " << output_dir);

    try {
        exam.setOutputDir(output_dir);
        exam.loadExamSetting(config_path);

        if (headless_mode) {
            LOG_INFO("Running in headless mode - executing all trials...");
            exam.runAllTrials();
            LOG_INFO("Headless mode completed.");
        } else {
            LOG_INFO("Exam setting loaded. Starting in paused state.");
            LOG_INFO("Use 'Start Next Trial' button to begin trials.");
            exam.startLoop();  // ViewerAppBase handles main loop
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading exam setting: " << e.what());
        return 1;
    }

    return 0;
}

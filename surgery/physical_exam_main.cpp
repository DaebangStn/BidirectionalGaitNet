#include "PhysicalExam.h"
#include "Log.h"
#include <pybind11/embed.h>
#include <iostream>
#include <string>

namespace py = pybind11;

void printUsage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <exam_setting.yaml>\n\n";
    std::cout << "Examples:\n";
    std::cout << "  With trials:    " << prog << " @data/config/knee_extension_exam.yaml\n";
    std::cout << "  Interactive:    " << prog << " @data/config/physical_exam_example.yaml\n";
}

int main(int argc, char** argv) {
    // Initialize Python interpreter
    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    py::module_::import("numpy");
    // Create and initialize physical examination
    PMuscle::PhysicalExam exam(1920, 1080);
    exam.initialize();

    // Parse command line arguments
    if (argc == 2) {
        // Trial mode: physical_exam <exam_setting.yaml>
        std::string config_path = argv[1];
        LOG_INFO("Loading exam setting from config: " << config_path);

        try {
            exam.loadExamSetting(config_path);
            LOG_INFO("Exam setting loaded. Starting in paused state.");
            LOG_INFO("Use 'Start Next Trial' button to begin trials.");
            exam.mainLoop();  // Start interactive loop with trials
        } catch (const std::exception& e) {
            LOG_ERROR("Error loading exam setting: " << e.what());
            return 1;
        }
    }
    else {
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}

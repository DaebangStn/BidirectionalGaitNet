#include "PhysicalExam.h"
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
    pybind11::print("[Python] sys.path:", sys.attr("path"));
    pybind11::print("[Python] interpreter path:", sys.attr("executable"));

    // Create and initialize physical examination
    PMuscle::PhysicalExam exam(1920, 1080);
    exam.initialize();

    // Parse command line arguments
    if (argc == 2) {
        // Trial mode: physical_exam <exam_setting.yaml>
        std::string config_path = argv[1];
        std::cout << "Loading exam setting from config: " << config_path << std::endl;

        try {
            exam.loadExamSetting(config_path);
            std::cout << "Exam setting loaded. Starting in paused state." << std::endl;
            std::cout << "Use 'Start Next Trial' button to begin trials." << std::endl;
            exam.mainLoop();  // Start interactive loop with trials
        } catch (const std::exception& e) {
            std::cerr << "Error loading exam setting: " << e.what() << std::endl;
            return 1;
        }
    }
    else {
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}

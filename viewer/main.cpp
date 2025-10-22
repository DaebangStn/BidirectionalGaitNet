#include "GLFWApp.h"
#include "Log.h"
#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char **argv)
{
    LOG_PRINT_LEVEL();  // Print current log level at startup

    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    py::module_::import("numpy");
    GLFWApp app(argc, argv);
    app.startLoop();

    return -1;
}
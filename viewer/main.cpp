#include "GLFWApp.h"
#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char **argv)
{
    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    py::module_::import("numpy");
    pybind11::print("[Python] sys.path:", sys.attr("path"));
    pybind11::print("[Python] interpreter path:", sys.attr("executable"));

    GLFWApp app(argc, argv);
    app.startLoop();

    return -1;
}
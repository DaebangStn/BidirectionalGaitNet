#include "RenderCkpt.h"
#include "Log.h"
#include <pybind11/embed.h>
#include <csignal>
#include <cstdlib>

namespace py = pybind11;

static volatile sig_atomic_t g_interrupted = 0;

void signalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM) {
        g_interrupted = 1;
        std::exit(0);
    }
}

int main(int argc, char **argv)
{
    // Set up signal handler BEFORE Python interpreter
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    LOG_PRINT_LEVEL();  // Print current log level at startup

    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    py::module_::import("numpy");
    RenderCkpt app(argc, argv);
    app.startLoop();

    return -1;
}

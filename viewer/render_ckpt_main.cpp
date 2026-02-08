#include "RenderCkpt.h"
#include "Log.h"
#include <csignal>
#include <cstdlib>


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
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    LOG_PRINT_LEVEL();  // Print current log level at startup

    RenderCkpt app(argc, argv);
    app.startLoop();

    return -1;
}

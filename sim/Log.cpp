#include "Log.h"

static int g_logLevel = LOG_LEVEL_INFO;

int getLogLevelImpl() { return g_logLevel; }
void setLogLevelImpl(int level) { g_logLevel = level; }

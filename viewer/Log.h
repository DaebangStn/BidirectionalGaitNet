#ifndef __LOG_H__
#define __LOG_H__

#include <iostream>

// ============================================================================
// Log Level Control System
// ============================================================================
// Define log levels (higher number = more verbose)
#define LOG_LEVEL_SILENT  0  // No logs
#define LOG_LEVEL_WARN    1  // Warnings only
#define LOG_LEVEL_INFO    2  // Info + warnings (default)
#define LOG_LEVEL_VERBOSE 3  // All details including frame counts and parameters

// Set current log level (change this line to control verbosity)
#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

// Logging macros - automatically filtered by log level
#define LOG_VERBOSE(msg) do { if(LOG_LEVEL >= LOG_LEVEL_VERBOSE) { std::cout << msg << std::endl; } } while(0)
#define LOG_INFO(msg)    do { if(LOG_LEVEL >= LOG_LEVEL_INFO)    { std::cout << msg << std::endl; } } while(0)
#define LOG_WARN(msg)    do { if(LOG_LEVEL >= LOG_LEVEL_WARN)    { std::cerr << msg << std::endl; } } while(0)

// Helper function to get current log level as string
inline const char* getLogLevelName() {
    #if LOG_LEVEL == LOG_LEVEL_SILENT
        return "SILENT";
    #elif LOG_LEVEL == LOG_LEVEL_WARN
        return "WARN";
    #elif LOG_LEVEL == LOG_LEVEL_INFO
        return "INFO";
    #elif LOG_LEVEL == LOG_LEVEL_VERBOSE
        return "VERBOSE";
    #else
        return "UNKNOWN";
    #endif
}

// Print current log level (call once at startup)
#define LOG_PRINT_LEVEL() do { std::cout << "[Log] Level: " << getLogLevelName() << " (" << LOG_LEVEL << ")" << std::endl; } while(0)

// Usage:
// - Set LOG_LEVEL to LOG_LEVEL_SILENT to suppress all output
// - Set LOG_LEVEL to LOG_LEVEL_WARN to only see warnings
// - Set LOG_LEVEL to LOG_LEVEL_INFO for normal operation (default)
// - Set LOG_LEVEL to LOG_LEVEL_VERBOSE for detailed debugging
//
// You can override LOG_LEVEL by defining it before including this header:
// #define LOG_LEVEL LOG_LEVEL_WARN
// #include "Log.h"
// ============================================================================

#endif

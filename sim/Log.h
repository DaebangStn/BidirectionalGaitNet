#ifndef __LOG_H__
#define __LOG_H__

#include <iostream>

// ============================================================================
// Log Level Control System
// ============================================================================
// Define log levels (higher number = more verbose)
#define LOG_LEVEL_SILENT  0  // No logs
#define LOG_LEVEL_ERROR   1  // Errors only
#define LOG_LEVEL_WARN    2  // Warnings + errors
#define LOG_LEVEL_INFO    3  // Info + warnings + errors (default)
#define LOG_LEVEL_VERBOSE 4  // All details including frame counts and parameters

// Compile-time maximum log level (can be overridden with -DLOG_LEVEL_MAX=...)
#ifndef LOG_LEVEL_MAX
#define LOG_LEVEL_MAX LOG_LEVEL_VERBOSE
#endif

// Runtime log level (defaults to INFO, can be changed at runtime)
// Defined in Log.cpp to ensure single instance across shared libraries
int getLogLevelImpl();
void setLogLevelImpl(int level);

inline int getLogLevel() { return getLogLevelImpl(); }
inline void setLogLevel(int level) { setLogLevelImpl(level); }

// Logging macros - filtered by both compile-time max and runtime level
#define LOG_VERBOSE(msg) do { if(LOG_LEVEL_MAX >= LOG_LEVEL_VERBOSE && getLogLevel() >= LOG_LEVEL_VERBOSE) { std::cout << msg << std::endl; } } while(0)
#define LOG_INFO(msg)    do { if(LOG_LEVEL_MAX >= LOG_LEVEL_INFO    && getLogLevel() >= LOG_LEVEL_INFO)    { std::cout << msg << std::endl; } } while(0)
#define LOG_WARN(msg)    do { if(LOG_LEVEL_MAX >= LOG_LEVEL_WARN    && getLogLevel() >= LOG_LEVEL_WARN)    { std::cerr << msg << std::endl; } } while(0)
#define LOG_ERROR(msg)   do { if(LOG_LEVEL_MAX >= LOG_LEVEL_ERROR   && getLogLevel() >= LOG_LEVEL_ERROR)   { std::cerr << "[ERROR] " << msg << std::endl; } } while(0)

// Helper function to get current log level as string
inline const char* getLogLevelName() {
    switch (getLogLevel()) {
        case LOG_LEVEL_SILENT:  return "SILENT";
        case LOG_LEVEL_ERROR:   return "ERROR";
        case LOG_LEVEL_WARN:    return "WARN";
        case LOG_LEVEL_INFO:    return "INFO";
        case LOG_LEVEL_VERBOSE: return "VERBOSE";
        default:                return "UNKNOWN";
    }
}

// Print current log level (call once at startup)
#define LOG_PRINT_LEVEL() do { std::cout << "[Log] Level: " << getLogLevelName() << " (" << getLogLevel() << ")" << std::endl; } while(0)

// ============================================================================

#endif

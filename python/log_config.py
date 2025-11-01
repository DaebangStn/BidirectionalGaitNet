"""
Python Log Level Configuration
Mirrors the C++ log system in viewer/Log.h
"""

import os

# Log levels (match C++ definitions)
LOG_LEVEL_SILENT = 0
LOG_LEVEL_WARN = 1
LOG_LEVEL_INFO = 2
LOG_LEVEL_VERBOSE = 3

# Read log level from environment variable or use default
_LOG_LEVEL = int(os.environ.get('LOG_LEVEL', LOG_LEVEL_INFO))

def get_log_level():
    """Get current log level"""
    return _LOG_LEVEL

def set_log_level(level):
    """Set log level programmatically"""
    global _LOG_LEVEL
    _LOG_LEVEL = level

def get_log_level_name():
    """Get log level name as string"""
    level_names = {
        LOG_LEVEL_SILENT: "SILENT",
        LOG_LEVEL_WARN: "WARN",
        LOG_LEVEL_INFO: "INFO",
        LOG_LEVEL_VERBOSE: "VERBOSE"
    }
    return level_names.get(_LOG_LEVEL, "UNKNOWN")

def log_verbose(msg):
    """Log verbose message (level 3+)"""
    if _LOG_LEVEL >= LOG_LEVEL_VERBOSE:
        print(msg)

def log_info(msg):
    """Log info message (level 2+)"""
    if _LOG_LEVEL >= LOG_LEVEL_INFO:
        print(msg)

def log_warn(msg):
    """Log warning message (level 1+)"""
    if _LOG_LEVEL >= LOG_LEVEL_WARN:
        print(msg, file=__import__('sys').stderr)

def log_error(msg):
    """Log error message (always printed to stderr)"""
    print(msg, file=__import__('sys').stderr)

def print_log_level():
    """Print current log level at startup"""
    print(f"[Log] Python Level: {get_log_level_name()} ({_LOG_LEVEL})")

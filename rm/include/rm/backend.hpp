#pragma once

#include "rm/handle.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rm {

// Abstract backend interface
// Implementations provide access to resources from different sources (local, FTP, etc.)
class Backend {
public:
    virtual ~Backend() = default;

    // Human-readable name for logging/debugging
    virtual std::string name() const = 0;

    // Whether results from this backend should be cached by ResourceManager
    // Returns true for remote backends (FTP), false for local filesystem
    virtual bool cached() const = 0;

    // Whether this backend is currently available/operational
    // For local backends: returns true if root directory exists
    // For remote backends: returns true (connectivity checked per-request)
    // Used to determine if fallback to next backend should occur
    virtual bool isAvailable() const = 0;

    // Check if a resource exists at the given path
    // Path is relative to the backend's root
    virtual bool exists(const std::string& path) = 0;

    // Check if a directory exists at the given path
    // Path is relative to the backend's root
    // Default implementation returns false (override for backends that support directories)
    virtual bool existsDir(const std::string& path) { return false; }

    // Resolve path to absolute filesystem path (for local backends)
    // Returns empty path if not supported or path doesn't exist
    virtual std::filesystem::path resolvePath(const std::string& path) { return {}; }

    // Fetch a resource, returning a handle to its data
    // Throws RMError on failure
    virtual ResourceHandle fetch(const std::string& path) = 0;

    // List resources matching a pattern (glob-style)
    // Returns relative paths within the backend
    virtual std::vector<std::string> list(const std::string& pattern) = 0;
};;

} // namespace rm

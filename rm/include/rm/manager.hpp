#pragma once

#include "rm/backend.hpp"
#include "rm/handle.hpp"
#include "rm/uri.hpp"
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rm {

// Resource Manager with prefix-based routing
// Supports URI prefixes like @data, @pid:id for routing to specific backends
// Caches results from remote backends (where cached() == true)
//
// Config format:
//   cache_dir: .tmp/rm_cache  # optional
//
//   backends:
//     project_data:
//       type: local
//       root: ./data
//     pid_local:
//       type: local
//       root: /mnt/blue8T/CP/RM
//     pid_ftp:
//       type: ftp
//       host: gait
//       ip: 192.168.x.x
//       username: user
//       password: pass
//
//   routes:
//     "@data":
//       backends: [project_data]
//     "@pid":
//       backends: [pid_local, pid_ftp]  # fallback chain
class ResourceManager {
public:
    // Create from YAML config file
    explicit ResourceManager(const std::string& config_path);

    // Destructor
    ~ResourceManager();

    // Non-copyable, non-moveable (mutex is not moveable)
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;
    ResourceManager(ResourceManager&&) = delete;
    ResourceManager& operator=(ResourceManager&&) = delete;

    // Check if resource exists
    // Supports prefix routing: @data/path, @pid:id/path
    bool exists(const std::string& uri_str);

    // Fetch resource using prefix routing
    // @data/path -> routes to data backend only
    // @pid:id/path -> routes to pid backends (local, then ftp fallback)
    // plain/path -> uses default local backend relative to config dir
    // For cached backends: checks cache first, writes to cache on miss
    // Throws RMError if not found
    ResourceHandle fetch(const std::string& uri_str);

    // List resources matching pattern
    // Returns unique paths (deduped across backends)
    std::vector<std::string> list(const std::string& pattern);

    // Clear cached files
    void clear_cache();

    // Get number of configured backends
    size_t backend_count() const { return named_backends_.size(); }

    // Resolve URI to full filesystem path (using first available backend)
    // Returns empty path if resource not found
    std::filesystem::path resolve(const std::string& uri_str);

    // Get backend names for a URI (based on prefix routing)
    std::vector<std::string> resolve_backend_names(const std::string& uri_str);

private:
    void load_config(const std::string& config_path);

    // Backend resolution for a URI
    std::vector<Backend*> resolve_backends(const URI& uri);

    // Cache management
    std::filesystem::path cache_path_for(const std::string& uri) const;
    void write_cache(const std::string& uri, const ResourceHandle& handle);

    // Named backends (from config)
    std::unordered_map<std::string, std::unique_ptr<Backend>> named_backends_;

    // Route mapping: prefix -> ordered list of backend names
    std::unordered_map<std::string, std::vector<std::string>> routes_;

    // Default backend for non-prefixed URIs (local relative to config dir)
    std::unique_ptr<Backend> default_backend_;

    std::filesystem::path cache_dir_;  // .tmp/rm_cache/
    std::filesystem::path config_dir_; // directory containing config file
    mutable std::mutex cache_mutex_;
};

} // namespace rm

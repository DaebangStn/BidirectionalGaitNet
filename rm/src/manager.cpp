#include "rm/manager.hpp"
#include "rm/error.hpp"
#include "rm/backends/local.hpp"
#include "rm/backends/ftp.hpp"
#include "rm/backends/pid.hpp"
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <fstream>
#include <iostream>

namespace rm {

ResourceManager::ResourceManager(const std::string& config_path) {
    load_config(config_path);
}

ResourceManager::~ResourceManager() = default;

void ResourceManager::load_config(const std::string& config_path) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        throw RMError(ErrorCode::ConfigError, config_path,
            std::string("Failed to load config: ") + e.what());
    }

    // Store config directory for relative path resolution
    config_dir_ = std::filesystem::path(config_path).parent_path();
    if (config_dir_.empty()) {
        config_dir_ = ".";
    }
    config_dir_ = std::filesystem::weakly_canonical(config_dir_);

    // Determine cache directory - default to .tmp/rm_cache
    if (config["cache_dir"]) {
        cache_dir_ = config["cache_dir"].as<std::string>();
        if (cache_dir_.is_relative()) {
            cache_dir_ = config_dir_ / cache_dir_;
        }
    } else {
        cache_dir_ = config_dir_ / ".." / ".tmp" / "rm_cache";
    }
    cache_dir_ = std::filesystem::weakly_canonical(cache_dir_);

    // Create cache directory
    if (!std::filesystem::exists(cache_dir_)) {
        std::filesystem::create_directories(cache_dir_);
    }

    std::cout << "[rm] Config directory: " << config_dir_ << std::endl;
    std::cout << "[rm] Cache directory: " << cache_dir_ << std::endl;

    // Load named backends
    if (config["backends"]) {
        for (const auto& item : config["backends"]) {
            std::string name = item.first.as<std::string>();
            const auto& backend_config = item.second;
            std::string type = backend_config["type"].as<std::string>();

            if (type == "local") {
                std::string root = backend_config["root"].as<std::string>();
                std::filesystem::path root_path(root);
                if (root_path.is_relative()) {
                    root_path = config_dir_ / root_path;
                }

                try {
                    named_backends_[name] = std::make_unique<LocalBackend>(root_path);
                    std::cout << "[rm] Backend '" << name << "': local " << root_path << std::endl;
                } catch (const RMError& e) {
                    std::cerr << "[rm] Warning: Failed to add backend '" << name << "': " << e.what() << std::endl;
                }

            } else if (type == "pid") {
                std::string root = backend_config["root"].as<std::string>();
                std::filesystem::path root_path(root);
                if (root_path.is_relative()) {
                    root_path = config_dir_ / root_path;
                }

                try {
                    named_backends_[name] = std::make_unique<PidBackend>(root_path);
                    std::cout << "[rm] Backend '" << name << "': pid " << root_path << std::endl;
                } catch (const RMError& e) {
                    std::cerr << "[rm] Warning: Failed to add backend '" << name << "': " << e.what() << std::endl;
                }

            } else if (type == "ftp") {
                FTPConfig ftp_config;
                ftp_config.host = backend_config["host"].as<std::string>();
                ftp_config.ip = backend_config["ip"].as<std::string>();
                ftp_config.username = backend_config["username"].as<std::string>();
                ftp_config.password = backend_config["password"].as<std::string>();
                ftp_config.port = backend_config["port"].as<int>(21);
                ftp_config.root = backend_config["root"].as<std::string>("");

                named_backends_[name] = std::make_unique<FTPBackend>(ftp_config);
                std::cout << "[rm] Backend '" << name << "': ftp " << ftp_config.host
                          << " (" << ftp_config.ip << ":" << ftp_config.port << ")" << std::endl;

            } else {
                std::cerr << "[rm] Warning: Unknown backend type '" << type << "' for '" << name << "'" << std::endl;
            }
        }
    }

    // Load routes
    if (config["routes"]) {
        for (const auto& item : config["routes"]) {
            std::string prefix = item.first.as<std::string>();
            const auto& route_config = item.second;

            std::vector<std::string> backend_names;
            if (route_config["backends"]) {
                for (const auto& backend_name : route_config["backends"]) {
                    std::string name = backend_name.as<std::string>();
                    if (named_backends_.find(name) == named_backends_.end()) {
                        std::cerr << "[rm] Warning: Route '" << prefix << "' references unknown backend '" << name << "'" << std::endl;
                    } else {
                        backend_names.push_back(name);
                    }
                }
            }

            if (!backend_names.empty()) {
                routes_[prefix] = std::move(backend_names);
                std::cout << "[rm] Route '" << prefix << "' -> [";
                for (size_t i = 0; i < routes_[prefix].size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << routes_[prefix][i];
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    // Create default backend for non-prefixed URIs (local relative to config dir)
    try {
        default_backend_ = std::make_unique<LocalBackend>(config_dir_);
        std::cout << "[rm] Default backend: local " << config_dir_ << std::endl;
    } catch (const RMError& e) {
        std::cerr << "[rm] Warning: Failed to create default backend: " << e.what() << std::endl;
    }

    std::cout << "[rm] Initialized with " << named_backends_.size() << " backend(s) and "
              << routes_.size() << " route(s)" << std::endl;
}

std::vector<Backend*> ResourceManager::resolve_backends(const URI& uri) {
    std::vector<Backend*> result;

    if (uri.has_prefix()) {
        // Look up route for this prefix
        auto it = routes_.find(uri.prefix());
        if (it != routes_.end()) {
            for (const auto& name : it->second) {
                auto backend_it = named_backends_.find(name);
                if (backend_it != named_backends_.end()) {
                    result.push_back(backend_it->second.get());
                }
            }
        }

        if (result.empty()) {
            throw RMError(ErrorCode::ConfigError, uri.to_string(),
                "No route configured for prefix '" + uri.prefix() + "'");
        }
    } else {
        // Non-prefixed URI: use default backend
        if (default_backend_) {
            result.push_back(default_backend_.get());
        }
    }

    return result;
}

std::filesystem::path ResourceManager::cache_path_for(const std::string& uri) const {
    // Convert URI to safe filename
    std::string safe_name;
    safe_name.reserve(uri.size());

    for (char c : uri) {
        if (c == '/' || c == '\\' || c == ':') {
            safe_name += '_';
        } else {
            safe_name += c;
        }
    }

    // Remove leading underscores
    size_t start = safe_name.find_first_not_of('_');
    if (start != std::string::npos) {
        safe_name = safe_name.substr(start);
    }

    return cache_dir_ / safe_name;
}

void ResourceManager::write_cache(const std::string& uri, const ResourceHandle& handle) {
    auto cache_file = cache_path_for(uri);

    // Create parent directory if needed
    auto parent = cache_file.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }

    // Write data to cache file
    std::ofstream out(cache_file, std::ios::binary);
    if (!out) {
        std::cerr << "[rm] Warning: Failed to write cache file: " << cache_file << std::endl;
        return;
    }

    const auto& data = handle.data();
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

void ResourceManager::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(cache_dir_, ec)) {
        if (entry.is_regular_file()) {
            std::filesystem::remove(entry.path(), ec);
        }
    }

    std::cout << "[rm] Cache cleared" << std::endl;
}

bool ResourceManager::exists(const std::string& uri_str) {
    URI uri = URI::parse(uri_str);
    std::string resolved = uri.resolved_path();

    // Check cache first
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (std::filesystem::exists(cache_path_for(uri_str))) {
            return true;
        }
    }

    // Check backends
    auto backends = resolve_backends(uri);
    for (auto* backend : backends) {
        if (backend->exists(resolved)) {
            return true;
        }
    }
    return false;
}

ResourceHandle ResourceManager::fetch(const std::string& uri_str) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    URI uri = URI::parse(uri_str);
    std::string resolved = uri.resolved_path();

    // Check cache first
    auto cached_path = cache_path_for(uri_str);
    if (std::filesystem::exists(cached_path)) {
        return ResourceHandle(cached_path, false);  // Don't own cached files
    }

    // Try backends in order
    auto backends = resolve_backends(uri);
    for (auto* backend : backends) {
        if (backend->exists(resolved)) {
            try {
                auto handle = backend->fetch(resolved);

                // Backend doesn't need caching: return directly
                if (!backend->cached()) {
                    return handle;
                }

                // Cached backend: write to cache, return cached path
                write_cache(uri_str, handle);
                return ResourceHandle(cached_path, false);  // Don't own cached files

            } catch (const RMError&) {
                // Try next backend
                continue;
            }
        }
    }

    throw RMError(ErrorCode::NotFound, uri_str, "Resource not found in any backend");
}

std::vector<std::string> ResourceManager::list(const std::string& pattern) {
    std::unordered_set<std::string> seen;
    std::vector<std::string> results;

    // Parse pattern to determine which backends to query
    URI uri = URI::parse(pattern);

    auto backends = resolve_backends(uri);
    std::string resolved = uri.resolved_path();

    for (auto* backend : backends) {
        auto entries = backend->list(resolved);
        for (auto& entry : entries) {
            if (seen.insert(entry).second) {
                results.push_back(std::move(entry));
            }
        }
    }

    return results;
}

std::filesystem::path ResourceManager::resolve(const std::string& uri_str) {
    URI uri = URI::parse(uri_str);
    std::string resolved = uri.resolved_path();

    auto backends = resolve_backends(uri);
    for (auto* backend : backends) {
        if (backend->exists(resolved)) {
            try {
                auto handle = backend->fetch(resolved);
                return handle.local_path();
            } catch (const RMError&) {
                continue;
            }
        }
    }

    return {};  // Empty path if not found
}

std::filesystem::path ResourceManager::resolveDir(const std::string& uri_str) {
    URI uri = URI::parse(uri_str);
    std::string resolved = uri.resolved_path();

    auto backends = resolve_backends(uri);
    for (auto* backend : backends) {
        if (backend->existsDir(resolved)) {
            return backend->resolvePath(resolved);
        }
    }

    return {};  // Empty path if not found
}

std::vector<std::string> ResourceManager::resolve_backend_names(const std::string& uri_str) {
    URI uri = URI::parse(uri_str);
    std::vector<std::string> names;

    auto backends = resolve_backends(uri);
    for (auto* backend : backends) {
        names.push_back(backend->name());
    }

    return names;
}

} // namespace rm

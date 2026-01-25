#pragma once

#include "rm/backend.hpp"
#include <filesystem>

namespace rm {

// FTP backend configuration
struct FTPConfig {
    std::string host;      // Logical host name (for identification)
    std::string ip;        // IP address or hostname
    std::string username;
    std::string password;
    int port = 21;
    std::string root;      // Root path on FTP server
    bool pid_style = false; // Use PID-style paths (canonical format: {pid}/{visit}/...)
};

// FTP backend using libcurl
// Returns in-memory data; caching is handled by ResourceManager
class FTPBackend : public Backend {
public:
    explicit FTPBackend(FTPConfig config);

    std::string name() const override;
    bool cached() const override { return true; }  // Manager should cache FTP results
    bool isAvailable() const override { return true; }  // Connectivity checked per-request
    bool exists(const std::string& path) override;
    ResourceHandle fetch(const std::string& path) override;  // Returns in-memory data
    std::vector<std::string> list(const std::string& pattern) override;

    const FTPConfig& config() const { return config_; }

private:
    std::string build_url(const std::string& path) const;
    std::string build_userpass() const;
    std::vector<std::byte> download_to_memory(const std::string& url);
    std::vector<std::string> list_directory(const std::string& path);

    FTPConfig config_;
};;

} // namespace rm

#include "rm/backends/ftp.hpp"
#include "rm/pid_path.hpp"
#include "rm/error.hpp"
#include "Log.h"
#include <curl/curl.h>
#include <sstream>
#include <algorithm>

namespace rm {

namespace {

// CURL write callback for memory buffer
size_t write_memory_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::vector<std::byte>* buffer = static_cast<std::vector<std::byte>*>(userp);
    const std::byte* data = static_cast<const std::byte*>(contents);
    buffer->insert(buffer->end(), data, data + total_size);
    return total_size;
}

// CURL write callback for string data (directory listing)
size_t write_string_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), total_size);
    return total_size;
}

// RAII wrapper for CURL handle
class CurlHandle {
public:
    CurlHandle() : curl_(curl_easy_init()) {
        if (!curl_) {
            throw RMError(ErrorCode::NetworkError, "Failed to initialize CURL");
        }
    }
    ~CurlHandle() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
    }
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;

    CURL* get() { return curl_; }
    operator CURL*() { return curl_; }

private:
    CURL* curl_;
};

} // anonymous namespace

FTPBackend::FTPBackend(FTPConfig config)
    : config_(std::move(config)) {}

std::string FTPBackend::name() const {
    return "ftp:" + config_.host;
}

std::string FTPBackend::build_url(const std::string& path) const {
    std::string url = "ftp://" + config_.ip + ":" + std::to_string(config_.port);

    // Add root path
    if (!config_.root.empty()) {
        if (config_.root[0] != '/') {
            url += "/";
        }
        url += config_.root;
    }

    // Transform path if pid_style is enabled
    std::string transformed_path = path;
    if (config_.pid_style && !path.empty()) {
        transformed_path = PidPathResolver::transform_path(path);
        if (transformed_path != path) {
            LOG_VERBOSE("[ftp] pid_style transform: " << path << " -> " << transformed_path);
        }
    }

    // Add requested path
    if (!transformed_path.empty()) {
        if (transformed_path[0] != '/' && (url.empty() || url.back() != '/')) {
            url += "/";
        }
        url += transformed_path;
    }

    return url;
}

std::string FTPBackend::build_userpass() const {
    return config_.username + ":" + config_.password;
}

bool FTPBackend::exists(const std::string& path) {
    // Check FTP server with HEAD request
    CurlHandle curl;
    std::string url = build_url(path);

    LOG_VERBOSE("[ftp] exists() checking: " << url);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_USERPWD, build_userpass().c_str());

    // Suppress output
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_string_callback);
    std::string dummy;
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &dummy);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        LOG_WARN("[ftp] exists() failed for " << url << ": " << curl_easy_strerror(res));
    }
    return res == CURLE_OK;
}

std::vector<std::byte> FTPBackend::download_to_memory(const std::string& url) {
    std::vector<std::byte> buffer;

    LOG_VERBOSE("[ftp] downloading: " << url);

    CurlHandle curl;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_memory_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_USERPWD, build_userpass().c_str());

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        LOG_ERROR("[ftp] download failed for " << url << ": " << curl_easy_strerror(res));
        throw RMError(ErrorCode::NetworkError, url,
            std::string("FTP download failed: ") + curl_easy_strerror(res));
    }

    LOG_VERBOSE("[ftp] downloaded " << buffer.size() << " bytes");
    return buffer;
}

ResourceHandle FTPBackend::fetch(const std::string& path) {
    std::string url = build_url(path);
    auto data = download_to_memory(url);
    return ResourceHandle(std::move(data));
}

std::vector<std::string> FTPBackend::list_directory(const std::string& path) {
    std::vector<std::string> results;

    std::string url = build_url(path);
    if (!url.empty() && url.back() != '/') {
        url += "/";
    }

    CurlHandle curl;
    std::string listing;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_string_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &listing);
    curl_easy_setopt(curl, CURLOPT_USERPWD, build_userpass().c_str());

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        return results;  // Empty on error
    }

    // Parse directory listing (Unix ls -l format)
    std::istringstream iss(listing);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;

        // Extract last token (filename)
        size_t last_space = line.find_last_of(" \t");
        if (last_space != std::string::npos) {
            std::string name = line.substr(last_space + 1);
            // Remove trailing newlines/slashes
            while (!name.empty() && (name.back() == '\r' || name.back() == '\n' || name.back() == '/')) {
                name.pop_back();
            }
            if (!name.empty() && name != "." && name != "..") {
                // Check if directory (starts with 'd') or file (starts with '-')
                bool is_dir = !line.empty() && line[0] == 'd';
                if (is_dir) {
                    name += "/";  // Mark directories
                }
                results.push_back(name);
            }
        }
    }

    return results;
}

std::vector<std::string> FTPBackend::list(const std::string& pattern) {
    std::vector<std::string> results;

    // Find base directory (up to first wildcard)
    std::string base_dir;
    size_t wildcard_pos = pattern.find_first_of("*?");
    if (wildcard_pos != std::string::npos) {
        size_t last_sep = pattern.rfind('/', wildcard_pos);
        if (last_sep != std::string::npos) {
            base_dir = pattern.substr(0, last_sep);
        }
    } else {
        // No wildcard - single file
        if (exists(pattern)) {
            results.push_back(pattern);
        }
        return results;
    }

    // List directory and filter
    auto entries = list_directory(base_dir);
    for (const auto& entry : entries) {
        // Skip directories for now (simple implementation)
        if (!entry.empty() && entry.back() == '/') continue;

        std::string full_path = base_dir.empty() ? entry : base_dir + "/" + entry;

        // Simple glob matching (just * for now)
        bool matches = true;
        if (wildcard_pos != std::string::npos) {
            // Extract pattern suffix after *
            size_t star_pos = pattern.find('*');
            if (star_pos != std::string::npos) {
                std::string prefix = pattern.substr(0, star_pos);
                std::string suffix = pattern.substr(star_pos + 1);

                matches = (full_path.find(prefix) == 0);
                if (matches && !suffix.empty()) {
                    matches = (full_path.length() >= suffix.length() &&
                              full_path.substr(full_path.length() - suffix.length()) == suffix);
                }
            }
        }

        if (matches) {
            results.push_back(full_path);
        }
    }

    return results;
}

} // namespace rm

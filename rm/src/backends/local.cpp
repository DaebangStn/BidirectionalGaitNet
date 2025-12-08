#include "rm/backends/local.hpp"
#include "rm/error.hpp"
#include <fstream>
#include <regex>

namespace rm {

LocalBackend::LocalBackend(std::filesystem::path root)
    : root_(std::move(root)) {
    // Normalize and verify root exists
    if (!std::filesystem::exists(root_)) {
        throw RMError(ErrorCode::ConfigError, root_.string(), "Local backend root does not exist");
    }
    root_ = std::filesystem::canonical(root_);
}

std::string LocalBackend::name() const {
    return "local:" + root_.string();
}

std::filesystem::path LocalBackend::resolve(const std::string& path) const {
    std::filesystem::path full_path;
    if (path.empty()) {
        full_path = root_;
    } else if (path[0] == '/') {
        // Absolute path within root: treat as relative
        full_path = root_ / path.substr(1);
    } else {
        full_path = root_ / path;
    }
    return full_path;
}

bool LocalBackend::exists(const std::string& path) {
    auto full_path = resolve(path);
    return std::filesystem::exists(full_path) && std::filesystem::is_regular_file(full_path);
}

ResourceHandle LocalBackend::fetch(const std::string& path) {
    auto full_path = resolve(path);

    if (!std::filesystem::exists(full_path)) {
        throw RMError(ErrorCode::NotFound, path, "File not found in local backend");
    }

    if (!std::filesystem::is_regular_file(full_path)) {
        throw RMError(ErrorCode::IOError, path, "Path is not a regular file");
    }

    // Return handle with path - data loaded lazily
    return ResourceHandle(std::filesystem::canonical(full_path), false);
}

bool LocalBackend::match_glob(const std::string& text, const std::string& pattern) const {
    // Convert glob pattern to regex
    // * matches any sequence of characters (not including /)
    // ** matches any sequence including /
    // ? matches single character
    std::string regex_str;
    regex_str.reserve(pattern.size() * 2);

    for (size_t i = 0; i < pattern.size(); ++i) {
        char c = pattern[i];
        switch (c) {
            case '*':
                if (i + 1 < pattern.size() && pattern[i + 1] == '*') {
                    regex_str += ".*";
                    ++i;  // Skip second *
                } else {
                    regex_str += "[^/]*";
                }
                break;
            case '?':
                regex_str += "[^/]";
                break;
            case '.':
            case '+':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '^':
            case '$':
            case '|':
            case '\\':
                regex_str += '\\';
                regex_str += c;
                break;
            default:
                regex_str += c;
                break;
        }
    }

    try {
        std::regex re(regex_str);
        return std::regex_match(text, re);
    } catch (const std::regex_error&) {
        return false;
    }
}

std::vector<std::string> LocalBackend::list(const std::string& pattern) {
    std::vector<std::string> results;
    std::error_code ec;

    // Find the base directory (up to first wildcard)
    std::string base_dir;
    std::string glob_pattern = pattern;
    size_t wildcard_pos = pattern.find_first_of("*?");

    if (wildcard_pos != std::string::npos) {
        // Has wildcard - extract base directory
        size_t last_sep = pattern.rfind('/', wildcard_pos);
        if (last_sep != std::string::npos) {
            base_dir = pattern.substr(0, last_sep);
        }
    } else {
        // No wildcard - check if it's a directory or file
        auto full_path = resolve(pattern);
        if (std::filesystem::is_directory(full_path)) {
            // List directory contents
            // Root directory (empty pattern): list directories only (for @pid use case)
            // Subdirectories: list both files and directories
            bool dirs_only = pattern.empty();

            for (auto& entry : std::filesystem::directory_iterator(full_path, ec)) {
                if (dirs_only && !entry.is_directory()) {
                    continue;  // Skip files when listing root
                }
                auto rel_path = std::filesystem::relative(entry.path(), root_, ec);
                if (!ec) {
                    results.push_back(rel_path.string());
                }
            }
            return results;
        } else if (std::filesystem::is_regular_file(full_path)) {
            // Single file
            results.push_back(pattern);
            return results;
        }
        // Path doesn't exist - return empty
        return results;
    }

    auto base_path = resolve(base_dir);
    if (!std::filesystem::exists(base_path) || !std::filesystem::is_directory(base_path)) {
        return results;
    }

    // Check if we need recursive iteration
    bool recursive = pattern.find("**") != std::string::npos;

    if (recursive) {
        for (auto& entry : std::filesystem::recursive_directory_iterator(base_path, ec)) {
            if (entry.is_regular_file()) {
                auto rel_path = std::filesystem::relative(entry.path(), root_, ec);
                if (!ec && match_glob(rel_path.string(), glob_pattern)) {
                    results.push_back(rel_path.string());
                }
            }
        }
    } else {
        for (auto& entry : std::filesystem::directory_iterator(base_path, ec)) {
            if (entry.is_regular_file()) {
                auto rel_path = std::filesystem::relative(entry.path(), root_, ec);
                if (!ec && match_glob(rel_path.string(), glob_pattern)) {
                    results.push_back(rel_path.string());
                }
            }
        }
    }

    return results;
}

} // namespace rm

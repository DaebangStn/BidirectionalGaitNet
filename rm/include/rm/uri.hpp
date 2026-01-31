#pragma once

#include <string>

namespace rm {

// URI for resource identification with prefix-based routing
// Supports:
//   - Prefix routing: "@data/skeleton/base.xml", "@pid:CP001/markers/trial.c3d"
//   - Relative paths: "skeleton/base.xml"
//   - Absolute paths: "/home/user/data/file.txt"
//   - scheme:path format: "file:/path" or "ftp:/path"
//
// Prefix syntax:
//   @prefix/path        -> prefix="@prefix", arg="", path="path"
//   @prefix:arg/path    -> prefix="@prefix", arg="arg", path="path"
class URI {
public:
    URI() = default;
    explicit URI(const std::string& str);

    // Parse a string into URI
    static URI parse(const std::string& str);

    // Check if path is relative (no leading /)
    bool is_relative() const;

    // Check if path is absolute
    bool is_absolute() const;

    // Get the scheme (empty for relative/absolute paths without scheme)
    const std::string& scheme() const { return scheme_; }

    // Get the path component (after prefix expansion)
    const std::string& path() const { return path_; }

    // Prefix routing support
    bool has_prefix() const { return !prefix_.empty(); }
    const std::string& prefix() const { return prefix_; }      // "@data", "@pid", etc.
    const std::string& prefix_arg() const { return prefix_arg_; }  // "CP001" for @pid:CP001

    // Get resolved path for backend lookup (includes prefix_arg if present)
    // @pid:CP001/markers/trial.c3d -> "CP001/markers/trial.c3d"
    std::string resolved_path() const;

    // Convert back to string representation
    std::string to_string() const;

    // Check if URI is empty
    bool empty() const { return path_.empty() && scheme_.empty() && prefix_.empty(); }

private:
    std::string scheme_;      // "file", "ftp", or empty
    std::string path_;        // path after prefix (e.g., "markers/trial.c3d")
    std::string prefix_;      // "@data", "@pid", or empty
    std::string prefix_arg_;  // "CP001" for @pid:CP001, empty otherwise
};

// Expand @pid:/path to @pid:{default_pid}/path when prefix_arg is empty
// Returns original URI unchanged if not a @pid: URI or if prefix_arg is already set
std::string expand_pid(const std::string& uri, const std::string& default_pid);

} // namespace rm

#include "rm/uri.hpp"
#include "rm/error.hpp"
#include <algorithm>

namespace rm {

URI::URI(const std::string& str) {
    *this = parse(str);
}

URI URI::parse(const std::string& str) {
    URI uri;

    if (str.empty()) {
        return uri;
    }

    std::string remaining = str;

    // Check for @prefix at start
    if (!remaining.empty() && remaining[0] == '@') {
        // Find end of prefix: either ':' (with arg) or '/' (without arg)
        size_t colon_pos = remaining.find(':');
        size_t slash_pos = remaining.find('/');

        if (colon_pos != std::string::npos && (slash_pos == std::string::npos || colon_pos < slash_pos)) {
            // @prefix:arg/path format
            uri.prefix_ = remaining.substr(0, colon_pos);  // "@prefix"

            // Extract arg (between : and /)
            if (slash_pos != std::string::npos) {
                uri.prefix_arg_ = remaining.substr(colon_pos + 1, slash_pos - colon_pos - 1);
                remaining = remaining.substr(slash_pos + 1);  // path after /
            } else {
                // @prefix:arg with no path (just the arg)
                uri.prefix_arg_ = remaining.substr(colon_pos + 1);
                remaining.clear();
            }
        } else if (slash_pos != std::string::npos) {
            // @prefix/path format (no arg)
            uri.prefix_ = remaining.substr(0, slash_pos);  // "@prefix"
            remaining = remaining.substr(slash_pos + 1);    // path after /
        } else {
            // Just @prefix with no path
            uri.prefix_ = remaining;
            remaining.clear();
        }

        uri.path_ = remaining;
        return uri;
    }

    // Check for scheme:path format (existing logic)
    size_t colon_pos = remaining.find(':');
    if (colon_pos != std::string::npos && colon_pos > 0) {
        std::string potential_scheme = remaining.substr(0, colon_pos);

        // Validate scheme: must be alphanumeric (no path separators)
        bool valid_scheme = true;
        for (char c : potential_scheme) {
            if (!std::isalnum(c) && c != '_' && c != '-') {
                valid_scheme = false;
                break;
            }
        }

        // Handle Windows drive letters (C:/ etc.) - not a scheme
        if (colon_pos == 1 && remaining.length() > 2 && (remaining[2] == '/' || remaining[2] == '\\')) {
            valid_scheme = false;
        }

        if (valid_scheme) {
            uri.scheme_ = potential_scheme;
            uri.path_ = remaining.substr(colon_pos + 1);

            // Remove leading slashes from path (normalize ftp://path to ftp:/path)
            while (!uri.path_.empty() && uri.path_[0] == '/') {
                // Keep at least one slash for absolute paths
                if (uri.path_.length() > 1 && uri.path_[1] != '/') {
                    break;
                }
                uri.path_ = uri.path_.substr(1);
            }

            return uri;
        }
    }

    // No prefix or scheme - treat as plain path
    uri.path_ = remaining;
    return uri;
}

bool URI::is_relative() const {
    if (!scheme_.empty()) {
        return false;  // Has scheme, not relative
    }
    if (has_prefix()) {
        return true;  // Prefixed URIs are resolved via routing
    }
    return !path_.empty() && path_[0] != '/';
}

bool URI::is_absolute() const {
    return !is_relative();
}

std::string URI::resolved_path() const {
    if (prefix_arg_.empty()) {
        return path_;
    }
    // Combine arg and path: "CP001" + "markers/trial.c3d" -> "CP001/markers/trial.c3d"
    if (path_.empty()) {
        return prefix_arg_;
    }
    return prefix_arg_ + "/" + path_;
}

std::string URI::to_string() const {
    std::string result;

    if (!prefix_.empty()) {
        result = prefix_;
        if (!prefix_arg_.empty()) {
            result += ":" + prefix_arg_;
        }
        if (!path_.empty()) {
            result += "/" + path_;
        }
        return result;
    }

    if (!scheme_.empty()) {
        return scheme_ + ":" + path_;
    }

    return path_;
}

std::string expand_pid(const std::string& uri, const std::string& default_pid) {
    if (default_pid.empty()) {
        return uri;
    }

    URI parsed = URI::parse(uri);

    // Only expand @pid: URIs with empty prefix_arg
    if (parsed.prefix() == "@pid" && parsed.prefix_arg().empty()) {
        // Rebuild as @pid:{default_pid}/path
        std::string result = "@pid:" + default_pid;
        if (!parsed.path().empty()) {
            result += "/" + parsed.path();
        }
        return result;
    }

    return uri;
}

} // namespace rm

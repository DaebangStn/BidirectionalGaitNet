#include "rm/backends/pid.hpp"
#include "rm/pid_path.hpp"
#include "rm/error.hpp"
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <yaml-cpp/yaml.h>

namespace rm {

// Known metadata fields that can be accessed via @pid:{id}/{field}
static const std::unordered_set<std::string> METADATA_FIELDS = {
    "name", "pid", "gmfcs", "ser_no", "weight"
};

PidBackend::PidBackend(std::filesystem::path root)
    : root_(std::move(root)) {
    // Normalize and verify root exists
    try {
        if (!std::filesystem::exists(root_)) {
            throw RMError(ErrorCode::ConfigError, root_.string(), "Pid backend root does not exist");
        }
        root_ = std::filesystem::canonical(root_);
    } catch (const std::filesystem::filesystem_error& e) {
        // Handle network mount errors, permission issues, etc.
        throw RMError(ErrorCode::ConfigError, root_.string(),
            std::string("Cannot access pid backend root: ") + e.what());
    }
}

std::string PidBackend::name() const {
    return "pid:" + root_.string();
}

std::optional<PidBackend::GaitPathComponents> PidBackend::parse_gait_path(const std::string& path) const {
    // Delegate to shared resolver
    auto result = PidPathResolver::parse_gait_path(path);
    if (!result) {
        return std::nullopt;
    }
    GaitPathComponents components;
    components.patient_id = result->patient_id;
    components.timepoint = result->timepoint;
    components.filename = result->filename;
    return components;
}

std::optional<PidBackend::H5PathComponents> PidBackend::parse_h5_path(const std::string& path) const {
    // Delegate to shared resolver
    auto result = PidPathResolver::parse_h5_path(path);
    if (!result) {
        return std::nullopt;
    }
    H5PathComponents components;
    components.patient_id = result->patient_id;
    components.timepoint = result->timepoint;
    components.filename = result->filename;
    return components;
}

std::optional<PidBackend::MetadataFieldRequest> PidBackend::parse_metadata_field(const std::string& path) const {
    // Pattern: {patient_id}/{field_name}
    // Examples:
    //   "12964246/name" -> {patient_id="12964246", field_name="name"}
    //   "12964246/gmfcs" -> {patient_id="12964246", field_name="gmfcs"}

    std::vector<std::string> parts;
    std::istringstream iss(path);
    std::string part;
    while (std::getline(iss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }

    // Need exactly 2 parts: {pid}/{field}
    if (parts.size() != 2) {
        return std::nullopt;
    }

    // Check if the second part is a known metadata field
    if (METADATA_FIELDS.find(parts[1]) == METADATA_FIELDS.end()) {
        return std::nullopt;
    }

    MetadataFieldRequest request;
    request.patient_id = parts[0];
    request.field_name = parts[1];

    return request;
}

std::string PidBackend::fetch_metadata_field(const MetadataFieldRequest& request) const {
    auto metadata_path = root_ / request.patient_id / "metadata.yaml";

    if (!std::filesystem::exists(metadata_path)) {
        throw RMError(ErrorCode::NotFound, request.patient_id + "/" + request.field_name,
            "metadata.yaml not found for patient " + request.patient_id);
    }

    try {
        YAML::Node config = YAML::LoadFile(metadata_path.string());

        if (!config[request.field_name]) {
            throw RMError(ErrorCode::NotFound, request.patient_id + "/" + request.field_name,
                "Field '" + request.field_name + "' not found in metadata.yaml");
        }

        // Convert value to string
        return config[request.field_name].as<std::string>();

    } catch (const YAML::Exception& e) {
        throw RMError(ErrorCode::IOError, metadata_path.string(),
            std::string("Failed to parse metadata.yaml: ") + e.what());
    }
}

std::filesystem::path PidBackend::transform_gait_path(const GaitPathComponents& components) const {
    // Build base path: {root}/{pid}/gait/{pre|post}
    auto base_path = root_ / components.patient_id / "gait" / components.timepoint;

    if (components.filename.empty()) {
        return base_path;
    }

    // For files, check both locations:
    // 1. gait/{pre|post}/{filename} (new location for Trimmed_unified.c3d)
    // 2. gait/{pre|post}/Generated_C3D_files/{filename} (legacy location)
    auto direct_path = base_path / components.filename;
    if (std::filesystem::exists(direct_path)) {
        return direct_path;
    }

    auto generated_path = base_path / "Generated_C3D_files" / components.filename;
    if (std::filesystem::exists(generated_path)) {
        return generated_path;
    }

    // Default to direct path (for new files to be created)
    return direct_path;
}

std::filesystem::path PidBackend::transform_h5_path(const H5PathComponents& components) const {
    // Build base path: {root}/{pid}/gait/{pre|post}/h5
    auto base_path = root_ / components.patient_id / "gait" / components.timepoint / "h5";

    if (components.filename.empty()) {
        return base_path;
    }

    // Return full path to file
    return base_path / components.filename;
}

std::filesystem::path PidBackend::resolve(const std::string& path) const {
    // Check if this is a gait path that needs transformation
    auto gait_components = parse_gait_path(path);
    if (gait_components) {
        return transform_gait_path(*gait_components);
    }

    // Check if this is an h5 path that needs transformation
    auto h5_components = parse_h5_path(path);
    if (h5_components) {
        return transform_h5_path(*h5_components);
    }

    // Non-gait/h5 path: resolve directly
    if (path.empty()) {
        return root_;
    } else if (path[0] == '/') {
        return root_ / path.substr(1);
    } else {
        return root_ / path;
    }
}

bool PidBackend::exists(const std::string& path) {
    // Check if this is a metadata field request
    auto metadata_request = parse_metadata_field(path);
    if (metadata_request) {
        auto metadata_path = root_ / metadata_request->patient_id / "metadata.yaml";
        if (!std::filesystem::exists(metadata_path)) {
            return false;
        }
        try {
            YAML::Node config = YAML::LoadFile(metadata_path.string());
            return config[metadata_request->field_name].IsDefined();
        } catch (...) {
            return false;
        }
    }

    auto full_path = resolve(path);
    return std::filesystem::exists(full_path) && std::filesystem::is_regular_file(full_path);
}

bool PidBackend::existsDir(const std::string& path) {
    auto full_path = resolve(path);
    return std::filesystem::exists(full_path) && std::filesystem::is_directory(full_path);
}

std::filesystem::path PidBackend::resolvePath(const std::string& path) {
    return resolve(path);
}

ResourceHandle PidBackend::fetch(const std::string& path) {
    // Check if this is a metadata field request
    auto metadata_request = parse_metadata_field(path);
    if (metadata_request) {
        std::string value = fetch_metadata_field(*metadata_request);
        // Return value as in-memory ResourceHandle
        std::vector<std::byte> data(value.size());
        std::transform(value.begin(), value.end(), data.begin(),
            [](char c) { return static_cast<std::byte>(c); });
        return ResourceHandle(std::move(data));
    }

    auto full_path = resolve(path);

    if (!std::filesystem::exists(full_path)) {
        throw RMError(ErrorCode::NotFound, path, "File not found in pid backend");
    }

    if (!std::filesystem::is_regular_file(full_path)) {
        throw RMError(ErrorCode::IOError, path, "Path is not a regular file");
    }

    return ResourceHandle(std::filesystem::canonical(full_path), false);
}

bool PidBackend::match_glob(const std::string& text, const std::string& pattern) const {
    // Convert glob pattern to regex
    std::string regex_str;
    regex_str.reserve(pattern.size() * 2);

    for (size_t i = 0; i < pattern.size(); ++i) {
        char c = pattern[i];
        switch (c) {
            case '*':
                if (i + 1 < pattern.size() && pattern[i + 1] == '*') {
                    regex_str += ".*";
                    ++i;
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

std::vector<std::string> PidBackend::list(const std::string& pattern) {
    std::vector<std::string> results;
    std::error_code ec;

    // Check if this is an h5 path
    auto h5_components = parse_h5_path(pattern);

    if (h5_components) {
        // H5 path: list *.h5 and *.hdf files recursively from gait/{pre|post}/h5/
        auto h5_dir = root_ / h5_components->patient_id / "gait" / h5_components->timepoint / "h5";

        if (!std::filesystem::exists(h5_dir) || !std::filesystem::is_directory(h5_dir)) {
            return results;
        }

        std::unordered_set<std::string> seen;

        auto is_h5_file = [](const std::string& filename) {
            if (filename.size() > 3 && filename.substr(filename.size() - 3) == ".h5") {
                return true;
            }
            if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".hdf") {
                return true;
            }
            return false;
        };

        // Recursive search for h5/hdf files
        for (auto& entry : std::filesystem::recursive_directory_iterator(h5_dir, ec)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                auto rel_path = std::filesystem::relative(entry.path(), h5_dir, ec);
                std::string rel_str = rel_path.string();

                // Check if we have a filename pattern (glob)
                if (!h5_components->filename.empty()) {
                    if (match_glob(rel_str, h5_components->filename) || match_glob(filename, h5_components->filename)) {
                        if (seen.insert(rel_str).second) {
                            results.push_back(rel_str);
                        }
                    }
                } else {
                    // List all .h5 and .hdf files
                    if (is_h5_file(filename)) {
                        if (seen.insert(rel_str).second) {
                            results.push_back(rel_str);
                        }
                    }
                }
            }
        }

        std::sort(results.begin(), results.end());
        return results;
    }

    // Check if this is a gait path
    auto gait_components = parse_gait_path(pattern);

    if (gait_components) {
        // Gait path: list c3d files from both locations:
        // 1. gait/{pre|post}/ (direct)
        // 2. gait/{pre|post}/Generated_C3D_files/ (legacy)
        auto base_path = root_ / gait_components->patient_id / "gait" / gait_components->timepoint;
        auto generated_path = base_path / "Generated_C3D_files";

        std::unordered_set<std::string> seen;  // To avoid duplicates

        auto collect_c3d_files = [&](const std::filesystem::path& dir_path) {
            if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
                return;
            }

            for (auto& entry : std::filesystem::directory_iterator(dir_path, ec)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();

                    // Check if we have a filename pattern (glob)
                    if (!gait_components->filename.empty()) {
                        if (match_glob(filename, gait_components->filename)) {
                            if (seen.find(filename) == seen.end()) {
                                results.push_back(filename);
                                seen.insert(filename);
                            }
                        }
                    } else {
                        // List all .c3d files
                        if (filename.size() > 4 &&
                            filename.substr(filename.size() - 4) == ".c3d") {
                            if (seen.find(filename) == seen.end()) {
                                results.push_back(filename);
                                seen.insert(filename);
                            }
                        }
                    }
                }
            }
        };

        // Collect from both locations (direct first, then Generated_C3D_files)
        collect_c3d_files(base_path);
        collect_c3d_files(generated_path);

        // Sort results for consistent ordering
        std::sort(results.begin(), results.end());
        return results;
    }

    // Non-gait path: use standard directory listing
    auto full_path = resolve(pattern);

    // Find the base directory (up to first wildcard)
    std::string glob_pattern = pattern;
    size_t wildcard_pos = pattern.find_first_of("*?");

    if (wildcard_pos != std::string::npos) {
        // Has wildcard - extract base directory
        std::string base_dir;
        size_t last_sep = pattern.rfind('/', wildcard_pos);
        if (last_sep != std::string::npos) {
            base_dir = pattern.substr(0, last_sep);
        }

        auto base_path = resolve(base_dir);
        if (!std::filesystem::exists(base_path) || !std::filesystem::is_directory(base_path)) {
            return results;
        }

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
    } else {
        // No wildcard - check if it's a directory or file
        if (std::filesystem::is_directory(full_path)) {
            bool dirs_only = pattern.empty();

            std::error_code iter_ec;
            for (auto& entry : std::filesystem::directory_iterator(full_path, iter_ec)) {
                if (iter_ec) break;  // Stop on iteration error
                if (dirs_only && !entry.is_directory()) {
                    continue;
                }
                // Return just the filename for consistency with gait path behavior
                results.push_back(entry.path().filename().string());
            }
        } else if (std::filesystem::is_regular_file(full_path)) {
            // Return just the filename
            results.push_back(std::filesystem::path(pattern).filename().string());
        }
    }

    return results;
}

} // namespace rm

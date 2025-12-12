#include "rm/backends/pid.hpp"
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
    "name", "pid", "gmfcs", "ser_no"
};

PidBackend::PidBackend(std::filesystem::path root)
    : root_(std::move(root)) {
    // Normalize and verify root exists
    if (!std::filesystem::exists(root_)) {
        throw RMError(ErrorCode::ConfigError, root_.string(), "Pid backend root does not exist");
    }
    root_ = std::filesystem::canonical(root_);
}

std::string PidBackend::name() const {
    return "pid:" + root_.string();
}

std::optional<PidBackend::GaitPathComponents> PidBackend::parse_gait_path(const std::string& path) const {
    // Pattern: {patient_id}/gait/{pre|post}[/{filename}]
    // Examples:
    //   "12964246/gait/pre" -> {patient_id="12964246", timepoint="pre", filename=""}
    //   "12964246/gait/post/walk01-Dynamic.c3d" -> {patient_id="12964246", timepoint="post", filename="walk01-Dynamic.c3d"}

    std::vector<std::string> parts;
    std::istringstream iss(path);
    std::string part;
    while (std::getline(iss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }

    // Need at least 3 parts: {pid}/gait/{pre|post}
    if (parts.size() < 3) {
        return std::nullopt;
    }

    // Check for gait path pattern
    if (parts[1] != "gait") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    GaitPathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

    // If there are more parts, the 4th is the filename
    if (parts.size() >= 4) {
        components.filename = parts[3];
    }

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
    // Build: {root}/{pid}/gait/{pre|post}/Generated_C3D_files[/{filename}]
    auto transformed = root_ / components.patient_id / "gait" /
                       components.timepoint / "Generated_C3D_files";

    if (!components.filename.empty()) {
        transformed /= components.filename;
    }

    return transformed;
}

std::filesystem::path PidBackend::resolve(const std::string& path) const {
    // Check if this is a gait path that needs transformation
    auto gait_components = parse_gait_path(path);
    if (gait_components) {
        return transform_gait_path(*gait_components);
    }

    // Non-gait path: resolve directly
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

    // Check if this is a gait path
    auto gait_components = parse_gait_path(pattern);

    if (gait_components) {
        // Gait path: list c3d files from Generated_C3D_files directory
        auto transformed_path = transform_gait_path(*gait_components);

        if (!std::filesystem::exists(transformed_path) ||
            !std::filesystem::is_directory(transformed_path)) {
            return results;  // Empty result if directory doesn't exist
        }

        // Check if we have a filename pattern (glob)
        if (!gait_components->filename.empty()) {
            // Pattern matching within the directory
            for (auto& entry : std::filesystem::directory_iterator(transformed_path, ec)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    if (match_glob(filename, gait_components->filename)) {
                        results.push_back(filename);
                    }
                }
            }
        } else {
            // List only Trimmed_*.c3d files in the directory
            for (auto& entry : std::filesystem::directory_iterator(transformed_path, ec)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    // Only return Trimmed_*.c3d files
                    if (filename.size() > 4 &&
                        filename.substr(filename.size() - 4) == ".c3d" &&
                        filename.substr(0, 8) == "Trimmed_") {
                        results.push_back(filename);
                    }
                }
            }
        }

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

            for (auto& entry : std::filesystem::directory_iterator(full_path, ec)) {
                if (dirs_only && !entry.is_directory()) {
                    continue;
                }
                auto rel_path = std::filesystem::relative(entry.path(), root_, ec);
                if (!ec) {
                    results.push_back(rel_path.string());
                }
            }
        } else if (std::filesystem::is_regular_file(full_path)) {
            results.push_back(pattern);
        }
    }

    return results;
}

} // namespace rm

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

std::optional<PidBackend::MetadataFieldRequest> PidBackend::parse_metadata_field(const std::string& path) const {
    // Pattern: {patient_id}/{field_name}
    // Examples:
    //   "12964246/name" -> {patient_id="12964246", field_name="name"}
    //   "12964246/gmfcs" -> {patient_id="12964246", field_name="gmfcs"}

    auto parts = PidPathResolver::split_path(path);

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

std::filesystem::path PidBackend::resolve(const std::string& path) const {
    // Transform path if it's an old-style path
    std::string transformed_path = PidPathResolver::transform_path(path);

    // Build absolute path
    if (transformed_path.empty()) {
        return root_;
    } else if (transformed_path[0] == '/') {
        return root_ / transformed_path.substr(1);
    } else {
        return root_ / transformed_path;
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
    return std::filesystem::exists(full_path);
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
    return PidPathResolver::match_glob(text, pattern);
}

std::vector<std::string> PidBackend::list(const std::string& pattern) {
    std::vector<std::string> results;
    std::error_code ec;

    // Transform the pattern if it's an old-style path
    std::string transformed_pattern = PidPathResolver::transform_path(pattern);

    // Find the base directory (up to first wildcard)
    std::string glob_pattern = transformed_pattern;
    size_t wildcard_pos = transformed_pattern.find_first_of("*?");

    std::filesystem::path base_path;
    std::string base_dir;

    if (wildcard_pos != std::string::npos) {
        // Has wildcard - extract base directory
        size_t last_sep = transformed_pattern.rfind('/', wildcard_pos);
        if (last_sep != std::string::npos) {
            base_dir = transformed_pattern.substr(0, last_sep);
        }
        base_path = resolve(base_dir);
    } else {
        // No wildcard - resolve full path
        base_path = resolve(transformed_pattern);
        base_dir = transformed_pattern;
    }

    if (!std::filesystem::exists(base_path)) {
        return results;
    }

    // If no wildcard and path is a directory, list its contents
    if (wildcard_pos == std::string::npos && std::filesystem::is_directory(base_path)) {
        for (auto& entry : std::filesystem::directory_iterator(base_path, ec)) {
            results.push_back(entry.path().filename().string());
        }
        std::sort(results.begin(), results.end());
        return results;
    }

    // If no wildcard and path is a file, return just that file
    if (wildcard_pos == std::string::npos && std::filesystem::is_regular_file(base_path)) {
        results.push_back(std::filesystem::path(transformed_pattern).filename().string());
        return results;
    }

    // Process glob pattern
    bool recursive = glob_pattern.find("**") != std::string::npos;

    auto process_entry = [&](const std::filesystem::directory_entry& entry) {
        if (!entry.is_regular_file()) {
            return;
        }

        auto rel_path = std::filesystem::relative(entry.path(), root_, ec);
        if (ec) return;

        std::string rel_str = rel_path.string();

        if (match_glob(rel_str, glob_pattern)) {
            // Return just the filename for simpler API
            results.push_back(entry.path().filename().string());
        }
    };

    if (!std::filesystem::is_directory(base_path)) {
        return results;
    }

    if (recursive) {
        for (auto& entry : std::filesystem::recursive_directory_iterator(base_path, ec)) {
            process_entry(entry);
        }
    } else {
        for (auto& entry : std::filesystem::directory_iterator(base_path, ec)) {
            process_entry(entry);
        }
    }

    std::sort(results.begin(), results.end());

    // Remove duplicates
    results.erase(std::unique(results.begin(), results.end()), results.end());

    return results;
}

} // namespace rm

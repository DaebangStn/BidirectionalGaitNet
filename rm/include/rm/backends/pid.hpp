#pragma once

#include "rm/backend.hpp"
#include <filesystem>
#include <optional>
#include <vector>

namespace rm {

// Patient data backend with automatic path transformations
// Handles:
//   - Gait data: {pid}/gait/{pre|post} -> {pid}/gait/{pre|post}/Generated_C3D_files
//   - Metadata fields: {pid}/name -> reads "name" from {pid}/metadata.yaml
class PidBackend : public Backend {
public:
    explicit PidBackend(std::filesystem::path root);

    std::string name() const override;
    bool cached() const override { return false; }  // Local files don't need caching
    bool exists(const std::string& path) override;
    ResourceHandle fetch(const std::string& path) override;
    std::vector<std::string> list(const std::string& pattern) override;

    const std::filesystem::path& root() const { return root_; }

private:
    struct GaitPathComponents {
        std::string patient_id;
        std::string timepoint;   // "pre" or "post"
        std::string filename;    // optional, empty for directory listing
    };

    struct MetadataFieldRequest {
        std::string patient_id;
        std::string field_name;  // e.g., "name", "pid", "gmfcs"
    };

    // Parse gait path pattern: {pid}/gait/{pre|post}[/{filename}]
    std::optional<GaitPathComponents> parse_gait_path(const std::string& path) const;

    // Parse metadata field request: {pid}/{field_name} where field_name is a known metadata field
    std::optional<MetadataFieldRequest> parse_metadata_field(const std::string& path) const;

    // Fetch metadata field value from {pid}/metadata.yaml
    std::string fetch_metadata_field(const MetadataFieldRequest& request) const;

    // Transform gait path to include Generated_C3D_files subdirectory
    std::filesystem::path transform_gait_path(const GaitPathComponents& components) const;

    // Resolve path to absolute filesystem path (handles both gait and non-gait paths)
    std::filesystem::path resolve(const std::string& path) const;

    // Glob pattern matching
    bool match_glob(const std::string& text, const std::string& pattern) const;

    std::filesystem::path root_;
};

} // namespace rm

#pragma once

#include "rm/backend.hpp"
#include <filesystem>
#include <optional>
#include <vector>

namespace rm {

// Patient data backend with direct path resolution
//
// New structure (visit-based):
//   {pid}/{visit}/{data_type}/{filename}
//
// Where:
//   - visit: "pre", "op1", "op2"
//   - data_type: "gait", "motion", "skeleton", "muscle", "ckpt", "dicom"
//
// Examples:
//   "12964246/pre/gait/walk01.c3d"
//   "12964246/pre/motion/Trimmed_walk02-Dynamic.h5"
//   "12964246/pre/skeleton/dynamic.yaml"
//   "12964246/metadata.yaml"      (patient-level metadata)
//   "12964246/pre/metadata.yaml"  (visit-level metadata)
//
// Backward Compatibility:
//   Old-style paths are automatically transformed to new-style:
//   - {pid}/gait/{pre|post}/... -> {pid}/{visit}/gait/...
//   - {pid}/h5/{pre|post}/...   -> {pid}/{visit}/motion/...
//   - {pid}/skeleton/{pre|post}/... -> {pid}/{visit}/skeleton/...
//
class PidBackend : public Backend {
public:
    explicit PidBackend(std::filesystem::path root);

    std::string name() const override;
    bool cached() const override { return false; }  // Local files don't need caching
    bool isAvailable() const override { return std::filesystem::exists(root_); }
    bool exists(const std::string& path) override;
    bool existsDir(const std::string& path) override;
    std::filesystem::path resolvePath(const std::string& path) override;
    ResourceHandle fetch(const std::string& path) override;
    std::vector<std::string> list(const std::string& pattern) override;

    const std::filesystem::path& root() const { return root_; }

private:
    struct MetadataFieldRequest {
        std::string patient_id;
        std::string field_name;  // e.g., "name", "pid", "gmfcs"
    };

    // Parse metadata field request: {pid}/{field_name} where field_name is a known metadata field
    std::optional<MetadataFieldRequest> parse_metadata_field(const std::string& path) const;

    // Fetch metadata field value from {pid}/metadata.yaml
    std::string fetch_metadata_field(const MetadataFieldRequest& request) const;

    // Resolve path to absolute filesystem path
    // Handles:
    //   1. Direct new-style paths (no transformation)
    //   2. Legacy old-style paths (auto-transformed via PidPathResolver)
    std::filesystem::path resolve(const std::string& path) const;

    // Glob pattern matching
    bool match_glob(const std::string& text, const std::string& pattern) const;

    std::filesystem::path root_;
};;

} // namespace rm

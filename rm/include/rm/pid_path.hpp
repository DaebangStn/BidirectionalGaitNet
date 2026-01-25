#pragma once

#include <string>
#include <optional>
#include <vector>

namespace rm {

// Simplified PID path resolver for visit-based flat structure
//
// New structure (visit-based):
//   {pid}/{visit}/{data_type}[/{filename}]
//
// Where:
//   - visit: "pre", "op1", "op2" (or "" for root-level access)
//   - data_type: "gait", "motion", "skeleton", "muscle", "ckpt", "dicom"
//   - filename: optional file name or glob pattern
//
// Examples:
//   "12964246/pre/gait/walk01.c3d"
//   "12964246/pre/motion/Trimmed_walk02-Dynamic.h5"
//   "12964246/pre/skeleton/dynamic.yaml"
//   "12964246/pre/metadata.yaml"  (visit-level metadata)
//   "12964246/metadata.yaml"      (patient-level metadata)
//
class PidPathResolver {
public:
    // Unified path components for all data types
    struct PathComponents {
        std::string patient_id;
        std::string visit;      // "pre", "op1", "op2", or "" for root
        std::string data_type;  // "gait", "motion", "skeleton", "muscle", "ckpt", "dicom", "metadata", "rom"
        std::string filename;   // file name or glob pattern
    };

    // Parse path into components
    // Patterns:
    //   {pid}                          -> patient_id only
    //   {pid}/metadata.yaml            -> root metadata
    //   {pid}/{visit}                  -> visit only
    //   {pid}/{visit}/metadata.yaml    -> visit metadata
    //   {pid}/{visit}/rom.yaml         -> visit rom
    //   {pid}/{visit}/{data_type}      -> data type directory
    //   {pid}/{visit}/{data_type}/{filename} -> specific file
    static std::optional<PathComponents> parse(const std::string& path);

    // Build path from components (direct - no transformation)
    static std::string build(const PathComponents& c);

    // Validation helpers
    static bool is_valid_visit(const std::string& v);
    static bool is_valid_data_type(const std::string& t);

    // Split path into parts
    static std::vector<std::string> split_path(const std::string& path);

    // Check if filename matches glob pattern
    static bool match_glob(const std::string& filename, const std::string& pattern);

    // ============================================================
    // Legacy API - for backward compatibility during migration
    // These will be removed after migration is complete
    // ============================================================

    struct GaitPathComponents {
        std::string patient_id;
        std::string timepoint;   // "pre" or "post"
        std::string filename;
    };

    struct H5PathComponents {
        std::string patient_id;
        std::string timepoint;
        std::string filename;
    };

    struct SkeletonPathComponents {
        std::string patient_id;
        std::string timepoint;
        std::string filename;
    };

    struct MusclePathComponents {
        std::string patient_id;
        std::string timepoint;
        std::string filename;
    };

    // Legacy parsers (detect old-style paths)
    static std::optional<GaitPathComponents> parse_gait_path(const std::string& path);
    static std::optional<H5PathComponents> parse_h5_path(const std::string& path);
    static std::optional<SkeletonPathComponents> parse_skeleton_path(const std::string& path);
    static std::optional<MusclePathComponents> parse_muscle_path(const std::string& path);

    // Legacy transformers (kept for reference but transformation is disabled)
    static std::string transform_h5_path(const H5PathComponents& components);
    static std::string transform_gait_path(const GaitPathComponents& components);
    static std::string transform_skeleton_path(const SkeletonPathComponents& components);
    static std::string transform_muscle_path(const MusclePathComponents& components);

    // Convert legacy "pre"/"post" to new visit names
    static std::string timepoint_to_visit(const std::string& timepoint);
};

} // namespace rm

#pragma once

#include <string>
#include <optional>
#include <vector>

namespace rm {

// Shared PID path transformation logic
// Used by both PidBackend (local) and FTPBackend (when pid_style=true)
class PidPathResolver {
public:
    struct GaitPathComponents {
        std::string patient_id;
        std::string timepoint;   // "pre" or "post"
        std::string filename;    // optional, empty for directory listing
    };

    struct H5PathComponents {
        std::string patient_id;
        std::string timepoint;   // "pre" or "post"
        std::string filename;    // optional, empty for directory listing
    };

    struct SkeletonPathComponents {
        std::string patient_id;
        std::string timepoint;   // "pre" or "post"
        std::string filename;    // optional, empty for directory listing
    };

    struct MusclePathComponents {
        std::string patient_id;
        std::string timepoint;   // "pre" or "post"
        std::string filename;    // optional, empty for directory listing
    };

    // Parse gait path pattern: {pid}/gait/{pre|post}[/{filename}]
    // Only matches gait paths for c3d file listing, NOT nested directories
    static std::optional<GaitPathComponents> parse_gait_path(const std::string& path);

    // Parse h5 path pattern: {pid}/h5/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/h5/
    static std::optional<H5PathComponents> parse_h5_path(const std::string& path);

    // Parse skeleton path pattern: {pid}/skeleton/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/skeleton/
    static std::optional<SkeletonPathComponents> parse_skeleton_path(const std::string& path);

    // Parse muscle path pattern: {pid}/muscle/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/muscle/
    static std::optional<MusclePathComponents> parse_muscle_path(const std::string& path);

    // Transform path using PID conventions (string-based, no filesystem operations)
    // - h5: {pid}/h5/{pre|post}/{file} -> {pid}/gait/{pre|post}/h5/{file}
    // - gait c3d: {pid}/gait/{pre|post}/{file}.c3d -> {pid}/gait/{pre|post}/Generated_C3D_files/{file}.c3d
    // - other: no change
    static std::string transform_path(const std::string& path);

    // Transform h5 path: {pid}/h5/{pre|post}/{file} -> {pid}/gait/{pre|post}/h5/{file}
    static std::string transform_h5_path(const H5PathComponents& components);

    // Transform gait path for c3d files (adds Generated_C3D_files if needed)
    static std::string transform_gait_path(const GaitPathComponents& components);

    // Transform skeleton path: {pid}/skeleton/{pre|post}/{file} -> {pid}/gait/{pre|post}/skeleton/{file}
    static std::string transform_skeleton_path(const SkeletonPathComponents& components);

    // Transform muscle path: {pid}/muscle/{pre|post}/{file} -> {pid}/gait/{pre|post}/muscle/{file}
    static std::string transform_muscle_path(const MusclePathComponents& components);

    // Split path into parts
    static std::vector<std::string> split_path(const std::string& path);
};

} // namespace rm

#include "rm/pid_path.hpp"
#include <sstream>

namespace rm {

std::vector<std::string> PidPathResolver::split_path(const std::string& path) {
    std::vector<std::string> parts;
    std::istringstream iss(path);
    std::string part;
    while (std::getline(iss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

std::optional<PidPathResolver::GaitPathComponents> PidPathResolver::parse_gait_path(const std::string& path) {
    // Pattern: {patient_id}/gait/{pre|post}[/{filename}]
    // Only matches gait paths for c3d file listing, NOT nested directories
    // Examples:
    //   "12964246/gait/pre" -> {patient_id="12964246", timepoint="pre", filename=""}
    //   "12964246/gait/post/walk01-Dynamic.c3d" -> {patient_id="12964246", timepoint="post", filename="walk01-Dynamic.c3d"}
    //   "12964246/gait/pre/*.c3d" -> {patient_id="12964246", timepoint="pre", filename="*.c3d"}
    //   "12964246/gait/pre/h5" -> NOT a gait path (subdirectory), returns nullopt

    auto parts = split_path(path);

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

    // If more than 4 parts, it's a nested directory path, not a gait path
    if (parts.size() > 4) {
        return std::nullopt;
    }

    // If exactly 4 parts, the 4th must look like a c3d file or wildcard pattern
    if (parts.size() == 4) {
        const std::string& filename = parts[3];
        bool is_c3d_pattern = false;

        // Check if it ends with .c3d
        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".c3d") {
            is_c3d_pattern = true;
        }
        // Check if it contains wildcards (glob pattern for c3d files)
        else if (filename.find('*') != std::string::npos || filename.find('?') != std::string::npos) {
            is_c3d_pattern = true;
        }

        // If 4th part doesn't look like a c3d file/pattern, treat as directory path
        if (!is_c3d_pattern) {
            return std::nullopt;
        }
    }

    GaitPathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

    // If there are 4 parts and we got here, the 4th is a valid c3d filename/pattern
    if (parts.size() == 4) {
        components.filename = parts[3];
    }

    return components;
}

std::optional<PidPathResolver::H5PathComponents> PidPathResolver::parse_h5_path(const std::string& path) {
    // Pattern: {patient_id}/h5/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/h5/
    // Examples:
    //   "12964246/h5/pre" -> list all *.h5, *.hdf files recursively
    //   "12964246/h5/post/walk01.h5" -> specific file
    //   "12964246/h5/pre/*.h5" -> glob pattern

    auto parts = split_path(path);

    // Need at least 3 parts: {pid}/h5/{pre|post}
    if (parts.size() < 3) {
        return std::nullopt;
    }

    // Check for h5 path pattern
    if (parts[1] != "h5") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    H5PathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

    // If there are 4+ parts, the rest is the filename/pattern
    if (parts.size() >= 4) {
        // Join remaining parts with /
        std::string filename;
        for (size_t i = 3; i < parts.size(); ++i) {
            if (!filename.empty()) filename += "/";
            filename += parts[i];
        }
        components.filename = filename;
    }

    return components;
}

std::string PidPathResolver::transform_h5_path(const H5PathComponents& components) {
    // Transform: {pid}/h5/{pre|post}/{file} -> {pid}/gait/{pre|post}/h5/{file}
    std::string result = components.patient_id + "/gait/" + components.timepoint + "/h5";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::string PidPathResolver::transform_gait_path(const GaitPathComponents& components) {
    // Transform: {pid}/gait/{pre|post}/{file}.c3d -> {pid}/gait/{pre|post}/Generated_C3D_files/{file}.c3d
    // Note: For local filesystem, we check if file exists directly first. For FTP, we always use Generated_C3D_files
    std::string result = components.patient_id + "/gait/" + components.timepoint;
    if (!components.filename.empty()) {
        result += "/Generated_C3D_files/" + components.filename;
    }
    return result;
}

std::optional<PidPathResolver::SkeletonPathComponents> PidPathResolver::parse_skeleton_path(const std::string& path) {
    // Pattern: {patient_id}/skeleton/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/skeleton/
    // Examples:
    //   "12964246/skeleton/pre" -> list all skeleton files
    //   "12964246/skeleton/post/base.xml" -> specific file

    auto parts = split_path(path);

    // Need at least 3 parts: {pid}/skeleton/{pre|post}
    if (parts.size() < 3) {
        return std::nullopt;
    }

    // Check for skeleton path pattern
    if (parts[1] != "skeleton") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    SkeletonPathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

    // If there are 4+ parts, the rest is the filename
    if (parts.size() >= 4) {
        std::string filename;
        for (size_t i = 3; i < parts.size(); ++i) {
            if (!filename.empty()) filename += "/";
            filename += parts[i];
        }
        components.filename = filename;
    }

    return components;
}

std::optional<PidPathResolver::MusclePathComponents> PidPathResolver::parse_muscle_path(const std::string& path) {
    // Pattern: {patient_id}/muscle/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/muscle/
    // Examples:
    //   "12964246/muscle/pre" -> list all muscle files
    //   "12964246/muscle/post/lower.xml" -> specific file

    auto parts = split_path(path);

    // Need at least 3 parts: {pid}/muscle/{pre|post}
    if (parts.size() < 3) {
        return std::nullopt;
    }

    // Check for muscle path pattern
    if (parts[1] != "muscle") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    MusclePathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

    // If there are 4+ parts, the rest is the filename
    if (parts.size() >= 4) {
        std::string filename;
        for (size_t i = 3; i < parts.size(); ++i) {
            if (!filename.empty()) filename += "/";
            filename += parts[i];
        }
        components.filename = filename;
    }

    return components;
}

std::string PidPathResolver::transform_skeleton_path(const SkeletonPathComponents& components) {
    // Transform: {pid}/skeleton/{pre|post}/{file} -> {pid}/gait/{pre|post}/skeleton/{file}
    std::string result = components.patient_id + "/gait/" + components.timepoint + "/skeleton";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::string PidPathResolver::transform_muscle_path(const MusclePathComponents& components) {
    // Transform: {pid}/muscle/{pre|post}/{file} -> {pid}/gait/{pre|post}/muscle/{file}
    std::string result = components.patient_id + "/gait/" + components.timepoint + "/muscle";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::string PidPathResolver::transform_path(const std::string& path) {
    // Check if this is a skeleton path that needs transformation
    auto skeleton_components = parse_skeleton_path(path);
    if (skeleton_components) {
        return transform_skeleton_path(*skeleton_components);
    }

    // Check if this is a muscle path that needs transformation
    auto muscle_components = parse_muscle_path(path);
    if (muscle_components) {
        return transform_muscle_path(*muscle_components);
    }

    // Check if this is an h5 path that needs transformation
    auto h5_components = parse_h5_path(path);
    if (h5_components) {
        return transform_h5_path(*h5_components);
    }

    // Check if this is a gait path with c3d file that needs transformation
    auto gait_components = parse_gait_path(path);
    if (gait_components && !gait_components->filename.empty()) {
        // For c3d files, transform to Generated_C3D_files path
        return transform_gait_path(*gait_components);
    }

    // No transformation needed
    return path;
}

} // namespace rm

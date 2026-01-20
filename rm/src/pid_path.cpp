#include "rm/pid_path.hpp"
#include <sstream>
#include <regex>
#include <unordered_set>

namespace rm {

// Valid visits for the new structure
static const std::unordered_set<std::string> VALID_VISITS = {
    "pre", "op1", "op2"
};

// Valid data types for the new structure
static const std::unordered_set<std::string> VALID_DATA_TYPES = {
    "gait", "motion", "skeleton", "muscle", "ckpt", "dicom"
};

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

bool PidPathResolver::is_valid_visit(const std::string& v) {
    return VALID_VISITS.count(v) > 0;
}

bool PidPathResolver::is_valid_data_type(const std::string& t) {
    return VALID_DATA_TYPES.count(t) > 0;
}

std::string PidPathResolver::timepoint_to_visit(const std::string& timepoint) {
    if (timepoint == "pre") return "pre";
    if (timepoint == "post") return "op1";
    return timepoint;  // Return as-is if not recognized
}

bool PidPathResolver::match_glob(const std::string& filename, const std::string& pattern) {
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
        return std::regex_match(filename, re);
    } catch (const std::regex_error&) {
        return false;
    }
}

std::optional<PidPathResolver::PathComponents> PidPathResolver::parse(const std::string& path) {
    auto parts = split_path(path);
    if (parts.empty()) {
        return std::nullopt;
    }

    PathComponents c;
    c.patient_id = parts[0];

    if (parts.size() == 1) {
        // Just patient ID: {pid}
        return c;
    }

    // Check if second part is a visit
    if (is_valid_visit(parts[1])) {
        c.visit = parts[1];

        if (parts.size() == 2) {
            // {pid}/{visit}
            return c;
        }

        // Check for visit-level files (metadata.yaml, rom.yaml)
        if (parts[2] == "metadata.yaml" || parts[2] == "rom.yaml") {
            c.data_type = parts[2].substr(0, parts[2].find('.'));  // "metadata" or "rom"
            c.filename = parts[2];
            return c;
        }

        // Check for data type
        if (is_valid_data_type(parts[2])) {
            c.data_type = parts[2];

            if (parts.size() == 3) {
                // {pid}/{visit}/{data_type}
                return c;
            }

            // Join remaining parts as filename
            std::string filename;
            for (size_t i = 3; i < parts.size(); ++i) {
                if (!filename.empty()) filename += "/";
                filename += parts[i];
            }
            c.filename = filename;
            return c;
        }

        // Unknown data type - treat rest as filename
        std::string filename;
        for (size_t i = 2; i < parts.size(); ++i) {
            if (!filename.empty()) filename += "/";
            filename += parts[i];
        }
        c.filename = filename;
        return c;
    }

    // Not a visit - could be root-level file (metadata.yaml)
    if (parts[1] == "metadata.yaml") {
        c.data_type = "metadata";
        c.filename = parts[1];
        return c;
    }

    // Not matching new structure - might be legacy path
    return std::nullopt;
}

std::string PidPathResolver::build(const PathComponents& c) {
    std::string result = c.patient_id;

    if (!c.visit.empty()) {
        result += "/" + c.visit;
    }

    if (!c.data_type.empty() && c.data_type != "metadata" && c.data_type != "rom") {
        result += "/" + c.data_type;
    }

    if (!c.filename.empty()) {
        result += "/" + c.filename;
    }

    return result;
}

// ============================================================
// Legacy API implementation (kept for backward compatibility)
// ============================================================

std::optional<PidPathResolver::GaitPathComponents> PidPathResolver::parse_gait_path(const std::string& path) {
    // Pattern: {patient_id}/gait/{pre|post}[/{filename}]
    auto parts = split_path(path);

    if (parts.size() < 3) {
        return std::nullopt;
    }

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

        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".c3d") {
            is_c3d_pattern = true;
        }
        else if (filename.find('*') != std::string::npos || filename.find('?') != std::string::npos) {
            is_c3d_pattern = true;
        }

        if (!is_c3d_pattern) {
            return std::nullopt;
        }
    }

    GaitPathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

    if (parts.size() == 4) {
        components.filename = parts[3];
    }

    return components;
}

std::optional<PidPathResolver::H5PathComponents> PidPathResolver::parse_h5_path(const std::string& path) {
    // Pattern: {patient_id}/h5/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/h5/
    auto parts = split_path(path);

    if (parts.size() < 3) {
        return std::nullopt;
    }

    if (parts[1] != "h5") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    H5PathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

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

std::string PidPathResolver::transform_h5_path(const H5PathComponents& components) {
    // OLD: {pid}/h5/{pre|post}/{file} -> {pid}/gait/{pre|post}/h5/{file}
    // NEW: Convert to {pid}/{visit}/motion/{file}
    std::string visit = timepoint_to_visit(components.timepoint);
    std::string result = components.patient_id + "/" + visit + "/motion";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::string PidPathResolver::transform_gait_path(const GaitPathComponents& components) {
    // OLD: {pid}/gait/{pre|post}/{file}.c3d -> {pid}/gait/{pre|post}/Generated_C3D_files/{file}.c3d
    // NEW: Convert to {pid}/{visit}/gait/{file}.c3d
    std::string visit = timepoint_to_visit(components.timepoint);
    std::string result = components.patient_id + "/" + visit + "/gait";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::optional<PidPathResolver::SkeletonPathComponents> PidPathResolver::parse_skeleton_path(const std::string& path) {
    // Pattern: {patient_id}/skeleton/{pre|post}[/{filename}]
    // Maps to: {pid}/gait/{pre|post}/skeleton/
    auto parts = split_path(path);

    if (parts.size() < 3) {
        return std::nullopt;
    }

    if (parts[1] != "skeleton") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    SkeletonPathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

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
    auto parts = split_path(path);

    if (parts.size() < 3) {
        return std::nullopt;
    }

    if (parts[1] != "muscle") {
        return std::nullopt;
    }

    if (parts[2] != "pre" && parts[2] != "post") {
        return std::nullopt;
    }

    MusclePathComponents components;
    components.patient_id = parts[0];
    components.timepoint = parts[2];

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
    // OLD: {pid}/skeleton/{pre|post}/{file} -> {pid}/gait/{pre|post}/skeleton/{file}
    // NEW: Convert to {pid}/{visit}/skeleton/{file}
    std::string visit = timepoint_to_visit(components.timepoint);
    std::string result = components.patient_id + "/" + visit + "/skeleton";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::string PidPathResolver::transform_muscle_path(const MusclePathComponents& components) {
    // OLD: {pid}/muscle/{pre|post}/{file} -> {pid}/gait/{pre|post}/muscle/{file}
    // NEW: Convert to {pid}/{visit}/muscle/{file}
    std::string visit = timepoint_to_visit(components.timepoint);
    std::string result = components.patient_id + "/" + visit + "/muscle";
    if (!components.filename.empty()) {
        result += "/" + components.filename;
    }
    return result;
}

std::string PidPathResolver::transform_path(const std::string& path) {
    // Try to detect and transform legacy paths to new structure

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
    if (gait_components) {
        return transform_gait_path(*gait_components);
    }

    // Check for old-style gait/metadata.yaml path pattern
    // OLD: {pid}/gait/metadata.yaml -> NOT TRANSFORMED (no equivalent)
    // OLD: {pid}/gait/{pre|post}/... -> Transform to {pid}/{visit}/...
    auto parts = split_path(path);
    if (parts.size() >= 3 && parts[1] == "gait") {
        std::string timepoint = parts[2];
        if (timepoint == "pre" || timepoint == "post") {
            std::string visit = timepoint_to_visit(timepoint);
            std::string result = parts[0] + "/" + visit;

            // Add remaining path parts
            for (size_t i = 3; i < parts.size(); ++i) {
                // Map h5 -> motion
                if (parts[i] == "h5") {
                    result += "/motion";
                } else {
                    result += "/" + parts[i];
                }
            }
            return result;
        }
    }

    // No transformation needed
    return path;
}

} // namespace rm

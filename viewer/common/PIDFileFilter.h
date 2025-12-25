#pragma once

#include <string>

namespace PIDNav {

/**
 * Strategy interface for filtering files by type in PID directories.
 *
 * Concrete implementations define which subdirectory to scan and which
 * files to include based on file extension matching.
 */
class FileFilter {
public:
    virtual ~FileFilter() = default;

    /**
     * Returns the subdirectory name within a PID's pre/post directories.
     * Example: "h5", "c3d", "skeleton"
     */
    virtual std::string getSubdirectory() const = 0;

    /**
     * Returns true if the given filename should be included in the file list.
     * Typically checks file extensions.
     */
    virtual bool matches(const std::string& filename) const = 0;

    /**
     * Returns the default filter string for file list filtering.
     * Empty string means no default filter applied.
     */
    virtual std::string getDefaultFilter() const { return ""; }
};

/**
 * Filter for HDF5 motion files (.h5, .hdf5)
 */
class HDFFileFilter : public FileFilter {
public:
    std::string getSubdirectory() const override {
        return "h5";
    }

    bool matches(const std::string& filename) const override {
        auto hasExtension = [](const std::string& str, const std::string& ext) {
            if (str.size() < ext.size()) return false;
            return str.compare(str.size() - ext.size(), ext.size(), ext) == 0;
        };
        return hasExtension(filename, ".h5") || hasExtension(filename, ".hdf5");
    }
};

/**
 * Filter for C3D motion capture files (.c3d)
 * Default filter for trimmed files.
 *
 * Note: C3D files use a different RM path structure than H5 files.
 * - Listing: @pid:{pid}/gait/{pre|post} (RM returns C3D files from Generated_C3D_files)
 * - Fetching: @pid:{pid}/gait/{pre|post}/{filename}.c3d (RM transforms internally)
 */
class C3DFileFilter : public FileFilter {
public:
    std::string getSubdirectory() const override {
        // Empty subdirectory - C3D files are listed at gait/{pre|post} level
        // RM internally handles the Generated_C3D_files directory
        return "";
    }

    bool matches(const std::string& filename) const override {
        auto hasExtension = [](const std::string& str, const std::string& ext) {
            if (str.size() < ext.size()) return false;
            return str.compare(str.size() - ext.size(), ext.size(), ext) == 0;
        };
        return hasExtension(filename, ".c3d");
    }

    std::string getDefaultFilter() const override {
        return "Trimmed_";
    }
};

/**
 * Filter for skeleton definition files (.yaml, .xml)
 */
class SkeletonFileFilter : public FileFilter {
public:
    std::string getSubdirectory() const override {
        return "skeleton";
    }

    bool matches(const std::string& filename) const override {
        auto hasExtension = [](const std::string& str, const std::string& ext) {
            if (str.size() < ext.size()) return false;
            return str.compare(str.size() - ext.size(), ext.size(), ext) == 0;
        };
        return hasExtension(filename, ".yaml") || hasExtension(filename, ".xml");
    }
};

} // namespace PIDNav

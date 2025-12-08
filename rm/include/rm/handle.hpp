#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace rm {

// RAII handle to a fetched resource
// Provides access to data and optionally a local file path
class ResourceHandle {
public:
    // Create handle with data loaded into memory
    explicit ResourceHandle(std::vector<std::byte> data);

    // Create handle with local file path (reads data lazily or from file)
    explicit ResourceHandle(std::filesystem::path local_path, bool owns_file = false);

    // Create handle with both data and local path
    ResourceHandle(std::vector<std::byte> data, std::filesystem::path local_path, bool owns_file = false);

    // Destructor: cleanup temp file if owned
    ~ResourceHandle();

    // Move-only semantics
    ResourceHandle(ResourceHandle&& other) noexcept;
    ResourceHandle& operator=(ResourceHandle&& other) noexcept;
    ResourceHandle(const ResourceHandle&) = delete;
    ResourceHandle& operator=(const ResourceHandle&) = delete;

    // Access the raw data (loads from file if needed)
    const std::vector<std::byte>& data() const;

    // Get local file path (may be empty if data-only)
    const std::filesystem::path& local_path() const { return local_path_; }

    // Convenience: get data as string_view
    std::string_view as_string() const;

    // Check if handle has valid data
    bool valid() const { return !data_.empty() || !local_path_.empty(); }

    // Get size in bytes
    size_t size() const;

private:
    void load_from_file() const;
    void cleanup();

    mutable std::vector<std::byte> data_;
    std::filesystem::path local_path_;
    bool owns_file_;
    mutable bool data_loaded_;
};

} // namespace rm

#include "rm/handle.hpp"
#include "rm/error.hpp"
#include <fstream>

namespace rm {

ResourceHandle::ResourceHandle(std::vector<std::byte> data)
    : data_(std::move(data))
    , local_path_()
    , owns_file_(false)
    , data_loaded_(true) {}

ResourceHandle::ResourceHandle(std::filesystem::path local_path, bool owns_file)
    : data_()
    , local_path_(std::move(local_path))
    , owns_file_(owns_file)
    , data_loaded_(false) {}

ResourceHandle::ResourceHandle(std::vector<std::byte> data, std::filesystem::path local_path, bool owns_file)
    : data_(std::move(data))
    , local_path_(std::move(local_path))
    , owns_file_(owns_file)
    , data_loaded_(true) {}

ResourceHandle::~ResourceHandle() {
    cleanup();
}

ResourceHandle::ResourceHandle(ResourceHandle&& other) noexcept
    : data_(std::move(other.data_))
    , local_path_(std::move(other.local_path_))
    , owns_file_(other.owns_file_)
    , data_loaded_(other.data_loaded_) {
    other.owns_file_ = false;  // Transfer ownership
}

ResourceHandle& ResourceHandle::operator=(ResourceHandle&& other) noexcept {
    if (this != &other) {
        cleanup();
        data_ = std::move(other.data_);
        local_path_ = std::move(other.local_path_);
        owns_file_ = other.owns_file_;
        data_loaded_ = other.data_loaded_;
        other.owns_file_ = false;
    }
    return *this;
}

void ResourceHandle::cleanup() {
    if (owns_file_ && !local_path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(local_path_, ec);
        // Ignore errors on cleanup
    }
}

void ResourceHandle::load_from_file() const {
    if (data_loaded_ || local_path_.empty()) {
        return;
    }

    std::ifstream file(local_path_, std::ios::binary | std::ios::ate);
    if (!file) {
        throw RMError(ErrorCode::IOError, local_path_.string(), "Failed to open file for reading");
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    data_.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data_.data()), size)) {
        throw RMError(ErrorCode::IOError, local_path_.string(), "Failed to read file data");
    }

    data_loaded_ = true;
}

const std::vector<std::byte>& ResourceHandle::data() const {
    load_from_file();
    return data_;
}

std::string_view ResourceHandle::as_string() const {
    const auto& d = data();
    return std::string_view(reinterpret_cast<const char*>(d.data()), d.size());
}

size_t ResourceHandle::size() const {
    if (data_loaded_) {
        return data_.size();
    }
    if (!local_path_.empty()) {
        std::error_code ec;
        auto sz = std::filesystem::file_size(local_path_, ec);
        if (!ec) {
            return static_cast<size_t>(sz);
        }
    }
    return 0;
}

} // namespace rm

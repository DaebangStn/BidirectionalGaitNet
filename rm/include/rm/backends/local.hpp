#pragma once

#include "rm/backend.hpp"
#include <filesystem>

namespace rm {

// Local filesystem backend
// Resolves paths relative to a root directory
class LocalBackend : public Backend {
public:
    explicit LocalBackend(std::filesystem::path root);

    std::string name() const override;
    bool cached() const override { return false; }  // Local files don't need caching
    bool exists(const std::string& path) override;
    bool existsDir(const std::string& path) override;
    std::filesystem::path resolvePath(const std::string& path) override;
    ResourceHandle fetch(const std::string& path) override;
    std::vector<std::string> list(const std::string& pattern) override;

    const std::filesystem::path& root() const { return root_; }

private:
    std::filesystem::path resolve(const std::string& path) const;
    bool match_glob(const std::string& text, const std::string& pattern) const;

    std::filesystem::path root_;
};

} // namespace rm

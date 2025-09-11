#pragma once

#include <string>
#include <unordered_map>

namespace PMuscle {

class URIResolver {
public:
    // Singleton access
    static URIResolver& getInstance();
    
    // Initialize the resolver with default schemes
    void initialize();
    
    // Resolve a URI to an absolute path
    // @data/file.xml -> /path/to/project/data/file.xml
    std::string resolve(const std::string& uri) const;
    
    // Check if a string is a URI (starts with @)
    bool isURI(const std::string& path) const;
    
    // Register a new scheme with its root path
    void registerScheme(const std::string& scheme, const std::string& rootPath);

private:
    URIResolver() = default;
    ~URIResolver() = default;
    URIResolver(const URIResolver&) = delete;
    URIResolver& operator=(const URIResolver&) = delete;
    
    // Get the data root path (from compile-time or fallback)
    std::string getDataRootPath() const;
    
    bool mInitialized = false;
    std::unordered_map<std::string, std::string> mSchemeRoots;
};

} // namespace PMuscle
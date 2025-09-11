#include "UriResolver.h"
#include <filesystem>
#include <iostream>

namespace PMuscle {

URIResolver& URIResolver::getInstance() {
    static URIResolver instance;
    return instance;
}

void URIResolver::initialize() {
    if (mInitialized) return;
    
    // Register default @data/ scheme
    std::string dataRoot = getDataRootPath();
    registerScheme("data", dataRoot);
    
    mInitialized = true;
    std::printf("URIResolver initialized with data root: %s\n", dataRoot.c_str());
}

std::string URIResolver::resolve(const std::string& uri) const {
    if (!isURI(uri)) {
        // Special case: check for */filename pattern and resolve to data/filename
        if (!uri.empty() && uri[0] == '*' && uri.length() > 1 && uri[1] == '/') {
            std::string filename = uri.substr(2); // Remove */
            auto it = mSchemeRoots.find("data");
            if (it != mSchemeRoots.end()) {
                std::filesystem::path fullPath = std::filesystem::path(it->second) / filename;
                return fullPath.string();
            }
        }
        
        // Backwards compatibility: check for ../data/ pattern and resolve to data/
        if (uri.find("../data/") == 0) {
            std::string filename = uri.substr(8); // Remove "../data/"
            auto it = mSchemeRoots.find("data");
            if (it != mSchemeRoots.end()) {
                std::filesystem::path fullPath = std::filesystem::path(it->second) / filename;
                std::printf("URIResolver: Backwards compatibility - resolving %s to %s\n", uri.c_str(), fullPath.string().c_str());
                return fullPath.string();
            }
        }
        
        return uri; // Not a URI, return as-is
    }
    
    std::string scheme, relativePath;
    
    // Handle both @scheme/path and scheme:path formats
    if (uri[0] == '@') {
        // @scheme/path format
        size_t schemeEnd = uri.find('/', 1);
        if (schemeEnd == std::string::npos) {
            std::printf("Warning: Invalid URI format: %s\n", uri.c_str());
            return uri;
        }
        scheme = uri.substr(1, schemeEnd - 1); // Remove @ and get scheme
        relativePath = uri.substr(schemeEnd + 1); // Get path after scheme/
    } else {
        // scheme:path format
        size_t colonPos = uri.find(':');
        if (colonPos == std::string::npos) {
            std::printf("Warning: Invalid URI format: %s\n", uri.c_str());
            return uri;
        }
        scheme = uri.substr(0, colonPos);
        relativePath = uri.substr(colonPos + 1);
    }
    
    auto it = mSchemeRoots.find(scheme);
    if (it == mSchemeRoots.end()) {
        std::printf("Warning: Unknown URI scheme: %s\n", scheme.c_str());
        return uri;
    }
    
    std::filesystem::path fullPath = std::filesystem::path(it->second) / relativePath;
    return fullPath.string();
}

bool URIResolver::isURI(const std::string& path) const {
    if (path.empty()) return false;
    
    // Check for @scheme/path format
    if (path[0] == '@') return true;
    
    // Check for scheme:path format (must have colon but not be an absolute path)
    size_t colonPos = path.find(':');
    if (colonPos != std::string::npos && colonPos > 0) {
        // Make sure it's not a Windows drive letter (C:) or absolute path
        if (colonPos == 1 && path.length() > 2 && (path[2] == '\\' || path[2] == '/')) {
            return false; // Windows drive letter
        }
        // Make sure scheme part doesn't contain path separators
        std::string scheme = path.substr(0, colonPos);
        return scheme.find('/') == std::string::npos && scheme.find('\\') == std::string::npos;
    }
    
    return false;
}

void URIResolver::registerScheme(const std::string& scheme, const std::string& rootPath) {
    mSchemeRoots[scheme] = rootPath;
}

std::string URIResolver::getDataRootPath() const {
#ifdef PROJECT_ROOT
    std::filesystem::path projectRoot = PROJECT_ROOT;
    std::filesystem::path dataPath = projectRoot / "data";
    return dataPath.string();
#elif defined(DATA_ROOT_PATH)
    return DATA_ROOT_PATH;
#else
    // Fallback to current working directory
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path dataPath = currentPath / "data";
    return dataPath.string();
#endif
}

} // namespace PMuscle
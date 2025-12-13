#include "rm/global.hpp"
#include "rm/error.hpp"
#include <iostream>

namespace rm {

ResourceManager& getManager() {
#ifdef PROJECT_ROOT
    static ResourceManager instance(std::string(PROJECT_ROOT) + "/data/rm_config.yaml");
#else
    static ResourceManager instance("data/rm_config.yaml");
#endif
    return instance;
}

std::string resolve(const std::string& uri) {
    // Handle non-URI paths (no @ prefix)
    if (uri.empty() || uri[0] != '@') {
        return uri;
    }

    try {
        auto path = getManager().resolve(uri);
        if (!path.empty()) {
            return path.string();
        }
    } catch (const RMError& e) {
        std::cerr << "[rm] Failed to resolve URI: " << uri << " - " << e.what() << std::endl;
    }
    return uri;  // Return original on failure
}

} // namespace rm

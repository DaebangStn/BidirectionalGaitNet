#include "sim/UriResolver.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    // Initialize UriResolver
    PMuscle::URIResolver::getInstance().initialize();
    
    // Test various URI patterns including backwards compatibility
    std::vector<std::string> testPaths = {
        "@data/skeleton_gaitnet_narrow_model.xml",
        "@data/motion/walk.bvh", 
        "@data/muscle_gaitnet.xml",
        "data:skeleton_gaitnet_narrow_model.xml",
        "*/skeleton_gaitnet_narrow_model.xml",
        "../data/skeleton_gaitnet_narrow_model.xml",  // Backwards compatibility test
        "../data/motion/walk.bvh",                     // Backwards compatibility test
        "../data/muscle_gaitnet.xml",                  // Backwards compatibility test
        "regular/path/file.xml"
    };
    
    std::cout << "=== URI Resolver Test ===" << std::endl;
    for (const auto& path : testPaths) {
        std::string resolved = PMuscle::URIResolver::getInstance().resolve(path);
        std::cout << "Input:  " << path << std::endl;
        std::cout << "Output: " << resolved << std::endl;
        std::cout << "Is URI: " << (PMuscle::URIResolver::getInstance().isURI(path) ? "Yes" : "No") << std::endl;
        std::cout << "---" << std::endl;
    }
    
    return 0;
}
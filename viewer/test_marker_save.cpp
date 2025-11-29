// Test program for marker save cyclic consistency
// Compile: add to CMakeLists.txt or compile manually
// Run: ./test_marker_save

#include "RenderCharacter.h"
#include "Log.h"
#include <iostream>

int main()
{
    std::string skelPath = "data/skeleton/base.xml";
    std::string markerPath = "data/marker/default.xml";
    std::string outputPath = "data/marker/test.xml";

    std::cout << "Testing marker save cyclic consistency..." << std::endl;
    std::cout << "Loading skeleton from: " << skelPath << std::endl;

    try {
        RenderCharacter character(skelPath);

        std::cout << "Loading markers from: " << markerPath << std::endl;
        character.loadMarkers(markerPath);

        const auto& markers = character.getMarkers();
        std::cout << "Loaded " << markers.size() << " markers" << std::endl;

        // Print first few markers for verification
        for (size_t i = 0; i < std::min(size_t(5), markers.size()); ++i) {
            std::cout << "  [" << i << "] " << markers[i].name
                      << " bn=" << markers[i].bodyNode->getName()
                      << " offset=(" << markers[i].offset[0] << ", "
                      << markers[i].offset[1] << ", " << markers[i].offset[2] << ")"
                      << std::endl;
        }

        std::cout << "Saving markers to: " << outputPath << std::endl;
        if (character.saveMarkersToXml(outputPath)) {
            std::cout << "SUCCESS: Markers saved!" << std::endl;
        } else {
            std::cout << "FAILED: Could not save markers" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nNow compare the files with: diff " << markerPath << " " << outputPath << std::endl;
    return 0;
}

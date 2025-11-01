// Simple test to verify transformation fix
#include "DARTHelper.h"
#include "Character.h"
#include "SurgeryExecutor.h"
#include <iostream>
#include <iomanip>

using namespace PMuscle;

void printPositions(const dart::dynamics::SkeletonPtr& skel, const std::string& label) {
    std::cout << label << ":" << std::endl;
    int count = std::min(5, (int)skel->getNumBodyNodes());
    for (int i = 0; i < count; i++) {
        auto bn = skel->getBodyNode(i);
        Eigen::Vector3d pos = bn->getWorldTransform().translation();
        std::cout << "  [" << i << "] " << bn->getName()
                  << " @ [" << std::fixed << std::setprecision(4)
                  << pos[0] << ", " << pos[1] << ", " << pos[2] << "]" << std::endl;
    }
}

int main() {
    std::cout << "=== Transformation Fix Test ===" << std::endl << std::endl;

    // 1. Load XML
    std::cout << "1. Loading XML skeleton..." << std::endl;
    auto skel_xml = BuildFromFile("@data/skeleton/base.xml", 0);
    if (!skel_xml) {
        std::cerr << "Failed to load XML" << std::endl;
        return 1;
    }
    printPositions(skel_xml, "XML Positions");

    // 2. Export to YAML
    std::cout << std::endl << "2. Exporting to YAML..." << std::endl;
    auto character = std::make_shared<Character>(skel_xml);
    SurgeryExecutor executor(character);
    executor.setOriginalSkeletonPath("@data/skeleton/base.xml");
    executor.exportSkeleton("@data/skeleton/test_transform_output.yaml");
    std::cout << "   Exported to test_transform_output.yaml" << std::endl;

    // 3. Load YAML
    std::cout << std::endl << "3. Loading YAML skeleton..." << std::endl;
    auto skel_yaml = BuildFromFile("@data/skeleton/test_transform_output.yaml", 0);
    if (!skel_yaml) {
        std::cerr << "Failed to load YAML" << std::endl;
        return 1;
    }
    printPositions(skel_yaml, "YAML Positions");

    // 4. Compare
    std::cout << std::endl << "4. Comparing positions..." << std::endl;
    int count = std::min(5, (int)skel_xml->getNumBodyNodes());
    bool match = true;
    double tolerance = 1e-4;

    for (int i = 0; i < count; i++) {
        auto bn_xml = skel_xml->getBodyNode(i);
        auto bn_yaml = skel_yaml->getBodyNode(i);
        Eigen::Vector3d pos_xml = bn_xml->getWorldTransform().translation();
        Eigen::Vector3d pos_yaml = bn_yaml->getWorldTransform().translation();
        double diff = (pos_xml - pos_yaml).norm();

        if (diff > tolerance) {
            std::cout << "  MISMATCH [" << i << "] " << bn_xml->getName()
                      << ": diff = " << diff << std::endl;
            match = false;
        }
    }

    if (match) {
        std::cout << "  ✓ All positions match!" << std::endl;
    } else {
        std::cout << "  ✗ Positions DO NOT match!" << std::endl;
        return 1;
    }

    std::cout << std::endl << "=== Test PASSED ===" << std::endl;
    return 0;
}

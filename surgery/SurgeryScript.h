#pragma once

#include "SurgeryOperation.h"
#include <string>
#include <vector>
#include <memory>

namespace PMuscle {

class SurgeryScript {
public:
    // Load operations from YAML file
    static std::vector<std::unique_ptr<SurgeryOperation>> loadFromFile(const std::string& filepath);
    
    // Save operations to YAML file
    static void saveToFile(
        const std::vector<std::unique_ptr<SurgeryOperation>>& ops,
        const std::string& filepath,
        const std::string& description = "");
    
    // Generate human-readable preview of operations
    static std::string preview(const std::vector<std::unique_ptr<SurgeryOperation>>& ops);
    
private:
    // Factory method to create operation from YAML node
    static std::unique_ptr<SurgeryOperation> createOperation(const YAML::Node& node);
};

} // namespace PMuscle


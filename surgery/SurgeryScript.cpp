#include "SurgeryScript.h"
#include "UriResolver.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace PMuscle {

std::vector<std::unique_ptr<SurgeryOperation>> SurgeryScript::loadFromFile(const std::string& filepath) {
    std::vector<std::unique_ptr<SurgeryOperation>> operations;
    
    // Resolve URI if needed
    URIResolver& resolver = URIResolver::getInstance();
    resolver.initialize();
    std::string resolved_path = resolver.resolve(filepath);
    
    std::cout << "[SurgeryScript] Loading from: " << resolved_path << std::endl;
    
    try {
        YAML::Node config = YAML::LoadFile(resolved_path);
        
        // Check version (optional)
        if (config["version"]) {
            std::string version = config["version"].as<std::string>();
            std::cout << "[SurgeryScript] Version: " << version << std::endl;
        }
        
        // Load description (optional)
        if (config["description"]) {
            std::string desc = config["description"].as<std::string>();
            std::cout << "[SurgeryScript] Description: " << desc << std::endl;
        }
        
        // Load operations
        if (!config["operations"]) {
            std::cerr << "[SurgeryScript] Error: No 'operations' section found" << std::endl;
            return operations;
        }
        
        const YAML::Node& ops_node = config["operations"];
        for (size_t i = 0; i < ops_node.size(); ++i) {
            try {
                auto op = createOperation(ops_node[i]);
                if (op) {
                    operations.push_back(std::move(op));
                }
            } catch (const std::exception& e) {
                std::cerr << "[SurgeryScript] Error parsing operation " << i 
                         << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "[SurgeryScript] Loaded " << operations.size() << " operation(s)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[SurgeryScript] Error loading file: " << e.what() << std::endl;
    }
    
    return operations;
}

void SurgeryScript::saveToFile(
    const std::vector<std::unique_ptr<SurgeryOperation>>& ops,
    const std::string& filepath,
    const std::string& description) {
    
    YAML::Node config;
    config["version"] = "1.0";
    
    if (!description.empty()) {
        config["description"] = description;
    }
    
    YAML::Node operations;
    for (const auto& op : ops) {
        operations.push_back(op->toYAML());
    }
    config["operations"] = operations;
    
    std::ofstream fout(filepath);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    fout << config;
    fout.close();
    
    std::cout << "[SurgeryScript] Saved " << ops.size() << " operation(s) to " << filepath << std::endl;
}

std::string SurgeryScript::preview(const std::vector<std::unique_ptr<SurgeryOperation>>& ops) {
    std::ostringstream oss;
    oss << "Surgery Script Preview\n";
    oss << "======================\n\n";
    oss << "Total operations: " << ops.size() << "\n\n";
    
    for (size_t i = 0; i < ops.size(); ++i) {
        oss << (i + 1) << ". " << ops[i]->getDescription() << "\n";
    }
    
    return oss.str();
}

std::unique_ptr<SurgeryOperation> SurgeryScript::createOperation(const YAML::Node& node) {
    if (!node["type"]) {
        throw std::runtime_error("Operation missing 'type' field");
    }
    
    std::string type = node["type"].as<std::string>();
    
    if (type == "reset_muscles") {
        return ResetMusclesOp::fromYAML(node);
    } else if (type == "distribute_passive_force") {
        return DistributePassiveForceOp::fromYAML(node);
    } else if (type == "relax_passive_force") {
        return RelaxPassiveForceOp::fromYAML(node);
    } else if (type == "remove_anchor") {
        return RemoveAnchorOp::fromYAML(node);
    } else if (type == "copy_anchor") {
        return CopyAnchorOp::fromYAML(node);
    } else if (type == "edit_anchor_position") {
        return EditAnchorPositionOp::fromYAML(node);
    } else if (type == "edit_anchor_weights") {
        return EditAnchorWeightsOp::fromYAML(node);
    } else if (type == "add_bodynode_to_anchor") {
        return AddBodyNodeToAnchorOp::fromYAML(node);
    } else if (type == "remove_bodynode_from_anchor") {
        return RemoveBodyNodeFromAnchorOp::fromYAML(node);
    } else if (type == "export_muscles") {
        return ExportMusclesOp::fromYAML(node);
    } else {
        std::cerr << "[SurgeryScript] Unknown operation type: " << type << std::endl;
        return nullptr;
    }
}

} // namespace PMuscle


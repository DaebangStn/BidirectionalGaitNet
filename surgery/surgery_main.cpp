#include <iostream>
#include <string>
#include <cstring>
#include "SurgeryExecutor.h"
#include "SurgeryScript.h"
#include "SurgeryOperation.h"

// Default values
const std::string DEFAULT_SKELETON = "@data/skeleton_gaitnet_narrow_model.xml";
const std::string DEFAULT_MUSCLE = "@data/muscle_gaitnet.xml";
const std::string DEFAULT_SCRIPT = "@data/recorded_surgery.yaml";

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --skeleton PATH    Path to skeleton XML file" << std::endl;
    std::cout << "                     (default: " << DEFAULT_SKELETON << ")" << std::endl;
    std::cout << "  --muscle PATH      Path to muscle XML file" << std::endl;
    std::cout << "                     (default: " << DEFAULT_MUSCLE << ")" << std::endl;
    std::cout << "  --script PATH      Path to surgery script YAML file" << std::endl;
    std::cout << "                     (default: " << DEFAULT_SCRIPT << ")" << std::endl;
    std::cout << "  --help, -h         Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Use all defaults" << std::endl;
    std::cout << "  " << programName << std::endl;
    std::cout << std::endl;
    std::cout << "  # Use custom script with default skeleton and muscle" << std::endl;
    std::cout << "  " << programName << " --script data/my_surgery.yaml" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Specify all parameters" << std::endl;
    std::cout << "  " << programName << " --skeleton data/skeleton_gaitnet_narrow_model.xml \\" << std::endl;
    std::cout << "                     --muscle data/muscle_gaitnet.xml \\" << std::endl;
    std::cout << "                     --script data/example_surgery.yaml" << std::endl;
}

bool hasExportOperation(const std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>& ops) {
    for (const auto& op : ops) {
        if (op->getType() == "export_muscles") {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    std::cout << "==============================================================================" << std::endl;
    std::cout << "Surgery Tool - Standalone Surgery Script Executor" << std::endl;
    std::cout << "==============================================================================" << std::endl;
    std::cout << std::endl;

    // Default values
    std::string skeleton_path = DEFAULT_SKELETON;
    std::string muscle_path = DEFAULT_MUSCLE;
    std::string script_path = DEFAULT_SCRIPT;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--skeleton") {
            if (i + 1 < argc) {
                skeleton_path = argv[++i];
            } else {
                std::cerr << "Error: --skeleton requires a path argument" << std::endl;
                return 1;
            }
        } else if (arg == "--muscle") {
            if (i + 1 < argc) {
                muscle_path = argv[++i];
            } else {
                std::cerr << "Error: --muscle requires a path argument" << std::endl;
                return 1;
            }
        } else if (arg == "--script") {
            if (i + 1 < argc) {
                script_path = argv[++i];
            } else {
                std::cerr << "Error: --script requires a path argument" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            std::cout << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Convert relative data/ paths to @data/ format for URIResolver
    auto convertPath = [](const std::string& path) -> std::string {
        if (path.find("data/") == 0) {
            return "@" + path;
        }
        return path;
    };
    
    skeleton_path = convertPath(skeleton_path);
    muscle_path = convertPath(muscle_path);
    script_path = convertPath(script_path);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Skeleton: " << skeleton_path << std::endl;
    std::cout << "  Muscles:  " << muscle_path << std::endl;
    std::cout << "  Script:   " << script_path << std::endl;
    std::cout << std::endl;

    try {
        // Create surgery executor
        std::cout << "Initializing surgery executor..." << std::endl;
        PMuscle::SurgeryExecutor executor;

        // Load character
        std::cout << "Loading character..." << std::endl;
        executor.loadCharacter(skeleton_path, muscle_path, mus);
        std::cout << "Character loaded successfully!" << std::endl;
        std::cout << std::endl;

        // Load surgery script
        std::cout << "Loading surgery script..." << std::endl;
        auto operations = PMuscle::SurgeryScript::loadFromFile(script_path);
        
        if (operations.empty()) {
            std::cerr << "Error: No operations loaded from script!" << std::endl;
            return 1;
        }

        std::cout << "Loaded " << operations.size() << " operation(s)" << std::endl;
        std::cout << std::endl;

        // Check for export operation
        if (!hasExportOperation(operations)) {
            std::cout << "WARNING: No export operation found in script!" << std::endl;
            std::cout << "         Muscles will be modified but not saved." << std::endl;
            std::cout << "         Add an 'export_muscles' operation to save results." << std::endl;
            std::cout << std::endl;
        }

        // Execute operations
        std::cout << "==============================================================================" << std::endl;
        std::cout << "Executing Surgery Script" << std::endl;
        std::cout << "==============================================================================" << std::endl;
        std::cout << std::endl;

        int successCount = 0;
        int failCount = 0;

        for (size_t i = 0; i < operations.size(); ++i) {
            std::cout << "Operation " << (i + 1) << "/" << operations.size() << ": " 
                      << operations[i]->getDescription() << std::endl;

            bool success = operations[i]->execute(&executor);
            
            if (success) {
                successCount++;
                std::cout << "  ✓ Success" << std::endl;
            } else {
                failCount++;
                std::cout << "  ✗ FAILED" << std::endl;
            }
            std::cout << std::endl;
        }

        // Summary
        std::cout << "==============================================================================" << std::endl;
        std::cout << "Execution Summary" << std::endl;
        std::cout << "==============================================================================" << std::endl;
        std::cout << "Total operations: " << operations.size() << std::endl;
        std::cout << "Successful:       " << successCount << std::endl;
        std::cout << "Failed:           " << failCount << std::endl;
        std::cout << std::endl;

        if (failCount == 0) {
            std::cout << "✓ All operations completed successfully!" << std::endl;
            
            // Warn if no export operation
            if (!hasExportOperation(operations)) {
                std::cout << std::endl;
                std::cout << "⚠ WARNING: No export_muscles operation found!" << std::endl;
                std::cout << "           Modified muscles were NOT saved to disk." << std::endl;
                std::cout << "           Add an 'export_muscles' operation to your script to save the results." << std::endl;
            }
            
            return 0;
        } else {
            std::cout << "✗ Some operations failed. Check the output above for details." << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


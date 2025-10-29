#include <iostream>
#include <string>
#include <cstring>
#include "SurgeryExecutor.h"
#include "SurgeryScript.h"
#include "SurgeryOperation.h"
#include "Log.h"

// Default values
const std::string DEFAULT_SKELETON = "@data/skeleton/gaitnet_narrow_model.xml";
const std::string DEFAULT_MUSCLE = "@data/muscle/distribute_lower_only.xml";
const std::string DEFAULT_SCRIPT = "@data/recorded_surgery.yaml";

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    LOG_INFO("");
    std::cout << "Options:" << std::endl;
    std::cout << "  --skeleton PATH    Path to skeleton XML file" << std::endl;
    std::cout << "                     (default: " << DEFAULT_SKELETON << ")" << std::endl;
    std::cout << "  --muscle PATH      Path to muscle XML file" << std::endl;
    std::cout << "                     (default: " << DEFAULT_MUSCLE << ")" << std::endl;
    std::cout << "  --script PATH      Path to surgery script YAML file" << std::endl;
    std::cout << "                     (default: " << DEFAULT_SCRIPT << ")" << std::endl;
    std::cout << "  --help, -h         Show this help message" << std::endl;
    LOG_INFO("");
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Use all defaults" << std::endl;
    std::cout << "  " << programName << std::endl;
    LOG_INFO("");
    std::cout << "  # Use custom script with default skeleton and muscle" << std::endl;
    std::cout << "  " << programName << " --script data/my_surgery.yaml" << std::endl;
    LOG_INFO("");
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
    LOG_INFO("==============================================================================");
    LOG_INFO("Surgery Tool - Standalone Surgery Script Executor");
    LOG_INFO("==============================================================================");
    LOG_INFO("");

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
                LOG_ERROR("Error: --skeleton requires a path argument");
                return 1;
            }
        } else if (arg == "--muscle") {
            if (i + 1 < argc) {
                muscle_path = argv[++i];
            } else {
                LOG_ERROR("Error: --muscle requires a path argument");
                return 1;
            }
        } else if (arg == "--script") {
            if (i + 1 < argc) {
                script_path = argv[++i];
            } else {
                LOG_ERROR("Error: --script requires a path argument");
                return 1;
            }
        } else {
            LOG_ERROR("Error: Unknown argument: " << arg);
            LOG_INFO("");
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

    LOG_INFO("Configuration:");
    LOG_INFO("  Skeleton: " << skeleton_path);
    LOG_INFO("  Muscles:  " << muscle_path);
    LOG_INFO("  Script:   " << script_path);
    LOG_INFO("");

    try {
        // Create surgery executor
        LOG_INFO("Initializing surgery executor...");
        PMuscle::SurgeryExecutor executor;

        // Load character
        LOG_INFO("Loading character...");
        executor.loadCharacter(skeleton_path, muscle_path, mus);
        LOG_INFO("Character loaded successfully!");
        LOG_INFO("");

        // Load surgery script
        LOG_INFO("Loading surgery script...");
        auto operations = PMuscle::SurgeryScript::loadFromFile(script_path);
        
        if (operations.empty()) {
            LOG_ERROR("Error: No operations loaded from script!");
            return 1;
        }

        LOG_INFO("Loaded " << operations.size() << " operation(s)");
        LOG_INFO("");

        // Check for export operation
        if (!hasExportOperation(operations)) {
            LOG_WARN("WARNING: No export operation found in script!");
            LOG_WARN("         Muscles will be modified but not saved.");
            LOG_WARN("         Add an 'export_muscles' operation to save results.");
            LOG_INFO("");
        }

        // Execute operations
        LOG_INFO("==============================================================================");
        LOG_INFO("Executing Surgery Script");
        LOG_INFO("==============================================================================");
        LOG_INFO("");

        int successCount = 0;
        int failCount = 0;

        for (size_t i = 0; i < operations.size(); ++i) {
            LOG_INFO("Operation " << (i + 1) << "/" << operations.size() << ": " << operations[i]->getDescription());

            bool success = operations[i]->execute(&executor);
            
            if (success) {
                successCount++;
                LOG_INFO("  ✓ Success");
            } else {
                failCount++;
                LOG_ERROR("  ✗ FAILED");
            }
            LOG_INFO("");
        }

        // Summary
        LOG_INFO("==============================================================================");
        LOG_INFO("Execution Summary");
        LOG_INFO("==============================================================================");
        LOG_INFO("Total operations: " << operations.size());
        LOG_INFO("Successful:       " << successCount);
        LOG_INFO("Failed:           " << failCount);
        LOG_INFO("");

        if (failCount == 0) {
            LOG_INFO("✓ All operations completed successfully!");
            
            // Warn if no export operation
            if (!hasExportOperation(operations)) {
                LOG_INFO("");
                LOG_WARN("⚠ WARNING: No export_muscles operation found!");
                LOG_WARN("           Modified muscles were NOT saved to disk.");
                LOG_WARN("           Add an 'export_muscles' operation to your script to save the results.");
            }
            
            return 0;
        } else {
            LOG_ERROR("✗ Some operations failed. Check the output above for details.");
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


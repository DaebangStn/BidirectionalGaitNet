/**
 * @file skeleton_global_export.cpp
 * @brief Export global transforms of skeleton body nodes to YAML
 *
 * This tool loads a skeleton YAML, sets T-pose (all DOFs = 0),
 * computes FK to get global transforms, and exports to skeleton/global/ directory.
 *
 * Usage:
 *   skeleton_global_export -i /path/to/skeleton.yaml
 *   skeleton_global_export -i "@pid:29792292/pre/skeleton/Trimmed_walk01-Dynamic.yaml"
 */

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <filesystem>

#include "DARTHelper.h"
#include "rm/rm.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    po::options_description desc("Skeleton Global Transform Exporter");
    desc.add_options()
        ("help,h", "Show help")
        ("input,i", po::value<std::string>(), "Input skeleton YAML (supports @pid: paths)")
        ("verbose,v", po::bool_switch(), "Verbose output");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    if (vm.count("help") || !vm.count("input")) {
        std::cout << "Skeleton Global Transform Exporter\n\n";
        std::cout << "Exports global transforms of skeleton body nodes in T-pose.\n\n";
        std::cout << "Usage: " << argv[0] << " -i <skeleton.yaml>\n\n";
        std::cout << desc << std::endl;
        std::cout << "\nExamples:\n";
        std::cout << "  " << argv[0] << " -i @pid:29792292/pre/skeleton/Trimmed_walk01-Dynamic.yaml\n";
        std::cout << "  " << argv[0] << " -i /path/to/skeleton.yaml -v\n";
        return vm.count("help") ? 0 : 1;
    }

    std::string inputPath = vm["input"].as<std::string>();
    bool verbose = vm["verbose"].as<bool>();

    // Resolve the input path
    std::string resolvedInput = rm::resolve(inputPath);
    if (verbose) {
        std::cout << "Input: " << inputPath << std::endl;
        std::cout << "Resolved: " << resolvedInput << std::endl;
    }

    if (!fs::exists(resolvedInput)) {
        std::cerr << "Error: Input file does not exist: " << resolvedInput << std::endl;
        return 1;
    }

    // Output to global/ subdirectory with same filename
    fs::path inputP(resolvedInput);
    fs::path globalDir = inputP.parent_path() / "global";
    fs::create_directories(globalDir);
    std::string outputPath = (globalDir / inputP.filename()).string();

    if (verbose) {
        std::cout << "Output: " << outputPath << std::endl;
    }

    // Load skeleton using BuildFromFile (same as RenderCharacter)
    auto skel = BuildFromFile(inputPath);
    if (!skel) {
        std::cerr << "Error: Failed to load skeleton: " << inputPath << std::endl;
        return 1;
    }

    if (verbose) {
        std::cout << "Skeleton loaded: " << skel->getName() << std::endl;
        std::cout << "Body nodes: " << skel->getNumBodyNodes() << std::endl;
        std::cout << "DOFs: " << skel->getNumDofs() << std::endl;
    }

    // Set T-pose (all DOFs = 0)
    skel->setPositions(Eigen::VectorXd::Zero(skel->getNumDofs()));

    // Load original YAML to get body sizes
    YAML::Node skelYaml = YAML::LoadFile(resolvedInput);
    std::map<std::string, std::vector<double>> bodySizes;

    if (skelYaml["skeleton"] && skelYaml["skeleton"]["nodes"]) {
        for (const auto& node : skelYaml["skeleton"]["nodes"]) {
            std::string name = node["name"].as<std::string>();
            if (node["body"] && node["body"]["size"]) {
                auto sizeNode = node["body"]["size"];
                bodySizes[name] = {
                    sizeNode[0].as<double>(),
                    sizeNode[1].as<double>(),
                    sizeNode[2].as<double>()
                };
            }
        }
    }

    // Generate output
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "source" << YAML::Value << inputP.filename().string();

    // Timestamp
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    out << YAML::Key << "timestamp" << YAML::Value << buf;

    // Metadata
    out << YAML::Key << "num_body_nodes" << YAML::Value << (int)skel->getNumBodyNodes();
    out << YAML::Key << "num_dofs" << YAML::Value << (int)skel->getNumDofs();

    out << YAML::Key << "nodes" << YAML::Value << YAML::BeginSeq;
    for (size_t i = 0; i < skel->getNumBodyNodes(); ++i) {
        auto* bn = skel->getBodyNode(i);
        std::string name = bn->getName();

        Eigen::Isometry3d T = bn->getWorldTransform();
        Eigen::Vector3d pos = T.translation();
        Eigen::AngleAxisd aa(T.rotation());

        out << YAML::BeginMap;
        out << YAML::Key << "name" << YAML::Value << name;

        // Body size in meters (from original YAML)
        if (bodySizes.find(name) != bodySizes.end()) {
            out << YAML::Key << "size_meters" << YAML::Flow << bodySizes[name];
        } else {
            // Fallback: get from body node shape if available
            bn->eachShapeNodeWith<dart::dynamics::VisualAspect>([&](dart::dynamics::ShapeNode* sn) {
                auto boxShape = std::dynamic_pointer_cast<dart::dynamics::BoxShape>(sn->getShape());
                if (boxShape) {
                    Eigen::Vector3d size = boxShape->getSize();
                    out << YAML::Key << "size_meters" << YAML::Flow
                        << std::vector<double>{size.x(), size.y(), size.z()};
                }
                return false;  // stop after first shape
            });
        }

        // Global position
        out << YAML::Key << "global_position" << YAML::Flow
            << std::vector<double>{pos.x(), pos.y(), pos.z()};

        // Translation w.r.t. parent body node (in parent's local frame)
        auto* parentBn = bn->getParentBodyNode();
        if (parentBn) {
            Eigen::Isometry3d parentT = parentBn->getWorldTransform();
            Eigen::Vector3d localPos = parentT.inverse() * pos;
            out << YAML::Key << "local_translation" << YAML::Flow
                << std::vector<double>{localPos.x(), localPos.y(), localPos.z()};
        } else {
            // Root node - local translation is same as global position
            out << YAML::Key << "local_translation" << YAML::Flow
                << std::vector<double>{pos.x(), pos.y(), pos.z()};
        }

        // Global rotation as axis-angle
        out << YAML::Key << "global_rotation" << YAML::BeginMap;
        out << YAML::Key << "axis" << YAML::Flow
            << std::vector<double>{aa.axis().x(), aa.axis().y(), aa.axis().z()};
        out << YAML::Key << "angle_deg" << YAML::Value << (aa.angle() * 180.0 / M_PI);
        out << YAML::EndMap;

        out << YAML::EndMap;

        if (verbose) {
            std::cout << "  " << name << ": pos=[" << pos.transpose()
                      << "] angle=" << (aa.angle() * 180.0 / M_PI) << " deg" << std::endl;
        }
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;

    // Write output file
    std::ofstream fout(outputPath);
    if (!fout.is_open()) {
        std::cerr << "Error: Failed to open output file: " << outputPath << std::endl;
        return 1;
    }
    fout << out.c_str();
    fout.close();

    std::cout << "Exported: " << outputPath << std::endl;
    return 0;
}

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <set>
#include <map>
#include <filesystem>
#include <cmath>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include "SurgeryExecutor.h"
#include "SurgeryScript.h"
#include "SurgeryOperation.h"
#include "Log.h"
#include "rm/rm.hpp"

// Default values
const std::string DEFAULT_SKELETON = "@data/skeleton/trimmed_unified.yaml";
const std::string DEFAULT_MUSCLE = "@data/muscle/base.yaml";
const std::string DEFAULT_SCRIPT = "@data/recorded_surgery.yaml";
const std::string DEFAULT_CONFIG = "@data/config/surgery_panel.yaml";

// ============================================================================
// Config structs for PID mode (mirrors SurgeryPanel structs)
// ============================================================================

struct ROMTrialEntry {
    std::string trial_name;
    float angle_deg = 0.0f;
};

struct TALSideConfig {
    std::vector<ROMTrialEntry> rom_trials;
    std::map<std::string, std::vector<std::string>> search_groups;  // group_name -> muscles
    bool enabled = false;
};

struct DHLSideConfig {
    std::string target_muscle;
    std::string donor_muscle;
    int donor_anchor = 3;
    std::vector<int> remove_anchors;
    std::vector<std::string> rom_trials;
    std::vector<std::string> muscles;
    float popliteal_angle_deg = 0.0f;
    bool enabled = false;
};

struct FDOSideConfig {
    std::string joint;
    std::string ref_muscle;
    int ref_anchor = 0;
    float rotation_axis[3] = {0, 1, 0};
    float search_direction[3] = {0, -1, 0};
    float angle_deg = 0.0f;
    bool enabled = false;
};

struct RFTSideConfig {
    std::vector<std::string> target_muscles;
    std::string donor_muscle;
    std::vector<int> remove_anchors;
    std::vector<int> copy_donor_anchors;
    bool enabled = false;
};

struct SEMLSConfig {
    TALSideConfig tal_left, tal_right;
    DHLSideConfig dhl_left, dhl_right;
    FDOSideConfig fdo_left, fdo_right;
    RFTSideConfig rft_left, rft_right;
};

// ============================================================================
// YAML config loaders (mirrors SurgeryPanel's static loaders)
// ============================================================================

static void loadTALSide(const YAML::Node& node, TALSideConfig& cfg) {
    if (!node) return;
    if (node["rom_trials"]) {
        cfg.rom_trials.clear();
        for (const auto& t : node["rom_trials"]) {
            ROMTrialEntry entry;
            entry.trial_name = t["trial"].as<std::string>();
            if (t["angle"]) entry.angle_deg = t["angle"].as<float>();
            cfg.rom_trials.push_back(entry);
        }
    }
    if (node["search_groups"]) {
        cfg.search_groups.clear();
        for (const auto& sg : node["search_groups"]) {
            std::string name = sg.first.as<std::string>();
            std::vector<std::string> muscles;
            for (const auto& m : sg.second) muscles.push_back(m.as<std::string>());
            cfg.search_groups[name] = muscles;
        }
    }
}

static void loadDHLSide(const YAML::Node& node, DHLSideConfig& cfg) {
    if (!node) return;
    if (node["target_muscle"]) cfg.target_muscle = node["target_muscle"].as<std::string>();
    if (node["donor_muscle"]) cfg.donor_muscle = node["donor_muscle"].as<std::string>();
    if (node["donor_anchor"]) cfg.donor_anchor = node["donor_anchor"].as<int>();
    if (node["remove_anchors"]) {
        cfg.remove_anchors.clear();
        for (const auto& a : node["remove_anchors"]) cfg.remove_anchors.push_back(a.as<int>());
    }
    if (node["rom_trials"]) {
        cfg.rom_trials.clear();
        for (const auto& t : node["rom_trials"]) cfg.rom_trials.push_back(t.as<std::string>());
    }
    if (node["muscles"]) {
        cfg.muscles.clear();
        for (const auto& m : node["muscles"]) cfg.muscles.push_back(m.as<std::string>());
    }
    if (node["popliteal_angle"]) cfg.popliteal_angle_deg = node["popliteal_angle"].as<float>();
}

static void loadFDOSide(const YAML::Node& node, FDOSideConfig& cfg) {
    if (!node) return;
    if (node["joint"]) cfg.joint = node["joint"].as<std::string>();
    if (node["ref_muscle"]) cfg.ref_muscle = node["ref_muscle"].as<std::string>();
    if (node["ref_anchor"]) cfg.ref_anchor = node["ref_anchor"].as<int>();
    if (node["rotation_axis"] && node["rotation_axis"].size() == 3) {
        for (int i = 0; i < 3; ++i) cfg.rotation_axis[i] = node["rotation_axis"][i].as<float>();
    }
    if (node["search_direction"] && node["search_direction"].size() == 3) {
        for (int i = 0; i < 3; ++i) cfg.search_direction[i] = node["search_direction"][i].as<float>();
    }
    if (node["angle_deg"]) cfg.angle_deg = node["angle_deg"].as<float>();
}

static void loadRFTSide(const YAML::Node& node, RFTSideConfig& cfg) {
    if (!node) return;
    if (node["target_muscles"]) {
        cfg.target_muscles.clear();
        for (const auto& m : node["target_muscles"]) cfg.target_muscles.push_back(m.as<std::string>());
    }
    if (node["donor_muscle"]) cfg.donor_muscle = node["donor_muscle"].as<std::string>();
    if (node["remove_anchors"]) {
        cfg.remove_anchors.clear();
        for (const auto& a : node["remove_anchors"]) cfg.remove_anchors.push_back(a.as<int>());
    }
    if (node["copy_donor_anchors"]) {
        cfg.copy_donor_anchors.clear();
        for (const auto& a : node["copy_donor_anchors"]) cfg.copy_donor_anchors.push_back(a.as<int>());
    }
}

// ============================================================================
// Parse procedure filter string → set of lowercase procedure names
// ============================================================================

static std::set<std::string> parseProcedureFilter(const std::string& filter_str) {
    std::set<std::string> result;
    if (filter_str.empty()) return result;
    std::istringstream ss(filter_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos) {
            std::string proc = token.substr(start, end - start + 1);
            // lowercase
            for (auto& c : proc) c = std::tolower(c);
            result.insert(proc);
        }
    }
    return result;
}

// ============================================================================
// Build TAL operations
// ============================================================================

static void buildTALOps(const TALSideConfig& cfg, const std::string& side_label,
                        std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>& ops) {
    if (!cfg.enabled || cfg.rom_trials.empty() || cfg.search_groups.empty()) return;

    LOG_INFO("[PID] Building TAL " << side_label << " operations");

    std::vector<PMuscle::ContractureOptOp::ROMTrialParam> rom_params;
    for (const auto& entry : cfg.rom_trials)
        rom_params.push_back({entry.trial_name, (double)entry.angle_deg});

    auto op = std::make_unique<PMuscle::ContractureOptOp>(
        cfg.search_groups, rom_params, "lt_rel");
    ops.push_back(std::move(op));
}

// ============================================================================
// Build DHL operations
// ============================================================================

static void buildDHLOps(const DHLSideConfig& cfg, const std::string& side_label,
                        std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>& ops) {
    if (!cfg.enabled) return;

    LOG_INFO("[PID] Building DHL " << side_label << " operations");

    // Step 1: Anchor transfer (remove anchors + copy anchor)
    if (!cfg.remove_anchors.empty() && !cfg.target_muscle.empty()) {
        for (int anchor_idx : cfg.remove_anchors) {
            ops.push_back(std::make_unique<PMuscle::RemoveAnchorOp>(cfg.target_muscle, anchor_idx));
        }
        ops.push_back(std::make_unique<PMuscle::CopyAnchorOp>(
            cfg.donor_muscle, cfg.donor_anchor, cfg.target_muscle));
    }

    // Step 2: Contracture optimization — each muscle is its own search group
    if (!cfg.rom_trials.empty() && !cfg.muscles.empty()) {
        std::vector<PMuscle::ContractureOptOp::ROMTrialParam> rom_params;
        for (const auto& trial_name : cfg.rom_trials)
            rom_params.push_back({trial_name, (double)cfg.popliteal_angle_deg});

        std::map<std::string, std::vector<std::string>> search_groups;
        for (const auto& m : cfg.muscles)
            search_groups[m] = {m};

        auto op = std::make_unique<PMuscle::ContractureOptOp>(
            search_groups, rom_params, "lm_contract");
        ops.push_back(std::move(op));
    }
}

// ============================================================================
// Build FDO operations
// ============================================================================

static void buildFDOOps(const FDOSideConfig& cfg, const std::string& side_label,
                        std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>& ops) {
    if (!cfg.enabled || cfg.angle_deg == 0.0f) return;

    LOG_INFO("[PID] Building FDO " << side_label << " (angle=" << cfg.angle_deg << " deg)");

    double angle_rad = cfg.angle_deg * M_PI / 180.0;
    Eigen::Vector3d search_dir(cfg.search_direction[0], cfg.search_direction[1], cfg.search_direction[2]);
    Eigen::Vector3d rot_axis(cfg.rotation_axis[0], cfg.rotation_axis[1], cfg.rotation_axis[2]);
    search_dir.normalize();
    rot_axis.normalize();

    auto op = std::make_unique<PMuscle::FDOCombinedOp>(
        cfg.ref_muscle, cfg.ref_anchor, search_dir, rot_axis, angle_rad);
    ops.push_back(std::move(op));
}

// ============================================================================
// Build RFT operations
// ============================================================================

static void buildRFTOps(const RFTSideConfig& cfg, const std::string& side_label,
                        std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>& ops) {
    if (!cfg.enabled || cfg.target_muscles.empty()) return;

    LOG_INFO("[PID] Building RFT " << side_label << " operations");

    for (const auto& target : cfg.target_muscles) {
        // Remove anchors from target (descending order)
        for (int anchor_idx : cfg.remove_anchors) {
            ops.push_back(std::make_unique<PMuscle::RemoveAnchorOp>(target, anchor_idx));
        }
        // Copy donor anchors (ascending order)
        for (int donor_anchor_idx : cfg.copy_donor_anchors) {
            ops.push_back(std::make_unique<PMuscle::CopyAnchorOp>(
                cfg.donor_muscle, donor_anchor_idx, target));
        }
    }
}

// ============================================================================
// Build all operations from PID
// ============================================================================

static std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>
buildOperationsFromPID(const std::string& pid, const std::string& config_path,
                       const std::string& procedure_filter_str) {
    std::vector<std::unique_ptr<PMuscle::SurgeryOperation>> ops;
    SEMLSConfig cfg;

    // 1. Load surgery_panel.yaml
    std::string resolved_config = rm::getManager().resolve(config_path);
    if (resolved_config.empty() || !std::filesystem::exists(resolved_config)) {
        LOG_ERROR("[PID] Surgery config not found: " << config_path);
        return ops;
    }

    LOG_INFO("[PID] Loading surgery config: " << resolved_config);
    try {
        YAML::Node config = YAML::LoadFile(resolved_config);
        if (config["tal"]) {
            loadTALSide(config["tal"]["left"], cfg.tal_left);
            loadTALSide(config["tal"]["right"], cfg.tal_right);
        }
        if (config["dhl"]) {
            loadDHLSide(config["dhl"]["left"], cfg.dhl_left);
            loadDHLSide(config["dhl"]["right"], cfg.dhl_right);
        }
        if (config["fdo"]) {
            loadFDOSide(config["fdo"]["left"], cfg.fdo_left);
            loadFDOSide(config["fdo"]["right"], cfg.fdo_right);
        }
        if (config["rft"]) {
            loadRFTSide(config["rft"]["left"], cfg.rft_left);
            loadRFTSide(config["rft"]["right"], cfg.rft_right);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[PID] Failed to load surgery config: " << e.what());
        return ops;
    }

    // 2. Load patient metadata → determine enabled surgeries
    std::string metaUri = "@pid:" + pid + "/op1/metadata.yaml";
    std::string metaPath = rm::getManager().resolve(metaUri);
    if (metaPath.empty() || !std::filesystem::exists(metaPath)) {
        LOG_ERROR("[PID] Patient metadata not found: " << metaUri);
        return ops;
    }

    LOG_INFO("[PID] Loading patient metadata: " << metaPath);
    try {
        YAML::Node metadata = YAML::LoadFile(metaPath);
        if (metadata["surgery"]) {
            for (const auto& s : metadata["surgery"]) {
                std::string name = s.as<std::string>();
                bool is_left  = name.size() > 3 && name.substr(name.size() - 3) == "_Lt";
                bool is_right = name.size() > 3 && name.substr(name.size() - 3) == "_Rt";
                std::string base = name;
                if (is_left || is_right) base = name.substr(0, name.size() - 3);

                if (base == "TAL") {
                    if (is_left)  cfg.tal_left.enabled = true;
                    if (is_right) cfg.tal_right.enabled = true;
                } else if (base == "DHL") {
                    if (is_left)  cfg.dhl_left.enabled = true;
                    if (is_right) cfg.dhl_right.enabled = true;
                } else if (base == "FDO" || base == "FVDO") {
                    if (is_left)  cfg.fdo_left.enabled = true;
                    if (is_right) cfg.fdo_right.enabled = true;
                } else if (base == "RFT") {
                    if (is_left)  cfg.rft_left.enabled = true;
                    if (is_right) cfg.rft_right.enabled = true;
                }
            }
        } else {
            LOG_WARN("[PID] No 'surgery' key in metadata");
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[PID] Failed to load patient metadata: " << e.what());
        return ops;
    }

    // 3. Load ROM angles from op1/rom.yaml → TAL dorsiflexion, DHL popliteal
    std::string romUri = "@pid:" + pid + "/op1/rom.yaml";
    std::string romPath = rm::getManager().resolve(romUri);
    if (!romPath.empty() && std::filesystem::exists(romPath)) {
        try {
            YAML::Node rom = YAML::LoadFile(romPath);
            auto loadSideAngles = [&](const YAML::Node& side, TALSideConfig& tal, DHLSideConfig& dhl) {
                if (!side) return;
                // TAL: dorsiflexion angles
                if (side["ankle"]) {
                    auto ankle = side["ankle"];
                    for (auto& entry : tal.rom_trials) {
                        if (entry.trial_name.find("dorsi_k0") != std::string::npos) {
                            if (ankle["dorsiflexion_knee0_r2"] && !ankle["dorsiflexion_knee0_r2"].IsNull())
                                entry.angle_deg = ankle["dorsiflexion_knee0_r2"].as<float>();
                        } else if (entry.trial_name.find("dorsi_k90") != std::string::npos) {
                            if (ankle["dorsiflexion_knee90_r2"] && !ankle["dorsiflexion_knee90_r2"].IsNull())
                                entry.angle_deg = ankle["dorsiflexion_knee90_r2"].as<float>();
                        }
                    }
                }
                // DHL: popliteal angle
                if (side["knee"] && side["knee"]["popliteal_bilateral"] && !side["knee"]["popliteal_bilateral"].IsNull()) {
                    dhl.popliteal_angle_deg = side["knee"]["popliteal_bilateral"].as<float>();
                }
            };
            if (rom["rom"]) {
                loadSideAngles(rom["rom"]["left"], cfg.tal_left, cfg.dhl_left);
                loadSideAngles(rom["rom"]["right"], cfg.tal_right, cfg.dhl_right);
            }
            LOG_INFO("[PID] Loaded ROM angles from: " << romPath);
        } catch (const std::exception& e) {
            LOG_ERROR("[PID] Failed to load ROM: " << e.what());
        }
    } else {
        LOG_WARN("[PID] ROM file not found: " << romUri);
    }

    // 4. FDO angle: pre.hip.anteversion - op1.hip.anteversion
    std::string preRomUri = "@pid:" + pid + "/pre/rom.yaml";
    std::string preRomPath = rm::getManager().resolve(preRomUri);
    if (!preRomPath.empty() && std::filesystem::exists(preRomPath) &&
        !romPath.empty() && std::filesystem::exists(romPath)) {
        try {
            YAML::Node preRom = YAML::LoadFile(preRomPath);
            YAML::Node op1Rom = YAML::LoadFile(romPath);
            auto loadFDOAngle = [](const YAML::Node& preSide, const YAML::Node& op1Side) -> float {
                if (!preSide || !preSide["hip"] || !preSide["hip"]["anteversion"] || preSide["hip"]["anteversion"].IsNull())
                    return 0.0f;
                if (!op1Side || !op1Side["hip"] || !op1Side["hip"]["anteversion"] || op1Side["hip"]["anteversion"].IsNull())
                    return 0.0f;
                float pre = preSide["hip"]["anteversion"].as<float>();
                float op1 = op1Side["hip"]["anteversion"].as<float>();
                return pre - op1;
            };
            if (preRom["rom"] && op1Rom["rom"]) {
                cfg.fdo_left.angle_deg = loadFDOAngle(preRom["rom"]["left"], op1Rom["rom"]["left"]);
                cfg.fdo_right.angle_deg = loadFDOAngle(preRom["rom"]["right"], op1Rom["rom"]["right"]);
                LOG_INFO("[PID] FDO angles from anteversion diff: L=" << cfg.fdo_left.angle_deg
                         << " R=" << cfg.fdo_right.angle_deg);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("[PID] Failed to load FDO anteversion: " << e.what());
        }
    }

    // 5. Apply procedure filter
    auto filter = parseProcedureFilter(procedure_filter_str);
    if (!filter.empty()) {
        if (filter.find("tal") == filter.end()) {
            cfg.tal_left.enabled = cfg.tal_right.enabled = false;
        }
        if (filter.find("dhl") == filter.end()) {
            cfg.dhl_left.enabled = cfg.dhl_right.enabled = false;
        }
        if (filter.find("fdo") == filter.end()) {
            cfg.fdo_left.enabled = cfg.fdo_right.enabled = false;
        }
        if (filter.find("rft") == filter.end()) {
            cfg.rft_left.enabled = cfg.rft_right.enabled = false;
        }
    }

    // Log enabled surgeries
    LOG_INFO("[PID] Enabled surgeries:");
    if (cfg.tal_left.enabled)  LOG_INFO("  TAL Left  (dorsi_k0=" << cfg.tal_left.rom_trials[0].angle_deg
                                        << ", dorsi_k90=" << (cfg.tal_left.rom_trials.size() > 1 ? cfg.tal_left.rom_trials[1].angle_deg : 0.0f) << ")");
    if (cfg.tal_right.enabled) LOG_INFO("  TAL Right (dorsi_k0=" << cfg.tal_right.rom_trials[0].angle_deg
                                        << ", dorsi_k90=" << (cfg.tal_right.rom_trials.size() > 1 ? cfg.tal_right.rom_trials[1].angle_deg : 0.0f) << ")");
    if (cfg.dhl_left.enabled)  LOG_INFO("  DHL Left  (popliteal=" << cfg.dhl_left.popliteal_angle_deg << ")");
    if (cfg.dhl_right.enabled) LOG_INFO("  DHL Right (popliteal=" << cfg.dhl_right.popliteal_angle_deg << ")");
    if (cfg.fdo_left.enabled)  LOG_INFO("  FDO Left  (angle=" << cfg.fdo_left.angle_deg << ")");
    if (cfg.fdo_right.enabled) LOG_INFO("  FDO Right (angle=" << cfg.fdo_right.angle_deg << ")");
    if (cfg.rft_left.enabled)  LOG_INFO("  RFT Left");
    if (cfg.rft_right.enabled) LOG_INFO("  RFT Right");

    // 6. Build operations in order: TAL → DHL → FDO → RFT
    buildTALOps(cfg.tal_left, "Left", ops);
    buildTALOps(cfg.tal_right, "Right", ops);

    buildDHLOps(cfg.dhl_left, "Left", ops);
    buildDHLOps(cfg.dhl_right, "Right", ops);

    buildFDOOps(cfg.fdo_left, "Left", ops);
    buildFDOOps(cfg.fdo_right, "Right", ops);

    buildRFTOps(cfg.rft_left, "Left", ops);
    buildRFTOps(cfg.rft_right, "Right", ops);

    // 7. Append export operations
    std::string muscle_export = "@pid:" + pid + "/pre/muscle/semls.yaml";
    std::string skeleton_export = "@pid:" + pid + "/pre/skeleton/semls.yaml";
    ops.push_back(std::make_unique<PMuscle::ExportMusclesOp>(muscle_export));
    ops.push_back(std::make_unique<PMuscle::ExportSkeletonOp>(skeleton_export));

    return ops;
}

// ============================================================================
// Utilities
// ============================================================================

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  PID mode:    " << programName << " --pid <PID> [--procedure tal,dhl,fdo,rft]" << std::endl;
    std::cout << "  Script mode: " << programName << " --script <path>" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --pid, -p PID          Patient ID - auto-generate surgery from metadata+ROM" << std::endl;
    std::cout << "  --procedure PROCS      Comma-separated procedures to run: tal,dhl,fdo,rft" << std::endl;
    std::cout << "  --config, -c PATH      Surgery panel config YAML (default: " << DEFAULT_CONFIG << ")" << std::endl;
    std::cout << "  --skeleton, -s PATH    Path to skeleton XML file" << std::endl;
    std::cout << "  --muscle, -m PATH      Path to muscle XML file" << std::endl;
    std::cout << "  --script PATH          Path to surgery script YAML file" << std::endl;
    std::cout << "  --help, -h             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "PID mode defaults:" << std::endl;
    std::cout << "  --skeleton  @pid:<PID>/pre/skeleton/trimmed_unified.yaml" << std::endl;
    std::cout << "  --muscle    @data/muscle/base.yaml" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Run all surgeries for a patient" << std::endl;
    std::cout << "  " << programName << " --pid 29792292" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run only TAL and DHL for a patient" << std::endl;
    std::cout << "  " << programName << " --pid 29792292 --procedure tal,dhl" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run a surgery script" << std::endl;
    std::cout << "  " << programName << " --script data/surgery/example.yaml" << std::endl;
}

bool hasExportOperation(const std::vector<std::unique_ptr<PMuscle::SurgeryOperation>>& ops) {
    for (const auto& op : ops) {
        std::string type = op->getType();
        if (type == "export_muscles" || type == "export_skeleton") {
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

    // Parse command-line arguments using Boost.Program_options
    namespace po = boost::program_options;

    po::options_description desc("Surgery Tool Options");
    desc.add_options()
        ("help,h", "Show this help message")
        ("pid,p", po::value<std::string>(), "Patient ID - auto-generate surgery from metadata+ROM")
        ("procedure", po::value<std::string>(), "Comma-separated procedures: tal,dhl,fdo,rft")
        ("config,c", po::value<std::string>(), "Surgery panel config YAML")
        ("skeleton,s", po::value<std::string>(), "Path to skeleton XML file")
        ("muscle,m", po::value<std::string>(), "Path to muscle XML file")
        ("script", po::value<std::string>(), "Path to surgery script YAML file");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            printUsage(argv[0]);
            return 0;
        }
    } catch (const po::error& e) {
        LOG_ERROR("Argument parsing error: " << e.what());
        std::cerr << desc << std::endl;
        return 1;
    }

    // Convert relative data/ paths to @data/ format for rm::resolve
    auto convertPath = [](const std::string& path) -> std::string {
        if (path.find("data/") == 0) {
            return "@" + path;
        }
        return path;
    };

    bool pid_mode = vm.count("pid") > 0;

    // Determine skeleton/muscle paths
    std::string skeleton_path, muscle_path;
    if (pid_mode) {
        std::string pid = vm["pid"].as<std::string>();
        // PID mode defaults
        skeleton_path = vm.count("skeleton")
            ? convertPath(vm["skeleton"].as<std::string>())
            : "@pid:" + pid + "/pre/skeleton/trimmed_unified.yaml";
        muscle_path = vm.count("muscle")
            ? convertPath(vm["muscle"].as<std::string>())
            : "@data/muscle/base.yaml";
    } else {
        skeleton_path = vm.count("skeleton")
            ? convertPath(vm["skeleton"].as<std::string>())
            : DEFAULT_SKELETON;
        muscle_path = vm.count("muscle")
            ? convertPath(vm["muscle"].as<std::string>())
            : DEFAULT_MUSCLE;
    }

    LOG_INFO("Configuration:");
    LOG_INFO("  Mode:     " << (pid_mode ? "PID" : "Script"));
    LOG_INFO("  Skeleton: " << skeleton_path);
    LOG_INFO("  Muscles:  " << muscle_path);
    LOG_INFO("");

    try {
        // Create surgery executor
        std::string generator_context;
        if (pid_mode) {
            generator_context = "surgery-cli-pid: " + vm["pid"].as<std::string>();
        } else {
            std::string script_path = vm.count("script")
                ? convertPath(vm["script"].as<std::string>())
                : DEFAULT_SCRIPT;
            size_t lastSlash = script_path.find_last_of("/\\");
            std::string script_name = (lastSlash != std::string::npos)
                ? script_path.substr(lastSlash + 1)
                : script_path;
            generator_context = "surgery-tool: " + script_name;
        }

        LOG_INFO("Initializing surgery executor...");
        PMuscle::SurgeryExecutor executor(generator_context);

        // Load character
        LOG_INFO("Loading character...");
        executor.loadCharacter(skeleton_path, muscle_path);
        LOG_INFO("");

        // Build or load operations
        std::vector<std::unique_ptr<PMuscle::SurgeryOperation>> operations;

        if (pid_mode) {
            std::string pid = vm["pid"].as<std::string>();
            std::string config_path = vm.count("config")
                ? convertPath(vm["config"].as<std::string>())
                : DEFAULT_CONFIG;
            std::string proc_filter = vm.count("procedure")
                ? vm["procedure"].as<std::string>()
                : "";

            LOG_INFO("Building operations from PID: " << pid);
            LOG_INFO("");
            operations = buildOperationsFromPID(pid, config_path, proc_filter);
        } else {
            std::string script_path = vm.count("script")
                ? convertPath(vm["script"].as<std::string>())
                : DEFAULT_SCRIPT;
            LOG_INFO("Loading surgery script: " << script_path);
            operations = PMuscle::SurgeryScript::loadFromFile(script_path);
        }

        if (operations.empty()) {
            LOG_ERROR("Error: No operations to execute!");
            return 1;
        }

        LOG_INFO("");
        LOG_INFO("Total " << operations.size() << " operation(s)");
        LOG_INFO("");

        // Execute operations
        LOG_INFO("==============================================================================");
        LOG_INFO("Executing Surgery Operations");
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

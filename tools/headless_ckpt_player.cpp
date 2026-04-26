#include "Environment.h"
#include "PolicyNet.h"

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path ckpt_dir;
    int steps = 1800;
    double reset_phase = -1.0;
    bool stochastic = false;
    fs::path csv_path;
};

fs::path resolve_checkpoint_dir(const std::string& arg) {
    fs::path path(arg);
    if (fs::is_directory(path)) {
        return path;
    }
    fs::path under_runs = fs::path("runs") / path;
    if (fs::is_directory(under_runs)) {
        return under_runs;
    }
    throw std::runtime_error("Checkpoint directory not found: " + arg);
}

Options parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: headless_ckpt_player <ckpt_dir> [--steps N] [--phase P] "
            "[--stochastic] [--csv PATH]");
    }

    Options opt;
    opt.ckpt_dir = resolve_checkpoint_dir(argv[1]);
    opt.csv_path = opt.ckpt_dir / "headless_rollout.csv";

    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--steps" && i + 1 < argc) {
            opt.steps = std::stoi(argv[++i]);
        } else if (key == "--phase" && i + 1 < argc) {
            opt.reset_phase = std::stod(argv[++i]);
        } else if (key == "--csv" && i + 1 < argc) {
            opt.csv_path = argv[++i];
        } else if (key == "--stochastic") {
            opt.stochastic = true;
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + key);
        }
    }

    if (opt.steps <= 0) {
        throw std::runtime_error("--steps must be positive");
    }
    if ((opt.reset_phase < 0.0 && opt.reset_phase != -1.0) || opt.reset_phase > 1.0) {
        throw std::runtime_error("--phase must be -1 for randomized reset, or in [0, 1]");
    }
    return opt;
}

bool finite_vector(const Eigen::VectorXd& v) {
    return v.allFinite();
}

double finite_or_nan(double v) {
    return std::isfinite(v) ? v : std::numeric_limits<double>::quiet_NaN();
}

std::string sanitize_column_name(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9')) {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "unnamed" : out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const fs::path metadata_path = opt.ckpt_dir / "metadata.yaml";
        const fs::path agent_path = opt.ckpt_dir / "agent.pt";
        if (!fs::is_regular_file(metadata_path)) {
            throw std::runtime_error("metadata.yaml not found: " + metadata_path.string());
        }
        if (!fs::is_regular_file(agent_path)) {
            throw std::runtime_error("agent.pt not found: " + agent_path.string());
        }

        std::cout << "[HeadlessCkpt] checkpoint: " << opt.ckpt_dir << "\n";
        std::cout << "[HeadlessCkpt] metadata:   " << metadata_path << "\n";
        std::cout << "[HeadlessCkpt] agent:      " << agent_path << "\n";

        Environment env(metadata_path.string());
        env.setParamDefault();
        env.reset(opt.reset_phase);

        const int obs_dim = static_cast<int>(env.getState().rows());
        const int act_dim = static_cast<int>(env.getAction().rows());
        auto skel = env.getCharacter()->getSkeleton();
        std::vector<std::string> dof_names;
        dof_names.reserve(skel->getNumDofs());
        for (std::size_t i = 0; i < skel->getNumDofs(); ++i) {
            std::string name = skel->getDof(i)->getName();
            if (name.empty()) {
                std::ostringstream ss;
                ss << "dof_" << i;
                name = ss.str();
            }
            dof_names.push_back(sanitize_column_name(name));
        }
        std::cout << "[HeadlessCkpt] obs_dim=" << obs_dim
                  << " act_dim=" << act_dim
                  << " sim_hz=" << env.getSimulationHz()
                  << " control_hz=" << env.getControlHz()
                  << " substeps=" << env.getNumSubSteps() << "\n";

        auto weights = loadStateDict(agent_path.string());
        if (weights.empty()) {
            throw std::runtime_error("Failed to load TorchScript state_dict: " + agent_path.string());
        }
        PolicyNet policy = std::make_shared<PolicyNetImpl>(obs_dim, act_dim);
        policy->load_state_dict(weights);

        fs::create_directories(opt.csv_path.parent_path());
        std::ofstream csv(opt.csv_path);
        csv << "step,time,reward,terminated,truncated,phase,cycles,root_x,root_y,"
               "root_z,com_x,com_y,com_z,target_vel,avg_vel_x,avg_vel_z,"
               "action_l2,action_max_abs,tau_l2,tau_max_abs";
        for (std::size_t i = 0; i < dof_names.size(); ++i) {
            csv << ",tau_" << i << "_" << dof_names[i];
        }
        csv << '\n';

        double reward_sum = 0.0;
        double min_root_y = std::numeric_limits<double>::infinity();
        double max_action_abs = 0.0;
        double max_tau_abs = 0.0;
        int completed_cycles = 0;
        int executed_steps = 0;
        std::string stop_reason = "max_steps";

        for (int step = 0; step < opt.steps; ++step) {
            Eigen::VectorXd state = env.getState();
            if (!finite_vector(state)) {
                stop_reason = "nonfinite_state_before_step";
                break;
            }

            auto [action_f, value, logprob] =
                policy->sample_action(state.cast<float>(), opt.stochastic);
            (void)value;
            (void)logprob;
            Eigen::VectorXd action = action_f.cast<double>();
            env.setAction(action);
            env.step();

            auto current_skel = env.getCharacter()->getSkeleton();
            const Eigen::VectorXd q = current_skel->getPositions();
            const Eigen::Vector3d com = current_skel->getCOM();
            const Eigen::Vector3d avg_vel = env.getAvgVelocity();
            const Eigen::VectorXd tau = env.getCachedSPDTorque();
            const auto& info = env.getInfoMap();
            const bool terminated = env.isTerminated();
            const bool truncated = env.isTruncated();

            const double reward = env.getReward();
            reward_sum += finite_or_nan(reward);
            min_root_y = std::min(min_root_y, q.size() > 4 ? q[4] : com[1]);
            max_action_abs = std::max(max_action_abs, action.cwiseAbs().maxCoeff());
            if (tau.size() > 0 && tau.allFinite()) {
                max_tau_abs = std::max(max_tau_abs, tau.cwiseAbs().maxCoeff());
            }

            completed_cycles = env.getGaitPhase()->getAdaptiveCycleCount();
            csv << step << ','
                << env.getWorld()->getTime() << ','
                << reward << ','
                << (terminated ? 1 : 0) << ','
                << (truncated ? 1 : 0) << ','
                << env.getGaitPhase()->getAdaptivePhase() << ','
                << completed_cycles << ','
                << (q.size() > 3 ? q[3] : 0.0) << ','
                << (q.size() > 4 ? q[4] : 0.0) << ','
                << (q.size() > 5 ? q[5] : 0.0) << ','
                << com[0] << ','
                << com[1] << ','
                << com[2] << ','
                << env.getTargetCOMVelocity() << ','
                << avg_vel[0] << ','
                << avg_vel[2] << ','
                << action.norm() << ','
                << action.cwiseAbs().maxCoeff() << ','
                << (tau.size() > 0 && tau.allFinite() ? tau.norm() : std::numeric_limits<double>::quiet_NaN()) << ','
                << (tau.size() > 0 && tau.allFinite() ? tau.cwiseAbs().maxCoeff() : std::numeric_limits<double>::quiet_NaN());
            for (int i = 0; i < static_cast<int>(dof_names.size()); ++i) {
                csv << ',' << (i < tau.size() ? tau[i] : std::numeric_limits<double>::quiet_NaN());
            }
            csv << '\n';

            ++executed_steps;

            if (!q.allFinite() || !current_skel->getVelocities().allFinite()) {
                stop_reason = "nonfinite_skeleton";
                break;
            }
            if (terminated) {
                auto it = info.find("terminated");
                stop_reason = it != info.end() ? "terminated" : "terminated";
                break;
            }
            if (truncated) {
                stop_reason = "truncated";
                break;
            }
        }

        csv.close();
        const double elapsed = env.getWorld()->getTime();
        const Eigen::Vector3d final_com = env.getCharacter()->getSkeleton()->getCOM();
        const Eigen::Vector3d avg_vel = env.getAvgVelocity();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[HeadlessCkpt] stop_reason=" << stop_reason
                  << " steps=" << executed_steps
                  << " sim_time=" << elapsed
                  << " reward_sum=" << reward_sum
                  << " cycles=" << completed_cycles << "\n";
        std::cout << "[HeadlessCkpt] final_com=(" << final_com.transpose() << ")"
                  << " avg_vel=(" << avg_vel.transpose() << ")"
                  << " target_vel=" << env.getTargetCOMVelocity() << "\n";
        std::cout << "[HeadlessCkpt] min_root_y=" << min_root_y
                  << " max_action_abs=" << max_action_abs
                  << " max_tau_abs=" << max_tau_abs << "\n";
        std::cout << "[HeadlessCkpt] wrote " << opt.csv_path << "\n";

        return stop_reason == "max_steps" || stop_reason == "truncated" ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "[HeadlessCkpt] ERROR: " << e.what() << "\n";
        return 1;
    }
}

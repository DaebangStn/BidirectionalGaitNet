#include "PolicyNet.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <random>

// Helper: Orthogonal initialization (matching PyTorch torch.nn.init.orthogonal_)
void orthogonal_init(torch::Tensor tensor, double gain = 1.0) {
    torch::NoGradGuard no_grad;

    if (tensor.ndimension() < 2) {
        throw std::runtime_error("Only tensors with 2 or more dimensions are supported");
    }

    int64_t rows = tensor.size(0);
    int64_t cols = tensor.numel() / rows;

    // Generate random tensor
    auto flat_tensor = torch::randn({rows, cols}, torch::kFloat32);

    // QR decomposition - returns (Q: rows x min(rows,cols), R: min(rows,cols) x cols)
    auto qr_result = torch::linalg_qr(flat_tensor, "reduced");
    auto q = std::get<0>(qr_result);
    auto r = std::get<1>(qr_result);

    // For the orthogonal init, we want to fix the signs
    // Extract diagonal from R
    int64_t diag_size = std::min(rows, cols);
    auto d = r.diagonal();

    // Multiply Q columns by sign of R diagonal
    q = q * d.sign().unsqueeze(0);

    // If rows < cols, we need to pad q
    if (rows < cols) {
        // Generate orthogonal complement for remaining columns
        auto remaining = cols - rows;
        auto q_extra = torch::randn({rows, remaining}, torch::kFloat32);
        q = torch::cat({q, q_extra}, 1);
    }

    // Scale by gain
    q = q * gain;

    // Reshape and copy
    tensor.copy_(q.reshape_as(tensor));
}

template<typename ModuleType>
void PolicyNetImpl::layer_init(ModuleType& layer, double std, double bias_const) {
    torch::NoGradGuard no_grad;

    // Orthogonal weight initialization
    orthogonal_init(layer->weight, std);

    // Constant bias initialization
    torch::nn::init::constant_(layer->bias, bias_const);
}

PolicyNetImpl::PolicyNetImpl(int num_states, int num_actions, torch::Device device)
    : num_states_(num_states), num_actions_(num_actions), device_(device) {

    // ===== THREADING CONFIGURATION =====
    // NOTE: Threading is already configured by Python/PyTorch before C++ module loads.
    // Attempting to reconfigure causes c10::Error exception that crashes the module.
    // Current threading config is set by Python environment and works correctly.

    // ===== CRITIC NETWORK =====
    // 3 hidden layers of 512 units with ReLU
    critic_fc1 = register_module("critic_fc1", torch::nn::Linear(num_states, 512));
    critic_fc2 = register_module("critic_fc2", torch::nn::Linear(512, 512));
    critic_fc3 = register_module("critic_fc3", torch::nn::Linear(512, 512));
    critic_value = register_module("critic_value", torch::nn::Linear(512, 1));

    // Initialize with std=sqrt(2), bias=0
    layer_init(critic_fc1, std::sqrt(2.0), 0.0);
    layer_init(critic_fc2, std::sqrt(2.0), 0.0);
    layer_init(critic_fc3, std::sqrt(2.0), 0.0);
    layer_init(critic_value, 1.0, 0.0);  // Output layer: std=1.0

    // ===== ACTOR NETWORK =====
    // 3 hidden layers of 512 units with ReLU
    actor_fc1 = register_module("actor_fc1", torch::nn::Linear(num_states, 512));
    actor_fc2 = register_module("actor_fc2", torch::nn::Linear(512, 512));
    actor_fc3 = register_module("actor_fc3", torch::nn::Linear(512, 512));
    actor_mean = register_module("actor_mean", torch::nn::Linear(512, num_actions));

    // Initialize with std=sqrt(2), bias=0
    layer_init(actor_fc1, std::sqrt(2.0), 0.0);
    layer_init(actor_fc2, std::sqrt(2.0), 0.0);
    layer_init(actor_fc3, std::sqrt(2.0), 0.0);
    layer_init(actor_mean, 0.01, 0.0);  // Output layer: std=0.01

    // ===== LOG STD PARAMETER =====
    // Initialize to ones, with special handling for upper body (ppo_hierarchical.py:139-143)
    auto init_log_std = torch::ones({num_actions}, torch::kFloat32);

    {
        torch::NoGradGuard no_grad;
        if (num_actions > 18) {
            // Upper body actions: scale by 0.5
            init_log_std.slice(0, 18, num_actions) *= 0.5;
        }

        // Last action (cascading): keep at 1.0
        if (num_actions > 0) {
            init_log_std[-1] = 1.0;
        }
    }

    // Register as parameter (not learnable in C++ inference, but loadable)
    actor_logstd = register_parameter("actor_logstd", init_log_std.unsqueeze(0));

    // Move to device
    this->to(device_);
}

void PolicyNetImpl::setLogStd(float value) {
    torch::NoGradGuard no_grad;
    actor_logstd.fill_(value);
}

torch::Tensor PolicyNetImpl::get_value(torch::Tensor x) {
    // Critic forward pass: obs → fc1 → ReLU → fc2 → ReLU → fc3 → ReLU → value
    auto h = torch::relu(critic_fc1->forward(x));
    h = torch::relu(critic_fc2->forward(h));
    h = torch::relu(critic_fc3->forward(h));
    return critic_value->forward(h);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PolicyNetImpl::get_action_and_value(torch::Tensor x, torch::Tensor action) {
    // Actor forward pass: obs → fc1 → ReLU → fc2 → ReLU → fc3 → ReLU → mean
    auto h = torch::relu(actor_fc1->forward(x));
    h = torch::relu(actor_fc2->forward(h));
    h = torch::relu(actor_fc3->forward(h));
    auto action_mean = actor_mean->forward(h);

    // Expand log_std to match batch size
    auto action_logstd = actor_logstd.expand_as(action_mean);
    auto action_std = torch::exp(action_logstd);

    // Sample action if not provided (manual Normal distribution)
    if (!action.defined()) {
        auto eps = torch::randn_like(action_mean);
        action = action_mean + action_std * eps;
    }

    // Compute log probability (sum over action dimensions)
    // log_prob = -0.5 * ((action - mean) / std)^2 - log(std) - 0.5 * log(2π)
    auto log_prob = -0.5 * torch::pow((action - action_mean) / action_std, 2)
                    - action_logstd - 0.5 * std::log(2.0 * M_PI);
    log_prob = log_prob.sum(-1);

    // Compute entropy (sum over action dimensions)
    // entropy = 0.5 * log(2π * e * std^2)
    auto entropy = 0.5 + 0.5 * std::log(2.0 * M_PI) + action_logstd;
    entropy = entropy.sum(-1);

    // Get value
    auto value = get_value(x);

    return std::make_tuple(action, log_prob, entropy, value);
}

// Thread-safe inference method
// Multiple threads can call this concurrently on the same PolicyNet instance:
// - torch::NoGradGuard ensures no gradient computation (logically read-only)
// - Forward passes are stateless tensor operations (thread-safe)
// - torch::randn_like() uses thread-local RNG (libtorch ≥1.9, thread-safe)
// - Each call creates independent temporary tensors (no shared mutable state)
// Note: Not marked const due to libtorch API (forward() is non-const), but is logically const
std::tuple<Eigen::VectorXf, float, float>
PolicyNetImpl::sample_action(const Eigen::VectorXf& obs, bool stochastic) {
    torch::NoGradGuard no_grad;  // Inference mode (thread-local flag)

    // Convert Eigen → Torch
    auto obs_tensor = torch::from_blob(
        (void*)obs.data(),
        {1, obs.size()},
        torch::kFloat32
    ).clone().to(device_);  // Clone to avoid aliasing, move to device

    // Actor forward pass
    auto h = torch::relu(actor_fc1->forward(obs_tensor));
    h = torch::relu(actor_fc2->forward(h));
    h = torch::relu(actor_fc3->forward(h));
    auto action_mean = actor_mean->forward(h);

    // Expand log_std
    auto action_logstd = actor_logstd.expand_as(action_mean);
    auto action_std = torch::exp(action_logstd);

    // Sample from Normal(mean, std) or use mean directly (deterministic)
    torch::Tensor action_tensor;
    torch::Tensor log_prob;
    if (stochastic) {
        // NOTE: torch::randn_like() is thread-safe (uses thread-local RNG in libtorch ≥1.9)
        auto eps = torch::randn_like(action_mean);
        action_tensor = action_mean + action_std * eps;

        // Compute log probability
        log_prob = -0.5 * torch::pow((action_tensor - action_mean) / action_std, 2)
                        - action_logstd - 0.5 * std::log(2.0 * M_PI);
        log_prob = log_prob.sum(-1);
    } else {
        // Deterministic: use mean directly
        action_tensor = action_mean;
        log_prob = torch::zeros({1}, torch::kFloat32).to(device_);
    }

    // Get value estimate
    auto value_tensor = get_value(obs_tensor);

    // Convert back to Eigen (CPU)
    action_tensor = action_tensor.to(torch::kCPU);
    value_tensor = value_tensor.to(torch::kCPU);
    log_prob = log_prob.to(torch::kCPU);

    Eigen::VectorXf action(num_actions_);
    auto action_accessor = action_tensor.accessor<float, 2>();
    for (int i = 0; i < num_actions_; ++i) {
        action[i] = action_accessor[0][i];
    }

    auto value_accessor = value_tensor.accessor<float, 2>();
    float value = value_accessor[0][0];

    auto logprob_accessor = log_prob.accessor<float, 1>();
    float logprob = logprob_accessor[0];

    return {action, value, logprob};
}

void PolicyNetImpl::load_state_dict(const py::dict& state_dict) {
    torch::NoGradGuard no_grad;

    for (auto item : state_dict) {
        std::string key = py::str(item.first);

        // Convert py::object to torch::Tensor
        auto py_tensor = item.second;
        torch::Tensor value;

        try {
            // Try direct cast first
            value = py_tensor.cast<torch::Tensor>();
            // Ensure tensor is on CPU (handle CUDA tensors)
            if (value.is_cuda()) {
                value = value.cpu();
            }
        } catch (...) {
            // If that fails, try converting from numpy
            py::array np_array = py_tensor.cast<py::array>();
            auto np_info = np_array.request();

            std::vector<int64_t> shape;
            for (int i = 0; i < np_info.ndim; ++i) {
                shape.push_back(np_info.shape[i]);
            }

            value = torch::from_blob(
                np_info.ptr,
                shape,
                torch::kFloat32
            ).clone();
        }

        // Load weights based on key
        if (key == "critic.0.weight") {
            critic_fc1->weight.copy_(value);
        } else if (key == "critic.0.bias") {
            critic_fc1->bias.copy_(value);
        } else if (key == "critic.2.weight") {
            critic_fc2->weight.copy_(value);
        } else if (key == "critic.2.bias") {
            critic_fc2->bias.copy_(value);
        } else if (key == "critic.4.weight") {
            critic_fc3->weight.copy_(value);
        } else if (key == "critic.4.bias") {
            critic_fc3->bias.copy_(value);
        } else if (key == "critic.6.weight") {
            critic_value->weight.copy_(value);
        } else if (key == "critic.6.bias") {
            critic_value->bias.copy_(value);
        } else if (key == "actor_mean.0.weight") {
            actor_fc1->weight.copy_(value);
        } else if (key == "actor_mean.0.bias") {
            actor_fc1->bias.copy_(value);
        } else if (key == "actor_mean.2.weight") {
            actor_fc2->weight.copy_(value);
        } else if (key == "actor_mean.2.bias") {
            actor_fc2->bias.copy_(value);
        } else if (key == "actor_mean.4.weight") {
            actor_fc3->weight.copy_(value);
        } else if (key == "actor_mean.4.bias") {
            actor_fc3->bias.copy_(value);
        } else if (key == "actor_mean.6.weight") {
            actor_mean->weight.copy_(value);
        } else if (key == "actor_mean.6.bias") {
            actor_mean->bias.copy_(value);
        } else if (key == "actor_logstd") {
            actor_logstd.copy_(value);
        }
        // Ignore unknown keys
    }

    // Ensure all parameters are on correct device
    this->to(device_);
}

void PolicyNetImpl::load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    torch::NoGradGuard no_grad;

    for (const auto& [key, value] : state_dict) {
        // Load weights based on key (same mapping as Python version)
        if (key == "critic.0.weight") {
            critic_fc1->weight.copy_(value);
        } else if (key == "critic.0.bias") {
            critic_fc1->bias.copy_(value);
        } else if (key == "critic.2.weight") {
            critic_fc2->weight.copy_(value);
        } else if (key == "critic.2.bias") {
            critic_fc2->bias.copy_(value);
        } else if (key == "critic.4.weight") {
            critic_fc3->weight.copy_(value);
        } else if (key == "critic.4.bias") {
            critic_fc3->bias.copy_(value);
        } else if (key == "critic.6.weight") {
            critic_value->weight.copy_(value);
        } else if (key == "critic.6.bias") {
            critic_value->bias.copy_(value);
        } else if (key == "actor_mean.0.weight") {
            actor_fc1->weight.copy_(value);
        } else if (key == "actor_mean.0.bias") {
            actor_fc1->bias.copy_(value);
        } else if (key == "actor_mean.2.weight") {
            actor_fc2->weight.copy_(value);
        } else if (key == "actor_mean.2.bias") {
            actor_fc2->bias.copy_(value);
        } else if (key == "actor_mean.4.weight") {
            actor_fc3->weight.copy_(value);
        } else if (key == "actor_mean.4.bias") {
            actor_fc3->bias.copy_(value);
        } else if (key == "actor_mean.6.weight") {
            actor_mean->weight.copy_(value);
        } else if (key == "actor_mean.6.bias") {
            actor_mean->bias.copy_(value);
        } else if (key == "actor_logstd") {
            actor_logstd.copy_(value);
        }
        // Ignore unknown keys
    }

    // Ensure all parameters are on correct device
    this->to(device_);
}

std::unordered_map<std::string, torch::Tensor>
loadStateDict(const std::string& path) {
    std::unordered_map<std::string, torch::Tensor> state_dict;

    try {
        auto module = torch::jit::load(path);

        // TorchScript saves buffers with underscores instead of dots
        // e.g., "critic_0_weight" instead of "critic.0.weight"
        for (const auto& item : module.named_buffers()) {
            std::string key = item.name;

            // Convert underscores back to dots for first two positions
            // critic_0_weight -> critic.0.weight
            // actor_mean_0_weight -> actor_mean.0.weight
            size_t pos = 0;
            int dot_count = 0;

            // Find and replace underscores that represent layer separators
            // The pattern is: <module>_<layer_idx>_<param_type>
            while ((pos = key.find('_', pos)) != std::string::npos) {
                // Check if next char is a digit (layer index)
                if (pos + 1 < key.size() && std::isdigit(key[pos + 1])) {
                    key[pos] = '.';
                    ++pos;
                    // Find the next underscore after the digit(s)
                    while (pos < key.size() && std::isdigit(key[pos])) ++pos;
                    if (pos < key.size() && key[pos] == '_') {
                        key[pos] = '.';
                    }
                }
                ++pos;
            }

            state_dict[key] = item.value.clone();
        }
    } catch (const c10::Error&) {
        // Not a TorchScript file, return empty map
        return {};
    } catch (const std::exception&) {
        // Other errors, return empty map
        return {};
    }

    return state_dict;
}

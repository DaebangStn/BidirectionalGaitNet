#pragma once

#include <torch/torch.h>
#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * PolicyNet: Actor-Critic network in C++ libtorch
 *
 * Matches Python Agent architecture from ppo_hierarchical.py:112-161
 * - Critic: 3x512 ReLU layers → value
 * - Actor: 3x512 ReLU layers → mean
 * - Log std as learnable parameter
 *
 * Usage:
 *   auto policy = std::make_shared<PolicyNetImpl>(num_states, num_actions);
 *   auto [action, value, logprob] = policy->sample_action(obs);
 *   policy->load_state_dict(python_state_dict);
 */

class PolicyNetImpl : public torch::nn::Module {
public:
    /**
     * Construct PolicyNet matching Python Agent architecture.
     *
     * @param num_states Observation dimension
     * @param num_actions Action dimension
     * @param device Device to run inference on (cpu/cuda)
     */
    PolicyNetImpl(int num_states, int num_actions, torch::Device device = torch::kCPU);

    /**
     * Sample action from policy (inference mode, no gradient).
     *
     * Thread-safe: Multiple threads can call concurrently on same PolicyNet instance.
     * Logically const: Does not modify network parameters during inference.
     * (Not marked const due to libtorch API limitations - forward() is non-const)
     *
     * @param obs Observation vector (float)
     * @return tuple<action, value, logprob>
     *   - action: Sampled action vector
     *   - value: State value estimate
     *   - logprob: Log probability of sampled action
     */
    std::tuple<Eigen::VectorXf, float, float> sample_action(const Eigen::VectorXf& obs);

    /**
     * Get value estimate (critic forward pass).
     *
     * @param x Observation tensor (batch_size, num_states)
     * @return Value estimates (batch_size, 1)
     */
    torch::Tensor get_value(torch::Tensor x);

    /**
     * Get action and value (for training, with gradients).
     *
     * @param x Observation tensor (batch_size, num_states)
     * @param action Optional action tensor for log_prob computation
     * @return tuple<action, log_prob, entropy, value>
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    get_action_and_value(torch::Tensor x, torch::Tensor action = {});

    /**
     * Load weights from PyTorch state_dict.
     *
     * @param state_dict Python dictionary with network weights
     */
    void load_state_dict(const py::dict& state_dict);

    /**
     * Get current device.
     */
    torch::Device device() const { return device_; }

private:
    // Initialize layer with orthogonal weights
    template<typename ModuleType>
    void layer_init(ModuleType& layer, double std = std::sqrt(2.0), double bias_const = 0.0);

    // Critic network: obs → value
    torch::nn::Linear critic_fc1{nullptr};
    torch::nn::Linear critic_fc2{nullptr};
    torch::nn::Linear critic_fc3{nullptr};
    torch::nn::Linear critic_value{nullptr};

    // Actor network: obs → action_mean
    torch::nn::Linear actor_fc1{nullptr};
    torch::nn::Linear actor_fc2{nullptr};
    torch::nn::Linear actor_fc3{nullptr};
    torch::nn::Linear actor_mean{nullptr};

    // Log std (learnable parameter, initialized specially)
    torch::Tensor actor_logstd;

    // Network metadata
    int num_states_;
    int num_actions_;
    torch::Device device_;
};

// Convenience alias
using PolicyNet = std::shared_ptr<PolicyNetImpl>;

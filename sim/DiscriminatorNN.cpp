#include "DiscriminatorNN.h"
#include <torch/torch.h>
#include <cmath>

// Xavier initialization for linear layers
static void xavier_init_disc(torch::nn::Linear& layer) {
    torch::nn::init::xavier_uniform_(layer->weight);
    torch::nn::init::zeros_(layer->bias);
}

DiscriminatorNNImpl::DiscriminatorNNImpl(int num_muscles, bool force_cpu)
    : num_muscles_(num_muscles),
      device_(torch::kCPU) {

    // Network architecture: 3-layer MLP (256 -> 256 -> 1)
    const int num_h1 = 256;
    const int num_h2 = 256;

    // Define layers
    fc1 = register_module("fc1", torch::nn::Linear(num_muscles, num_h1));
    fc2 = register_module("fc2", torch::nn::Linear(num_h1, num_h2));
    fc_out = register_module("fc_out", torch::nn::Linear(num_h2, 1));

    // Xavier initialization
    xavier_init_disc(fc1);
    xavier_init_disc(fc2);
    xavier_init_disc(fc_out);

    // Check for CUDA availability (unless force_cpu is true)
    if (!force_cpu && torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        to(device_);
    }
}

torch::Tensor DiscriminatorNNImpl::forward(torch::Tensor activations) {
    // Optional normalization
    if (use_normalizer_ && norm_mean_.defined() && norm_std_.defined()) {
        activations = (activations - norm_mean_) / (norm_std_ + 1e-8f);
    }

    // Forward through layers with LeakyReLU
    auto x = torch::leaky_relu(fc1->forward(activations), 0.2);
    x = torch::leaky_relu(fc2->forward(x), 0.2);
    x = fc_out->forward(x);  // No activation on output (logit)

    return x;
}

float DiscriminatorNNImpl::forward_no_grad(const Eigen::VectorXf& activations) {
    torch::NoGradGuard no_grad;  // Disable gradient computation

    // Convert Eigen (float) → torch::Tensor
    auto act_tensor = torch::from_blob(
        const_cast<float*>(activations.data()),
        {activations.size()},
        torch::kFloat32
    ).clone().to(device_);

    // Forward pass
    auto logit = forward(act_tensor);

    // Return scalar logit value
    return logit.item<float>();
}

float DiscriminatorNNImpl::compute_reward(const Eigen::VectorXf& activations) {
    // For ADD-style energy efficiency:
    // - Input: current muscle activations
    // - Demo: zero activations (ideal minimal energy)
    // - Diff: 0 - activations = -activations
    // But since discriminator sees normalized difference,
    // we directly pass activations (as negative of demo - agent = -(0 - act) = act)
    // Actually for ADD: disc sees (demo - agent), demo=0, so disc sees -activations
    // We need to negate the activations for the discriminator input

    Eigen::VectorXf neg_activations = -activations;

    // Get logit from discriminator
    float logit = forward_no_grad(neg_activations);

    // Compute probability using sigmoid
    float prob = 1.0f / (1.0f + std::exp(-logit));

    // ADD reward formula: disc_r = -log(max(1 - prob, 0.0001))
    // When prob → 1 (agent looks like demo/efficient) → high reward
    // When prob → 0 (agent looks fake/inefficient) → low reward
    // Note: reward_scale is applied in Environment.cpp (additive mode only)
    float disc_r = -std::log(std::max(1.0f - prob, 0.0001f));

    return disc_r;
}

void DiscriminatorNNImpl::load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    // Load each parameter by name
    // Python Sequential uses indices: fc.0, fc.2, fc.4 (skipping activation layers)
    for (const auto& pair : state_dict) {
        const std::string& name = pair.first;
        const torch::Tensor& param = pair.second;

        // Map Python parameter names to C++ module names
        torch::Tensor* target_param = nullptr;

        if (name == "fc.0.weight") target_param = &fc1->weight;
        else if (name == "fc.0.bias") target_param = &fc1->bias;
        else if (name == "fc.2.weight") target_param = &fc2->weight;
        else if (name == "fc.2.bias") target_param = &fc2->bias;
        else if (name == "fc.4.weight") target_param = &fc_out->weight;
        else if (name == "fc.4.bias") target_param = &fc_out->bias;

        if (target_param != nullptr) {
            // Copy parameter values
            target_param->data().copy_(param.to(device_));
        }
    }
}

void DiscriminatorNNImpl::setNormalizer(const Eigen::VectorXf& mean, const Eigen::VectorXf& std) {
    // Convert Eigen vectors to torch tensors
    norm_mean_ = torch::from_blob(
        const_cast<float*>(mean.data()),
        {mean.size()},
        torch::kFloat32
    ).clone().to(device_);

    norm_std_ = torch::from_blob(
        const_cast<float*>(std.data()),
        {std.size()},
        torch::kFloat32
    ).clone().to(device_);
}

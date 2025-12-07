#pragma once

#include <torch/torch.h>
#include <Eigen/Dense>
#include <memory>

/**
 * C++ implementation of ADD-style Discriminator for energy-efficient control.
 *
 * Learns to distinguish "necessary" vs "excessive" disc_obs (e.g., muscle activations + upper body torques).
 * Input: disc_obs (normalized difference from zero)
 * Output: single logit (high = looks like zero/efficient)
 *
 * Reward formula: disc_r = -log(max(1 - sigmoid(logit), 0.0001)) * scale
 */

class DiscriminatorNNImpl : public torch::nn::Module {
public:
    /**
     * Construct DiscriminatorNN with specified architecture.
     *
     * @param disc_obs_dim Discriminator observation dimension (e.g., num_muscles + upper_body_dim)
     * @param force_cpu If true, force CPU execution even if CUDA is available
     */
    DiscriminatorNNImpl(int disc_obs_dim, bool force_cpu = true);

    /**
     * Forward pass: activations -> logit.
     * Returns single logit value (unbounded).
     */
    torch::Tensor forward(torch::Tensor activations);

    /**
     * No-gradient forward pass for inference during rollout.
     * Converts Eigen vector to torch tensor, runs inference, returns logit.
     *
     * @param activations Muscle activations as Eigen vector (float)
     * @return Logit value (scalar)
     */
    float forward_no_grad(const Eigen::VectorXf& activations);

    /**
     * Compute discriminator reward from activations.
     * Uses ADD style: diff = 0 - activations = -activations
     * Reward: disc_r = -log(max(1 - sigmoid(logit), 0.0001))
     *
     * @param activations Muscle activations as Eigen vector
     * @return Discriminator reward (positive when activations look "efficient")
     */
    float compute_reward(const Eigen::VectorXf& activations);

    /**
     * Load weights from Python state_dict.
     * Called from BatchRolloutEnv::update_discriminator_weights().
     *
     * @param state_dict Map of parameter name -> tensor
     */
    void load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict);

    /**
     * Set optional normalizer parameters.
     * When enabled, input is normalized: (x - mean) / std before forward pass.
     *
     * @param mean Mean vector for normalization
     * @param std Standard deviation vector for normalization
     */
    void setNormalizer(const Eigen::VectorXf& mean, const Eigen::VectorXf& std);

    /**
     * Enable/disable input normalization.
     */
    void setUseNormalizer(bool use) { use_normalizer_ = use; }

    /**
     * Check if normalizer is enabled.
     */
    bool usesNormalizer() const { return use_normalizer_; }

    /**
     * Get discriminator observation dimension.
     */
    int getDiscObsDim() const { return disc_obs_dim_; }

private:
    // Network architecture: 3-layer MLP (256 -> 256 -> 1)
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_out{nullptr};

    // Network configuration
    int disc_obs_dim_;

    // Device management
    torch::Device device_;

    // Optional normalization
    bool use_normalizer_ = false;
    torch::Tensor norm_mean_;
    torch::Tensor norm_std_;
};

// Using std::shared_ptr for automatic memory management
using DiscriminatorNN = std::shared_ptr<DiscriminatorNNImpl>;

/**
 * Factory function to create DiscriminatorNN instance.
 *
 * @param disc_obs_dim Discriminator observation dimension
 * @param force_cpu If true, force CPU execution (recommended for multi-process)
 */
inline DiscriminatorNN make_discriminator_nn(int disc_obs_dim, bool force_cpu = true) {
    return std::make_shared<DiscriminatorNNImpl>(disc_obs_dim, force_cpu);
}

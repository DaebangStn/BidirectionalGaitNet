#pragma once

#include <torch/torch.h>
#include <Eigen/Dense>
#include <memory>

/**
 * C++ implementation of MuscleNN for hierarchical muscle control.
 *
 * Replicates the Python PyTorch architecture for thread-safe inference
 * without Python GIL constraints. Supports dynamic weight updates from
 * Python training code.
 */

class MuscleNNImpl : public torch::nn::Module {
public:
    /**
     * Construct MuscleNN with specified architecture.
     *
     * @param num_total_muscle_related_dofs Dimension of muscle forces (JtA)
     * @param num_dofs Dimension of desired torques
     * @param num_muscles Number of muscles (output dimension)
     * @param is_cascaded Whether this is a cascading (hierarchical) network
     * @param force_cpu If true, force CPU execution even if CUDA is available
     */
    MuscleNNImpl(int num_total_muscle_related_dofs, int num_dofs, int num_muscles, bool is_cascaded = false, bool force_cpu = false);

    /**
     * Forward pass without activation functions.
     * Used internally for standard forward().
     */
    torch::Tensor forward_wo_relu(torch::Tensor muscle_tau, torch::Tensor tau);

    /**
     * Forward pass with cascading input without activation.
     * Used for hierarchical control.
     */
    torch::Tensor forward_with_prev_out_wo_relu(
        torch::Tensor muscle_tau, torch::Tensor tau, torch::Tensor prev_out, double weight = 1.0);

    /**
     * Standard forward pass with activation functions.
     * Returns: relu(tanh(forward_wo_relu(...)))
     */
    torch::Tensor forward(torch::Tensor muscle_tau, torch::Tensor tau);

    /**
     * Forward pass with cascading (hierarchical) input.
     * Returns: relu(tanh(forward_with_prev_out_wo_relu(...)))
     */
    torch::Tensor forward_with_prev_out(
        torch::Tensor muscle_tau, torch::Tensor tau, torch::Tensor prev_out, double weight = 1.0);

    /**
     * No-gradient forward pass for inference (C++ version).
     * Converts Eigen vectors to torch tensors, runs inference, returns as Eigen.
     *
     * @param muscle_tau Muscle forces (JtA_reduced) as Eigen vector
     * @param tau Desired torques as Eigen vector
     * @param prev_out Previous network output (for cascading), nullptr for standard mode
     * @param weight Cascading weight (default 1.0)
     * @return Unnormalized activations as Eigen vector (float)
     */
    Eigen::VectorXf unnormalized_no_grad_forward(
        const Eigen::VectorXd& muscle_tau,
        const Eigen::VectorXd& tau,
        const Eigen::VectorXf* prev_out = nullptr,
        double weight = 1.0);

    /**
     * Apply activation filter: relu(tanh(x))
     * Used to convert unnormalized activations to final muscle activations.
     *
     * @param unnormalized Unnormalized activation as Eigen vector
     * @return Filtered activation (0 to 1 range) as Eigen vector
     */
    Eigen::VectorXf forward_filter(const Eigen::VectorXf& unnormalized);

    /**
     * Load weights from Python state_dict (via pybind11).
     * Called from Environment::setMuscleNetworkWeight().
     *
     * @param state_dict Python dictionary of parameter name -> numpy array
     */
    void load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict);

    /**
     * Initialize weights with Xavier uniform initialization.
     */
    void initialize_weights();

private:
    // Network architecture (4-layer MLP)
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    // Normalization parameters
    torch::Tensor std_muscle_tau;
    torch::Tensor std_tau;

    // Network configuration
    int num_total_muscle_related_dofs_;
    int num_dofs_;
    int num_muscles_;
    bool is_cascaded_;

    // Device management
    torch::Device device_;
};

// Using std::shared_ptr for automatic memory management
using MuscleNN = std::shared_ptr<MuscleNNImpl>;

/**
 * Factory function to create MuscleNN instance.
 *
 * @param force_cpu If true, force CPU execution even if CUDA is available (useful for multi-process scenarios)
 */
inline MuscleNN make_muscle_nn(int num_total_muscle_related_dofs, int num_dofs, int num_muscles, bool is_cascaded = false, bool force_cpu = false) {
    return std::make_shared<MuscleNNImpl>(num_total_muscle_related_dofs, num_dofs, num_muscles, is_cascaded, force_cpu);
}

#include "MuscleNN.h"
#include <torch/torch.h>

// Xavier initialization for linear layers
void xavier_init(torch::nn::Linear& layer) {
    torch::nn::init::xavier_uniform_(layer->weight);
    torch::nn::init::zeros_(layer->bias);
}

MuscleNNImpl::MuscleNNImpl(int num_total_muscle_related_dofs, int num_dofs, int num_muscles, bool is_cascaded)
    : num_total_muscle_related_dofs_(num_total_muscle_related_dofs),
      num_dofs_(num_dofs),
      num_muscles_(num_muscles),
      is_cascaded_(is_cascaded),
      device_(torch::kCPU) {

    // Network architecture: 4-layer MLP with LeakyReLU
    const int num_h1 = 256;
    const int num_h2 = 256;
    const int num_h3 = 256;

    // Input dimension depends on cascading mode
    int input_dim = num_total_muscle_related_dofs + num_dofs;
    if (is_cascaded) {
        input_dim += num_muscles + 1;  // prev_out + weight
    }

    // Define layers
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, num_h1));
    fc2 = register_module("fc2", torch::nn::Linear(num_h1, num_h2));
    fc3 = register_module("fc3", torch::nn::Linear(num_h2, num_h3));
    fc4 = register_module("fc4", torch::nn::Linear(num_h3, num_muscles));

    // Initialize normalization parameters (std = 200 for all dimensions)
    std_muscle_tau = torch::ones({num_total_muscle_related_dofs}) * 200.0f;
    std_tau = torch::ones({num_dofs}) * 200.0f;

    // Check for CUDA availability
    if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        to(device_);
        std_muscle_tau = std_muscle_tau.to(device_);
        std_tau = std_tau.to(device_);
    }

    // Xavier initialization
    initialize_weights();
}

void MuscleNNImpl::initialize_weights() {
    xavier_init(fc1);
    xavier_init(fc2);
    xavier_init(fc3);
    xavier_init(fc4);
}

torch::Tensor MuscleNNImpl::forward_wo_relu(torch::Tensor muscle_tau, torch::Tensor tau) {
    // Normalize inputs
    muscle_tau = muscle_tau / std_muscle_tau;
    tau = tau / std_tau;

    // Concatenate inputs
    auto x = torch::cat({muscle_tau, tau}, -1);

    // Forward through layers
    x = torch::leaky_relu(fc1->forward(x), 0.2);
    x = torch::leaky_relu(fc2->forward(x), 0.2);
    x = torch::leaky_relu(fc3->forward(x), 0.2);
    x = fc4->forward(x);

    return x;
}

torch::Tensor MuscleNNImpl::forward_with_prev_out_wo_relu(
    torch::Tensor muscle_tau, torch::Tensor tau, torch::Tensor prev_out, double weight) {

    // Normalize inputs
    muscle_tau = muscle_tau / std_muscle_tau;
    tau = tau / std_tau;

    // Create weight tensor
    auto weight_tensor = torch::full({1}, weight, torch::TensorOptions().dtype(torch::kFloat32).device(muscle_tau.device()));

    // Concatenate: [0.5 * prev_out, weight, muscle_tau, tau]
    auto x = torch::cat({0.5 * prev_out, weight_tensor, muscle_tau, tau}, -1);

    // Forward through layers
    x = torch::leaky_relu(fc1->forward(x), 0.2);
    x = torch::leaky_relu(fc2->forward(x), 0.2);
    x = torch::leaky_relu(fc3->forward(x), 0.2);
    auto out = fc4->forward(x);

    // Add residual connection: out = prev_out + weight * network_output
    return prev_out + weight * out;
}

torch::Tensor MuscleNNImpl::forward(torch::Tensor muscle_tau, torch::Tensor tau) {
    return torch::relu(torch::tanh(forward_wo_relu(muscle_tau, tau)));
}

torch::Tensor MuscleNNImpl::forward_with_prev_out(
    torch::Tensor muscle_tau, torch::Tensor tau, torch::Tensor prev_out, double weight) {
    return torch::relu(torch::tanh(forward_with_prev_out_wo_relu(muscle_tau, tau, prev_out, weight)));
}

Eigen::VectorXf MuscleNNImpl::unnormalized_no_grad_forward(
    const Eigen::VectorXd& muscle_tau,
    const Eigen::VectorXd& tau,
    const Eigen::VectorXf* prev_out,
    double weight) {

    torch::NoGradGuard no_grad;  // Disable gradient computation

    // Convert Eigen (double) → torch::Tensor (float32)
    auto muscle_tau_tensor = torch::from_blob(
        const_cast<double*>(muscle_tau.data()),
        {muscle_tau.size()},
        torch::kFloat64
    ).to(torch::kFloat32).to(device_);

    auto tau_tensor = torch::from_blob(
        const_cast<double*>(tau.data()),
        {tau.size()},
        torch::kFloat64
    ).to(torch::kFloat32).to(device_);

    torch::Tensor out;

    if (prev_out == nullptr) {
        // Standard mode (non-cascading)
        out = forward_wo_relu(muscle_tau_tensor, tau_tensor);
    } else {
        // Cascading mode with previous output
        auto prev_out_tensor = torch::from_blob(
            const_cast<float*>(prev_out->data()),
            {prev_out->size()},
            torch::kFloat32
        ).to(device_);

        out = forward_with_prev_out_wo_relu(muscle_tau_tensor, tau_tensor, prev_out_tensor, weight);
    }

    // Convert result back to Eigen (float32)
    out = out.to(torch::kCPU);
    Eigen::VectorXf result(out.size(0));
    std::memcpy(result.data(), out.data_ptr<float>(), out.size(0) * sizeof(float));

    return result;
}

Eigen::VectorXf MuscleNNImpl::forward_filter(const Eigen::VectorXf& unnormalized) {
    torch::NoGradGuard no_grad;

    // Convert Eigen → torch::Tensor
    auto tensor = torch::from_blob(
        const_cast<float*>(unnormalized.data()),
        {unnormalized.size()},
        torch::kFloat32
    ).clone();  // Clone to avoid modifying original

    // Apply activation: relu(tanh(x))
    auto filtered = torch::relu(torch::tanh(tensor));

    // Convert back to Eigen
    Eigen::VectorXf result(filtered.size(0));
    std::memcpy(result.data(), filtered.data_ptr<float>(), filtered.size(0) * sizeof(float));

    return result;
}

void MuscleNNImpl::load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    // Load each parameter by name
    for (const auto& pair : state_dict) {
        const std::string& name = pair.first;
        const torch::Tensor& param = pair.second;

        // Map Python parameter names to C++ module names
        torch::Tensor* target_param = nullptr;

        if (name == "fc.0.weight") target_param = &fc1->weight;
        else if (name == "fc.0.bias") target_param = &fc1->bias;
        else if (name == "fc.2.weight") target_param = &fc2->weight;
        else if (name == "fc.2.bias") target_param = &fc2->bias;
        else if (name == "fc.4.weight") target_param = &fc3->weight;
        else if (name == "fc.4.bias") target_param = &fc3->bias;
        else if (name == "fc.6.weight") target_param = &fc4->weight;
        else if (name == "fc.6.bias") target_param = &fc4->bias;

        if (target_param != nullptr) {
            // Copy parameter values
            target_param->data().copy_(param.to(device_));
        }
    }
}

"""
Muscle network trainer for hierarchical PPO.

Standalone module for learning muscle activation patterns from
desired torques using supervised learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from typing import Dict, List
# from python.ray_model import MuscleNN
from ppo.muscle_nn import MuscleNN


class MuscleLearner:
    """
    Muscle network trainer using supervised learning.

    Learns to map desired torques to muscle activations through
    gradient descent on muscle tuple data collected during rollouts.
    """

    def __init__(
        self,
        num_actuator_action: int,
        num_muscles: int,
        num_muscle_dofs: int,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 128,
        is_cascaded: bool = False,
    ):
        """
        Initialize muscle learner.

        Args:
            num_actuator_action: Number of actuator DOFs
            num_muscles: Number of muscles
            num_muscle_dofs: Number of muscle-related DOFs
            learning_rate: Adam optimizer learning rate (default: 1e-4)
            num_epochs: Number of training epochs per update (default: 3)
            batch_size: Minibatch size for SGD (default: 128)
            is_cascaded: Whether to use cascading mode (default: False)
        """
        self.device = torch.device("cuda")
        self.num_actuator_action = num_actuator_action
        self.num_muscles = num_muscles
        self.num_muscle_dofs = num_muscle_dofs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.is_cascaded = is_cascaded
        self.learning_rate = learning_rate

        # Create muscle network model
        self.model = MuscleNN(
            num_muscle_dofs,
            num_actuator_action,
            num_muscles,
            is_cascaded=is_cascaded
        ).to(self.device)

        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Register gradient clipping hooks
        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.model.train()

    def learn(self, muscle_tuples_list: List[List]) -> Dict:
        """
        Train muscle network on collected tuples from all environments.

        Args:
            muscle_tuples_list: List of tuple lists from each environment.
                Each environment provides lists of numpy arrays:
                - Non-cascading: [tau_des_list, JtA_reduced_list, JtA_list]
                - Cascading: [..., prev_out_list, weight_list]

        Returns:
            Training statistics dictionary with keys:
            - num_tuples: Number of training samples
            - loss_muscle: Average total loss
            - loss_reg: Average regularization loss
            - loss_target: Average target tracking loss
            - loss_act: Average activation regularization loss
            - time: Timing statistics
        """
        start_time = time.perf_counter()

        # Flatten and concatenate tuples from all environments
        num_components = 5 if self.is_cascaded else 3
        all_tuples = [[] for _ in range(num_components)]

        for env_tuples in muscle_tuples_list:
            for component_idx in range(num_components):
                # env_tuples[component_idx] is a list of numpy arrays
                all_tuples[component_idx].extend(env_tuples[component_idx])

        # Convert to tensors
        # Stack arrays to create (num_samples, feature_dim) tensors
        tau_des = torch.tensor(
            np.stack([arr for arr in all_tuples[0]]),
            device=self.device,
            dtype=torch.float32
        )
        JtA_reduced = torch.tensor(
            np.stack([arr for arr in all_tuples[1]]),
            device=self.device,
            dtype=torch.float32
        )
        JtA = torch.tensor(
            np.stack([arr for arr in all_tuples[2]]),
            device=self.device,
            dtype=torch.float32
        )

        prev_out = None
        weight = None
        if self.is_cascaded:
            prev_out = torch.tensor(
                np.stack([arr for arr in all_tuples[3]]),
                device=self.device,
                dtype=torch.float32
            )
            weight = torch.tensor(
                np.stack([arr for arr in all_tuples[4]]),
                device=self.device,
                dtype=torch.float32
            )

        converting_time = (time.perf_counter() - start_time) * 1000
        learning_start = time.perf_counter()

        # Training loop
        num_samples = len(tau_des)
        indices = np.arange(num_samples)

        loss_avg = 0.0
        loss_reg_avg = 0.0
        loss_target_avg = 0.0
        loss_act_avg = 0.0

        num_batches = num_samples // self.batch_size

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)

            epoch_loss = 0.0
            epoch_loss_reg = 0.0
            epoch_loss_target = 0.0
            epoch_loss_act = 0.0

            for batch_idx in range(num_batches):
                # Get batch indices
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_indices = torch.from_numpy(
                    indices[batch_start:batch_end]
                ).to(self.device)

                # Get batch data
                batch_tau_des = tau_des[batch_indices]
                batch_JtA_reduced = JtA_reduced[batch_indices]
                batch_JtA = JtA[batch_indices]

                # Forward pass
                if self.is_cascaded:
                    batch_weight = weight[batch_indices]
                    batch_prev_out = prev_out[batch_indices]
                    activation_wo_relu = self.model.forward_with_prev_out_wo_relu(
                        batch_JtA_reduced,
                        batch_tau_des,
                        batch_prev_out,
                        batch_weight
                    ).unsqueeze(2)
                else:
                    activation_wo_relu = self.model.forward_wo_relu(
                        batch_JtA_reduced,
                        batch_tau_des
                    ).unsqueeze(2)

                # Apply activation function
                activation = torch.relu(torch.tanh(activation_wo_relu))

                # Compute actual torque from muscle activations
                tau = torch.bmm(batch_JtA, activation).squeeze(-1)

                # Loss computation
                loss_reg_wo_relu = activation_wo_relu.pow(2).mean()
                loss_target = ((tau - batch_tau_des) / 100.0).pow(2).mean()
                loss_reg_act = activation.pow(2).mean()

                # Total loss
                loss = 0.01 * loss_reg_act + loss_target + 0.01 * loss_reg_wo_relu

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Additional gradient clipping (redundant with hooks but safe)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)

                self.optimizer.step()

                # Accumulate stats
                epoch_loss += loss.item()
                epoch_loss_reg += loss_reg_wo_relu.item()
                epoch_loss_target += loss_target.item()
                epoch_loss_act += loss_reg_act.item()

            # Average over batches
            loss_avg += epoch_loss / num_batches
            loss_reg_avg += epoch_loss_reg / num_batches
            loss_target_avg += epoch_loss_target / num_batches
            loss_act_avg += epoch_loss_act / num_batches

        # Average over epochs
        loss_avg /= self.num_epochs
        loss_reg_avg /= self.num_epochs
        loss_target_avg /= self.num_epochs
        loss_act_avg /= self.num_epochs

        learning_time = (time.perf_counter() - learning_start) * 1000
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            'num_tuples': num_samples,
            'loss_muscle': loss_avg,
            'loss_reg': loss_reg_avg,
            'loss_target': loss_target_avg,
            'loss_act': loss_act_avg,
            'time': {
                'converting_time_ms': converting_time,
                'learning_time_ms': learning_time,
                'total_ms': total_time
            }
        }

    def get_weights(self) -> Dict:
        """Get model weights as numpy arrays for distribution to environments."""
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def get_state_dict(self) -> Dict:
        """
        Get model state dict with CPU tensors for C++ environment update.

        Note: Converts GPU tensors to CPU for distribution to C++ simulation environments.
        Training happens on GPU, but C++ environments require CPU tensors.
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict) -> None:
        """Load model weights from numpy arrays."""
        weights_tensor = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in weights.items()
        }
        self.model.load_state_dict(weights_tensor)

    def get_optimizer_weights(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights: Dict) -> None:
        """Load optimizer state from checkpoint."""
        self.optimizer.load_state_dict(weights)

    def save(self, path: str) -> None:
        """
        Save model and optimizer to disk.

        Args:
            path: Path to save model weights
        """
        path = Path(path)
        torch.save(self.model.state_dict(), path)
        torch.save(
            self.optimizer.state_dict(),
            path.with_suffix(".opt" + path.suffix)
        )

    def load(self, path: str) -> None:
        """
        Load model and optimizer from disk.

        Args:
            path: Path to saved model weights
        """
        path = Path(path)
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.optimizer.load_state_dict(
            torch.load(
                path.with_suffix(".opt" + path.suffix),
                map_location=self.device
            )
        )

"""
ADD-style discriminator for energy-efficient muscle activation learning.

Learns to distinguish "necessary" vs "excessive" muscle activations.
Demo = zero activations (ideal minimal energy), Agent = current activations.
Provides reward signal to encourage energy efficiency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional


class DiscriminatorNN(nn.Module):
    """
    ADD-style discriminator network.

    Architecture: input_dim -> 256 -> 256 -> 1 (logit)
    Input: Normalized difference (demo - agent) = (-activations) for energy efficiency.
    Output: Logit (high = looks like demo/efficient, low = looks fake/inefficient).
    """

    def __init__(self, num_muscles: int):
        super().__init__()
        self.num_muscles = num_muscles

        # Network architecture: 3-layer MLP matching C++ implementation
        self.fc = nn.Sequential(
            nn.Linear(num_muscles, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output: single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: activations -> logit."""
        return self.fc(x)


class DiscriminatorLearner:
    """
    ADD-style discriminator trainer.

    Trains discriminator to distinguish demo (zero activations) from agent (current activations).
    For ADD style, the input to discriminator is the difference (demo - agent) = -activations.
    """

    def __init__(
        self,
        num_muscles: int,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 256,
        buffer_size: int = 100000,
        grad_penalty: float = 10.0,
        logit_reg: float = 0.01,
        weight_decay: float = 0.0001,
    ):
        """
        Initialize discriminator learner.

        Args:
            num_muscles: Number of muscles (input dimension)
            learning_rate: Adam optimizer learning rate
            num_epochs: Training epochs per update
            batch_size: Minibatch size for training
            buffer_size: Replay buffer capacity for negative samples
            grad_penalty: Gradient penalty coefficient
            logit_reg: Logit regularization coefficient
            weight_decay: Weight decay coefficient
        """
        self.device = torch.device("cuda")
        self.num_muscles = num_muscles
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.grad_penalty = grad_penalty
        self.logit_reg = logit_reg
        self.weight_decay = weight_decay

        # Create discriminator model
        self.model = DiscriminatorNN(num_muscles).to(self.device)

        # Adam optimizer with weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Replay buffer for negative samples (past agent observations)
        self.replay_buffer: Optional[torch.Tensor] = None
        self.replay_idx = 0

        self.model.train()

    def _add_to_replay_buffer(self, disc_obs: torch.Tensor) -> None:
        """Add observations to replay buffer."""
        batch_size = disc_obs.shape[0]

        if self.replay_buffer is None:
            # Initialize buffer
            self.replay_buffer = torch.zeros(
                (self.buffer_size, self.num_muscles),
                device=self.device
            )

        # Circular buffer insertion
        if self.replay_idx + batch_size <= self.buffer_size:
            self.replay_buffer[self.replay_idx:self.replay_idx + batch_size] = disc_obs
            self.replay_idx += batch_size
        else:
            # Wrap around
            remaining = self.buffer_size - self.replay_idx
            self.replay_buffer[self.replay_idx:] = disc_obs[:remaining]
            overflow = batch_size - remaining
            self.replay_buffer[:overflow] = disc_obs[remaining:]
            self.replay_idx = overflow

    def _sample_replay_buffer(self, num_samples: int) -> Optional[torch.Tensor]:
        """Sample from replay buffer."""
        if self.replay_buffer is None:
            return None

        valid_size = min(self.replay_idx, self.buffer_size)
        if valid_size < num_samples:
            return None

        indices = torch.randint(0, valid_size, (num_samples,), device=self.device)
        return self.replay_buffer[indices]

    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for discriminator training."""
        batch_size = real_samples.shape[0]

        # Random interpolation
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Compute discriminator output on interpolated samples
        disc_out = self.model(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_out,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_out),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient penalty: (||grad|| - 1)^2
        gradient_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()
        return gradient_penalty

    def learn(self, disc_obs: np.ndarray) -> Dict:
        """
        Train discriminator on collected disc_obs (muscle activations).

        ADD style:
        - Positive samples (demo): zero difference = zeros
        - Negative samples (agent): demo - agent = 0 - activations = -activations

        Args:
            disc_obs: numpy array of shape (num_samples, num_muscles)
                      Current muscle activations from rollout

        Returns:
            Training statistics dictionary
        """
        start_time = time.perf_counter()

        # Convert to tensor
        disc_obs_tensor = torch.tensor(
            disc_obs,
            device=self.device,
            dtype=torch.float32
        )

        # Add to replay buffer
        self._add_to_replay_buffer(disc_obs_tensor)

        num_samples = disc_obs_tensor.shape[0]
        num_batches = max(1, num_samples // self.batch_size)

        converting_time = (time.perf_counter() - start_time) * 1000
        learning_start = time.perf_counter()

        # Training stats
        loss_avg = 0.0
        loss_pos_avg = 0.0
        loss_neg_avg = 0.0
        loss_gp_avg = 0.0
        accuracy_avg = 0.0

        indices = np.arange(num_samples)

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)

            epoch_loss = 0.0
            epoch_loss_pos = 0.0
            epoch_loss_neg = 0.0
            epoch_loss_gp = 0.0
            epoch_accuracy = 0.0

            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, num_samples)
                batch_size = batch_end - batch_start
                batch_indices = torch.from_numpy(indices[batch_start:batch_end]).to(self.device)

                # Get batch of agent activations
                agent_activations = disc_obs_tensor[batch_indices]

                # ADD style: diff = demo - agent = 0 - activations = -activations
                neg_samples = -agent_activations  # Negative samples (fake)

                # Positive samples: demo = zeros (perfect match = zero difference)
                pos_samples = torch.zeros_like(neg_samples)

                # Sample from replay buffer for additional negative samples
                replay_samples = self._sample_replay_buffer(batch_size // 2)
                if replay_samples is not None:
                    neg_samples = torch.cat([neg_samples, -replay_samples], dim=0)
                    pos_samples = torch.cat([
                        pos_samples,
                        torch.zeros(batch_size // 2, self.num_muscles, device=self.device)
                    ], dim=0)

                # Forward pass
                pos_logits = self.model(pos_samples)  # Should output high (real)
                neg_logits = self.model(neg_samples)  # Should output low (fake)

                # BCE loss
                # pos: target=1 -> -log(sigmoid(logit))
                # neg: target=0 -> -log(1 - sigmoid(logit))
                loss_pos = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits)
                )
                loss_neg = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits)
                )

                disc_loss = 0.5 * (loss_pos + loss_neg)

                # Gradient penalty
                if self.grad_penalty > 0:
                    gp = self._compute_gradient_penalty(pos_samples, neg_samples[:pos_samples.shape[0]])
                    disc_loss = disc_loss + self.grad_penalty * gp
                else:
                    gp = torch.tensor(0.0)

                # Logit regularization
                if self.logit_reg > 0:
                    logit_norm = (pos_logits.pow(2).mean() + neg_logits.pow(2).mean()) / 2
                    disc_loss = disc_loss + self.logit_reg * logit_norm

                # Backward pass
                self.optimizer.zero_grad()
                disc_loss.backward()
                self.optimizer.step()

                # Compute accuracy
                with torch.no_grad():
                    pos_correct = (torch.sigmoid(pos_logits) > 0.5).float().mean()
                    neg_correct = (torch.sigmoid(neg_logits) < 0.5).float().mean()
                    accuracy = (pos_correct + neg_correct) / 2

                # Accumulate stats
                epoch_loss += disc_loss.item()
                epoch_loss_pos += loss_pos.item()
                epoch_loss_neg += loss_neg.item()
                epoch_loss_gp += gp.item() if isinstance(gp, torch.Tensor) else gp
                epoch_accuracy += accuracy.item()

            # Average over batches
            loss_avg += epoch_loss / num_batches
            loss_pos_avg += epoch_loss_pos / num_batches
            loss_neg_avg += epoch_loss_neg / num_batches
            loss_gp_avg += epoch_loss_gp / num_batches
            accuracy_avg += epoch_accuracy / num_batches

        # Average over epochs
        loss_avg /= self.num_epochs
        loss_pos_avg /= self.num_epochs
        loss_neg_avg /= self.num_epochs
        loss_gp_avg /= self.num_epochs
        accuracy_avg /= self.num_epochs

        learning_time = (time.perf_counter() - learning_start) * 1000
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            'num_samples': num_samples,
            'loss_disc': loss_avg,
            'loss_pos': loss_pos_avg,
            'loss_neg': loss_neg_avg,
            'loss_gp': loss_gp_avg,
            'accuracy': accuracy_avg,
            'replay_buffer_size': min(self.replay_idx, self.buffer_size),
            'time': {
                'converting_time_ms': converting_time,
                'learning_time_ms': learning_time,
                'total_ms': total_time
            }
        }

    def compute_reward(self, disc_obs: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Compute discriminator reward for given observations.

        Args:
            disc_obs: numpy array of shape (num_samples, num_muscles)
            scale: Reward scale factor

        Returns:
            numpy array of rewards, shape (num_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(disc_obs, device=self.device, dtype=torch.float32)

            # ADD style: input is -activations (diff = demo - agent)
            neg_obs = -obs_tensor

            logits = self.model(neg_obs)
            probs = torch.sigmoid(logits).squeeze(-1)

            # ADD reward: disc_r = -log(max(1 - prob, 0.0001)) * scale
            rewards = -torch.log(torch.clamp(1 - probs, min=0.0001)) * scale

        self.model.train()
        return rewards.cpu().numpy()

    def get_weights(self) -> Dict:
        """Get model weights as numpy arrays."""
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def get_state_dict(self) -> Dict:
        """Get model state dict with CPU tensors for C++ environment update."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict) -> None:
        """Load model weights from numpy arrays."""
        weights_tensor = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in weights.items()
        }
        self.model.load_state_dict(weights_tensor)

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

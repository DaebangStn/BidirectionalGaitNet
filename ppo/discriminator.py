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
    Input: Normalized difference (demo - agent) = (-disc_obs) for energy efficiency.
    Output: Logit (high = looks like demo/efficient, low = looks fake/inefficient).
    """

    def __init__(self, disc_obs_dim: int):
        super().__init__()
        self.disc_obs_dim = disc_obs_dim

        # Network architecture: 3-layer MLP matching C++ implementation
        self.fc = nn.Sequential(
            nn.Linear(disc_obs_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output: single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: disc_obs -> logit."""
        return self.fc(x)


class DiscriminatorLearner:
    """
    ADD-style discriminator trainer.

    Trains discriminator to distinguish demo (zero disc_obs) from agent (current disc_obs).
    For ADD style, the input to discriminator is the difference (demo - agent) = -disc_obs.
    """

    def __init__(
        self,
        disc_obs_dim: int,
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
            disc_obs_dim: Discriminator observation dimension (e.g., num_muscles + upper_body_dim)
            learning_rate: Adam optimizer learning rate
            num_epochs: Training epochs per update
            batch_size: Minibatch size for training
            buffer_size: Replay buffer capacity for negative samples
            grad_penalty: Gradient penalty coefficient
            logit_reg: Logit regularization coefficient
            weight_decay: Weight decay coefficient
        """
        self.device = torch.device("cuda")
        self.disc_obs_dim = disc_obs_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.grad_penalty = grad_penalty
        self.logit_reg = logit_reg
        self.weight_decay = weight_decay

        # Create discriminator model
        self.model = DiscriminatorNN(disc_obs_dim).to(self.device)

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
                (self.buffer_size, self.disc_obs_dim),
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

    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor):
        """Compute gradient penalty for discriminator training.

        Returns:
            Tuple of (gradient_penalty, grad_norm_mean, grad_norm_std)
        """
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

        # Per-sample gradient norms
        grad_norms = gradients.norm(2, dim=1)

        # Gradient penalty: (||grad|| - 1)^2
        gradient_penalty = (grad_norms - 1).pow(2).mean()

        # Stats for logging (detached)
        grad_norm_mean = grad_norms.mean().item()
        grad_norm_std = grad_norms.std().item()

        return gradient_penalty, grad_norm_mean, grad_norm_std

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
        loss_gp_raw_avg = 0.0      # Raw GP term (before scaling)
        loss_gp_scaled_avg = 0.0   # Scaled GP contribution (grad_penalty * gp)
        grad_norm_mean_avg = 0.0   # Averaged over all batches/epochs
        grad_norm_std_avg = 0.0    # Averaged over all batches/epochs
        accuracy_avg = 0.0

        indices = np.arange(num_samples)

        for _ in range(self.num_epochs):
            np.random.shuffle(indices)

            epoch_loss = 0.0
            epoch_loss_pos = 0.0
            epoch_loss_neg = 0.0
            epoch_loss_gp_raw = 0.0
            epoch_loss_gp_scaled = 0.0
            epoch_grad_norm_mean = 0.0
            epoch_grad_norm_std = 0.0
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
                        torch.zeros(batch_size // 2, self.disc_obs_dim, device=self.device)
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
                    gp_raw, gp_grad_norm_mean, gp_grad_norm_std = self._compute_gradient_penalty(
                        pos_samples, neg_samples[:pos_samples.shape[0]])
                    gp_scaled = self.grad_penalty * gp_raw
                    disc_loss = disc_loss + gp_scaled
                else:
                    gp_raw = 0.0
                    gp_scaled = 0.0
                    gp_grad_norm_mean = 0.0
                    gp_grad_norm_std = 0.0

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

                # Accumulate stats (use .item() for tensors to detach from graph)
                epoch_loss += disc_loss.item()
                epoch_loss_pos += loss_pos.item()
                epoch_loss_neg += loss_neg.item()
                epoch_loss_gp_raw += gp_raw.item() if isinstance(gp_raw, torch.Tensor) else gp_raw
                epoch_loss_gp_scaled += gp_scaled.item() if isinstance(gp_scaled, torch.Tensor) else gp_scaled
                epoch_grad_norm_mean += gp_grad_norm_mean  # Already a float from _compute_gradient_penalty
                epoch_grad_norm_std += gp_grad_norm_std    # Already a float from _compute_gradient_penalty
                epoch_accuracy += accuracy.item()

            # Average over batches
            loss_avg += epoch_loss / num_batches
            loss_pos_avg += epoch_loss_pos / num_batches
            loss_neg_avg += epoch_loss_neg / num_batches
            loss_gp_raw_avg += epoch_loss_gp_raw / num_batches
            loss_gp_scaled_avg += epoch_loss_gp_scaled / num_batches
            grad_norm_mean_avg += epoch_grad_norm_mean / num_batches
            grad_norm_std_avg += epoch_grad_norm_std / num_batches
            accuracy_avg += epoch_accuracy / num_batches

        # Average over epochs
        loss_avg /= self.num_epochs
        loss_pos_avg /= self.num_epochs
        loss_neg_avg /= self.num_epochs
        loss_gp_raw_avg /= self.num_epochs
        loss_gp_scaled_avg /= self.num_epochs
        grad_norm_mean_avg /= self.num_epochs
        grad_norm_std_avg /= self.num_epochs
        accuracy_avg /= self.num_epochs

        learning_time = (time.perf_counter() - learning_start) * 1000
        total_time = (time.perf_counter() - start_time) * 1000

        # ===== DIAGNOSTIC METRICS (computed on final batch outputs) =====
        eps = 1e-6

        with torch.no_grad():
            # D_fake distribution (sigmoid of neg_logits)
            D_fake = torch.sigmoid(neg_logits).squeeze()
            D_fake_mean = D_fake.mean().item()
            D_fake_std = D_fake.std().item() if D_fake.numel() > 1 else 0.0
            D_fake_p10 = torch.quantile(D_fake, 0.1).item() if D_fake.numel() > 1 else D_fake_mean
            D_fake_p90 = torch.quantile(D_fake, 0.9).item() if D_fake.numel() > 1 else D_fake_mean

            # D_pos distribution
            D_pos = torch.sigmoid(pos_logits).squeeze()

            # Logit margin: logit(D_pos) - logit(D_neg)
            def safe_logit(x):
                return torch.log(x + eps) - torch.log(1 - x + eps)
            logit_margin = (safe_logit(D_pos) - safe_logit(D_fake)).mean().item()

            # r_disc reward distribution
            r_disc = -torch.log(1 - D_fake + eps)
            r_disc_mean = r_disc.mean().item()
            r_disc_std = r_disc.std().item() if r_disc.numel() > 1 else 0.0
            r_disc_p10 = torch.quantile(r_disc, 0.1).item() if r_disc.numel() > 1 else r_disc_mean
            r_disc_p90 = torch.quantile(r_disc, 0.9).item() if r_disc.numel() > 1 else r_disc_mean

        return {
            'num_samples': num_samples,
            'loss_disc': loss_avg,
            'loss_pos': loss_pos_avg,
            'loss_neg': loss_neg_avg,
            'loss_gp_raw': loss_gp_raw_avg,       # Raw GP term (before scaling)
            'loss_gp_scaled': loss_gp_scaled_avg, # Scaled GP (grad_penalty * gp)
            'accuracy': accuracy_avg,
            'replay_buffer_size': min(self.replay_idx, self.buffer_size),
            # Diagnostic metrics
            'D_fake_mean': D_fake_mean,
            'D_fake_std': D_fake_std,
            'D_fake_p10': D_fake_p10,
            'D_fake_p90': D_fake_p90,
            'logit_margin': logit_margin,
            'r_disc_mean': r_disc_mean,
            'r_disc_std': r_disc_std,
            'r_disc_p10': r_disc_p10,
            'r_disc_p90': r_disc_p90,
            'grad_norm_mean': grad_norm_mean_avg,  # Averaged over all batches/epochs
            'grad_norm_std': grad_norm_std_avg,    # Averaged over all batches/epochs
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

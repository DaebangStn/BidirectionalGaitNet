import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from python.learn.hdf5_dataset import HDF5DataModule
from python.learn.network import RegressionNet
from python.learn.logger import SimpleLogger


class Trainer:
    """Trainer for regression MLP with multi-component loss."""

    def __init__(self, model: RegressionNet, data: HDF5DataModule, logger: SimpleLogger,
                 ckpt_dir: Path, lr: float = 1e-3, min_lr: float = 1e-4,
                 l1_weight: float = 0.0, l2_weight: float = 0.0,
                 grad_weight: float = 0.0, recon_weight: float = 1.0,
                 recon_delta: float = 0.5, recon_type: str = 'mse',
                 recon_start_epoch: int = 0, log_period: int = 50,
                 max_epochs: int = 1000, save_period: int = 1000):
        """
        Args:
            model: RegressionNet model
            data: HDF5DataModule for data loading
            logger: SimpleLogger for metrics
            ckpt_dir: Directory to save checkpoints
            lr: Initial learning rate
            min_lr: Minimum learning rate for scheduler
            l1_weight: Weight for L1 regularization
            l2_weight: Weight for L2 regularization
            grad_weight: Weight for gradient penalty
            recon_weight: Weight for reconstruction loss
            recon_delta: Delta parameter for robust losses
            recon_type: Type of reconstruction loss ('mse', 'huber', 'cauchy', 'welsch', 'geman', 'tukey')
            recon_start_epoch: Epoch to start using robust loss (before that, use Huber)
            log_period: Frequency of logging (in epochs)
            max_epochs: Maximum number of training epochs
            save_period: Frequency of checkpoint saving (in epochs)
        """
        self.model = model
        self.data = data
        self.logger = logger
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimization
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            threshold=1e-3,
            threshold_mode='rel',
            min_lr=min_lr,
        )

        # Loss hyperparameters
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.grad_weight = grad_weight
        self.recon_weight = recon_weight
        self.recon_delta = recon_delta
        self.recon_type = recon_type
        self.recon_start_epoch = recon_start_epoch

        # Training config
        self.log_period = log_period
        self.max_epochs = max_epochs
        self.save_period = save_period

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Optimizer: Adam(lr={lr})")
        print(f"[Trainer] Loss: {recon_type} (weight={recon_weight}) + L1={l1_weight} + L2={l2_weight} + GP={grad_weight}")

    @staticmethod
    def cauchy_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Cauchy robust loss function."""
        delta *= 1.2
        r = input - target
        loss = 0.5 * delta**2 * torch.log1p((r / delta)**2)
        return loss.mean()

    @staticmethod
    def welsch_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Welsch robust loss function."""
        delta *= 1.8
        r = input - target
        loss = 0.5 * delta**2 * (1 - torch.exp(-(r / delta)**2))
        return loss.mean()

    @staticmethod
    def geman_mcclure_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Geman-McClure robust loss function."""
        delta *= 1.2
        r = input - target
        loss = 0.5 * r**2 / (1 + (r / delta)**2)
        return loss.mean()

    @staticmethod
    def tukey_biweight_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Tukey biweight robust loss function."""
        delta *= 3.5
        r = input - target
        mask = r.abs() <= delta
        loss = torch.zeros_like(r)
        z = (r / delta)**2
        loss[mask] = (delta**2 / 6) * (1 - (1 - z[mask])**3)
        loss[~mask] = delta**2 / 6
        return loss.mean()

    def compute_reconstruction_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Compute reconstruction loss based on configured loss type.

        Args:
            y_pred: Predicted outputs
            y_true: Ground truth outputs
            epoch: Current epoch (for conditional loss switching)

        Returns:
            Reconstruction loss
        """
        if self.recon_type == 'mse':
            return nn.functional.mse_loss(y_pred, y_true)
        elif self.recon_type == 'huber':
            return nn.functional.huber_loss(y_pred, y_true, delta=self.recon_delta)
        elif self.recon_type == 'geman':
            if epoch < self.recon_start_epoch:
                return nn.functional.huber_loss(y_pred, y_true, delta=self.recon_delta)
            else:
                return self.geman_mcclure_loss(y_pred, y_true, delta=self.recon_delta)
        elif self.recon_type == 'cauchy':
            return self.cauchy_loss(y_pred, y_true, delta=self.recon_delta)
        elif self.recon_type == 'welsch':
            if epoch < self.recon_start_epoch:
                return nn.functional.huber_loss(y_pred, y_true, delta=self.recon_delta)
            else:
                return self.welsch_loss(y_pred, y_true, delta=self.recon_delta)
        elif self.recon_type == 'tukey':
            if epoch < self.recon_start_epoch:
                return nn.functional.huber_loss(y_pred, y_true, delta=self.recon_delta)
            else:
                return self.tukey_biweight_loss(y_pred, y_true, delta=self.recon_delta)
        else:
            raise ValueError(f"Invalid reconstruction loss type: {self.recon_type}")

    def compute_regularization_losses(self) -> tuple:
        """
        Compute L1 and L2 regularization losses.

        Returns:
            (l1_loss, l2_loss)
        """
        l1_reg = torch.zeros(1, device=self.device)
        l2_reg = torch.zeros(1, device=self.device)

        for param in self.model.parameters():
            l1_reg += torch.mean(torch.abs(param))
            l2_reg += torch.mean(torch.square(param))

        return l1_reg, l2_reg

    def compute_gradient_penalty(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty to encourage smooth predictions.

        Args:
            x: Input tensor (requires_grad=True)
            y_pred: Predicted output

        Returns:
            Gradient penalty loss
        """
        grad_input = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Clamp gradients to prevent explosion
        grad_input = torch.clamp(grad_input, min=-1e3, max=1e3)
        gp_loss = torch.sum(torch.square(grad_input), dim=-1).mean()

        return gp_loss

    def train_step(self, batch: tuple, epoch: int) -> dict:
        """
        Perform a single training step.

        Args:
            batch: (inputs, targets) tuple
            epoch: Current epoch number

        Returns:
            Dictionary of losses
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Enable gradient computation for inputs (needed for gradient penalty)
        x.requires_grad = True

        # Forward pass
        y_pred = self.model(x, normalize=True, denormalize=False)

        # Compute reconstruction loss
        recon_loss = self.compute_reconstruction_loss(y_pred, y, epoch)

        # Compute regularization losses
        l1_loss, l2_loss = self.compute_regularization_losses()

        # Compute gradient penalty
        gp_loss = self.compute_gradient_penalty(x, y_pred) if self.grad_weight > 0 else torch.zeros(1, device=self.device)

        # Total loss
        total_loss = (self.recon_weight * recon_loss +
                     self.l1_weight * l1_loss +
                     self.l2_weight * l2_loss +
                     self.grad_weight * gp_loss)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss/train': total_loss.item(),
            'loss/reconstruction': recon_loss.item(),
            'loss/l1': l1_loss.item(),
            'loss/l2': l2_loss.item(),
            'loss/grad_penalty': gp_loss.item(),
        }

    def validate(self, epoch: int) -> float:
        """
        Validate model on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.data.val_dataloader():
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                y_pred = self.model(x, normalize=True, denormalize=False)
                mse = nn.functional.mse_loss(y_pred, y)
                total_val_loss += mse.item()
                num_batches += 1

        avg_val_loss = total_val_loss / max(num_batches, 1)

        if epoch % self.log_period == 0:
            self.logger.log_metrics({'loss/val': avg_val_loss}, step=epoch)

        return avg_val_loss

    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.get_config(),
        }

        filename = self.ckpt_dir / f"ep_{epoch:06d}.ckpt"
        torch.save(checkpoint, filename)

        if epoch % self.log_period == 0:
            print(f"[Trainer] Checkpoint saved: {filename}")

    def train(self):
        """Main training loop."""
        print(f"\n[Trainer] Starting training for {self.max_epochs} epochs...")

        for epoch in tqdm(range(self.max_epochs), desc="Training", ncols=100):
            self.model.train()
            epoch_losses = {}

            # Training
            for batch in self.data.train_dataloader():
                losses = self.train_step(batch, epoch)
                for key, value in losses.items():
                    epoch_losses[key] = epoch_losses.get(key, 0.0) + value

            # Average losses over batches
            num_batches = len(self.data.train_dataloader())
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            # Validation
            val_loss = self.validate(epoch)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Logging
            if epoch % self.log_period == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_losses['lr'] = current_lr
                self.logger.log_metrics(epoch_losses, step=epoch)

                # Print progress
                print(f"\nEpoch {epoch}: train_loss={epoch_losses['loss/train']:.6f}, "
                      f"val_loss={val_loss:.6f}, lr={current_lr:.6f}")

            # Checkpointing
            if epoch % self.save_period == 0 or epoch == self.max_epochs - 1:
                self.save_checkpoint(epoch)

        print(f"\n[Trainer] Training completed!")
        self.logger.print_summary()
        self.logger.close()

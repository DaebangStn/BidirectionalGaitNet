import torch
import torch.nn as nn
from typing import List, Dict, Any


class ResidualMLP(nn.Module):
    """Multi-layer perceptron with residual connections."""

    def __init__(self, layers: List[int], input_dim: int, output_dim: int, activation: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        # Input layer
        self.input_layer = nn.Linear(input_dim, layers[0])
        self.input_activation = activation

        # Hidden layers with residual connections
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))

            # Add skip connection if dimensions match, otherwise use a projection
            if layers[i-1] == layers[i]:
                self.skip_connections.append(nn.Identity())
            else:
                self.skip_connections.append(nn.Linear(layers[i-1], layers[i]))

        # Output layer
        self.output_layer = nn.Linear(layers[-1], output_dim)

        # Activation for hidden layers - create a new instance to avoid sharing
        self.activation = activation.__class__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.input_activation(self.input_layer(x))

        # Hidden layers with residual connections
        for i, (layer, skip) in enumerate(zip(self.layers, self.skip_connections)):
            residual = skip(x)
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to last hidden layer
                x = self.activation(x)
            x = x + residual  # Add residual connection

        # Output layer
        x = self.output_layer(x)
        return x


class SequentialMLP(nn.Module):
    """Standard sequential MLP without residual connections."""

    def __init__(self, layers: List[int], input_dim: int, output_dim: int, activation: nn.Module):
        super().__init__()
        mlp = nn.Sequential()

        for i, layer in enumerate(layers):
            if i == 0:
                mlp.add_module("input", nn.Linear(input_dim, layer))
                mlp.add_module("act0", activation)
            else:
                mlp.add_module(f"hidden{i}", nn.Linear(layers[i - 1], layer))
                mlp.add_module(f"act{i}", activation if i < len(layers) - 1 else nn.Tanh())

        mlp.add_module("output", nn.Linear(layers[-1], output_dim))
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RegressionNet(nn.Module):
    """Regression network with normalization and configurable architecture."""

    def __init__(self, input_dim: int, output_dim: int, layers: List[int],
                 input_mean: torch.Tensor, input_std: torch.Tensor,
                 target_mean: torch.Tensor, target_std: torch.Tensor,
                 residual: bool = False, activation: str = 'gelu'):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            layers: Hidden layer dimensions (e.g., [64, 32])
            input_mean: Mean for input normalization
            input_std: Std for input normalization
            target_mean: Mean for target denormalization
            target_std: Std for target denormalization
            residual: Whether to use residual connections
            activation: Activation function ('gelu', 'silu', 'relu', 'tanh')
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.residual = residual

        # Normalization parameters (registered as buffers, not parameters)
        self.register_buffer('input_mean', input_mean)
        self.register_buffer('input_std', input_std)
        self.register_buffer('target_mean', target_mean)
        self.register_buffer('target_std', target_std)

        # Activation function
        activation_map = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        self._activation = activation_map.get(activation.lower(), nn.GELU())

        # Build network
        if residual:
            self.mlp = ResidualMLP(layers, input_dim, output_dim, self._activation)
        else:
            self.mlp = SequentialMLP(layers, input_dim, output_dim, self._activation)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using stored statistics."""
        return (x - self.input_mean) / self.input_std

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output using stored statistics."""
        return y * self.target_std + self.target_mean

    def forward(self, x: torch.Tensor, normalize: bool = True, denormalize: bool = False) -> torch.Tensor:
        """
        Forward pass with optional normalization/denormalization.

        Args:
            x: Input tensor
            normalize: Whether to normalize input
            denormalize: Whether to denormalize output

        Returns:
            Output tensor
        """
        if normalize:
            x = self.normalize_input(x)

        y = self.mlp(x)

        if denormalize:
            y = self.denormalize_output(y)

        return y

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for checkpointing."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'layers': self.layers,
            'residual': self.residual,
            'input_mean': self.input_mean.cpu().numpy().tolist(),
            'input_std': self.input_std.cpu().numpy().tolist(),
            'target_mean': self.target_mean.cpu().numpy().tolist(),
            'target_std': self.target_std.cpu().numpy().tolist(),
        }

    def load_from_checkpoint(self, checkpoint_path: str, device: str = 'cuda'):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        return self

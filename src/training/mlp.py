"""Simple MLP classifier for feature classification."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Simple MLP classifier for frozen features.

    A straightforward multi-layer perceptron for classification
    of features extracted from pretrained models.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (B, input_dim)

        Returns:
            Logits of shape (B, output_dim)
        """
        return self.mlp(x)


class LinearClassifier(nn.Module):
    """Simple linear classifier (single layer).

    Useful as a baseline for linear probing.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 100):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ResidualMLP(nn.Module):
    """MLP with residual connections for better gradient flow."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 100,
        num_blocks: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_proj(x))

        for block in self.blocks:
            x = block(x)

        return self.output(x)


class ResidualBlock(nn.Module):
    """Single residual block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn(self.norm1(x))
        return self.norm2(x)

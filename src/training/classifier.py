"""
Multi-view fusion classifier
"""
import torch
import torch.nn as nn
from typing import List


class MultiViewClassifier(nn.Module):
    """Fusion classifier for multi-view features"""

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int = 196,
        hidden_dims: List[int] = [1024, 512],
        dropout: List[float] = [0.5, 0.3],
    ):
        """
        Args:
            feature_dims: List of feature dimensions for each view
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rates for each hidden layer
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.input_dim = sum(feature_dims)
        self.num_classes = num_classes

        # Build MLP layers
        layers = []
        prev_dim = self.input_dim

        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_dims, dropout)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, *features):
        """
        Forward pass

        Args:
            *features: Variable number of feature tensors

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Concatenate all features
        fused = torch.cat(features, dim=1)

        return self.classifier(fused)

    def get_fused_dim(self):
        """Get fused feature dimension"""
        return self.input_dim


class SingleViewClassifier(nn.Module):
    """Single-view classifier for baseline"""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 196,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        """
        Forward pass

        Args:
            features: Feature tensor [batch_size, feature_dim]

        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        return self.classifier(features)

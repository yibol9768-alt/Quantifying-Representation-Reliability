"""Configuration for CIFAR-100 experiments."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Experiment configuration."""

    # Model settings
    model_type: str = "mae"  # mae, clip, dino, fusion

    # Training settings
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 128
    weight_decay: float = 0.01

    # MLP settings
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3

    # Data settings
    data_dir: str = "./data"
    num_workers: int = 4

    # Device settings
    device: str = "cuda:0"

    # Feature dimensions
    FEATURE_DIMS = {
        "mae": 768,
        "clip": 512,
        "dino": 768,
    }

    @property
    def feature_dim(self) -> int:
        """Get feature dimension based on model type."""
        if self.model_type == "fusion":
            return sum(self.FEATURE_DIMS.values())  # 2048
        return self.FEATURE_DIMS.get(self.model_type, 768)

    # CIFAR-100 classes
    NUM_CLASSES: int = 100

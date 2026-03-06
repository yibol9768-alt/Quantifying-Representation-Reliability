"""
Configuration settings
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Config:
    """Main configuration"""

    # Device
    device: str = "cuda"

    # Data
    data_root: str = "/root/autodl-tmp/data"
    dataset: str = "stanford_cars"  # Dataset name
    num_classes: int = 196  # Will be set based on dataset

    # Feature extraction
    feature_dir: str = "/root/autodl-tmp/features"
    models_to_use: List[str] = field(default_factory=lambda: ["clip", "dino", "mae"])

    # Training
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2

    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    dropout: List[float] = field(default_factory=lambda: [0.5, 0.3])

    # Output
    output_dir: str = "/root/autodl-tmp/outputs"
    checkpoint_dir: str = "/root/autodl-tmp/outputs/checkpoints"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        import os
        from src.data import DATASET_INFO

        # Set num_classes based on dataset
        if self.dataset in DATASET_INFO:
            self.num_classes = DATASET_INFO[self.dataset]["num_classes"]

        # Create directories
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# Model configurations
MODEL_CONFIGS = {
    "clip": {
        "feature_dim": 512,
        "model_name": "ViT-B/32",
    },
    "dino": {
        "feature_dim": 768,
        "model_name": "dino_vitb16",
    },
    "mae": {
        "feature_dim": 768,
        "model_name": "facebook/vit-mae-base",
    },
}


# Dataset configurations
DATASET_CONFIGS = {
    "stanford_cars": {
        "num_classes": 196,
        "data_path": "stanford_cars",
        "type": "fine-grained",
    },
    "cifar10": {
        "num_classes": 10,
        "data_path": "cifar10",
        "type": "general",
    },
    "cifar100": {
        "num_classes": 100,
        "data_path": "cifar100",
        "type": "general",
    },
    "flowers102": {
        "num_classes": 102,
        "data_path": "flowers102",
        "type": "fine-grained",
    },
    "pets": {
        "num_classes": 37,
        "data_path": "pets",
        "type": "fine-grained",
    },
    "food101": {
        "num_classes": 101,
        "data_path": "food101",
        "type": "fine-grained",
    },
}


# Experiment configurations
EXPERIMENTS = {
    "single_clip": {
        "name": "CLIP Single",
        "models": ["clip"],
        "feature_dims": [512],
    },
    "single_dino": {
        "name": "DINO Single",
        "models": ["dino"],
        "feature_dims": [768],
    },
    "single_mae": {
        "name": "MAE Single",
        "models": ["mae"],
        "feature_dims": [768],
    },
    "fusion_clip_dino": {
        "name": "CLIP + DINO",
        "models": ["clip", "dino"],
        "feature_dims": [512, 768],
    },
    "fusion_all": {
        "name": "CLIP + DINO + MAE",
        "models": ["clip", "dino", "mae"],
        "feature_dims": [512, 768, 768],
    },
}


def get_config(dataset: str = "stanford_cars", **kwargs) -> Config:
    """
    Get config with specific dataset

    Args:
        dataset: Dataset name
        **kwargs: Additional config overrides

    Returns:
        Config instance
    """
    config = Config(**kwargs)
    config.dataset = dataset

    # Update num_classes based on dataset
    from src.data import DATASET_INFO
    if dataset in DATASET_INFO:
        config.num_classes = DATASET_INFO[dataset]["num_classes"]

    return config

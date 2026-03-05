"""
Configuration settings
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Main configuration"""

    # Device
    device: str = "cuda"

    # Data
    data_root: str = "stanford_cars"
    num_classes: int = 196

    # Feature extraction
    feature_dir: str = "features"
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
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        import os
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

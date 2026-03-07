"""
Pre-trained vision model wrappers

Supports: CLIP, DINO, MAE
"""

from .base import BaseModel
from .clip_model import CLIPModel
from .dino_model import DINOModel
from .mae_model import MAEModel
from .multilayer_models import (
    CLIPMultiLayerModel,
    DINOMultiLayerModel,
    MAEMultiLayerModel,
)

__all__ = [
    "BaseModel",
    "CLIPModel",
    "DINOModel",
    "MAEModel",
    "CLIPMultiLayerModel",
    "DINOMultiLayerModel",
    "MAEMultiLayerModel",
]

# Model configurations
MODEL_CONFIGS = {
    "clip": {
        "class": CLIPModel,
        "feature_dim": 512,
        "model_name": "ViT-B/32",
    },
    "dino": {
        "class": DINOModel,
        "feature_dim": 768,
        "model_name": "dino_vitb16",
    },
    "mae": {
        "class": MAEModel,
        "feature_dim": 768,
        "model_name": "vit-mae-base",
    },
}


def get_model(model_type: str, device: str = "cuda"):
    """Get model instance by type"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_type]
    return config["class"](device=device)

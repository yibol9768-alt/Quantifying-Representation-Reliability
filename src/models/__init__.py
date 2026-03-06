"""
Pre-trained visual model wrappers

Support: CLIP, DINO, MAE, and multi-layer variants for COMM fusion
"""

from .base import BaseModel
from .clip_model import CLIPModel
from .dino_model import DINOModel
from .mae_model import MAEModel
from .clip_multilayer import CLIPMultiLayerModel, LLNLayerscale
from .dino_multilayer import DINOMultiLayerModel
from .mae_multilayer import MAEMultiLayerModel

__all__ = [
    "BaseModel",
    "CLIPModel",
    "DINOModel",
    "MAEModel",
    "CLIPMultiLayerModel",
    "DINOMultiLayerModel",
    "MAEMultiLayerModel",
    "LLNLayerscale",
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

# Multi-layer model configurations for COMM fusion
MULTILAYER_MODEL_CONFIGS = {
    "clip_multilayer": {
        "class": CLIPMultiLayerModel,
        "num_layers": 12,  # ViT-B has 12 layers
        "hidden_dim": 768,
        "output_dim": 512,
        "layers_to_extract": list(range(12)),  # All layers
    },
    "dino_multilayer": {
        "class": DINOMultiLayerModel,
        "num_layers": 12,  # ViT-B has 12 layers
        "hidden_dim": 768,
        "output_dim": 768,
        "layers_to_extract": list(range(6, 12)),  # Deep layers (6-11)
    },
    "mae_multilayer": {
        "class": MAEMultiLayerModel,
        "num_layers": 12,  # ViT-Base has 12 layers
        "hidden_dim": 768,
        "output_dim": 768,
        "layers_to_extract": list(range(6, 12)),  # Deep layers (6-11), same as DINO
    },
}


def get_model(model_type: str, device: str = "cuda"):
    """Get model instance by type"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_type]
    return config["class"](device=device)


def get_multilayer_model(model_type: str, device: str = "cuda"):
    """Get multi-layer model instance by type"""
    if model_type not in MULTILAYER_MODEL_CONFIGS:
        raise ValueError(f"Unknown multi-layer model type: {model_type}. Available: {list(MULTILAYER_MODEL_CONFIGS.keys())}")
    config = MULTILAYER_MODEL_CONFIGS[model_type]
    return config["class"](device=device)

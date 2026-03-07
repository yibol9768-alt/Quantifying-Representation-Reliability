"""
Training module
"""

from .classifier import MultiViewClassifier, SingleViewClassifier
from .fusion import (
    COMMTokenFusion,
    MMViTTokenFusion,
    create_fusion_model,
)
from .trainer import Trainer, FeatureDataset

__all__ = [
    "MultiViewClassifier",
    "SingleViewClassifier",
    "MMViTTokenFusion",
    "COMMTokenFusion",
    "create_fusion_model",
    "Trainer",
    "FeatureDataset",
]

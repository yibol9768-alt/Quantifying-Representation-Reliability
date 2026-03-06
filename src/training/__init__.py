"""
Training module
"""

from .classifier import MultiViewClassifier, SingleViewClassifier
from .trainer import Trainer, FeatureDataset
from .comm_fusion import (
    COMMClassifier,
    ConcatFusionClassifier,
    WeightedSumFusionClassifier,
    LLNLayerscaleFusion,
)

__all__ = [
    "MultiViewClassifier",
    "SingleViewClassifier",
    "Trainer",
    "FeatureDataset",
    "COMMClassifier",
    "ConcatFusionClassifier",
    "WeightedSumFusionClassifier",
    "LLNLayerscaleFusion",
]

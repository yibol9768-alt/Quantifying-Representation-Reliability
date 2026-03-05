"""
Training module
"""

from .classifier import MultiViewClassifier, SingleViewClassifier
from .trainer import Trainer, FeatureDataset

__all__ = [
    "MultiViewClassifier",
    "SingleViewClassifier",
    "Trainer",
    "FeatureDataset",
]

"""
Training module
"""

from .classifier import MultiViewClassifier, SingleViewClassifier
from .trainer import Trainer, FeatureDataset
from .comm_fusion import (
    COMMFusionClassifier,
    ConcatFusionClassifier,
    WeightedSumFusionClassifier,
    LLNLayerscaleFusion,
)
from .comm3_fusion import (
    COMM3FusionClassifier,
    AlignmentMLP,
)
from .mmvit_fusion import (
    MMViTFusionClassifier,
    MMViTLiteFusionClassifier,
    MultiViewCrossAttention,
    MultiscaleFusionBlock,
)

__all__ = [
    "MultiViewClassifier",
    "SingleViewClassifier",
    "Trainer",
    "FeatureDataset",
    "COMMFusionClassifier",
    "ConcatFusionClassifier",
    "WeightedSumFusionClassifier",
    "LLNLayerscaleFusion",
    "COMM3FusionClassifier",
    "AlignmentMLP",
    "MMViTFusionClassifier",
    "MMViTLiteFusionClassifier",
    "MultiViewCrossAttention",
    "MultiscaleFusionBlock",
]

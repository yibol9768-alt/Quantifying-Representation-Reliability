"""
Training module - Unified fusion classifiers
"""

from .fusion import (
    # Factory
    create_fusion_model,
    # Main classes
    ConcatFusion,
    WeightedSumFusion,
    COMMFusion,
    MMViTFusion,
    MMViTLiteFusion,
    MMViTCrossAttentionBlock,
    # Components
    LLNLayerscaleFusion,
    AlignmentMLP,
    # Backward compatibility
    MultiViewClassifier,
    SingleViewClassifier,
    COMMFusionClassifier,
    COMM3FusionClassifier,
    MMViTFusionClassifier,
    MMViTLiteFusionClassifier,
)

from .trainer import Trainer, FeatureDataset

__all__ = [
    "create_fusion_model",
    "ConcatFusion",
    "WeightedSumFusion",
    "COMMFusion",
    "MMViTFusion",
    "MMViTLiteFusion",
    "MMViTCrossAttentionBlock",
    "LLNLayerscaleFusion",
    "AlignmentMLP",
    "Trainer",
    "FeatureDataset",
    "MultiViewClassifier",
    "SingleViewClassifier",
    "COMMFusionClassifier",
    "COMM3FusionClassifier",
    "MMViTFusionClassifier",
    "MMViTLiteFusionClassifier",
]

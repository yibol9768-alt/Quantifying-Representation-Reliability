"""Simple MLP classifier for HuggingFace Trainer."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """MLP classifier for frozen features.

    Compatible with HuggingFace Trainer.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 100,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (B, feature_dim)

        Returns:
            Logits (B, num_classes)
        """
        return self.classifier(features)


class FeatureClassifier(nn.Module):
    """Full model: Feature Extractor + MLP Classifier.

    This wraps the feature extractor and classifier together
    for end-to-end forward pass.
    """

    def __init__(
        self,
        model_type: str = "mae",
        num_classes: int = 100,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Feature extractor
        from .extractors import get_extractor
        self.extractor = get_extractor(model_type)

        # Classifier
        self.classifier = MLPClassifier(
            feature_dim=self.extractor.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features (no grad for frozen backbone)."""
        return self.extractor(pixel_values)

    def forward(
        self,
        pixel_values: torch.Tensor = None,
        features: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """Forward pass compatible with HF Trainer.

        Args:
            pixel_values: Raw images (will extract features)
            features: Pre-extracted features (if provided, skip extraction)
            labels: Ground truth labels

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        if features is None:
            features = self.extract_features(pixel_values)

        logits = self.classifier(features)

        output = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

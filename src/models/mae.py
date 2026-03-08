"""MAE (Masked Autoencoder) feature extractor."""

import torch
import torch.nn as nn
import timm


class MAEExtractor(nn.Module):
    """MAE ViT feature extractor using timm.

    Uses pretrained MAE ViT-Base model from timm.
    Feature dimension: 768
    """

    def __init__(self, model_name: str = "vit_base_patch16_224.mae"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Feature tensor of shape (B, 768)
        """
        # Get patch tokens and CLS token
        features = self.model.forward_features(x)

        # Global average pooling over patch tokens
        # features shape: (B, 196+1, 768) -> (B, 768)
        cls_token = features[:, 0]  # CLS token

        return cls_token


class MAEExtractorV2(nn.Module):
    """Alternative MAE extractor using facebook's official weights.

    Uses the official MAE pretrained weights.
    """

    def __init__(self):
        super().__init__()
        # Use timm with facebook mae weights
        self.model = timm.create_model(
            "vit_base_patch16_224.mae",
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Feature tensor of shape (B, 768)
        """
        return self.model(x)

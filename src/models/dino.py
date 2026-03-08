"""DINO (Self-Distillation with No Labels) feature extractor."""

import torch
import torch.nn as nn
import timm


class DINOExtractor(nn.Module):
    """DINO ViT feature extractor using timm.

    Uses pretrained DINO ViT-B/16 model.
    Feature dimension: 768
    """

    def __init__(self, model_name: str = "vit_base_patch16_224.dino"):
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
        # Get CLS token
        features = self.model.forward_features(x)

        # CLS token is the first token
        cls_token = features[:, 0]

        return cls_token


class DINOv2Extractor(nn.Module):
    """DINOv2 feature extractor (newer version).

    Uses DINOv2 ViT-B/14 model with better features.
    Feature dimension: 768
    """

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        features = self.model.forward_features(x)
        cls_token = features[:, 0]
        return cls_token


class DINOExtractorTorchHub(nn.Module):
    """DINO extractor using PyTorch Hub (official Facebook implementation).

    This uses the official DINO models from Facebook Research.
    """

    def __init__(self, model_name: str = "dino_vitb16"):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.model(x)

"""CLIP feature extractor."""

import torch
import torch.nn as nn
import clip


class CLIPExtractor(nn.Module):
    """CLIP visual encoder feature extractor.

    Uses OpenAI's pretrained CLIP ViT-B/16 model.
    Feature dimension: 512
    """

    def __init__(self, model_name: str = "ViT-B/16"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features from input images.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Feature tensor of shape (B, 512)
        """
        # CLIP expects specific normalization
        # If input is already normalized, this may need adjustment
        features = self.model.encode_image(x)

        # Normalize features (optional, but often helpful)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.float()


class CLIPExtractorV2(nn.Module):
    """CLIP extractor using OpenCLIP for more model options.

    Supports various CLIP variants through OpenCLIP.
    """

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except ImportError:
            raise ImportError(
                "OpenCLIP not installed. Install with: pip install open-clip-torch"
            )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual features."""
        features = self.model.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

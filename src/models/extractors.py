"""Feature extractors using HuggingFace Transformers - offline mode."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoImageProcessor,
    ViTMAEModel,
    CLIPVisionModel,
    Dinov2Model,
)


class FeatureExtractor(nn.Module):
    """Feature extractor using local HuggingFace models.

    Models must be downloaded manually. See README for download instructions.
    """

    # Local model paths
    MODEL_PATHS = {
        "mae": {
            "path": "./models/vit-mae-base",  # Local path
            "hf_name": "facebook/vit-mae-base",
            "dim": 768,
        },
        "clip": {
            "path": "./models/clip-vit-base-patch16",
            "hf_name": "openai/clip-vit-base-patch16",
            "dim": 512,
        },
        "dino": {
            "path": "./models/dinov2-base",
            "hf_name": "facebook/dinov2-base",
            "dim": 768,
        },
    }

    def __init__(self, model_type: str = "mae"):
        super().__init__()
        self.model_type = model_type

        config = self.MODEL_PATHS[model_type]
        model_path = config["path"]
        self.feature_dim = config["dim"]

        # Check if model exists locally
        from pathlib import Path
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Please download manually:\n"
                f"  huggingface-cli download {config['hf_name']} --local-dir {model_path}\n"
                f"See README for details."
            )

        # Load model from local path
        if model_type == "mae":
            self.model = ViTMAEModel.from_pretrained(model_path, local_files_only=True)
        elif model_type == "clip":
            self.model = CLIPVisionModel.from_pretrained(model_path, local_files_only=True)
        elif model_type == "dino":
            self.model = Dinov2Model.from_pretrained(model_path, local_files_only=True)

        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)

        # Freeze backbone
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        outputs = self.model(pixel_values=pixel_values)

        if self.model_type == "mae":
            return outputs.last_hidden_state[:, 0]
        elif self.model_type == "clip":
            return outputs.pooler_output
        elif self.model_type == "dino":
            return outputs.last_hidden_state[:, 0]
        else:
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0]


class MultiModelExtractor(nn.Module):
    """Extract and fuse features from multiple models.

    Fusion strategy: concatenation of L2-normalized features.
    """

    def __init__(self, model_types: list = ["mae", "clip", "dino"]):
        super().__init__()
        self.model_types = model_types

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name) for name in model_types
        })

        self.feature_dim = sum(
            FeatureExtractor.MODEL_PATHS[name]["dim"]
            for name in model_types
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract and fuse features."""
        features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            if normalize:
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            features.append(feat)

        return torch.cat(features, dim=-1)


def get_extractor(model_type: str) -> nn.Module:
    """Factory function to create feature extractor."""
    if model_type == "fusion":
        return MultiModelExtractor(["mae", "clip", "dino"])
    else:
        return FeatureExtractor(model_type)

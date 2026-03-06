"""
MAE model with multi-layer feature extraction for extended COMM fusion

Extended to support 3-model fusion: CLIP + DINO + MAE
Based on original COMM: "From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models"
https://arxiv.org/abs/2310.08825

Note: Uses last 6 layers (7-12) for MAE, similar to DINO, as deep layers contain
more semantic features relevant for classification tasks.
"""
import torch
import torch.nn as nn
from transformers import ViTMAEModel, AutoImageProcessor
from typing import List, Dict, Optional
from PIL import Image
import os

from .base import BaseModel


class MAEMultiLayerModel(BaseModel):
    """
    MAE model that extracts features from specified transformer layers.

    For extended COMM fusion with 3 models, we extract features from deep layers
    (last 6 layers, similar to DINO) for better semantic representation.

    Default: layers 7-12 for ViT-Base (12 layers total)
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "facebook/vit-mae-base",
        layers_to_extract: Optional[List[int]] = None,
    ):
        """
        Args:
            device: Device to use
            model_name: MAE model name (default: facebook/vit-mae-base)
            layers_to_extract: List of layer indices to extract (0-indexed).
                               If None, extract deep layers (last 6 layers).
        """
        super().__init__(device)
        self.model_name = model_name
        self.layers_to_extract = layers_to_extract

        # Setup dimensions based on model variant
        self._setup_dimensions()

        # Default: extract deep layers (last 6 layers, similar to DINO)
        if self.layers_to_extract is None:
            self.layers_to_extract = list(range(self.num_layers - 6, self.num_layers))

        self.load_model()

    def _setup_dimensions(self):
        """Setup feature dimensions based on model variant"""
        # MAE ViT-Base: 12 layers, hidden dim 768
        # MAE ViT-Large: 24 layers, hidden dim 1024
        if "large" in self.model_name.lower():
            self.num_layers = 24
            self.hidden_dim = 1024
        else:
            # ViT-Base (default)
            self.num_layers = 12
            self.hidden_dim = 768

        self.feature_dim = self.hidden_dim

    def load_model(self):
        """Load MAE model from Hugging Face (with mirror support)"""
        mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            mirror=mirror
        )
        self.model = ViTMAEModel.from_pretrained(
            self.model_name,
            mirror=mirror
        ).to(self.device)
        self.model.eval()

    def get_transform(self):
        """Get MAE preprocessing transform"""
        return self.processor

    def extract_multilayer_features(
        self,
        image: Image.Image,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multi-layer features from MAE model.

        Args:
            image: PIL Image

        Returns:
            Dict mapping layer index to feature tensor
        """
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract multi-layer features
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract CLS token from each specified layer
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1, batch, seq_len, hidden)
        features = {}

        for layer_idx in self.layers_to_extract:
            # Get CLS token representation
            cls_feature = hidden_states[layer_idx][:, 0, :]  # [1, hidden_dim]

            # L2 normalize (consistent with single-layer MAE)
            cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)

            features[layer_idx] = cls_feature

        return features

    def extract_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Extract single-layer feature (last layer) for backward compatibility.

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Feature vector [1, 768]
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        feature = outputs.last_hidden_state[:, 0, :]
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature

"""
DINO model with multi-layer feature extraction for COMM fusion

Based on: "From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models"
https://arxiv.org/abs/2310.08825

Note: COMM uses DINOv2 features from layers 19-24 (deep layers only).
For ViT-B models with 12 layers, we use layers 7-12.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Dict, Optional
from PIL import Image

from .base import BaseModel


class DINOMultiLayerModel(BaseModel):
    """
    DINO model that extracts features from specified transformer layers.

    For COMM fusion, we extract features from deep layers (19-24 for ViT-L, 7-12 for ViT-B)
    and apply LLN-Layerscale-MLP fusion strategy.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "dino_vitb16",
        layers_to_extract: Optional[List[int]] = None,
    ):
        """
        Args:
            device: Device to use
            model_name: DINO model name (e.g., "dino_vitb16", "dino_vits16")
            layers_to_extract: List of layer indices to extract (0-indexed).
                               If None, extract deep layers (last 6 layers).
        """
        super().__init__(device)
        self.model_name = model_name
        self.layers_to_extract = layers_to_extract

        # Setup dimensions based on model variant
        self._setup_dimensions()

        # Default: extract deep layers (last 6 layers, following COMM paper)
        # COMM uses layers 19-24 for ViT-L (24 layers), so for ViT-B we use 7-12
        if self.layers_to_extract is None:
            self.layers_to_extract = list(range(self.num_layers - 6, self.num_layers))

        self.load_model()

    def _setup_dimensions(self):
        """Setup feature dimensions based on model variant"""
        # DINO ViT-B has 12 layers, hidden dim 768
        # ViT-S has 12 layers, hidden dim 384
        if "vitb" in self.model_name.lower():
            self.num_layers = 12
            self.hidden_dim = 768
        elif "vits" in self.model_name.lower():
            self.num_layers = 12
            self.hidden_dim = 384
        else:
            # Default to ViT-B
            self.num_layers = 12
            self.hidden_dim = 768

        self.feature_dim = self.hidden_dim

    def load_model(self):
        """Load DINO model from torch hub"""
        self.model = torch.hub.load('facebookresearch/dino:main', self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # DINO preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_transform(self):
        """Get DINO preprocessing transform"""
        return self.preprocess

    def extract_multilayer_features(
        self,
        image: Image.Image,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract features from multiple layers of DINO visual encoder.

        Args:
            image: PIL Image

        Returns:
            Dict mapping layer indices to feature tensors [1, hidden_dim]
        """
        # Preprocess
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        features = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output shape: [batch, seq_len, hidden_dim]
                features[layer_idx] = output.detach()
            return hook

        hooks = []

        # Register hooks for specified layers
        # DINO ViT structure: model.blocks[i]
        for i in self.layers_to_extract:
            hook = self.model.blocks[i].register_forward_hook(make_hook(i))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(image_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Extract CLS token (first token) from each layer
        result = {}
        for layer_idx, feat in features.items():
            # feat: [1, num_patches+1, hidden_dim]
            # Take CLS token (index 0)
            cls_token = feat[:, 0, :]  # [1, hidden_dim]
            result[layer_idx] = cls_token

        return result

    def extract_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Extract final layer feature (for compatibility with BaseModel)

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Normalized feature vector [1, hidden_dim]
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(image_input)

        # L2 normalize
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature

    def extract_batch_multilayer_features(
        self,
        images: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract multi-layer features from a batch of images.

        Args:
            images: Preprocessed image tensor [batch_size, 3, H, W]

        Returns:
            Dict mapping layer indices to feature tensors [batch_size, hidden_dim]
        """
        features = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                features[layer_idx] = output.detach()
            return hook

        hooks = []

        # Register hooks
        for i in self.layers_to_extract:
            hook = self.model.blocks[i].register_forward_hook(make_hook(i))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(images.to(self.device))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Extract CLS tokens
        result = {}
        for layer_idx, feat in features.items():
            cls_token = feat[:, 0, :]  # [batch, hidden_dim]
            result[layer_idx] = cls_token

        return result

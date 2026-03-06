"""
CLIP model with multi-layer feature extraction for COMM fusion

Based on: "From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models"
https://arxiv.org/abs/2310.08825
"""
import torch
import torch.nn as nn
import clip
from typing import List, Dict, Optional, Tuple
from PIL import Image

from .base import BaseModel


class CLIPMultiLayerModel(BaseModel):
    """
    CLIP model that extracts features from all transformer layers.

    For COMM fusion, we extract features from all layers of CLIP ViT
    and apply LLN-Layerscale fusion strategy.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "ViT-B/32",
        layers_to_extract: Optional[List[int]] = None,
    ):
        """
        Args:
            device: Device to use
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-B/16")
            layers_to_extract: List of layer indices to extract (0-indexed).
                               If None, extract all layers.
        """
        super().__init__(device)
        self.model_name = model_name
        self.layers_to_extract = layers_to_extract
        self.load_model()

        # Get model dimensions
        self._setup_dimensions()

    def _setup_dimensions(self):
        """Setup feature dimensions based on model variant"""
        # CLIP ViT-B has 12 layers, hidden dim 768 (visual) or 512 (projected)
        # ViT-L has 24 layers, hidden dim 1024
        if "ViT-B" in self.model_name:
            self.num_layers = 12
            self.hidden_dim = 768
            self.output_dim = 512
        elif "ViT-L" in self.model_name:
            self.num_layers = 24
            self.hidden_dim = 1024
            self.output_dim = 768
        else:
            # Default to ViT-B
            self.num_layers = 12
            self.hidden_dim = 768
            self.output_dim = 512

        self.feature_dim = self.hidden_dim  # Use hidden dim for multi-layer features

        # Default: extract all layers
        if self.layers_to_extract is None:
            self.layers_to_extract = list(range(self.num_layers))

    def load_model(self):
        """Load CLIP model from OpenAI"""
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

    def get_transform(self):
        """Get CLIP preprocessing transform"""
        return self.preprocess

    def _register_hooks(self) -> Dict[str, torch.Tensor]:
        """Register forward hooks to extract intermediate layer outputs"""
        features = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                features[f"layer_{layer_idx}"] = output
            return hook

        hooks = []

        # Get visual transformer layers
        if hasattr(self.model.visual, 'transformer'):
            # CLIP ViT structure
            for i, block in enumerate(self.model.visual.transformer.resblocks):
                if i in self.layers_to_extract:
                    hook = block.register_forward_hook(make_hook(i))
                    hooks.append(hook)

        return features, hooks

    def extract_multilayer_features(
        self,
        image: Image.Image,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple layers of CLIP visual encoder.

        Args:
            image: PIL Image

        Returns:
            Dict mapping layer indices to feature tensors
            Each tensor is [1, num_patches+1, hidden_dim] or pooled to [1, hidden_dim]
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
        for i, block in enumerate(self.model.visual.transformer.resblocks):
            if i in self.layers_to_extract:
                hook = block.register_forward_hook(make_hook(i))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model.encode_image(image_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Extract CLS token (first token) from each layer
        result = {}
        for layer_idx, feat in features.items():
            # feat: [1, num_patches+1, hidden_dim]
            # Take CLS token (index 0)
            cls_token = feat[:, 0, :]  # [1, hidden_dim]
            result[f"layer_{layer_idx}"] = cls_token

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
            feature = self.model.encode_image(image_input)

        # Project to hidden dim if needed (for consistency with multi-layer features)
        # Actually, just return the CLIP output (already projected)
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
        for i, block in enumerate(self.model.visual.transformer.resblocks):
            if i in self.layers_to_extract:
                hook = block.register_forward_hook(make_hook(i))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model.encode_image(images.to(self.device))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Extract CLS tokens
        result = {}
        for layer_idx, feat in features.items():
            cls_token = feat[:, 0, :]  # [batch, hidden_dim]
            result[layer_idx] = cls_token

        return result


class LLNLayerscale(nn.Module):
    """
    LLN-Layerscale module for multi-layer feature fusion.

    Applies Linear + LayerNorm to align feature spaces,
    then uses learnable scale parameters for weighted summation.

    From COMM paper:
        v̄ = Σ αᵢ · Linear(LN(vⁱ))
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """
        Args:
            num_layers: Number of layers to fuse
            hidden_dim: Input feature dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LayerNorm for each layer (shared or separate?)
        # Paper uses separate LN for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Linear projection for each layer
        self.linear_projs = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim, bias=True) for _ in range(num_layers)
        ])

        # Learnable scale parameters (initialized to 1/num_layers)
        self.scales = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-layer features.

        Args:
            layer_features: List of tensors [batch, hidden_dim] for each layer

        Returns:
            Fused feature tensor [batch, output_dim]
        """
        assert len(layer_features) == self.num_layers

        # Normalize scales to sum to 1 (softmax)
        scale_weights = torch.softmax(self.scales, dim=0)

        # Apply LLN and weighted sum
        fused = torch.zeros(layer_features[0].shape[0], self.output_dim,
                           device=layer_features[0].device)

        for i, feat in enumerate(layer_features):
            # Linear(LN(feat))
            normalized = self.layer_norms[i](feat)
            projected = self.linear_projs[i](normalized)

            # Weighted sum
            fused = fused + scale_weights[i] * projected

        return fused

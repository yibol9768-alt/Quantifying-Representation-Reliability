"""Base feature extractor for pretrained vision models."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    CLIPVisionModel,
    ConvNextModel,
    Data2VecVisionModel,
    DeiTModel,
    Dinov2Model,
    ResNetModel,
    SiglipVisionModel,
    SwinModel,
    BeitModel,
    ViTMAEModel,
    ViTModel,
)

# Map architecture family name -> HuggingFace model class
_MODEL_CLASSES = {
    "vit": ViTModel,
    "deit": DeiTModel,
    "swin": SwinModel,
    "beit": BeitModel,
    "data2vec": Data2VecVisionModel,
    "mae": ViTMAEModel,
    "dinov2": Dinov2Model,
    "clip": CLIPVisionModel,
    "siglip": SiglipVisionModel,
    "convnext": ConvNextModel,
    "resnet": ResNetModel,
}


def _infer_square_size(num_tokens: int) -> Optional[int]:
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side
    return None


def _resize_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Resize patch tokens to a target length while preserving 2D layout if possible."""
    target_tokens = max(1, int(target_tokens))
    if tokens.size(1) == target_tokens:
        return tokens

    side_in = _infer_square_size(tokens.size(1))
    side_out = _infer_square_size(target_tokens)

    if side_in is not None and side_out is not None:
        bsz, _, dim = tokens.shape
        x = tokens.transpose(1, 2).reshape(bsz, dim, side_in, side_in)
        x = F.interpolate(x, size=(side_out, side_out), mode="bilinear", align_corners=False)
        return x.flatten(2).transpose(1, 2)

    x = tokens.transpose(1, 2)  # [B, D, N]
    x = F.adaptive_avg_pool1d(x, target_tokens)
    return x.transpose(1, 2)


def _split_cls_token(tokens: torch.Tensor, has_cls: bool) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if has_cls and tokens.size(1) > 0:
        return tokens[:, :1], tokens[:, 1:]
    return None, tokens


def _merge_cls_token(cls_token: Optional[torch.Tensor], patch_tokens: torch.Tensor) -> torch.Tensor:
    if cls_token is None:
        return patch_tokens
    return torch.cat([cls_token, patch_tokens], dim=1)


def _apply_depthwise_pool(tokens: torch.Tensor, has_cls: bool, pool: nn.Conv2d) -> torch.Tensor:
    """Apply depthwise 2D pooling on token map; fallback to 1D pooling for non-square tokens."""
    cls_token, patches = _split_cls_token(tokens, has_cls)
    if patches.size(1) == 0:
        return tokens

    side = _infer_square_size(patches.size(1))
    if side is not None:
        bsz, _, dim = patches.shape
        x = patches.transpose(1, 2).reshape(bsz, dim, side, side)
        x = pool(x)
        pooled = x.flatten(2).transpose(1, 2)
    else:
        stride_h, stride_w = pool.stride
        reduce = max(1, int(stride_h) * int(stride_w))
        target_tokens = max(1, patches.size(1) // reduce)
        pooled = _resize_tokens(patches, target_tokens)

    return _merge_cls_token(cls_token, pooled)


def _add_positional_embedding(tokens: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
    if pos_embed.size(1) != tokens.size(1):
        pos_embed = F.interpolate(
            pos_embed.transpose(1, 2),
            size=tokens.size(1),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return tokens + pos_embed


def _valid_num_heads(dim: int, preferred: int) -> int:
    preferred = max(1, min(preferred, dim))
    for h in range(preferred, 0, -1):
        if dim % h == 0:
            return h
    return 1


class FeatureExtractor(nn.Module):
    """Feature extractor using local HuggingFace models.

    Models must be downloaded manually. See README for download instructions.
    """

    MODEL_PATHS = {
        # ── Vision Transformer (supervised) ──────────────────────────
        "vit": {
            "path": "vit-base-patch16",
            "hf_name": "google/vit-base-patch16-224",
            "arch": "vit", "dim": 768,
        },
        "vit_large": {
            "path": "vit-large-patch16",
            "hf_name": "google/vit-large-patch16-224",
            "arch": "vit", "dim": 1024,
        },
        # ── DeiT (data-efficient ViT) ───────────────────────────────
        "deit_small": {
            "path": "deit-small-patch16",
            "hf_name": "facebook/deit-small-patch16-224",
            "arch": "deit", "dim": 384,
        },
        "deit_base": {
            "path": "deit-base-patch16",
            "hf_name": "facebook/deit-base-patch16-224",
            "arch": "deit", "dim": 768,
        },
        # ── Swin Transformer ────────────────────────────────────────
        "swin_tiny": {
            "path": "swin-tiny",
            "hf_name": "microsoft/swin-tiny-patch4-window7-224",
            "arch": "swin", "dim": 768,
        },
        "swin": {
            "path": "swin-base",
            "hf_name": "microsoft/swin-base-patch4-window7-224",
            "arch": "swin", "dim": 1024,
        },
        # ── BEiT ────────────────────────────────────────────────────
        "beit": {
            "path": "beit-base",
            "hf_name": "microsoft/beit-base-patch16-224-pt22k",
            "arch": "beit", "dim": 768,
        },
        "beit_large": {
            "path": "beit-large",
            "hf_name": "microsoft/beit-large-patch16-224-pt22k",
            "arch": "beit", "dim": 1024,
        },
        # ── Data2Vec Vision ─────────────────────────────────────────
        "data2vec": {
            "path": "data2vec-vision-base",
            "hf_name": "facebook/data2vec-vision-base",
            "arch": "data2vec", "dim": 768,
        },
        # ── MAE (self-supervised) ───────────────────────────────────
        "mae": {
            "path": "vit-mae-base",
            "hf_name": "facebook/vit-mae-base",
            "arch": "mae", "dim": 768,
        },
        "mae_large": {
            "path": "vit-mae-large",
            "hf_name": "facebook/vit-mae-large",
            "arch": "mae", "dim": 1024,
        },
        # ── DINOv2 (self-supervised) ────────────────────────────────
        "dinov2_small": {
            "path": "dinov2-small",
            "hf_name": "facebook/dinov2-small",
            "arch": "dinov2", "dim": 384,
        },
        "dino": {
            "path": "dinov2-base",
            "hf_name": "facebook/dinov2-base",
            "arch": "dinov2", "dim": 768,
        },
        "dinov2_large": {
            "path": "dinov2-large",
            "hf_name": "facebook/dinov2-large",
            "arch": "dinov2", "dim": 1024,
        },
        # ── CLIP (contrastive language-image) ───────────────────────
        "clip_base32": {
            "path": "clip-vit-base-patch32",
            "hf_name": "openai/clip-vit-base-patch32",
            "arch": "clip", "dim": 768,
        },
        "clip": {
            "path": "clip-vit-base-patch16",
            "hf_name": "openai/clip-vit-base-patch16",
            "arch": "clip", "dim": 768,
        },
        "clip_large": {
            "path": "clip-vit-large-patch14",
            "hf_name": "openai/clip-vit-large-patch14",
            "arch": "clip", "dim": 1024,
        },
        "openclip": {
            "path": "openclip-vit-b32",
            "hf_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "arch": "clip", "dim": 768,
        },
        # ── SigLIP ──────────────────────────────────────────────────
        "siglip": {
            "path": "siglip-base-patch16-224",
            "hf_name": "google/siglip-base-patch16-224",
            "arch": "siglip", "dim": 768,
        },
        # ── ConvNeXt (modern CNN) ───────────────────────────────────
        "convnext_tiny": {
            "path": "convnext-tiny",
            "hf_name": "facebook/convnext-tiny-224",
            "arch": "convnext", "dim": 768,
        },
        "convnext": {
            "path": "convnext-base",
            "hf_name": "facebook/convnext-base-224",
            "arch": "convnext", "dim": 1024,
        },
        "convnext_large": {
            "path": "convnext-large",
            "hf_name": "facebook/convnext-large-224",
            "arch": "convnext", "dim": 1536,
        },
        # ── ResNet (classic CNN baseline) ───────────────────────────
        "resnet50": {
            "path": "resnet-50",
            "hf_name": "microsoft/resnet-50",
            "arch": "resnet", "dim": 2048,
        },
        "resnet101": {
            "path": "resnet-101",
            "hf_name": "microsoft/resnet-101",
            "arch": "resnet", "dim": 2048,
        },
    }

    def __init__(
        self,
        model_type: str = "mae",
        normalize_input: bool = False,
        model_dir: str = "./models",
    ):
        super().__init__()
        if model_type not in self.MODEL_PATHS:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model_type = model_type
        self.normalize_input = normalize_input
        config = self.MODEL_PATHS[model_type]
        model_path = Path(model_dir) / config["path"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Please download manually:\n"
                f"  huggingface-cli download {config['hf_name']} --local-dir {model_path}\n"
                f"See MODEL_ZOO.md for details."
            )

        # Load model via architecture family lookup
        arch = config.get("arch", model_type)
        model_cls = _MODEL_CLASSES.get(arch)
        if model_cls is None:
            raise ValueError(
                f"Unknown architecture '{arch}' for model '{model_type}'. "
                f"Supported: {list(_MODEL_CLASSES.keys())}"
            )
        self.model = model_cls.from_pretrained(str(model_path), local_files_only=True)

        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)

        image_mean = self.processor.image_mean if self.processor.image_mean is not None else [0.485, 0.456, 0.406]
        image_std = self.processor.image_std if self.processor.image_std is not None else [0.229, 0.224, 0.225]
        if len(image_mean) == 1:
            image_mean = image_mean * 3
        if len(image_std) == 1:
            image_std = image_std * 3
        self.register_buffer("_image_mean", torch.tensor(image_mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("_image_std", torch.tensor(image_std).view(1, -1, 1, 1), persistent=False)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.token_dim = getattr(self.model.config, "hidden_size", config["dim"])
        self.feature_dim = self.token_dim
        self.num_hidden_layers = int(getattr(self.model.config, "num_hidden_layers", 12))

    def _maybe_normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return pixel_values

        x = pixel_values
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        mean = self._image_mean.to(device=x.device, dtype=x.dtype)
        std = self._image_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + 1e-6)

    @staticmethod
    def _split_cls_and_patches(sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if sequence.size(1) <= 1:
            return sequence[:, 0], sequence.new_zeros(sequence.size(0), 0, sequence.size(-1))
        return sequence[:, 0], sequence[:, 1:]

    @torch.no_grad()
    def _run_model(self, pixel_values: torch.Tensor, output_hidden_states: bool = False):
        x = self._maybe_normalize(pixel_values)
        return self.model(pixel_values=x, output_hidden_states=output_hidden_states)

    @staticmethod
    def _ensure_matrix(features: torch.Tensor) -> torch.Tensor:
        """Flatten backbone outputs to [B, D] for downstream fusion/classification."""
        if features.ndim <= 2:
            return features
        if all(dim == 1 for dim in features.shape[2:]):
            return features.flatten(1)
        return features.reshape(features.size(0), -1)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract a single global feature vector."""
        outputs = self._run_model(pixel_values, output_hidden_states=False)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return self._ensure_matrix(outputs.pooler_output)
        return self._ensure_matrix(outputs.last_hidden_state[:, 0])

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract cacheable tensors from frozen backbones."""
        return {"features": self.forward(pixel_values)}

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Rebuild classifier input features from cached tensors."""
        return cached_inputs["features"]

    def release_backbones(self):
        """Free backbone modules once cache extraction is complete."""
        self.model = None
        self.processor = None

    @torch.no_grad()
    def extract_last_tokens(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract CLS token and patch tokens from the last layer."""
        outputs = self._run_model(pixel_values, output_hidden_states=False)
        cls_token, patch_tokens = self._split_cls_and_patches(outputs.last_hidden_state)
        return {"cls": cls_token, "patches": patch_tokens}

    @torch.no_grad()
    def extract_hidden_tokens(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Extract patch tokens from all hidden layers (excluding embedding layer)."""
        outputs = self._run_model(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if hidden_states is None:
            return [self._split_cls_and_patches(outputs.last_hidden_state)[1]]

        token_layers: List[torch.Tensor] = []
        for hidden in hidden_states[1:]:
            _, patch_tokens = self._split_cls_and_patches(hidden)
            token_layers.append(patch_tokens)
        return token_layers

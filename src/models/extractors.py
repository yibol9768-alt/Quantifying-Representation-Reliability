"""Feature extractors and paper-inspired fusion modules for classification."""

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
    Data2VecVisionModel,
    Dinov2Model,
    SiglipVisionModel,
    ViTMAEModel,
    ViTModel,
    SwinModel,
    BeitModel,
    ConvNextModel,
)


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
        # Vision Transformer series
        "vit": {
            "path": "vit-base-patch16",
            "hf_name": "google/vit-base-patch16-224",
            "dim": 768,
        },
        "swin": {
            "path": "swin-base",
            "hf_name": "microsoft/swin-base-patch4-window7-224",
            "dim": 1024,
        },
        "beit": {
            "path": "beit-base",
            "hf_name": "microsoft/beit-base-patch16-224-pt22k",
            "dim": 768,
        },
        "data2vec": {
            "path": "data2vec-vision-base",
            "hf_name": "facebook/data2vec-vision-base",
            "dim": 768,
        },
        # Self-supervised series
        "mae": {
            "path": "vit-mae-base",
            "hf_name": "facebook/vit-mae-base",
            "dim": 768,
        },
        "dino": {
            "path": "dinov2-base",
            "hf_name": "facebook/dinov2-base",
            "dim": 768,
        },
        # CLIP series
        "clip": {
            "path": "clip-vit-base-patch16",
            "hf_name": "openai/clip-vit-base-patch16",
            "dim": 768,
        },
        "openclip": {
            "path": "openclip-vit-b32",
            "hf_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "dim": 768,
        },
        "siglip": {
            "path": "siglip-base-patch16-224",
            "hf_name": "google/siglip-base-patch16-224",
            "dim": 768,
        },
        # Modern CNN
        "convnext": {
            "path": "convnext-base",
            "hf_name": "facebook/convnext-base-224",
            "dim": 1024,
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
                f"  hf download {config['hf_name']} --local-dir {model_path}\n"
                f"See README for details."
            )

        # Load model based on type
        if model_type == "mae":
            self.model = ViTMAEModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "clip":
            self.model = CLIPVisionModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "openclip":
            self.model = CLIPVisionModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "siglip":
            self.model = SiglipVisionModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "dino":
            self.model = Dinov2Model.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "vit":
            self.model = ViTModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "swin":
            self.model = SwinModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "beit":
            self.model = BeitModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "data2vec":
            self.model = Data2VecVisionModel.from_pretrained(str(model_path), local_files_only=True)
        elif model_type == "convnext":
            self.model = ConvNextModel.from_pretrained(str(model_path), local_files_only=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract a single global feature vector."""
        outputs = self._run_model(pixel_values, output_hidden_states=False)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

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


class MultiModelConcatExtractor(nn.Module):
    """Baseline fusion: concatenate L2-normalized global features."""

    def __init__(
        self,
        model_types: Sequence[str],
        output_dim: Optional[int] = None,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })
        self.concat_dim = sum(self.extractors[name].feature_dim for name in self.model_types)
        if output_dim is not None:
            self.output_proj = nn.Sequential(nn.LayerNorm(self.concat_dim), nn.Linear(self.concat_dim, output_dim))
            self.feature_dim = output_dim
            self.trainable = True
        else:
            self.output_proj = nn.Identity()
            self.feature_dim = self.concat_dim
            self.trainable = False

    def forward(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            if normalize:
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            features.append(feat)
        fused = torch.cat(features, dim=-1)
        return self.output_proj(fused)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache per-branch global features for offline fusion training."""
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(
        self,
        cached_inputs: Dict[str, torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Fuse cached branch features without touching backbones."""
        features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            if normalize:
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            features.append(feat)
        fused = torch.cat(features, dim=-1)
        return self.output_proj(fused)

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class MultiModelProjectedConcatExtractor(nn.Module):
    """Baseline A: Project features to same dimension, then concatenate.

    Formula: z = [P_c(f_c); P_d(f_d); P_m(f_m)]

    More reasonable than raw concat because:
    - Different encoders have different output dimensions
    - Different feature distributions
    - Projection makes them more comparable
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 256,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        # Individual projections for each model
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim)
            )
            for name in self.model_types
        })

        self.feature_dim = proj_dim * len(self.model_types)
        self.proj_dim = proj_dim
        self.trainable = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)
        return torch.cat(projected_features, dim=-1)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache projected features for offline fusion training."""
        cached = {}
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            cached[f"feat_{name}"] = feat
        return cached

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse cached features with projection."""
        projected_features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)
        return torch.cat(projected_features, dim=-1)

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class MultiModelWeightedSumExtractor(nn.Module):
    """Baseline B: Learnable weighted sum of projected features.

    Formula: z = α_c·z_c + α_d·z_d + α_m·z_m
    where α = softmax(w) and w is learnable.

    Very simple but effective baseline.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        # Individual projections for each model
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim)
            )
            for name in self.model_types
        })

        # Learnable weights (scalar for each model)
        self.weight_params = nn.Parameter(torch.zeros(len(self.model_types)))

        self.feature_dim = proj_dim
        self.proj_dim = proj_dim
        self.trainable = True

    def _get_weights(self) -> torch.Tensor:
        """Get normalized weights via softmax."""
        return torch.softmax(self.weight_params, dim=0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        weights = self._get_weights()
        fused = torch.zeros(pixel_values.size(0), self.proj_dim, device=pixel_values.device)

        for i, name in enumerate(self.model_types):
            feat = self.extractors[name](pixel_values)
            projected_feat = self.projections[name](feat)
            fused = fused + weights[i] * projected_feat

        return fused

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache features for offline fusion training."""
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse cached features with learnable weights."""
        weights = self._get_weights()
        device = next(iter(cached_inputs.values())).device
        fused = torch.zeros(cached_inputs[f"feat_{self.model_types[0]}"].size(0),
                           self.proj_dim, device=device)

        for i, name in enumerate(self.model_types):
            feat = cached_inputs[f"feat_{name}"]
            projected_feat = self.projections[name](feat)
            fused = fused + weights[i] * projected_feat

        return fused

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class MultiModelGatedFusionExtractor(nn.Module):
    """Baseline C: Gated fusion with sample-wise adaptive weights.

    Formula:
        g = softmax(MLP([z_c; z_d; z_m]))
        z = g_c·z_c + g_d·z_d + g_m·z_m

    More flexible than fixed weighted sum.
    The model learns to trust different encoders for different images.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        hidden_dim: int = 128,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        # Individual projections for each model
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim)
            )
            for name in self.model_types
        })

        # Gate network: takes concatenated features, outputs weights
        total_dim = proj_dim * len(self.model_types)
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.model_types))
        )

        self.feature_dim = proj_dim
        self.proj_dim = proj_dim
        self.trainable = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Get all projected features
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)

        # Compute gate weights
        concat_features = torch.cat(projected_features, dim=-1)
        gate_logits = self.gate_network(concat_features)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, num_models]

        # Apply gated fusion
        fused = torch.zeros_like(projected_features[0])
        for i, proj_feat in enumerate(projected_features):
            fused = fused + gate_weights[:, i:i+1] * proj_feat

        return fused

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache features for offline fusion training."""
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse cached features with gated fusion."""
        # Get all projected features
        projected_features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)

        # Compute gate weights
        concat_features = torch.cat(projected_features, dim=-1)
        gate_logits = self.gate_network(concat_features)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        # Apply gated fusion
        fused = torch.zeros_like(projected_features[0])
        for i, proj_feat in enumerate(projected_features):
            fused = fused + gate_weights[:, i:i+1] * proj_feat

        return fused

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class MultiModelDifferenceAwareExtractor(nn.Module):
    """Baseline D: Concat with pairwise differences.

    Formula: z = [z_c; z_d; z_m; z_c-z_d; z_c-z_m; z_d-z_m]

    Explicitly models "difference information" between representations.
    Very suitable for representation reliability/complementarity analysis.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 256,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        # Individual projections for each model
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim)
            )
            for name in self.model_types
        })

        num_models = len(self.model_types)
        # Original features + pairwise differences
        num_diffs = num_models * (num_models - 1) // 2
        self.feature_dim = proj_dim * (num_models + num_diffs)
        self.proj_dim = proj_dim
        self.trainable = True

    def _compute_differences(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute pairwise differences between features."""
        differences = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                differences.append(features[i] - features[j])
        return differences

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Get projected features
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)

        # Add pairwise differences
        diff_features = self._compute_differences(projected_features)

        # Concatenate all
        all_features = projected_features + diff_features
        return torch.cat(all_features, dim=-1)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache features for offline fusion training."""
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse cached features with differences."""
        # Get projected features
        projected_features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)

        # Add pairwise differences
        diff_features = self._compute_differences(projected_features)

        # Concatenate all
        all_features = projected_features + diff_features
        return torch.cat(all_features, dim=-1)

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class MultiModelHadamardExtractor(nn.Module):
    """Baseline E: Concat with element-wise products (Hadamard interaction).

    Formula: z = [z_c; z_d; z_m; z_c⊙z_d; z_c⊙z_m; z_d⊙z_m]

    Original features preserve individual information.
    Product terms explicitly model which dimensions co-activate.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 256,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        # Individual projections for each model
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim)
            )
            for name in self.model_types
        })

        num_models = len(self.model_types)
        # Original features + pairwise products
        num_pairs = num_models * (num_models - 1) // 2
        self.feature_dim = proj_dim * (num_models + num_pairs)
        self.proj_dim = proj_dim
        self.trainable = True

    def _compute_products(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute pairwise element-wise products between features."""
        products = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                products.append(features[i] * features[j])
        return products

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Get projected features
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)

        # Add pairwise products
        product_features = self._compute_products(projected_features)

        # Concatenate all
        all_features = projected_features + product_features
        return torch.cat(all_features, dim=-1)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache features for offline fusion training."""
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse cached features with Hadamard interactions."""
        # Get projected features
        projected_features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            projected_feat = self.projections[name](feat)
            projected_features.append(projected_feat)

        # Add pairwise products
        product_features = self._compute_products(projected_features)

        # Concatenate all
        all_features = projected_features + product_features
        return torch.cat(all_features, dim=-1)

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class MultiModelLateFusionExtractor(nn.Module):
    """Baseline F: Late Fusion (logit-level fusion/ensemble).

    Instead of fusing features, each encoder gets its own classifier.
    Final prediction is the average of all classifier logits.

    Formula: y = (y_c + y_d + y_m) / 3

    This is NOT a feature-level fusion. It's an ensemble method that
    proves whether fusion gains come from representation complementarity
    or just from voting effects.

    Note: This requires special handling in the training loop since
    it maintains multiple classifiers instead of one.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        num_classes: int,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        self.num_classes = num_classes
        self.feature_dims = {
            name: self.extractors[name].feature_dim
            for name in self.model_types
        }

        # This extractor doesn't produce a single feature_dim
        # It needs special handling in the training loop
        self.trainable = False  # Extractors are frozen
        self.requires_multiple_classifiers = True  # Signal to training loop

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache features for each model separately."""
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract all features - returns dict for late fusion handling."""
        # For late fusion, we return a dict instead of single tensor
        # The training loop needs to handle this specially
        return {
            name: self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return cached features as dict for late fusion."""
        return {
            name: cached_inputs[f"feat_{name}"]
            for name in self.model_types
        }

    def release_backbones(self):
        """Drop frozen backbones once their outputs are cached."""
        self.extractors = None


class ResidualTokenMLP(nn.Module):
    """Residual token-wise MLP used for branch alignment blocks."""

    def __init__(self, dim: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class TokenFeedForward(nn.Module):
    """Standard transformer FFN without an internal residual shortcut."""

    def __init__(self, dim: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class COMMStrictFusionExtractor(nn.Module):
    """COMM-inspired classifier fusion with CLIP-anchor token enhancement."""

    def __init__(
        self,
        model_types: Sequence[str],
        dino_mlp_blocks: int = 2,
        dino_mlp_ratio: float = 8.0,
        output_dim: Optional[int] = None,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("COMM fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })
        self.anchor_model = "clip" if "clip" in self.model_types else self.model_types[0]
        self.anchor_dim = self.extractors[self.anchor_model].token_dim

        self.layer_indices: Dict[str, List[int]] = {}
        self.lln_modules = nn.ModuleDict()
        self.layer_scale_logits = nn.ParameterDict()
        self.aligners = nn.ModuleDict()
        self.branch_gate_logits = nn.ParameterDict()

        for name in self.model_types:
            depth = self.extractors[name].num_hidden_layers
            indices = self._select_layer_indices(name, depth)
            self.layer_indices[name] = indices

            dim = self.extractors[name].token_dim
            lln_list = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
                for _ in indices
            ])
            self.lln_modules[name] = lln_list
            self.layer_scale_logits[name] = nn.Parameter(torch.zeros(len(indices)))
            gate_init = 0.0 if name == self.anchor_model else -2.0
            self.branch_gate_logits[name] = nn.Parameter(torch.full((1,), gate_init))

            if name == self.anchor_model and dim == self.anchor_dim:
                self.aligners[name] = nn.Identity()
            else:
                align_layers: List[nn.Module] = []
                if dino_mlp_blocks > 0:
                    align_layers.extend(ResidualTokenMLP(dim, dino_mlp_ratio) for _ in range(dino_mlp_blocks))
                if dim != self.anchor_dim:
                    align_layers.append(nn.Linear(dim, self.anchor_dim))
                if not align_layers:
                    align_layers.append(nn.Identity())
                self.aligners[name] = nn.Sequential(*align_layers)

        self.final_norm = nn.LayerNorm(self.anchor_dim)
        target_dim = self.anchor_dim if output_dim is None else output_dim
        self.final_proj = nn.Linear(self.anchor_dim, target_dim)
        self.feature_dim = target_dim
        self.trainable = True

    @staticmethod
    def _build_mlp_aligner(dim: int, num_blocks: int, ratio: float) -> nn.Module:
        if num_blocks <= 0:
            return nn.Identity()
        return nn.Sequential(*[ResidualTokenMLP(dim, ratio) for _ in range(num_blocks)])

    @staticmethod
    def _select_layer_indices(model_name: str, depth: int) -> List[int]:
        if depth <= 0:
            return [0]
        if model_name == "clip":
            # Paper uses all CLIP layers.
            return list(range(depth))
        # For DINO and optional extra backbones, keep only deeper layers to
        # preserve the "semantic enhancement" role instead of flattening all layers.
        keep = min(6, depth)
        return list(range(depth - keep, depth))

    def _merge_model_layers(self, model_name: str, token_layers: List[torch.Tensor]) -> torch.Tensor:
        indices = self.layer_indices[model_name]
        modules = self.lln_modules[model_name]
        if len(indices) != len(modules):
            raise RuntimeError(f"Layer config mismatch for {model_name}.")

        transformed_layers = []
        for module, idx in zip(modules, indices):
            idx = max(0, min(idx, len(token_layers) - 1))
            transformed_layers.append(module(token_layers[idx]))

        stacked = torch.stack(transformed_layers, dim=0)  # [L, B, N, D]
        weights = torch.softmax(self.layer_scale_logits[model_name], dim=0).view(-1, 1, 1, 1)
        merged = (stacked * weights).sum(dim=0)
        return self.aligners[model_name](merged)

    def _merge_cached_layers(self, model_name: str, cached_layers: torch.Tensor) -> torch.Tensor:
        modules = self.lln_modules[model_name]
        if cached_layers.ndim != 4:
            raise ValueError(
                f"Expected cached COMM layers for {model_name} to have shape [B, L, N, D], "
                f"got {tuple(cached_layers.shape)}."
            )
        if cached_layers.size(1) != len(modules):
            raise RuntimeError(
                f"Cached layer count mismatch for {model_name}: "
                f"{cached_layers.size(1)} vs expected {len(modules)}."
            )

        transformed_layers = []
        for idx, module in enumerate(modules):
            transformed_layers.append(module(cached_layers[:, idx]))

        stacked = torch.stack(transformed_layers, dim=0)
        weights = torch.softmax(self.layer_scale_logits[model_name], dim=0).view(-1, 1, 1, 1)
        merged = (stacked * weights).sum(dim=0)
        return self.aligners[model_name](merged)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        branch_tokens: Dict[str, torch.Tensor] = {}
        target_tokens: Optional[int] = None

        for name in self.model_types:
            token_layers = self.extractors[name].extract_hidden_tokens(pixel_values)
            merged = self._merge_model_layers(name, token_layers)
            branch_tokens[name] = merged
            target_tokens = merged.size(1) if target_tokens is None else min(target_tokens, merged.size(1))

        assert target_tokens is not None
        fused = _resize_tokens(branch_tokens[self.anchor_model], target_tokens)
        for name in self.model_types:
            if name == self.anchor_model:
                continue
            aligned = _resize_tokens(branch_tokens[name], target_tokens)
            gate = torch.sigmoid(self.branch_gate_logits[name]).view(1, 1, 1)
            fused = fused + gate * aligned

        fused = self.final_proj(self.final_norm(fused))
        return fused.mean(dim=1)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache only the hidden token layers consumed by COMM."""
        cached_inputs: Dict[str, torch.Tensor] = {}
        for name in self.model_types:
            token_layers = self.extractors[name].extract_hidden_tokens(pixel_values)
            selected_layers = []
            for idx in self.layer_indices[name]:
                safe_idx = max(0, min(idx, len(token_layers) - 1))
                selected_layers.append(token_layers[safe_idx])
            cached_inputs[f"layers_{name}"] = torch.stack(selected_layers, dim=1)
        return cached_inputs

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse cached hidden token stacks without rerunning backbones."""
        branch_tokens: Dict[str, torch.Tensor] = {}
        target_tokens: Optional[int] = None

        for name in self.model_types:
            merged = self._merge_cached_layers(name, cached_inputs[f"layers_{name}"])
            branch_tokens[name] = merged
            target_tokens = merged.size(1) if target_tokens is None else min(target_tokens, merged.size(1))

        assert target_tokens is not None
        fused = _resize_tokens(branch_tokens[self.anchor_model], target_tokens)
        for name in self.model_types:
            if name == self.anchor_model:
                continue
            aligned = _resize_tokens(branch_tokens[name], target_tokens)
            gate = torch.sigmoid(self.branch_gate_logits[name]).view(1, 1, 1)
            fused = fused + gate * aligned
        fused = self.final_proj(self.final_norm(fused))
        return fused.mean(dim=1)

    def release_backbones(self):
        """Drop frozen backbones once token caches are available."""
        self.extractors = None


class MultiHeadPoolingAttention(nn.Module):
    """Multi-head pooling attention used in MMViT self/cross blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        q_stride: int = 1,
        kv_stride: int = 1,
        pool_kernel: int = 3,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = _valid_num_heads(dim, num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, out_dim)

        padding = pool_kernel // 2
        self.q_pool = nn.Conv2d(
            dim, dim, kernel_size=pool_kernel, stride=q_stride, padding=padding, groups=dim, bias=False
        )
        self.k_pool = nn.Conv2d(
            dim, dim, kernel_size=pool_kernel, stride=kv_stride, padding=padding, groups=dim, bias=False
        )
        self.v_pool = nn.Conv2d(
            dim, dim, kernel_size=pool_kernel, stride=kv_stride, padding=padding, groups=dim, bias=False
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def forward(self, x: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        q = _apply_depthwise_pool(self.q_proj(x), has_cls, self.q_pool)
        k = _apply_depthwise_pool(self.k_proj(x), has_cls, self.k_pool)
        v = _apply_depthwise_pool(self.v_proj(x), has_cls, self.v_pool)

        qh = self._reshape_heads(q)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)

        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, vh)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), self.dim)
        return self.out_proj(out)


class TokenDownsample(nn.Module):
    """Residual path downsample for scaled self-attention blocks."""

    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        self.pool = None
        if stride > 1:
            self.pool = nn.Conv2d(
                in_dim, in_dim, kernel_size=3, stride=stride, padding=1, groups=in_dim, bias=False
            )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        if self.pool is not None:
            x = _apply_depthwise_pool(x, has_cls, self.pool)
        return self.proj(x)


class MMViTSelfBlock(nn.Module):
    """Self-attention block; scaled block uses stride-2 pooling and channel doubling."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scaled: bool = False,
    ):
        super().__init__()
        stride = 2 if scaled else 1
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadPoolingAttention(
            dim=dim,
            num_heads=num_heads,
            q_stride=stride,
            kv_stride=1,
            pool_kernel=3,
            out_dim=out_dim,
        )
        self.skip = TokenDownsample(in_dim=dim, out_dim=out_dim, stride=stride)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = TokenFeedForward(out_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, has_cls: bool = False) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), has_cls=has_cls)
        x = self.skip(x, has_cls=has_cls) + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MMViTCrossBlock(nn.Module):
    """Cross-attention block fusing multiple views at the same scale stage."""

    def __init__(self, num_views: int, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_views = num_views
        self.num_heads = _valid_num_heads(dim, num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.pre_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_views)])
        self.q_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_views)])
        self.k_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_views)])
        self.v_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_views)])

        self.q_pool = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
            for _ in range(num_views)
        ])
        self.k_pool = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
            for _ in range(num_views)
        ])
        self.v_pool = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
            for _ in range(num_views)
        ])

        self.out_proj = nn.Linear(dim, dim)
        self.post_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_views)])
        self.mlps = nn.ModuleList([TokenFeedForward(dim, mlp_ratio) for _ in range(num_views)])

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x = x.view(bsz, seq_len, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        return x

    def forward(self, views: List[torch.Tensor], has_cls_flags: List[bool]) -> List[torch.Tensor]:
        q_list, k_list, v_list = [], [], []
        lengths = []

        for idx in range(self.num_views):
            x = self.pre_norm[idx](views[idx])
            q = _apply_depthwise_pool(self.q_proj[idx](x), has_cls_flags[idx], self.q_pool[idx])
            k = _apply_depthwise_pool(self.k_proj[idx](x), has_cls_flags[idx], self.k_pool[idx])
            v = _apply_depthwise_pool(self.v_proj[idx](x), has_cls_flags[idx], self.v_pool[idx])
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
            lengths.append(q.size(1))

        q_cat = torch.cat(q_list, dim=1)
        k_cat = torch.cat(k_list, dim=1)
        v_cat = torch.cat(v_list, dim=1)

        qh = self._reshape_heads(q_cat)
        kh = self._reshape_heads(k_cat)
        vh = self._reshape_heads(v_cat)

        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)
        out = out.permute(0, 2, 1, 3).reshape(out.size(0), out.size(2), q_cat.size(-1))
        out = self.out_proj(out)

        out_views = list(torch.split(out, lengths, dim=1))
        fused_views: List[torch.Tensor] = []
        for idx in range(self.num_views):
            y = views[idx] + out_views[idx]
            y = y + self.mlps[idx](self.post_norm[idx](y))
            fused_views.append(y)
        return fused_views


class MMViTStage(nn.Module):
    """One MMViT scale stage."""

    def __init__(
        self,
        num_views: int,
        dim: int,
        next_dim: Optional[int],
        n_self_blocks: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        with_cross: bool = True,
        with_scale: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.self_blocks = nn.ModuleList()
        for _ in range(n_self_blocks):
            self.self_blocks.append(nn.ModuleList([
                MMViTSelfBlock(dim=dim, out_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, scaled=False)
                for _ in range(num_views)
            ]))

        self.cross_block = MMViTCrossBlock(num_views, dim, num_heads, mlp_ratio) if with_cross else None
        self.scale_blocks = None
        if with_scale:
            if next_dim is None:
                raise ValueError("next_dim is required when with_scale=True")
            self.scale_blocks = nn.ModuleList([
                MMViTSelfBlock(dim=dim, out_dim=next_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, scaled=True)
                for _ in range(num_views)
            ])

    def forward(self, views: List[torch.Tensor], has_cls_flags: List[bool]) -> List[torch.Tensor]:
        for block_group in self.self_blocks:
            views = [block_group[i](views[i], has_cls_flags[i]) for i in range(self.num_views)]

        if self.cross_block is not None:
            views = self.cross_block(views, has_cls_flags)

        if self.scale_blocks is not None:
            views = [self.scale_blocks[i](views[i], has_cls_flags[i]) for i in range(self.num_views)]

        return views


class MMViTStrictFusionExtractor(nn.Module):
    """MMViT-inspired multiscale multiview token fusion for classification."""

    def __init__(
        self,
        model_types: Sequence[str],
        base_dim: int = 96,
        mlp_ratio: float = 4.0,
        num_heads: int = 8,
        max_position_tokens: int = 256,
        output_dim: Optional[int] = None,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("MMViT fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })
        self.num_views = len(self.model_types)

        self.input_proj = nn.ModuleDict({
            name: nn.Linear(self.extractors[name].token_dim, base_dim)
            for name in self.model_types
        })
        self.view_embed = nn.Parameter(torch.zeros(1, self.num_views, base_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, base_dim))
        self.pos_embed = nn.ParameterList([
            nn.Parameter(
                torch.zeros(1, max_position_tokens + (1 if idx == 0 else 0), base_dim)
            )
            for idx in range(self.num_views)
        ])

        stage_dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        stage_heads = [_valid_num_heads(dim, num_heads) for dim in stage_dims]
        stage_self_counts = [0, 0, 9, 1]  # Paper: total 16 blocks with stage layout [0,0,9,1].

        self.stages = nn.ModuleList([
            MMViTStage(
                num_views=self.num_views,
                dim=stage_dims[0],
                next_dim=stage_dims[1],
                n_self_blocks=stage_self_counts[0],
                num_heads=stage_heads[0],
                mlp_ratio=mlp_ratio,
                with_cross=True,
                with_scale=True,
            ),
            MMViTStage(
                num_views=self.num_views,
                dim=stage_dims[1],
                next_dim=stage_dims[2],
                n_self_blocks=stage_self_counts[1],
                num_heads=stage_heads[1],
                mlp_ratio=mlp_ratio,
                with_cross=True,
                with_scale=True,
            ),
            MMViTStage(
                num_views=self.num_views,
                dim=stage_dims[2],
                next_dim=stage_dims[3],
                n_self_blocks=stage_self_counts[2],
                num_heads=stage_heads[2],
                mlp_ratio=mlp_ratio,
                with_cross=True,
                with_scale=True,
            ),
            MMViTStage(
                num_views=self.num_views,
                dim=stage_dims[3],
                next_dim=None,
                n_self_blocks=stage_self_counts[3],
                num_heads=stage_heads[3],
                mlp_ratio=mlp_ratio,
                with_cross=False,
                with_scale=False,
            ),
        ])

        self.final_norm = nn.LayerNorm(stage_dims[-1])
        target_dim = stage_dims[-1] if output_dim is None else output_dim
        self.final_proj = nn.Linear(stage_dims[-1], target_dim)
        self.feature_dim = target_dim
        self.trainable = True
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        for pos_embed in self.pos_embed:
            nn.init.trunc_normal_(pos_embed, std=0.02)

    def _build_view_tokens(self, pixel_values: torch.Tensor) -> Tuple[List[torch.Tensor], List[bool]]:
        patch_tokens = []
        for name in self.model_types:
            tokens = self.extractors[name].extract_last_tokens(pixel_values)["patches"]
            patch_tokens.append(tokens)

        high_res_tokens = patch_tokens[0].size(1)
        high_res_side = _infer_square_size(high_res_tokens)
        views: List[torch.Tensor] = []
        has_cls_flags: List[bool] = []

        for idx, name in enumerate(self.model_types):
            if high_res_side is not None:
                target_side = max(2, round(high_res_side / (2 ** idx)))
                target_tokens = target_side * target_side
            else:
                target_tokens = max(4, high_res_tokens // (4 ** idx))
            tokens = _resize_tokens(patch_tokens[idx], target_tokens)
            tokens = self.input_proj[name](tokens)
            tokens = tokens + self.view_embed[:, idx:idx + 1]

            if idx == 0:
                cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
                tokens = torch.cat([cls_token, tokens], dim=1)
                has_cls_flags.append(True)
            else:
                has_cls_flags.append(False)

            tokens = _add_positional_embedding(tokens, self.pos_embed[idx])
            views.append(tokens)

        return views, has_cls_flags

    def _build_view_tokens_from_cache(
        self,
        cached_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[bool]]:
        patch_tokens = [cached_inputs[f"patches_{name}"] for name in self.model_types]
        high_res_tokens = patch_tokens[0].size(1)
        high_res_side = _infer_square_size(high_res_tokens)
        views: List[torch.Tensor] = []
        has_cls_flags: List[bool] = []

        for idx, name in enumerate(self.model_types):
            if high_res_side is not None:
                target_side = max(2, round(high_res_side / (2 ** idx)))
                target_tokens = target_side * target_side
            else:
                target_tokens = max(4, high_res_tokens // (4 ** idx))
            tokens = _resize_tokens(patch_tokens[idx], target_tokens)
            tokens = self.input_proj[name](tokens)
            tokens = tokens + self.view_embed[:, idx:idx + 1]

            if idx == 0:
                cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
                tokens = torch.cat([cls_token, tokens], dim=1)
                has_cls_flags.append(True)
            else:
                has_cls_flags.append(False)

            tokens = _add_positional_embedding(tokens, self.pos_embed[idx])
            views.append(tokens)

        return views, has_cls_flags

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        views, has_cls_flags = self._build_view_tokens(pixel_values)
        for stage in self.stages:
            views = stage(views, has_cls_flags)

        # Paper uses CLS from the first view for classification.
        cls_feature = views[0][:, 0]
        return self.final_proj(self.final_norm(cls_feature))

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache last-layer patch tokens for offline MMViT training."""
        return {
            f"patches_{name}": self.extractors[name].extract_last_tokens(pixel_values)["patches"]
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run MMViT fusion on cached patch tokens."""
        views, has_cls_flags = self._build_view_tokens_from_cache(cached_inputs)
        for stage in self.stages:
            views = stage(views, has_cls_flags)
        cls_feature = views[0][:, 0]
        return self.final_proj(self.final_norm(cls_feature))

    def release_backbones(self):
        """Drop frozen backbones once token caches are available."""
        self.extractors = None


class MultiModelTopKRouterExtractor(nn.Module):
    """Top-K Sparse Router inspired by Switch Transformer / V-MoE.

    A router MLP takes the mean of all projected features and outputs N logits.
    Only the top-k models (by logit value) are used for each sample, with
    softmax-normalized weights. A straight-through estimator keeps gradients
    flowing through the discrete selection. A load-balancing auxiliary loss
    discourages routing collapse.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        hidden_dim: int = 128,
        router_k: int = 2,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        n = len(self.model_types)
        if n < 2:
            raise ValueError("Fusion requires at least two models.")
        self.router_k = min(router_k, n)

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim),
            )
            for name in self.model_types
        })

        # Router: input = mean of projected features, output = per-model logits
        self.router = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n),
        )

        self.feature_dim = proj_dim
        self.proj_dim = proj_dim
        self.trainable = True
        self.aux_loss: Optional[torch.Tensor] = None

    def _route(self, projected_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute routed fusion from projected features. Sets self.aux_loss."""
        n = len(self.model_types)
        stacked = torch.stack(projected_features, dim=1)  # [B, N, D]
        router_input = stacked.mean(dim=1)  # [B, D]
        logits = self.router(router_input)  # [B, N]

        # Soft weights over all models (for gradient flow)
        soft_weights = torch.softmax(logits, dim=-1)  # [B, N]

        # Hard top-k mask (no gradient through selection)
        _, topk_indices = logits.topk(self.router_k, dim=-1)  # [B, k]
        hard_mask = torch.zeros_like(logits).scatter_(1, topk_indices, 1.0)  # [B, N]

        # Straight-through: forward uses hard mask, backward uses soft weights
        weights = hard_mask * soft_weights  # zero out non-selected
        # Re-normalize so selected weights sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        # Straight-through estimator: gradient flows through soft_weights
        weights = soft_weights + (weights - soft_weights).detach()

        # Weighted combination
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, D]

        # Load-balancing loss: L_bal = N * sum(f_i * P_i)
        # f_i = fraction of samples routed to expert i
        # P_i = mean routing probability for expert i
        if self.training:
            f = hard_mask.mean(dim=0)  # [N] fraction of samples selecting each model
            P = soft_weights.mean(dim=0)  # [N] mean probability per model
            self.aux_loss = n * (f * P).sum()
        else:
            self.aux_loss = None

        return fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_features.append(self.projections[name](feat))
        return self._route(projected_features)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected_features = [
            self.projections[name](cached_inputs[f"feat_{name}"])
            for name in self.model_types
        ]
        return self._route(projected_features)

    def release_backbones(self):
        self.extractors = None


class MultiModelMoERouterExtractor(nn.Module):
    """Soft MoE Router with load balancing, inspired by GShard / ST-MoE.

    All models contribute (soft routing) but weights are adaptive per sample.
    Includes three auxiliary losses:
      - Load-balancing loss to encourage uniform expert utilization
      - Entropy regularization to encourage routing diversity
      - Router z-loss to prevent logits from growing too large
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        hidden_dim: int = 128,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        n = len(self.model_types)
        if n < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim),
            )
            for name in self.model_types
        })

        # Router network
        self.router = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n),
        )

        self.feature_dim = proj_dim
        self.proj_dim = proj_dim
        self.trainable = True
        self.aux_loss: Optional[torch.Tensor] = None

    def _route(self, projected_features: List[torch.Tensor]) -> torch.Tensor:
        """Soft routing with auxiliary losses."""
        n = len(self.model_types)
        stacked = torch.stack(projected_features, dim=1)  # [B, N, D]
        router_input = stacked.mean(dim=1)  # [B, D]
        logits = self.router(router_input)  # [B, N]
        weights = torch.softmax(logits, dim=-1)  # [B, N]

        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, D]

        if self.training:
            # Load-balancing: L_bal = N * sum(f_i * P_i)
            # For soft routing, f_i = mean weight assigned to expert i
            f = weights.mean(dim=0)  # [N]
            P = weights.mean(dim=0)  # same for soft routing
            balance_loss = n * (f * P).sum()

            # Entropy regularization: encourage high entropy (uniform) routing
            # H = -sum(p * log(p)), maximize entropy => minimize -H
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            # Target: max entropy = log(N). Loss = (log(N) - H) / log(N)
            max_entropy = math.log(n)
            entropy_loss = (max_entropy - entropy) / max_entropy

            # Router z-loss: prevent logits from growing too large
            z_loss = (logits ** 2).mean()

            self.aux_loss = balance_loss + 0.1 * entropy_loss + 0.001 * z_loss
        else:
            self.aux_loss = None

        return fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_features.append(self.projections[name](feat))
        return self._route(projected_features)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected_features = [
            self.projections[name](cached_inputs[f"feat_{name}"])
            for name in self.model_types
        ]
        return self._route(projected_features)

    def release_backbones(self):
        self.extractors = None


class MultiModelAttentionRouterExtractor(nn.Module):
    """Self-Attention Based Router inspired by FusionFM attention-based gating.

    Each model's projected feature is treated as a token. Multi-head
    self-attention lets the model features interact, and attention outputs
    are pooled to produce per-model routing weights.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        num_heads: int = 4,
        model_dir: str = "./models",
    ):
        super().__init__()
        self.model_types = list(model_types)
        n = len(self.model_types)
        if n < 2:
            raise ValueError("Fusion requires at least two models.")

        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim),
            )
            for name in self.model_types
        })

        # Learnable position embeddings for each model slot
        self.model_pos_embed = nn.Parameter(torch.randn(1, n, proj_dim) * 0.02)

        # Multi-head self-attention
        num_heads = _valid_num_heads(proj_dim, num_heads)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(proj_dim)

        # Gate head: project each attended token to a scalar weight
        self.gate_head = nn.Linear(proj_dim, 1)

        self.feature_dim = proj_dim
        self.proj_dim = proj_dim
        self.trainable = True
        self.aux_loss: Optional[torch.Tensor] = None

    def _route(self, projected_features: List[torch.Tensor]) -> torch.Tensor:
        """Attention-based routing."""
        stacked = torch.stack(projected_features, dim=1)  # [B, N, D]

        # Add positional embeddings
        tokens = stacked + self.model_pos_embed

        # Self-attention with residual
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)  # [B, N, D]

        # Compute routing weights from attended representations
        gate_logits = self.gate_head(tokens).squeeze(-1)  # [B, N]
        weights = torch.softmax(gate_logits, dim=-1)  # [B, N]

        # Weighted combination of original projected features (not attended)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, D]

        self.aux_loss = None
        return fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_features.append(self.projections[name](feat))
        return self._route(projected_features)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected_features = [
            self.projections[name](cached_inputs[f"feat_{name}"])
            for name in self.model_types
        ]
        return self._route(projected_features)

    def release_backbones(self):
        self.extractors = None


def get_extractor(
    model_type: str,
    fusion_method: str = "concat",
    fusion_models: Optional[Sequence[str]] = None,
    fusion_kwargs: Optional[Dict] = None,
    model_dir: str = "./models",
) -> nn.Module:
    """Factory function to create extractors.

    Args:
        model_type: Single model type (`mae`, `clip`, `dino`) or `fusion`.
        fusion_method: One of `concat`, `proj_concat`, `weighted_sum`, `gated`,
                      `difference_concat`, `hadamard_concat`, `comm`, `mmvit` when model_type is `fusion`.
        fusion_models: List of model types for fusion, e.g. ["clip", "dino"].
        fusion_kwargs: Extra kwargs for paper-inspired classifier fusion implementations.
    """
    if model_type != "fusion":
        return FeatureExtractor(model_type, normalize_input=False, model_dir=model_dir)

    model_types = list(fusion_models) if fusion_models is not None else ["mae", "clip", "dino"]
    fusion_method = fusion_method.lower()
    fusion_kwargs = {} if fusion_kwargs is None else dict(fusion_kwargs)

    # Simple baselines
    if fusion_method == "concat":
        return MultiModelConcatExtractor(
            model_types,
            output_dim=fusion_kwargs.get("fusion_output_dim"),
            model_dir=model_dir,
        )
    if fusion_method == "proj_concat":
        return MultiModelProjectedConcatExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 256),
            model_dir=model_dir,
        )
    if fusion_method == "weighted_sum":
        return MultiModelWeightedSumExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 512),
            model_dir=model_dir,
        )
    if fusion_method == "gated":
        return MultiModelGatedFusionExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 512),
            hidden_dim=fusion_kwargs.get("hidden_dim", 128),
            model_dir=model_dir,
        )
    if fusion_method == "difference_concat":
        return MultiModelDifferenceAwareExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 256),
            model_dir=model_dir,
        )
    if fusion_method == "hadamard_concat":
        return MultiModelHadamardExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 256),
            model_dir=model_dir,
        )
    if fusion_method == "late_fusion":
        return MultiModelLateFusionExtractor(
            model_types,
            num_classes=fusion_kwargs.get("num_classes", 100),
            model_dir=model_dir,
        )

    # Paper-inspired methods
    if fusion_method == "comm":
        return COMMStrictFusionExtractor(
            model_types=model_types,
            dino_mlp_blocks=fusion_kwargs.get("comm_dino_mlp_blocks", 2),
            dino_mlp_ratio=fusion_kwargs.get("comm_dino_mlp_ratio", 8.0),
            output_dim=fusion_kwargs.get("fusion_output_dim"),
            model_dir=model_dir,
        )
    if fusion_method == "mmvit":
        return MMViTStrictFusionExtractor(
            model_types=model_types,
            base_dim=fusion_kwargs.get("mmvit_base_dim", 96),
            mlp_ratio=fusion_kwargs.get("mmvit_mlp_ratio", 4.0),
            num_heads=fusion_kwargs.get("mmvit_num_heads", 8),
            max_position_tokens=fusion_kwargs.get("mmvit_max_position_tokens", 256),
            output_dim=fusion_kwargs.get("fusion_output_dim"),
            model_dir=model_dir,
        )

    # Dynamic routing / MoE methods
    if fusion_method == "topk_router":
        return MultiModelTopKRouterExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 512),
            hidden_dim=fusion_kwargs.get("hidden_dim", 128),
            router_k=fusion_kwargs.get("router_k", 2),
            model_dir=model_dir,
        )
    if fusion_method == "moe_router":
        return MultiModelMoERouterExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 512),
            hidden_dim=fusion_kwargs.get("hidden_dim", 128),
            model_dir=model_dir,
        )
    if fusion_method == "attention_router":
        return MultiModelAttentionRouterExtractor(
            model_types,
            proj_dim=fusion_kwargs.get("proj_dim", 512),
            num_heads=fusion_kwargs.get("attention_router_heads", 4),
            model_dir=model_dir,
        )

    raise ValueError(
        f"Unsupported fusion method: {fusion_method}. "
        f"Choose from ['concat', 'proj_concat', 'weighted_sum', 'gated', "
        f"'difference_concat', 'hadamard_concat', 'late_fusion', 'comm', 'mmvit', "
        f"'topk_router', 'moe_router', 'attention_router']."
    )

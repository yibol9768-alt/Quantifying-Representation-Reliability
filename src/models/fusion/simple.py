"""Simple fusion baselines: concat, projected concat, weighted sum, gated."""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..extractor import FeatureExtractor


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
            feat = FeatureExtractor._ensure_matrix(cached_inputs[f"feat_{name}"])
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
            feat = FeatureExtractor._ensure_matrix(cached_inputs[f"feat_{name}"])
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
            feat = FeatureExtractor._ensure_matrix(cached_inputs[f"feat_{name}"])
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
            feat = FeatureExtractor._ensure_matrix(cached_inputs[f"feat_{name}"])
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

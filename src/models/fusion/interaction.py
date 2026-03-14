"""Interaction-based fusion: difference concat, hadamard, bilinear."""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..extractor import FeatureExtractor


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


class MultiModelBilinearExtractor(nn.Module):
    """Baseline G: Concat with pairwise Bilinear Pooling (Outer Product).

    Formula: z = [z_c; z_d; z_m; sqrt_norm(z_c ⊗ z_d); sqrt_norm(z_c ⊗ z_m); ...]

    Based on Bilinear CNN (Lin et al., ICCV 2015). The outer product captures
    all cross-dimensional interactions between two feature vectors, which is
    strictly more expressive than Hadamard (element-wise) products.
    Signed square-root and L2 normalization suppress burstiness.

    Note: proj_dim defaults to 64 because the outer product squares the
    dimension (64 -> 4096 per pair). With 512 it would be 262,144 per pair.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 64,
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

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim),
            )
            for name in self.model_types
        })

        num_models = len(self.model_types)
        num_pairs = num_models * (num_models - 1) // 2
        self.feature_dim = (proj_dim * num_models) + (num_pairs * (proj_dim ** 2))
        self.proj_dim = proj_dim
        self.trainable = True

    def _compute_bilinear_pairs(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pairwise outer products with signed sqrt and L2 normalization."""
        bilinear_features = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                outer = torch.bmm(
                    features[i].unsqueeze(2), features[j].unsqueeze(1)
                ).flatten(1)
                signed_sqrt = torch.sign(outer) * torch.sqrt(torch.abs(outer) + 1e-8)
                bilinear_features.append(F.normalize(signed_sqrt, p=2, dim=-1))
        return bilinear_features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected_features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            projected_features.append(self.projections[name](feat))
        bilinear_pairs = self._compute_bilinear_pairs(projected_features)
        return torch.cat(projected_features + bilinear_pairs, dim=-1)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected_features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            projected_features.append(self.projections[name](feat))
        bilinear_pairs = self._compute_bilinear_pairs(projected_features)
        return torch.cat(projected_features + bilinear_pairs, dim=-1)

    def release_backbones(self):
        self.extractors = None

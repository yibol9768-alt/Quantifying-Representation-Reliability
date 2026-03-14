"""Attention-based fusion: FiLM, context gating, LMF, SE-fusion, late fusion."""

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..extractor import FeatureExtractor


class MultiModelFiLMExtractor(nn.Module):
    """FiLM Fusion: Feature-wise Linear Modulation (Perez et al., AAAI 2018).

    One model's features generate affine parameters (scale gamma + shift beta)
    that modulate the other models' features. Strictly more expressive than
    gating (which only scales by 0-1), because FiLM also shifts.

    Formula: z_i = gamma_i * proj(x_i) + beta_i
    where (gamma, beta) are predicted from the mean of all other models.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
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
        # FiLM generators: each model gets gamma & beta from the mean of others
        self.film_generators = nn.ModuleDict({
            name: nn.Linear(proj_dim, proj_dim * 2)
            for name in self.model_types
        })
        self.feature_dim = proj_dim * n
        self.proj_dim = proj_dim
        self.trainable = True

    def _fuse(self, projected: Dict[str, torch.Tensor]) -> torch.Tensor:
        all_feats = torch.stack([projected[n] for n in self.model_types], dim=1)  # [B,N,D]
        modulated = []
        for i, name in enumerate(self.model_types):
            # Context = mean of all OTHER models
            mask = torch.ones(len(self.model_types), device=all_feats.device)
            mask[i] = 0
            context = (all_feats * mask.view(1, -1, 1)).sum(1) / mask.sum()
            gamma_beta = self.film_generators[name](context)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            modulated.append(gamma * projected[name] + beta)
        return torch.cat(modulated, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected = {n: self.projections[n](self.extractors[n](pixel_values)) for n in self.model_types}
        return self._fuse(projected)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f"feat_{n}": self.extractors[n](pixel_values) for n in self.model_types}

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = {n: self.projections[n](cached_inputs[f"feat_{n}"]) for n in self.model_types}
        return self._fuse(projected)

    def release_backbones(self):
        self.extractors = None


class MultiModelContextGatingExtractor(nn.Module):
    """Context Gating: per-dimension self-gating on concatenated features.

    From Miech et al., 2017 (YouTube-8M challenge winner).
    Unlike Gated Fusion (model-level scalar gates), this applies a
    per-dimension gate on the full concatenated feature vector, capturing
    cross-model dimensional interactions.

    Formula: z_out = sigmoid(W * z + b) * z,  where z = concat(proj(x_i))
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
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.extractors[name].feature_dim),
                nn.Linear(self.extractors[name].feature_dim, proj_dim),
            )
            for name in self.model_types
        })
        concat_dim = proj_dim * len(self.model_types)
        self.context_gate = nn.Linear(concat_dim, concat_dim)
        self.feature_dim = concat_dim
        self.proj_dim = proj_dim
        self.trainable = True

    def _fuse(self, projected_features: List[torch.Tensor]) -> torch.Tensor:
        z = torch.cat(projected_features, dim=-1)
        gate = torch.sigmoid(self.context_gate(z))
        return gate * z

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = [self.projections[n](self.extractors[n](pixel_values)) for n in self.model_types]
        return self._fuse(feats)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f"feat_{n}": self.extractors[n](pixel_values) for n in self.model_types}

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [self.projections[n](cached_inputs[f"feat_{n}"]) for n in self.model_types]
        return self._fuse(feats)

    def release_backbones(self):
        self.extractors = None


class MultiModelLMFExtractor(nn.Module):
    """Low-rank Multimodal Fusion (Liu et al., ACL 2018).

    Captures higher-order (not just pairwise) interactions across all models
    via low-rank tensor decomposition. Each rank component projects all models
    into a shared subspace, takes element-wise product, then sums across ranks.

    Formula: h = sum_r( (W1_r * z_1) * (W2_r * z_2) * ... * (WM_r * z_M) )
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        rank: int = 16,
        output_dim: int = 512,
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
        # Low-rank factors: each model gets a (proj_dim+1) -> (rank * output_dim) mapping
        # +1 for bias trick (append 1 to feature vector)
        self.lmf_factors = nn.ModuleDict({
            name: nn.Linear(proj_dim + 1, rank * output_dim, bias=False)
            for name in self.model_types
        })
        self.rank = rank
        self.output_dim = output_dim
        self.feature_dim = output_dim
        self.proj_dim = proj_dim
        self.trainable = True

    def _fuse(self, projected: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(projected.values())).size(0)
        device = next(iter(projected.values())).device
        fusion_output = torch.ones(batch_size, self.rank * self.output_dim, device=device)
        for name in self.model_types:
            # Append 1 for bias trick
            feat_with_bias = torch.cat([projected[name], torch.ones(batch_size, 1, device=device)], dim=-1)
            factor = self.lmf_factors[name](feat_with_bias)  # [B, rank*output_dim]
            fusion_output = fusion_output * factor
        # Reshape to [B, rank, output_dim] and sum over rank
        fusion_output = fusion_output.view(batch_size, self.rank, self.output_dim).sum(dim=1)
        return fusion_output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        projected = {n: self.projections[n](self.extractors[n](pixel_values)) for n in self.model_types}
        return self._fuse(projected)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f"feat_{n}": self.extractors[n](pixel_values) for n in self.model_types}

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = {n: self.projections[n](cached_inputs[f"feat_{n}"]) for n in self.model_types}
        return self._fuse(projected)

    def release_backbones(self):
        self.extractors = None


class MultiModelSEFusionExtractor(nn.Module):
    """SE-Fusion: Squeeze-and-Excitation channel attention for fusion.

    From Hu et al., CVPR 2018. Treats M model features as M channels,
    applies SE-style bottleneck attention to compute content-dependent
    per-model importance weights.

    Unlike Gated (MLP on full concat), this uses a bottleneck on global
    statistics — more parameter-efficient and specifically designed for
    channel recalibration.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        proj_dim: int = 512,
        reduction: int = 4,
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
        # SE bottleneck: squeeze (mean pool) -> excite (bottleneck MLP) -> scale
        bottleneck = max(1, n // reduction)
        self.se_block = nn.Sequential(
            nn.Linear(n, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n),
            nn.Sigmoid(),
        )
        self.feature_dim = proj_dim * n
        self.proj_dim = proj_dim
        self.trainable = True

    def _fuse(self, projected_features: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(projected_features, dim=1)  # [B, N, D]
        # Squeeze: mean over feature dim -> [B, N]
        squeezed = stacked.mean(dim=-1)
        # Excite: bottleneck MLP -> [B, N]
        scale = self.se_block(squeezed)
        # Scale each model's features
        scaled = stacked * scale.unsqueeze(-1)  # [B, N, D]
        return scaled.reshape(scaled.size(0), -1)  # [B, N*D]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = [self.projections[n](self.extractors[n](pixel_values)) for n in self.model_types]
        return self._fuse(feats)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f"feat_{n}": self.extractors[n](pixel_values) for n in self.model_types}

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [self.projections[n](cached_inputs[f"feat_{n}"]) for n in self.model_types]
        return self._fuse(feats)

    def release_backbones(self):
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

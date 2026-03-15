"""Redundancy-Aware Feature Bottleneck fusion methods.

Three bottleneck approaches applied after multi-model concatenation:

1. PCABottleneckExtractor: Non-parametric PCA compression (computed on train set)
2. LinearBottleneckExtractor: Learnable linear projection (4608 → d)
3. VIBBottleneckExtractor: Variational Information Bottleneck
   - Maximizes I(Z;Y) while minimizing I(X;Z)
   - Stochastic encoding: z = μ + σ·ε, ε ~ N(0,I)
   - KL regularization compresses redundant dimensions to prior

Theory:
    The IB objective L = CE(y, f(z)) + β·KL(q(z|x) || p(z)) naturally
    balances task-relevant information preservation (CE term) with
    redundancy compression (KL term). β controls the trade-off:
    - β=0: no compression, equivalent to standard concat
    - β→∞: maximum compression, z collapses to prior
    - Optimal β: removes redundant dimensions while preserving task signal

References:
    - Alemi et al., "Deep Variational Information Bottleneck", ICLR 2017
    - Tishby & Zaslavsky, "Deep Learning and the IB Principle", 2015
"""

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..extractor import FeatureExtractor


class _ConcatBase(nn.Module):
    """Shared base: extract and concatenate features from multiple models."""

    def __init__(self, model_types: Sequence[str], model_dir: str = "./models"):
        super().__init__()
        self.model_types = list(model_types)
        if len(self.model_types) < 2:
            raise ValueError("Fusion requires at least two models.")
        self.extractors = nn.ModuleDict({
            name: FeatureExtractor(name, normalize_input=True, model_dir=model_dir)
            for name in self.model_types
        })
        self.concat_dim = sum(
            self.extractors[name].feature_dim for name in self.model_types
        )

    def _concat_features(
        self, pixel_values: torch.Tensor, normalize: bool = True,
    ) -> torch.Tensor:
        features = []
        for name in self.model_types:
            feat = self.extractors[name](pixel_values)
            if normalize:
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            features.append(feat)
        return torch.cat(features, dim=-1)

    def _concat_from_cache(
        self, cached_inputs: Dict[str, torch.Tensor], normalize: bool = True,
    ) -> torch.Tensor:
        features = []
        for name in self.model_types:
            feat = cached_inputs[f"feat_{name}"]
            if normalize:
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            features.append(feat)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def extract_cache_batch(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            f"feat_{name}": self.extractors[name](pixel_values)
            for name in self.model_types
        }

    def release_backbones(self):
        self.extractors = None


class LinearBottleneckExtractor(_ConcatBase):
    """Concat + learnable linear bottleneck projection.

    4608-dim concat → LayerNorm → Linear → bottleneck_dim
    """

    def __init__(
        self,
        model_types: Sequence[str],
        bottleneck_dim: int = 512,
        model_dir: str = "./models",
    ):
        super().__init__(model_types, model_dir)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(self.concat_dim),
            nn.Linear(self.concat_dim, bottleneck_dim),
        )
        self.feature_dim = bottleneck_dim
        self.trainable = True
        self.aux_loss = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        concat_feat = self._concat_features(pixel_values)
        return self.bottleneck(concat_feat)

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat_feat = self._concat_from_cache(cached_inputs)
        return self.bottleneck(concat_feat)


class VIBBottleneckExtractor(_ConcatBase):
    """Concat + Variational Information Bottleneck.

    Encodes concatenated features into a stochastic bottleneck:
        z ~ N(μ(x), σ²(x))  during training
        z = μ(x)             during inference

    The KL divergence KL(q(z|x) || N(0,I)) acts as a compression
    regularizer, pushing unused dimensions toward the prior.

    The aux_loss field exposes the KL term so the training loop
    can weight it with a β coefficient.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        bottleneck_dim: int = 512,
        model_dir: str = "./models",
    ):
        super().__init__(model_types, model_dir)

        self.encoder_norm = nn.LayerNorm(self.concat_dim)
        self.fc_mu = nn.Linear(self.concat_dim, bottleneck_dim)
        self.fc_logvar = nn.Linear(self.concat_dim, bottleneck_dim)

        self.feature_dim = bottleneck_dim
        self.trainable = True
        self.aux_loss: Optional[torch.Tensor] = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode through VIB: returns z, sets self.aux_loss."""
        h = self.encoder_norm(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.training:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps

            # KL divergence: KL(N(μ,σ²) || N(0,I))
            # = 0.5 * Σ(μ² + σ² - 1 - log(σ²))
            kl = 0.5 * torch.mean(
                mu.pow(2) + logvar.exp() - 1.0 - logvar
            )
            self.aux_loss = kl
        else:
            z = mu
            self.aux_loss = None

        return z

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        concat_feat = self._concat_features(pixel_values)
        return self._encode(concat_feat)

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat_feat = self._concat_from_cache(cached_inputs)
        return self._encode(concat_feat)


class PCABottleneckExtractor(_ConcatBase):
    """Concat + PCA bottleneck (non-parametric).

    PCA components are fit on the training set, then applied to both
    train and test. Since PCA is non-parametric, only the classifier
    is trained.

    Call fit_pca() after caching features and before training.
    """

    def __init__(
        self,
        model_types: Sequence[str],
        bottleneck_dim: int = 512,
        model_dir: str = "./models",
    ):
        super().__init__(model_types, model_dir)
        self.bottleneck_dim = bottleneck_dim
        self.feature_dim = bottleneck_dim
        self.trainable = False  # PCA is not learnable
        self.aux_loss = None

        # Will be set by fit_pca()
        self.register_buffer("pca_mean", torch.zeros(self.concat_dim))
        self.register_buffer("pca_components", torch.zeros(bottleneck_dim, self.concat_dim))
        self._pca_fitted = False

    def fit_pca(self, features: torch.Tensor):
        """Fit PCA on training features.

        Args:
            features: [N, concat_dim] concatenated training features.
        """
        # Center
        mean = features.mean(dim=0)
        centered = features - mean

        # SVD for PCA
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        components = Vt[:self.bottleneck_dim]  # [k, concat_dim]

        self.pca_mean.copy_(mean)
        self.pca_components.copy_(components)
        self._pca_fitted = True

        # Print explained variance
        total_var = (S ** 2).sum()
        explained_var = (S[:self.bottleneck_dim] ** 2).sum()
        ratio = explained_var / total_var
        print(f"  PCA: {self.concat_dim} → {self.bottleneck_dim} dims, "
              f"explained variance: {ratio:.4f}")

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if not self._pca_fitted:
            raise RuntimeError("Call fit_pca() before forward pass.")
        centered = x - self.pca_mean
        return centered @ self.pca_components.T  # [B, bottleneck_dim]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        concat_feat = self._concat_features(pixel_values)
        return self._project(concat_feat)

    def forward_from_cache(self, cached_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat_feat = self._concat_from_cache(cached_inputs)
        return self._project(concat_feat)

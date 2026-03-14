"""Dynamic routing fusion: Top-K, MoE, and attention-based routers."""

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..extractor import FeatureExtractor, _valid_num_heads


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

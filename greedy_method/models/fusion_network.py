"""
models/fusion_network.py
========================
Multi-view weighted feature fusion (Section: 网络架构设计).

Architecture (for a subset of K active encoders):

  Input images x
      │
      ├─── encoder_1 (frozen) ──► f_1 ──► Linear_1 ──► z_1 ─┐
      ├─── encoder_2 (frozen) ──► f_2 ──► Linear_2 ──► z_2 ─┤
      │         ...                                           │ weighted sum
      └─── encoder_K (frozen) ──► f_K ──► Linear_K ──► z_K ─┘
                                                              │
                               α = Softmax([w_1,...,w_K])    │
                               Z_fused = Σ α_i · z_i   ──────►  Z_fused
                                                              │
                                                        classifier head
                                                        (see heads.py)

Trainable parameters (all lightweight):
  • Linear projection per encoder : output_dim_i → fusion_dim
  • Scalar fusion weight w_i per encoder
  • Downstream classifier head (external, passed in heads.py)

The fusion module is designed to accept **pre-extracted cached features**
(Dict[str, Tensor]) to avoid re-running large encoders during training.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Projection head (one per encoder)
# ---------------------------------------------------------------------------
class LinearProjection(nn.Module):
    """Maps an encoder's raw feature to the shared fusion_dim space."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Multi-view fusion module
# ---------------------------------------------------------------------------
class MultiViewFusion(nn.Module):
    """
    Adaptive weighted fusion of heterogeneous encoder features.

    Parameters
    ----------
    active_models : list of encoder names from MODEL_REGISTRY to fuse.
    fusion_dim    : common projection dimensionality (d in the paper).
    """

    def __init__(self, active_models: List[str], fusion_dim: int = 512) -> None:
        super().__init__()
        if not active_models:
            raise ValueError("active_models must not be empty.")

        self.active_models = list(active_models)
        self.fusion_dim    = fusion_dim

        # One projection per encoder
        self.projections = nn.ModuleDict({
            name: LinearProjection(MODEL_REGISTRY[name]["output_dim"], fusion_dim)
            for name in self.active_models
        })

        # Learnable scalar fusion weights (one per encoder, initialised to 0)
        # After Softmax these become uniform α = 1/K at init
        self.fusion_weights = nn.Parameter(
            torch.zeros(len(self.active_models))
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_models(self) -> int:
        return len(self.active_models)

    def get_alphas(self) -> torch.Tensor:
        """Return normalised fusion weights α_i = softmax(w_i)."""
        return F.softmax(self.fusion_weights, dim=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : dict mapping encoder name → raw feature tensor (B, output_dim_i).
                   Must contain entries for ALL active_models.

        Returns
        -------
        Z_fused : (B, fusion_dim)  weighted sum of projected features.
        """
        alphas = self.get_alphas()  # (K,)
        z_fused: Optional[torch.Tensor] = None

        for i, name in enumerate(self.active_models):
            f_i = features[name]                   # (B, output_dim_i)
            z_i = self.projections[name](f_i)      # (B, fusion_dim)
            weighted = alphas[i] * z_i             # broadcast scalar × vector

            z_fused = weighted if z_fused is None else z_fused + weighted

        return z_fused  # (B, fusion_dim)

    # ------------------------------------------------------------------
    # Convenience: add a new encoder without re-initialising existing ones
    # ------------------------------------------------------------------
    def expand(self, new_model_name: str) -> "MultiViewFusion":
        """
        Return a NEW MultiViewFusion that adds one encoder to the current set.
        Existing projection weights and fusion scalars are copied (not shared).
        """
        new_models = self.active_models + [new_model_name]
        new_fusion = MultiViewFusion(new_models, self.fusion_dim)

        # Copy existing projection weights
        for name in self.active_models:
            new_fusion.projections[name].load_state_dict(
                self.projections[name].state_dict()
            )

        # Copy existing fusion weight scalars (new one stays at 0)
        with torch.no_grad():
            old_k = len(self.active_models)
            new_fusion.fusion_weights[:old_k].copy_(self.fusion_weights)

        return new_fusion

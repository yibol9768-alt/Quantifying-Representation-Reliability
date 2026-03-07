"""
Token-level fusion models for MMViT and COMM.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_if_concatenated(
    features: Tuple[torch.Tensor, ...],
    feature_dims: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Keep backward compatibility for concat baseline:
    accept either split feature tensors or one concatenated tensor.
    """
    if (
        feature_dims is not None
        and len(features) == 1
        and features[0].dim() == 2
        and features[0].shape[-1] == sum(feature_dims)
    ):
        return torch.split(features[0], feature_dims, dim=-1)
    return features


def _pool_tokens_to_length(tokens: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adaptive token pooling while preserving CLS token.
    tokens: [B, T, C], where tokens[:, 0] is CLS.
    """
    if tokens.size(1) == target_length:
        return tokens
    if tokens.size(1) <= 1 or target_length <= 1:
        cls = tokens[:, :1]
        if target_length == 1:
            return cls
        pad = cls.repeat(1, target_length - 1, 1)
        return torch.cat([cls, pad], dim=1)

    cls = tokens[:, :1]
    patch = tokens[:, 1:]
    patch_target = max(1, target_length - 1)
    patch = patch.transpose(1, 2)  # [B, C, T-1]
    pooled = F.adaptive_avg_pool1d(patch, patch_target).transpose(1, 2)
    return torch.cat([cls, pooled], dim=1)


class _TokenFFNLayerScale(nn.Module):
    """Token-wise FFN with LayerScale residual."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, layerscale_init: float = 1e-4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(layerscale_init * torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.ffn(self.norm(x))


class _CrossViewTokenBlock(nn.Module):
    """
    Cross-view token communication block:
    each view attends to all views' tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        layerscale_init: float = 1e-4,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_ctx = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = _TokenFFNLayerScale(
            hidden_dim=hidden_dim,
            dropout=dropout,
            layerscale_init=layerscale_init,
        )
        self.gamma_attn = nn.Parameter(layerscale_init * torch.ones(hidden_dim))

    def forward(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        ctx = torch.cat(views, dim=1)
        ctx = self.norm_ctx(ctx)
        outs: List[torch.Tensor] = []
        for view in views:
            q = self.norm_q(view)
            attn_out, _ = self.attn(query=q, key=ctx, value=ctx)
            x = view + self.gamma_attn * attn_out
            x = self.ffn(x)
            outs.append(x)
        return outs


class MMViTTokenFusion(nn.Module):
    """
    Token-level MMViT-style fusion:
    per-view token projections + multi-scale cross-view attention.
    """

    def __init__(
        self,
        feature_shapes: List[Tuple[int, int]],
        view_layout: List[List[int]],
        num_classes: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        self.feature_shapes = feature_shapes
        self.view_layout = view_layout
        self.num_views = len(view_layout)
        if any(len(group) != 1 for group in view_layout):
            raise ValueError("MMViT expects exactly one token tensor per view")

        # One token source per view
        in_dims = [feature_shapes[g[0]][1] for g in view_layout]
        self.projections = nn.ModuleList([nn.Linear(d, hidden_dim) for d in in_dims])

        # Three-scale hierarchy
        self.scales = [1, 2, 4]
        self.blocks = nn.ModuleList(
            [
                _CrossViewTokenBlock(hidden_dim=hidden_dim, num_heads=num_heads)
                for _ in range(depth * len(self.scales))
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.num_views * len(self.scales)),
            nn.Linear(hidden_dim * self.num_views * len(self.scales), hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        self.depth = depth

    def _downsample_scale(self, tokens: torch.Tensor, scale: int) -> torch.Tensor:
        if scale <= 1:
            return tokens
        cls = tokens[:, :1]
        patch = tokens[:, 1:]
        target_patch = max(1, patch.size(1) // scale)
        patch = patch.transpose(1, 2)
        patch = F.adaptive_avg_pool1d(patch, target_patch).transpose(1, 2)
        return torch.cat([cls, patch], dim=1)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != len(self.feature_shapes):
            raise ValueError(f"Expected {len(self.feature_shapes)} inputs, got {len(features)}")

        view_tokens = [features[group[0]] for group in self.view_layout]  # [B, T, D]
        view_tokens = [proj(x) for proj, x in zip(self.projections, view_tokens)]

        stage_cls: List[torch.Tensor] = []
        block_idx = 0
        for scale in self.scales:
            stage_views = [self._downsample_scale(x, scale) for x in view_tokens]
            for _ in range(self.depth):
                stage_views = self.blocks[block_idx](stage_views)
                block_idx += 1
            stage_cls.extend([x[:, 0] for x in stage_views])

        fused = torch.cat(stage_cls, dim=-1)
        return self.classifier(fused)


class COMMTokenFusion(nn.Module):
    """
    Token-level COMM-style fusion:
    multi-layer alignment + LLN/LayerScale + cross-view communication.
    """

    def __init__(
        self,
        feature_shapes: List[Tuple[int, int]],
        view_layout: List[List[int]],
        num_classes: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        self.feature_shapes = feature_shapes
        self.view_layout = view_layout
        self.num_views = len(view_layout)

        in_dims = [shape[1] for shape in feature_shapes]
        self.projections = nn.ModuleList([nn.Linear(d, hidden_dim) for d in in_dims])
        self.refine = nn.ModuleList(
            [_TokenFFNLayerScale(hidden_dim=hidden_dim, dropout=0.1) for _ in feature_shapes]
        )

        self.layer_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(group))) for group in view_layout]
        )

        self.cross_blocks = nn.ModuleList(
            [_CrossViewTokenBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(depth)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.num_views),
            nn.Linear(hidden_dim * self.num_views, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def _merge_layers_in_view(self, tensors: List[torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        target_len = min(t.size(1) for t in tensors)
        aligned = [_pool_tokens_to_length(t, target_len) for t in tensors]
        weights = torch.softmax(logits, dim=0)
        return sum(w * t for w, t in zip(weights, aligned))

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != len(self.feature_shapes):
            raise ValueError(f"Expected {len(self.feature_shapes)} inputs, got {len(features)}")

        projected = [proj(x) for proj, x in zip(self.projections, features)]
        projected = [block(x) for block, x in zip(self.refine, projected)]

        view_tokens: List[torch.Tensor] = []
        for view_idx, group in enumerate(self.view_layout):
            layer_tokens = [projected[i] for i in group]
            merged = self._merge_layers_in_view(layer_tokens, self.layer_logits[view_idx])
            view_tokens.append(merged)

        target_len = min(v.size(1) for v in view_tokens)
        view_tokens = [_pool_tokens_to_length(v, target_len) for v in view_tokens]

        for block in self.cross_blocks:
            view_tokens = block(view_tokens)

        cls_repr = [v[:, 0] for v in view_tokens]
        fused = torch.cat(cls_repr, dim=-1)
        return self.classifier(fused)


def _infer_view_layout_from_order(
    feature_order: Sequence[str],
) -> List[List[int]]:
    """
    Group feature indices by view prefix.
    e.g., clip_layer_0_features -> clip, dino_tokens_features -> dino
    """
    groups: Dict[str, List[int]] = {}
    ordered_names: List[str] = []
    for idx, key in enumerate(feature_order):
        if "_layer_" in key:
            view_name = key.split("_layer_")[0]
        elif key.endswith("_tokens_features"):
            view_name = key[: -len("_tokens_features")]
        elif key.endswith("_features"):
            view_name = key[: -len("_features")]
        else:
            view_name = key

        if view_name not in groups:
            groups[view_name] = []
            ordered_names.append(view_name)
        groups[view_name].append(idx)

    return [groups[name] for name in ordered_names]


def create_fusion_model(
    method: str,
    num_classes: int,
    feature_dims: Optional[List[int]] = None,
    feature_shapes: Optional[List[Tuple[int, int]]] = None,
    view_layout: Optional[List[List[int]]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Factory for fusion models."""
    method = method.lower()
    if method == "concat":
        if feature_dims is None:
            raise ValueError("concat method requires feature_dims")
        from src.training.classifier import MultiViewClassifier
        return MultiViewClassifier(feature_dims=feature_dims, num_classes=num_classes)

    if feature_shapes is None:
        raise ValueError(f"{method} requires feature_shapes")

    if view_layout is None:
        if feature_order is None:
            raise ValueError(f"{method} requires either view_layout or feature_order")
        view_layout = _infer_view_layout_from_order(feature_order)

    if method in {"mmvit", "mmvit3"}:
        return MMViTTokenFusion(
            feature_shapes=feature_shapes,
            view_layout=view_layout,
            num_classes=num_classes,
        )

    if method in {"comm", "comm3"}:
        return COMMTokenFusion(
            feature_shapes=feature_shapes,
            view_layout=view_layout,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown fusion method: {method}")

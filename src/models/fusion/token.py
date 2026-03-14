"""Token-level fusion: COMM and MMViT inspired methods."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..extractor import FeatureExtractor, _infer_square_size, _resize_tokens, _split_cls_token, _merge_cls_token, _apply_depthwise_pool, _add_positional_embedding, _valid_num_heads


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

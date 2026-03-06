"""
Unified Multi-Model Fusion Classifiers

Supported methods:
1. Concat - Simple concatenation (baseline)
2. WeightedSum - Learnable weighted sum
3. COMM - Multi-layer fusion (COMM paper, 2 or 3 models)
4. MMViT - Cross-attention fusion (MMViT paper, precise implementation)

References:
- COMM: https://arxiv.org/abs/2310.08825
- MMViT: https://arxiv.org/abs/2305.00104
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


# =============================================================================
# 1. CONCAT FUSION (Baseline)
# =============================================================================

class ConcatFusion(nn.Module):
    """Simple concatenation fusion"""
    
    def __init__(self, feature_dims: List[int], num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        input_dim = sum(feature_dims)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, *features) -> torch.Tensor:
        x = torch.cat(features, dim=-1)
        return self.classifier(x)


# =============================================================================
# 2. WEIGHTED SUM FUSION
# =============================================================================

class WeightedSumFusion(nn.Module):
    """Learnable weighted sum fusion"""
    
    def __init__(self, feature_dims: List[int], num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.num_views = len(feature_dims)
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        self.weights = nn.Parameter(torch.ones(self.num_views) / self.num_views)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, *features) -> torch.Tensor:
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))
        return self.classifier(fused)


# =============================================================================
# 3. COMM MULTI-LAYER FUSION (Supports 2 or 3 models)
# =============================================================================

class LLNLayerscaleFusion(nn.Module):
    """
    LLN-Layerscale: Linear + LayerNorm + Learnable Scale
    
    From COMM paper: v̄ = Σ αᵢ · Linear(LN(vⁱ))
    """
    
    def __init__(self, num_layers: int, input_dim: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        self.linear_projs = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_layers)
        ])
        self.scales = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.scales, dim=0)
        fused = torch.zeros(layer_features[0].shape[0], self.linear_projs[0].out_features,
                           device=layer_features[0].device)
        for i, feat in enumerate(layer_features):
            fused = fused + weights[i] * self.linear_projs[i](self.layer_norms[i](feat))
        return fused


class AlignmentMLP(nn.Module):
    """MLP for feature alignment"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class COMMFusion(nn.Module):
    """
    COMM Multi-layer Fusion (supports 2 or 3 models)
    
    Architecture:
        CLIP (12 layers) → LLN-Layerscale → v̄₁
        DINO (6 layers)  → LLN-Layerscale → MLP → v̄₂
        [MAE (6 layers)  → LLN-Layerscale → MLP → v̄₃]  (optional)
        
        [v̄₁, v̄₂, (v̄₃)] → Classification
    """
    
    def __init__(
        self,
        num_classes: int,
        clip_hidden_dim: int = 768,
        clip_output_dim: int = 512,
        clip_num_layers: int = 12,
        dino_hidden_dim: int = 768,
        dino_num_layers: int = 6,
        mae_hidden_dim: int = 768,
        mae_num_layers: int = 6,
        use_mae: bool = False,
        mlp_hidden_dim: int = 2048,
        hidden_dims: List[int] = [1024, 512],
        dropout: List[float] = [0.5, 0.3],
    ):
        super().__init__()
        self.use_mae = use_mae
        
        # CLIP fusion (all 12 layers)
        self.clip_fusion = LLNLayerscaleFusion(
            num_layers=clip_num_layers,
            input_dim=clip_hidden_dim,
            output_dim=clip_output_dim,
        )
        
        # DINO fusion (last 6 layers)
        self.dino_fusion = LLNLayerscaleFusion(
            num_layers=dino_num_layers,
            input_dim=dino_hidden_dim,
            output_dim=clip_output_dim,
        )
        self.dino_mlp = AlignmentMLP(clip_output_dim, mlp_hidden_dim, clip_output_dim)
        
        # MAE fusion (optional)
        if use_mae:
            self.mae_fusion = LLNLayerscaleFusion(
                num_layers=mae_num_layers,
                input_dim=mae_hidden_dim,
                output_dim=clip_output_dim,
            )
            self.mae_mlp = AlignmentMLP(clip_output_dim, mlp_hidden_dim, clip_output_dim)
            fused_dim = clip_output_dim * 3
        else:
            self.mae_fusion = None
            self.mae_mlp = None
            fused_dim = clip_output_dim * 2
        
        # Classification head
        layers = []
        prev_dim = fused_dim
        for h, d in zip(hidden_dims, dropout):
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(d),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(
        self,
        clip_features: List[torch.Tensor],
        dino_features: List[torch.Tensor],
        mae_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        clip_fused = self.clip_fusion(clip_features)
        dino_fused = self.dino_fusion(dino_features)
        dino_aligned = self.dino_mlp(dino_fused)
        
        if self.use_mae and mae_features is not None:
            mae_fused = self.mae_fusion(mae_features)
            mae_aligned = self.mae_mlp(mae_fused)
            fused = torch.cat([clip_fused, dino_aligned, mae_aligned], dim=-1)
        else:
            fused = torch.cat([clip_fused, dino_aligned], dim=-1)
        
        return self.classifier(fused)


# =============================================================================
# 4. MMViT CROSS-ATTENTION FUSION (Precise Implementation)
# =============================================================================

class MMViTCrossAttentionBlock(nn.Module):
    """
    MMViT Cross-Attention Block (precise implementation from paper)
    
    From MMViT paper:
    "At each scale stage, we use a cross-attention block to fuse information 
    across different views."
    
    Architecture:
        For each view i:
            Q_i = Linear_i(view_i)
            K = Concat(Linear_k(view_k) for all k)
            V = Concat(Linear_v(view_k) for all k)
            
            Attention_i = Softmax(Q_i @ K^T / sqrt(d)) @ V
            Output_i = LayerNorm(view_i + Attention_i)
    
    Then aggregate all views.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_views: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_views = num_views
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Per-view query projections
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_views)
        ])
        
        # Shared key-value projections (one per view, then concatenated)
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_views)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_views)
        ])
        
        # Output projections
        self.out_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_views)
        ])
        
        # Layer norms
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_views)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_views)])
        
        # FFN (shared)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, view_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            view_features: List of [batch, hidden_dim] tensors
        
        Returns:
            List of updated [batch, hidden_dim] tensors
        """
        batch_size = view_features[0].shape[0]
        
        # Stack views: [batch, num_views, hidden_dim]
        stacked = torch.stack(view_features, dim=1)
        
        outputs = []
        
        for i in range(self.num_views):
            # Query from view i
            q = self.q_projs[i](view_features[i])  # [batch, hidden_dim]
            q = q.view(batch_size, self.num_heads, self.head_dim)
            
            # Key and Value from all views
            k_list = [self.k_projs[j](view_features[j]) for j in range(self.num_views)]
            v_list = [self.v_projs[j](view_features[j]) for j in range(self.num_views)]
            
            k = torch.stack(k_list, dim=1)  # [batch, num_views, hidden_dim]
            v = torch.stack(v_list, dim=1)
            
            k = k.view(batch_size, self.num_views, self.num_heads, self.head_dim)
            v = v.view(batch_size, self.num_views, self.num_heads, self.head_dim)
            
            # Attention: [batch, heads, head_dim] @ [batch, views, heads, head_dim]^T
            # -> [batch, heads, views]
            attn = torch.einsum('bhd,bnhd->bhn', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply to values: [batch, heads, views] @ [batch, views, heads, head_dim]
            # -> [batch, heads, head_dim]
            out = torch.einsum('bhn,bnhd->bhd', attn, v)
            out = out.reshape(batch_size, self.hidden_dim)
            
            # Output projection
            out = self.out_projs[i](out)
            
            # Residual + Norm
            out = self.norm1[i](view_features[i] + out)
            
            # FFN + Residual + Norm
            out = self.norm2[i](out + self.ffn(out))
            
            outputs.append(out)
        
        return outputs


class MMViTFusion(nn.Module):
    """
    MMViT: Multiscale Multiview Vision Transformers (Precise Implementation)
    
    From paper (arXiv:2305.00104):
    "Our model encodes different views of the input signal and builds several 
    channel-resolution feature stages to process the multiple views of the input 
    at different resolutions in parallel."
    
    For our use case (pre-trained model features):
        - Views = Different pre-trained models (CLIP, DINO, MAE)
        - Cross-attention fuses information across views
    
    Architecture:
        View1 (CLIP) ─┐
        View2 (DINO) ─┼──→ [Project to hidden_dim] → [Cross-Attention × N] → [Aggregate] → Classify
        View3 (MAE)  ─┘
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_views = len(feature_dims)
        self.hidden_dim = hidden_dim
        
        # Project each view to hidden_dim
        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for dim in feature_dims
        ])
        
        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            MMViTCrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_views=self.num_views,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
        # Final aggregation and classification
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_views, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, *features) -> torch.Tensor:
        # Project views
        view_features = [proj(f) for proj, f in zip(self.view_projections, features)]
        
        # Apply cross-attention blocks
        for block in self.cross_attn_blocks:
            view_features = block(view_features)
        
        # Aggregate: concatenate all views
        aggregated = torch.cat(view_features, dim=-1)
        
        # Final norm and classify
        return self.classifier(aggregated)


class MMViTLiteFusion(nn.Module):
    """
    MMViT-Lite: Single cross-attention block
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_views = len(feature_dims)
        
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        
        self.cross_attn = MMViTCrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_views=self.num_views,
            dropout=dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * self.num_views, num_classes),
        )
    
    def forward(self, *features) -> torch.Tensor:
        view_features = [proj(f) for proj, f in zip(self.projections, features)]
        view_features = self.cross_attn(view_features)
        aggregated = torch.cat(view_features, dim=-1)
        return self.classifier(aggregated)


# =============================================================================
# 5. FACTORY FUNCTION
# =============================================================================

def create_fusion_model(
    method: str,
    feature_dims: List[int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion models.
    
    Args:
        method: 'concat', 'weighted_sum', 'comm', 'comm3', 'mmvit', 'mmvit_lite'
        feature_dims: Feature dimensions for each model
        num_classes: Number of classes
    """
    method = method.lower()
    
    if method == 'concat':
        return ConcatFusion(feature_dims, num_classes, **kwargs)
    elif method == 'weighted_sum':
        return WeightedSumFusion(feature_dims, num_classes, **kwargs)
    elif method == 'comm':
        return COMMFusion(num_classes=num_classes, use_mae=False, **kwargs)
    elif method == 'comm3':
        return COMMFusion(num_classes=num_classes, use_mae=True, **kwargs)
    elif method == 'mmvit':
        return MMViTFusion(feature_dims, num_classes, **kwargs)
    elif method == 'mmvit_lite':
        return MMViTLiteFusion(feature_dims, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 6. BACKWARD COMPATIBILITY
# =============================================================================

MultiViewClassifier = ConcatFusion
SingleViewClassifier = ConcatFusion
COMMFusionClassifier = COMMFusion
COMM3FusionClassifier = COMMFusion
MMViTFusionClassifier = MMViTFusion
MMViTLiteFusionClassifier = MMViTLiteFusion

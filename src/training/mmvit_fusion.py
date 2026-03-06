"""
MMViT-style Multi-view Fusion Classifier

Based on: "MMViT: Multiscale Multiview Vision Transformers"
https://arxiv.org/abs/2305.00104

Key idea: Use cross-attention to fuse features from different "views" (models)

Architecture:
    CLIP features ──┐
    DINO features ──┼──→ [Cross-Attention Fusion] ──→ Classification
    MAE features  ──┘

Cross-Attention: Each view can attend to all other views
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class MultiViewCrossAttention(nn.Module):
    """
    Cross-attention module for fusing multiple views.
    
    Each view attends to all other views, then aggregates.
    Similar to MMViT's cross-attention block.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dims: List of feature dimensions for each view
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_views = len(feature_dims)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Project each view to hidden_dim
        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ) for dim in feature_dims
        ])
        
        # Cross-attention: each view can attend to others
        # Q from view i, K/V from all views
        self.q_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_views)
        ])
        self.k_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_views)
        ])
        self.v_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_views)
        ])
        
        # Output projection for each view
        self.out_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_views)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, view_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple view features via cross-attention.
        
        Args:
            view_features: List of tensors [batch, dim_i] for each view
        
        Returns:
            Fused features [batch, hidden_dim]
        """
        batch_size = view_features[0].shape[0]
        
        # Project all views to hidden_dim
        projected = [
            proj(feat) for proj, feat in zip(self.view_projections, view_features)
        ]  # Each: [batch, hidden_dim]
        
        # Stack for easier computation
        stacked = torch.stack(projected, dim=1)  # [batch, num_views, hidden_dim]
        
        # Cross-attention
        attended_outputs = []
        
        for i in range(self.num_views):
            # Query from view i
            q = self.q_projections[i](projected[i])  # [batch, hidden_dim]
            
            # Key and Value from all views
            k = torch.cat([self.k_projections[j](projected[j]) for j in range(self.num_views)], dim=1)
            v = torch.cat([self.v_projections[j](projected[j]) for j in range(self.num_views)], dim=1)
            # k, v: [batch, num_views * hidden_dim]
            
            # Reshape for multi-head attention
            q = q.view(batch_size, self.num_heads, self.head_dim)  # [batch, heads, head_dim]
            k = k.view(batch_size, self.num_views, self.num_heads, self.head_dim)
            v = v.view(batch_size, self.num_views, self.num_heads, self.head_dim)
            
            # Attention: [batch, heads, 1, head_dim] @ [batch, heads, head_dim, num_views]
            # -> [batch, heads, 1, num_views]
            attn_scores = torch.einsum('bhd,bnhd->bhn', q, k) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, heads, num_views]
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values: [batch, heads, num_views] @ [batch, num_views, heads, head_dim]
            # -> [batch, heads, head_dim]
            attended = torch.einsum('bhn,bnhd->bhd', attn_weights, v)
            attended = attended.reshape(batch_size, self.hidden_dim)
            
            # Output projection
            out = self.out_projections[i](attended)
            attended_outputs.append(out)
        
        # Aggregate all attended outputs
        fused = torch.stack(attended_outputs, dim=1).mean(dim=1)  # [batch, hidden_dim]
        fused = self.norm(fused)
        
        return fused


class MultiscaleFusionBlock(nn.Module):
    """
    Multiscale fusion block inspired by MMViT.
    
    Processes features at multiple scales and fuses them.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 512, 1024],
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions at each scale
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_scales = len(hidden_dims)
        
        # Project to different scales
        self.scale_projections = nn.ModuleList([
            nn.Linear(input_dim, dim) for dim in hidden_dims
        ])
        
        # Cross-scale attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=max(hidden_dims),
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Fuse scales
        self.fusion = nn.Linear(sum(hidden_dims), max(hidden_dims))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, input_dim]
        
        Returns:
            Fused features [batch, max_hidden_dim]
        """
        # Project to multiple scales
        scale_features = [proj(x) for proj in self.scale_projections]
        
        # Pad to same dimension for attention
        max_dim = max(f.shape[-1] for f in scale_features)
        padded = [F.pad(f, (0, max_dim - f.shape[-1])) for f in scale_features]
        
        # Stack for attention
        stacked = torch.stack(padded, dim=1)  # [batch, num_scales, max_dim]
        
        # Self-attention across scales
        attn_out, _ = self.cross_attention(stacked, stacked, stacked)
        
        # Concatenate original scale features
        concat = torch.cat(scale_features, dim=-1)
        
        # Fuse
        fused = self.fusion(concat)
        
        return fused


class MMViTFusionClassifier(nn.Module):
    """
    MMViT-style Multi-view Fusion Classifier.
    
    Fuses features from multiple pre-trained models using:
    1. Cross-attention between views
    2. Multiscale processing
    3. Final classification head
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_multiscale: bool = True,
    ):
        """
        Args:
            feature_dims: Feature dimensions for each model (e.g., [512, 768, 768] for CLIP+DINO+MAE)
            num_classes: Number of output classes
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
            use_multiscale: Whether to use multiscale fusion
        """
        super().__init__()
        
        self.num_views = len(feature_dims)
        self.hidden_dim = hidden_dim
        self.use_multiscale = use_multiscale
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            MultiViewCrossAttention(
                feature_dims=feature_dims if i == 0 else [hidden_dim] * self.num_views,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for i in range(num_layers)
        ])
        
        # Multiscale fusion (optional)
        if use_multiscale:
            self.multiscale = MultiscaleFusionBlock(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim // 2, hidden_dim, hidden_dim * 2],
                num_heads=num_heads,
                dropout=dropout,
            )
            classifier_input_dim = hidden_dim * 2
        else:
            self.multiscale = None
            classifier_input_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, *view_features) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            *view_features: Variable number of feature tensors [batch, dim_i]
        
        Returns:
            Logits [batch, num_classes]
        """
        # List of features
        features = list(view_features)
        
        # Apply cross-attention layers
        for cross_attn in self.cross_attention_layers:
            # Each layer takes individual view features and produces fused output
            # But we need to maintain per-view representations for next layer
            fused = cross_attn(features)
            # Split fused back to per-view for next layer
            features = [fused for _ in range(self.num_views)]
        
        # Final fused representation
        x = features[0]
        
        # Multiscale processing
        if self.multiscale is not None:
            x = self.multiscale(x)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


class MMViTLiteFusionClassifier(nn.Module):
    """
    Lightweight version of MMViT fusion.
    
    Simpler architecture with single cross-attention layer.
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
        
        # Project all views to hidden_dim
        self.view_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in feature_dims
        ])
        
        # Cross-view self-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, *view_features) -> torch.Tensor:
        # Project views
        projected = [proj(feat) for proj, feat in zip(self.view_projections, view_features)]
        
        # Stack: [batch, num_views, hidden_dim]
        stacked = torch.stack(projected, dim=1)
        
        # Self-attention across views
        attn_out, _ = self.cross_attention(stacked, stacked, stacked)
        stacked = self.norm1(stacked + attn_out)
        
        # FFN
        stacked = self.norm2(stacked + self.ffn(stacked))
        
        # Global pooling across views
        fused = stacked.mean(dim=1)  # [batch, hidden_dim]
        
        # Classify
        return self.classifier(fused)

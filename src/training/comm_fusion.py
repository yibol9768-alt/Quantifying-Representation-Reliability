"""
COMM Fusion Classifier

Based on: "From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models"
https://arxiv.org/abs/2310.08825

COMM = CLIP and DINO with Multi-level features Merging

Key components:
1. LLN-Layerscale: Linear + LayerNorm + learnable scale weights for multi-layer fusion
2. MLP alignment: Align DINO features to CLIP feature space
3. Final fusion: Concatenate CLIP and aligned DINO features
"""
import torch
import torch.nn as nn
from typing import List, Optional


class LLNLayerscaleFusion(nn.Module):
    """
    LLN-Layerscale module for multi-layer feature fusion.

    From COMM paper:
        v̄ = Σ αᵢ · Linear(LN(vⁱ))

    Where:
        - LN: LayerNorm
        - Linear: Linear projection
        - α: Learnable scale parameters (normalized via softmax)
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        output_dim: int,
    ):
        """
        Args:
            num_layers: Number of layers to fuse
            input_dim: Input feature dimension (hidden dim of the model)
            output_dim: Output feature dimension
        """
        super().__init__()

        self.num_layers = num_layers

        # LayerNorm for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])

        # Linear projection for each layer
        self.linear_projs = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=True) for _ in range(num_layers)
        ])

        # Learnable scale parameters
        # Initialize to uniform distribution
        self.scales = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-layer features.

        Args:
            layer_features: List of tensors [batch, input_dim] for each layer

        Returns:
            Fused feature tensor [batch, output_dim]
        """
        assert len(layer_features) == self.num_layers, \
            f"Expected {self.num_layers} layer features, got {len(layer_features)}"

        batch_size = layer_features[0].shape[0]
        output_device = layer_features[0].device

        # Normalize scales using softmax
        scale_weights = torch.softmax(self.scales, dim=0)

        # Initialize output
        fused = torch.zeros(batch_size, self.linear_projs[0].out_features,
                           device=output_device)

        # Apply LLN and weighted sum
        for i, feat in enumerate(layer_features):
            # LayerNorm
            normalized = self.layer_norms[i](feat)
            # Linear projection
            projected = self.linear_projs[i](normalized)
            # Weighted sum
            fused = fused + scale_weights[i] * projected

        return fused


class DINOAlignmentMLP(nn.Module):
    """
    MLP module to align DINO features to CLIP feature space.

    From COMM paper:
        "we employ an MLP layer to project the features of DINOv2
         and concatenate the output features with that of CLIP"

    Architecture: Linear -> GELU -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """
        Args:
            input_dim: DINO feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (should match CLIP feature dim)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Align DINO features.

        Args:
            x: DINO features [batch, input_dim]

        Returns:
            Aligned features [batch, output_dim]
        """
        return self.mlp(x)


class COMMFusionClassifier(nn.Module):
    """
    COMM Fusion Classifier for CLIP + DINO multi-level features.

    Architecture:
        1. Extract multi-layer features from CLIP (all layers) and DINO (deep layers)
        2. Apply LLN-Layerscale fusion for each model separately
        3. Align DINO features using MLP
        4. Concatenate CLIP and aligned DINO features
        5. Final classification head

    From paper:
        v̄₁ = Σᵢ₌₁²⁴ αᵢ · Linear(LN(v₁ⁱ))  # CLIP all layers
        v̄₂ = Σⱼ₌₁₉²⁴ βⱼ · Linear(LN(v₂ʲ))  # DINO layers 19-24
        v̄ = [v̄₁, MLP(v̄₂)]  # Concatenate
    """

    def __init__(
        self,
        clip_hidden_dim: int = 768,  # ViT-B hidden dim
        clip_output_dim: int = 512,  # CLIP output dim (projected)
        clip_num_layers: int = 12,  # ViT-B has 12 layers
        dino_hidden_dim: int = 768,  # ViT-B hidden dim
        dino_num_layers: int = 6,  # Use last 6 layers (7-12 for ViT-B)
        num_classes: int = 196,
        hidden_dims: List[int] = [1024, 512],
        dropout: List[float] = [0.5, 0.3],
        mlp_hidden_dim: int = 2048,  # MLP expansion factor * output_dim
    ):
        """
        Args:
            clip_hidden_dim: CLIP ViT hidden dimension
            clip_output_dim: CLIP output dimension (for fusion)
            clip_num_layers: Number of CLIP layers to fuse
            dino_hidden_dim: DINO ViT hidden dimension
            dino_num_layers: Number of DINO layers to fuse
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions for classification head
            dropout: Dropout rates
            mlp_hidden_dim: Hidden dimension for DINO alignment MLP
        """
        super().__init__()

        # CLIP multi-layer fusion (LLN-Layerscale)
        self.clip_fusion = LLNLayerscaleFusion(
            num_layers=clip_num_layers,
            input_dim=clip_hidden_dim,
            output_dim=clip_output_dim,
        )

        # DINO multi-layer fusion (LLN-Layerscale)
        self.dino_fusion = LLNLayerscaleFusion(
            num_layers=dino_num_layers,
            input_dim=dino_hidden_dim,
            output_dim=clip_output_dim,  # Project to same dim as CLIP
        )

        # DINO alignment MLP
        # Paper uses 2-layer MLP with expansion ratio of 4 (for ViT-L: 1024 -> 4096 -> 1024)
        self.dino_mlp = DINOAlignmentMLP(
            input_dim=clip_output_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=clip_output_dim,
        )

        # Fused dimension
        self.fused_dim = clip_output_dim * 2  # CLIP + DINO (both projected to clip_output_dim)

        # Classification head
        layers = []
        prev_dim = self.fused_dim

        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_dims, dropout)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        clip_layer_features: List[torch.Tensor],
        dino_layer_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with multi-layer features.

        Args:
            clip_layer_features: List of CLIP layer features [batch, hidden_dim]
            dino_layer_features: List of DINO layer features [batch, hidden_dim]

        Returns:
            Logits [batch, num_classes]
        """
        # Fuse CLIP multi-layer features
        clip_fused = self.clip_fusion(clip_layer_features)

        # Fuse DINO multi-layer features
        dino_fused = self.dino_fusion(dino_layer_features)

        # Align DINO features with MLP
        dino_aligned = self.dino_mlp(dino_fused)

        # Concatenate
        fused = torch.cat([clip_fused, dino_aligned], dim=1)

        # Classify
        logits = self.classifier(fused)

        return logits


class ConcatFusionClassifier(nn.Module):
    """
    Simple concatenation fusion classifier (baseline).

    This is the original fusion method: concatenate final layer features
    from multiple models and pass through MLP classifier.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int = 196,
        hidden_dims: List[int] = [1024, 512],
        dropout: List[float] = [0.5, 0.3],
    ):
        """
        Args:
            feature_dims: List of feature dimensions for each model
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rates
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.input_dim = sum(feature_dims)

        # Build MLP
        layers = []
        prev_dim = self.input_dim

        for hidden_dim, drop_rate in zip(hidden_dims, dropout):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, *features) -> torch.Tensor:
        """
        Forward pass.

        Args:
            *features: Variable number of feature tensors

        Returns:
            Logits [batch, num_classes]
        """
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)


class WeightedSumFusionClassifier(nn.Module):
    """
    Weighted sum fusion classifier.

    Alternative fusion: learn weights to combine features, then classify.
    Simpler than COMM but more flexible than concatenation.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int = 196,
        fusion_dim: int = 512,
        hidden_dims: List[int] = [1024, 512],
        dropout: List[float] = [0.5, 0.3],
    ):
        """
        Args:
            feature_dims: List of feature dimensions for each model
            num_classes: Number of output classes
            fusion_dim: Dimension to project all features to before fusion
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rates
        """
        super().__init__()

        self.num_models = len(feature_dims)

        # Project each model's features to fusion_dim
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)

        # Classification head
        layers = []
        prev_dim = fusion_dim

        for hidden_dim, drop_rate in zip(hidden_dims, dropout):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, *features) -> torch.Tensor:
        """
        Forward pass.

        Args:
            *features: Variable number of feature tensors

        Returns:
            Logits [batch, num_classes]
        """
        # Project all features
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Weighted sum
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = sum(w * p for w, p in zip(weights, projected))

        # Classify
        return self.classifier(fused)

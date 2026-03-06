"""
COMM3 Fusion Classifier - Extended COMM with 3 models (CLIP + DINO + MAE)

Based on original COMM: "From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models"
https://arxiv.org/abs/2310.08825

Extended to include MAE alongside CLIP and DINO for richer feature fusion.

Key components:
1. LLN-Layerscale: Linear + LayerNorm + learnable scale weights for multi-layer fusion
2. MLP alignment: Align DINO and MAE features to CLIP feature space
3. Final fusion: Concatenate CLIP, aligned DINO, and aligned MAE features

Layer selection:
    - CLIP: All 12 layers (multi-modal training makes all layers useful)
    - DINO: Last 6 layers (7-12) - deep semantic features
    - MAE: Last 6 layers (7-12) - similar to DINO, self-supervised
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
        self.scales = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-layer features.

        Args:
            layer_features: List of tensors [batch, input_dim] for each layer

        Returns:
            Fused feature tensor [batch, output_dim]
        """
        batch_size = layer_features[0].shape[0]
        output_device = layer_features[0].device

        # Normalize scales using softmax
        scale_weights = torch.softmax(self.scales, dim=0)

        # Initialize output
        fused = torch.zeros(batch_size, self.linear_projs[0].out_features,
                           device=output_device)

        # Apply LLN and weighted sum
        for i, feat in enumerate(layer_features):
            normalized = self.layer_norms[i](feat)
            projected = self.linear_projs[i](normalized)
            fused = fused + scale_weights[i] * projected

        return fused


class AlignmentMLP(nn.Module):
    """
    MLP module to align features (DINO/MAE) to CLIP feature space.

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
            input_dim: Input feature dimension (DINO or MAE)
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
        Align features.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Aligned features [batch, output_dim]
        """
        return self.mlp(x)


class COMM3FusionClassifier(nn.Module):
    """
    COMM3 Fusion Classifier for CLIP + DINO + MAE multi-level features.

    Architecture:
        1. Extract multi-layer features from CLIP (all 12), DINO (last 6), MAE (last 6)
        2. Apply LLN-Layerscale fusion for each model separately
        3. Align DINO and MAE features using MLPs to CLIP space
        4. Concatenate CLIP, aligned DINO, and aligned MAE features
        5. Final classification head

    Args:
        num_classes: Number of output classes
        clip_output_dim: CLIP fused feature dimension (default: 512)
        dino_output_dim: DINO fused feature dimension (default: 512)
        mae_output_dim: MAE fused feature dimension (default: 512)
        alignment_hidden_dim: Hidden dimension for alignment MLPs (default: 512)
        classifier_hidden_dim: Hidden dimension for final classifier (default: 512)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        num_classes: int = 100,
        clip_output_dim: int = 512,
        dino_output_dim: int = 512,
        mae_output_dim: int = 512,
        alignment_hidden_dim: int = 512,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # LLN-Layerscale fusion for CLIP (12 layers -> clip_output_dim)
        self.clip_fusion = LLNLayerscaleFusion(
            num_layers=12,
            input_dim=768,  # CLIP ViT-B hidden dim
            output_dim=clip_output_dim,
        )

        # LLN-Layerscale fusion for DINO (6 layers -> dino_output_dim)
        self.dino_fusion = LLNLayerscaleFusion(
            num_layers=6,
            input_dim=768,  # DINO ViT-B hidden dim
            output_dim=dino_output_dim,
        )

        # LLN-Layerscale fusion for MAE (6 layers -> mae_output_dim)
        self.mae_fusion = LLNLayerscaleFusion(
            num_layers=6,
            input_dim=768,  # MAE ViT-Base hidden dim
            output_dim=mae_output_dim,
        )

        # Alignment MLPs: DINO and MAE -> CLIP space
        self.dino_alignment = AlignmentMLP(
            input_dim=dino_output_dim,
            hidden_dim=alignment_hidden_dim,
            output_dim=clip_output_dim,
        )

        self.mae_alignment = AlignmentMLP(
            input_dim=mae_output_dim,
            hidden_dim=alignment_hidden_dim,
            output_dim=clip_output_dim,
        )

        # Calculate total fused dimension
        total_dim = clip_output_dim + clip_output_dim + clip_output_dim  # 3 * clip_output_dim

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        clip_layer_features: List[torch.Tensor],
        dino_layer_features: List[torch.Tensor],
        mae_layer_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            clip_layer_features: List of 12 CLIP layer features [batch, 768]
            dino_layer_features: List of 6 DINO layer features [batch, 768]
            mae_layer_features: List of 6 MAE layer features [batch, 768]

        Returns:
            Logits [batch, num_classes]
        """
        # Fuse multi-layer features for each model
        clip_fused = self.clip_fusion(clip_layer_features)      # [batch, clip_output_dim]
        dino_fused = self.dino_fusion(dino_layer_features)      # [batch, dino_output_dim]
        mae_fused = self.mae_fusion(mae_layer_features)         # [batch, mae_output_dim]

        # Align DINO and MAE to CLIP space
        dino_aligned = self.dino_alignment(dino_fused)           # [batch, clip_output_dim]
        mae_aligned = self.mae_alignment(mae_fused)             # [batch, clip_output_dim]

        # Concatenate all three
        combined = torch.cat([clip_fused, dino_aligned, mae_aligned], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits

    def get_fused_dim(self) -> int:
        """Get the total fused feature dimension after concatenation"""
        # After alignment: all three are aligned to clip_output_dim
        return self.classifier[0].in_features

"""
特征融合架构。

包含三个核心组件：
  1. FeatureProjector  — 异构特征维度对齐（线性投影到公共隐空间）
  2. AdaptiveWeights   — 可学习 Softmax 归一化权重
  3. FusionModule      — 完整融合管线：投影 → 加权聚合
"""
import torch
import torch.nn as nn
from typing import List, Dict


class FeatureProjector(nn.Module):
    """
    为每个模型维护独立的线性投影层，将异构维度映射到公共 d_proj 维空间。

    z_i = Linear_i(f_i),  i ∈ {1, ..., k}
    """

    def __init__(self, feat_dims: Dict[str, int], d_proj: int):
        """
        Args:
            feat_dims: {model_name: feat_dim}
            d_proj:    公共隐空间维度
        """
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: nn.Linear(dim, d_proj)
            for name, dim in feat_dims.items()
        })
        self.d_proj = d_proj

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {model_name: (B, feat_dim_i)}

        Returns:
            projected: {model_name: (B, d_proj)}
        """
        return {name: self.projectors[name](feat)
                for name, feat in features.items()}


class AdaptiveWeights(nn.Module):
    """
    可学习的自适应融合权重。

    α_i = softmax(w)_i，确保 α_i ∈ (0,1) 且 Σα_i = 1
    """

    def __init__(self, model_names: List[str]):
        super().__init__()
        self.model_names = list(model_names)
        # 可学习标量参数，初始化为 0（softmax 后均匀分布）
        self.raw_weights = nn.Parameter(torch.zeros(len(model_names)))

    def forward(self) -> Dict[str, torch.Tensor]:
        """返回归一化后的权重 {model_name: α_i}"""
        alphas = torch.softmax(self.raw_weights, dim=0)
        return {name: alphas[i] for i, name in enumerate(self.model_names)}

    def get_weight_dict(self) -> Dict[str, float]:
        """获取当前权重（用于日志记录）"""
        with torch.no_grad():
            alphas = torch.softmax(self.raw_weights, dim=0)
            return {name: alphas[i].item()
                    for i, name in enumerate(self.model_names)}


class FusionModule(nn.Module):
    """
    完整的特征融合模块。

    Forward:
        {model_name: (B, feat_dim_i)} → Z_fused (B, d_proj)

    流程:
        1. 线性投影对齐: z_i = Linear_i(f_i)
        2. Softmax 归一化权重: α_i = softmax(w_i)
        3. 加权聚合: Z_fused = Σ α_i · z_i
    """

    def __init__(self, feat_dims: Dict[str, int], d_proj: int):
        """
        Args:
            feat_dims: {model_name: feat_dim}  当前参与融合的模型及其特征维度
            d_proj:    公共隐空间维度
        """
        super().__init__()
        self.model_names = list(feat_dims.keys())
        self.projector = FeatureProjector(feat_dims, d_proj)
        self.weights = AdaptiveWeights(self.model_names)
        self.d_proj = d_proj

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {model_name: (B, feat_dim_i)}

        Returns:
            z_fused: (B, d_proj)
        """
        projected = self.projector(features)     # {name: (B, d_proj)}
        alphas = self.weights()                   # {name: scalar}

        # Z_fused = Σ α_i · z_i
        z_fused = torch.zeros_like(next(iter(projected.values())))
        for name in self.model_names:
            z_fused = z_fused + alphas[name] * projected[name]

        return z_fused

    def get_weight_dict(self) -> Dict[str, float]:
        return self.weights.get_weight_dict()

"""
models/heads.py
===============
唯一的分类头：基于欧氏距离的原型分类器（无可学习参数）。

对应论文第5节：「基于类原型距离度量的无参分类范式」

完整流程：
  A) 用 support set 构建每类的特征质心 P_c = mean(Z_fused of class c)
  B) 计算 query 与每个原型的平方欧氏距离 d(x_q, P_c)
  C) 负距离经 Softmax 转换为概率分布
  D) 最小化交叉熵损失，梯度反传到上游 fusion 模块
     （更新投影头 + 融合权重，不更新冻结编码器）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    """
    非参数化原型分类器，本身没有任何可学习参数。
    梯度通过距离公式流回 Z_fused，进而更新 fusion 模块。
    """

    def __init__(self) -> None:
        super().__init__()

    def build_prototypes(
        self,
        support_features: torch.Tensor,  # (N*K, D)
        support_labels:   torch.Tensor,  # (N*K,)  取值 0..N-1
        n_way:            int,
    ) -> torch.Tensor:
        """
        P_c = (1/K) Σ_{x_i ∈ S_c} Z_fused(x_i)
        返回 (N, D) 的原型矩阵。
        """
        D = support_features.size(-1)
        prototypes = torch.zeros(n_way, D, device=support_features.device)
        for c in range(n_way):
            mask = support_labels == c
            prototypes[c] = support_features[mask].mean(0)
        return prototypes  # (N, D)

    def forward(
        self,
        support_features: torch.Tensor,  # (N*K, D)
        support_labels:   torch.Tensor,  # (N*K,)
        query_features:   torch.Tensor,  # (N*Q, D)
        n_way:            int,
    ) -> torch.Tensor:
        """
        返回 log-概率矩阵 (N*Q, N)。

        d(x_q, P_c) = ||Z_fused(x_q) - P_c||_2^2
        p(y=c|x_q)  = softmax(-d(x_q, P_c))
        """
        prototypes = self.build_prototypes(support_features, support_labels, n_way)
        dists = _squared_euclidean(query_features, prototypes)  # (N*Q, N)
        return F.log_softmax(-dists, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        support_features: torch.Tensor,
        support_labels:   torch.Tensor,
        query_features:   torch.Tensor,
        n_way:            int,
    ) -> torch.Tensor:
        """返回 query 样本的预测类别索引 (N*Q,)。"""
        log_probs = self.forward(support_features, support_labels, query_features, n_way)
        return log_probs.argmax(dim=-1)


# ---------------------------------------------------------------------------
# 工具函数：高效平方欧氏距离矩阵
# ---------------------------------------------------------------------------
def _squared_euclidean(
    queries:    torch.Tensor,  # (M, D)
    prototypes: torch.Tensor,  # (N, D)
) -> torch.Tensor:
    """
    利用恒等式 ||a-b||^2 = ||a||^2 + ||b||^2 - 2·a·b^T 高效计算。
    返回 (M, N) 距离矩阵。
    """
    q_sq  = (queries    ** 2).sum(dim=1, keepdim=True)    # (M, 1)
    p_sq  = (prototypes ** 2).sum(dim=1, keepdim=True).T  # (1, N)
    cross = queries @ prototypes.T                          # (M, N)
    return (q_sq + p_sq - 2 * cross).clamp(min=0)

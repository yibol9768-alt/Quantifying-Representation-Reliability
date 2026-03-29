"""
基于类原型欧氏距离的无参分类器。

流程：
  1. 在 support set 上构建类原型 P_c = mean(Z_fused of class c)
  2. 对 query 样本计算到各原型的欧氏距离
  3. 负距离经 Softmax 转为概率分布
  4. 交叉熵损失驱动融合模块的参数更新
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def build_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    构建类原型。

    P_c = (1/K) Σ_{x_i ∈ S_c} Z_fused(x_i)

    Args:
        features:    (N, d_proj)
        labels:      (N,)
        num_classes: 类别总数

    Returns:
        prototypes: (num_classes, d_proj)
    """
    d = features.shape[1]
    prototypes = torch.zeros(num_classes, d, device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = features[mask].mean(dim=0)
    return prototypes


def prototype_logits(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    计算 query 到各原型的负欧氏距离（作为 logits）。

    logits(x_q, c) = -||Z_fused(x_q) - P_c||_2

    Args:
        query_features: (N_q, d_proj)
        prototypes:     (N_cls, d_proj)

    Returns:
        logits: (N_q, N_cls)  负距离
    """
    # cdist: (1, N_q, d) vs (1, N_cls, d) → (1, N_q, N_cls) → (N_q, N_cls)
    dists = torch.cdist(
        query_features.unsqueeze(0),
        prototypes.unsqueeze(0),
    ).squeeze(0)
    return -dists


def prototype_loss(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """
    原型分类交叉熵损失。

    L = -(1/|Q|) Σ log p(y_q | x_q)
    p(y=c | x_q) = softmax(-d(x_q, P_c))

    Args:
        query_features: (N_q, d_proj)
        query_labels:   (N_q,)
        prototypes:     (N_cls, d_proj)

    Returns:
        loss:     标量交叉熵损失
        accuracy: float 分类准确率
    """
    logits = prototype_logits(query_features, prototypes)
    loss = F.cross_entropy(logits, query_labels)

    preds = logits.argmax(dim=1)
    accuracy = (preds == query_labels).float().mean().item()

    return loss, accuracy

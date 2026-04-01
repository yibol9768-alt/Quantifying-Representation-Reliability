"""
任务相关性评估 R̂(m)：单模型 few-shot 准确率。

对每个候选模型独立运行 few-shot 评估：
  冻结特征 → 线性投影 → 类原型 → 欧氏距离分类 → 准确率
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict


def _build_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    构建类原型 P_c = mean(features of class c)

    Args:
        features:    (N, d_proj) 投影后特征
        labels:      (N,) 标签
        num_classes: 类别数

    Returns:
        prototypes: (num_classes, d_proj)
    """
    prototypes = torch.zeros(num_classes, features.shape[1], device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = features[mask].mean(dim=0)
    return prototypes


def _euclidean_classify(
    query_feats: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """基于欧氏距离的最近原型分类"""
    # (N_query, 1, d) - (1, N_class, d) → (N_query, N_class)
    dists = torch.cdist(query_feats.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
    return dists.argmin(dim=1)


def evaluate_single_model(
    support_feats: torch.Tensor,
    support_labels: torch.Tensor,
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    num_classes: int,
    d_proj: int = 512,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda",
) -> float:
    """
    单模型 few-shot 评估流程。

    1. 训练线性投影层（support set 上原型分类交叉熵）
    2. 在 query set 上计算分类准确率

    Returns:
        accuracy: float ∈ [0, 1]
    """
    feat_dim = support_feats.shape[1]
    proj = nn.Linear(feat_dim, d_proj).to(device)

    support_feats = support_feats.to(device)
    support_labels = support_labels.to(device)
    query_feats = query_feats.to(device)
    query_labels = query_labels.to(device)

    optimizer = optim.Adam(proj.parameters(), lr=lr, weight_decay=1e-4)

    # 训练投影层
    proj.train()
    for _ in range(epochs):
        z_support = proj(support_feats)          # (N_s, d_proj)
        prototypes = _build_prototypes(z_support, support_labels, num_classes)

        # 负欧氏距离 → softmax → 交叉熵
        dists = torch.cdist(z_support.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        logits = -dists  # (N_s, num_classes)
        loss = nn.functional.cross_entropy(logits, support_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    proj.eval()
    with torch.no_grad():
        z_support = proj(support_feats)
        z_query = proj(query_feats)
        prototypes = _build_prototypes(z_support, support_labels, num_classes)
        preds = _euclidean_classify(z_query, prototypes)
        accuracy = (preds == query_labels).float().mean().item()

    return accuracy


def compute_relevance_scores(
    support_features: Dict[str, torch.Tensor],
    query_features: Dict[str, torch.Tensor],
    support_labels: torch.Tensor,
    query_labels: torch.Tensor,
    model_names: list,
    num_classes: int,
    d_proj: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    计算所有模型的归一化任务相关性分数 R̂(m)。

    Returns:
        {model_name: R̂(m)} 其中最优模型 R̂ = 1.0
    """
    raw_acc = {}

    for name in model_names:
        s_feats = support_features[name]
        q_feats = query_features[name]

        acc = evaluate_single_model(
            s_feats, support_labels, q_feats, query_labels,
            num_classes=num_classes, d_proj=d_proj, device=device,
        )
        raw_acc[name] = acc
        print(f"    {name}: Acc = {acc:.4f}")

    # 归一化：R̂(m) = Acc(m) / max Acc
    max_acc = max(raw_acc.values())
    relevance = {name: acc / max_acc for name, acc in raw_acc.items()}

    return relevance

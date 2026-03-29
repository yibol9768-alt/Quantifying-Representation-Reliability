"""
analysis/metrics.py
===================
指标计算工具，仅针对 few-shot 情节场景。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> float:
    """计算 top-k 准确率。"""
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()


def episode_stats(episode_accs: List[float]) -> Tuple[float, float]:
    """
    计算多个 few-shot 情节的均值准确率和 95% 置信区间。
    返回 (mean, ci_95)。
    """
    arr  = np.array(episode_accs)
    mean = arr.mean()
    ci   = 1.96 * arr.std() / np.sqrt(len(arr))
    return float(mean), float(ci)

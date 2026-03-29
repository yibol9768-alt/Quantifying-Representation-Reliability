"""
CKA (Centered Kernel Alignment) 计算模块。

用于衡量两个模型特征表示的相似度，作为冗余代理 D̂(m, S)。
"""
import torch
import numpy as np
from typing import Dict


def _centering_matrix(n: int) -> np.ndarray:
    """H = I_n - (1/n) * 1 * 1^T"""
    return np.eye(n) - np.ones((n, n)) / n


def _hsic(K: np.ndarray, L: np.ndarray, H: np.ndarray) -> float:
    """
    HSIC(K, L) = 1/(n-1)^2 * tr(K̃ L̃)
    其中 K̃ = HKH, L̃ = HLH
    """
    n = K.shape[0]
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    return np.trace(K_centered @ L_centered) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    线性 CKA：衡量两组特征的表示相似度。

    Args:
        X: (n, d1) 模型 1 的特征矩阵
        Y: (n, d2) 模型 2 的特征矩阵

    Returns:
        CKA ∈ [0, 1]，越大越相似
    """
    n = X.shape[0]
    assert Y.shape[0] == n, "样本数必须一致"

    H = _centering_matrix(n)
    K = X @ X.T   # (n, n)
    L = Y @ Y.T   # (n, n)

    hsic_kl = _hsic(K, L, H)
    hsic_kk = _hsic(K, K, H)
    hsic_ll = _hsic(L, L, H)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


def compute_cka_matrix(
    features: Dict[str, torch.Tensor],
    model_names: list,
) -> np.ndarray:
    """
    计算所有模型对的 CKA 相似度矩阵。

    Args:
        features:    {model_name: (N, feat_dim)} 的特征字典
        model_names: 模型名称列表，确定矩阵顺序

    Returns:
        cka_matrix: (N_models, N_models) 的对称矩阵
    """
    N = len(model_names)
    cka_matrix = np.zeros((N, N))

    # 转为 numpy
    feat_np = {name: features[name].numpy() for name in model_names}

    for i in range(N):
        cka_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            cka_val = linear_cka(feat_np[model_names[i]], feat_np[model_names[j]])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            print(f"    CKA({model_names[i]}, {model_names[j]}) = {cka_val:.4f}")

    return cka_matrix

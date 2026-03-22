"""SVCCA: Singular Vector Canonical Correlation Analysis.

Implements the SVCCA representation similarity metric from:
    Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis
    for Deep Learning Dynamics and Interpretability", NeurIPS 2017.

SVCCA first applies PCA to denoise each representation, then computes
CCA between the reduced representations. The similarity is the mean
canonical correlation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

from ._torch_backend import run_with_fallback


def svcca_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    pca_threshold: float = 0.99,
    pca_max_dim: int = 128,
) -> float:
    """Compute SVCCA similarity between two feature matrices.

    Args:
        X: [N, d1] feature matrix.
        Y: [N, d2] feature matrix.
        pca_threshold: Fraction of variance to retain in PCA step.
        pca_max_dim: Maximum PCA dimensions to keep.

    Returns:
        SVCCA similarity in [0, 1] (mean canonical correlation).
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        Yt = torch.as_tensor(Y, dtype=dtype, device=device)
        assert Xt.shape[0] == Yt.shape[0]

        Xt = Xt - Xt.mean(dim=0, keepdim=True)
        Yt = Yt - Yt.mean(dim=0, keepdim=True)

        X_pca = _pca_reduce_torch(Xt, pca_threshold, pca_max_dim)
        Y_pca = _pca_reduce_torch(Yt, pca_threshold, pca_max_dim)
        correlations = _cca_torch(X_pca, Y_pca)
        if correlations.numel() == 0:
            return 0.0
        return float(correlations.mean().item())

    return run_with_fallback(_impl)


def _pca_reduce(
    X: np.ndarray,
    threshold: float,
    max_dim: int,
) -> np.ndarray:
    """Reduce dimensionality via PCA, keeping enough components to
    explain `threshold` fraction of variance."""
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / max(var.sum(), 1e-10)

    # Find number of components
    k = int(np.searchsorted(cumvar, threshold) + 1)
    k = min(k, max_dim, len(S))
    k = max(k, 1)

    return U[:, :k] * S[:k]


def _cca(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute canonical correlations between X and Y."""
    n = X.shape[0]
    d1 = X.shape[1]
    d2 = Y.shape[1]
    k = min(d1, d2, n)

    if k == 0:
        return np.array([])

    # QR decomposition for numerical stability
    Q1, _ = np.linalg.qr(X)
    Q2, _ = np.linalg.qr(Y)

    Q1 = Q1[:, :d1]
    Q2 = Q2[:, :d2]

    # SVD of Q1^T Q2
    _, S, _ = np.linalg.svd(Q1.T @ Q2, full_matrices=False)
    correlations = np.clip(S[:k], 0.0, 1.0)

    return correlations


def svcca_pairwise_matrix(
    features: Dict[str, np.ndarray],
    pca_threshold: float = 0.99,
    pca_max_dim: int = 128,
) -> Tuple[List[str], np.ndarray]:
    """Compute pairwise SVCCA similarity matrix.

    Args:
        features: {model_name: [N, d] feature array}.
        pca_threshold: Variance threshold for PCA step.
        pca_max_dim: Max PCA dimensions.

    Returns:
        (model_names, svcca_matrix) where svcca_matrix is [M, M].
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> Tuple[List[str], np.ndarray]:
        names = list(features.keys())
        reduced = {}
        for name, feat in features.items():
            tensor = torch.as_tensor(feat, dtype=dtype, device=device)
            tensor = tensor - tensor.mean(dim=0, keepdim=True)
            reduced[name] = _pca_reduce_torch(tensor, pca_threshold, pca_max_dim)

        M = len(names)
        matrix = np.zeros((M, M), dtype=np.float64)
        for i in range(M):
            matrix[i, i] = 1.0
            for j in range(i + 1, M):
                corr = _cca_torch(reduced[names[i]], reduced[names[j]])
                val = 0.0 if corr.numel() == 0 else float(corr.mean().item())
                matrix[i, j] = val
                matrix[j, i] = val
        return names, matrix

    return run_with_fallback(_impl)


def _pca_reduce_torch(
    X: torch.Tensor,
    threshold: float,
    max_dim: int,
) -> torch.Tensor:
    """Torch PCA reduction mirroring the NumPy implementation."""
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    var = S.square()
    cumvar = torch.cumsum(var, dim=0) / torch.clamp(var.sum(), min=1e-10)
    k = int(torch.searchsorted(cumvar, torch.tensor(threshold, dtype=X.dtype, device=X.device)).item() + 1)
    k = min(k, max_dim, int(S.numel()))
    k = max(k, 1)
    return U[:, :k] * S[:k]


def _cca_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Torch CCA implementation."""
    n = int(X.shape[0])
    d1 = int(X.shape[1])
    d2 = int(Y.shape[1])
    k = min(d1, d2, n)
    if k == 0:
        return torch.empty(0, dtype=X.dtype, device=X.device)

    Q1, _ = torch.linalg.qr(X, mode="reduced")
    Q2, _ = torch.linalg.qr(Y, mode="reduced")
    _, S, _ = torch.linalg.svd(Q1.transpose(0, 1) @ Q2, full_matrices=False)
    return torch.clamp(S[:k], min=0.0, max=1.0)

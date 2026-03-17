"""SVCCA: Singular Vector Canonical Correlation Analysis.

Implements the SVCCA representation similarity metric from:
    Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis
    for Deep Learning Dynamics and Interpretability", NeurIPS 2017.

SVCCA first applies PCA to denoise each representation, then computes
CCA between the reduced representations. The similarity is the mean
canonical correlation.
"""

import numpy as np
from typing import Dict, List, Tuple


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
    assert X.shape[0] == Y.shape[0]

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # PCA denoising
    X_pca = _pca_reduce(X, pca_threshold, pca_max_dim)
    Y_pca = _pca_reduce(Y, pca_threshold, pca_max_dim)

    # CCA
    correlations = _cca(X_pca, Y_pca)

    if len(correlations) == 0:
        return 0.0

    return float(np.mean(correlations))


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
    names = list(features.keys())
    M = len(names)
    matrix = np.zeros((M, M))

    for i in range(M):
        matrix[i, i] = 1.0
        for j in range(i + 1, M):
            val = svcca_similarity(
                features[names[i]], features[names[j]],
                pca_threshold=pca_threshold, pca_max_dim=pca_max_dim,
            )
            matrix[i, j] = val
            matrix[j, i] = val

    return names, matrix

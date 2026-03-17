"""CKA: Centered Kernel Alignment for representation redundancy measurement.

Provides linear CKA computation and pairwise CKA matrix construction.
Used as the default redundancy metric in the model selection framework.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
"""

import numpy as np
from typing import Dict, List, Tuple


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two feature matrices.

    Automatically selects feature-space or kernel-space formulation
    based on dimensions.

    Args:
        X: [N, d1] feature matrix.
        Y: [N, d2] feature matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    n = X.shape[0]

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    d_max = max(X.shape[1], Y.shape[1])
    if d_max <= n:
        # Feature-space
        YtX = Y.T @ X
        XtX = X.T @ X
        YtY = Y.T @ Y
        numerator = (YtX * YtX).sum()
        denominator = np.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())
    else:
        # Kernel-space
        K = X @ X.T
        L = Y @ Y.T
        H = np.eye(n) - 1.0 / n
        K = H @ K @ H
        L = H @ L @ H
        numerator = (K * L).sum()
        denominator = np.sqrt((K * K).sum() * (L * L).sum())

    if denominator < 1e-10:
        return 0.0

    return float(np.clip(numerator / denominator, 0.0, 1.0))


def cka_pairwise_matrix(
    features: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    """Compute pairwise CKA matrix for all model features.

    Args:
        features: {model_name: [N, d] feature array}.

    Returns:
        (model_names, cka_matrix) where cka_matrix is [M, M].
    """
    names = list(features.keys())
    M = len(names)
    matrix = np.zeros((M, M))

    for i in range(M):
        matrix[i, i] = 1.0
        for j in range(i + 1, M):
            val = linear_cka(features[names[i]], features[names[j]])
            matrix[i, j] = val
            matrix[j, i] = val

    return names, matrix

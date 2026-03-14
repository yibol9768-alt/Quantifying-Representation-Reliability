"""Centered Kernel Alignment (CKA) for measuring representation similarity.

Supports both feature-space (d < N) and kernel-space (d > N) computation.
Includes optional PCA projection to handle the d >> N degeneracy problem
when using flattened patch tokens.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two feature matrices.

    Automatically selects the more efficient formulation:
      - Feature-space (d < N): O(N·d^2), computes d×d cross-covariance
      - Kernel-space  (d > N): O(N^2·d), computes N×N Gram matrices

    Args:
        X: Feature matrix [N, d1].
        Y: Feature matrix [N, d2].

    Returns:
        CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    n = X.shape[0]

    # Center features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    d_max = max(X.shape[1], Y.shape[1])
    if d_max <= n:
        # Feature-space: CKA = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
        YtX = Y.T @ X
        XtX = X.T @ X
        YtY = Y.T @ Y
        numerator = (YtX * YtX).sum()
        denominator = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())
    else:
        # Kernel-space: use N×N Gram matrices
        K = X @ X.T
        L = Y @ Y.T
        # Center Gram matrices: K' = HKH
        H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
        K = H @ K @ H
        L = H @ L @ H
        numerator = (K * L).sum()
        denominator = torch.sqrt((K * K).sum() * (L * L).sum())

    if denominator < 1e-10:
        return 0.0

    return (numerator / denominator).clamp(0.0, 1.0).item()


def pca_reduce(X: torch.Tensor, n_components: int = 256) -> torch.Tensor:
    """Reduce dimensionality via PCA (truncated SVD on centered data).

    Args:
        X: [N, d] feature matrix (will be centered).
        n_components: Target dimensionality.

    Returns:
        [N, n_components] projected features.
    """
    if X.shape[1] <= n_components:
        return X

    X = X - X.mean(dim=0, keepdim=True)
    # Use SVD: X = U S V^T, projected = U[:, :k] * S[:k]
    U, S, _ = torch.svd_lowrank(X, q=n_components)
    return U * S.unsqueeze(0)


def compute_cka_matrix(
    features_dict: Dict[str, torch.Tensor],
    pca_dim: Optional[int] = 256,
) -> Tuple[list, np.ndarray]:
    """Compute pairwise CKA matrix for all models.

    Args:
        features_dict: {model_name: [N, dim] tensor}.
            All tensors must share the same N.
        pca_dim: If set and dim > pca_dim, apply PCA reduction first.
            This avoids the d >> N degeneracy of linear CKA.
            Set to None to skip PCA.

    Returns:
        (model_names, cka_matrix) where cka_matrix is [M, M] numpy array.
    """
    model_names = list(features_dict.keys())
    n_models = len(model_names)
    cka_matrix = np.zeros((n_models, n_models))

    # Prepare features: CPU float32, optional PCA
    features = {}
    for name, feat in features_dict.items():
        f = feat.float().cpu()
        if pca_dim is not None and f.shape[1] > pca_dim:
            orig_dim = f.shape[1]
            f = pca_reduce(f, pca_dim)
            print(f"    PCA: {name} {orig_dim} -> {f.shape[1]}")
        features[name] = f

    for i in range(n_models):
        cka_matrix[i, i] = 1.0
        for j in range(i + 1, n_models):
            cka_val = linear_CKA(features[model_names[i]], features[model_names[j]])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            print(f"    CKA({model_names[i]}, {model_names[j]}) = {cka_val:.4f}")

    return model_names, cka_matrix

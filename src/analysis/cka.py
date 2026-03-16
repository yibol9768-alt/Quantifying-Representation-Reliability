"""Centered Kernel Alignment (CKA) for measuring representation similarity.

Supports both feature-space (d < N) and kernel-space (d > N) computation.
Includes optional PCA projection to handle the d >> N degeneracy problem
when using flattened patch tokens.

Also provides a class-conditional CKA variant that averages per-class CKA
values. This is useful as a label-conditioned similarity diagnostic when one
wants to inspect whether within-class representation alignment changes across
model pairs, without claiming to estimate high-dimensional conditional mutual
information.
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


def _prepare_features(
    features_dict: Dict[str, torch.Tensor],
    pca_dim: Optional[int] = 256,
) -> Tuple[list, Dict[str, torch.Tensor]]:
    """Convert features to CPU float32 and apply optional PCA once per model."""
    model_names = list(features_dict.keys())
    features = {}
    for name, feat in features_dict.items():
        f = feat.float().cpu()
        if pca_dim is not None and f.shape[1] > pca_dim:
            orig_dim = f.shape[1]
            f = pca_reduce(f, pca_dim)
            print(f"    PCA: {name} {orig_dim} -> {f.shape[1]}")
        features[name] = f
    return model_names, features


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
    model_names, features = _prepare_features(features_dict, pca_dim=pca_dim)
    n_models = len(model_names)
    cka_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        cka_matrix[i, i] = 1.0
        for j in range(i + 1, n_models):
            cka_val = linear_CKA(features[model_names[i]], features[model_names[j]])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            print(f"    CKA({model_names[i]}, {model_names[j]}) = {cka_val:.4f}")

    return model_names, cka_matrix


def compute_class_conditional_cka_matrix(
    features_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    pca_dim: Optional[int] = 256,
    min_class_samples: int = 2,
) -> Tuple[list, np.ndarray, dict]:
    """Compute class-conditional CKA by averaging per-class CKA values.

    For discrete labels Y, this function computes a weighted class-wise average
    E_y[ CKA(F_i^(y), F_j^(y)) ], where F^(y) denotes the subset of features
    whose labels equal y. The weighting is empirical class frequency among
    classes with enough samples.

    Important: this is only a label-conditioned similarity diagnostic. It is
    not an estimator of I(F_i; F_j | Y), nor should it be used as such in
    theoretical claims.

    Args:
        features_dict: {model_name: [N, dim] tensor}. All tensors share labels.
        labels: [N] integer labels aligned with feature rows.
        pca_dim: Optional PCA dimension applied once per model before slicing.
        min_class_samples: Minimum samples needed for a class to contribute.

    Returns:
        (model_names, ccka_matrix, metadata)
    """
    labels = labels.view(-1).cpu()
    model_names, features = _prepare_features(features_dict, pca_dim=pca_dim)
    n_models = len(model_names)
    ccka_matrix = np.zeros((n_models, n_models))

    if not features:
        raise ValueError("features_dict must be non-empty")
    n_samples = next(iter(features.values())).shape[0]
    if labels.shape[0] != n_samples:
        raise ValueError("labels must align with features")

    labels_np = labels.numpy()
    unique_labels, counts = np.unique(labels_np, return_counts=True)

    class_indices = {}
    skipped_classes = []
    valid_class_counts = {}
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        if count < min_class_samples:
            skipped_classes.append(int(label))
            continue
        idx = torch.from_numpy(np.where(labels_np == label)[0]).long()
        class_indices[int(label)] = idx
        valid_class_counts[int(label)] = int(count)

    if not class_indices:
        raise ValueError(
            f"No classes have at least {min_class_samples} samples; "
            "cannot compute class-conditional CKA."
        )

    total_weight = float(sum(valid_class_counts.values()))

    for i in range(n_models):
        ccka_matrix[i, i] = 1.0
        for j in range(i + 1, n_models):
            weighted_sum = 0.0
            for label, idx in class_indices.items():
                weight = valid_class_counts[label] / total_weight
                cka_val = linear_CKA(
                    features[model_names[i]].index_select(0, idx),
                    features[model_names[j]].index_select(0, idx),
                )
                weighted_sum += weight * cka_val

            ccka_matrix[i, j] = weighted_sum
            ccka_matrix[j, i] = weighted_sum
            print(
                f"    ccCKA({model_names[i]}, {model_names[j]}) = "
                f"{weighted_sum:.4f}"
            )

    metadata = {
        "num_total_samples": int(n_samples),
        "num_total_classes": int(len(unique_labels)),
        "num_valid_classes": int(len(class_indices)),
        "num_skipped_classes": int(len(skipped_classes)),
        "valid_sample_coverage": total_weight / float(n_samples),
        "min_class_samples": int(min_class_samples),
        "valid_class_counts": valid_class_counts,
        "skipped_classes": skipped_classes,
    }
    return model_names, ccka_matrix, metadata

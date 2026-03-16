"""Pairwise conditional mutual information utilities.

This module implements a practical low-order approximation to the third term
in the exact decomposition

    I(F_m; Y | F_S) = I(F_m; Y) - I(F_m; F_S) + I(F_m; F_S | Y).

Directly estimating the set-level term I(F_m; F_S | Y) is hard in our
high-dimensional feature setting. Following the low-order approximation spirit
of Brown et al. (JMLR 2012), we instead estimate pairwise conditional terms

    avg_{j in S} I(F_m; F_j | Y).

For tractability, the estimator below assumes class-conditional Gaussianity
after PCA reduction, and computes

    I(F_i; F_j | Y) = sum_y p(y) I(F_i; F_j | Y=y),

where each within-class mutual information is estimated from regularized
Gaussian covariance matrices.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .cka import pca_reduce


def _prepare_features(
    features_dict: Dict[str, torch.Tensor],
    pca_dim: Optional[int],
) -> Tuple[list, Dict[str, np.ndarray]]:
    """Convert features to float64 numpy arrays with optional PCA reduction."""
    model_names = list(features_dict.keys())
    prepared = {}
    for name, feat in features_dict.items():
        x = feat.float().cpu()
        if pca_dim is not None and x.shape[1] > pca_dim:
            orig_dim = x.shape[1]
            x = pca_reduce(x, pca_dim)
            print(f"    CMI PCA: {name} {orig_dim} -> {x.shape[1]}")
        prepared[name] = x.numpy().astype(np.float64, copy=False)
    return model_names, prepared


def _regularized_covariance(X: np.ndarray, reg: float) -> np.ndarray:
    """Estimate a covariance matrix with scale-aware diagonal regularization."""
    n, d = X.shape
    if n <= 1:
        raise ValueError("Need at least 2 samples to estimate covariance.")

    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / float(max(n - 1, 1))

    scale = np.trace(cov) / float(d) if d > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov = cov + (reg * scale) * np.eye(d, dtype=np.float64)
    return cov


def _stable_logdet(cov: np.ndarray, reg: float) -> float:
    """Compute a robust log-determinant, increasing ridge if needed."""
    base = cov
    eye = np.eye(base.shape[0], dtype=np.float64)
    jitter = 0.0
    for _ in range(6):
        sign, logdet = np.linalg.slogdet(base + jitter * eye)
        if sign > 0 and np.isfinite(logdet):
            return float(logdet)
        jitter = reg if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Failed to obtain a positive definite covariance.")


def gaussian_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    reg: float = 1e-3,
) -> float:
    """Estimate I(X; Y) under a Gaussian assumption with ridge covariance."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    cov_x = _regularized_covariance(X, reg=reg)
    cov_y = _regularized_covariance(Y, reg=reg)
    cov_xy = _regularized_covariance(np.concatenate([X, Y], axis=1), reg=reg)

    mi = 0.5 * (
        _stable_logdet(cov_x, reg=reg)
        + _stable_logdet(cov_y, reg=reg)
        - _stable_logdet(cov_xy, reg=reg)
    )
    return float(max(0.0, mi))


def compute_pairwise_class_conditional_mi_matrix(
    features_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    pca_dim: Optional[int] = 32,
    min_class_samples: int = 8,
    reg: float = 1e-3,
) -> Tuple[list, np.ndarray, np.ndarray, dict]:
    """Estimate pairwise I(F_i; F_j | Y) under class-conditional Gaussianity.

    Args:
        features_dict: {model_name: [N, d] feature tensors}.
        labels: [N] integer labels aligned with feature rows.
        pca_dim: PCA target dimension before conditional MI estimation.
        min_class_samples: Minimum samples per class to contribute.
        reg: Diagonal covariance regularization strength.

    Returns:
        (model_names, raw_matrix, norm_matrix, metadata)
    """
    labels = labels.view(-1).cpu().numpy()
    model_names, features = _prepare_features(features_dict, pca_dim=pca_dim)
    n_models = len(model_names)
    raw = np.zeros((n_models, n_models), dtype=np.float64)

    if not features:
        raise ValueError("features_dict must be non-empty")
    n_samples = next(iter(features.values())).shape[0]
    if labels.shape[0] != n_samples:
        raise ValueError("labels must align with feature rows.")

    unique_labels, counts = np.unique(labels, return_counts=True)
    class_indices = {}
    valid_counts = {}
    skipped_classes = []
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        if count < min_class_samples:
            skipped_classes.append(int(label))
            continue
        class_indices[int(label)] = np.where(labels == label)[0]
        valid_counts[int(label)] = int(count)

    if not class_indices:
        raise ValueError(
            f"No classes have at least {min_class_samples} samples; "
            "cannot estimate class-conditional MI."
        )

    total_weight = float(sum(valid_counts.values()))
    for i in range(n_models):
        raw[i, i] = 0.0
        for j in range(i + 1, n_models):
            score = 0.0
            for label, idx in class_indices.items():
                weight = valid_counts[label] / total_weight
                x = features[model_names[i]][idx]
                y = features[model_names[j]][idx]
                score += weight * gaussian_mutual_information(x, y, reg=reg)

            raw[i, j] = score
            raw[j, i] = score
            print(
                f"    PCMI({model_names[i]}, {model_names[j]}) = {score:.4f}"
            )

    # Min-max normalize off-diagonal entries for scoring compatibility.
    norm = raw.copy()
    if n_models > 1:
        tri = raw[np.triu_indices(n_models, k=1)]
        lo = float(np.min(tri))
        hi = float(np.max(tri))
        if hi > lo:
            norm = (raw - lo) / (hi - lo)
        else:
            norm = np.zeros_like(raw)
        np.fill_diagonal(norm, 0.0)

    metadata = {
        "num_total_samples": int(n_samples),
        "num_total_classes": int(len(unique_labels)),
        "num_valid_classes": int(len(class_indices)),
        "num_skipped_classes": int(len(skipped_classes)),
        "valid_sample_coverage": total_weight / float(n_samples),
        "min_class_samples": int(min_class_samples),
        "pca_dim": None if pca_dim is None else int(pca_dim),
        "reg": float(reg),
        "valid_class_counts": valid_counts,
        "skipped_classes": skipped_classes,
        "raw_offdiag_min": float(np.min(tri)) if n_models > 1 else 0.0,
        "raw_offdiag_max": float(np.max(tri)) if n_models > 1 else 0.0,
    }
    return model_names, raw, norm, metadata

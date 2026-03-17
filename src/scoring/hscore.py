"""H-Score: An information-theoretic transferability metric.

Implements the H-Score from:
    Bao et al., "An Information-Theoretic Approach to Transferability in
    Task Transfer Learning", ICML 2019.

H-Score measures transferability by comparing the inter-class variance
of features to total feature redundancy:
    H(f) = tr(cov(f)^{-1} * cov(E[f|y]))
where cov(f) is the total feature covariance and cov(E[f|y]) is the
covariance of class-conditional means.
"""

import numpy as np


def hscore(
    features: np.ndarray,
    labels: np.ndarray,
    reg: float = 1e-3,
) -> float:
    """Compute the H-Score transferability metric.

    Args:
        features: [N, d] feature matrix.
        labels: [N] integer class labels.
        reg: Regularization for covariance inversion.

    Returns:
        H-Score (higher = better transferability).
    """
    features = features.astype(np.float64)
    N, d = features.shape
    classes = np.unique(labels)
    C = len(classes)

    if C < 2:
        return 0.0

    # Overall feature covariance
    f_centered = features - features.mean(axis=0, keepdims=True)
    cov_f = (f_centered.T @ f_centered) / max(N - 1, 1)
    cov_f += reg * np.eye(d)

    # Class-conditional means
    class_means = np.zeros((C, d), dtype=np.float64)
    class_weights = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = labels == c
        class_means[i] = features[mask].mean(axis=0)
        class_weights[i] = mask.sum() / N

    # Covariance of class-conditional means (between-class covariance)
    weighted_mean = (class_weights[:, None] * class_means).sum(axis=0)
    diff = class_means - weighted_mean
    cov_between = (class_weights[:, None] * diff).T @ diff

    # H-Score = tr(cov_f^{-1} @ cov_between)
    try:
        cov_f_inv = np.linalg.inv(cov_f)
    except np.linalg.LinAlgError:
        cov_f_inv = np.linalg.pinv(cov_f)

    score = float(np.trace(cov_f_inv @ cov_between))
    return max(score, 0.0)

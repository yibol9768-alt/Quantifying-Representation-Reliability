"""GBC: Gaussian Bhattacharyya Coefficient for transferability estimation.

Implements the GBC transferability metric from:
    Pandy et al., "Transferability Estimation using Bhattacharyya Class
    Separability", CVPR 2022.

GBC fits a Gaussian distribution per class on the extracted features,
then computes the average pairwise Bhattacharyya distance between all
class pairs. Higher separability implies better transferability.
"""

import numpy as np
from itertools import combinations


def gbc_score(
    features: np.ndarray,
    labels: np.ndarray,
    reg: float = 1e-3,
) -> float:
    """Compute the GBC transferability score.

    Args:
        features: [N, d] feature matrix.
        labels: [N] integer class labels.
        reg: Regularization added to diagonal of covariance matrices.

    Returns:
        GBC score (higher = better class separability = better transferability).
    """
    features = features.astype(np.float64)
    classes = np.unique(labels)
    C = len(classes)

    if C < 2:
        return 0.0

    # Fit per-class Gaussians
    means = {}
    covs = {}
    for c in classes:
        mask = labels == c
        X_c = features[mask]
        n_c = X_c.shape[0]
        if n_c < 2:
            return 0.0
        mu = X_c.mean(axis=0)
        Xc = X_c - mu
        cov = (Xc.T @ Xc) / max(n_c - 1, 1) + reg * np.eye(features.shape[1])
        means[c] = mu
        covs[c] = cov

    # Average pairwise Bhattacharyya distance
    total_dist = 0.0
    n_pairs = 0
    for ci, cj in combinations(classes, 2):
        dist = _bhattacharyya_distance(means[ci], covs[ci], means[cj], covs[cj])
        total_dist += dist
        n_pairs += 1

    return float(total_dist / max(n_pairs, 1))


def _bhattacharyya_distance(
    mu1: np.ndarray,
    cov1: np.ndarray,
    mu2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """Compute Bhattacharyya distance between two Gaussians.

    DB(p, q) = (1/8)(mu1-mu2)^T Sigma^{-1} (mu1-mu2) + (1/2) ln(det(Sigma)/sqrt(det(S1)*det(S2)))
    where Sigma = (cov1 + cov2) / 2.
    """
    diff = mu1 - mu2
    sigma = (cov1 + cov2) / 2.0

    # Use slogdet for numerical stability
    sign_s, logdet_s = np.linalg.slogdet(sigma)
    sign_1, logdet_1 = np.linalg.slogdet(cov1)
    sign_2, logdet_2 = np.linalg.slogdet(cov2)

    if sign_s <= 0 or sign_1 <= 0 or sign_2 <= 0:
        return 0.0

    try:
        sigma_inv = np.linalg.solve(sigma, diff)
    except np.linalg.LinAlgError:
        return 0.0

    term1 = 0.125 * float(diff @ sigma_inv)
    term2 = 0.5 * (logdet_s - 0.5 * (logdet_1 + logdet_2))

    return float(max(term1 + term2, 0.0))

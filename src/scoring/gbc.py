"""GBC: Gaussian Bhattacharyya Coefficient for transferability estimation.

Implements the GBC transferability metric from:
    Pandy et al., "Transferability Estimation using Bhattacharyya Class
    Separability", CVPR 2022.

GBC fits a Gaussian distribution per class on the extracted features,
then computes the average pairwise Bhattacharyya distance between all
class pairs. Higher separability implies better transferability.
"""

from itertools import combinations

import numpy as np
import torch


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
    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp32 on CUDA is much faster and is sufficient for this ranking heuristic.
    dtype = torch.float32 if device.type == "cuda" else torch.float64

    features_t = torch.as_tensor(features, dtype=dtype, device=device)
    labels_t = torch.as_tensor(labels, device=device)

    means = []
    covs = []
    logdets = []
    signs = []
    feat_dim = int(features_t.shape[1])
    eye = torch.eye(feat_dim, dtype=dtype, device=device)

    for c in classes.tolist():
        mask = labels_t == int(c)
        X_c = features_t[mask]
        n_c = int(X_c.shape[0])
        if n_c < 2:
            return 0.0

        mu = X_c.mean(dim=0)
        Xc = X_c - mu
        cov = (Xc.transpose(0, 1) @ Xc) / max(n_c - 1, 1)
        cov = cov + reg * eye
        sign, logdet = torch.linalg.slogdet(cov)

        means.append(mu)
        covs.append(cov)
        signs.append(sign)
        logdets.append(logdet)

    means_t = torch.stack(means, dim=0)
    covs_t = torch.stack(covs, dim=0)
    signs_t = torch.stack(signs, dim=0)
    logdets_t = torch.stack(logdets, dim=0)

    pair_indices = list(combinations(range(len(classes)), 2))
    if not pair_indices:
        return 0.0

    pairs_i = torch.tensor([i for i, _ in pair_indices], device=device, dtype=torch.long)
    pairs_j = torch.tensor([j for _, j in pair_indices], device=device, dtype=torch.long)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    # Keep the batched sigma tensors bounded; high-d models like ResNet need small batches.
    max_sigma_bytes = 64 * 1024 * 1024
    batch_size = max(1, max_sigma_bytes // max(feat_dim * feat_dim * elem_size, 1))

    total = torch.zeros((), dtype=dtype, device=device)
    valid_pairs = 0

    for start in range(0, len(pair_indices), batch_size):
        end = min(start + batch_size, len(pair_indices))
        idx_i = pairs_i[start:end]
        idx_j = pairs_j[start:end]

        cov_i = covs_t[idx_i]
        cov_j = covs_t[idx_j]
        sigma = 0.5 * (cov_i + cov_j)

        sign_s, logdet_s = torch.linalg.slogdet(sigma)
        valid = (sign_s > 0) & (signs_t[idx_i] > 0) & (signs_t[idx_j] > 0)
        if not torch.any(valid):
            continue

        valid_idx_i = idx_i[valid]
        valid_idx_j = idx_j[valid]
        sigma_valid = sigma[valid]
        diff_valid = means_t[valid_idx_i] - means_t[valid_idx_j]

        try:
            sigma_inv_diff = torch.linalg.solve(
                sigma_valid,
                diff_valid.unsqueeze(-1),
            ).squeeze(-1)
        except RuntimeError:
            # Fall back to CPU if the CUDA solve kernel becomes unstable.
            return _gbc_score_cpu(features, labels, reg=reg)

        term1 = 0.125 * (diff_valid * sigma_inv_diff).sum(dim=1)
        term2 = 0.5 * (
            logdet_s[valid]
            - 0.5 * (logdets_t[valid_idx_i] + logdets_t[valid_idx_j])
        )
        distances = torch.clamp(term1 + term2, min=0.0)
        total = total + distances.sum()
        valid_pairs += int(distances.numel())

    if valid_pairs == 0:
        return 0.0
    return float((total / valid_pairs).item())


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


def _gbc_score_cpu(
    features: np.ndarray,
    labels: np.ndarray,
    reg: float = 1e-3,
) -> float:
    """Reference NumPy implementation used as a safe fallback."""
    features = features.astype(np.float64)
    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0

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

    total_dist = 0.0
    n_pairs = 0
    for ci, cj in combinations(classes, 2):
        total_dist += _bhattacharyya_distance(means[ci], covs[ci], means[cj], covs[cj])
        n_pairs += 1

    return float(total_dist / max(n_pairs, 1))

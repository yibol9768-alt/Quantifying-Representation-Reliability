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
import torch

from ._torch_backend import run_with_fallback


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
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        feats = torch.as_tensor(features, dtype=dtype, device=device)
        labels_t = torch.as_tensor(labels, device=device)
        N, d = feats.shape
        classes = torch.unique(labels_t)
        C = int(classes.numel())

        if C < 2:
            return 0.0

        f_centered = feats - feats.mean(dim=0, keepdim=True)
        cov_f = (f_centered.transpose(0, 1) @ f_centered) / max(N - 1, 1)
        cov_f = cov_f + reg * torch.eye(d, dtype=dtype, device=device)

        class_means = []
        class_weights = []
        for c in classes:
            mask = labels_t == c
            class_means.append(feats[mask].mean(dim=0))
            class_weights.append(mask.sum().to(dtype) / N)

        class_means_t = torch.stack(class_means, dim=0)
        class_weights_t = torch.stack(class_weights, dim=0)
        weighted_mean = (class_weights_t[:, None] * class_means_t).sum(dim=0)
        diff = class_means_t - weighted_mean
        cov_between = (class_weights_t[:, None] * diff).transpose(0, 1) @ diff

        try:
            solved = torch.linalg.solve(cov_f, cov_between)
        except RuntimeError:
            solved = torch.linalg.pinv(cov_f) @ cov_between

        score = torch.trace(solved)
        return float(torch.clamp(score, min=0.0).item())

    return run_with_fallback(_impl)

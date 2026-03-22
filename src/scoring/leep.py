"""LEEP: Log Expected Empirical Prediction.

Implements the LEEP transferability metric from:
    Nguyen et al., "LEEP: A New Measure to Evaluate Transferability of
    Learned Representations", ICML 2020.

LEEP requires the source model's softmax probability outputs on the
target dataset. It computes the expected log-likelihood of target labels
under an empirical joint distribution P(y, z) derived from the source
predictions z and target labels y.
"""

import numpy as np
import torch

from ._torch_backend import run_with_fallback


def leep_score(
    source_probs: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    """Compute the LEEP score.

    Args:
        source_probs: [N, C_source] softmax probability matrix from the
            source model evaluated on the target dataset. Each row sums to 1.
        target_labels: [N] integer target labels in {0, ..., C_target - 1}.

    Returns:
        LEEP score (higher = better transferability). Typically negative.
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        probs = torch.as_tensor(source_probs, dtype=dtype, device=device)
        labels_t = torch.as_tensor(target_labels, device=device)
        N, C_source = probs.shape
        target_classes = torch.unique(labels_t)
        C_target = int(target_classes.numel())

        y_idx = torch.empty_like(labels_t, dtype=torch.long)
        for i, c in enumerate(target_classes):
            y_idx[labels_t == c] = i

        p_z = probs.mean(dim=0)
        p_yz = torch.zeros((C_target, C_source), dtype=dtype, device=device)
        for c_idx in range(C_target):
            mask = y_idx == c_idx
            if torch.any(mask):
                p_yz[c_idx] = probs[mask].mean(dim=0) * mask.to(dtype).mean()

        p_y_given_z = torch.zeros_like(p_yz)
        nonzero = p_z > 1e-12
        p_y_given_z[:, nonzero] = p_yz[:, nonzero] / p_z[nonzero]

        probs_y = p_y_given_z[y_idx]
        sample_probs = torch.sum(probs * probs_y, dim=1).clamp_min(1e-30)
        return float(torch.log(sample_probs).mean().item())

    return run_with_fallback(_impl)

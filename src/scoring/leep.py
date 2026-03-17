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
    N, C_source = source_probs.shape
    target_classes = np.unique(target_labels)
    C_target = len(target_classes)

    # Build label map to contiguous indices
    label_map = {int(c): i for i, c in enumerate(target_classes)}
    y_idx = np.array([label_map[int(l)] for l in target_labels])

    # Empirical joint P(y, z) = (1/N) sum_i 1[y_i = y] * theta_i(z)
    # P(z) = (1/N) sum_i theta_i(z)  [marginal over source classes]
    # P(y | z) = P(y, z) / P(z)

    # Compute empirical marginal P(z): [C_source]
    p_z = source_probs.mean(axis=0)  # [C_source]

    # Compute P(y, z): [C_target, C_source]
    p_yz = np.zeros((C_target, C_source), dtype=np.float64)
    for c_idx in range(C_target):
        mask = y_idx == c_idx
        if mask.any():
            p_yz[c_idx] = source_probs[mask].mean(axis=0) * mask.mean()

    # P(y | z) = P(y, z) / P(z), with safe division
    p_y_given_z = np.zeros_like(p_yz)
    nonzero = p_z > 1e-12
    p_y_given_z[:, nonzero] = p_yz[:, nonzero] / p_z[nonzero]

    # LEEP = (1/N) sum_i log( sum_z theta_i(z) * P(y_i | z) )
    # For each sample i, compute sum_z theta_i(z) * P(y_i | z)
    leep = 0.0
    for i in range(N):
        yi = y_idx[i]
        # sum over source classes z: theta_i(z) * P(y_i | z)
        prob = float(source_probs[i] @ p_y_given_z[yi])
        prob = max(prob, 1e-30)
        leep += np.log(prob)

    return float(leep / N)

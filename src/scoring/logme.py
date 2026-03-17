"""LogME: Log Maximum Evidence for transferability estimation.

Implements the LogME algorithm from:
    You et al., "LogME: Practical Assessment of Pre-trained Models for
    Transfer Learning", ICML 2021.

Given extracted features F [N, d] and labels y [N], LogME computes the
log marginal likelihood of a Bayesian linear regression model, which
serves as a training-free proxy for transfer performance.
"""

import numpy as np


def logme_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute the LogME score for a classification task.

    Args:
        features: [N, d] feature matrix (float).
        labels: [N] integer class labels.

    Returns:
        LogME score (higher = better transferability).
    """
    N, d = features.shape
    classes = np.unique(labels)
    C = len(classes)

    # One-hot encode: [N, C]
    Y = np.zeros((N, C), dtype=np.float64)
    for i, c in enumerate(classes):
        Y[labels == c, i] = 1.0

    # SVD of features (shared across label columns)
    U, S, _ = np.linalg.svd(features.astype(np.float64), full_matrices=False)
    S2 = S ** 2

    evidences = []
    for c in range(C):
        ev = _evidence_regression(U, S2, Y[:, c], N)
        evidences.append(ev)

    return float(np.mean(evidences))


def _evidence_regression(
    U: np.ndarray,
    S2: np.ndarray,
    y: np.ndarray,
    N: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> float:
    """Compute log evidence for a single regression target via fixed-point iteration."""
    k = len(S2)
    Uty = U.T @ y
    Uty2 = Uty ** 2
    yty = float(y @ y)

    alpha = 1.0
    beta = 1.0
    log_ev_old = -np.inf

    for _ in range(max_iter):
        denom = alpha + beta * S2
        gamma = float(np.sum(beta * S2 / denom))
        m_norm2 = float(np.sum(beta ** 2 * S2 * Uty2 / denom ** 2))
        rss = yty - float(np.sum(beta * S2 * Uty2 / denom))
        rss = max(rss, 1e-10)

        alpha_new = max(float(gamma / (m_norm2 + 1e-10)), 1e-10)
        beta_new = max(float((N - gamma) / rss), 1e-10)

        log_ev = 0.5 * (
            k * np.log(alpha_new)
            + N * np.log(beta_new)
            - beta_new * rss
            - alpha_new * m_norm2
            - float(np.sum(np.log(alpha_new + beta_new * S2)))
            - N * np.log(2.0 * np.pi)
        )

        if abs(log_ev - log_ev_old) < tol:
            return float(log_ev)

        alpha = alpha_new
        beta = beta_new
        log_ev_old = log_ev

    return float(log_ev_old)

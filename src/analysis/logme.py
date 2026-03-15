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
    # F = U S V^T, where U [N, k], S [k], V [d, k], k = min(N, d)
    U, S, _ = np.linalg.svd(features.astype(np.float64), full_matrices=False)
    S2 = S ** 2  # [k]

    # Compute log evidence for each label column, then average
    evidences = []
    for c in range(C):
        y_c = Y[:, c]
        ev = _evidence_regression(U, S2, y_c, N)
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
    """Compute log evidence for a single regression target via fixed-point iteration.

    Uses the SVD-based formulation to efficiently solve the Bayesian
    linear regression evidence maximization over hyperparameters alpha, beta.

    Args:
        U: [N, k] left singular vectors of F.
        S2: [k] squared singular values of F.
        y: [N] target vector (one column of one-hot matrix).
        N: number of samples.
        max_iter: maximum fixed-point iterations.
        tol: convergence tolerance for log evidence change.

    Returns:
        Maximized log evidence (scalar).
    """
    k = len(S2)

    # Precompute U^T y
    Uty = U.T @ y  # [k]
    Uty2 = Uty ** 2  # [k]
    yty = float(y @ y)

    # Initialize alpha, beta
    alpha = 1.0
    beta = 1.0

    log_ev_old = -np.inf

    for _ in range(max_iter):
        # Effective number of well-determined parameters
        # gamma_i = beta * s_i^2 / (alpha + beta * s_i^2)
        # gamma = sum(gamma_i)
        denom = alpha + beta * S2  # [k]
        gamma = float(np.sum(beta * S2 / denom))

        # Posterior mean contribution in SVD space:
        # m_hat = beta * diag(1/denom) * S * U^T y
        # ||m_hat||^2 = beta^2 * sum(s_i^2 * (U^T y)_i^2 / denom_i^2)
        m_norm2 = float(np.sum(beta ** 2 * S2 * Uty2 / denom ** 2))

        # Residual sum of squares in SVD form:
        # RSS = ||y - F m_hat||^2 = y^T y - beta * sum(s_i^2 * (U^T y)_i^2 / denom_i)
        rss = yty - float(np.sum(beta * S2 * Uty2 / denom))
        rss = max(rss, 1e-10)

        # Update alpha and beta
        alpha_new = float(gamma / (m_norm2 + 1e-10))
        beta_new = float((N - gamma) / rss)

        # Ensure positivity
        alpha_new = max(alpha_new, 1e-10)
        beta_new = max(beta_new, 1e-10)

        # Compute log evidence:
        # log p(y|alpha,beta) = 0.5 * [k*log(alpha) + N*log(beta)
        #   - beta*RSS - alpha*||m||^2 - sum(log(alpha + beta*s_i^2))
        #   - (N - k)*log(beta) - N*log(2*pi)]
        # Simplified:
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

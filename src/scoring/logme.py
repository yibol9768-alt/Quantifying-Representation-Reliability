"""LogME: Log Maximum Evidence for transferability estimation.

Implements the LogME algorithm from:
    You et al., "LogME: Practical Assessment of Pre-trained Models for
    Transfer Learning", ICML 2021.

Given extracted features F [N, d] and labels y [N], LogME computes the
log marginal likelihood of a Bayesian linear regression model, which
serves as a training-free proxy for transfer performance.
"""

import numpy as np
import torch

from ._torch_backend import run_with_fallback


def logme_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute the LogME score for a classification task.

    Args:
        features: [N, d] feature matrix (float).
        labels: [N] integer class labels.

    Returns:
        LogME score (higher = better transferability).
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        feats = torch.as_tensor(features, dtype=dtype, device=device)
        labels_t = torch.as_tensor(labels, device=device)
        N = int(feats.shape[0])
        classes = torch.unique(labels_t)
        C = int(classes.numel())

        Y = torch.zeros((N, C), dtype=dtype, device=device)
        for i, c in enumerate(classes):
            Y[:, i] = (labels_t == c).to(dtype)

        U, S, _ = torch.linalg.svd(feats, full_matrices=False)
        S2 = S.square()

        evidences = []
        for c in range(C):
            evidences.append(_evidence_regression_torch(U, S2, Y[:, c], N, dtype))

        return float(torch.stack(evidences).mean().item())

    return run_with_fallback(_impl)


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


def _evidence_regression_torch(
    U: torch.Tensor,
    S2: torch.Tensor,
    y: torch.Tensor,
    N: int,
    dtype: torch.dtype,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Torch implementation of fixed-point evidence iteration."""
    k = int(S2.numel())
    Uty = U.transpose(0, 1) @ y
    Uty2 = Uty.square()
    yty = y @ y

    alpha = torch.tensor(1.0, dtype=dtype, device=U.device)
    beta = torch.tensor(1.0, dtype=dtype, device=U.device)
    log_ev_old = torch.tensor(float("-inf"), dtype=dtype, device=U.device)
    eps = torch.tensor(1e-10, dtype=dtype, device=U.device)
    two_pi = torch.tensor(2.0 * np.pi, dtype=dtype, device=U.device)

    for _ in range(max_iter):
        denom = alpha + beta * S2
        gamma = torch.sum(beta * S2 / denom)
        m_norm2 = torch.sum(beta.square() * S2 * Uty2 / denom.square())
        rss = yty - torch.sum(beta * S2 * Uty2 / denom)
        rss = torch.clamp(rss, min=eps)

        alpha_new = torch.clamp(gamma / (m_norm2 + eps), min=eps)
        beta_new = torch.clamp((N - gamma) / rss, min=eps)

        log_ev = 0.5 * (
            k * torch.log(alpha_new)
            + N * torch.log(beta_new)
            - beta_new * rss
            - alpha_new * m_norm2
            - torch.sum(torch.log(alpha_new + beta_new * S2))
            - N * torch.log(two_pi)
        )

        if torch.abs(log_ev - log_ev_old).item() < tol:
            return log_ev

        alpha = alpha_new
        beta = beta_new
        log_ev_old = log_ev

    return log_ev_old

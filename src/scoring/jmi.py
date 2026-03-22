"""JMI: Joint Mutual Information model selection.

Implements the JMI criterion from:
    Brown et al., "Conditional Likelihood Maximisation: A Unifying
    Framework for Information Theoretic Feature Selection", JMLR 2012.

Adapted to the model selection level: each model is a "feature group".
The JMI criterion selects the model maximizing:
    J_jmi(m | S) = sum_{j in S} I(F_m, F_j; Y)
                 = sum_{j in S} [I(F_m; Y) + I(F_j; Y | F_m)]

In practice, we approximate I(F_m, F_j; Y) = I(F_m; Y) + I(F_j; Y) - I(F_j; F_m)
+ I(F_j; F_m | Y), using Gaussian estimators.
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from ._torch_backend import run_with_fallback


def _gaussian_entropy(X: np.ndarray, reg: float = 1e-3) -> float:
    """Gaussian entropy H(X) = 0.5 * log det(2*pi*e * Sigma)."""
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        N, d = Xt.shape
        Xc = Xt - Xt.mean(dim=0, keepdim=True)
        cov = (Xc.transpose(0, 1) @ Xc) / max(N - 1, 1) + reg * torch.eye(d, dtype=dtype, device=device)
        _, logdet = torch.linalg.slogdet(cov)
        const = torch.tensor(2 * np.pi * np.e, dtype=dtype, device=device)
        val = 0.5 * (d * torch.log(const) + logdet)
        return float(val.item())

    return run_with_fallback(_impl)


def _joint_mi_with_label(
    X: np.ndarray,
    Y_feat: np.ndarray,
    labels: np.ndarray,
    reg: float = 1e-3,
) -> float:
    """Estimate I(X, Y_feat; labels) under Gaussian assumptions.

    I(X, Y; Z) = H(X, Y) + H(Z) - H(X, Y, Z)
    where Z is discrete (labels), so we use:
    I(X, Y; Z) = H(X, Y) - H(X, Y | Z)
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        Yt = torch.as_tensor(Y_feat, dtype=dtype, device=device)
        labels_t = torch.as_tensor(labels, device=device)
        N = int(Xt.shape[0])
        XY = torch.cat([Xt, Yt], dim=1)

        h_xy = _gaussian_entropy_torch(XY, reg)
        classes = torch.unique(labels_t)
        h_xy_given_z = torch.zeros((), dtype=dtype, device=device)

        for c in classes:
            mask = labels_t == c
            n_c = int(mask.sum())
            if n_c < 2:
                continue
            p_c = n_c / N
            h_xy_given_z = h_xy_given_z + p_c * _gaussian_entropy_torch(XY[mask], reg)

        return float(torch.clamp(h_xy - h_xy_given_z, min=0.0).item())

    return run_with_fallback(_impl)


def jmi_select(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    max_models: int = 6,
    pca_dim: Optional[int] = 64,
    reg: float = 1e-3,
) -> List[str]:
    """Select models using the JMI criterion.

    At each step, select:
        m* = argmax_{m not in S} sum_{j in S} I(F_m, F_j; Y)

    For the first model, select by maximum I(F_m; Y).

    Args:
        features: {model_name: [N, d] feature array}.
        labels: [N] integer class labels.
        max_models: Maximum number of models to select.
        pca_dim: If set, reduce features via PCA before estimation.
        reg: Covariance regularization.

    Returns:
        Ordered list of selected model names.
    """
    names = list(features.keys())
    labels = np.asarray(labels).ravel()

    # PCA reduction
    feats = {}
    for name, f in features.items():
        f = f.astype(np.float64)
        if pca_dim is not None and f.shape[1] > pca_dim:
            f = _pca(f, pca_dim)
        feats[name] = f

    N = next(iter(feats.values())).shape[0]
    classes = np.unique(labels)

    # Relevance I(F_m; Y) for first-step selection
    relevance = {}
    for name in names:
        h_f = _gaussian_entropy(feats[name], reg)
        h_f_given_y = 0.0
        for c in classes:
            mask = labels == c
            n_c = mask.sum()
            if n_c < 2:
                continue
            p_c = n_c / N
            h_f_given_y += p_c * _gaussian_entropy(feats[name][mask], reg)
        relevance[name] = max(h_f - h_f_given_y, 0.0)

    # First model: highest relevance
    selected = []
    remaining = set(names)
    max_models = min(max_models, len(names))

    first = max(names, key=lambda m: relevance[m])
    selected.append(first)
    remaining.remove(first)

    # Pre-compute pairwise joint MI: I(F_i, F_j; Y)
    jmi_cache = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                val = _joint_mi_with_label(feats[ni], feats[nj], labels, reg)
                jmi_cache[(ni, nj)] = val
                jmi_cache[(nj, ni)] = val

    # Greedy selection
    while len(selected) < max_models and remaining:
        best_model = None
        best_score = -np.inf

        for m in remaining:
            score = sum(jmi_cache.get((m, s), 0.0) for s in selected)
            if score > best_score:
                best_score = score
                best_model = m

        selected.append(best_model)
        remaining.remove(best_model)

    return selected


def _pca(X: np.ndarray, k: int) -> np.ndarray:
    """Simple PCA to k dimensions."""
    def _impl(device: torch.device, dtype: torch.dtype) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        Xt = Xt - Xt.mean(dim=0, keepdim=True)
        U, S, _ = torch.linalg.svd(Xt, full_matrices=False)
        kk = min(k, int(U.shape[1]))
        reduced = U[:, :kk] * S[:kk]
        return reduced.cpu().numpy().astype(np.float64, copy=False)

    return run_with_fallback(_impl)


def _gaussian_entropy_torch(X: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    """Torch Gaussian entropy helper for batched JMI kernels."""
    N, d = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = (Xc.transpose(0, 1) @ Xc) / max(N - 1, 1)
    cov = cov + reg * torch.eye(d, dtype=X.dtype, device=X.device)
    _, logdet = torch.linalg.slogdet(cov)
    const = torch.tensor(2 * np.pi * np.e, dtype=X.dtype, device=X.device)
    return 0.5 * (d * torch.log(const) + logdet)

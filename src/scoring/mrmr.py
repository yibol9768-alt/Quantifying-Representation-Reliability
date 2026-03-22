"""mRMR: Max-Relevance Min-Redundancy model selection.

Adapts the classic mRMR feature selection framework (Peng et al., 2005;
Brown et al., JMLR 2012) to the model selection level, where each model
is treated as a "feature group" and mutual information is estimated at
the model representation level.

The mRMR criterion selects the model maximizing:
    J_mrmr(m | S) = I(F_m; Y) - (1/|S|) * sum_{j in S} I(F_m; F_j)

We use Gaussian MI estimators for tractability.
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from ._torch_backend import run_with_fallback


def _gaussian_mi_xy(X: np.ndarray, Y_onehot: np.ndarray, reg: float = 1e-3) -> float:
    """Estimate I(X; Y) where Y is discrete, using class-conditional Gaussians.

    I(X; Y) = H(X) - H(X|Y) = H(X) - sum_y p(y) H(X|Y=y)
    Under Gaussian assumptions, H(X) = 0.5*log(det(2*pi*e*Sigma)).
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        Yt = torch.as_tensor(Y_onehot, dtype=dtype, device=device)
        N, d = Xt.shape

        Xc = Xt - Xt.mean(dim=0, keepdim=True)
        eye = torch.eye(d, dtype=dtype, device=device)
        cov_x = (Xc.transpose(0, 1) @ Xc) / max(N - 1, 1) + reg * eye
        _, logdet_x = torch.linalg.slogdet(cov_x)

        labels_t = Yt.argmax(dim=1) if Yt.ndim > 1 else Yt.to(torch.long)
        classes = torch.unique(labels_t)

        h_x_given_y = torch.zeros((), dtype=dtype, device=device)
        for c in classes:
            mask = labels_t == c
            n_c = int(mask.sum())
            if n_c < 2:
                continue
            p_c = n_c / N
            Xc_class = Xt[mask] - Xt[mask].mean(dim=0, keepdim=True)
            cov_c = (Xc_class.transpose(0, 1) @ Xc_class) / max(n_c - 1, 1) + reg * eye
            _, logdet_c = torch.linalg.slogdet(cov_c)
            h_x_given_y = h_x_given_y + p_c * logdet_c

        mi = 0.5 * (logdet_x - h_x_given_y)
        return float(torch.clamp(mi, min=0.0).item())

    return run_with_fallback(_impl)


def _gaussian_mi_ff(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3) -> float:
    """Estimate I(X; Y) between two continuous feature matrices under Gaussianity."""
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        Yt = torch.as_tensor(Y, dtype=dtype, device=device)
        N = int(Xt.shape[0])
        d1 = int(Xt.shape[1])
        d2 = int(Yt.shape[1])

        Xc = Xt - Xt.mean(dim=0, keepdim=True)
        Yc = Yt - Yt.mean(dim=0, keepdim=True)

        cov_x = (Xc.transpose(0, 1) @ Xc) / max(N - 1, 1) + reg * torch.eye(d1, dtype=dtype, device=device)
        cov_y = (Yc.transpose(0, 1) @ Yc) / max(N - 1, 1) + reg * torch.eye(d2, dtype=dtype, device=device)
        XY = torch.cat([Xc, Yc], dim=1)
        cov_xy = (XY.transpose(0, 1) @ XY) / max(N - 1, 1) + reg * torch.eye(d1 + d2, dtype=dtype, device=device)

        _, logdet_x = torch.linalg.slogdet(cov_x)
        _, logdet_y = torch.linalg.slogdet(cov_y)
        _, logdet_xy = torch.linalg.slogdet(cov_xy)

        mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
        return float(torch.clamp(mi, min=0.0).item())

    return run_with_fallback(_impl)


def mrmr_select(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    max_models: int = 6,
    pca_dim: Optional[int] = 64,
    reg: float = 1e-3,
) -> List[str]:
    """Select models using the mRMR criterion.

    At each step, select:
        m* = argmax_{m not in S} [ I(F_m; Y) - (1/|S|) sum_{j in S} I(F_m; F_j) ]

    Args:
        features: {model_name: [N, d] feature array}.
        labels: [N] integer class labels.
        max_models: Maximum number of models to select.
        pca_dim: If set, reduce features to this dimension via PCA before MI estimation.
        reg: Covariance regularization.

    Returns:
        Ordered list of selected model names.
    """
    names = list(features.keys())

    # Optional PCA reduction for stability
    feats = {}
    for name, f in features.items():
        f = f.astype(np.float64)
        if pca_dim is not None and f.shape[1] > pca_dim:
            f = _pca(f, pca_dim)
        feats[name] = f

    N = next(iter(feats.values())).shape[0]
    labels = np.asarray(labels).ravel()

    # One-hot for relevance computation
    classes = np.unique(labels)
    Y_onehot = np.zeros((N, len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Y_onehot[labels == c, i] = 1.0

    # Pre-compute relevance I(F_m; Y) for all models
    relevance = {}
    for name in names:
        relevance[name] = _gaussian_mi_xy(feats[name], Y_onehot, reg=reg)

    # Pre-compute pairwise redundancy I(F_i; F_j) for all pairs
    redundancy = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                val = _gaussian_mi_ff(feats[ni], feats[nj], reg=reg)
                redundancy[(ni, nj)] = val
                redundancy[(nj, ni)] = val

    # Greedy selection
    selected = []
    remaining = set(names)
    max_models = min(max_models, len(names))

    # First model: highest relevance
    first = max(names, key=lambda m: relevance[m])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < max_models and remaining:
        best_model = None
        best_score = -np.inf

        for m in remaining:
            rel = relevance[m]
            red = np.mean([redundancy.get((m, s), 0.0) for s in selected])
            score = rel - red
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

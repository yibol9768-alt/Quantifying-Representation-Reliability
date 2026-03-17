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
from typing import Dict, List, Optional


def _gaussian_mi_xy(X: np.ndarray, Y_onehot: np.ndarray, reg: float = 1e-3) -> float:
    """Estimate I(X; Y) where Y is discrete, using class-conditional Gaussians.

    I(X; Y) = H(X) - H(X|Y) = H(X) - sum_y p(y) H(X|Y=y)
    Under Gaussian assumptions, H(X) = 0.5*log(det(2*pi*e*Sigma)).
    """
    N, d = X.shape
    X = X.astype(np.float64)

    # H(X): entropy of overall distribution
    Xc = X - X.mean(axis=0, keepdims=True)
    cov_x = (Xc.T @ Xc) / max(N - 1, 1) + reg * np.eye(d)
    _, logdet_x = np.linalg.slogdet(cov_x)

    # H(X|Y): weighted sum of per-class entropies
    classes = np.unique(np.argmax(Y_onehot, axis=1)) if Y_onehot.ndim > 1 else np.unique(Y_onehot)
    labels = np.argmax(Y_onehot, axis=1) if Y_onehot.ndim > 1 else Y_onehot

    h_x_given_y = 0.0
    for c in classes:
        mask = labels == c
        n_c = mask.sum()
        if n_c < 2:
            continue
        p_c = n_c / N
        Xc_class = X[mask] - X[mask].mean(axis=0, keepdims=True)
        cov_c = (Xc_class.T @ Xc_class) / max(n_c - 1, 1) + reg * np.eye(d)
        _, logdet_c = np.linalg.slogdet(cov_c)
        h_x_given_y += p_c * logdet_c

    mi = 0.5 * (logdet_x - h_x_given_y)
    return float(max(mi, 0.0))


def _gaussian_mi_ff(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3) -> float:
    """Estimate I(X; Y) between two continuous feature matrices under Gaussianity."""
    N = X.shape[0]
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    d1 = X.shape[1]
    d2 = Y.shape[1]

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    cov_x = (Xc.T @ Xc) / max(N - 1, 1) + reg * np.eye(d1)
    cov_y = (Yc.T @ Yc) / max(N - 1, 1) + reg * np.eye(d2)

    XY = np.concatenate([Xc, Yc], axis=1)
    cov_xy = (XY.T @ XY) / max(N - 1, 1) + reg * np.eye(d1 + d2)

    _, logdet_x = np.linalg.slogdet(cov_x)
    _, logdet_y = np.linalg.slogdet(cov_y)
    _, logdet_xy = np.linalg.slogdet(cov_xy)

    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return float(max(mi, 0.0))


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
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(k, U.shape[1])
    return U[:, :k] * S[:k]

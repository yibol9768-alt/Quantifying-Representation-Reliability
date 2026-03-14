"""Centered Kernel Alignment (CKA) for measuring representation similarity."""

import torch
import numpy as np
from typing import Dict


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: Feature matrix of shape [N, d1].
        Y: Feature matrix of shape [N, d2].

    Returns:
        CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"

    # Center the features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = (YtX * YtX).sum()
    denominator = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())

    if denominator < 1e-10:
        return 0.0

    return (numerator / denominator).item()


def compute_cka_matrix(features_dict: Dict[str, torch.Tensor]) -> tuple:
    """Compute pairwise CKA matrix for all models.

    Args:
        features_dict: {model_name: [N, dim] tensor} for each model.

    Returns:
        (model_names, cka_matrix) where cka_matrix is a numpy array of shape [M, M].
    """
    model_names = list(features_dict.keys())
    n_models = len(model_names)
    cka_matrix = np.zeros((n_models, n_models))

    # Move all features to CPU float32 for consistent computation
    features = {}
    for name, feat in features_dict.items():
        features[name] = feat.float().cpu()

    for i in range(n_models):
        cka_matrix[i, i] = 1.0
        for j in range(i + 1, n_models):
            cka_val = linear_CKA(features[model_names[i]], features[model_names[j]])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val

    return model_names, cka_matrix

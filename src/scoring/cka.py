"""CKA: Centered Kernel Alignment for representation redundancy measurement.

Provides linear CKA computation and pairwise CKA matrix construction.
Used as the default redundancy metric in the model selection framework.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

from ._torch_backend import run_with_fallback


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two feature matrices.

    Automatically selects feature-space or kernel-space formulation
    based on dimensions.

    Args:
        X: [N, d1] feature matrix.
        Y: [N, d2] feature matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> float:
        Xt = torch.as_tensor(X, dtype=dtype, device=device)
        Yt = torch.as_tensor(Y, dtype=dtype, device=device)
        assert Xt.shape[0] == Yt.shape[0], "X and Y must have same number of samples"
        n = int(Xt.shape[0])

        Xt = Xt - Xt.mean(dim=0, keepdim=True)
        Yt = Yt - Yt.mean(dim=0, keepdim=True)

        d_max = max(int(Xt.shape[1]), int(Yt.shape[1]))
        if d_max <= n:
            yt_x = Yt.transpose(0, 1) @ Xt
            xt_x = Xt.transpose(0, 1) @ Xt
            yt_y = Yt.transpose(0, 1) @ Yt
            numerator = yt_x.square().sum()
            denominator = torch.sqrt(xt_x.square().sum() * yt_y.square().sum())
        else:
            K = Xt @ Xt.transpose(0, 1)
            L = Yt @ Yt.transpose(0, 1)
            H = torch.eye(n, dtype=dtype, device=device) - (1.0 / n)
            K = H @ K @ H
            L = H @ L @ H
            numerator = (K * L).sum()
            denominator = torch.sqrt((K * K).sum() * (L * L).sum())

        if denominator.item() < 1e-10:
            return 0.0
        return float(torch.clamp(numerator / denominator, min=0.0, max=1.0).item())

    return run_with_fallback(_impl)


def cka_pairwise_matrix(
    features: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    """Compute pairwise CKA matrix for all model features.

    Args:
        features: {model_name: [N, d] feature array}.

    Returns:
        (model_names, cka_matrix) where cka_matrix is [M, M].
    """
    def _impl(device: torch.device, dtype: torch.dtype) -> Tuple[List[str], np.ndarray]:
        names = list(features.keys())
        tensors = {
            name: torch.as_tensor(feat, dtype=dtype, device=device)
            for name, feat in features.items()
        }
        M = len(names)
        matrix = np.zeros((M, M), dtype=np.float64)

        centered = {
            name: tensor - tensor.mean(dim=0, keepdim=True)
            for name, tensor in tensors.items()
        }

        stats = {}
        for name, tensor in centered.items():
            n = int(tensor.shape[0])
            d = int(tensor.shape[1])
            if max(d, d) <= n:
                gram = tensor.transpose(0, 1) @ tensor
                denom_sq = gram.square().sum()
                stats[name] = ("feature", tensor, denom_sq)
            else:
                gram = tensor @ tensor.transpose(0, 1)
                H = torch.eye(n, dtype=dtype, device=device) - (1.0 / n)
                gram = H @ gram @ H
                denom_sq = (gram * gram).sum()
                stats[name] = ("kernel", gram, denom_sq)

        for i in range(M):
            matrix[i, i] = 1.0
            for j in range(i + 1, M):
                kind_i, data_i, denom_i = stats[names[i]]
                kind_j, data_j, denom_j = stats[names[j]]

                if kind_i == "feature" and kind_j == "feature":
                    numerator = (data_j.transpose(0, 1) @ data_i).square().sum()
                else:
                    gram_i = data_i if kind_i == "kernel" else (centered[names[i]] @ centered[names[i]].transpose(0, 1))
                    gram_j = data_j if kind_j == "kernel" else (centered[names[j]] @ centered[names[j]].transpose(0, 1))
                    numerator = (gram_i * gram_j).sum()

                denominator = torch.sqrt(denom_i * denom_j)
                val = 0.0 if denominator.item() < 1e-10 else float(
                    torch.clamp(numerator / denominator, min=0.0, max=1.0).item()
                )
                matrix[i, j] = val
                matrix[j, i] = val

        return names, matrix

    return run_with_fallback(_impl)

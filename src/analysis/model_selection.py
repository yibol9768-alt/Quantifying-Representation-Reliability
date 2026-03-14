"""CKA-guided model selection strategies."""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _avg_pairwise_cka(cka_matrix: np.ndarray, indices: List[int]) -> float:
    """Compute average pairwise CKA for a set of model indices."""
    if len(indices) < 2:
        return 0.0
    vals = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            vals.append(cka_matrix[indices[i], indices[j]])
    return float(np.mean(vals))


def greedy_selection(
    cka_matrix: np.ndarray,
    model_names: List[str],
    start_model: str,
    max_redundancy: float = 0.25,
) -> Tuple[List[str], List[str], List[dict]]:
    """Strategy A: Greedy selection with redundancy-based stopping.

    Starting from start_model, iteratively add the model with the lowest
    average CKA to the current set. Always computes the full ordering.
    Recommends stopping when the next model's avg CKA to the existing
    set exceeds max_redundancy.

    Args:
        cka_matrix: [M, M] CKA similarity matrix.
        model_names: List of model names corresponding to matrix rows/cols.
        start_model: Name of the starting model.
        max_redundancy: Maximum avg CKA of new model to existing set.
            Models exceeding this are still included in the full ordering
            but excluded from the recommended subset.

    Returns:
        (recommended, full_order, trace) where:
          - recommended: models selected before hitting the redundancy limit
          - full_order: all models in greedy diversity order
          - trace: per-step details (model, avg_cka_to_set, set_diversity)
    """
    name_to_idx = {name: i for i, name in enumerate(model_names)}
    selected = [start_model]
    remaining = [m for m in model_names if m != start_model]
    cutoff = None  # index where recommended subset ends

    trace = [{
        "step": 1,
        "model": start_model,
        "avg_cka_to_set": 0.0,
        "set_diversity": 1.0,
    }]

    step = 2
    while remaining:
        best_model = None
        best_avg_cka = float("inf")

        for candidate in remaining:
            c_idx = name_to_idx[candidate]
            cka_vals = [cka_matrix[name_to_idx[s], c_idx] for s in selected]
            avg_cka = float(np.mean(cka_vals))
            if avg_cka < best_avg_cka:
                best_avg_cka = avg_cka
                best_model = candidate

        # Check redundancy threshold for recommended cutoff
        if cutoff is None and best_avg_cka > max_redundancy and len(selected) >= 2:
            cutoff = len(selected)

        selected.append(best_model)
        remaining.remove(best_model)

        trial_indices = [name_to_idx[m] for m in selected]
        set_diversity = 1.0 - _avg_pairwise_cka(cka_matrix, trial_indices)

        trace.append({
            "step": step,
            "model": best_model,
            "avg_cka_to_set": best_avg_cka,
            "set_diversity": set_diversity,
        })
        step += 1

    if cutoff is None:
        cutoff = len(selected)

    recommended = selected[:cutoff]
    return recommended, selected, trace


def max_diversity_selection(
    cka_matrix: np.ndarray,
    model_names: List[str],
    k: int,
) -> List[str]:
    """Strategy B: Select k models that maximize diversity (minimize avg pairwise CKA).

    Uses greedy approximation: start with the pair having lowest CKA,
    then iteratively add the model with lowest average CKA to the current set.

    Args:
        cka_matrix: [M, M] CKA similarity matrix.
        model_names: List of model names.
        k: Number of models to select.

    Returns:
        List of k selected model names.
    """
    n = len(model_names)
    if k >= n:
        return list(model_names)

    name_to_idx = {name: i for i, name in enumerate(model_names)}

    # Find the pair with lowest CKA
    min_cka = float("inf")
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if cka_matrix[i, j] < min_cka:
                min_cka = cka_matrix[i, j]
                best_pair = (i, j)

    selected = [model_names[best_pair[0]], model_names[best_pair[1]]]

    # Greedily add models
    while len(selected) < k:
        remaining = [m for m in model_names if m not in selected]
        best_model = None
        best_avg_cka = float("inf")

        for candidate in remaining:
            c_idx = name_to_idx[candidate]
            cka_vals = [cka_matrix[name_to_idx[s], c_idx] for s in selected]
            avg_cka = float(np.mean(cka_vals))
            if avg_cka < best_avg_cka:
                best_avg_cka = avg_cka
                best_model = candidate

        selected.append(best_model)

    return selected


def task_adaptive_selection(
    cka_matrices: Dict[str, np.ndarray],
    model_names: List[str],
    start_model: str,
    max_redundancy: float = 0.25,
) -> Dict[str, Tuple[List[str], List[str], List[dict]]]:
    """Strategy C: Per-dataset greedy selection.

    Apply greedy_selection independently per dataset, since different tasks
    may benefit from different model subsets.

    Args:
        cka_matrices: {dataset_name: [M, M] CKA matrix}.
        model_names: List of model names (same order for all matrices).
        start_model: Starting model for greedy selection.
        max_redundancy: Maximum avg CKA threshold.

    Returns:
        {dataset_name: (recommended, full_order, trace)}.
    """
    results = {}
    for dataset, matrix in cka_matrices.items():
        results[dataset] = greedy_selection(matrix, model_names, start_model, max_redundancy)
    return results

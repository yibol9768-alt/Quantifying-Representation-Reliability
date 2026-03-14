"""CKA-guided model selection strategies."""

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional


def greedy_selection(
    cka_matrix: np.ndarray,
    model_names: List[str],
    start_model: str,
    threshold: float = 0.85,
) -> List[str]:
    """Strategy A: Greedy selection starting from the best single model.

    Starting from start_model, iteratively add the model with the lowest
    average CKA to the current set. Stop when the average pairwise CKA
    of the set would exceed the threshold.

    Args:
        cka_matrix: [M, M] CKA similarity matrix.
        model_names: List of model names corresponding to matrix rows/cols.
        start_model: Name of the starting model.
        threshold: Maximum allowed average pairwise CKA. Stop before exceeding.

    Returns:
        List of selected model names in order of addition.
    """
    name_to_idx = {name: i for i, name in enumerate(model_names)}
    selected = [start_model]
    remaining = [m for m in model_names if m != start_model]

    while remaining:
        best_model = None
        best_avg_cka = float("inf")

        for candidate in remaining:
            c_idx = name_to_idx[candidate]
            # Average CKA between candidate and all currently selected models
            cka_vals = [cka_matrix[name_to_idx[s], c_idx] for s in selected]
            avg_cka = np.mean(cka_vals)
            if avg_cka < best_avg_cka:
                best_avg_cka = avg_cka
                best_model = candidate

        # Check if adding this model would exceed threshold
        trial_set = selected + [best_model]
        trial_indices = [name_to_idx[m] for m in trial_set]
        pairwise_ckas = []
        for i_idx in range(len(trial_indices)):
            for j_idx in range(i_idx + 1, len(trial_indices)):
                pairwise_ckas.append(cka_matrix[trial_indices[i_idx], trial_indices[j_idx]])
        avg_pairwise = np.mean(pairwise_ckas)

        if avg_pairwise > threshold and len(selected) >= 2:
            break

        selected.append(best_model)
        remaining.remove(best_model)

    return selected


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
            avg_cka = np.mean(cka_vals)
            if avg_cka < best_avg_cka:
                best_avg_cka = avg_cka
                best_model = candidate

        selected.append(best_model)

    return selected


def task_adaptive_selection(
    cka_matrices: Dict[str, np.ndarray],
    model_names: List[str],
    start_model: str,
    threshold: float = 0.85,
) -> Dict[str, List[str]]:
    """Strategy C: Per-dataset greedy selection.

    Apply greedy_selection independently per dataset, since different tasks
    may benefit from different model subsets.

    Args:
        cka_matrices: {dataset_name: [M, M] CKA matrix}.
        model_names: List of model names (same order for all matrices).
        start_model: Starting model for greedy selection.
        threshold: CKA threshold for greedy selection.

    Returns:
        {dataset_name: selected_model_list}.
    """
    results = {}
    for dataset, matrix in cka_matrices.items():
        results[dataset] = greedy_selection(matrix, model_names, start_model, threshold)
    return results

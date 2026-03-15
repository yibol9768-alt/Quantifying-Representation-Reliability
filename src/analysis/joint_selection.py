"""Diversity × Relevance Joint Model Selection Framework.

Implements Marginal Utility-based Model Selection (MUMS):

    U(m | S, T) = R(m, T)^α · (1 - ρ(m, S | D))^β

where:
    R(m, T)       = task relevance of model m (normalized single-model accuracy)
    ρ(m, S | D)   = avg CKA of m to selected set S on dataset D
    α, β          = exponents controlling relevance/diversity trade-off

The multiplicative form ensures that zero relevance (noise model) yields
zero utility regardless of diversity — matching our empirical finding that
DINO on SVHN is harmful despite being "diverse" from existing models.

Information-theoretic justification:
    ΔI(m, S) = I(f_m; Y | f_S)
             ≈ I(f_m; Y) · [1 - I(f_m; f_S) / H(f_m)]
             = Relevance(m) · Novelty(m, S)

Stopping criteria (Stage 2 — Subset Size Selection):
    The greedy selection can be stopped early using one of three criteria:
    1. Utility threshold: stop when U(m*|S) < τ
    2. Marginal gain ratio: stop when U(m*|S) / U(m_1) < τ
    3. Validation callback: stop when val_acc(S∪{m*}) ≤ val_acc(S)

References:
    - Carbonell & Goldberg (1998), MMR: analogous additive form for IR
    - Kulesza & Taskar (2012), DPP: quality × diversity kernel decomposition
    - Nemhauser et al. (1978): greedy submodular maximization guarantees
    - Caruana et al. (2004), Ensemble Selection from Libraries of Models
"""

import json
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


def normalize_relevance(
    single_model_acc: Dict[str, float],
) -> Dict[str, float]:
    """Normalize single-model accuracies to [0, 1] via min-max scaling.

    Args:
        single_model_acc: {model_name: accuracy_percent} on target task.

    Returns:
        {model_name: normalized_relevance} in [0, 1].
    """
    vals = list(single_model_acc.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {m: 1.0 for m in single_model_acc}
    return {m: (v - lo) / (hi - lo) for m, v in single_model_acc.items()}


def marginal_utility(
    candidate: str,
    selected: List[str],
    cka_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Compute marginal utility of adding candidate to selected set.

    U(m | S) = R(m)^α · (1 - avg_CKA(m, S))^β

    Args:
        candidate: model name to evaluate.
        selected: already selected model names.
        cka_matrix: [M, M] pairwise CKA matrix.
        model_names: model names corresponding to matrix indices.
        relevance: {model: normalized_relevance} scores.
        alpha: relevance exponent (higher → favor relevance more).
        beta: diversity exponent (higher → favor diversity more).

    Returns:
        Marginal utility score.
    """
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    c_idx = name_to_idx[candidate]

    # Relevance term
    rel = relevance.get(candidate, 0.0)

    # Redundancy term: avg CKA to selected set
    if not selected:
        redundancy = 0.0
    else:
        cka_vals = [cka_matrix[name_to_idx[s], c_idx] for s in selected]
        redundancy = float(np.mean(cka_vals))

    novelty = 1.0 - redundancy

    return (rel ** alpha) * (novelty ** beta)


def joint_greedy_selection(
    cka_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 1.0,
    start_model: Optional[str] = None,
    stop: Optional[str] = None,
    stop_threshold: float = 0.1,
    val_callback: Optional[Callable[[List[str]], float]] = None,
    patience: int = 1,
) -> Tuple[List[str], List[dict], int]:
    """Joint Diversity × Relevance greedy model selection.

    At each step, select the model with highest marginal utility:
        U(m | S) = R(m)^α · (1 - avg_CKA(m, S))^β

    Supports optional early stopping to automatically determine subset size K*.

    Args:
        cka_matrix: [M, M] pairwise CKA matrix.
        model_names: model names corresponding to matrix indices.
        relevance: {model: normalized_relevance} in [0, 1].
        alpha: relevance exponent.
        beta: diversity exponent.
        start_model: if None, start with highest-relevance model.
        stop: stopping criterion, one of:
            - None: no early stopping, rank all models (default).
            - "utility": stop when U(m*|S) < stop_threshold.
            - "gain_ratio": stop when U(m*|S) / U(step1) < stop_threshold.
            - "validation": stop when val_callback(S∪{m*}) ≤ val_callback(S)
              for `patience` consecutive steps. Requires val_callback.
        stop_threshold: threshold for "utility" and "gain_ratio" stopping.
            For "utility": absolute utility threshold (default 0.1).
            For "gain_ratio": ratio relative to first step (default 0.1 = 10%).
        val_callback: function that takes a list of model names and returns
            a validation metric (higher is better). Required when stop="validation".
        patience: number of consecutive non-improving steps before stopping
            (only used with stop="validation", default 1).

    Returns:
        (ordered_selection, trace, k_star) where:
            ordered_selection: all models ranked by utility.
            trace: per-step details including stopping info.
            k_star: recommended subset size.
    """
    if stop == "validation" and val_callback is None:
        raise ValueError("val_callback is required when stop='validation'")

    name_to_idx = {n: i for i, n in enumerate(model_names)}

    # Start model: highest relevance or specified
    if start_model is None:
        start_model = max(
            (m for m in model_names if m in relevance),
            key=lambda m: relevance[m],
        )

    selected = [start_model]
    remaining = [m for m in model_names if m != start_model]

    first_utility = relevance.get(start_model, 0.0)

    # Evaluate validation for initial model if using validation stopping
    best_val_score = None
    best_val_k = 1  # K at which best validation was seen
    no_improve_count = 0
    if stop == "validation":
        best_val_score = val_callback(selected)

    trace = [{
        "step": 1,
        "model": start_model,
        "relevance": relevance.get(start_model, 0.0),
        "avg_cka_to_set": 0.0,
        "novelty": 1.0,
        "utility": first_utility,
        "stopped": False,
    }]

    stopped = False
    selected_k = None  # index where we stopped (None = no stop)
    step = 2
    while remaining:
        best_model = None
        best_utility = -float("inf")
        best_info = {}

        for candidate in remaining:
            u = marginal_utility(
                candidate, selected, cka_matrix, model_names,
                relevance, alpha, beta,
            )
            c_idx = name_to_idx[candidate]
            cka_vals = [cka_matrix[name_to_idx[s], c_idx] for s in selected]
            avg_cka = float(np.mean(cka_vals))

            if u > best_utility:
                best_utility = u
                best_model = candidate
                best_info = {
                    "relevance": relevance.get(candidate, 0.0),
                    "avg_cka_to_set": avg_cka,
                    "novelty": 1.0 - avg_cka,
                }

        # Check stopping criteria before adding
        should_stop = False
        stop_reason = ""

        if stop == "utility" and best_utility < stop_threshold:
            should_stop = True
            stop_reason = (
                f"utility {best_utility:.4f} < threshold {stop_threshold}"
            )

        elif stop == "gain_ratio" and first_utility > 0:
            ratio = best_utility / first_utility
            if ratio < stop_threshold:
                should_stop = True
                stop_reason = (
                    f"gain_ratio {ratio:.4f} < threshold {stop_threshold}"
                )

        elif stop == "validation" and not stopped:
            # Tentatively add and evaluate
            trial_set = selected + [best_model]
            val_score = val_callback(trial_set)
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_k = len(trial_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    should_stop = True
                    stop_reason = (
                        f"no improvement for {patience} steps "
                        f"(best={best_val_score:.4f} at K={best_val_k}, "
                        f"current={val_score:.4f})"
                    )

        if should_stop and not stopped:
            stopped = True
            selected_k = len(selected)  # K* = current subset size

        selected.append(best_model)
        remaining.remove(best_model)

        trace.append({
            "step": step,
            "model": best_model,
            "utility": best_utility,
            "stopped": stopped,
            "stop_reason": stop_reason if should_stop else "",
            **best_info,
        })
        step += 1

    # Determine K*
    if stop == "validation" and stopped:
        # Use the K that achieved best validation score
        k_star = best_val_k
    elif stopped:
        k_star = selected_k
    else:
        k_star = len(selected)

    return selected, trace, k_star


def select_subset(
    cka_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 1.0,
    stop: str = "gain_ratio",
    stop_threshold: float = 0.1,
    val_callback: Optional[Callable[[List[str]], float]] = None,
    patience: int = 1,
) -> Tuple[List[str], int, List[dict]]:
    """Select optimal model subset using MUMS with automatic stopping.

    Two-stage method:
        Stage 1 (Ordering): Greedy selection by marginal utility.
        Stage 2 (Selection): Stop when stopping criterion triggers.

    Args:
        cka_matrix: [M, M] pairwise CKA matrix.
        model_names: model names corresponding to matrix indices.
        relevance: {model: normalized_relevance} in [0, 1].
        alpha: relevance exponent.
        beta: diversity exponent.
        stop: stopping criterion ("utility", "gain_ratio", or "validation").
        stop_threshold: threshold value for the chosen criterion.
        val_callback: validation function for stop="validation".
        patience: consecutive non-improving steps before stopping
            (only for stop="validation", default 1).

    Returns:
        (full_ordering, k_star, trace) where:
            full_ordering: all models ranked by utility.
            k_star: recommended subset size (use full_ordering[:k_star]).
            trace: per-step details including stopping info.
    """
    full_ordering, trace, k_star = joint_greedy_selection(
        cka_matrix, model_names, relevance,
        alpha=alpha, beta=beta,
        stop=stop, stop_threshold=stop_threshold,
        val_callback=val_callback,
        patience=patience,
    )
    return full_ordering, k_star, trace


def compare_orderings(
    cka_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    alpha_values: List[float] = [0.5, 1.0, 1.5, 2.0],
    beta_values: List[float] = [0.5, 1.0, 1.5, 2.0],
) -> Dict[str, List[str]]:
    """Generate orderings for a grid of (alpha, beta) values.

    Also generates three baseline orderings:
        - "relevance_only": pure relevance ranking (α=1, β=0)
        - "diversity_only": pure CKA diversity ranking (α=0, β=1)
        - "original": fixed order clip→dino→mae→siglip→convnext→data2vec

    Returns:
        {label: ordered_model_list}
    """
    ORIGINAL_ORDER = ["clip", "dino", "mae", "siglip", "convnext", "data2vec"]

    results = {}

    # Baselines
    # Relevance only
    rel_order, _, _ = joint_greedy_selection(
        cka_matrix, model_names, relevance, alpha=1.0, beta=0.0,
    )
    results["relevance_only"] = rel_order

    # Diversity only (CKA greedy from clip)
    div_order, _, _ = joint_greedy_selection(
        cka_matrix, model_names, relevance, alpha=0.0, beta=1.0,
        start_model="clip",
    )
    results["diversity_only"] = div_order

    # Original
    orig = [m for m in ORIGINAL_ORDER if m in model_names]
    results["original"] = orig

    # Grid search
    for alpha in alpha_values:
        for beta in beta_values:
            label = f"a{alpha:.1f}_b{beta:.1f}"
            order, _, _ = joint_greedy_selection(
                cka_matrix, model_names, relevance, alpha=alpha, beta=beta,
            )
            results[label] = order

    return results


def compute_cumulative_scores(
    ordering: List[str],
    cka_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
) -> List[dict]:
    """For a given ordering, compute cumulative diversity and relevance at each step.

    Returns list of dicts with step, models, avg_relevance, set_diversity, cum_score.
    """
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    selected_indices = []
    scores = []

    for step, model in enumerate(ordering, 1):
        selected_indices.append(name_to_idx[model])

        # Average relevance of selected set
        avg_rel = np.mean([relevance.get(ordering[i], 0.0) for i in range(step)])

        # Set diversity: 1 - avg pairwise CKA
        if len(selected_indices) < 2:
            diversity = 1.0
        else:
            cka_vals = []
            for i in range(len(selected_indices)):
                for j in range(i + 1, len(selected_indices)):
                    cka_vals.append(cka_matrix[selected_indices[i], selected_indices[j]])
            diversity = 1.0 - float(np.mean(cka_vals))

        scores.append({
            "step": step,
            "model": model,
            "models": ordering[:step],
            "avg_relevance": float(avg_rel),
            "set_diversity": float(diversity),
            "combined_score": float(avg_rel * diversity),
        })

    return scores

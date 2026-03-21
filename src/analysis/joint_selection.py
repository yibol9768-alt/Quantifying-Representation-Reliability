"""Diversity × Relevance Joint Model Selection Framework.

Implements Marginal Utility-based Model Selection (MUMS):

    U(m | S, T) = R(m, T)^α · (1 - ρ(m, S | D))^β

where:
    R(m, T)       = task relevance of model m (normalized single-model accuracy)
    ρ(m, S | D)   = avg CKA of m to selected set S on dataset D
    α, β          = exponents controlling relevance/diversity trade-off

The multiplicative form ensures that zero relevance (noise model) yields
zero utility regardless of diversity — matching our empirical finding that
a model can be novel yet still unhelpful for the task.

Information-theoretic note:
    The exact decomposition is
        I(f_m; Y | f_S)
      = I(f_m; Y) - I(f_m; f_S) + I(f_m; f_S | Y).

    The original multiplicative MUMS score implemented here only uses
    tractable proxies for the first two terms:
        - relevance proxy for I(f_m; Y)
        - CKA-based redundancy proxy for I(f_m; f_S)

    It does not include a proxy for the class-conditional term
    I(f_m; f_S | Y). Class-conditional CKA can be analyzed separately as a
    diagnostic, but it is not folded into the original MUMS score.

    For a three-term low-order approximation, see
    conditional_joint_greedy_selection below, which replaces the set-level
    third term by an average pairwise conditional term following the spirit
    of Brown et al. (2012).

References:
    - Carbonell & Goldberg (1998), MMR: analogous additive form for IR
    - Kulesza & Taskar (2012), DPP: quality × diversity kernel decomposition
    - Nemhauser et al. (1978): greedy submodular maximization guarantees
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple


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


def conditional_marginal_utility(
    candidate: str,
    selected: List[str],
    cka_matrix: np.ndarray,
    conditional_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    lambda_red: float = 1.0,
    eta_cond: float = 1.0,
) -> float:
    """Compute a three-term low-order utility.

    Score(m | S) = relevance(m)
                 - lambda_red * avg_redundancy(m, S)
                 + eta_cond * avg_conditional_term(m, S)

    The conditional term is intended to approximate the pairwise low-order
    correction avg_{j in S} I(F_m; F_j | Y), not the full set-level quantity.
    """
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    c_idx = name_to_idx[candidate]

    rel = relevance.get(candidate, 0.0)
    if not selected:
        redundancy = 0.0
        conditional = 0.0
    else:
        redundancy = float(np.mean([cka_matrix[name_to_idx[s], c_idx] for s in selected]))
        conditional = float(np.mean([conditional_matrix[name_to_idx[s], c_idx] for s in selected]))

    return rel - lambda_red * redundancy + eta_cond * conditional


def joint_greedy_selection(
    cka_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 1.0,
    start_model: Optional[str] = None,
) -> Tuple[List[str], List[dict]]:
    """Joint Diversity × Relevance greedy model selection.

    At each step, select the model with highest marginal utility:
        U(m | S) = R(m)^α · (1 - avg_CKA(m, S))^β

    Args:
        cka_matrix: [M, M] pairwise CKA matrix.
        model_names: model names corresponding to matrix indices.
        relevance: {model: normalized_relevance} in [0, 1].
        alpha: relevance exponent.
        beta: diversity exponent.
        start_model: if None, start with highest-relevance model.

    Returns:
        (ordered_selection, trace) where trace has per-step details.
    """
    name_to_idx = {n: i for i, n in enumerate(model_names)}

    # Start model: highest relevance or specified
    if start_model is not None and start_model not in model_names:
        raise ValueError(f"start_model '{start_model}' not found in model_names")

    if start_model is None:
        start_model = max(
            (m for m in model_names if m in relevance),
            key=lambda m: relevance[m],
        )

    selected = [start_model]
    remaining = [m for m in model_names if m != start_model]

    trace = [{
        "step": 1,
        "model": start_model,
        "relevance": relevance.get(start_model, 0.0),
        "avg_cka_to_set": 0.0,
        "novelty": 1.0,
        "utility": relevance.get(start_model, 0.0),
    }]

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

        selected.append(best_model)
        remaining.remove(best_model)

        trace.append({
            "step": step,
            "model": best_model,
            "utility": best_utility,
            **best_info,
        })
        step += 1

    return selected, trace


def conditional_joint_greedy_selection(
    cka_matrix: np.ndarray,
    conditional_matrix: np.ndarray,
    model_names: List[str],
    relevance: Dict[str, float],
    lambda_red: float = 1.0,
    eta_cond: float = 1.0,
    start_model: Optional[str] = None,
) -> Tuple[List[str], List[dict]]:
    """Greedy selection with a low-order three-term approximation.

    At each step, select the model with highest score:

        score(m | S) = R(m) - lambda_red * D(m, S) + eta_cond * C(m, S)

    where:
        D(m, S) = avg CKA redundancy to selected set
        C(m, S) = avg pairwise conditional term to selected set
    """
    name_to_idx = {n: i for i, n in enumerate(model_names)}

    if start_model is not None and start_model not in model_names:
        raise ValueError(f"start_model '{start_model}' not found in model_names")

    if start_model is None:
        start_model = max(
            (m for m in model_names if m in relevance),
            key=lambda m: relevance[m],
        )

    selected = [start_model]
    remaining = [m for m in model_names if m != start_model]
    trace = [{
        "step": 1,
        "model": start_model,
        "relevance": relevance.get(start_model, 0.0),
        "avg_cka_to_set": 0.0,
        "avg_conditional_to_set": 0.0,
        "utility": relevance.get(start_model, 0.0),
    }]

    step = 2
    while remaining:
        best_model = None
        best_utility = -float("inf")
        best_info = {}

        for candidate in remaining:
            u = conditional_marginal_utility(
                candidate,
                selected,
                cka_matrix,
                conditional_matrix,
                model_names,
                relevance,
                lambda_red=lambda_red,
                eta_cond=eta_cond,
            )
            c_idx = name_to_idx[candidate]
            avg_cka = float(np.mean([cka_matrix[name_to_idx[s], c_idx] for s in selected]))
            avg_cond = float(np.mean([conditional_matrix[name_to_idx[s], c_idx] for s in selected]))

            if u > best_utility:
                best_utility = u
                best_model = candidate
                best_info = {
                    "relevance": relevance.get(candidate, 0.0),
                    "avg_cka_to_set": avg_cka,
                    "avg_conditional_to_set": avg_cond,
                }

        selected.append(best_model)
        remaining.remove(best_model)
        trace.append({
            "step": step,
            "model": best_model,
            "utility": best_utility,
            **best_info,
        })
        step += 1

    return selected, trace


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
    rel_order, _ = joint_greedy_selection(
        cka_matrix, model_names, relevance, alpha=1.0, beta=0.0,
    )
    results["relevance_only"] = rel_order

    # Diversity only (CKA greedy from clip)
    div_start = "clip" if "clip" in model_names else max(
        (m for m in model_names if m in relevance),
        key=lambda m: relevance[m],
    )
    div_order, _ = joint_greedy_selection(
        cka_matrix, model_names, relevance, alpha=0.0, beta=1.0,
        start_model=div_start,
    )
    results["diversity_only"] = div_order

    # Original
    orig = [m for m in ORIGINAL_ORDER if m in model_names]
    results["original"] = orig

    # Grid search
    for alpha in alpha_values:
        for beta in beta_values:
            label = f"a{alpha:.1f}_b{beta:.1f}"
            order, _ = joint_greedy_selection(
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

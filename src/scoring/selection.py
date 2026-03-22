"""Unified model selection framework.

Provides a single entry point `greedy_select` that supports multiple
combinations of relevance scoring, redundancy measurement, and selection
strategies for comparing different model selection approaches.

Supported combinations:
    - relevance_redundancy: score = R(m) - lambda * D(m, S)
    - three_term_low_order: score = R(m) - lambda * D(m, S) + eta * C(m, S)
    - mrmr: Max-Relevance Min-Redundancy (information-theoretic)
    - jmi: Joint Mutual Information (information-theoretic)
    - relevance_only: score = R(m), no redundancy penalty
    - random: random ordering baseline
"""

import random as _random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .logme import logme_score
from .leep import leep_score
from .gbc import gbc_score
from .hscore import hscore
from .cka import linear_cka, cka_pairwise_matrix
from .svcca import svcca_similarity, svcca_pairwise_matrix
from .mrmr import mrmr_select
from .jmi import jmi_select
from src.analysis.conditional_mi import compute_pairwise_class_conditional_mi_matrix


def _compute_relevance(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    method: str,
    source_probs: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute per-model relevance scores.

    Args:
        features: {model_name: [N, d] features}.
        labels: [N] integer labels.
        method: One of "logme", "leep", "gbc", "hscore".
        source_probs: Required for LEEP; {model_name: [N, C_source] softmax probs}.

    Returns:
        {model_name: relevance_score}.
    """
    scores = {}
    for name, feat in features.items():
        if method == "logme":
            scores[name] = logme_score(feat, labels)
        elif method == "leep":
            if source_probs is None or name not in source_probs:
                raise ValueError(f"LEEP requires source_probs for model '{name}'")
            scores[name] = leep_score(source_probs[name], labels)
        elif method == "gbc":
            scores[name] = gbc_score(feat, labels)
        elif method == "hscore":
            scores[name] = hscore(feat, labels)
        else:
            raise ValueError(f"Unknown relevance method: {method}")

    return scores


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _compute_redundancy_matrix(
    features: Dict[str, np.ndarray],
    method: str,
) -> Tuple[List[str], np.ndarray]:
    """Compute pairwise redundancy matrix.

    Args:
        features: {model_name: [N, d] features}.
        method: One of "cka", "svcca".

    Returns:
        (model_names, redundancy_matrix [M, M]).
    """
    if method == "cka":
        return cka_pairwise_matrix(features)
    elif method == "svcca":
        return svcca_pairwise_matrix(features)
    else:
        raise ValueError(f"Unknown redundancy method: {method}")


def _compute_conditional_matrix(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    pca_dim: int = 32,
    min_class_samples: int = 8,
    reg: float = 1e-3,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
    """Compute a low-order pairwise conditional matrix.

    The returned normalized matrix is meant to approximate
    avg_{j in S} I(F_m; F_j | Y) for three-term scoring.
    """
    feature_tensors = {
        name: torch.as_tensor(feat, dtype=torch.float32)
        for name, feat in features.items()
    }
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    names, raw, norm, meta = compute_pairwise_class_conditional_mi_matrix(
        feature_tensors,
        labels_t,
        pca_dim=pca_dim,
        min_class_samples=min_class_samples,
        reg=reg,
    )
    return names, raw, norm, meta


def greedy_select(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    relevance_method: str = "logme",
    redundancy_method: str = "cka",
    selection_method: str = "relevance_redundancy",
    max_models: int = 6,
    lambda_param: float = 1.0,
    eta_cond: float = 1.0,
    conditional_pca_dim: int = 32,
    conditional_min_class_samples: int = 8,
    conditional_reg: float = 1e-3,
    source_probs: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 42,
) -> Tuple[List[str], Dict]:
    """Greedy model subset selection with configurable scoring.

    Args:
        features: {model_name: [N, d] numpy feature array}.
        labels: [N] integer class labels.
        relevance_method: "logme", "leep", "gbc", "hscore".
        redundancy_method: "cka", "svcca".
        selection_method: "relevance_redundancy", "three_term_low_order",
            "mrmr", "jmi", "relevance_only", "random".
        max_models: Maximum number of models to select.
        lambda_param: Trade-off parameter for redundancy penalty.
        eta_cond: Weight on the low-order conditional term when
            selection_method="three_term_low_order".
        conditional_pca_dim: PCA dimension before pairwise conditional MI.
        conditional_min_class_samples: Min samples/class for the third term.
        conditional_reg: Covariance regularization for the third term.
        source_probs: Required if relevance_method="leep".
        seed: Random seed for "random" selection.

    Returns:
        (selected_models, metadata) where selected_models is an ordered
        list of model names and metadata contains scoring details.
    """
    names = list(features.keys())
    labels = np.asarray(labels).ravel()
    max_models = min(max_models, len(names))

    metadata = {
        "relevance_method": relevance_method,
        "redundancy_method": redundancy_method,
        "selection_method": selection_method,
        "max_models": max_models,
        "lambda_param": lambda_param,
        "eta_cond": eta_cond,
        "num_candidates": len(names),
    }

    # Delegate to specialized methods
    if selection_method == "mrmr":
        selected = mrmr_select(features, labels, max_models=max_models)
        metadata["method_type"] = "information_theoretic"
        metadata["relevance_method"] = None
        metadata["redundancy_method"] = None
        metadata["lambda_param"] = None
        return selected, metadata

    if selection_method == "jmi":
        selected = jmi_select(features, labels, max_models=max_models)
        metadata["method_type"] = "information_theoretic"
        metadata["relevance_method"] = None
        metadata["redundancy_method"] = None
        metadata["lambda_param"] = None
        return selected, metadata

    if selection_method == "random":
        rng = _random.Random(seed)
        selected = list(names)
        rng.shuffle(selected)
        selected = selected[:max_models]
        metadata["method_type"] = "baseline"
        return selected, metadata

    # Compute relevance scores
    relevance = _compute_relevance(features, labels, relevance_method, source_probs)
    norm_relevance = _normalize_scores(relevance)
    metadata["relevance_scores"] = relevance
    metadata["normalized_relevance"] = norm_relevance

    if selection_method == "relevance_only":
        selected = sorted(names, key=lambda m: relevance[m], reverse=True)[:max_models]
        metadata["method_type"] = "relevance_only"
        return selected, metadata

    if selection_method == "relevance_redundancy":
        # Compute redundancy matrix
        red_names, red_matrix = _compute_redundancy_matrix(features, redundancy_method)
        name_to_idx = {n: i for i, n in enumerate(red_names)}
        metadata["redundancy_matrix"] = red_matrix.tolist()
        metadata["method_type"] = "relevance_redundancy"

        # Greedy selection: score = R(m) - lambda * avg_redundancy(m, S)
        selected = []
        remaining = set(names)
        trace = []

        # First model: highest relevance
        first = max(sorted(names), key=lambda m: (norm_relevance[m], relevance[m], m))
        selected.append(first)
        remaining.remove(first)
        trace.append({
            "step": 1,
            "model": first,
            "relevance": norm_relevance[first],
            "avg_redundancy": 0.0,
            "score": norm_relevance[first],
        })

        while len(selected) < max_models and remaining:
            best_model = None
            best_score = -np.inf
            best_info = {}

            for m in remaining:
                rel = norm_relevance[m]
                m_idx = name_to_idx[m]
                red_vals = [red_matrix[name_to_idx[s], m_idx] for s in selected]
                avg_red = float(np.mean(red_vals))
                score = rel - lambda_param * avg_red

                if score > best_score or (
                    score == best_score and best_model is not None and m < best_model
                ):
                    best_score = score
                    best_model = m
                    best_info = {
                        "relevance": rel,
                        "avg_redundancy": avg_red,
                        "score": score,
                    }

            selected.append(best_model)
            remaining.remove(best_model)
            trace.append({
                "step": len(selected),
                "model": best_model,
                **best_info,
            })

        metadata["trace"] = trace
        return selected, metadata

    if selection_method == "three_term_low_order":
        red_names, red_matrix = _compute_redundancy_matrix(features, redundancy_method)
        cond_names, cond_raw, cond_norm, cond_meta = _compute_conditional_matrix(
            features,
            labels,
            pca_dim=conditional_pca_dim,
            min_class_samples=conditional_min_class_samples,
            reg=conditional_reg,
        )
        if red_names != cond_names:
            raise ValueError(
                "Redundancy and conditional matrices use different model orders: "
                f"{red_names} vs {cond_names}"
            )
        name_to_idx = {n: i for i, n in enumerate(red_names)}
        metadata["redundancy_matrix"] = red_matrix.tolist()
        metadata["conditional_matrix"] = cond_norm.tolist()
        metadata["conditional_matrix_raw"] = cond_raw.tolist()
        metadata["conditional_meta"] = cond_meta
        metadata["method_type"] = "three_term_low_order"

        selected = []
        remaining = set(names)
        trace = []

        first = max(sorted(names), key=lambda m: (norm_relevance[m], relevance[m], m))
        selected.append(first)
        remaining.remove(first)
        trace.append({
            "step": 1,
            "model": first,
            "relevance": norm_relevance[first],
            "avg_redundancy": 0.0,
            "avg_conditional": 0.0,
            "score": norm_relevance[first],
        })

        while len(selected) < max_models and remaining:
            best_model = None
            best_score = -np.inf
            best_info = {}

            for m in remaining:
                rel = norm_relevance[m]
                m_idx = name_to_idx[m]
                red_vals = [red_matrix[name_to_idx[s], m_idx] for s in selected]
                cond_vals = [cond_norm[name_to_idx[s], m_idx] for s in selected]
                avg_red = float(np.mean(red_vals))
                avg_cond = float(np.mean(cond_vals))
                score = rel - lambda_param * avg_red + eta_cond * avg_cond

                if score > best_score or (
                    score == best_score and best_model is not None and m < best_model
                ):
                    best_score = score
                    best_model = m
                    best_info = {
                        "relevance": rel,
                        "avg_redundancy": avg_red,
                        "avg_conditional": avg_cond,
                        "score": score,
                    }

            selected.append(best_model)
            remaining.remove(best_model)
            trace.append({
                "step": len(selected),
                "model": best_model,
                **best_info,
            })

        metadata["trace"] = trace
        return selected, metadata

    raise ValueError(f"Unknown selection_method: {selection_method}")

"""Evaluation metrics and correlation computation."""

from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr


@dataclass
class CorrelationResult:
    """Container for correlation results."""
    kendall_tau: float
    spearman_rho: float
    pearson_r: float
    p_value_kendall: float
    p_value_spearman: float
    p_value_pearson: float


def compute_correlation(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    method: str = "kendall",
) -> CorrelationResult:
    """
    Compute correlation between predicted scores and ground truth reliability.

    Args:
        predicted: Predicted reliability scores (N,)
        ground_truth: Ground truth downstream performance (N,)
        method: Primary correlation method for reporting

    Returns:
        CorrelationResult with all correlation metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(predicted) | np.isnan(ground_truth))
    pred = predicted[mask]
    gt = ground_truth[mask]

    if len(pred) < 10:
        return CorrelationResult(
            kendall_tau=0.0,
            spearman_rho=0.0,
            pearson_r=0.0,
            p_value_kendall=1.0,
            p_value_spearman=1.0,
            p_value_pearson=1.0,
        )

    # Compute correlations
    tau, p_tau = kendalltau(pred, gt)
    rho, p_rho = spearmanr(pred, gt)
    r, p_r = pearsonr(pred, gt)

    return CorrelationResult(
        kendall_tau=tau,
        spearman_rho=rho,
        pearson_r=r,
        p_value_kendall=p_tau,
        p_value_spearman=p_rho,
        p_value_pearson=p_r,
    )


def evaluate_method(
    nc_scores: np.ndarray,
    downstream_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate how well NC scores correlate with downstream performance.

    Args:
        nc_scores: NC scores for each sample
        downstream_scores: Downstream performance (e.g., negative Brier score)

    Returns:
        Dictionary with correlation metrics
    """
    result = compute_correlation(nc_scores, downstream_scores)

    return {
        "kendall_tau": result.kendall_tau,
        "spearman_rho": result.spearman_rho,
        "pearson_r": result.pearson_r,
        "p_value": result.p_value_kendall,
    }


class ResultAggregator:
    """Aggregate and compare results across multiple methods and settings."""

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}

    def add_result(
        self,
        method_name: str,
        setting: str,
        nc_scores: np.ndarray,
        downstream_scores: np.ndarray,
    ):
        """Add a result for a method in a specific setting."""
        metrics = evaluate_method(nc_scores, downstream_scores)
        key = f"{method_name}/{setting}"
        self.results[key] = metrics

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all results."""
        return self.results

    def compare_methods(self, metric: str = "kendall_tau") -> Dict[str, float]:
        """Compare all methods by a specific metric."""
        return {
            key: vals[metric]
            for key, vals in self.results.items()
        }

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.results).T
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

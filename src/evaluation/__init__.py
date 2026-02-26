"""Evaluation methods for representation reliability."""

from .nc import NeighborhoodConsistency, NCConfig, load_representations, save_representations
from .downstream import (
    DownstreamTask,
    BinaryClassificationTask,
    MultiClassificationTask,
)
from .metrics import (
    compute_correlation,
    evaluate_method,
    ResultAggregator,
    CorrelationResult,
)

__all__ = [
    "NeighborhoodConsistency",
    "NCConfig",
    "load_representations",
    "save_representations",
    "DownstreamTask",
    "BinaryClassificationTask",
    "MultiClassificationTask",
    "compute_correlation",
    "evaluate_method",
    "ResultAggregator",
    "CorrelationResult",
]

"""Model selection scoring methods for multi-model feature fusion.

This package provides training-free transferability metrics and
representation similarity measures used in the greedy model selection
framework described in the paper.

Relevance scores (higher = better transferability):
    - LogME: Bayesian log marginal likelihood (You et al., ICML 2021)
    - LEEP: Log Expected Empirical Prediction (Nguyen et al., NeurIPS 2020)
    - GBC: Gaussian Bhattacharyya Coefficient (Pandy et al., CVPR 2022)
    - H-Score: Inter-class variance / feature redundancy (Bao et al., ICML 2019)

Redundancy scores (higher = more similar/redundant):
    - CKA: Centered Kernel Alignment (Kornblith et al., ICML 2019)
    - SVCCA: Singular Vector CCA (Raghu et al., NeurIPS 2017)

Information-theoretic selection:
    - mRMR: Max-Relevance Min-Redundancy at model group level
    - JMI: Joint Mutual Information

Unified selection:
    - selection.greedy_select: configurable greedy model subset selection
"""

from .logme import logme_score
from .leep import leep_score
from .gbc import gbc_score
from .hscore import hscore
from .cka import linear_cka, cka_pairwise_matrix
from .svcca import svcca_similarity, svcca_pairwise_matrix
from .mrmr import mrmr_select
from .jmi import jmi_select
from .selection import greedy_select

__all__ = [
    "logme_score",
    "leep_score",
    "gbc_score",
    "hscore",
    "linear_cka",
    "cka_pairwise_matrix",
    "svcca_similarity",
    "svcca_pairwise_matrix",
    "mrmr_select",
    "jmi_select",
    "greedy_select",
]

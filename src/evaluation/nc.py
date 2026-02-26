"""Neighborhood Consistency (NC) evaluation method."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pickle
import logging

import numpy as np
from sklearn.preprocessing import normalize

from ..utils import (
    batch_cosine_distances,
    batch_euclidean_distances,
    fast_sort,
    np_sort,
    pairwise_jaccard,
)


@dataclass
class NCConfig:
    """Configuration for NC evaluation."""
    n_ref: int = 5000  # Number of reference points
    k_list: List[int] = None  # k values for k-NN
    distance: str = "cosine"  # Distance metric
    seed: int = 42  # Random seed
    use_numba: bool = True  # Use numba for speedup

    def __post_init__(self):
        if self.k_list is None:
            self.k_list = [1, 10, 50, 100]


class NeighborhoodConsistency:
    """
    Compute Neighborhood Consistency (NC) scores.

    NC measures how consistently a test point's neighbors appear
    across different representation spaces.
    """

    def __init__(self, config: Optional[NCConfig] = None):
        self.config = config or NCConfig()

    def compute_nc(
        self,
        ref_representations: List[np.ndarray],
        eval_representations: List[np.ndarray],
        n_pretrain: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute NC scores for all test points.

        Args:
            ref_representations: List of reference representations from M models
                                 Each array has shape (N_ref, D_i)
            eval_representations: List of evaluation representations from M models
                                  Each array has shape (N_test, D_i)
            n_pretrain: Total number of pretraining samples (for random selection)

        Returns:
            Dictionary mapping (method, k) to NC scores
        """
        n_ensembles = len(ref_representations)
        n_test = eval_representations[0].shape[0]
        n_ref = self.config.n_ref

        # Select reference points
        np.random.seed(self.config.seed)
        if n_pretrain is not None and n_pretrain > n_ref:
            ref_idx = np.random.choice(n_pretrain, size=n_ref, replace=False)
        else:
            ref_idx = np.arange(min(n_ref, ref_representations[0].shape[0]))

        # Compute distance matrices and find neighbors
        nb_idx_list = []
        d_sorted_list = []

        for i, (ref_rep, eval_rep) in enumerate(zip(ref_representations, eval_representations)):
            # Select reference subset
            ref_subset = ref_rep[ref_idx]

            # Normalize if using cosine distance
            if self.config.distance == "cosine":
                ref_subset = normalize(ref_subset)
                eval_rep = normalize(eval_rep)

            # Compute distances
            if self.config.distance == "cosine":
                d_mat = batch_cosine_distances(eval_rep, ref_subset)
            else:
                d_mat = batch_euclidean_distances(eval_rep, ref_subset)

            # Sort to find neighbors
            if self.config.use_numba:
                nb_idx, d_sorted = fast_sort(d_mat)
            else:
                nb_idx, d_sorted = np_sort(d_mat)

            nb_idx_list.append(nb_idx)
            d_sorted_list.append(d_sorted)

        # Compute NC scores
        results = {}

        for k in self.config.k_list:
            # Get k-nearest neighbors for each ensemble member
            knn_idx = np.array([nb_idx[:, :k] for nb_idx in nb_idx_list])

            # Compute pairwise Jaccard similarity
            nc_scores = np.zeros(n_test)
            for i_sample in range(n_test):
                jaccard_sims = pairwise_jaccard(knn_idx[:, i_sample])
                nc_scores[i_sample] = np.mean(jaccard_sims)

            results[f"NC_k{k}"] = nc_scores

        # Also compute average distance (for comparison)
        d_sorted_arr = np.array(d_sorted_list)
        for k in self.config.k_list:
            avg_dist = np.mean(d_sorted_arr[:, :, :k], axis=-1)
            results[f"AvgDist_k{k}"] = -np.mean(avg_dist, axis=0)  # Negative for consistency

        return results

    def compute_feature_variance(
        self,
        eval_representations: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute feature variance across ensembles.

        This baseline measures representation consistency directly.
        """
        # Stack representations
        stacked = np.stack(eval_representations, axis=0)  # (M, N, D)

        # Compute variance per sample
        variance = np.var(stacked, axis=0)  # (N, D)

        # Sum over feature dimension
        return -np.sum(variance, axis=-1)  # (N,) negative for consistency


def load_representations(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load representations from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["emb"], data.get("label")


def save_representations(path: str, embeddings: np.ndarray, labels: Optional[np.ndarray] = None):
    """Save representations to pickle file."""
    data = {"emb": embeddings}
    if labels is not None:
        data["label"] = labels

    with open(path, "wb") as f:
        pickle.dump(data, f)

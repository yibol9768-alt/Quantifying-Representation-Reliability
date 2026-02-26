"""Utility functions for distance computation and sorting."""

from typing import List, Tuple
import numba as nb
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def get_minibatch(arr: np.ndarray, batch_size: int):
    """Generate minibatches from array."""
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]


def batch_distances(
    x: np.ndarray,
    y: np.ndarray,
    dist_fn,
    batch_size: int = 1024
) -> np.ndarray:
    """Compute distances in batches to save memory."""
    if len(y) <= batch_size:
        return dist_fn(x, y)
    y_minibatches = get_minibatch(y, batch_size)
    distances = []
    for y_minibatch in y_minibatches:
        distances.append(dist_fn(x, y_minibatch))
    return np.concatenate(distances, axis=1)


def batch_cosine_distances(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024
) -> np.ndarray:
    """Compute cosine distances in batches."""
    return batch_distances(x, y, cosine_distances, batch_size)


def batch_euclidean_distances(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024
) -> np.ndarray:
    """Compute euclidean distances in batches."""
    return batch_distances(x, y, euclidean_distances, batch_size)


@nb.njit(parallel=True)
def fast_sort(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fast parallel sorting using numba."""
    arg_idx = np.empty(a.shape, dtype=np.uint32)
    sorted_a = np.empty(a.shape, dtype=np.float32)
    for i in nb.prange(a.shape[0]):
        idx = np.argsort(a[i, :])
        arg_idx[i, :] = idx
        sorted_a[i, :] = a[i, :][idx]
    return arg_idx, sorted_a


def np_sort(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standard numpy sorting."""
    arg_idx = np.argsort(a, axis=1)
    sorted_a = np.empty(a.shape, dtype=np.float32)
    for i, idx in enumerate(arg_idx):
        sorted_a[i, :] = a[i, :][idx]
    return arg_idx, sorted_a


def pairwise_jaccard(lists: List[np.ndarray]) -> List[float]:
    """Compute pairwise Jaccard similarity between sets."""
    sets = [set(lst_) for lst_ in lists]
    jaccard_sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            n_intersections = len(sets[i].intersection(sets[j]))
            n_unions = len(sets[i].union(sets[j]))
            jaccard_sims.append(n_intersections / n_unions)
    return jaccard_sims

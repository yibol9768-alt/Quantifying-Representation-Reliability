
from typing import List, Dict

import pandas as pd
import numpy as np
import numba as nb
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


# Distance
def _get_minibatch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]


def _batch_distances(x, y, dist_fn, batch_size):
    if len(y) <= batch_size:
        return dist_fn(x, y)
    y_minibatches = _get_minibatch(y, batch_size)
    distances = []
    for y_minibatch in y_minibatches:
        distances.append(dist_fn(x, y_minibatch))
    return np.concatenate(distances, axis=1)


def batch_cosine_distances(x: np.ndarray, y: np.ndarray, batch_size: int = 1024):
    return _batch_distances(x, y, cosine_distances, batch_size)


def batch_euclidean_distances(x: np.ndarray, y: np.ndarray, batch_size: int = 1024):
    return _batch_distances(x, y, euclidean_distances, batch_size)


# Sort
@nb.njit(parallel=True)
def fastSort(a: np.ndarray):
    arg_idx = np.empty(a.shape, dtype=np.ushort)
    sorted_a = np.empty(a.shape, dtype=np.float32)
    for i in nb.prange(a.shape[0]):
        idx = np.argsort(a[i, :])
        arg_idx[i, :] = idx
        sorted_a[i, :] = a[i, :][idx]
    return arg_idx, sorted_a


def npSort(a: np.ndarray):
    arg_idx = np.argsort(a, axis=1)
    sorted_a = np.empty(a.shape, dtype=np.float32)
    for i, idx in enumerate(arg_idx):
        sorted_a[i, :] = a[i, :][idx]
    return arg_idx, sorted_a


# Set Similarity
def pairwise_jaccard(lsts: List[np.ndarray]):
    sets = [set(lst_) for lst_ in lsts]
    jaccard_sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            n_intersections = len(sets[i].intersection(sets[j]))
            n_unions = len(sets[i].union(sets[j]))
            jaccard_sims.append(n_intersections / n_unions)
    return jaccard_sims


def df_to_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return dict(zip(df.columns, df.values.T))

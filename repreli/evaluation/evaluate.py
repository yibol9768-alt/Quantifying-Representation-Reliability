
from typing import List, Tuple
import pickle
import logging
from functools import lru_cache
from itertools import combinations

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize

from evaluation.density import ProbabilityDensity
from utils import batch_cosine_distances, batch_euclidean_distances, fastSort, npSort, pairwise_jaccard
from evaluation.evaluator import ResultStorage


@lru_cache(maxsize=1)
def _preprocess(
        methods: Tuple[str],
        n_pretrain: int,
        n_ref: int,
        ref_rep_paths: Tuple[str],
        eval_rep_paths: Tuple[str],
        dist: str,
        seed: int,
        use_numba: bool,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], List[ProbabilityDensity]]:
    logging.info('Pre-processing data...')

    # Select the reference points
    np.random.seed(seed)
    ref_idx = np.random.choice(n_pretrain, size=n_ref, replace=False)

    # Compute the distance
    ref_eval_list = []
    d_mat_list = []
    density_estimator_list = []
    for ref_pth, eval_pth in zip(ref_rep_paths, eval_rep_paths):
        # Load the representations
        with open(ref_pth, 'rb') as f:
            data = pickle.load(f)
            rep_ref = data['emb'][ref_idx]

        with open(eval_pth, 'rb') as f:
            data = pickle.load(f)
            rep_eval = data['emb']

        if dist == 'cosine':
            rep_ref = normalize(rep_ref)
            rep_eval = normalize(rep_eval)

        ref_eval_list.append(rep_eval)

        # Compute the distance
        if 'Dist' in methods:
            if dist == 'cosine':
                d = batch_cosine_distances(rep_eval, rep_ref)
            else:
                d = batch_euclidean_distances(rep_eval, rep_ref)

            d_mat_list.append(d)

        # Fit the density estimators
        if 'LL' in methods:
            estimator = ProbabilityDensity(method='movMF' if dist == 'cosine' else 'gmm')
            estimator.fit(rep_ref)
            density_estimator_list.append(estimator)

    # Sort distances and find the neighbors
    d_sorted_list = []
    nb_idx_list = []
    for d in d_mat_list:
        if use_numba:
            nb_idx, d_sorted = fastSort(d)
        else:
            nb_idx, d_sorted = npSort(d)

        nb_idx_list.append(nb_idx)
        d_sorted_list.append(d_sorted)

    return ref_eval_list, np.array(d_sorted_list), nb_idx_list, density_estimator_list


def evaluate_score(
        methods: List[str],
        n_pretrain: int,
        n_ref: int,
        k_list: List[int],
        ref_rep_paths: List[str],
        eval_rep_paths: List[str],
        dist: str,
        seed: int,
        use_numba: bool = False,
        **kwargs,
) -> ResultStorage:
    n_ensembles = len(ref_rep_paths)
    ref_eval_list, d_sorted_arr, nb_idx_list, density_estimator_list = _preprocess(
        tuple(methods), n_pretrain, n_ref, tuple(ref_rep_paths), tuple(eval_rep_paths), dist, seed, use_numba
    )
    logging.info('Complete Pre-processing...')

    # Initialize the result storage
    results = ResultStorage()

    if 'Dist' in methods:
        # Compute the NC score
        n_test, n_ref = nb_idx_list[0].shape
        for k in k_list:
            knn_idx = np.array([nb_idx[:, :k] for nb_idx in nb_idx_list])
            pw_jaccard_sim = np.zeros(n_test)
            for i_emb in range(n_test):
                jaccard_sims = pairwise_jaccard(knn_idx[:, i_emb])
                pw_jaccard_sim[i_emb] = np.mean(jaccard_sims)

            results.add_score(pw_jaccard_sim, {'method': 'NC', 'k': k})

        # Compute Dist/AvgDist score
        for k in k_list:
            avgd_k = -np.mean(d_sorted_arr[:, :, :k], axis=-1)  # save negative score
            for i_ens in range(n_ensembles):
                results.add_score(avgd_k[i_ens], {'method': 'AvgDist', 'k': k, 'i_ens': i_ens})
            results.add_score(np.mean(avgd_k, axis=0), {'method': 'ens-AvgDist', 'k': k})

    # Compute the L2-norm
    if 'L2Norm' in methods and dist != 'cosine':
        for i_ens in range(n_ensembles):
            results.add_score(np.linalg.norm(ref_eval_list[i_ens], axis=-1), {'method': 'L2Norm', 'i_ens': i_ens})
        norm_score = np.linalg.norm(ref_eval_list, axis=-1)
        results.add_score(np.mean(norm_score, axis=0), {'method': 'ens-L2Norm'})

    # Compute the feature variance
    if 'FV' in methods:
        results.add_score(-np.var(ref_eval_list, axis=0).sum(axis=-1), {'method': 'FV'})

    # Compute LL
    if 'LL' in methods:
        lls = []
        for i_ens in range(n_ensembles):
            ll_score = density_estimator_list[i_ens].predict(ref_eval_list[i_ens])
            lls.append(ll_score)
            results.add_score(ll_score, {'method': 'LL', 'i_ens': i_ens})
        results.add_score(np.mean(lls, axis=0), {'method': 'ens-LL'})

    return results


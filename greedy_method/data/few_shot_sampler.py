"""
data/few_shot_sampler.py
========================
Episodic sampler for N-way K-shot few-shot learning.

Terminology
-----------
  N-way  : number of classes per episode
  K-shot : number of support samples per class
  Q      : number of query samples per class

Usage
-----
    sampler = EpisodeSampler(dataset, n_way=5, k_shot=5, n_query=15,
                             n_episodes=600)
    loader  = DataLoader(dataset, batch_sampler=sampler)

Each iteration yields a flat batch of size N*(K+Q) whose first N*K items
are the support set and the remaining N*Q are the query set.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List

import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# Build a label → indices map from any Dataset
# ---------------------------------------------------------------------------
def _build_class_map(dataset: Dataset) -> Dict[int, List[int]]:
    """Return {label: [sample_indices]} for the given dataset."""
    class_map: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_map[int(label)].append(idx)
    return dict(class_map)


# ---------------------------------------------------------------------------
# Episode Sampler
# ---------------------------------------------------------------------------
class EpisodeSampler(Sampler):
    """
    Generates episodes for prototypical / metric-learning training.

    Each episode contains:
      • support set : N * K samples (first N*K indices in the batch)
      • query set   : N * Q samples (last  N*Q indices in the batch)

    Classes are re-labelled 0 … N-1 within each episode.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset)
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.n_query    = n_query
        self.n_episodes = n_episodes

        self.class_map = _build_class_map(dataset)
        eligible = [
            cls for cls, idxs in self.class_map.items()
            if len(idxs) >= k_shot + n_query
        ]
        if len(eligible) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes with >= {k_shot + n_query} samples, "
                f"but only {len(eligible)} classes qualify."
            )
        self.eligible_classes = eligible
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_episodes):
            # Sample N classes
            classes = self.rng.sample(self.eligible_classes, self.n_way)

            support_indices: List[int] = []
            query_indices:   List[int] = []

            for cls in classes:
                pool = list(self.class_map[cls])
                selected = self.rng.sample(pool, self.k_shot + self.n_query)
                support_indices.extend(selected[: self.k_shot])
                query_indices.extend(selected[self.k_shot :])

            yield support_indices + query_indices


# ---------------------------------------------------------------------------
# Batch parser (used inside the training loop)
# ---------------------------------------------------------------------------
def parse_episode_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_way:  int,
    k_shot: int,
    n_query: int,
    device: torch.device,
):
    """
    Split a flat episode batch into support and query tensors and
    create episodic labels (0 … N-1) for both sets.

    Returns
    -------
    support_images : (N*K, C, H, W)
    support_labels : (N*K,)        – episodic 0-based class indices
    query_images   : (N*Q, C, H, W)
    query_labels   : (N*Q,)        – episodic 0-based class indices
    """
    n_support = n_way * k_shot

    sup_imgs = images[:n_support].to(device)
    qry_imgs = images[n_support:].to(device)

    # Build episodic labels: support is laid out as [class0 × K, class1 × K, ...]
    sup_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
    qry_labels = torch.arange(n_way, device=device).repeat_interleave(n_query)

    return sup_imgs, sup_labels, qry_imgs, qry_labels

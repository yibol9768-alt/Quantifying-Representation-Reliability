"""
engine/few_shot_engine.py
=========================
Episode-based few-shot training engine (Section: 改进模块 few-shot + 欧氏距离).

Implements prototypical network training:
  • For each episode: build prototypes from support set, classify queries
    via negative Euclidean distance + Softmax
  • Loss: cross-entropy over query predictions
  • Gradients update projection heads and fusion weights (not the base encoders)

The engine operates on **pre-extracted cached features** for speed, but
also supports raw-image mode for completeness.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.fusion_network import MultiViewFusion
from models.heads import PrototypeHead
from data.few_shot_sampler import EpisodeSampler, parse_episode_batch
from engine.feature_cache import collate_cached

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Few-shot engine
# ---------------------------------------------------------------------------
class FewShotEngine:
    """
    Trains and evaluates the fusion module using prototypical episodes.

    Parameters
    ----------
    fusion      : MultiViewFusion  (projection heads + fusion weights)
    n_way       : Number of classes per episode
    k_shot      : Support samples per class
    n_query     : Query samples per class
    device      : Computation device
    """

    def __init__(
        self,
        fusion:  MultiViewFusion,
        n_way:   int,
        k_shot:  int,
        n_query: int,
        device:  torch.device,
        lr:      float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.fusion   = fusion.to(device)
        self.head     = PrototypeHead()
        self.n_way    = n_way
        self.k_shot   = k_shot
        self.n_query  = n_query
        self.device   = device

        self.optimizer = AdamW(
            fusion.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ------------------------------------------------------------------
    def train_episodes(
        self,
        loader:       DataLoader,    # yields (features_dict, labels) via EpisodeSampler
        n_episodes:   int,
        scheduler:    Optional[object] = None,
    ) -> Tuple[float, float]:
        """
        Run one pass over `n_episodes` episodes.
        Returns (mean_loss, episode_accuracy).
        """
        self.fusion.train()

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for episode_idx, (features_dict, _labels) in enumerate(loader):
            if episode_idx >= n_episodes:
                break

            # Move to device
            features_dict = {k: v.to(self.device) for k, v in features_dict.items()}

            # Fuse features: (N*(K+Q), D)
            z_fused = self.fusion(features_dict)

            # Split into support and query
            n_support = self.n_way * self.k_shot
            sup_feats = z_fused[:n_support]
            qry_feats = z_fused[n_support:]

            # Build episodic labels
            sup_labels = torch.arange(
                self.n_way, device=self.device
            ).repeat_interleave(self.k_shot)
            qry_labels = torch.arange(
                self.n_way, device=self.device
            ).repeat_interleave(self.n_query)

            # Prototype classification
            log_probs = self.head(sup_feats, sup_labels, qry_feats, self.n_way)
            loss = F.nll_loss(log_probs, qry_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Metrics
            total_loss    += loss.item()
            preds          = log_probs.argmax(dim=-1)
            total_correct += (preds == qry_labels).sum().item()
            total_samples += qry_labels.size(0)

        n_eps = min(episode_idx + 1, n_episodes)
        return total_loss / n_eps, total_correct / total_samples

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_episodes(
        self,
        loader:     DataLoader,
        n_episodes: int,
    ) -> Tuple[float, float]:
        """
        Evaluate over `n_episodes` episodes.
        Returns (mean_accuracy, 95%_confidence_interval).
        """
        self.fusion.eval()

        episode_accs: List[float] = []

        for episode_idx, (features_dict, _labels) in enumerate(loader):
            if episode_idx >= n_episodes:
                break

            features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
            z_fused = self.fusion(features_dict)

            n_support = self.n_way * self.k_shot
            sup_feats = z_fused[:n_support]
            qry_feats = z_fused[n_support:]

            sup_labels = torch.arange(
                self.n_way, device=self.device
            ).repeat_interleave(self.k_shot)
            qry_labels = torch.arange(
                self.n_way, device=self.device
            ).repeat_interleave(self.n_query)

            log_probs = self.head(sup_feats, sup_labels, qry_feats, self.n_way)
            preds     = log_probs.argmax(dim=-1)
            acc       = (preds == qry_labels).float().mean().item()
            episode_accs.append(acc)

        accs   = torch.tensor(episode_accs)
        mean   = accs.mean().item()
        ci_95  = 1.96 * accs.std().item() / (len(accs) ** 0.5)
        return mean, ci_95


# ---------------------------------------------------------------------------
# Build episode DataLoader from cached features
# ---------------------------------------------------------------------------
def build_episode_loader(
    features_dict: Dict[str, torch.Tensor],   # {name: (N, D)}
    labels:        torch.Tensor,              # (N,)
    n_way:         int,
    k_shot:        int,
    n_query:       int,
    n_episodes:    int,
    seed:          int = 42,
) -> DataLoader:
    """
    Build an episodic DataLoader from cached features.

    Returns a DataLoader where each batch is one episode of size
    N * (K + Q) samples – ordered [support samples, query samples].
    """
    from engine.feature_cache import CachedFeatureDataset
    from data.few_shot_sampler import EpisodeSampler

    # Build a simple wrapped dataset where __getitem__ returns (feature_dict, label)
    dataset = CachedFeatureDataset(features_dict, labels)

    sampler = EpisodeSampler(
        dataset    = dataset,
        n_way      = n_way,
        k_shot     = k_shot,
        n_query    = n_query,
        n_episodes = n_episodes,
        seed       = seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_cached,
    )

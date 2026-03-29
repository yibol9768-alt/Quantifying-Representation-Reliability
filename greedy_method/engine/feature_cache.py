"""
engine/feature_cache.py
=======================
Extract encoder features once, save to disk, reload on demand.

This is the key performance optimisation for greedy search:
  • Each of the 10 encoders is run once over a dataset split
  • Features are saved as {cache_dir}/{dataset}/{split}/{model_name}.pt
  • Subsequent training runs load features from disk (microseconds vs seconds)

Cached file format (a dict saved with torch.save):
  {
      "features": Tensor(N, output_dim),  # float32
      "labels":   Tensor(N,),             # int64
  }
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.encoder_zoo import FrozenEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single encoder extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(
    encoder:    FrozenEncoder,
    dataset:    Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    device:     torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run `encoder` over the entire `dataset` and return (features, labels).

    Returns
    -------
    features : (N, output_dim)  float32
    labels   : (N,)             int64
    """
    encoder = encoder.to(device)
    encoder.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    all_features = []
    all_labels   = []

    for images, labels in tqdm(loader, desc=f"  {encoder.name}", leave=False):
        images = images.to(device)
        feats  = encoder(images).cpu()     # keep on CPU to save GPU memory
        all_features.append(feats)
        all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0)  # (N, D)
    labels   = torch.cat(all_labels,   dim=0)  # (N,)
    return features, labels


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------
class FeatureCache:
    """
    Manages on-disk caching of encoder features for a single dataset.

    Directory layout:
        cache_dir/
          dataset_name/
            train/
              deit_small.pt
              dinov2_small.pt
              ...
            val/
              ...
            test/
              ...
    """

    def __init__(
        self,
        cache_dir:    str,
        dataset_name: str,
    ) -> None:
        self.base_dir = os.path.join(cache_dir, dataset_name)

    def _path(self, split: str, model_name: str) -> str:
        return os.path.join(self.base_dir, split, f"{model_name}.pt")

    def exists(self, split: str, model_name: str) -> bool:
        return os.path.isfile(self._path(split, model_name))

    # ------------------------------------------------------------------
    def save(
        self,
        split:      str,
        model_name: str,
        features:   torch.Tensor,
        labels:     torch.Tensor,
    ) -> None:
        path = self._path(split, model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"features": features, "labels": labels}, path)
        logger.debug(f"Saved cache: {path}  shape={tuple(features.shape)}")

    # ------------------------------------------------------------------
    def load(
        self,
        split:      str,
        model_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self._path(split, model_name)
        data = torch.load(path, map_location="cpu")
        return data["features"], data["labels"]

    # ------------------------------------------------------------------
    def load_split(
        self,
        split:       str,
        model_names: list,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load cached features for multiple models in one call.

        Returns
        -------
        features_dict : {model_name: Tensor(N, D)}
        labels        : Tensor(N,)  (same for all models)
        """
        features_dict: Dict[str, torch.Tensor] = {}
        labels = None

        for name in model_names:
            feats, lbls = self.load(split, name)
            features_dict[name] = feats
            if labels is None:
                labels = lbls

        return features_dict, labels


# ---------------------------------------------------------------------------
# Build / populate cache for all encoders
# ---------------------------------------------------------------------------
def build_cache_for_dataset(
    encoders:    Dict[str, FrozenEncoder],
    datasets:    Dict[str, Dataset],      # {"train": ds, "val": ds, "test": ds}
    cache:       FeatureCache,
    batch_size:  int = 256,
    num_workers: int = 4,
    device:      torch.device = torch.device("cpu"),
    force:       bool = False,
) -> None:
    """
    For each (encoder, split) pair: extract features if not cached, save to disk.

    Parameters
    ----------
    force : if True, re-extract even if cache already exists.
    """
    for split_name, dataset in datasets.items():
        for model_name, encoder in encoders.items():
            if not force and cache.exists(split_name, model_name):
                logger.info(f"Cache hit  – {split_name}/{model_name}")
                continue

            logger.info(f"Extracting – {split_name}/{model_name}")
            feats, labels = extract_features(
                encoder, dataset, batch_size, num_workers, device
            )
            cache.save(split_name, model_name, feats, labels)
            encoder.cpu()  # free GPU memory after extraction
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# In-memory cached feature dataset (used as DataLoader source)
# ---------------------------------------------------------------------------
class CachedFeatureDataset(Dataset):
    """
    Wraps pre-loaded feature tensors as a Dataset.

    Each sample is (features_dict, label) where features_dict maps
    model_name → 1-D feature tensor for that sample.
    """

    def __init__(
        self,
        features_dict: Dict[str, torch.Tensor],  # {name: (N, D)}
        labels:        torch.Tensor,              # (N,)
    ) -> None:
        # Verify consistency
        n = labels.shape[0]
        for name, feats in features_dict.items():
            assert feats.shape[0] == n, (
                f"Feature tensor for '{name}' has {feats.shape[0]} rows, "
                f"expected {n}."
            )
        self.features_dict = features_dict
        self.labels        = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sample = {name: feats[idx] for name, feats in self.features_dict.items()}
        return sample, self.labels[idx]


def collate_cached(batch):
    """
    Custom collate_fn for CachedFeatureDataset.
    Returns ({name: Tensor(B, D)}, Tensor(B,)).
    """
    samples, labels = zip(*batch)
    keys = list(samples[0].keys())
    features_batch = {
        k: torch.stack([s[k] for s in samples]) for k in keys
    }
    labels_batch = torch.stack(list(labels))
    return features_batch, labels_batch

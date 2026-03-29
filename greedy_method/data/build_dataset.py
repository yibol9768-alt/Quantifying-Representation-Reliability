"""
data/build_dataset.py
=====================
Dataset factory: returns (train_dataset, val_dataset, test_dataset) for any
registered dataset.

Design notes:
  - Each split is constructed independently with its own transform to avoid
    data-leakage from augmentation into val/test.
  - For datasets without a canonical val split, a deterministic random split
    is carved from the training set using a fixed seed.
  - All datasets use a unified 224×224 ImageNet-normalised transform so that
    the same cached features are compatible across all encoders.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split
import torchvision.transforms as T
import torchvision.datasets as D

from configs.config import DATASET_REGISTRY, TrainingConfig


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_dataset(
    name: str,
    cfg: TrainingConfig,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Returns (train_dataset, val_dataset, test_dataset).

    All three splits are ready-to-use PyTorch Dataset objects.
    """
    assert name in DATASET_REGISTRY, (
        f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
    )
    root = os.path.join(cfg.data_root, name)
    os.makedirs(root, exist_ok=True)

    tr = build_train_transform(cfg.image_size)
    ev = build_eval_transform(cfg.image_size)

    _builders = {
        "gtsrb":      _build_gtsrb,
        "svhn":       _build_svhn,
        "dtd":        _build_dtd,
        "eurosat":    _build_eurosat,
        "pets":       _build_pets,
        "country211": _build_country211,
        "imagenet":   _build_imagenet,
    }
    return _builders[name](root, tr, ev, cfg.val_fraction, cfg.seed)


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------
def _build_gtsrb(root, tr, ev, val_frac, seed):
    # torchvision internally appends /gtsrb/, so pass parent directory
    parent = os.path.dirname(root)
    train_all_tr = D.GTSRB(parent, split="train", transform=tr, download=True)
    train_all_ev = D.GTSRB(parent, split="train", transform=ev, download=True)
    test_ds      = D.GTSRB(parent, split="test",  transform=ev, download=True)
    train_idx, val_idx = _random_split_indices(len(train_all_tr), val_frac, seed)
    return Subset(train_all_tr, train_idx), Subset(train_all_ev, val_idx), test_ds


def _build_svhn(root, tr, ev, val_frac, seed):
    train_all_tr = D.SVHN(root, split="train", transform=tr, download=True)
    train_all_ev = D.SVHN(root, split="train", transform=ev, download=True)
    test_ds      = D.SVHN(root, split="test",  transform=ev, download=True)
    train_idx, val_idx = _random_split_indices(len(train_all_tr), val_frac, seed)
    return Subset(train_all_tr, train_idx), Subset(train_all_ev, val_idx), test_ds


def _build_dtd(root, tr, ev, val_frac, seed):
    # DTD provides official train / val / test splits
    # torchvision internally appends /dtd/dtd/, so pass parent directory
    parent = os.path.dirname(root)
    train_ds = D.DTD(parent, split="train", transform=tr, download=True)
    val_ds   = D.DTD(parent, split="val",   transform=ev, download=True)
    test_ds  = D.DTD(parent, split="test",  transform=ev, download=True)
    return train_ds, val_ds, test_ds


def _build_eurosat(root, tr, ev, val_frac, seed):
    # EuroSAT has no official split; create train/val/test from the full set
    # torchvision internally appends /eurosat/2750/, so pass parent directory
    parent = os.path.dirname(root)
    full_tr = D.EuroSAT(parent, transform=tr, download=True)
    full_ev = D.EuroSAT(parent, transform=ev, download=True)
    n       = len(full_tr)
    n_test  = int(n * 0.20)
    n_val   = int(n * val_frac)
    n_train = n - n_test - n_val
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return (
        Subset(full_tr, train_idx),
        Subset(full_ev, val_idx),
        Subset(full_ev, test_idx),
    )


def _build_pets(root, tr, ev, val_frac, seed):
    train_all_tr = D.OxfordIIITPet(root, split="trainval", transform=tr, download=True)
    train_all_ev = D.OxfordIIITPet(root, split="trainval", transform=ev, download=True)
    test_ds      = D.OxfordIIITPet(root, split="test",     transform=ev, download=True)
    train_idx, val_idx = _random_split_indices(len(train_all_tr), val_frac, seed)
    return Subset(train_all_tr, train_idx), Subset(train_all_ev, val_idx), test_ds


def _build_country211(root, tr, ev, val_frac, seed):
    if not hasattr(D, "Country211"):
        raise RuntimeError(
            "torchvision >= 0.13 is required for Country211. "
            "Upgrade: pip install --upgrade torchvision"
        )
    train_ds = D.Country211(root, split="train", transform=tr, download=True)
    val_ds   = D.Country211(root, split="valid", transform=ev, download=True)
    test_ds  = D.Country211(root, split="test",  transform=ev, download=True)
    return train_ds, val_ds, test_ds


def _build_imagenet(root, tr, ev, val_frac, seed):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet not found at '{root}'.\n"
            "Please download ILSVRC2012 manually and place it as:\n"
            f"  {train_dir}/  (training images)\n"
            f"  {val_dir}/   (validation images)"
        )
    train_all_tr = D.ImageFolder(train_dir, transform=tr)
    train_all_ev = D.ImageFolder(train_dir, transform=ev)
    test_ds      = D.ImageFolder(val_dir,   transform=ev)
    train_idx, val_idx = _random_split_indices(len(train_all_tr), val_frac, seed)
    return Subset(train_all_tr, train_idx), Subset(train_all_ev, val_idx), test_ds


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _random_split_indices(
    n: int,
    val_fraction: float,
    seed: int,
) -> Tuple[list, list]:
    """Return (train_indices, val_indices) via a deterministic shuffle."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_val   = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    return trn_idx, val_idx

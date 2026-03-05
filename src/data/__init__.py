"""
Data loading module

Supports multiple datasets:
- Stanford Cars (original)
- CIFAR-10/100
- Flowers-102
- Oxford-IIIT Pets
- Food-101
"""

from .datasets import (
    get_dataset,
    list_datasets,
    DATASET_REGISTRY,
    DATASET_INFO,
    StanfordCarsDataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    Flowers102Dataset,
    PetsDataset,
    Food101Dataset,
)

# For backward compatibility
from .dataset import StanfordCarsDataset as OldStanfordCarsDataset

__all__ = [
    "get_dataset",
    "list_datasets",
    "DATASET_REGISTRY",
    "DATASET_INFO",
    "StanfordCarsDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "Flowers102Dataset",
    "PetsDataset",
    "Food101Dataset",
]

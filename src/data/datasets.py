"""Dataset utilities for loading common benchmarks."""

from typing import Tuple, Optional
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class StandardDataset(Dataset):
    """Wrapper for standard vision datasets with preprocessing."""

    DATASET_INFO = {
        "cifar10": {"n_classes": 10, "img_size": 32, "n_train": 50000, "n_test": 10000},
        "cifar100": {"n_classes": 100, "img_size": 32, "n_train": 50000, "n_test": 10000},
        "stl10": {"n_classes": 10, "img_size": 96, "n_train": 5000, "n_test": 8000},
        "imagenet": {"n_classes": 1000, "img_size": 224, "n_train": 1281167, "n_test": 50000},
    }

    def __init__(
        self,
        name: str,
        root: str = "./data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ):
        self.name = name.lower()
        self.root = Path(root)
        self.train = train

        # Default transform for ViT-based models
        if transform is None:
            transform = self._get_default_transform()

        self.transform = transform

        # Load dataset
        self.dataset = self._load_dataset(download)

        self.data = self.dataset.data
        self.targets = np.array(self.dataset.targets)
        self.info = self.DATASET_INFO[self.name]

    def _get_default_transform(self, img_size: int = 224) -> transforms.Compose:
        """Get default transform for ViT models."""
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_dataset(self, download: bool):
        """Load the underlying dataset."""
        name = self.name

        if name == "cifar10":
            return datasets.CIFAR10(
                self.root, train=self.train, download=download, transform=self.transform
            )
        elif name == "cifar100":
            return datasets.CIFAR100(
                self.root, train=self.train, download=download, transform=self.transform
            )
        elif name == "stl10":
            split = "train" if self.train else "test"
            return datasets.STL10(
                self.root, split=split, download=download, transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        return self.dataset[idx]

    @property
    def n_classes(self) -> int:
        return self.info["n_classes"]

    @property
    def n_samples(self) -> int:
        return len(self)


def get_dataloader(
    dataset_name: str,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 4,
    **kwargs,
):
    """Get a DataLoader for a dataset."""
    from torch.utils.data import DataLoader

    dataset = StandardDataset(dataset_name, train=train, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )

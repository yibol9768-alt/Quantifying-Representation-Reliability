"""Multi-dataset loader for feature classification."""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Tuple, Optional
from pathlib import Path


def get_transforms(model_type: str = "mae", train: bool = False):
    """Get transforms based on model type.

    Args:
        model_type: Type of model (mae, clip, dino, fusion)
        train: Whether to use training augmentations

    Returns:
        torchvision transforms
    """
    if model_type == "clip":
        # CLIP uses 224x224 and specific normalization
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        if train:
            return transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        # MAE and DINO use ImageNet normalization
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        if train:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])


# ==================== CIFAR-10 ====================
def get_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== CIFAR-100 ====================
def get_cifar100_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-100 train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== STL-10 ====================
def get_stl10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get STL-10 train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    train_dataset = datasets.STL10(
        root=data_dir, split="train", download=True, transform=train_transform
    )
    test_dataset = datasets.STL10(
        root=data_dir, split="test", download=True, transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== Tiny ImageNet ====================
def get_tiny_imagenet_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get Tiny ImageNet train and test dataloaders.

    Note: Tiny ImageNet needs manual download from:
    http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    data_path = Path(data_dir) / "tiny-imagenet-200"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Tiny ImageNet not found at {data_path}.\n"
            "Please download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
            f"And extract to: {data_path}"
        )

    train_dataset = datasets.ImageFolder(
        data_path / "train", transform=train_transform
    )

    # Tiny ImageNet test uses val folder with proper structure
    test_dataset = datasets.ImageFolder(
        data_path / "val", transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== Caltech-101 ====================
def get_caltech101_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get Caltech-101 train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    full_dataset = datasets.Caltech101(
        root=data_dir, download=True
    )

    # Split into train/test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Need to wrap with transform
    class TransformDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img.convert('RGB'))
            return img, label

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # Wrap with transforms
    train_dataset = TransformDataset(
        torch.utils.data.Subset(full_dataset, train_dataset.indices),
        train_transform
    )
    test_dataset = TransformDataset(
        torch.utils.data.Subset(full_dataset, test_dataset.indices),
        test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== Flowers-102 ====================
def get_flowers102_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get Oxford Flowers-102 train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    # Flowers102 uses 'train' and 'test' splits
    train_dataset = datasets.Flowers102(
        root=data_dir, split="train", download=True, transform=train_transform
    )
    test_dataset = datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== Food-101 ====================
def get_food101_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get Food-101 train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    train_dataset = datasets.Food101(
        root=data_dir, split="train", download=True, transform=train_transform
    )
    test_dataset = datasets.Food101(
        root=data_dir, split="test", download=True, transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== Oxford Pets ====================
def get_pets_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get Oxford-IIIT Pets train and test dataloaders."""
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    # OxfordIIITPet uses 'trainval' and 'test' splits
    train_dataset = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", download=True, transform=train_transform
    )
    test_dataset = datasets.OxfordIIITPet(
        root=data_dir, split="test", download=True, transform=test_transform
    )

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== CUB-200 ====================
def get_cub200_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get CUB-200-2011 train and test dataloaders.

    Note: CUB-200 needs manual download from:
    https://data.caltech.edu/records/65de2-v2p15
    """
    train_transform = get_transforms(model_type, train=True)
    test_transform = get_transforms(model_type, train=False)

    data_path = Path(data_dir) / "CUB_200_2011"

    if not data_path.exists():
        raise FileNotFoundError(
            f"CUB-200 not found at {data_path}.\n"
            "Please download from: https://data.caltech.edu/records/65de2-v2p15\n"
            f"And extract to: {data_path}"
        )

    # Read train/test split
    images_folder = data_path / "images"
    train_test_split = data_path / "train_test_split.txt"

    if not train_test_split.exists():
        raise FileNotFoundError(f"Train/test split file not found: {train_test_split}")

    # Parse split file
    with open(train_test_split) as f:
        lines = f.readlines()

    train_indices = []
    test_indices = []
    for line in lines:
        img_id, is_training = line.strip().split()
        if is_training == "1":
            train_indices.append(int(img_id) - 1)  # 0-indexed
        else:
            test_indices.append(int(img_id) - 1)

    # Create full dataset
    full_dataset = datasets.ImageFolder(images_folder)

    # Split based on indices
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Wrap with transforms
    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = TransformSubset(train_dataset, train_transform)
    test_dataset = TransformSubset(test_dataset, test_transform)

    return _create_loaders(train_dataset, test_dataset, batch_size, num_workers)


# ==================== Helper Functions ====================
def _create_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders from datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# ==================== Unified Interface ====================
def get_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for the specified dataset.

    Args:
        dataset_name: Name of dataset (cifar10, cifar100, stl10, etc.)
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of worker processes
        model_type: Type of model for transforms

    Returns:
        train_loader, test_loader
    """
    dataset_loaders = {
        "cifar10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
        "stl10": get_stl10_loaders,
        "tiny_imagenet": get_tiny_imagenet_loaders,
        "caltech101": get_caltech101_loaders,
        "flowers102": get_flowers102_loaders,
        "food101": get_food101_loaders,
        "pets": get_pets_loaders,
        "cub200": get_cub200_loaders,
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(dataset_loaders.keys())}"
        )

    return dataset_loaders[dataset_name](
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model_type=model_type
    )

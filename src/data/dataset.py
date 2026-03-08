"""CIFAR-100 dataset loader."""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_transforms(model_type: str = "mae"):
    """Get transforms based on model type.

    Args:
        model_type: Type of model (mae, clip, dino, fusion)

    Returns:
        torchvision transforms
    """
    if model_type == "clip":
        # CLIP uses 224x224 and specific normalization
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])
    else:
        # MAE and DINO use ImageNet normalization
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])


def get_cifar100_loaders(data_dir: str, batch_size: int, num_workers: int, model_type: str = "mae"):
    """Get CIFAR-100 train and test dataloaders.

    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        model_type: Type of model for transforms

    Returns:
        train_loader, test_loader
    """
    transform = get_transforms(model_type)

    # Training data with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

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

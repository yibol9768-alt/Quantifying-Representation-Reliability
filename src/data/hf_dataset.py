"""Dataset utilities - manual download only."""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple
from pathlib import Path


# Dataset info
DATASET_INFO = {
    # Existing datasets
    "cifar10": {"num_classes": 10},
    "cifar100": {"num_classes": 100},
    "stl10": {"num_classes": 10},
    "caltech101": {"num_classes": 101},
    "flowers102": {"num_classes": 102},
    "food101": {"num_classes": 101},
    "pets": {"num_classes": 37},
    "tiny_imagenet": {"num_classes": 200},
    "cub200": {"num_classes": 200},
    # CLIP paper datasets
    "mnist": {"num_classes": 10},
    "svhn": {"num_classes": 10},
    "sun397": {"num_classes": 397},
    "stanford_cars": {"num_classes": 196},
    "dtd": {"num_classes": 47},
    "eurosat": {"num_classes": 10},
    "gtsrb": {"num_classes": 43},
    "country211": {"num_classes": 211},
    "aircraft": {"num_classes": 100},
    "resisc45": {"num_classes": 45},
}


def get_transforms(model_type: str = "mae", train: bool = False, dataset: str = None):
    """Get transforms based on model type and dataset.

    Args:
        model_type: Model type for normalization (mae, clip, fusion, etc.)
        train: Whether to use training augmentations
        dataset: Dataset name for special handling (e.g., MNIST grayscale)

    Returns:
        transforms.Compose object
    """
    # Get normalization based on model type
    if model_type == "fusion":
        normalize = None
    elif model_type in ("clip", "clip_large", "openclip"):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=mean, std=std)

    # MNIST special handling: grayscale to RGB conversion
    is_mnist = dataset == "mnist"

    if train:
        ops = []
        if is_mnist:
            # MNIST is 28x28, resize to 256 first
            ops.extend([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            ])
        else:
            ops.extend([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
        ops.append(transforms.ToTensor())
        if normalize is not None:
            ops.append(normalize)
        return transforms.Compose(ops)
    else:
        ops = []
        if is_mnist:
            ops.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            ])
        else:
            ops.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
        ops.append(transforms.ToTensor())
        if normalize is not None:
            ops.append(normalize)
        return transforms.Compose(ops)


class ImageFolderDataset(Dataset):
    """Generic image folder dataset.

    Expected structure:
    data/
    └── cifar100/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.png
        │   │   └── img2.png
        │   └── class2/
        └── test/
            ├── class1/
            └── class2/
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae"
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for specified dataset.

    Args:
        dataset: Dataset name
        data_dir: Data directory (must contain pre-downloaded data)
        batch_size: Batch size
        num_workers: Number of workers
        model_type: Model type for transforms

    Returns:
        train_loader, test_loader
    """
    train_tf = get_transforms(model_type, train=True, dataset=dataset)
    test_tf = get_transforms(model_type, train=False, dataset=dataset)

    dataset_path = Path(data_dir) / dataset

    # Check if dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please download manually. See README for instructions."
        )

    # Load from image folders
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train_set = ImageFolderDataset(str(train_path), transform=train_tf)
    test_set = ImageFolderDataset(str(test_path), transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Loaded {dataset}: {len(train_set)} train, {len(test_set)} test, {len(train_set.classes)} classes")

    return train_loader, test_loader

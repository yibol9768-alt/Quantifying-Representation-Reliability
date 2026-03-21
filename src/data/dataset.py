"""Dataset utilities - manual download only."""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    # Normalization groups: model_type -> (mean, std)
    _CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    _HALF_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    _IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    _NORM_MAP = {
        # CLIP-style normalization
        "clip": _CLIP_NORM, "clip_base32": _CLIP_NORM, "clip_large": _CLIP_NORM,
        "openclip": _CLIP_NORM,
        # 0.5/0.5 normalization
        "beit": _HALF_NORM, "beit_large": _HALF_NORM,
        "data2vec": _HALF_NORM,
        "siglip": _HALF_NORM,
    }

    if model_type == "fusion":
        normalize = None
    else:
        mean, std = _NORM_MAP.get(model_type, _IMAGENET_NORM)
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

    def __init__(
        self,
        root: str,
        transform=None,
        fewshot_min: Optional[int] = None,
        fewshot_max: Optional[int] = None,
        seed: int = 42,
        samples: Optional[List[Tuple[Path, int]]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        if samples is None:
            self.samples = self._scan_samples()
        else:
            self.samples = list(samples)

        if samples is None and fewshot_min is not None and fewshot_max is not None:
            self.samples = self._apply_fewshot(
                self.samples,
                fewshot_min=fewshot_min,
                fewshot_max=fewshot_max,
                seed=seed,
            )

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    @staticmethod
    def _apply_fewshot(
        samples: List[Tuple[Path, int]],
        *,
        fewshot_min: int,
        fewshot_max: int,
        seed: int,
    ) -> List[Tuple[Path, int]]:
        """Keep only a small deterministic subset of train images per class."""
        if fewshot_min <= 0 or fewshot_max <= 0:
            raise ValueError("fewshot_min and fewshot_max must be positive integers.")
        if fewshot_min > fewshot_max:
            raise ValueError("fewshot_min must be <= fewshot_max.")

        by_class = defaultdict(list)
        for img_path, label in samples:
            by_class[label].append((img_path, label))

        selected = []
        rng = random.Random(seed)
        for label in sorted(by_class):
            class_samples = sorted(by_class[label], key=lambda item: str(item[0]))
            rng.shuffle(class_samples)
            target_count = rng.randint(fewshot_min, fewshot_max)
            selected.extend(class_samples[:min(target_count, len(class_samples))])

        selected.sort(key=lambda item: (item[1], str(item[0])))
        return selected

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
    model_type: str = "mae",
    fewshot_min: Optional[int] = None,
    fewshot_max: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for specified dataset.

    Args:
        dataset: Dataset name
        data_dir: Data directory (must contain pre-downloaded data)
        batch_size: Batch size
        num_workers: Number of workers
        model_type: Model type for transforms
        fewshot_min: Min train images per class. None disables few-shot.
        fewshot_max: Max train images per class. None disables few-shot.
        seed: Sampling seed for deterministic few-shot subsets.

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

    train_set = ImageFolderDataset(
        str(train_path),
        transform=train_tf,
        fewshot_min=fewshot_min,
        fewshot_max=fewshot_max,
        seed=seed,
    )
    test_set = ImageFolderDataset(str(test_path), transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if fewshot_min is not None and fewshot_max is not None:
        print(
            f"Loaded {dataset}: {len(train_set)} few-shot train "
            f"({fewshot_min}-{fewshot_max} per class), {len(test_set)} test, "
            f"{len(train_set.classes)} classes"
        )
    else:
        print(f"Loaded {dataset}: {len(train_set)} train, {len(test_set)} test, {len(train_set.classes)} classes")

    return train_loader, test_loader


def _split_train_val_samples(
    samples: List[Tuple[Path, int]],
    *,
    val_ratio: float,
    split_seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Deterministically split train samples into train/val per class."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    if val_ratio == 0.0:
        return list(samples), []

    by_class: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(split_seed)
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    for label in sorted(by_class):
        class_samples = sorted(by_class[label], key=lambda item: str(item[0]))
        rng.shuffle(class_samples)

        if len(class_samples) <= 1:
            train_samples.extend(class_samples)
            continue

        val_count = int(math.ceil(len(class_samples) * val_ratio))
        val_count = min(max(val_count, 1), len(class_samples) - 1)

        val_split = class_samples[:val_count]
        train_split = class_samples[val_count:]

        train_samples.extend(train_split)
        val_samples.extend(val_split)

    train_samples.sort(key=lambda item: (item[1], str(item[0])))
    val_samples.sort(key=lambda item: (item[1], str(item[0])))
    return train_samples, val_samples


def get_train_val_test_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae",
    fewshot_min: Optional[int] = None,
    fewshot_max: Optional[int] = None,
    seed: int = 42,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Get train/val/test dataloaders with a deterministic train-val split."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    train_tf = get_transforms(model_type, train=True, dataset=dataset)
    eval_tf = get_transforms(model_type, train=False, dataset=dataset)

    dataset_path = Path(data_dir) / dataset
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please download manually. See README for instructions."
        )
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    base_train = ImageFolderDataset(
        str(train_path),
        transform=None,
        fewshot_min=fewshot_min,
        fewshot_max=fewshot_max,
        seed=seed,
    )
    train_samples, val_samples = _split_train_val_samples(
        base_train.samples,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    train_set = ImageFolderDataset(
        str(train_path),
        transform=train_tf,
        samples=train_samples,
    )
    val_set = None
    if val_samples:
        val_set = ImageFolderDataset(
            str(train_path),
            transform=eval_tf,
            samples=val_samples,
        )
    test_set = ImageFolderDataset(str(test_path), transform=eval_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_count = 0 if val_set is None else len(val_set)
    print(
        f"Loaded {dataset}: {len(train_set)} train, {val_count} val, "
        f"{len(test_set)} test, {len(train_set.classes)} classes"
    )
    return train_loader, val_loader, test_loader


def get_feature_split_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    model_type: str = "mae",
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> Dict[str, DataLoader]:
    """Get eval-only train/val/test loaders for feature extraction and analysis."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    eval_tf = get_transforms(model_type, train=False, dataset=dataset)
    dataset_path = Path(data_dir) / dataset
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please download manually. See README for instructions."
        )
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    base_train = ImageFolderDataset(str(train_path), transform=None)
    train_samples, val_samples = _split_train_val_samples(
        base_train.samples,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    split_datasets = {
        "train": ImageFolderDataset(str(train_path), transform=eval_tf, samples=train_samples),
        "test": ImageFolderDataset(str(test_path), transform=eval_tf),
    }
    if val_samples:
        split_datasets["val"] = ImageFolderDataset(
            str(train_path),
            transform=eval_tf,
            samples=val_samples,
        )

    loaders = {}
    for split_name, split_dataset in split_datasets.items():
        loaders[split_name] = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    print(
        f"Loaded feature splits for {dataset}: "
        f"{len(split_datasets['train'])} train, "
        f"{len(split_datasets.get('val', []))} val, "
        f"{len(split_datasets['test'])} test"
    )
    return loaders

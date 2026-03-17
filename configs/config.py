"""Configuration for feature classification experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Dataset configurations
DATASET_CONFIGS = {
    # Existing datasets
    "cifar10": {
        "num_classes": 10,
        "img_size": 32,
        "description": "CIFAR-10: 10 classes, 32x32 images"
    },
    "cifar100": {
        "num_classes": 100,
        "img_size": 32,
        "description": "CIFAR-100: 100 classes, 32x32 images"
    },
    "stl10": {
        "num_classes": 10,
        "img_size": 96,
        "description": "STL-10: 10 classes, 96x96 images"
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "img_size": 64,
        "description": "Tiny ImageNet: 200 classes, 64x64 images"
    },
    "caltech101": {
        "num_classes": 101,
        "img_size": 224,
        "description": "Caltech-101: 101 object categories"
    },
    "flowers102": {
        "num_classes": 102,
        "img_size": 224,
        "description": "Oxford Flowers-102: 102 flower categories"
    },
    "food101": {
        "num_classes": 101,
        "img_size": 224,
        "description": "Food-101: 101 food categories"
    },
    "pets": {
        "num_classes": 37,
        "img_size": 224,
        "description": "Oxford-IIIT Pets: 37 pet breeds"
    },
    "cub200": {
        "num_classes": 200,
        "img_size": 224,
        "description": "CUB-200-2011: 200 bird species"
    },
    # CLIP paper datasets
    "mnist": {
        "num_classes": 10,
        "img_size": 224,
        "description": "MNIST: 10 handwritten digit classes"
    },
    "svhn": {
        "num_classes": 10,
        "img_size": 224,
        "description": "SVHN: 10 street view house number classes"
    },
    "sun397": {
        "num_classes": 397,
        "img_size": 224,
        "description": "SUN397: 397 scene categories"
    },
    "stanford_cars": {
        "num_classes": 196,
        "img_size": 224,
        "description": "Stanford Cars: 196 car classes"
    },
    "dtd": {
        "num_classes": 47,
        "img_size": 224,
        "description": "DTD: 47 texture categories"
    },
    "eurosat": {
        "num_classes": 10,
        "img_size": 224,
        "description": "EuroSAT: 10 satellite image classes"
    },
    "gtsrb": {
        "num_classes": 43,
        "img_size": 224,
        "description": "GTSRB: 43 traffic sign classes"
    },
    "country211": {
        "num_classes": 211,
        "img_size": 224,
        "description": "Country211: 211 country identification"
    },
    "aircraft": {
        "num_classes": 100,
        "img_size": 224,
        "description": "FGVC Aircraft: 100 aircraft types"
    },
    "resisc45": {
        "num_classes": 45,
        "img_size": 224,
        "description": "Resisc45: 45 remote sensing scene classes"
    },
}


@dataclass
class Config:
    """Experiment configuration."""

    # Model settings
    model_type: str = "mae"  # mae, clip, dino, fusion

    # Dataset settings
    dataset: str = "cifar100"

    # Training settings
    epochs: int = 10
    lr: float = 0.001
    batch_size: int = 128
    weight_decay: float = 0.01

    # MLP settings
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3

    # Data settings
    data_dir: str = "./data"
    num_workers: int = 4

    # Device settings
    device: str = "cuda:0"

    # Feature dimensions
    FEATURE_DIMS: Dict = field(default_factory=lambda: {
        # Vision Transformer (supervised)
        "vit": 768, "vit_large": 1024,
        # DeiT
        "deit_small": 384, "deit_base": 768,
        # Swin Transformer
        "swin_tiny": 768, "swin": 1024,
        # BEiT
        "beit": 768, "beit_large": 1024,
        # Data2Vec
        "data2vec": 768,
        # MAE (self-supervised)
        "mae": 768, "mae_large": 1024,
        # DINOv2 (self-supervised)
        "dinov2_small": 384, "dino": 768, "dinov2_large": 1024,
        # CLIP
        "clip_base32": 768, "clip": 768, "clip_large": 1024, "openclip": 768,
        # SigLIP
        "siglip": 768,
        # ConvNeXt
        "convnext_tiny": 768, "convnext": 1024, "convnext_large": 1536,
        # ResNet
        "resnet50": 2048, "resnet101": 2048,
    })

    @property
    def feature_dim(self) -> int:
        """Get feature dimension based on model type."""
        if self.model_type == "fusion":
            return sum(self.FEATURE_DIMS.values())  # default 2304 (mae+clip+dino)
        return self.FEATURE_DIMS.get(self.model_type, 768)

    @property
    def num_classes(self) -> int:
        """Get number of classes for current dataset."""
        return DATASET_CONFIGS.get(self.dataset, {}).get("num_classes", 100)

    @property
    def dataset_info(self) -> str:
        """Get dataset description."""
        return DATASET_CONFIGS.get(self.dataset, {}).get("description", "Unknown dataset")

    @staticmethod
    def list_datasets() -> List[str]:
        """List all available datasets."""
        return list(DATASET_CONFIGS.keys())

    @staticmethod
    def get_dataset_info() -> str:
        """Get formatted dataset information."""
        lines = ["Available datasets:"]
        for name, info in DATASET_CONFIGS.items():
            lines.append(f"  - {name}: {info['description']}")
        return "\n".join(lines)

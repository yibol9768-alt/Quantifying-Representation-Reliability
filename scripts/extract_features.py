#!/usr/bin/env python3
"""
使用新模型（CLIP、DINO、MAE等）提取特征，保存为原代码兼容的格式。

输出格式与原 repreli 代码一致：
    pickle 文件，包含 {'emb': np.ndarray, 'label': np.ndarray}

Usage:
    python scripts/extract_features.py --models clip_vit_b16 dinov2_vit_b14 --dataset cifar10
"""

import argparse
import os
import sys
from pathlib import Path
import pickle

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模型封装
from src.models import get_model, list_available_models


def get_transform(img_size: int = 224):
    """获取适用于 ViT 模型的图像变换"""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def extract_and_save(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    batch_size: int = 64,
    device: str = "cuda",
    train: bool = True,
    **model_kwargs,
):
    """提取特征并保存为原代码兼容的 pkl 格式"""
    from torchvision import datasets

    split = "train" if train else "test"
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}, Split: {split}")
    print(f"{'='*50}")

    # 加载模型
    print("Loading model...")
    extractor = get_model(model_name, device=device, **model_kwargs)
    print(f"Feature dim: {extractor.feature_dim}")

    # 加载数据集
    print("Loading dataset...")
    transform = get_transform(img_size=224)

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
        n_classes = 10
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root="./data", train=train, download=True, transform=transform
        )
        n_classes = 100
    elif dataset_name == "stl10":
        split_name = "train" if train else "test"
        dataset = datasets.STL10(
            root="./data", split=split_name, download=True, transform=transform
        )
        n_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 提取特征
    print("Extracting features...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting"):
            features = extractor.extract_features(images)
            all_features.append(features)
            all_labels.append(labels.numpy())

    # 合并
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Extracted: {features.shape}, Labels: {labels.shape}")

    # 保存为原代码兼容的格式
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{model_name}_{dataset_name}_{split}.pkl"
    )

    data = {"emb": features, "label": labels}
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved to: {output_path}")
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Extract features using pretrained models")

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to use (e.g., clip_vit_b16 dinov2_vit_b14 mae_vit_b16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "stl10"],
        help="Dataset name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./features",
        help="Output directory for features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which split to extract",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for m in list_available_models():
            print(f"  - {m}")
        return

    # 验证模型
    available = list_available_models()
    for model in args.models:
        if model not in available:
            print(f"Error: Unknown model '{model}'")
            print(f"Available: {available}")
            sys.exit(1)

    # 提取特征
    model_kwargs = {}
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for model_name in args.models:
        for split in splits:
            extract_and_save(
                model_name=model_name,
                dataset_name=args.dataset,
                output_dir=args.output,
                batch_size=args.batch_size,
                device=args.device,
                train=(split == "train"),
                **model_kwargs,
            )

    print("\n" + "=" * 50)
    print("Feature extraction complete!")
    print("=" * 50)
    print(f"\nOutput files in {args.output}:")
    for f in sorted(os.listdir(args.output)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()

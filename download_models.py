"""Download models and datasets for the server."""

import os
import subprocess
from pathlib import Path


def download_model(model_name: str, local_dir: str):
    """Download model from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}...")
    print(f"{'='*60}")

    Path(local_dir).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        model_name,
        "--local-dir", local_dir
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Downloaded to {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False

    return True


def download_all_models():
    """Download all required models."""
    models = {
        "facebook/vit-mae-base": "./models/vit-mae-base",
        "openai/clip-vit-base-patch16": "./models/clip-vit-base-patch16",
        "facebook/dinov2-base": "./models/dinov2-base",
    }

    print("="*60)
    print("Downloading all models...")
    print("="*60)

    for hf_name, local_path in models.items():
        download_model(hf_name, local_path)

    print("\n" + "="*60)
    print("All models downloaded!")
    print("="*60)


def download_cifar100():
    """Download and convert CIFAR-100 to image folders."""
    import torchvision
    from PIL import Image
    from tqdm import tqdm

    print("\n" + "="*60)
    print("Downloading CIFAR-100...")
    print("="*60)

    raw_dir = "./data_raw"
    output_dir = "./data/cifar100"

    # Download raw data
    train_set = torchvision.datasets.CIFAR100(raw_dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(raw_dir, train=False, download=True)

    # Convert to image folders
    for split, dataset in [("train", train_set), ("test", test_set)]:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class folders
        for class_name in dataset.classes:
            (split_dir / class_name).mkdir(exist_ok=True)

        # Save images
        print(f"Converting {split} set...")
        for i in tqdm(range(len(dataset))):
            img, label = dataset[i]
            class_name = dataset.classes[label]
            img_path = split_dir / class_name / f"{i}.png"
            img.save(img_path)

    print(f"\n✓ CIFAR-100 saved to {output_dir}")
    print(f"  Train: {len(train_set)} images")
    print(f"  Test: {len(test_set)} images")


def download_cifar10():
    """Download and convert CIFAR-10 to image folders."""
    import torchvision
    from PIL import Image
    from tqdm import tqdm

    print("\n" + "="*60)
    print("Downloading CIFAR-10...")
    print("="*60)

    raw_dir = "./data_raw"
    output_dir = "./data/cifar10"

    train_set = torchvision.datasets.CIFAR10(raw_dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(raw_dir, train=False, download=True)

    for split, dataset in [("train", train_set), ("test", test_set)]:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_name in dataset.classes:
            (split_dir / class_name).mkdir(exist_ok=True)

        print(f"Converting {split} set...")
        for i in tqdm(range(len(dataset))):
            img, label = dataset[i]
            class_name = dataset.classes[label]
            img_path = split_dir / class_name / f"{i}.png"
            img.save(img_path)

    print(f"\n✓ CIFAR-10 saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download models and datasets")
    parser.add_argument("--models", action="store_true", help="Download all models")
    parser.add_argument("--cifar10", action="store_true", help="Download CIFAR-10")
    parser.add_argument("--cifar100", action="store_true", help="Download CIFAR-100")
    parser.add_argument("--all", action="store_true", help="Download everything")

    args = parser.parse_args()

    if args.all:
        download_all_models()
        download_cifar10()
        download_cifar100()
    else:
        if args.models:
            download_all_models()
        if args.cifar10:
            download_cifar10()
        if args.cifar100:
            download_cifar100()

    if not any([args.models, args.cifar10, args.cifar100, args.all]):
        print("Usage:")
        print("  python download_models.py --all       # Download everything")
        print("  python download_models.py --models    # Download models only")
        print("  python download_models.py --cifar100  # Download CIFAR-100")
        print("  python download_models.py --cifar10   # Download CIFAR-10")

"""Download models and datasets for the server."""

import subprocess
import shutil
from pathlib import Path

DEFAULT_STORAGE_DIR = "."


def download_model(model_name: str, local_dir: str):
    """Download model from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}...")
    print(f"{'='*60}")

    Path(local_dir).parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("hf") is not None:
        cmd = [
            "hf", "download",
            model_name,
            "--local-dir", local_dir,
        ]
    elif shutil.which("huggingface-cli") is not None:
        cmd = [
            "huggingface-cli", "download",
            model_name,
            "--local-dir", local_dir,
        ]
    else:
        print("✗ Neither `hf` nor `huggingface-cli` is installed.")
        print("  Install one with: pip install -U huggingface_hub")
        return False

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Downloaded to {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False

    return True


def get_storage_paths(storage_dir: str):
    """Resolve large-file directories from a shared storage root."""
    root = Path(storage_dir)
    return {
        "root": root,
        "models": root / "models",
        "data": root / "data",
        "data_raw": root / "data_raw",
    }


def download_all_models(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download all required models."""
    paths = get_storage_paths(storage_dir)
    models = {
        # Supported backbones validated by the current extractor implementation.
        "facebook/vit-mae-base": str(paths["models"] / "vit-mae-base"),
        "openai/clip-vit-base-patch16": str(paths["models"] / "clip-vit-base-patch16"),
        "facebook/dinov2-base": str(paths["models"] / "dinov2-base"),
        # Vision Transformer series
        "google/vit-base-patch16-224": str(paths["models"] / "vit-base-patch16"),
        "microsoft/swin-base-patch4-window7-224": str(paths["models"] / "swin-base"),
        "microsoft/beit-base-patch16-224-pt22k": str(paths["models"] / "beit-base"),
        "facebook/data2vec-vision-base": str(paths["models"] / "data2vec-vision-base"),
        # CLIP series
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": str(paths["models"] / "openclip-vit-b32"),
        "google/siglip-base-patch16-224": str(paths["models"] / "siglip-base-patch16-224"),
        # Modern CNN
        "facebook/convnext-base-224": str(paths["models"] / "convnext-base"),
    }

    print("="*60)
    print("Downloading all models...")
    print("="*60)

    for hf_name, local_path in models.items():
        download_model(hf_name, local_path)

    print("\n" + "="*60)
    print("All models downloaded!")
    print("="*60)


def download_cifar100(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert CIFAR-100 to image folders."""
    import torchvision
    from PIL import Image
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading CIFAR-100...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "cifar100")

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


def download_cifar10(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert CIFAR-10 to image folders."""
    import torchvision
    from PIL import Image
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading CIFAR-10...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "cifar10")

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


def _convert_to_image_folders(dataset, output_dir: Path, split_name: str):
    """Convert a torchvision dataset to image folder format."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Get class names
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    elif hasattr(dataset, 'classes'):
        # For some datasets like MNIST
        class_names = [str(i) for i in range(getattr(dataset, 'num_classes', 10))]
    else:
        # Fallback: infer from unique labels
        class_names = sorted(set([dataset[i][1] for i in range(len(dataset))]))

    # Create class folders
    for class_name in class_names:
        (split_dir / str(class_name)).mkdir(exist_ok=True)

    # Save images
    print(f"Converting {split_name} set...")
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        class_name = str(class_names[label] if label < len(class_names) else label)
        img_path = split_dir / class_name / f"{i}.png"
        img.save(img_path)


def download_mnist(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert MNIST to image folders (grayscale to RGB)."""
    import torchvision
    from PIL import Image
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading MNIST...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "mnist")

    train_set = torchvision.datasets.MNIST(raw_dir, train=True, download=True)
    test_set = torchvision.datasets.MNIST(raw_dir, train=False, download=True)

    # Convert and save as RGB
    for split, dataset in [("train", train_set), ("test", test_set)]:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            (split_dir / str(i)).mkdir(exist_ok=True)

        print(f"Converting {split} set...")
        for i in tqdm(range(len(dataset))):
            img, label = dataset[i]
            # Convert grayscale to RGB
            img_rgb = Image.merge("RGB", (img, img, img))
            img_path = split_dir / str(label) / f"{i}.png"
            img_rgb.save(img_path)

    print(f"\n✓ MNIST saved to {output_dir}")
    print(f"  Train: {len(train_set)} images")
    print(f"  Test: {len(test_set)} images")


def download_svhn(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert SVHN to image folders."""
    import torchvision
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading SVHN...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "svhn")

    train_set = torchvision.datasets.SVHN(raw_dir, split='train', download=True)
    test_set = torchvision.datasets.SVHN(raw_dir, split='test', download=True)

    # SVHN uses labels 0-9 but starting from 1 internally
    for split, dataset in [("train", train_set), ("test", test_set)]:
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            (split_dir / str(i)).mkdir(exist_ok=True)

        print(f"Converting {split} set...")
        for i in tqdm(range(len(dataset))):
            img, label = dataset[i]
            # SVHN labels are 1-10, convert to 0-9
            label = int(label) - 1 if int(label) == 10 else int(label)
            img_path = split_dir / str(label) / f"{i}.png"
            img.save(img_path)

    print(f"\n✓ SVHN saved to {output_dir}")


def download_dtd(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert DTD (Describable Textures Dataset)."""
    import torchvision
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading DTD...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "dtd")

    # DTD has train/test/val splits
    train_set = torchvision.datasets.DTD(root=raw_dir, split='train', download=True)
    test_set = torchvision.datasets.DTD(root=raw_dir, split='test', download=True)
    val_set = torchvision.datasets.DTD(root=raw_dir, split='val', download=True)

    # Combine train and val for training
    for split, dataset in [("train", train_set), ("test", test_set)]:
        _convert_to_image_folders(dataset, Path(output_dir), split)

    print(f"\n✓ DTD saved to {output_dir}")


def download_eurosat(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert EuroSAT."""
    import torchvision
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading EuroSAT...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "eurosat")

    # EuroSAT has train/test/val splits, use all for now
    train_set = torchvision.datasets.EuroSAT(root=raw_dir, split='train', download=True)
    test_set = torchvision.datasets.EuroSAT(root=raw_dir, split='test', download=True)

    for split, dataset in [("train", train_set), ("test", test_set)]:
        _convert_to_image_folders(dataset, Path(output_dir), split)

    print(f"\n✓ EuroSAT saved to {output_dir}")


def download_gtsrb(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert GTSRB (German Traffic Sign Recognition Benchmark)."""
    import torchvision
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading GTSRB...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "gtsrb")

    train_set = torchvision.datasets.GTSRB(root=raw_dir, split='train', download=True)
    test_set = torchvision.datasets.GTSRB(root=raw_dir, split='test', download=True)

    for split, dataset in [("train", train_set), ("test", test_set)]:
        _convert_to_image_folders(dataset, Path(output_dir), split)

    print(f"\n✓ GTSRB saved to {output_dir}")


def download_country211(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert Country211."""
    import torchvision
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading Country211...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "country211")

    train_set = torchvision.datasets.Country211(root=raw_dir, split='train', download=True)
    test_set = torchvision.datasets.Country211(root=raw_dir, split='test', download=True)

    for split, dataset in [("train", train_set), ("test", test_set)]:
        _convert_to_image_folders(dataset, Path(output_dir), split)

    print(f"\n✓ Country211 saved to {output_dir}")


def download_resisc45(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Download and convert Resisc45 (Remote Sensing Image Scene Classification)."""
    import torchvision
    from tqdm import tqdm

    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Downloading Resisc45...")
    print("="*60)

    raw_dir = str(paths["data_raw"])
    output_dir = str(paths["data"] / "resisc45")

    train_set = torchvision.datasets.RESISC45(root=raw_dir, split='train', download=True)
    test_set = torchvision.datasets.RESISC45(root=raw_dir, split='test', download=True)

    for split, dataset in [("train", train_set), ("test", test_set)]:
        _convert_to_image_folders(dataset, Path(output_dir), split)

    print(f"\n✓ Resisc45 saved to {output_dir}")


def print_manual_download_instructions(storage_dir: str = DEFAULT_STORAGE_DIR):
    """Print instructions for datasets that require manual download."""
    paths = get_storage_paths(storage_dir)

    print("\n" + "="*60)
    print("Manual Dataset Download Instructions")
    print("="*60)

    print("\nThe following datasets require manual download:")
    print("\n1. SUN397")
    print("   Download: http://vision.princeton.edu/projects/2010/SUN/download.html")
    print(f"   Extract to: {paths['data'] / 'sun397'}")
    print("   Structure: sun397/train/<class>/images/ and sun397/test/<class>/images/")

    print("\n2. Stanford Cars")
    print("   Download: https://ai.stanford.edu/~jkrause/cars/car_dataset.html")
    print(f"   Extract to: {paths['data'] / 'stanford_cars'}")
    print("   Structure: stanford_cars/train/<class>/ and stanford_cars/test/<class>/")

    print("\n3. FGVC Aircraft")
    print("   Download: https://www.robots.ox.ac.uk/~vgg/data/aircraft/")
    print(f"   Extract to: {paths['data'] / 'aircraft'}")
    print("   Structure: aircraft/train/<class>/ and aircraft/test/<class>/")

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download models and datasets")
    parser.add_argument("--storage_dir", type=str, default=DEFAULT_STORAGE_DIR,
                        help="Root directory for large files: models/, data/, data_raw/")
    parser.add_argument("--models", action="store_true", help="Download all models")
    # Existing datasets
    parser.add_argument("--cifar10", action="store_true", help="Download CIFAR-10")
    parser.add_argument("--cifar100", action="store_true", help="Download CIFAR-100")
    # CLIP datasets (torchvision supported)
    parser.add_argument("--mnist", action="store_true", help="Download MNIST")
    parser.add_argument("--svhn", action="store_true", help="Download SVHN")
    parser.add_argument("--dtd", action="store_true", help="Download DTD")
    parser.add_argument("--eurosat", action="store_true", help="Download EuroSAT")
    parser.add_argument("--gtsrb", action="store_true", help="Download GTSRB")
    parser.add_argument("--country211", action="store_true", help="Download Country211")
    parser.add_argument("--resisc45", action="store_true", help="Download Resisc45")
    # Manual download datasets
    parser.add_argument("--manual_instructions", action="store_true", help="Print manual download instructions")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--all_datasets", action="store_true", help="Download all torchvision-supported datasets")

    args = parser.parse_args()

    if args.all:
        download_all_models(args.storage_dir)
        download_cifar10(args.storage_dir)
        download_cifar100(args.storage_dir)
        download_mnist(args.storage_dir)
        download_svhn(args.storage_dir)
        download_dtd(args.storage_dir)
        download_eurosat(args.storage_dir)
        download_gtsrb(args.storage_dir)
        download_country211(args.storage_dir)
        download_resisc45(args.storage_dir)
        print_manual_download_instructions(args.storage_dir)
    else:
        if args.models:
            download_all_models(args.storage_dir)
        if args.cifar10:
            download_cifar10(args.storage_dir)
        if args.cifar100:
            download_cifar100(args.storage_dir)
        if args.mnist:
            download_mnist(args.storage_dir)
        if args.svhn:
            download_svhn(args.storage_dir)
        if args.dtd:
            download_dtd(args.storage_dir)
        if args.eurosat:
            download_eurosat(args.storage_dir)
        if args.gtsrb:
            download_gtsrb(args.storage_dir)
        if args.country211:
            download_country211(args.storage_dir)
        if args.resisc45:
            download_resisc45(args.storage_dir)
        if args.all_datasets:
            download_cifar10(args.storage_dir)
            download_cifar100(args.storage_dir)
            download_mnist(args.storage_dir)
            download_svhn(args.storage_dir)
            download_dtd(args.storage_dir)
            download_eurosat(args.storage_dir)
            download_gtsrb(args.storage_dir)
            download_country211(args.storage_dir)
            download_resisc45(args.storage_dir)
        if args.manual_instructions:
            print_manual_download_instructions(args.storage_dir)

    if not any([
        args.models, args.cifar10, args.cifar100, args.all, args.all_datasets,
        args.mnist, args.svhn, args.dtd, args.eurosat, args.gtsrb,
        args.country211, args.resisc45, args.manual_instructions
    ]):
        print("Usage:")
        print("  python download_models.py --all                    # Download everything")
        print("  python download_models.py --models                 # Download all models")
        print("  python download_models.py --all_datasets           # Download all torchvision-supported datasets")
        print("  python download_models.py --cifar100               # Download CIFAR-100")
        print("  python download_models.py --cifar10                # Download CIFAR-10")
        print("  python download_models.py --mnist                  # Download MNIST")
        print("  python download_models.py --svhn                   # Download SVHN")
        print("  python download_models.py --dtd                    # Download DTD")
        print("  python download_models.py --eurosat                # Download EuroSAT")
        print("  python download_models.py --gtsrb                  # Download GTSRB")
        print("  python download_models.py --country211             # Download Country211")
        print("  python download_models.py --resisc45               # Download Resisc45")
        print("  python download_models.py --manual_instructions    # Print manual download instructions")
        print("  python download_models.py --all --storage_dir /path/to/bigfiles")

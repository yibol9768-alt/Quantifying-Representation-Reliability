"""
Extract multi-layer features for COMM fusion

This script extracts features from multiple layers of CLIP and DINO models,
which are required for the COMM fusion strategy.

Usage:
    python scripts/2_extract_comm.py --dataset stanford_cars --split train
    python scripts/2_extract_comm.py --dataset cifar10 --split test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from src.models import CLIPMultiLayerModel, DINOMultiLayerModel
from src.data import get_dataset
from src.utils import get_device, ensure_dir


def extract_comm_features(
    dataset_name: str,
    split: str,
    output_path: str,
    device: str = "cuda",
):
    """
    Extract multi-layer features for COMM fusion.

    Args:
        dataset_name: Dataset name
        split: Data split ("train" or "test")
        output_path: Output path for features
        device: Device to use
    """
    # Load dataset
    dataset = get_dataset(dataset_name, "data")

    if split == "train":
        image_paths, labels = dataset.load_train_data()
    else:
        image_paths, labels = dataset.load_test_data()

    print(f"Dataset: {dataset_name}, Split: {split}")
    print(f"Samples: {len(image_paths)}, Classes: {dataset.num_classes}")

    # Load multi-layer models
    print("\nLoading CLIP multi-layer model...")
    clip_model = CLIPMultiLayerModel(device=device)

    print("Loading DINO multi-layer model...")
    dino_model = DINOMultiLayerModel(device=device)

    # Initialize storage
    # CLIP: 12 layers, DINO: 6 layers (7-12)
    clip_layer_features = {f"clip_layer_{i}": [] for i in range(12)}
    dino_layer_features = {f"dino_layer_{i}": [] for i in range(6, 12)}
    labels_list = []

    # Extract features
    print(f"\nExtracting multi-layer features...")
    with torch.no_grad():
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                img = dataset.get_image(img_path)

                # Extract CLIP multi-layer features
                clip_features = clip_model.extract_multilayer_features(img)
                for layer_idx in range(12):
                    feat = clip_features[f"layer_{layer_idx}"]
                    clip_layer_features[f"clip_layer_{layer_idx}"].append(feat.cpu())

                # Extract DINO multi-layer features
                dino_features = dino_model.extract_multilayer_features(img)
                for layer_idx in range(6, 12):
                    feat = dino_features[layer_idx]
                    dino_layer_features[f"dino_layer_{layer_idx}"].append(feat.cpu())

                labels_list.append(label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Concatenate features
    result = {}

    # CLIP layer features
    for layer_idx in range(12):
        key = f"clip_layer_{layer_idx}"
        result[key] = torch.cat(clip_layer_features[key], dim=0)
        print(f"  {key}: {result[key].shape}")

    # DINO layer features
    for layer_idx in range(6, 12):
        key = f"dino_layer_{layer_idx}"
        result[key] = torch.cat(dino_layer_features[key], dim=0)
        print(f"  {key}: {result[key].shape}")

    result["labels"] = torch.tensor(labels_list, dtype=torch.long)
    result["dataset"] = dataset_name
    result["num_classes"] = dataset.num_classes

    # Save features
    ensure_dir(os.path.dirname(output_path))
    torch.save(result, output_path)
    print(f"\nFeatures saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract multi-layer features for COMM fusion")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "stanford_cars",
            "cifar10",
            "cifar100",
            "flowers102",
            "pets",
            "food101",
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Data split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()

    # Default output path
    if args.output is None:
        args.output = f"features/{args.dataset}_comm_{args.split}.pt"

    print(f"Extracting COMM features for {args.dataset} ({args.split})")
    print(f"Device: {device}")

    extract_comm_features(
        dataset_name=args.dataset,
        split=args.split,
        output_path=args.output,
        device=device,
    )

    print(f"\nDone! Features saved to: {args.output}")


if __name__ == "__main__":
    main()

"""Extract and save frozen features for all models on all datasets.

This script produces the feature files needed by run_selection_comparison.py.

Output structure:
    {output_dir}/{dataset}/{split}/{model_name}.pt   # [N, d] float32 tensor
    {output_dir}/{dataset}/{split}/labels.pt         # [N] int64 tensor
    {output_dir}/{dataset}/protocol.json

Usage:
    python experiments/extract_features.py \
        --storage_dir /path/to/storage \
        --output_dir /path/to/storage/data/features \
        --datasets stl10,pets,eurosat,dtd,gtsrb,svhn,country211 \
        --models clip,dino,mae,siglip,convnext,data2vec,vit,swin,beit,dinov2_large,dinov2_small,mae_large,deit_small,resnet50,clip_large

    # Quick test with 1 model 1 dataset:
    python experiments/extract_features.py \
        --storage_dir /path/to/storage \
        --output_dir /path/to/storage/data/features \
        --datasets stl10 \
        --models clip
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.extractor import FeatureExtractor
from src.data.dataset import get_feature_split_dataloaders


ALL_RECOMMENDED_MODELS = [
    "clip", "dino", "mae", "siglip", "convnext", "data2vec",
    "vit", "swin", "beit",
    "dinov2_large", "dinov2_small", "mae_large",
    "deit_small", "resnet50", "clip_large",
]

ALL_DATASETS = ["stl10", "pets", "eurosat", "dtd", "gtsrb", "svhn", "country211"]


def _flatten_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    """Normalize extracted features to [N, D] before persisting them."""
    if features.ndim <= 2:
        return features
    if all(dim == 1 for dim in features.shape[2:]):
        return features.flatten(1)
    return features.reshape(features.size(0), -1)


def extract_features(
    model_name: str,
    dataset: str,
    split: str,
    data_dir: str,
    model_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    val_ratio: float,
    split_seed: int,
) -> tuple:
    """Extract features for one model on one dataset.

    Returns:
        (features [N, d], labels [N])
    """
    extractor = FeatureExtractor(
        model_type=model_name,
        normalize_input=False,
        model_dir=model_dir,
    ).to(device)

    split_loaders = get_feature_split_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model_type=model_name,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    if split not in split_loaders:
        raise ValueError(f"Split '{split}' is not available for dataset '{dataset}'.")
    loader = split_loaders[split]

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {model_name}/{dataset}/{split}", leave=False):
            images = images.to(device)
            feats = extractor(images)  # [B, d]
            all_features.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0).float()
    features = _flatten_feature_tensor(features)
    labels = torch.cat(all_labels, dim=0).long()  # [N]

    # Cleanup GPU memory
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Extract features for selection experiments")
    parser.add_argument("--storage_dir", type=str, default=None,
                        help="Shared storage root (sets model_dir and data_dir)")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Model weights directory")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for feature .pt files")
    parser.add_argument("--datasets", type=str,
                        default=",".join(ALL_DATASETS),
                        help="Comma-separated dataset list")
    parser.add_argument("--models", type=str,
                        default=",".join(ALL_RECOMMENDED_MODELS),
                        help="Comma-separated model list")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve storage paths
    if args.storage_dir:
        from pathlib import Path
        root = Path(args.storage_dir)
        args.model_dir = str(root / "models")
        args.data_dir = str(root / "data")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    print("=" * 60)
    print("Feature Extraction for Selection Experiments")
    print("=" * 60)
    print(f"Models:   {len(models)} -> {models}")
    print(f"Datasets: {len(datasets)} -> {datasets}")
    print(f"Output:   {args.output_dir}")
    print(f"Device:   {device}")
    print(f"Protocol: Train/Val/Test (val_ratio={args.validation_ratio}, split_seed={args.split_seed})")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset}")
        print(f"{'='*40}")

        out_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)

        protocol_path = os.path.join(out_dir, "protocol.json")
        with open(protocol_path, "w", encoding="utf-8") as f:
            json.dump({
                "selection_split": "train",
                "selection_eval_split": "val",
                "final_eval_split": "test",
                "validation_ratio": args.validation_ratio,
                "split_seed": args.split_seed,
            }, f, indent=2)

        for split in ("train", "val", "test"):
            split_dir = os.path.join(out_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            labels_path = os.path.join(split_dir, "labels.pt")

            for model_name in models:
                out_path = os.path.join(split_dir, f"{model_name}.pt")

                if os.path.exists(out_path) and os.path.exists(labels_path):
                    print(f"  [SKIP] {split}/{model_name} (already exists)")
                    continue

                print(f"  [EXTRACT] {split}/{model_name}...")
                try:
                    features, labels = extract_features(
                        model_name=model_name,
                        dataset=dataset,
                        split=split,
                        data_dir=args.data_dir,
                        model_dir=args.model_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        val_ratio=args.validation_ratio,
                        split_seed=args.split_seed,
                    )

                    torch.save(features, out_path)
                    print(f"    -> {out_path} [{features.shape[0]} x {features.shape[1]}]")
                    torch.save(labels, labels_path)
                    print(f"    -> {labels_path} [{labels.shape[0]}]")
                except Exception as e:
                    print(f"    [FAIL] {split}/{model_name}: {e}")
                    continue

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Output directory: {args.output_dir}")
    print("")
    print("Next step: run selection comparison")
    print(f"  python experiments/run_selection_comparison.py \\")
    print(f"      --data_root {args.output_dir} \\")
    print(f"      --datasets {','.join(datasets)} \\")
    print(f"      --max_models 10")
    print("=" * 60)


if __name__ == "__main__":
    main()

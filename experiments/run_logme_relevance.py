#!/usr/bin/env python3
"""Compute LogME-based relevance scores for each (dataset, model) pair.

Extracts pooled features via FeatureExtractor.forward() and computes LogME
scores as a training-free proxy for single-model task relevance R(m, T).

Outputs:
    logme_scores.json: {dataset: {model: score}}

Usage:
    python experiments/run_logme_relevance.py \
        --datasets stl10,gtsrb,svhn,pets,eurosat,dtd,country211 \
        --models clip,dino,mae,siglip,convnext,data2vec \
        --model_dir ./models --data_dir ./data \
        --output_dir ./results/logme --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.extractor import FeatureExtractor
from src.data.dataset import get_feature_split_dataloaders
from src.analysis.logme import logme_score


def extract_pooled_features(
    model_type: str,
    dataset: str,
    data_dir: str,
    model_dir: str,
    device: str,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    sample_seed: int,
    selection_split: str,
    validation_ratio: float,
    split_seed: int,
) -> tuple:
    """Extract pooled features and labels from a model.

    Returns:
        features: np.ndarray [N, d]
        labels: np.ndarray [N]
    """
    extractor = FeatureExtractor(model_type=model_type, model_dir=model_dir)
    extractor.eval().to(device)
    split_loaders = get_feature_split_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model_type=model_type,
        val_ratio=validation_ratio,
        split_seed=split_seed,
    )
    dataloader = split_loaders[selection_split]

    if max_samples > 0 and len(dataloader.dataset) > max_samples:
        generator = torch.Generator().manual_seed(sample_seed)
        subset_indices = torch.randperm(
            len(dataloader.dataset), generator=generator
        )[:max_samples]
        dataloader = DataLoader(
            Subset(dataloader.dataset, subset_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    all_features = []
    all_labels = []
    for images, targets in dataloader:
        images = images.to(device)
        feat = extractor(images)  # [B, d]
        all_features.append(feat.cpu().numpy())
        all_labels.append(targets.numpy())

    extractor.to("cpu")
    del extractor
    if "cuda" in device:
        torch.cuda.empty_cache()

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="LogME relevance scoring")
    parser.add_argument("--datasets", type=str,
                        default="stl10,gtsrb,svhn,pets,eurosat,dtd,country211",
                        help="Comma-separated dataset names")
    parser.add_argument("--models", type=str,
                        default="clip,dino,mae,siglip,convnext,data2vec",
                        help="Comma-separated model names")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results/logme")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples per dataset (0=all)")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--selection_split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== LogME Relevance Scoring ===")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Device: {args.device}")
    print(f"Selection split: {args.selection_split}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"Output: {args.output_dir}")
    print()

    all_scores = {}

    for dataset in datasets:
        print(f"--- Dataset: {dataset} ---")

        ds_scores = {}
        for model_name in models:
            t0 = time.time()
            print(f"  {model_name}: extracting...", end=" ", flush=True)

            features, labels = extract_pooled_features(
                model_type=model_name,
                dataset=dataset,
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                sample_seed=args.sample_seed,
                selection_split=args.selection_split,
                validation_ratio=args.validation_ratio,
                split_seed=args.split_seed,
            )
            print(f"features={features.shape},", end=" ", flush=True)

            score = logme_score(features, labels)
            elapsed = time.time() - t0
            ds_scores[model_name] = score
            print(f"LogME={score:.4f} ({elapsed:.1f}s)")

        all_scores[dataset] = ds_scores
        print()

    # Save results
    output_path = os.path.join(args.output_dir, "logme_scores.json")
    with open(output_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"Saved LogME scores to {output_path}")

    protocol_path = os.path.join(args.output_dir, "protocol.json")
    with open(protocol_path, "w") as f:
        json.dump({
            "selection_split": args.selection_split,
            "validation_ratio": args.validation_ratio,
            "split_seed": args.split_seed,
            "sample_seed": args.sample_seed,
        }, f, indent=2)
    print(f"Saved protocol to {protocol_path}")

    # Print summary table
    print("\n=== Summary ===")
    header = f"{'Dataset':<12}" + "".join(f"{m:>12}" for m in models)
    print(header)
    print("-" * len(header))
    for ds in datasets:
        if ds in all_scores:
            row = f"{ds:<12}" + "".join(
                f"{all_scores[ds].get(m, 0):>12.4f}" for m in models
            )
            print(row)


if __name__ == "__main__":
    main()

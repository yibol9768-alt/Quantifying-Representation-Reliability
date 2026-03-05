"""
Extract features from a single model

Usage:
    python scripts/1_extract_single.py --model clip --dataset stanford_cars
    python scripts/1_extract_single.py --model dino --dataset cifar10
    python scripts/1_extract_single.py --model mae --dataset flowers102
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureExtractor
from src.utils import get_device, ensure_dir
from configs.config import Config


def main():
    parser = argparse.ArgumentParser(description="Extract features from a single model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["clip", "dino", "mae"],
        help="Model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stanford_cars",
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
        help="Output path (default: features/{dataset}_{model}_{split}.pt)",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()

    print(f"Extracting {args.model} features for {args.dataset} ({args.split})")
    print(f"Device: {device}")

    # Default output path
    if args.output is None:
        args.output = f"features/{args.dataset}_{args.model}_{args.split}.pt"

    ensure_dir(os.path.dirname(args.output))

    # Extract
    extractor = FeatureExtractor(device=device, dataset=args.dataset)
    features = extractor.extract(
        model_types=args.model,
        split=args.split,
        output_path=args.output,
    )

    print(f"\nDone! Features saved to: {args.output}")


if __name__ == "__main__":
    main()

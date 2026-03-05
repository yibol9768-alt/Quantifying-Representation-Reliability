"""
Extract features from multiple models

Usage:
    python scripts/2_extract_multi.py --models clip dino --dataset stanford_cars
    python scripts/2_extract_multi.py --models clip dino mae --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureExtractor
from src.utils import get_device, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Extract features from multiple models")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=["clip", "dino", "mae"],
        help="Model types (space-separated)",
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
        help="Output path",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()

    model_str = "_".join(args.models)
    print(f"Extracting {model_str} features for {args.dataset} ({args.split})")
    print(f"Device: {device}")

    # Default output path
    if args.output is None:
        args.output = f"features/{args.dataset}_{model_str}_{args.split}.pt"

    ensure_dir(os.path.dirname(args.output))

    # Extract
    extractor = FeatureExtractor(device=device, dataset=args.dataset)
    features = extractor.extract(
        model_types=args.models,
        split=args.split,
        output_path=args.output,
    )

    print(f"\nDone! Features saved to: {args.output}")


if __name__ == "__main__":
    main()

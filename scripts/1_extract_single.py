"""
Extract features from a single model

Usage:
    python scripts/1_extract_single.py --model clip --split train
    python scripts/1_extract_single.py --model dino --split train
    python scripts/1_extract_single.py --model mae --split train
    python scripts/1_extract_single.py --model clip --split test
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
        help="Output path (default: features/{model}_{split}.pt)",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    config = Config()

    print(f"Extracting {args.model} features for {args.split} split...")
    print(f"Device: {device}")

    # Default output path
    if args.output is None:
        args.output = os.path.join(config.feature_dir, f"{args.model}_{args.split}.pt")

    ensure_dir(os.path.dirname(args.output))

    # Extract
    extractor = FeatureExtractor(device=device, data_root=config.data_root)
    features = extractor.extract(
        model_types=args.model,
        split=args.split,
        output_path=args.output,
    )

    print(f"\nDone! Features saved to: {args.output}")


if __name__ == "__main__":
    main()

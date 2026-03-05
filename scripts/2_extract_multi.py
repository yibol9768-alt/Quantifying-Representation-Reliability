"""
Extract features from multiple models

Usage:
    python scripts/2_extract_multi.py --models clip dino --split train
    python scripts/2_extract_multi.py --models clip dino mae --split train
    python scripts/2_extract_multi.py --models clip dino mae --split test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureExtractor
from src.utils import get_device, ensure_dir
from configs.config import Config


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
    config = Config()

    model_str = "_".join(args.models)
    print(f"Extracting {model_str} features for {args.split} split...")
    print(f"Device: {device}")

    # Default output path
    if args.output is None:
        args.output = os.path.join(config.feature_dir, f"{model_str}_{args.split}.pt")

    ensure_dir(os.path.dirname(args.output))

    # Extract
    extractor = FeatureExtractor(device=device, data_root=config.data_root)
    features = extractor.extract(
        model_types=args.models,
        split=args.split,
        output_path=args.output,
    )

    print(f"\nDone! Features saved to: {args.output}")


if __name__ == "__main__":
    main()

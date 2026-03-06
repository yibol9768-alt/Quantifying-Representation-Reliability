"""
Feature extraction script

Usage:
    # Single model
    python scripts/extract.py --model clip --dataset cifar10
    
    # Multiple models (for fusion)
    python scripts/extract.py --models clip dino --dataset cifar10
    
    # All models
    python scripts/extract.py --all --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureExtractor
from src.utils import get_device, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Extract features from pre-trained models")
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"],
                             help="Single model type")
    model_group.add_argument("--models", type=str, nargs="+", 
                             choices=["clip", "dino", "mae"],
                             help="Multiple model types for fusion")
    model_group.add_argument("--all", action="store_true",
                             help="Extract all models")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "flowers102", "pets", "stanford_cars", "food101"],
                        help="Dataset name")
    
    # Split
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Data split")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="features",
                        help="Output directory")

    args = parser.parse_args()

    # Determine models to extract
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    elif args.all:
        models = ["clip", "dino", "mae"]
    
    # Setup
    device = get_device()
    ensure_dir(args.output_dir)
    
    print(f"Extracting features for {args.dataset} ({args.split})")
    print(f"Models: {' + '.join(models).upper()}")
    print(f"Device: {device}")
    print()

    # Extract
    extractor = FeatureExtractor(device=device, dataset=args.dataset)
    
    if len(models) == 1:
        # Single model
        output_path = os.path.join(args.output_dir, f"{args.dataset}_{models[0]}_{args.split}.pt")
        extractor.extract(model_types=models[0], split=args.split, output_path=output_path)
    else:
        # Multiple models (for fusion)
        model_str = "_".join(models)
        output_path = os.path.join(args.output_dir, f"{args.dataset}_{model_str}_{args.split}.pt")
        extractor.extract_multi(model_types=models, split=args.split, output_path=output_path)
    
    print(f"\nDone! Features saved to: {output_path}")


if __name__ == "__main__":
    main()

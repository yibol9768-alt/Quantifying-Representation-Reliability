"""
Train a multi-view fusion model

Usage:
    python scripts/4_train_fusion.py --models clip dino --dataset stanford_cars
    python scripts/4_train_fusion.py --models clip dino mae --dataset cifar10
    python scripts/4_train_fusion.py --models clip dino --dataset stanford_cars --method weighted_sum

Fusion methods:
    - concat: Simple concatenation (default, same as before)
    - weighted_sum: Learnable weighted sum fusion
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import MultiViewClassifier, Trainer, WeightedSumFusionClassifier
from src.features import FeatureExtractor
from src.data import DATASET_INFO
from src.utils import get_device, set_seed, print_model_info
from configs.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Train multi-view fusion model")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=["clip", "dino", "mae"],
        help="Model types to fuse (space-separated)",
    )
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
        "--method",
        type=str,
        default="concat",
        choices=["concat", "weighted_sum"],
        help="Fusion method: concat (simple concatenation) or weighted_sum (learnable weights)",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="features",
        help="Feature directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output checkpoint path",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    set_seed(42)

    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    model_str = "_".join(args.models)
    print(f"Training {model_str.upper()} fusion model on {args.dataset}")
    print(f"Fusion method: {args.method}")
    print(f"Device: {device}, Classes: {num_classes}")

    # Load features
    train_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_train.pt")
    test_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_test.pt")

    print(f"\nLoading features from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    if not os.path.exists(train_path):
        print(f"\nError: {train_path} not found!")
        print(f"Please run: python scripts/2_extract_multi.py --models {' '.join(args.models)} --dataset {args.dataset} --split train")
        sys.exit(1)

    if not os.path.exists(test_path):
        print(f"\nError: {test_path} not found!")
        print(f"Please run: python scripts/2_extract_multi.py --models {' '.join(args.models)} --dataset {args.dataset} --split test")
        sys.exit(1)

    train_features = FeatureExtractor.load_features(train_path)
    test_features = FeatureExtractor.load_features(test_path)

    # Build model based on fusion method
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]

    if args.method == "concat":
        # Original concatenation method
        model = MultiViewClassifier(
            feature_dims=feature_dims,
            num_classes=num_classes,
        )
        print(f"Input feature dim: {sum(feature_dims)} ({args.models})")

    elif args.method == "weighted_sum":
        # Learnable weighted sum fusion
        model = WeightedSumFusionClassifier(
            feature_dims=feature_dims,
            num_classes=num_classes,
            fusion_dim=512,  # Project all to same dimension
        )
        print(f"Using weighted sum fusion with projection dim 512")

    print_model_info(model)

    # Train
    trainer = Trainer(model, device=device, lr=args.lr, weight_decay=1e-4)

    output_path = args.output
    if output_path is None:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        output_path = f"outputs/checkpoints/{args.dataset}_{model_str}_{args.method}_fusion.pth"

    history = trainer.fit(
        train_features=train_features,
        test_features=test_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save final model
    trainer.save(output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()

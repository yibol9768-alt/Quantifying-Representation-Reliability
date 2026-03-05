"""
Train a multi-view fusion model

Usage:
    python scripts/4_train_fusion.py --models clip dino
    python scripts/4_train_fusion.py --models clip dino mae
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import MultiViewClassifier, Trainer
from src.features import FeatureExtractor
from src.utils import get_device, set_seed, print_model_info
from configs.config import Config, MODEL_CONFIGS


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
    config = Config()

    model_str = "_".join(args.models)
    print(f"Training {model_str.upper()} fusion model")
    print(f"Device: {device}")

    # Load features
    model_str = "_".join(args.models)
    train_path = os.path.join(args.feature_dir, f"{model_str}_train.pt")
    test_path = os.path.join(args.feature_dir, f"{model_str}_test.pt")

    print(f"\nLoading features from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    train_features = FeatureExtractor.load_features(train_path)
    test_features = FeatureExtractor.load_features(test_path)

    # Build model
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
    model = MultiViewClassifier(
        feature_dims=feature_dims,
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    print_model_info(model)
    print(f"Input feature dim: {sum(feature_dims)} ({args.models})")

    # Train
    trainer = Trainer(model, device=device, lr=args.lr, weight_decay=1e-4)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(config.checkpoint_dir, f"{model_str}_fusion.pth")

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

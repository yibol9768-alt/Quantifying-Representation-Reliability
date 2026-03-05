"""
Train a single model classifier

Usage:
    python scripts/3_train_single.py --model clip
    python scripts/3_train_single.py --model dino
    python scripts/3_train_single.py --model mae
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import SingleViewClassifier, Trainer
from src.features import FeatureExtractor
from src.utils import get_device, set_seed, print_model_info
from configs.config import Config, MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Train single model classifier")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["clip", "dino", "mae"],
        help="Model type",
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

    print(f"Training {args.model.upper()} single model classifier")
    print(f"Device: {device}")

    # Load features
    train_path = os.path.join(args.feature_dir, f"{args.model}_train.pt")
    test_path = os.path.join(args.feature_dir, f"{args.model}_test.pt")

    print(f"\nLoading features from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    train_features = FeatureExtractor.load_features(train_path)
    test_features = FeatureExtractor.load_features(test_path)

    # Build model
    feature_dim = MODEL_CONFIGS[args.model]["feature_dim"]
    model = SingleViewClassifier(
        feature_dim=feature_dim,
        num_classes=config.num_classes,
    )

    print_model_info(model)

    # Train
    trainer = Trainer(model, device=device, lr=args.lr, weight_decay=1e-4)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(config.checkpoint_dir, f"{args.model}_single.pth")

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

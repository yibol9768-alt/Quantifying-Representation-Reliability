"""
Training script

Usage:
    # Single model
    python scripts/train.py --model clip --dataset cifar10
    
    # Multi-model fusion
    python scripts/train.py --models clip dino --dataset cifar10
    
    # Three models
    python scripts/train.py --models clip dino mae --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import SingleViewClassifier, MultiViewClassifier, Trainer
from src.features import FeatureExtractor
from src.data import DATASET_INFO
from src.utils import get_device, set_seed, print_model_info
from configs.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"],
                             help="Single model type")
    model_group.add_argument("--models", type=str, nargs="+",
                             choices=["clip", "dino", "mae"],
                             help="Multiple model types for fusion")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "flowers102", "pets", "stanford_cars", "food101"],
                        help="Dataset name")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--feature-dir", type=str, default="features", help="Feature directory")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")

    args = parser.parse_args()

    # Setup
    device = get_device()
    set_seed(42)
    num_classes = DATASET_INFO[args.dataset]["num_classes"]

    # Determine if single or fusion
    if args.model:
        models = [args.model]
        is_single = True
    else:
        models = args.models
        is_single = False

    model_str = "_".join(models)
    print(f"Training {model_str.upper()} on {args.dataset}")
    print(f"Mode: {'Single' if is_single else 'Fusion'}")
    print(f"Device: {device}, Classes: {num_classes}")

    # Load features
    train_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_train.pt")
    test_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_test.pt")

    print(f"\nLoading features:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    for path in [train_path, test_path]:
        if not os.path.exists(path):
            print(f"\nError: {path} not found!")
            cmd = f"python scripts/extract.py --models {' '.join(models)} --dataset {args.dataset} --split {{}}"
            print(f"Please run:")
            print(f"  {cmd.format('train')}")
            print(f"  {cmd.format('test')}")
            sys.exit(1)

    train_features = FeatureExtractor.load_features(train_path)
    test_features = FeatureExtractor.load_features(test_path)

    # Build model
    if is_single:
        feature_dim = MODEL_CONFIGS[models[0]]["feature_dim"]
        model = SingleViewClassifier(feature_dim=feature_dim, num_classes=num_classes)
    else:
        feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in models]
        model = MultiViewClassifier(feature_dims=feature_dims, num_classes=num_classes)
        print(f"Input feature dims: {feature_dims} (total: {sum(feature_dims)})")

    print_model_info(model)

    # Train
    trainer = Trainer(model, device=device, lr=args.lr, weight_decay=1e-4)

    output_path = args.output
    if output_path is None:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        suffix = "single" if is_single else "fusion"
        output_path = f"outputs/checkpoints/{args.dataset}_{model_str}_{suffix}.pth"

    history = trainer.fit(
        train_features=train_features,
        test_features=test_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    trainer.save(output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"Best accuracy: {max(history['test_acc']):.2f}%")


if __name__ == "__main__":
    main()

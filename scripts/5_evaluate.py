"""
Evaluate a trained model

Usage:
    python scripts/5_evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_single.pth --model clip --dataset cifar10
    python scripts/5_evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_mae_fusion.pth --models clip dino mae --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import MultiViewClassifier, SingleViewClassifier, Trainer
from src.features import FeatureExtractor
from src.data import DATASET_INFO
from src.utils import get_device
from configs.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["clip", "dino", "mae"],
        help="Model types (for fusion models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["clip", "dino", "mae"],
        help="Single model type",
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
        "--feature-dir",
        type=str,
        default="features",
        help="Feature directory",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Single model classifier",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    num_classes = DATASET_INFO[args.dataset]["num_classes"]

    print(f"Evaluating: {args.checkpoint}")
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")
    print(f"Device: {device}")

    # Determine model type from checkpoint name if not specified
    if args.models is None and args.model is None:
        checkpoint_name = os.path.basename(args.checkpoint)

        # Try to extract from checkpoint name
        found_models = [m for m in ["clip", "dino", "mae"] if m in checkpoint_name]

        if "fusion" in checkpoint_name or len(found_models) > 1:
            args.models = found_models
        elif len(found_models) == 1:
            args.model = found_models[0]
        else:
            raise ValueError(
                "Cannot determine model types from checkpoint name. "
                "Please specify --model or --models"
            )

    if args.model:
        args.models = [args.model]

    if args.models is None:
        raise ValueError("Please specify --model or --models")

    print(f"Model types: {args.models}")

    # Load test features
    if len(args.models) == 1:
        test_path = os.path.join(args.feature_dir, f"{args.dataset}_{args.models[0]}_test.pt")
    else:
        model_str = "_".join(args.models)
        test_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_test.pt")

    if not os.path.exists(test_path):
        print(f"Error: Test features not found: {test_path}")
        print("Please extract features first.")
        sys.exit(1)

    print(f"Loading test features from: {test_path}")
    test_features = FeatureExtractor.load_features(test_path)

    # Build model
    is_single = len(args.models) == 1 or args.single

    if is_single:
        feature_dim = MODEL_CONFIGS[args.models[0]]["feature_dim"]
        model = SingleViewClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
        )
    else:
        feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
        model = MultiViewClassifier(
            feature_dims=feature_dims,
            num_classes=num_classes,
        )

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint loaded from: {args.checkpoint}")

    # Evaluate
    trainer = Trainer(model, device=device)

    from torch.utils.data import DataLoader
    from src.training import FeatureDataset

    test_dataset = FeatureDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    accuracy = trainer.evaluate(test_loader)

    print("\n" + "=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {' + '.join(args.models)}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

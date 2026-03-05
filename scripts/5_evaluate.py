"""
Evaluate a trained model

Usage:
    python scripts/5_evaluate.py --checkpoint outputs/checkpoints/clip_single.pth --model clip
    python scripts/5_evaluate.py --checkpoint outputs/checkpoints/clip_dino_mae_fusion.pth --models clip dino mae
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import MultiViewClassifier, SingleViewClassifier, Trainer
from src.features import FeatureExtractor
from src.utils import get_device
from configs.config import Config, MODEL_CONFIGS


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
    config = Config()

    print(f"Evaluating: {args.checkpoint}")
    print(f"Device: {device}")

    # Determine model type from checkpoint name if not specified
    if args.models is None:
        checkpoint_name = os.path.basename(args.checkpoint)
        if "single" in checkpoint_name or args.single:
            # Extract model name from checkpoint
            for m in ["clip", "dino", "mae"]:
                if m in checkpoint_name:
                    args.models = [m]
                    break
        else:
            # Fusion model - extract all model names
            args.models = [m for m in ["clip", "dino", "mae"] if m in checkpoint_name]

    if args.models is None:
        raise ValueError("Cannot determine model types from checkpoint name. Please specify --models")

    print(f"Model types: {args.models}")

    # Load test features
    model_str = "_".join(args.models)
    test_path = os.path.join(args.feature_dir, f"{model_str}_test.pt")

    if not os.path.exists(test_path):
        print(f"Warning: {test_path} not found, trying single model features...")
        test_path = os.path.join(args.feature_dir, f"{args.models[0]}_test.pt")

    print(f"Loading test features from: {test_path}")
    test_features = FeatureExtractor.load_features(test_path)

    # Build model
    is_single = len(args.models) == 1 or args.single

    if is_single:
        feature_dim = MODEL_CONFIGS[args.models[0]]["feature_dim"]
        model = SingleViewClassifier(
            feature_dim=feature_dim,
            num_classes=config.num_classes,
        )
    else:
        feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
        model = MultiViewClassifier(
            feature_dims=feature_dims,
            num_classes=config.num_classes,
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
    print(f"Model: {model_str}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

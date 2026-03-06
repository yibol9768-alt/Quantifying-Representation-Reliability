"""
Evaluation script

Usage:
    # Evaluate single model
    python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_single.pth --model clip --dataset cifar10
    
    # Evaluate fusion model
    python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_fusion.pth --models clip dino --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.training import SingleViewClassifier, MultiViewClassifier
from src.features import FeatureExtractor
from src.data import DATASET_INFO
from src.utils import get_device
from configs.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"],
                             help="Single model type")
    model_group.add_argument("--models", type=str, nargs="+",
                             choices=["clip", "dino", "mae"],
                             help="Multiple model types (fusion)")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--feature-dir", type=str, default="features",
                        help="Feature directory")

    args = parser.parse_args()

    # Setup
    device = get_device()
    num_classes = DATASET_INFO[args.dataset]["num_classes"]

    # Determine models
    if args.model:
        models = [args.model]
        is_single = True
    else:
        models = args.models
        is_single = False

    model_str = "_".join(models)
    print(f"Evaluating {model_str.upper()} on {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load features
    test_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_test.pt")
    
    if not os.path.exists(test_path):
        print(f"\nError: {test_path} not found!")
        sys.exit(1)

    test_features = FeatureExtractor.load_features(test_path)

    # Build model
    if is_single:
        feature_dim = MODEL_CONFIGS[models[0]]["feature_dim"]
        model = SingleViewClassifier(feature_dim=feature_dim, num_classes=num_classes)
    else:
        feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in models]
        model = MultiViewClassifier(feature_dims=feature_dims, num_classes=num_classes)

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluate
    from torch.utils.data import DataLoader, TensorDataset
    
    X = test_features["features"]
    y = test_features["labels"]
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            
            if is_single:
                outputs = model(features)
            else:
                # Split features for each model
                dims = [MODEL_CONFIGS[m]["feature_dim"] for m in models]
                splits = torch.split(features, dims, dim=1)
                outputs = model(*splits)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    main()

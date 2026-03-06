"""
Evaluation script

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/model.pth --method single --model clip --dataset cifar10
    python scripts/evaluate.py --checkpoint outputs/checkpoints/model.pth --method comm --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def evaluate_single(args):
    """Evaluate single model"""
    from src.training import SingleViewClassifier
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    data = torch.load(f"features/{args.dataset}_{args.model}_test.pt")
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dim = MODEL_CONFIGS[args.model]["feature_dim"]
    
    model = SingleViewClassifier(feature_dim=feature_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device).eval()
    
    X = data["features"].float().to(args.device)
    y = data["labels"].long().to(args.device)
    
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    
    return acc


def evaluate_concat(args):
    """Evaluate concat fusion"""
    from src.training import MultiViewClassifier
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    model_str = "_".join(args.models)
    data = torch.load(f"features/{args.dataset}_{model_str}_test.pt")
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
    
    model = MultiViewClassifier(feature_dims=feature_dims, num_classes=num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device).eval()
    
    X = data["features"].float().to(args.device)
    y = data["labels"].long().to(args.device)
    
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    
    return acc


def evaluate_comm(args):
    """Evaluate COMM fusion"""
    from src.training import COMMFusionClassifier
    from src.data import DATASET_INFO
    
    data = torch.load(f"features/{args.dataset}_comm_test.pt")
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    
    model = COMMFusionClassifier(
        clip_hidden_dim=768, clip_output_dim=512, clip_num_layers=12,
        dino_hidden_dim=768, dino_num_layers=6,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device).eval()
    
    clip_feats = [data[f"clip_layer_{j}"].float().to(args.device) for j in range(12)]
    dino_feats = [data[f"dino_layer_{j}"].float().to(args.device) for j in range(6, 12)]
    y = data["labels"].long().to(args.device)
    
    with torch.no_grad():
        pred = model(clip_feats, dino_feats).argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    
    return acc


def evaluate_comm3(args):
    """Evaluate COMM3 fusion"""
    from src.training import COMM3FusionClassifier
    from src.data import DATASET_INFO
    
    data = torch.load(f"features/{args.dataset}_comm3_test.pt")
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    
    model = COMM3FusionClassifier(
        clip_hidden_dim=768, clip_output_dim=512, clip_num_layers=12,
        dino_hidden_dim=768, dino_num_layers=6,
        mae_hidden_dim=768, mae_num_layers=6,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device).eval()
    
    clip_feats = [data[f"clip_layer_{j}"].float().to(args.device) for j in range(12)]
    dino_feats = [data[f"dino_layer_{j}"].float().to(args.device) for j in range(6, 12)]
    mae_feats = [data[f"mae_layer_{j}"].float().to(args.device) for j in range(6, 12)]
    y = data["labels"].long().to(args.device)
    
    with torch.no_grad():
        pred = model(clip_feats, dino_feats, mae_feats).argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["single", "concat", "comm", "comm3"])
    parser.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    parser.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.method == "single":
        acc = evaluate_single(args)
    elif args.method == "concat":
        acc = evaluate_concat(args)
    elif args.method == "comm":
        acc = evaluate_comm(args)
    elif args.method == "comm3":
        acc = evaluate_comm3(args)
    
    print(f"\nTest Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()

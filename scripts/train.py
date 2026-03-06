"""
Unified Training Script

Usage:
    # Single model
    python scripts/train.py --model clip --dataset cifar10
    
    # Two models
    python scripts/train.py --models clip dino --dataset cifar10 --method concat
    python scripts/train.py --models clip dino --dataset cifar10 --method mmvit
    python scripts/train.py --models clip dino --dataset cifar10 --method comm
    
    # Three models
    python scripts/train.py --models clip dino mae --dataset cifar10 --method concat
    python scripts/train.py --models clip dino mae --dataset cifar10 --method mmvit
    python scripts/train.py --models clip dino mae --dataset cifar10 --method comm3

Methods:
    concat       - Simple concatenation (baseline)
    weighted_sum - Learnable weighted sum
    mmvit        - Cross-attention fusion (MMViT)
    mmvit_lite   - Lightweight cross-attention
    comm         - Multi-layer fusion (2 models)
    comm3        - Multi-layer fusion (3 models)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_feature_path(dataset: str, models: list, split: str, method: str) -> str:
    """Get feature file path based on method"""
    model_str = "_".join(models)
    
    if method in ['comm', 'comm3']:
        # COMM methods use special multi-layer features
        return f"features/{dataset}_{method}_{split}.pt"
    else:
        # Other methods use standard concatenated features
        return f"features/{dataset}_{model_str}_{split}.pt"


def train_standard(args):
    """Train standard fusion (concat, weighted_sum, mmvit, mmvit_lite)"""
    from src.training import create_fusion_model
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    model_str = "_".join(args.models)
    
    # Load features
    train_path = get_feature_path(args.dataset, args.models, "train", args.method)
    test_path = get_feature_path(args.dataset, args.models, "test", args.method)
    
    print(f"Loading features:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
    
    print(f"\nTraining {args.method.upper()} fusion on {args.dataset}")
    print(f"Models: {' + '.join(args.models).upper()}")
    print(f"Feature dims: {feature_dims}")
    
    # Create model
    model = create_fusion_model(
        method=args.method,
        feature_dims=feature_dims,
        num_classes=num_classes,
    )
    
    return train_model(model, train_data, test_data, args, name=f"{model_str}_{args.method}")


def train_comm(args):
    """Train COMM fusion (multi-layer features)"""
    from src.training import create_fusion_model
    from src.data import DATASET_INFO
    
    use_mae = args.method == 'comm3'
    method_name = "COMM3" if use_mae else "COMM"
    
    # Load multi-layer features
    train_data = torch.load(f"features/{args.dataset}_{args.method}_train.pt")
    test_data = torch.load(f"features/{args.dataset}_{args.method}_test.pt")
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    
    print(f"\nTraining {method_name} fusion on {args.dataset}")
    
    model = create_fusion_model(
        method=args.method,
        feature_dims=[],  # COMM doesn't use this
        num_classes=num_classes,
        use_mae=use_mae,
    )
    
    return train_comm_model(model, train_data, test_data, args, name=args.method)


def train_comm_model(model, train_data, test_data, args, name="comm"):
    """Training loop for COMM models (multi-layer features)"""
    device = args.device
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    use_mae = 'mae_layer_6' in train_data
    
    best_acc = 0
    best_state = None
    
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(len(train_data["labels"]))
        
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            
            # Prepare multi-layer features
            clip_feats = [train_data[f"clip_layer_{j}"][batch_idx].float().to(device) for j in range(12)]
            dino_feats = [train_data[f"dino_layer_{j}"][batch_idx].float().to(device) for j in range(6, 12)]
            mae_feats = None
            
            if use_mae:
                mae_feats = [train_data[f"mae_layer_{j}"][batch_idx].float().to(device) for j in range(6, 12)]
            
            labels = train_data["labels"][batch_idx].long().to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(clip_feats, dino_feats, mae_feats), labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            clip_feats = [test_data[f"clip_layer_{j}"].float().to(device) for j in range(12)]
            dino_feats = [test_data[f"dino_layer_{j}"].float().to(device) for j in range(6, 12)]
            mae_feats = None
            
            if use_mae:
                mae_feats = [test_data[f"mae_layer_{j}"].float().to(device) for j in range(6, 12)]
            
            labels = test_data["labels"].long().to(device)
            pred = model(clip_feats, dino_feats, mae_feats).argmax(dim=1)
            acc = (pred == labels).float().mean().item() * 100
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    model.load_state_dict(best_state)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    path = f"outputs/checkpoints/{args.dataset}_{name}.pth"
    torch.save(best_state, path)
    print(f"\nSaved to: {path}")
    return best_acc


def train_model(model, train_data, test_data, args, name="model"):
    """Generic training loop for standard fusion methods"""
    device = args.device
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    X_train = train_data["features"].float().to(device)
    y_train = train_data["labels"].long().to(device)
    X_test = test_data["features"].float().to(device)
    y_test = test_data["labels"].long().to(device)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(args.epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item() * 100
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    model.load_state_dict(best_state)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    path = f"outputs/checkpoints/{args.dataset}_{name}.pth"
    torch.save(best_state, path)
    print(f"\nSaved to: {path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    
    # Model selection
    parser.add_argument("--model", type=str, choices=["clip", "dino", "mae"],
                        help="Single model (baseline)")
    parser.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"],
                        help="Multiple models for fusion")
    
    # Method selection
    parser.add_argument("--method", type=str, default="concat",
                        choices=["concat", "weighted_sum", "mmvit", "mmvit_lite", "comm", "comm3"],
                        help="Fusion method")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "flowers102", "pets", "stanford_cars", "food101"])
    
    # Training params
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model is None and args.models is None:
        print("Error: Must specify --model or --models")
        sys.exit(1)
    
    # Single model (use concat as wrapper)
    if args.model:
        args.models = [args.model]
        args.method = "concat"
    
    # COMM methods need specific feature files
    if args.method in ['comm', 'comm3']:
        train_comm(args)
    else:
        train_standard(args)


if __name__ == "__main__":
    main()

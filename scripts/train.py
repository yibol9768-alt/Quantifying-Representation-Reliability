"""
Training script

Usage:
    python scripts/train.py --model clip --dataset cifar10
    python scripts/train.py --models clip dino --dataset cifar10 --method concat
    python scripts/train.py --models clip dino --dataset cifar10 --method mmvit
    python scripts/train.py --method comm --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_model(model, train_data, test_data, args, name="model"):
    """Generic training loop"""
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


def train_single(args):
    """Train single model classifier"""
    from src.training import SingleViewClassifier
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    train_data = torch.load(f"features/{args.dataset}_{args.model}_train.pt")
    test_data = torch.load(f"features/{args.dataset}_{args.model}_test.pt")
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dim = MODEL_CONFIGS[args.model]["feature_dim"]
    
    print(f"Training {args.model.upper()} on {args.dataset}")
    model = SingleViewClassifier(feature_dim=feature_dim, num_classes=num_classes)
    return train_model(model, train_data, test_data, args, name=args.model)


def train_concat(args):
    """Train concat fusion classifier"""
    from src.training import MultiViewClassifier
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    model_str = "_".join(args.models)
    train_data = torch.load(f"features/{args.dataset}_{model_str}_train.pt")
    test_data = torch.load(f"features/{args.dataset}_{model_str}_test.pt")
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
    
    print(f"Training {'+'.join(args.models).upper()} CONCAT on {args.dataset}")
    model = MultiViewClassifier(feature_dims=feature_dims, num_classes=num_classes)
    return train_model(model, train_data, test_data, args, name=f"{model_str}_concat")


def train_mmvit(args):
    """Train MMViT fusion classifier (cross-attention based)"""
    from src.training import MMViTFusionClassifier
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    model_str = "_".join(args.models)
    train_data = torch.load(f"features/{args.dataset}_{model_str}_train.pt")
    test_data = torch.load(f"features/{args.dataset}_{model_str}_test.pt")
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
    
    print(f"Training {'+'.join(args.models).upper()} MMViT on {args.dataset}")
    print(f"Using cross-attention fusion")
    
    model = MMViTFusionClassifier(
        feature_dims=feature_dims,
        num_classes=num_classes,
        hidden_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    )
    return train_model(model, train_data, test_data, args, name=f"{model_str}_mmvit")


def train_mmvit_lite(args):
    """Train MMViT-Lite fusion classifier"""
    from src.training import MMViTLiteFusionClassifier
    from src.data import DATASET_INFO
    from configs.config import MODEL_CONFIGS
    
    model_str = "_".join(args.models)
    train_data = torch.load(f"features/{args.dataset}_{model_str}_train.pt")
    test_data = torch.load(f"features/{args.dataset}_{model_str}_test.pt")
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in args.models]
    
    print(f"Training {'+'.join(args.models).upper()} MMViT-Lite on {args.dataset}")
    
    model = MMViTLiteFusionClassifier(
        feature_dims=feature_dims,
        num_classes=num_classes,
        hidden_dim=512,
        num_heads=8,
    )
    return train_model(model, train_data, test_data, args, name=f"{model_str}_mmvit_lite")


def train_comm(args):
    """Train COMM fusion classifier"""
    from src.training import COMMFusionClassifier
    from src.data import DATASET_INFO
    
    train_data = torch.load(f"features/{args.dataset}_comm_train.pt")
    test_data = torch.load(f"features/{args.dataset}_comm_test.pt")
    
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    
    print(f"Training COMM fusion on {args.dataset}")
    
    model = COMMFusionClassifier(
        clip_hidden_dim=768, clip_output_dim=512, clip_num_layers=12,
        dino_hidden_dim=768, dino_num_layers=6,
        num_classes=num_classes
    )
    return train_comm_model(model, train_data, test_data, args)


def train_comm_model(model, train_data, test_data, args):
    """Training loop for COMM (multi-layer features)"""
    device = args.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(len(train_data["labels"]))
        
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            
            clip_feats = [train_data[f"clip_layer_{j}"][batch_idx].float().to(device) for j in range(12)]
            dino_feats = [train_data[f"dino_layer_{j}"][batch_idx].float().to(device) for j in range(6, 12)]
            labels = train_data["labels"][batch_idx].long().to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(clip_feats, dino_feats), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            clip_feats = [test_data[f"clip_layer_{j}"].float().to(device) for j in range(12)]
            dino_feats = [test_data[f"dino_layer_{j}"].float().to(device) for j in range(6, 12)]
            labels = test_data["labels"].long().to(device)
            
            pred = model(clip_feats, dino_feats).argmax(dim=1)
            acc = (pred == labels).float().mean().item() * 100
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Test Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    model.load_state_dict(best_state)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    path = f"outputs/checkpoints/{args.dataset}_comm.pth"
    torch.save(best_state, path)
    print(f"\nSaved to: {path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Training")
    
    parser.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    parser.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])
    parser.add_argument("--method", type=str, choices=["concat", "mmvit", "mmvit_lite", "comm", "comm3"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Determine training mode
    if args.model:
        train_single(args)
    elif args.models:
        if args.method == "mmvit":
            train_mmvit(args)
        elif args.method == "mmvit_lite":
            train_mmvit_lite(args)
        else:  # concat (default)
            train_concat(args)
    elif args.method == "comm":
        train_comm(args)
    elif args.method == "comm3":
        print("COMM3 not implemented yet")
    else:
        print("Error: Must specify --model, --models, or --method")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Feature extraction script

Usage:
    # Single model
    python scripts/extract.py --model clip --dataset cifar10
    
    # Multi-model (concat fusion)
    python scripts/extract.py --models clip dino --dataset cifar10
    
    # COMM multi-layer features
    python scripts/extract.py --method comm --dataset cifar10
    
    # COMM3 (3 models) multi-layer features
    python scripts/extract.py --method comm3 --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm


def extract_single(args):
    """Extract single model features"""
    from src.models import get_model
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting {args.model.upper()} features for {args.dataset} ({args.split})")
    
    model = get_model(args.model, device=args.device)
    features_list, labels_list = [], []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting"):
            img = dataset.get_image(img_path)
            feat = model.extract_feature(img)
            features_list.append(feat.cpu())
    
    features = torch.cat(features_list, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {"features": features, "labels": labels_tensor, "model": args.model}


def extract_multi(args):
    """Extract multi-model features (concat fusion)"""
    from src.models import get_model
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting {'+'.join(args.models).upper()} features for {args.dataset} ({args.split})")
    
    models = {m: get_model(m, device=args.device) for m in args.models}
    features_list = {m: [] for m in args.models}
    labels_list = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting"):
            img = dataset.get_image(img_path)
            for m in args.models:
                feat = models[m].extract_feature(img)
                features_list[m].append(feat.cpu())
            labels_list.append(labels[image_paths.index(img_path)] if img_path in image_paths else 0)
    
    # Concatenate features
    all_features = torch.cat([torch.cat(features_list[m], dim=0) for m in args.models], dim=1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {"features": all_features, "labels": labels_tensor, "models": args.models}


def extract_comm(args):
    """Extract COMM multi-layer features (CLIP + DINO)"""
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting COMM multi-layer features for {args.dataset} ({args.split})")
    
    clip_model = CLIPMultiLayerModel(device=args.device)
    dino_model = DINOMultiLayerModel(device=args.device)
    
    clip_features = {f"clip_layer_{i}": [] for i in range(12)}
    dino_features = {f"dino_layer_{i}": [] for i in range(6, 12)}
    labels_list = []
    
    with torch.no_grad():
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Extracting"):
            img = dataset.get_image(img_path)
            
            # CLIP multi-layer
            clip_feats = clip_model.extract_multilayer_features(img)
            for i in range(12):
                clip_features[f"clip_layer_{i}"].append(clip_feats[f"layer_{i}"].cpu())
            
            # DINO multi-layer
            dino_feats = dino_model.extract_multilayer_features(img)
            for i in range(6, 12):
                dino_features[f"dino_layer_{i}"].append(dino_feats[i].cpu())
            
            labels_list.append(label)
    
    result = {k: torch.cat(v, dim=0) for k, v in clip_features.items()}
    result.update({k: torch.cat(v, dim=0) for k, v in dino_features.items()})
    result["labels"] = torch.tensor(labels_list, dtype=torch.long)
    result["method"] = "comm"
    
    return result


def extract_comm3(args):
    """Extract COMM3 multi-layer features (CLIP + DINO + MAE)"""
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel, MAEMultiLayerModel
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting COMM3 multi-layer features for {args.dataset} ({args.split})")
    
    clip_model = CLIPMultiLayerModel(device=args.device)
    dino_model = DINOMultiLayerModel(device=args.device)
    mae_model = MAEMultiLayerModel(device=args.device)
    
    clip_features = {f"clip_layer_{i}": [] for i in range(12)}
    dino_features = {f"dino_layer_{i}": [] for i in range(6, 12)}
    mae_features = {f"mae_layer_{i}": [] for i in range(6, 12)}
    labels_list = []
    
    with torch.no_grad():
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Extracting"):
            img = dataset.get_image(img_path)
            
            # CLIP multi-layer
            clip_feats = clip_model.extract_multilayer_features(img)
            for i in range(12):
                clip_features[f"clip_layer_{i}"].append(clip_feats[f"layer_{i}"].cpu())
            
            # DINO multi-layer
            dino_feats = dino_model.extract_multilayer_features(img)
            for i in range(6, 12):
                dino_features[f"dino_layer_{i}"].append(dino_feats[i].cpu())
            
            # MAE multi-layer
            mae_feats = mae_model.extract_multilayer_features(img)
            for i in range(6, 12):
                mae_features[f"mae_layer_{i}"].append(mae_feats[i].cpu())
            
            labels_list.append(label)
    
    result = {k: torch.cat(v, dim=0) for k, v in clip_features.items()}
    result.update({k: torch.cat(v, dim=0) for k, v in dino_features.items()})
    result.update({k: torch.cat(v, dim=0) for k, v in mae_features.items()})
    result["labels"] = torch.tensor(labels_list, dtype=torch.long)
    result["method"] = "comm3"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Feature extraction")
    
    # Method selection
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"],
                              help="Single model")
    method_group.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"],
                              help="Multiple models for concat fusion")
    method_group.add_argument("--method", type=str, choices=["comm", "comm3"],
                              help="Multi-layer fusion method")
    
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "flowers102", "pets", "stanford_cars", "food101"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="features")
    
    args = parser.parse_args()
    
    # Extract
    if args.model:
        data = extract_single(args)
        name = f"{args.dataset}_{args.model}_{args.split}.pt"
    elif args.models:
        data = extract_multi(args)
        name = f"{args.dataset}_{'_'.join(args.models)}_{args.split}.pt"
    elif args.method == "comm":
        data = extract_comm(args)
        name = f"{args.dataset}_comm_{args.split}.pt"
    elif args.method == "comm3":
        data = extract_comm3(args)
        name = f"{args.dataset}_comm3_{args.split}.pt"
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, name)
    torch.save(data, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

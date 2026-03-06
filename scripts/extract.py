"""
Feature extraction script (optimized with batching)

Usage:
    python scripts/extract.py --model clip --dataset cifar10
    python scripts/extract.py --models clip dino --dataset cifar10
    python scripts/extract.py --method comm --dataset cifar10
    python scripts/extract.py --method comm3 --dataset cifar10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_single(args):
    """Extract single model features with batching"""
    from src.models import get_model
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting {args.model.upper()} features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")
    
    model = get_model(args.model, device=args.device)
    model.eval()
    
    features_list = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Extracting"):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_images = torch.stack([model.get_transform()(dataset.get_image(p)) for p in batch_paths])
        batch_images = batch_images.to(args.device)
        
        with torch.no_grad():
            features = model.extract_batch_features(batch_images)
        features_list.append(features.cpu())
    
    features = torch.cat(features_list, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {"features": features, "labels": labels_tensor, "model": args.model}


def extract_multi(args):
    """Extract multi-model features with batching"""
    from src.models import get_model
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting {'+'.join(args.models).upper()} features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")
    
    models = {m: get_model(m, device=args.device) for m in args.models}
    for m in models.values():
        m.eval()
    
    features_list = {m: [] for m in args.models}
    
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Extracting"):
        batch_paths = image_paths[i:i+args.batch_size]
        
        for m in args.models:
            batch_images = torch.stack([models[m].get_transform()(dataset.get_image(p)) for p in batch_paths])
            batch_images = batch_images.to(args.device)
            
            with torch.no_grad():
                features = models[m].extract_batch_features(batch_images)
            features_list[m].append(features.cpu())
    
    all_features = torch.cat([torch.cat(features_list[m], dim=0) for m in args.models], dim=1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {"features": all_features, "labels": labels_tensor, "models": args.models}


def extract_comm(args):
    """Extract COMM multi-layer features with batching"""
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel
    from src.data import get_dataset
    import torchvision.transforms as T
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting COMM multi-layer features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")
    
    clip_model = CLIPMultiLayerModel(device=args.device)
    dino_model = DINOMultiLayerModel(device=args.device)
    clip_model.eval()
    dino_model.eval()
    
    # Preprocessing transforms
    clip_transform = clip_model.get_transform()
    dino_transform = dino_model.get_transform()
    
    clip_features = {f"clip_layer_{i}": [] for i in range(12)}
    dino_features = {f"dino_layer_{i}": [] for i in range(6, 12)}
    
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Extracting"):
        batch_paths = image_paths[i:i+args.batch_size]
        
        # CLIP batch
        clip_batch = torch.stack([clip_transform(dataset.get_image(p)) for p in batch_paths])
        clip_batch = clip_batch.to(args.device)
        
        with torch.no_grad():
            clip_feats = clip_model.extract_batch_multilayer_features(clip_batch)
        for j in range(12):
            clip_features[f"clip_layer_{j}"].append(clip_feats[j].cpu())
        
        # DINO batch
        dino_batch = torch.stack([dino_transform(dataset.get_image(p)) for p in batch_paths])
        dino_batch = dino_batch.to(args.device)
        
        with torch.no_grad():
            dino_feats = dino_model.extract_batch_multilayer_features(dino_batch)
        for j in range(6, 12):
            dino_features[f"dino_layer_{j}"].append(dino_feats[j].cpu())
    
    result = {k: torch.cat(v, dim=0) for k, v in clip_features.items()}
    result.update({k: torch.cat(v, dim=0) for k, v in dino_features.items()})
    result["labels"] = torch.tensor(labels, dtype=torch.long)
    result["method"] = "comm"
    
    return result


def extract_comm3(args):
    """Extract COMM3 multi-layer features with batching"""
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel, MAEMultiLayerModel
    from src.data import get_dataset
    
    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    
    print(f"Extracting COMM3 multi-layer features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")
    
    clip_model = CLIPMultiLayerModel(device=args.device)
    dino_model = DINOMultiLayerModel(device=args.device)
    mae_model = MAEMultiLayerModel(device=args.device)
    clip_model.eval()
    dino_model.eval()
    mae_model.eval()
    
    clip_transform = clip_model.get_transform()
    dino_transform = dino_model.get_transform()
    mae_transform = mae_model.get_transform()
    
    clip_features = {f"clip_layer_{i}": [] for i in range(12)}
    dino_features = {f"dino_layer_{i}": [] for i in range(6, 12)}
    mae_features = {f"mae_layer_{i}": [] for i in range(6, 12)}
    
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Extracting"):
        batch_paths = image_paths[i:i+args.batch_size]
        
        # CLIP batch
        clip_batch = torch.stack([clip_transform(dataset.get_image(p)) for p in batch_paths])
        clip_batch = clip_batch.to(args.device)
        
        with torch.no_grad():
            clip_feats = clip_model.extract_batch_multilayer_features(clip_batch)
        for j in range(12):
            clip_features[f"clip_layer_{j}"].append(clip_feats[j].cpu())
        
        # DINO batch
        dino_batch = torch.stack([dino_transform(dataset.get_image(p)) for p in batch_paths])
        dino_batch = dino_batch.to(args.device)
        
        with torch.no_grad():
            dino_feats = dino_model.extract_batch_multilayer_features(dino_batch)
        for j in range(6, 12):
            dino_features[f"dino_layer_{j}"].append(dino_feats[j].cpu())
        
        # MAE batch
        mae_batch = torch.stack([mae_transform(dataset.get_image(p)) for p in batch_paths])
        mae_batch = mae_batch.to(args.device)
        
        with torch.no_grad():
            mae_feats = mae_model.extract_batch_multilayer_features(mae_batch)
        for j in range(6, 12):
            mae_features[f"mae_layer_{j}"].append(mae_feats[j].cpu())
    
    result = {k: torch.cat(v, dim=0) for k, v in clip_features.items()}
    result.update({k: torch.cat(v, dim=0) for k, v in dino_features.items()})
    result.update({k: torch.cat(v, dim=0) for k, v in mae_features.items()})
    result["labels"] = torch.tensor(labels, dtype=torch.long)
    result["method"] = "comm3"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Feature extraction")
    
    # Method selection
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    method_group.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])
    method_group.add_argument("--method", type=str, choices=["comm", "comm3"])
    
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for GPU processing")
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
    print(f"Feature shape: {data.get('features', list(data.values())[0]).shape if 'features' in data else 'multi-layer'}")


if __name__ == "__main__":
    main()

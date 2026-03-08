"""Main entry point - optimized for GPU."""

import argparse
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from configs.config import Config, DATASET_CONFIGS
from src.models.extractors import get_extractor
from src.models.mlp import MLPClassifier
from src.data.hf_dataset import get_dataloaders
from src.training.offline_cache import (
    CachedShardDataset,
    GroupedShardSampler,
    build_split_cache,
    cleanup_cache_dir,
    move_to_device,
)

DEFAULT_MODEL_DIR = "./models"
DEFAULT_DATA_DIR = "./data"
DEFAULT_CACHE_DIR = "./cache/offline"


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Classification")

    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", type=str, default="mae",
                        choices=["mae", "clip", "dino", "fusion"])
    parser.add_argument("--fusion_method", type=str, default="concat",
                        choices=["concat", "comm", "mmvit"],
                        help="Fusion method when --model fusion")
    parser.add_argument("--fusion_models", type=str, default="mae,clip,dino",
                        help="Comma-separated model list for fusion, e.g. clip,dino or mae,clip,dino")
    parser.add_argument("--fusion_output_dim", type=int, default=1024,
                        help="Unified fusion output dim for fair horizontal comparison")
    parser.add_argument("--disable_fusion_harmonization", action="store_true",
                        help="Disable fair-comparison harmonization for fusion methods")
    parser.add_argument("--comm_dino_mlp_blocks", type=int, default=2,
                        help="COMM: number of residual MLP blocks for DINO/text-space alignment")
    parser.add_argument("--comm_dino_mlp_ratio", type=float, default=8.0,
                        help="COMM: expansion ratio of each DINO MLP block")
    parser.add_argument("--mmvit_base_dim", type=int, default=96,
                        help="MMViT: base embedding dim of stage-1")
    parser.add_argument("--mmvit_mlp_ratio", type=float, default=4.0,
                        help="MMViT: MLP expansion ratio in attention blocks")
    parser.add_argument("--mmvit_num_heads", type=int, default=8,
                        help="MMViT: preferred number of attention heads per stage")
    parser.add_argument("--mmvit_max_position_tokens", type=int, default=256,
                        help="MMViT: reference length of learnable positional embeddings")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--storage_dir", type=str, default=None,
                        help="Root directory for large files: models/, data/, cache/")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_precompute", action="store_true",
                        help="Optional fallback: disable offline cache and train from raw images")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Directory for disk-backed offline caches")
    parser.add_argument("--cache_dtype", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="Storage dtype for offline cache tensors")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuilding offline cache files")
    parser.add_argument("--cleanup_cache", action="store_true",
                        help="Delete generated cache files after training finishes")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (faster on modern GPUs)")

    return parser.parse_args()


def parse_fusion_models(fusion_models: str):
    """Parse fusion model list from CLI."""
    valid_models = {"mae", "clip", "dino"}
    models = [name.strip().lower() for name in fusion_models.split(",") if name.strip()]

    # Preserve order while removing duplicates.
    seen = set()
    unique_models = []
    for name in models:
        if name not in seen:
            unique_models.append(name)
            seen.add(name)

    if not unique_models:
        raise ValueError("fusion_models is empty. Example: --fusion_models clip,dino")
    if any(name not in valid_models for name in unique_models):
        raise ValueError(f"Invalid model in fusion_models: {unique_models}. Valid: {sorted(valid_models)}")
    if len(unique_models) < 2:
        raise ValueError("Fusion requires at least 2 models.")
    if len(unique_models) > 3:
        raise ValueError("At most 3 models are supported.")

    return unique_models


def get_checkpoint_name(args) -> str:
    """Build checkpoint name."""
    if args.model == "fusion":
        model_tag = "-".join(args.fusion_model_list)
        if use_fusion_harmonization(args):
            return (
                f"{args.dataset}_fusion-{args.fusion_method}_{model_tag}"
                f"_dim{args.fusion_output_dim}_best.pth"
            )
        return f"{args.dataset}_fusion-{args.fusion_method}_{model_tag}_best.pth"
    return f"{args.dataset}_{args.model}_best.pth"


def get_fusion_kwargs(args) -> dict:
    """Build fusion kwargs from CLI arguments."""
    return {
        "comm_dino_mlp_blocks": args.comm_dino_mlp_blocks,
        "comm_dino_mlp_ratio": args.comm_dino_mlp_ratio,
        "mmvit_base_dim": args.mmvit_base_dim,
        "mmvit_mlp_ratio": args.mmvit_mlp_ratio,
        "mmvit_num_heads": args.mmvit_num_heads,
        "mmvit_max_position_tokens": args.mmvit_max_position_tokens,
        "fusion_output_dim": args.fusion_output_dim if use_fusion_harmonization(args) else None,
    }


def is_trainable_fusion(args) -> bool:
    """Whether current setup uses trainable fusion modules."""
    if args.model != "fusion":
        return False
    # In harmonized mode, concat has a trainable projection for fair comparison.
    if use_fusion_harmonization(args):
        return True
    return args.fusion_method in {"comm", "mmvit"}


def use_fusion_harmonization(args) -> bool:
    """Whether to harmonize fusion settings for fair horizontal comparison."""
    return args.model == "fusion" and (not args.disable_fusion_harmonization)


def set_random_seed(seed: int):
    """Set random seed for fair and reproducible comparisons."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB used / {reserved:.2f}GB reserved")


def resolve_storage_paths(args):
    """Resolve model/data/cache paths from a shared storage root."""
    if args.storage_dir is None:
        args.model_dir = DEFAULT_MODEL_DIR
        return

    storage_root = Path(args.storage_dir)
    args.model_dir = str(storage_root / "models")

    if args.data_dir == DEFAULT_DATA_DIR:
        args.data_dir = str(storage_root / "data")
    if args.cache_dir == DEFAULT_CACHE_DIR:
        args.cache_dir = str(storage_root / "cache" / "offline")


def get_cache_storage_dtype(args) -> torch.dtype:
    """Map CLI cache dtype to torch dtype."""
    return torch.float16 if args.cache_dtype == "fp16" else torch.float32


def get_cache_name(args) -> str:
    """Build a cache directory name for the current experiment."""
    if args.model == "fusion":
        model_tag = "-".join(args.fusion_model_list)
        parts = [args.dataset, "fusion", args.fusion_method, model_tag]
        if use_fusion_harmonization(args):
            parts.append(f"dim{args.fusion_output_dim}")
    else:
        parts = [args.dataset, args.model]

    parts.append(f"seed{args.seed}")
    parts.append(f"cache{args.cache_dtype}")
    return "_".join(parts).replace(".", "p")


def has_trainable_extractor(extractor) -> bool:
    """Check whether the cached extractor still has trainable layers."""
    return any(param.requires_grad for param in extractor.parameters())


def save_offline_checkpoint(args, classifier, extractor):
    """Save classifier and any trainable post-cache fusion layers."""
    payload = {
        "classifier": classifier.state_dict(),
        "args": vars(args),
    }
    if has_trainable_extractor(extractor):
        payload["extractor"] = extractor.state_dict()
    torch.save(payload, get_checkpoint_name(args))


def build_cached_loaders(args, extractor, device, use_fp16):
    """Build disk-backed offline caches and dataloaders."""
    cache_root = Path(args.cache_dir) / get_cache_name(args)
    train_split_dir = cache_root / "train"
    test_split_dir = cache_root / "test"

    print(f"\nLoading dataset {args.dataset}...")
    image_train_loader, image_test_loader = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.loader_model_type
    )

    storage_dtype = get_cache_storage_dtype(args)

    print(f"\n[Step 1/3] Building train cache at {train_split_dir}")
    build_split_cache(
        extractor=extractor,
        dataloader=image_train_loader,
        split_dir=train_split_dir,
        split="train",
        device=device,
        storage_dtype=storage_dtype,
        use_fp16=use_fp16,
        recache=args.rebuild_cache,
    )

    print(f"\n[Step 2/3] Building test cache at {test_split_dir}")
    build_split_cache(
        extractor=extractor,
        dataloader=image_test_loader,
        split_dir=test_split_dir,
        split="test",
        device=device,
        storage_dtype=storage_dtype,
        use_fp16=use_fp16,
        recache=args.rebuild_cache,
    )

    extractor.release_backbones()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = CachedShardDataset(train_split_dir)
    test_dataset = CachedShardDataset(test_split_dir)
    train_sampler = GroupedShardSampler(train_dataset, shuffle=True, seed=args.seed)
    test_sampler = GroupedShardSampler(test_dataset, shuffle=False, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    return train_loader, test_loader, cache_root


def train_with_offline_cache(args, config):
    """Train from disk-backed offline caches."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_fp16}")
    print(f"Cache dtype: {args.cache_dtype}")
    print_gpu_usage()

    # Feature extractor
    print(f"\nLoading {args.model} model...")
    extractor = get_extractor(
        args.model,
        fusion_method=args.fusion_method,
        fusion_models=args.fusion_model_list,
        fusion_kwargs=get_fusion_kwargs(args),
        model_dir=args.model_dir,
    ).to(device)
    print(f"Feature dimension: {extractor.feature_dim}")
    print_gpu_usage()

    train_loader, test_loader, cache_root = build_cached_loaders(args, extractor, device, use_fp16)

    # Classifier
    classifier = MLPClassifier(
        feature_dim=extractor.feature_dim,
        num_classes=config.num_classes,
        hidden_dim=args.hidden_dim
    ).to(device)

    trainable_extractor = has_trainable_extractor(extractor)
    optim_params = list(classifier.parameters())
    if trainable_extractor:
        trainable_params = [param for param in extractor.parameters() if param.requires_grad]
        optim_params.extend(trainable_params)
        print(f"Trainable fusion params: {sum(param.numel() for param in trainable_params):,}")

    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("\n" + "=" * 60)
    print("Starting training (offline cache mode)...")
    print("=" * 60)

    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if trainable_extractor:
            extractor.train()
        else:
            extractor.eval()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for cached_inputs, labels in pbar:
            cached_inputs = move_to_device(cached_inputs, device)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if trainable_extractor:
                if use_fp16:
                    with autocast():
                        features = extractor.forward_from_cache(cached_inputs)
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    features = extractor.forward_from_cache(cached_inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    if use_fp16:
                        with autocast():
                            features = extractor.forward_from_cache(cached_inputs)
                    else:
                        features = extractor.forward_from_cache(cached_inputs)

                if use_fp16:
                    with autocast():
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*train_correct/train_total:.1f}%"
            })

        scheduler.step()

        # Evaluate
        classifier.eval()
        extractor.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for cached_inputs, labels in test_loader:
                cached_inputs = move_to_device(cached_inputs, device)
                labels = labels.to(device, non_blocking=True)

                if use_fp16:
                    with autocast():
                        features = extractor.forward_from_cache(cached_inputs)
                        outputs = classifier(features)
                else:
                    features = extractor.forward_from_cache(cached_inputs)
                    outputs = classifier(features)

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            save_offline_checkpoint(args, classifier, extractor)

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Best {best_acc:.2f}%")

    if args.cleanup_cache:
        cleanup_cache_dir(cache_root)
        print(f"\nRemoved offline cache: {cache_root}")

    return best_acc


def train_online(args, config):
    """Train with online feature extraction."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_fp16}")
    print_gpu_usage()

    # Load data
    print(f"\nLoading dataset {args.dataset}...")
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.loader_model_type
    )

    # Feature extractor
    print(f"\nLoading {args.model} model...")
    extractor = get_extractor(
        args.model,
        fusion_method=args.fusion_method,
        fusion_models=args.fusion_model_list,
        fusion_kwargs=get_fusion_kwargs(args),
        model_dir=args.model_dir,
    ).to(device)
    fusion_trainable = is_trainable_fusion(args)
    if fusion_trainable:
        extractor.train()
    else:
        extractor.eval()
    print(f"Feature dimension: {extractor.feature_dim}")
    print_gpu_usage()

    # Classifier
    classifier = MLPClassifier(
        feature_dim=extractor.feature_dim,
        num_classes=config.num_classes,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Training
    optim_params = list(classifier.parameters())
    if fusion_trainable:
        trainable_extractor_params = [p for p in extractor.parameters() if p.requires_grad]
        optim_params.extend(trainable_extractor_params)
        print(f"Trainable fusion params: {sum(p.numel() for p in trainable_extractor_params):,}")

    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("\n" + "=" * 60)
    print("Starting training (online mode)...")
    print("=" * 60)

    for epoch in range(args.epochs):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if fusion_trainable:
                if use_fp16:
                    with autocast():
                        features = extractor(images)
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    features = extractor(images)
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    if use_fp16:
                        with autocast():
                            features = extractor(images)
                    else:
                        features = extractor(images)

                if use_fp16:
                    with autocast():
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*train_correct/train_total:.1f}%"
            })

        scheduler.step()

        # Evaluate
        classifier.eval()
        extractor.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if use_fp16:
                    with autocast():
                        features = extractor(images)
                        outputs = classifier(features)
                else:
                    features = extractor(images)
                    outputs = classifier(features)

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        if fusion_trainable:
            extractor.train()
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(classifier.state_dict(), get_checkpoint_name(args))

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Best {best_acc:.2f}%")

    return best_acc


def main():
    args = parse_args()
    set_random_seed(args.seed)
    resolve_storage_paths(args)

    if args.model == "fusion":
        args.fusion_model_list = parse_fusion_models(args.fusion_models)
        args.loader_model_type = "fusion"
    else:
        args.fusion_model_list = None
        args.loader_model_type = args.model

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = Config(
        dataset=args.dataset,
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    print("=" * 60)
    print("Feature Classification with HuggingFace Transformers")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({config.num_classes} classes)")
    print(f"Model: {args.model}")
    if args.model == "fusion":
        print(f"Fusion method: {args.fusion_method}")
        print(f"Fusion models: {args.fusion_model_list}")
        print(f"Fusion harmonization: {use_fusion_harmonization(args)}")
        if use_fusion_harmonization(args):
            print(f"Unified fusion output dim: {args.fusion_output_dim}")
        print(f"Fusion trainable: {is_trainable_fusion(args)}")
    print(f"Seed: {args.seed}")
    print(f"Offline cache mode: {not args.no_precompute}")
    print(f"Mixed precision (fp16): {args.fp16}")
    print(f"Storage dir: {args.storage_dir if args.storage_dir is not None else '(repo defaults)'}")
    print(f"Model dir: {args.model_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Cache dtype: {args.cache_dtype}")
    print(f"Rebuild cache: {args.rebuild_cache}")
    print(f"Cleanup cache: {args.cleanup_cache}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Train
    if not args.no_precompute:
        best_acc = train_with_offline_cache(args, config)
    else:
        best_acc = train_online(args, config)

    print("\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

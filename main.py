"""Main entry point - optimized for GPU."""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time

from configs.config import Config, DATASET_CONFIGS
from src.models.extractors import get_extractor
from src.models.mlp import MLPClassifier
from src.data.hf_dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Classification")

    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", type=str, default="mae",
                        choices=["mae", "clip", "dino", "fusion"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precompute", action="store_true",
                        help="Pre-compute features for faster training")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (faster on modern GPUs)")

    return parser.parse_args()


def print_gpu_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB used / {reserved:.2f}GB reserved")


@torch.no_grad()
def precompute_features(extractor, dataloader, device, use_fp16=False):
    """Pre-compute features for entire dataset - optimized."""
    all_features = []
    all_labels = []

    extractor.eval()
    total_time = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Extracting features")
    for images, labels in pbar:
        start = time.time()

        images = images.to(device, non_blocking=True)

        if use_fp16:
            with autocast():
                features = extractor(images)
        else:
            features = extractor(images)

        # Keep on GPU for now, move at end
        all_features.append(features)
        all_labels.append(labels.to(device, non_blocking=True))

        batch_time = time.time() - start
        total_time += batch_time
        n_batches += 1

        pbar.set_postfix({
            "batch_time": f"{batch_time:.3f}s",
            "avg_time": f"{total_time/n_batches:.3f}s"
        })

    # Move to CPU at end
    features = torch.cat(all_features, dim=0).cpu()
    labels = torch.cat(all_labels, dim=0).cpu()

    return features, labels


def train_with_precomputed_features(args, config):
    """Train with pre-computed features (fastest)."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler() if use_fp16 else None

    print(f"\nDevice: {device}")
    print(f"Mixed precision: {use_fp16}")
    print_gpu_usage()

    # Load data
    print(f"\nLoading dataset {args.dataset}...")
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.model
    )

    # Feature extractor
    print(f"\nLoading {args.model} model...")
    extractor = get_extractor(args.model).to(device)
    extractor.eval()
    print(f"Feature dimension: {extractor.feature_dim}")
    print_gpu_usage()

    # Classifier
    classifier = MLPClassifier(
        feature_dim=extractor.feature_dim,
        num_classes=config.num_classes,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Pre-compute features
    print("\n[Step 1/2] Pre-computing training features...")
    train_features, train_labels = precompute_features(
        extractor, train_loader, device, use_fp16
    )

    print("\n[Step 2/2] Pre-computing test features...")
    test_features, test_labels = precompute_features(
        extractor, test_loader, device, use_fp16
    )

    print(f"\nTrain features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")

    # Free extractor memory
    del extractor
    torch.cuda.empty_cache()

    # Create feature loaders (num_workers=0 for in-memory data)
    train_feat_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_feat_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_features, test_labels),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Training
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(args.epochs):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_feat_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for features, labels in pbar:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

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

            train_loss += loss.item() * features.size(0)
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
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for features, labels in test_feat_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if use_fp16:
                    with autocast():
                        outputs = classifier(features)
                else:
                    outputs = classifier(features)

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(classifier.state_dict(), f"{args.dataset}_{args.model}_best.pth")

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Best {best_acc:.2f}%")

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
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.model
    )

    # Feature extractor
    print(f"\nLoading {args.model} model...")
    extractor = get_extractor(args.model).to(device)
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
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
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

            # Extract features
            with torch.no_grad():
                if use_fp16:
                    with autocast():
                        features = extractor(images)
                else:
                    features = extractor(images)

            # Train classifier
            optimizer.zero_grad()

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

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(classifier.state_dict(), f"{args.dataset}_{args.model}_best.pth")

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Best {best_acc:.2f}%")

    return best_acc


def main():
    args = parse_args()

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
    print(f"Precompute: {args.precompute}")
    print(f"Mixed precision (fp16): {args.fp16}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Train
    if args.precompute:
        best_acc = train_with_precomputed_features(args, config)
    else:
        best_acc = train_online(args, config)

    print("\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Main entry point - simplified with HuggingFace Transformers."""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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

    return parser.parse_args()


def train_with_precomputed_features(args, config):
    """Train with pre-computed features (faster)."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.model
    )

    # Feature extractor
    extractor = get_extractor(args.model).to(device)
    extractor.eval()

    # Classifier
    classifier = MLPClassifier(
        feature_dim=extractor.feature_dim,
        num_classes=config.num_classes,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Pre-compute features
    print("\nPre-computing features...")
    train_features, train_labels = precompute_features(extractor, train_loader, device)
    test_features, test_labels = precompute_features(extractor, test_loader, device)

    # Create feature loaders
    train_feat_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=args.batch_size, shuffle=True
    )
    test_feat_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_features, test_labels),
        batch_size=args.batch_size, shuffle=False
    )

    # Training
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in tqdm(train_feat_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Evaluate
        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for features, labels in test_feat_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(classifier.state_dict(), f"{args.dataset}_{args.model}_best.pth")

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%")

    return best_acc


def train_online(args, config):
    """Train with online feature extraction (more flexible)."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.model
    )

    # Feature extractor
    extractor = get_extractor(args.model).to(device)
    extractor.eval()

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

    for epoch in range(args.epochs):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Extract features (no grad)
            with torch.no_grad():
                features = extractor(images)

            # Train classifier
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Evaluate
        classifier.eval()
        extractor.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
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

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%")

    return best_acc


@torch.no_grad()
def precompute_features(extractor, dataloader, device):
    """Pre-compute features for entire dataset."""
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = extractor(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def main():
    args = parse_args()

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
    print(f"Device: {args.device}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Train
    if args.precompute:
        best_acc = train_with_precomputed_features(args, config)
    else:
        best_acc = train_online(args, config)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Accuracy: {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

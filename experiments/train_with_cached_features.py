#!/usr/bin/env python3
"""
快速训练脚本 - 使用预提取的特征
用法: python experiments/train_with_cached_features.py
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.models.mlp import MLPClassifier
from src.models.extractors import get_extractor


class CachedFeatureDataset(Dataset):
    """从磁盘加载预提取的特征"""

    def __init__(self, cache_dir: str, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split

        # 加载所有shard文件
        split_dir = self.cache_dir / split
        self.shards = sorted(list(split_dir.glob("shard_*.pt")))

        if not self.shards:
            raise ValueError(f"No shards found in {split_dir}")

        # 加载数据索引
        self.samples = []
        self.labels = []

        for shard_file in self.shards:
            shard_data = torch.load(shard_file)
            shard_indices = shard_data['indices']
            shard_labels = shard_data['labels']
            num_samples = len(shard_indices)

            start_idx = len(self.samples)
            self.samples.extend([(shard_file, i) for i in range(num_samples)])
            self.labels.extend(shard_labels.tolist())
            end_idx = len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_file, sample_idx = self.samples[idx]
        label = self.labels[idx]

        # 加载shard
        shard_data = torch.load(shard_file)
        features = shard_data['features'][sample_idx]

        return features, label


def train_single_model(
    model: str,
    dataset: str,
    feature_cache_dir: str,
    num_classes: int,
    feature_dim: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    seed: int = 42,
    device: str = "cuda:0",
    results_dir: str = "./results"
):
    """训练单个模型的分类器"""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cache_path = Path(feature_cache_dir) / f"{dataset}_{model}_seed{seed}"

    if not cache_path.exists():
        raise FileNotFoundError(f"Features not found: {cache_path}")

    # 加载数据
    print(f"Loading features from {cache_path}")
    train_set = CachedFeatureDataset(str(cache_path), "train")
    test_set = CachedFeatureDataset(str(cache_path), "test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_set)}, Test: {len(test_set)}")

    # 创建分类器
    classifier = MLPClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    results = []

    for epoch in range(epochs):
        # 训练
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*train_correct/train_total:.1f}%"})

        scheduler.step()

        # 验证
        classifier.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / train_total,
            "train_acc": train_acc,
            "test_loss": test_loss / test_total,
            "test_acc": test_acc,
        })

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save(classifier.state_dict(), f"{model}_{dataset}_best.pth")

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Best {best_acc:.2f}%")

    return best_acc, results


def main():
    parser = argparse.ArgumentParser(description="Train with cached features")
    parser.add_argument("--feature_cache_dir", type=str, required=True,
                        help="Directory containing cached features")
    parser.add_argument("--model", type=str, default="clip",
                        help="Model to train")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        help="Dataset to use")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Number of classes")
    parser.add_argument("--feature_dim", type=int, default=768,
                        help="Feature dimension")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="./results")

    args = parser.parse_args()

    # 训练
    best_acc, results = train_single_model(
        model=args.model,
        dataset=args.dataset,
        feature_cache_dir=args.feature_cache_dir,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
    )

    # 保存结果
    result_path = Path(args.results_dir) / f"{args.model}_{args.dataset}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, 'w') as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "best_acc": best_acc,
            "history": results,
        }, f, indent=2)

    print(f"\nBest Accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()

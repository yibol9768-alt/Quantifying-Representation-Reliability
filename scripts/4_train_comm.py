"""
Train COMM fusion model

This script trains a COMM fusion classifier using multi-layer features
from CLIP and DINO models.

COMM = CLIP and DINO with Multi-level features Merging

Usage:
    python scripts/4_train_comm.py --dataset stanford_cars
    python scripts/4_train_comm.py --dataset cifar10 --epochs 50

Fusion methods:
    - comm: Full COMM fusion (LLN-Layerscale + MLP alignment)
    - concat: Simple concatenation fusion (baseline)
    - weighted_sum: Weighted sum fusion
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.training import COMMClassifier, ConcatFusionClassifier, WeightedSumFusionClassifier, Trainer
from src.data import DATASET_INFO
from src.utils import get_device, set_seed, print_model_info


class COMMFeatureDataset(Dataset):
    """Dataset for COMM multi-layer features"""

    def __init__(self, features_dict: dict):
        """
        Args:
            features_dict: Dict with multi-layer features and labels
        """
        self.labels = features_dict["labels"]

        # Extract CLIP layer features (12 layers)
        self.clip_features = []
        for i in range(12):
            key = f"clip_layer_{i}"
            if key in features_dict:
                self.clip_features.append(features_dict[key])
            else:
                raise ValueError(f"Missing CLIP layer feature: {key}")

        # Extract DINO layer features (6 layers: 7-12)
        self.dino_features = []
        for i in range(6, 12):
            key = f"dino_layer_{i}"
            if key in features_dict:
                self.dino_features.append(features_dict[key])
            else:
                raise ValueError(f"Missing DINO layer feature: {key}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # CLIP features: list of 12 tensors [hidden_dim]
        clip_feats = [feat[idx] for feat in self.clip_features]
        # DINO features: list of 6 tensors [hidden_dim]
        dino_feats = [feat[idx] for feat in self.dino_features]
        label = self.labels[idx]

        return clip_feats, dino_feats, label


class COMMTrainer:
    """Trainer for COMM fusion models"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for clip_feats, dino_feats, labels in train_loader:
            # Move to device
            clip_feats = [[f.to(self.device).float() for f in clip_batch] for clip_batch in clip_feats]
            dino_feats = [[f.to(self.device).float() for f in dino_batch] for dino_batch in dino_feats]
            labels = labels.to(self.device).long()

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(clip_feats, dino_feats)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0

        for clip_feats, dino_feats, labels in data_loader:
            # Move to device
            clip_feats = [[f.to(self.device).float() for f in clip_batch] for clip_batch in clip_feats]
            dino_feats = [[f.to(self.device).float() for f in dino_batch] for dino_batch in dino_feats]
            labels = labels.to(self.device).long()

            # Forward pass
            logits = self.model(clip_feats, dino_feats)

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def fit(
        self,
        train_features: dict,
        test_features: dict,
        epochs: int = 30,
        batch_size: int = 256,
        val_split: float = 0.2,
    ) -> dict:
        """Train model"""
        # Create datasets
        train_dataset = COMMFeatureDataset(train_features)
        test_dataset = COMMFeatureDataset(test_features)

        # Split train into train/val
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {len(test_dataset)}")

        # Training loop
        history = {"train_loss": [], "val_acc": [], "test_acc": []}
        best_val_acc = 0

        for epoch in range(epochs):
            import time
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_acc = self.evaluate(val_loader)
            test_acc = self.evaluate(test_loader)

            # Record
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)

            # Print progress
            epoch_time = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save("best_comm_model.pth")

        print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")
        return history

    def save(self, path: str):
        """Save model weights"""
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def main():
    parser = argparse.ArgumentParser(description="Train COMM fusion model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "stanford_cars",
            "cifar10",
            "cifar100",
            "flowers102",
            "pets",
            "food101",
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="comm",
        choices=["comm", "concat", "weighted_sum"],
        help="Fusion method (comm=full COMM, concat=baseline concatenation, weighted_sum=weighted sum)",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="features",
        help="Feature directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output checkpoint path",
    )

    args = parser.parse_args()

    # Setup
    device = get_device()
    set_seed(42)

    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    print(f"Training COMM fusion model on {args.dataset}")
    print(f"Fusion method: {args.method}")
    print(f"Device: {device}, Classes: {num_classes}")

    # Load features
    train_path = os.path.join(args.feature_dir, f"{args.dataset}_comm_train.pt")
    test_path = os.path.join(args.feature_dir, f"{args.dataset}_comm_test.pt")

    print(f"\nLoading features from:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    if not os.path.exists(train_path):
        print(f"\nError: {train_path} not found!")
        print(f"Please run: python scripts/2_extract_comm.py --dataset {args.dataset} --split train")
        sys.exit(1)

    if not os.path.exists(test_path):
        print(f"\nError: {test_path} not found!")
        print(f"Please run: python scripts/2_extract_comm.py --dataset {args.dataset} --split test")
        sys.exit(1)

    train_features = torch.load(train_path, map_location="cpu")
    test_features = torch.load(test_path, map_location="cpu")

    # Build model based on method
    if args.method == "comm":
        model = COMMClassifier(
            clip_hidden_dim=768,
            clip_output_dim=512,
            clip_num_layers=12,
            dino_hidden_dim=768,
            dino_num_layers=6,
            num_classes=num_classes,
        )
    elif args.method == "concat":
        # Simple concatenation of all layer features
        model = ConcatFusionClassifier(
            clip_num_layers=12,
            clip_hidden_dim=768,
            dino_num_layers=6,
            dino_hidden_dim=768,
            num_classes=num_classes,
        )
    elif args.method == "weighted_sum":
        model = WeightedSumFusionClassifier(
            feature_dims=[768, 768],  # CLIP and DINO hidden dims
            num_classes=num_classes,
            fusion_dim=512,
        )

    print_model_info(model)

    # Train
    trainer = COMMTrainer(model, device=device, lr=args.lr, weight_decay=1e-4)

    output_path = args.output
    if output_path is None:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        output_path = f"outputs/checkpoints/{args.dataset}_comm_{args.method}.pth"

    history = trainer.fit(
        train_features=train_features,
        test_features=test_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save final model
    trainer.save(output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()

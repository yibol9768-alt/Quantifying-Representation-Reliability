"""
Training module
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.training.classifier import MultiViewClassifier, SingleViewClassifier


class FeatureDataset(Dataset):
    """Dataset for pre-extracted features"""

    def __init__(self, features_dict: Dict[str, torch.Tensor]):
        """
        Args:
            features_dict: Dict with feature tensors and labels
        """
        self.features = []
        self.labels = None

        # Get all feature keys (exclude 'labels')
        feature_keys = [k for k in features_dict.keys() if k.endswith('_features')]

        for key in sorted(feature_keys):
            self.features.append(features_dict[key])

        if 'labels' in features_dict:
            self.labels = features_dict['labels']

    def __len__(self):
        return len(self.labels) if self.labels is not None else 0

    def __getitem__(self, idx):
        if self.labels is not None:
            return tuple(f[idx] for f in self.features) + (self.labels[idx],)
        return tuple(f[idx] for f in self.features)


class Trainer:
    """Training and evaluation handler"""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """
        Args:
            model: Model to train
            device: Device to use
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Move to device
            if len(batch) == 2:  # Single view
                features, labels = batch
                features = features.to(self.device).float()
            else:  # Multi view
                *features, labels = batch
                features = [f.to(self.device).float() for f in features]

            labels = labels.to(self.device).long()

            # Forward pass
            self.optimizer.zero_grad()

            if isinstance(features, list):
                logits = self.model(*features)
            else:
                logits = self.model(features)

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

        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            # Move to device
            if len(batch) == 2:  # Single view
                features, labels = batch
                features = features.to(self.device).float()
            else:  # Multi view
                *features, labels = batch
                features = [f.to(self.device).float() for f in features]

            labels = labels.to(self.device).long()

            # Forward pass
            if isinstance(features, list):
                logits = self.model(*features)
            else:
                logits = self.model(features)

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def fit(
        self,
        train_features: Dict[str, torch.Tensor],
        test_features: Dict[str, torch.Tensor],
        epochs: int = 30,
        batch_size: int = 256,
        val_split: float = 0.2,
    ) -> Dict:
        """
        Train model

        Args:
            train_features: Training features dict
            test_features: Test features dict
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split ratio

        Returns:
            Dict with training history
        """
        # Create datasets
        train_dataset = FeatureDataset(train_features)
        test_dataset = FeatureDataset(test_features)

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
                self.save("best_model.pth")

        print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")
        return history

    def save(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    @torch.no_grad()
    def predict(
        self,
        features: Dict[str, torch.Tensor],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Make predictions

        Args:
            features: Feature dict
            batch_size: Batch size

        Returns:
            torch.Tensor: Predictions
        """
        dataset = FeatureDataset(features)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []

        for batch in loader:
            if len(batch) == 2:  # Single view
                feats, _ = batch
                feats = feats.to(self.device).float()
            else:  # Multi view
                *feats, _ = batch
                feats = [f.to(self.device).float() for f in feats]

            if isinstance(feats, list):
                logits = self.model(*feats)
            else:
                logits = self.model(feats)

            _, preds = torch.max(logits, 1)
            all_preds.append(preds.cpu())

        return torch.cat(all_preds)

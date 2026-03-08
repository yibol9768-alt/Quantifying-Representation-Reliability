"""Trainer for feature classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MAEExtractor, CLIPExtractor, DINOExtractor
from .mlp import MLPClassifier


class FeatureTrainer:
    """Trainer for feature-based classification.

    Handles feature extraction from frozen backbones and
    training of the MLP classifier.
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int = 100,
        hidden_dim: int = 512,
        device: str = "cuda:0"
    ):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize feature extractors
        self.extractors = {}
        if model_type in ["mae", "fusion"]:
            self.extractors["mae"] = MAEExtractor().to(self.device)
        if model_type in ["clip", "fusion"]:
            self.extractors["clip"] = CLIPExtractor().to(self.device)
        if model_type in ["dino", "fusion"]:
            self.extractors["dino"] = DINOExtractor().to(self.device)

        # Calculate feature dimension
        if model_type == "fusion":
            feature_dim = 768 + 512 + 768  # 2048
        else:
            feature_dims = {"mae": 768, "clip": 512, "dino": 768}
            feature_dim = feature_dims[model_type]

        # Initialize classifier
        self.classifier = MLPClassifier(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes
        ).to(self.device)

        print(f"Using device: {self.device}")
        print(f"Model type: {model_type}, Feature dim: {feature_dim}")

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Input images of shape (B, 3, H, W)

        Returns:
            Concatenated features of shape (B, D)
        """
        features = []

        for name, extractor in self.extractors.items():
            feat = extractor(images)
            features.append(feat)

        if len(features) == 1:
            return features[0]

        return torch.cat(features, dim=-1)

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Average loss and accuracy
        """
        self.classifier.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract features
            features = self.extract_features(images)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate the model.

        Args:
            test_loader: Test data loader
            criterion: Loss function

        Returns:
            Average loss and accuracy
        """
        self.classifier.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Extract features
            features = self.extract_features(images)

            # Forward pass
            outputs = self.classifier(features)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 0.01
    ) -> Dict:
        """Full training loop.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for optimizer

        Returns:
            Training history dictionary
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "best_acc": 0.0
        }

        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Evaluate
            test_loss, test_acc = self.evaluate(test_loader, criterion)

            # Update scheduler
            scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            # Save best
            if test_acc > history["best_acc"]:
                history["best_acc"] = test_acc
                self.save_checkpoint("best_model.pth")

            # Print summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
            print(f"Best Acc: {history['best_acc']*100:.2f}%")

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "classifier": self.classifier.state_dict(),
            "model_type": self.model_type
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint["classifier"])

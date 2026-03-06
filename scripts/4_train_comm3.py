"""
Train COMM3 fusion model (CLIP + DINO + MAE with multi-layer features)

Extended from original COMM (CLIP + DINO) to include MAE.

Usage:
    python scripts/4_train_comm3.py --dataset cifar100
    python scripts/4_train_comm3.py --dataset flowers102 --epochs 50

Architecture:
    - CLIP: 12 layers -> LLN fusion -> 512D
    - DINO: 6 layers (7-12) -> LLN fusion -> 512D -> MLP alignment -> 512D
    - MAE: 6 layers (7-12) -> LLN fusion -> 512D -> MLP alignment -> 512D
    - Concat: 512 + 512 + 512 = 1536D -> MLP classifier
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.training import COMM3FusionClassifier, Trainer
from src.data import DATASET_INFO
from src.utils import get_device, set_seed, print_model_info


class COMM3FeatureDataset(Dataset):
    """Dataset for COMM3 multi-layer features (CLIP + DINO + MAE)"""

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

        # Extract MAE layer features (6 layers: 7-12)
        self.mae_features = []
        for i in range(6, 12):
            key = f"mae_layer_{i}"
            if key in features_dict:
                self.mae_features.append(features_dict[key])
            else:
                raise ValueError(f"Missing MAE layer feature: {key}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Return features and label for a single sample"""
        # Collect features from all layers for each model
        clip_feats = [self.clip_features[l][idx] for l in range(12)]
        dino_feats = [self.dino_features[l][idx] for l in range(6)]
        mae_feats = [self.mae_features[l][idx] for l in range(6)]

        label = self.labels[idx]

        return {
            'clip_layer_features': clip_feats,
            'dino_layer_features': dino_feats,
            'mae_layer_features': mae_feats,
            'label': label,
        }


def collate_fn(batch):
    """Custom collate function for COMM3 features"""
    clip_layer_features = [[] for _ in range(12)]
    dino_layer_features = [[] for _ in range(6)]
    mae_layer_features = [[] for _ in range(6)]
    labels = []

    for item in batch:
        for l in range(12):
            clip_layer_features[l].append(item['clip_layer_features'][l])
        for l in range(6):
            dino_layer_features[l].append(item['dino_layer_features'][l])
            mae_layer_features[l].append(item['mae_layer_features'][l])
        labels.append(item['label'])

    # Stack features for each layer
    clip_stacked = [torch.stack(clip_layer_features[l]) for l in range(12)]
    dino_stacked = [torch.stack(dino_layer_features[l]) for l in range(6)]
    mae_stacked = [torch.stack(mae_layer_features[l]) for l in range(6)]
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'clip_layer_features': clip_stacked,
        'dino_layer_features': dino_stacked,
        'mae_layer_features': mae_stacked,
        'labels': labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train COMM3 fusion model")
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
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--clip-output-dim",
        type=int,
        default=512,
        help="CLIP fused feature dimension",
    )
    parser.add_argument(
        "--dino-output-dim",
        type=int,
        default=512,
        help="DINO fused feature dimension",
    )
    parser.add_argument(
        "--mae-output-dim",
        type=int,
        default=512,
        help="MAE fused feature dimension",
    )
    parser.add_argument(
        "--alignment-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for alignment MLPs",
    )
    parser.add_argument(
        "--classifier-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for final classifier",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    # Get dataset info
    dataset_info = DATASET_INFO.get(args.dataset, {})
    num_classes = dataset_info.get('num_classes', 100)

    print("=" * 70)
    print(f"Training COMM3 (CLIP+DINO+MAE) Fusion on {args.dataset}")
    print("=" * 70)
    print(f"Device: {device}, Classes: {num_classes}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")

    # Feature paths
    train_features_path = f"features/{args.dataset}_comm3_train.pt"
    test_features_path = f"features/{args.dataset}_comm3_test.pt"

    print(f"\nLoading features from:")
    print(f"  Train: {train_features_path}")
    print(f"  Test: {test_features_path}")

    # Load features
    train_features = torch.load(train_features_path, map_location='cpu', weights_only=False)
    test_features = torch.load(test_features_path, map_location='cpu', weights_only=False)

    # Create datasets
    train_dataset = COMM3FeatureDataset(train_features)
    test_dataset = COMM3FeatureDataset(test_features)

    # Split train into train/val
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Create model
    model = COMM3FusionClassifier(
        num_classes=num_classes,
        clip_output_dim=args.clip_output_dim,
        dino_output_dim=args.dino_output_dim,
        mae_output_dim=args.mae_output_dim,
        alignment_hidden_dim=args.alignment_hidden_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print_model_info(model)
    print(f"Total fused dimension: {model.get_fused_dim()}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    # Create checkpoint directory
    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.dataset}_comm3_fusion.pth")

    # Custom forward function for COMM3
    def forward_fn(batch, model):
        clip_feats = [f.to(device) for f in batch['clip_layer_features']]
        dino_feats = [f.to(device) for f in batch['dino_layer_features']]
        mae_feats = [f.to(device) for f in batch['mae_layer_features']]
        return model(clip_feats, dino_feats, mae_feats)

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        forward_fn=forward_fn,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Best Val Acc: {history['best_val_acc']:.2f}%")
    print(f"Final Test Acc: {history['test_acc']:.2f}%")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

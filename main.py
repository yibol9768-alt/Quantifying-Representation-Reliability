"""Main entry point for CIFAR-100 feature classification."""

import argparse
import torch

from configs.config import Config
from src.data.dataset import get_cifar100_loaders
from src.training.trainer import FeatureTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CIFAR-100 Feature Classification")

    parser.add_argument(
        "--model",
        type=str,
        default="mae",
        choices=["mae", "clip", "dino", "fusion"],
        help="Model type for feature extraction"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store CIFAR-100 data"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create config
    config = Config(
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
        data_dir=args.data_dir,
        num_workers=args.num_workers
    )

    print("=" * 60)
    print("CIFAR-100 Feature Classification")
    print("=" * 60)
    print(f"Model: {config.model_type}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.lr}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"Device: {config.device}")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWarning: CUDA not available, using CPU")

    # Load data
    print("\nLoading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        model_type=config.model_type
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize trainer
    print("\nInitializing model...")
    trainer = FeatureTrainer(
        model_type=config.model_type,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.hidden_dim,
        device=config.device
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Test Accuracy: {history['best_acc']*100:.2f}%")
    print("=" * 60)

    # Save final model
    trainer.save_checkpoint(f"{config.model_type}_final.pth")
    print(f"Model saved to: {config.model_type}_final.pth")


if __name__ == "__main__":
    main()

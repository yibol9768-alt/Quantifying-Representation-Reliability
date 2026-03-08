"""Main entry point for feature classification experiments."""

import argparse
import torch

from configs.config import Config, DATASET_CONFIGS
from src.data.dataset import get_dataloaders
from src.training.trainer import FeatureTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature Classification with MAE/CLIP/DINO"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use for training"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="mae",
        choices=["mae", "clip", "dino", "fusion"],
        help="Model type for feature extraction"
    )

    # Training settings
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

    # MLP settings
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP"
    )

    # System settings
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
        help="Directory to store datasets"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Utility
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List available datasets and exit"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # List datasets mode
    if args.list_datasets:
        print(Config.get_dataset_info())
        return

    # Create config
    config = Config(
        dataset=args.dataset,
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
    print("Feature Classification Experiment")
    print("=" * 60)
    print(f"Dataset: {config.dataset}")
    print(f"  -> {config.dataset_info}")
    print(f"Model: {config.model_type}")
    print(f"Feature Dim: {config.feature_dim}")
    print(f"Num Classes: {config.num_classes}")
    print("-" * 60)
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
    print(f"\nLoading {config.dataset} dataset...")
    try:
        train_loader, test_loader = get_dataloaders(
            dataset_name=config.dataset,
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            model_type=config.model_type
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize trainer
    print("\nInitializing model...")
    trainer = FeatureTrainer(
        model_type=config.model_type,
        num_classes=config.num_classes,
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
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model_type}")
    print(f"Best Test Accuracy: {history['best_acc']*100:.2f}%")
    print("=" * 60)

    # Save final model
    model_name = f"{config.dataset}_{config.model_type}_final.pth"
    trainer.save_checkpoint(model_name)
    print(f"Model saved to: {model_name}")


if __name__ == "__main__":
    main()

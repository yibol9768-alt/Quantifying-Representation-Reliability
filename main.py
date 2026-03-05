"""
Main entry point for the multi-view fusion project

Usage:
    # Run complete pipeline on Stanford Cars
    python main.py --mode full --dataset stanford_cars --models clip dino mae

    # Run on CIFAR-10
    python main.py --mode full --dataset cifar10 --models clip dino

    # Extract features only
    python main.py --mode extract --dataset cifar10 --models clip dino mae

    # Train only (requires pre-extracted features)
    python main.py --mode train --dataset cifar10 --models clip dino mae

    # Evaluate only
    python main.py --mode evaluate --checkpoint outputs/checkpoints/cifar10_clip_dino_mae_fusion.pth
"""
import argparse
import os
import sys
import subprocess


def run_command(cmd, description):
    """Run a shell command with description"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        sys.exit(1)


def extract_single_models(dataset, models, split):
    """Extract features for single models"""
    for model in models:
        run_command(
            ["python", "scripts/1_extract_single.py", "--model", model, "--dataset", dataset, "--split", split],
            f"Extracting {model} features ({dataset} - {split})",
        )


def extract_fusion_models(dataset, models, split):
    """Extract features for fusion"""
    run_command(
        ["python", "scripts/2_extract_multi.py", "--models"] + models + ["--dataset", dataset, "--split", split],
        f"Extracting {' + '.join(models)} features ({dataset} - {split})",
    )


def train_single_model(dataset, model):
    """Train single model classifier"""
    run_command(
        ["python", "scripts/3_train_single.py", "--model", model, "--dataset", dataset],
        f"Training {model} single model on {dataset}",
    )


def train_fusion_model(dataset, models):
    """Train fusion model"""
    run_command(
        ["python", "scripts/4_train_fusion.py", "--models"] + models + ["--dataset", dataset],
        f"Training {' + '.join(models)} fusion model on {dataset}",
    )


def evaluate_model(checkpoint, models, dataset):
    """Evaluate model"""
    if len(models) == 1:
        run_command(
            ["python", "scripts/5_evaluate.py", "--checkpoint", checkpoint, "--model", models[0], "--dataset", dataset, "--single"],
            f"Evaluating {models[0]} model on {dataset}",
        )
    else:
        run_command(
            ["python", "scripts/5_evaluate.py", "--checkpoint", checkpoint, "--models"] + models + ["--dataset", dataset],
            f"Evaluating {' + '.join(models)} fusion model on {dataset}",
        )


def list_available_datasets():
    """List all available datasets"""
    from src.data import DATASET_INFO

    print("\nAvailable datasets:")
    print("-" * 60)
    for name, info in DATASET_INFO.items():
        print(f"  {name:20} {info['num_classes']:3} classes  {info['type']:15} {info['description']}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view fusion project main entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full --dataset stanford_cars --models clip dino mae
  python main.py --mode full --dataset cifar10 --models clip dino
  python main.py --mode extract --dataset cifar10 --models clip dino
  python main.py --mode train --dataset cifar10 --models clip dino
  python main.py --mode list
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["extract", "train", "evaluate", "full", "list"],
        help="Operation mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stanford_cars",
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
        "--models",
        type=str,
        nargs="+",
        default=["clip", "dino", "mae"],
        choices=["clip", "dino", "mae"],
        help="Models to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (for evaluate mode)",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only run single model experiments",
    )
    parser.add_argument(
        "--fusion-only",
        action="store_true",
        help="Only run fusion experiments",
    )

    args = parser.parse_args()

    # List mode
    if args.mode == "list" or args.mode is None:
        list_available_datasets()
        return

    # Full pipeline
    if args.mode == "full":
        print("\n" + "=" * 60)
        print(f"MULTI-VIEW FUSION: FULL PIPELINE")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Models: {args.models}")

        # Extract features
        print("\n--- Step 1: Feature Extraction ---")
        for split in ["train", "test"]:
            extract_single_models(args.dataset, args.models, split)

        # Train single models
        if not args.fusion_only:
            print("\n--- Step 2: Single Model Training ---")
            for model in args.models:
                train_single_model(args.dataset, model)

        # Train fusion model
        if not args.single_only and len(args.models) > 1:
            print("\n--- Step 3: Fusion Model Training ---")
            train_fusion_model(args.dataset, args.models)

        # Evaluate all
        print("\n--- Step 4: Evaluation ---")
        os.makedirs("outputs/checkpoints", exist_ok=True)

        if not args.fusion_only:
            for model in args.models:
                checkpoint = f"outputs/checkpoints/{args.dataset}_{model}_single.pth"
                if os.path.exists(checkpoint):
                    evaluate_model(checkpoint, [model], args.dataset)

        if not args.single_only and len(args.models) > 1:
            model_str = "_".join(args.models)
            checkpoint = f"outputs/checkpoints/{args.dataset}_{model_str}_fusion.pth"
            if os.path.exists(checkpoint):
                evaluate_model(checkpoint, args.models, args.dataset)

    # Extract only
    elif args.mode == "extract":
        for split in ["train", "test"]:
            extract_single_models(args.dataset, args.models, split)

    # Train only
    elif args.mode == "train":
        if not args.fusion_only:
            for model in args.models:
                train_single_model(args.dataset, model)

        if not args.single_only and len(args.models) > 1:
            train_fusion_model(args.dataset, args.models)

    # Evaluate only
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            model_str = "_".join(args.models)
            args.checkpoint = f"outputs/checkpoints/{args.dataset}_{model_str}_fusion.pth"

        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        evaluate_model(args.checkpoint, args.models, args.dataset)

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Main entry point for the multi-view fusion project

Usage:
    # Complete pipeline
    python main.py --mode full --dataset cifar10 --models clip dino
    
    # Extract features only
    python main.py --mode extract --dataset cifar10 --models clip dino
    
    # Train only
    python main.py --mode train --dataset cifar10 --models clip dino
    
    # Evaluate only
    python main.py --mode evaluate --dataset cifar10 --models clip dino
"""
import argparse
import os
import sys
import subprocess


def run_cmd(cmd, desc):
    """Run shell command"""
    print(f"\n{'='*60}")
    print(f" {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {desc} failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Multi-view fusion pipeline")
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["extract", "train", "evaluate", "full"],
                        help="Running mode")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "flowers102", "pets", "stanford_cars", "food101"],
                        help="Dataset name")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=["clip", "dino", "mae"],
                        help="Model types")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    model_str = "_".join(args.models)
    
    if args.mode in ["extract", "full"]:
        # Extract train features
        run_cmd(
            ["python", "scripts/extract.py", "--models"] + args.models + 
            ["--dataset", args.dataset, "--split", "train"],
            f"Extracting features (train)"
        )
        # Extract test features
        run_cmd(
            ["python", "scripts/extract.py", "--models"] + args.models + 
            ["--dataset", args.dataset, "--split", "test"],
            f"Extracting features (test)"
        )
    
    if args.mode in ["train", "full"]:
        run_cmd(
            ["python", "scripts/train.py", "--models"] + args.models +
            ["--dataset", args.dataset, "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size), "--lr", str(args.lr)],
            f"Training model"
        )
    
    if args.mode in ["evaluate", "full"]:
        suffix = "single" if len(args.models) == 1 else "fusion"
        checkpoint = f"outputs/checkpoints/{args.dataset}_{model_str}_{suffix}.pth"
        run_cmd(
            ["python", "scripts/evaluate.py", "--checkpoint", checkpoint,
             "--models"] + args.models + ["--dataset", args.dataset],
            f"Evaluating model"
        )
    
    print(f"\n{'='*60}")
    print(" Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

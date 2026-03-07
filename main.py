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
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "dali"],
        help="Feature extraction backend",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="concat",
        choices=["concat", "mmvit", "mmvit3", "comm", "comm3"],
        help="Fusion method for multi-model training",
    )
    
    args = parser.parse_args()

    method_requirements = {
        "comm": ["clip", "dino"],
        "mmvit": ["clip", "dino"],
        "comm3": ["clip", "dino", "mae"],
        "mmvit3": ["clip", "dino", "mae"],
    }
    if args.method in method_requirements and args.models != method_requirements[args.method]:
        required = " ".join(method_requirements[args.method])
        print(f"Error: --method {args.method} requires --models {required}")
        sys.exit(1)
    
    model_str = "_".join(args.models)
    
    if args.mode in ["extract", "full"]:
        if args.method in {"comm", "comm3", "mmvit", "mmvit3"} and len(args.models) > 1:
            train_cmd = ["python", "scripts/extract.py", "--method", args.method, "--dataset", args.dataset, "--split", "train", "--backend", args.backend]
            test_cmd = ["python", "scripts/extract.py", "--method", args.method, "--dataset", args.dataset, "--split", "test", "--backend", args.backend]
        elif len(args.models) == 1:
            train_cmd = ["python", "scripts/extract.py", "--model", args.models[0], "--dataset", args.dataset, "--split", "train", "--backend", args.backend]
            test_cmd = ["python", "scripts/extract.py", "--model", args.models[0], "--dataset", args.dataset, "--split", "test", "--backend", args.backend]
        else:
            train_cmd = ["python", "scripts/extract.py", "--models"] + args.models + ["--dataset", args.dataset, "--split", "train", "--backend", args.backend]
            test_cmd = ["python", "scripts/extract.py", "--models"] + args.models + ["--dataset", args.dataset, "--split", "test", "--backend", args.backend]

        run_cmd(train_cmd, "Extracting features (train)")
        run_cmd(test_cmd, "Extracting features (test)")
    
    if args.mode in ["train", "full"]:
        run_cmd(
            ["python", "scripts/train.py", "--models"] + args.models +
            ["--dataset", args.dataset, "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size), "--lr", str(args.lr),
             "--method", args.method],
            f"Training model"
        )
    
    if args.mode in ["evaluate", "full"]:
        suffix = "single" if len(args.models) == 1 else args.method
        checkpoint = f"outputs/checkpoints/{args.dataset}_{model_str}_{suffix}.pth"
        run_cmd(
            ["python", "scripts/evaluate.py", "--checkpoint", checkpoint,
             "--models"] + args.models + ["--dataset", args.dataset, "--method", args.method],
            f"Evaluating model"
        )
    
    print(f"\n{'='*60}")
    print(" Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
Main entry point

Usage:
    # Complete pipeline
    python main.py --mode full --dataset cifar10 --method clip
    python main.py --mode full --dataset cifar10 --method concat --models clip dino
    python main.py --mode full --dataset cifar10 --method comm
    python main.py --mode full --dataset cifar10 --method comm3
    
    # Individual steps
    python main.py --mode extract --dataset cifar10 --method comm
    python main.py --mode train --dataset cifar10 --method comm
    python main.py --mode evaluate --dataset cifar10 --method comm
"""
import argparse
import os
import sys
import subprocess


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f" {desc}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-view fusion pipeline")
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["extract", "train", "evaluate", "full"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["single", "concat", "comm", "comm3"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    parser.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    # Build commands based on method
    if args.method == "single":
        if not args.model:
            print("Error: --model required for single method")
            sys.exit(1)
        extract_arg = ["--model", args.model]
        train_arg = ["--model", args.model]
        ckpt_name = f"{args.dataset}_{args.model}_single.pth"
        
    elif args.method == "concat":
        if not args.models:
            print("Error: --models required for concat method")
            sys.exit(1)
        extract_arg = ["--models"] + args.models
        train_arg = ["--models"] + args.models
        ckpt_name = f"{args.dataset}_{'_'.join(args.models)}_concat.pth"
        
    elif args.method == "comm":
        extract_arg = ["--method", "comm"]
        train_arg = ["--method", "comm"]
        ckpt_name = f"{args.dataset}_comm.pth"
        
    elif args.method == "comm3":
        extract_arg = ["--method", "comm3"]
        train_arg = ["--method", "comm3"]
        ckpt_name = f"{args.dataset}_comm3.pth"
    
    # Execute
    if args.mode in ["extract", "full"]:
        for split in ["train", "test"]:
            run_cmd(
                ["python", "scripts/extract.py"] + extract_arg +
                ["--dataset", args.dataset, "--split", split],
                f"Extracting features ({split})"
            )
    
    if args.mode in ["train", "full"]:
        run_cmd(
            ["python", "scripts/train.py"] + train_arg +
            ["--dataset", args.dataset, "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size), "--lr", str(args.lr)],
            "Training model"
        )
    
    if args.mode in ["evaluate", "full"]:
        run_cmd(
            ["python", "scripts/evaluate.py",
             "--checkpoint", f"outputs/checkpoints/{ckpt_name}",
             "--method", args.method,
             "--dataset", args.dataset] +
            (["--model", args.model] if args.model else []) +
            (["--models"] + args.models if args.models else []),
            "Evaluating model"
        )
    
    print(f"\n{'='*60}")
    print(" Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

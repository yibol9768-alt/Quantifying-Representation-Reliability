"""
Main entry point for the multi-view fusion project

Usage:
    # Run complete pipeline (extract + train + evaluate)
    python main.py --mode full --models clip dino mae

    # Extract features only
    python main.py --mode extract --models clip dino mae

    # Train only (requires pre-extracted features)
    python main.py --mode train --models clip dino mae

    # Evaluate only
    python main.py --mode evaluate --checkpoint outputs/checkpoints/clip_dino_mae_fusion.pth
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


def extract_single_models(models, split):
    """Extract features for single models"""
    for model in models:
        run_command(
            ["python", "scripts/1_extract_single.py", "--model", model, "--split", split],
            f"Extracting {model} features ({split})",
        )


def extract_fusion_models(models, split):
    """Extract features for fusion"""
    model_str = " ".join(models)
    run_command(
        ["python", "scripts/2_extract_multi.py", "--models"] + models + ["--split", split],
        f"Extracting {' + '.join(models)} features ({split})",
    )


def train_single_model(model):
    """Train single model classifier"""
    run_command(
        ["python", "scripts/3_train_single.py", "--model", model],
        f"Training {model} single model",
    )


def train_fusion_model(models):
    """Train fusion model"""
    run_command(
        ["python", "scripts/4_train_fusion.py", "--models"] + models,
        f"Training {' + '.join(models)} fusion model",
    )


def evaluate_model(checkpoint, models):
    """Evaluate model"""
    if len(models) == 1:
        run_command(
            ["python", "scripts/5_evaluate.py", "--checkpoint", checkpoint, "--single"],
            f"Evaluating {models[0]} model",
        )
    else:
        run_command(
            ["python", "scripts/5_evaluate.py", "--checkpoint", checkpoint, "--models"] + models,
            f"Evaluating {' + '.join(models)} fusion model",
        )


def main():
    parser = argparse.ArgumentParser(description="Multi-view fusion project main entry")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["extract", "train", "evaluate", "full"],
        help="Operation mode",
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

    # Full pipeline
    if args.mode == "full":
        print("\n" + "=" * 60)
        print("MULTI-VIEW FUSION: FULL PIPELINE")
        print("=" * 60)
        print(f"Models: {args.models}")

        # Extract features
        print("\n--- Step 1: Feature Extraction ---")
        for split in ["train", "test"]:
            extract_single_models(args.models, split)

        # Train single models
        if not args.fusion_only:
            print("\n--- Step 2: Single Model Training ---")
            for model in args.models:
                train_single_model(model)

        # Train fusion model
        if not args.single_only:
            print("\n--- Step 3: Fusion Model Training ---")
            train_fusion_model(args.models)

        # Evaluate all
        print("\n--- Step 4: Evaluation ---")
        from configs.config import Config
        config = Config()
        model_str = "_".join(args.models)

        if not args.fusion_only:
            for model in args.models:
                checkpoint = os.path.join(config.checkpoint_dir, f"{model}_single.pth")
                if os.path.exists(checkpoint):
                    evaluate_model(checkpoint, [model])

        if not args.single_only:
            checkpoint = os.path.join(config.checkpoint_dir, f"{model_str}_fusion.pth")
            if os.path.exists(checkpoint):
                evaluate_model(checkpoint, args.models)

    # Extract only
    elif args.mode == "extract":
        for split in ["train", "test"]:
            extract_single_models(args.models, split)

    # Train only
    elif args.mode == "train":
        if not args.fusion_only:
            for model in args.models:
                train_single_model(model)

        if not args.single_only and len(args.models) > 1:
            train_fusion_model(args.models)

    # Evaluate only
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            from configs.config import Config
            config = Config()
            model_str = "_".join(args.models)
            args.checkpoint = os.path.join(config.checkpoint_dir, f"{model_str}_fusion.pth")

        evaluate_model(args.checkpoint, args.models)

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

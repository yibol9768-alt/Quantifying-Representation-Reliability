#!/usr/bin/env python3
"""
Download pretrained model weights.

This script downloads weights for all supported models.
Models are cached in HuggingFace cache or torch hub cache by default.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models clip_vit_b16 dinov2_vit_l14
    python scripts/download_models.py --family clip dino
    python scripts/download_models.py --cache_dir ./model_cache
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import list_available_models, get_models_by_family


def download_clip_models(cache_dir: str = None):
    """Download CLIP models from HuggingFace."""
    print("\n" + "=" * 50)
    print("Downloading CLIP models...")
    print("=" * 50)

    from transformers import CLIPModel, CLIPProcessor

    models = {
        "clip_vit_b16": "openai/clip-vit-base-patch16",
        "clip_vit_b32": "openai/clip-vit-base-patch32",
        "clip_vit_l14": "openai/clip-vit-large-patch14",
        "clip_vit_l14_336": "openai/clip-vit-large-patch14-336",
    }

    for name, checkpoint in models.items():
        print(f"\n  Downloading {name} ({checkpoint})...")
        try:
            CLIPModel.from_pretrained(checkpoint, cache_dir=cache_dir)
            CLIPProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
            print(f"  ✓ {name} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")


def download_dino_models(cache_dir: str = None):
    """Download DINO models from HuggingFace."""
    print("\n" + "=" * 50)
    print("Downloading DINO models...")
    print("=" * 50)

    from transformers import AutoModel

    models = {
        "dino_vit_b16": "facebook/dino-vitb16",
        "dino_vit_b8": "facebook/dino-vitb8",
    }

    for name, checkpoint in models.items():
        print(f"\n  Downloading {name} ({checkpoint})...")
        try:
            AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir)
            print(f"  ✓ {name} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")


def download_dinov2_models(cache_dir: str = None):
    """Download DINOv2 models from torch hub."""
    print("\n" + "=" * 50)
    print("Downloading DINOv2 models...")
    print("=" * 50)

    import torch

    models = {
        "dinov2_vit_s14": "dinov2_small",
        "dinov2_vit_b14": "dinov2_base",
        "dinov2_vit_l14": "dinov2_large",
        "dinov2_vit_g14": "dinov2_giant",
    }

    for name, model_id in models.items():
        print(f"\n  Downloading {name}...")
        try:
            torch.hub.load("facebookresearch/dinov2", model_id, pretrained=True)
            print(f"  ✓ {name} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")


def download_mae_models(cache_dir: str = None):
    """Download MAE models from HuggingFace."""
    print("\n" + "=" * 50)
    print("Downloading MAE models...")
    print("=" * 50)

    from transformers import ViTMAEModel

    models = {
        "mae_vit_b16": "facebook/vit-mae-base",
        "mae_vit_l16": "facebook/vit-mae-large",
        "mae_vit_h14": "facebook/vit-mae-huge",
    }

    for name, checkpoint in models.items():
        print(f"\n  Downloading {name} ({checkpoint})...")
        try:
            ViTMAEModel.from_pretrained(checkpoint, cache_dir=cache_dir)
            print(f"  ✓ {name} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")


def download_ibot_models(cache_dir: str = None):
    """Download iBOT models (manual download required)."""
    print("\n" + "=" * 50)
    print("iBOT models require manual download")
    print("=" * 50)

    urls = {
        "ibot_vit_b16": "https://lf3-static.bytednsdoc.com/obj/eden-cn/hjeh7pldnulm/iBOT/checkpoint_teacher.pth",
        "ibot_vit_l16": "https://lf3-static.bytednsdoc.com/obj/eden-cn/hjeh7pldnulm/iBOT/checkpoint_teacher_large.pth",
    }

    print("\niBOT weights must be downloaded manually:")
    for name, url in urls.items():
        print(f"  {name}: {url}")
    print("\nAfter downloading, place the files and use:")
    print("  python scripts/extract_features.py --model ibot_vit_b16 --weights_path /path/to/checkpoint.pth")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to download (default: all)",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        choices=["clip", "dino", "dinov2", "mae", "ibot"],
        help="Download all models from specific families",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for family in ["clip", "dino", "dinov2", "mae", "ibot"]:
            models = get_models_by_family(family)
            print(f"\n  {family.upper()}:")
            for m in models:
                print(f"    - {m}")
        return

    # Create cache directory if specified
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    # Determine which families to download
    if args.family:
        families = args.family
    elif args.models:
        # Infer families from model names
        families = set()
        for model in args.models:
            for family in ["clip", "dino", "dinov2", "mae", "ibot"]:
                if model.startswith(family):
                    families.add(family)
    else:
        families = ["clip", "dino", "dinov2", "mae", "ibot"]

    print("=" * 50)
    print("Model Download Script")
    print("=" * 50)
    print(f"Families to download: {families}")
    if args.cache_dir:
        print(f"Cache directory: {args.cache_dir}")

    # Download models
    if "clip" in families:
        download_clip_models(args.cache_dir)

    if "dino" in families:
        download_dino_models(args.cache_dir)

    if "dinov2" in families:
        download_dinov2_models(args.cache_dir)

    if "mae" in families:
        download_mae_models(args.cache_dir)

    if "ibot" in families:
        download_ibot_models(args.cache_dir)

    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

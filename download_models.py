#!/usr/bin/env python3
"""
Download all required pre-trained models with proxy acceleration

This script downloads:
1. OpenAI CLIP (ViT-B/32)
2. Facebook DINO (dino_vitb16)
3. Facebook MAE (vit-mae-base)
"""

import os
import sys
import torch
import clip
from PIL import Image

def setup_environment():
    """Setup environment variables for faster download in China"""
    # Hugging Face mirror and cache
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"

    # PyTorch hub cache
    os.environ["TORCH_HUB_DIR"] = "/root/autodl-tmp/torchhub"

    # Create cache directories
    os.makedirs("/root/autodl-tmp/huggingface", exist_ok=True)
    os.makedirs("/root/autodl-tmp/torchhub", exist_ok=True)

    # Set proxies if provided
    # You can set http_proxy and https_proxy environment variables
    # export http_proxy=http://your-proxy:port
    # export https_proxy=http://your-proxy:port

    print("Environment setup complete")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'default')}")
    print(f"TORCH_HUB_DIR: {os.environ.get('TORCH_HUB_DIR', 'default')}")

def download_clip(device="cuda"):
    """Download OpenAI CLIP model"""
    print("\n" + "="*60)
    print("Downloading CLIP model (ViT-B/32)...")
    print("="*60)

    try:
        # CLIP will cache in ~/.cache/clip by default, we can't easily change this
        # but we can ensure models are downloaded to /root/autodl-tmp
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ CLIP model downloaded successfully!")
        print(f"  Model: ViT-B/32")
        print(f"  Feature dim: 512")
        print(f"  Cache location: ~/.cache/clip (can be symlinked to /root/autodl-tmp)")
        return model
    except Exception as e:
        print(f"✗ CLIP download failed: {e}")
        return None

def download_dino(device="cuda"):
    """Download Facebook DINO model"""
    print("\n" + "="*60)
    print("Downloading DINO model (dino_vitb16)...")
    print("="*60)

    try:
        # Set torch hub cache dir to /root/autodl-tmp
        torch_hub_dir = "/root/autodl-tmp/torchhub"
        os.makedirs(torch_hub_dir, exist_ok=True)
        os.environ["TORCH_HUB_DIR"] = torch_hub_dir

        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
        model = model.to(device)
        model.eval()
        print("✓ DINO model downloaded successfully!")
        print(f"  Model: dino_vitb16")
        print(f"  Feature dim: 768")
        print(f"  Storage: {torch_hub_dir}")
        return model
    except Exception as e:
        print(f"✗ DINO download failed: {e}")
        return None

def download_mae(device="cuda"):
    """Download Facebook MAE model"""
    print("\n" + "="*60)
    print("Downloading MAE model (vit-mae-base)...")
    print("="*60)

    try:
        from transformers import ViTMAEModel, AutoImageProcessor

        # Set cache dir for transformers to /root/autodl-tmp
        cache_dir = "/root/autodl-tmp/transformers"
        os.makedirs(cache_dir, exist_ok=True)

        # Also set HF_HOME environment variable
        os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)

        processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", cache_dir=cache_dir)
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", cache_dir=cache_dir)
        model = model.to(device)
        model.eval()

        print("✓ MAE model downloaded successfully!")
        print(f"  Model: facebook/vit-mae-base")
        print(f"  Feature dim: 768")
        print(f"  Storage: {cache_dir}")
        return model
    except Exception as e:
        print(f"✗ MAE download failed: {e}")
        return None

def verify_models():
    """Verify all models are working"""
    print("\n" + "="*60)
    print("Verifying models...")
    print("="*60)

    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')

    try:
        # Test CLIP
        import clip
        model, _ = clip.load("ViT-B/32", device="cuda")
        print("✓ CLIP verified")
    except Exception as e:
        print(f"✗ CLIP verification failed: {e}")

    try:
        # Test DINO
        import torch
        from torchvision import transforms
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print("✓ DINO verified")
    except Exception as e:
        print(f"✗ DINO verification failed: {e}")

    try:
        # Test MAE
        from transformers import ViTMAEModel, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        print("✓ MAE verified")
    except Exception as e:
        print(f"✗ MAE verification failed: {e}")

def main():
    """Main download function"""
    print("="*60)
    print("Model Download Script for Quantifying Representation Reliability")
    print("="*60)

    # Setup environment
    setup_environment()

    # Create cache directories in /root/autodl-tmp
    os.makedirs("/root/autodl-tmp/huggingface", exist_ok=True)
    os.makedirs("/root/autodl-tmp/torchhub", exist_ok=True)
    os.makedirs("/root/autodl-tmp/transformers", exist_ok=True)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if device == "cpu":
        print("Warning: CUDA not available, using CPU. Downloads will still work.")

    # Download models
    results = {}

    print("\n" + "="*60)
    print("Starting model downloads...")
    print("="*60)

    results["clip"] = download_clip(device) is not None
    results["dino"] = download_dino(device) is not None
    results["mae"] = download_mae(device) is not None

    # Verify models
    verify_models()

    # Create symlinks for CLIP cache
    print("\n" + "="*60)
    print("Setting up model cache symlinks...")
    print("="*60)
    try:
        clip_cache_src = os.path.expanduser("~/.cache/clip")
        if os.path.exists(clip_cache_src):
            clip_cache_dst = "/root/autodl-tmp/clip"
            if not os.path.exists(clip_cache_dst):
                os.system(f"cp -r {clip_cache_src} {clip_cache_dst}")
                print(f"✓ CLIP cache copied to: {clip_cache_dst}")
            else:
                print(f"✓ CLIP cache already exists at: {clip_cache_dst}")
    except Exception as e:
        print(f"Note: Could not setup CLIP cache symlink: {e}")

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{model_name.upper()}: {status}")

    # Exit with appropriate code
    if all(results.values()):
        print("\n✓ All models downloaded successfully!")
        return 0
    else:
        print("\n✗ Some models failed to download. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

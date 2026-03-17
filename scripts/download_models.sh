#!/bin/bash
# Download pretrained models from HuggingFace.
#
# Usage:
#   bash scripts/download_models.sh ./models              # all 25 models
#   bash scripts/download_models.sh ./models --original    # original 6
#   bash scripts/download_models.sh ./models --recommended # balanced 15
#   bash scripts/download_models.sh ./models --all         # all 25
#
# For China mainland users, set mirror first:
#   export HF_ENDPOINT=https://hf-mirror.com

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 MODEL_DIR [--original|--recommended|--all]}"
MODE="${2:---all}"

mkdir -p "$MODEL_DIR"

download_model() {
    local hf_name=$1
    local local_path=$2
    local target="$MODEL_DIR/$local_path"

    if [ -d "$target" ] && [ -f "$target/config.json" ]; then
        echo "[SKIP] $hf_name -> $target (already exists)"
        return 0
    fi

    echo "[DOWN] $hf_name -> $target"
    huggingface-cli download "$hf_name" --local-dir "$target" || {
        echo "[FAIL] $hf_name"
        return 1
    }
}

# ── Original 6 models ───────────────────────────────────────────────
download_original() {
    echo "=== Downloading original 6 models ==="
    download_model "openai/clip-vit-base-patch16"              "clip-vit-base-patch16"
    download_model "facebook/dinov2-base"                      "dinov2-base"
    download_model "facebook/vit-mae-base"                     "vit-mae-base"
    download_model "google/siglip-base-patch16-224"            "siglip-base-patch16-224"
    download_model "facebook/convnext-base-224"                "convnext-base"
    download_model "facebook/data2vec-vision-base"             "data2vec-vision-base"
}

# ── Recommended 15 models (original 6 + 9 diverse additions) ────────
download_recommended() {
    download_original

    echo ""
    echo "=== Downloading 9 additional recommended models ==="
    # Large variants (strong models)
    download_model "facebook/dinov2-large"                     "dinov2-large"
    download_model "openai/clip-vit-large-patch14"             "clip-vit-large-patch14"
    download_model "facebook/vit-mae-large"                    "vit-mae-large"
    # Different architectures
    download_model "google/vit-base-patch16-224"               "vit-base-patch16"
    download_model "microsoft/beit-base-patch16-224-pt22k"     "beit-base"
    download_model "microsoft/swin-base-patch4-window7-224"    "swin-base"
    # Small/weak models (for selection contrast)
    download_model "facebook/deit-small-patch16-224"           "deit-small-patch16"
    download_model "facebook/dinov2-small"                     "dinov2-small"
    # Classic CNN baseline
    download_model "microsoft/resnet-50"                       "resnet-50"
}

# ── All 24 models ───────────────────────────────────────────────────
download_all() {
    download_recommended

    echo ""
    echo "=== Downloading remaining 10 models ==="
    # ViT large
    download_model "google/vit-large-patch16-224"              "vit-large-patch16"
    # DeiT base
    download_model "facebook/deit-base-patch16-224"            "deit-base-patch16"
    # Swin tiny
    download_model "microsoft/swin-tiny-patch4-window7-224"    "swin-tiny"
    # BEiT large
    download_model "microsoft/beit-large-patch16-224-pt22k"    "beit-large"
    # CLIP variants
    download_model "openai/clip-vit-base-patch32"              "clip-vit-base-patch32"
    download_model "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"    "openclip-vit-b32"
    # ConvNeXt variants
    download_model "facebook/convnext-tiny-224"                "convnext-tiny"
    download_model "facebook/convnext-large-224"               "convnext-large"
    # ResNet 101
    download_model "microsoft/resnet-101"                      "resnet-101"
    # MAE large (may already be downloaded in recommended)
    # (covered above)
}

echo "============================================"
echo "Model Download"
echo "  Target directory: $MODEL_DIR"
echo "  Mode: $MODE"
echo "============================================"
echo ""

case "$MODE" in
    --original)    download_original ;;
    --recommended) download_recommended ;;
    --all)         download_all ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 MODEL_DIR [--original|--recommended|--all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Download complete!"
echo ""
echo "Verify with:"
echo "  python -c \"from src.models.extractor import FeatureExtractor; import os"
echo "  for n,c in FeatureExtractor.MODEL_PATHS.items():"
echo "    s='OK' if os.path.isdir(f'$MODEL_DIR/{c[\"path\"]}') else 'MISSING'"
echo "    print(f'  [{s:7s}] {n:16s}')\""
echo "============================================"

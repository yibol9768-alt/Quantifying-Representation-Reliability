#!/bin/bash
# Single-Model Baselines: Run all models individually on all datasets
#
# This provides the Relevance(m, T) scores needed for
# the Diversity × Relevance joint model selection framework.
#
# Usage:
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_single_model.sh
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_single_model.sh --quick
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_single_model.sh --original

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"

# ── Model pools ──────────────────────────────────────────────────────
# Original 6 models (backward compatible)
MODELS_ORIGINAL="clip dino mae siglip convnext data2vec"

# Full 25-model zoo
MODELS_ALL="
  vit vit_large
  deit_small deit_base
  swin_tiny swin
  beit beit_large
  data2vec
  mae mae_large
  dinov2_small dino dinov2_large
  clip_base32 clip clip_large openclip
  siglip
  convnext_tiny convnext convnext_large
  resnet50 resnet101
"

# Quick subset: representative models from each family
MODELS_QUICK="clip dino mae siglip convnext resnet50 vit deit_small dinov2_large clip_large"

# ── Datasets ─────────────────────────────────────────────────────────
DATASETS_FULL="stl10 pets eurosat dtd gtsrb svhn country211"
DATASETS_QUICK="stl10 gtsrb svhn"

run_single() {
    local dataset=$1
    local model=$2
    local fewshot_flag=$3

    echo ""
    echo "============================================================"
    echo "  Single model: $model on $dataset"
    echo "============================================================"

    $BASE_CMD --dataset "$dataset" --model "$model" \
        $fewshot_flag \
        || echo "FAILED: $model on $dataset"
}

quick_mode() {
    echo "=== Quick: Single Model Baselines (3 datasets × ${#} models) ==="
    for ds in $DATASETS_QUICK; do
        for model in $MODELS_QUICK; do
            run_single "$ds" "$model" "--disable_fewshot"
        done
    done
}

original_mode() {
    echo "=== Original 6 models (7 datasets × 6 models = 42 runs) ==="
    for ds in $DATASETS_FULL; do
        for model in $MODELS_ORIGINAL; do
            run_single "$ds" "$model" "--disable_fewshot"
        done
    done
}

full_mode() {
    local model_count=$(echo $MODELS_ALL | wc -w | tr -d ' ')
    local ds_count=$(echo $DATASETS_FULL | wc -w | tr -d ' ')
    echo "=== Full: Single Model Baselines ($ds_count datasets × $model_count models) ==="
    for ds in $DATASETS_FULL; do
        for model in $MODELS_ALL; do
            run_single "$ds" "$model" "--disable_fewshot"
        done
    done
}

case "${1:---full}" in
    --quick)    quick_mode ;;
    --original) original_mode ;;
    --full)     full_mode ;;
    *)
        echo "Usage: $0 [--quick|--original|--full]"
        echo "  --quick:    3 datasets × 10 representative models"
        echo "  --original: 7 datasets × 6 original models (backward compatible)"
        echo "  --full:     7 datasets × 24 models (full model zoo)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Single model baselines complete!"
echo "  Results in: $STORAGE_DIR/results/"
echo "  Next: collect results and feed into joint_selection.py"
echo "============================================================"

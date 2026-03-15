#!/bin/bash
# Single-Model Baselines: Run all 6 models individually on all datasets
#
# This provides the Relevance(m, T) scores needed for
# the Diversity × Relevance joint model selection framework.
#
# Usage:
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_single_model.sh
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_single_model.sh --quick

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"

MODELS="clip dino mae siglip convnext data2vec"
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
    echo "=== Quick: Single Model Baselines (3 datasets × 6 models = 18 runs) ==="
    for ds in $DATASETS_QUICK; do
        for model in $MODELS; do
            run_single "$ds" "$model" "--disable_fewshot"
        done
    done
}

full_mode() {
    echo "=== Full: Single Model Baselines (7 datasets × 6 models = 42 runs) ==="
    for ds in $DATASETS_FULL; do
        for model in $MODELS; do
            run_single "$ds" "$model" "--disable_fewshot"
        done
    done
}

case "${1:---full}" in
    --quick) quick_mode ;;
    --full)  full_mode ;;
    *)
        echo "Usage: $0 [--quick|--full]"
        echo "  --quick: 3 datasets × 6 models = 18 runs"
        echo "  --full:  7 datasets × 6 models = 42 runs"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Single model baselines complete!"
echo "  Results in: $STORAGE_DIR/results/"
echo "  Next: collect results and feed into joint_selection.py"
echo "============================================================"

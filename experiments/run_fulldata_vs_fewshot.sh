#!/bin/bash
# Full-data vs Few-shot 对比实验
# 验证假设：few-shot 是复杂路由方法表现不佳的根本原因
#
# Usage:
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_fulldata_vs_fewshot.sh
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_fulldata_vs_fewshot.sh --quick

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"
MODELS_6="clip,dino,mae,siglip,convnext,data2vec"

DATASETS_FULL="stl10 pets eurosat dtd gtsrb svhn country211"
DATASETS_QUICK="stl10 gtsrb svhn"

METHODS="concat gated moe_router"

run() {
    local dataset=$1
    local method=$2
    local fewshot_flag=$3  # "" or "--disable_fewshot"
    local tag
    if [ -z "$fewshot_flag" ]; then tag="10shot"; else tag="fulldata"; fi

    echo ""
    echo "============================================================"
    echo "  $dataset | $method | $tag"
    echo "============================================================"

    $BASE_CMD --dataset "$dataset" --model fusion \
        --fusion_method "$method" --fusion_models "$MODELS_6" \
        $fewshot_flag \
        || echo "FAILED: $dataset $method $tag"
}

quick_mode() {
    echo "=== Quick: Full-data vs Few-shot ==="
    for ds in $DATASETS_QUICK; do
        for method in $METHODS; do
            run "$ds" "$method" ""                  # 10-shot
            run "$ds" "$method" "--disable_fewshot"  # full data
        done
    done
}

full_mode() {
    echo "=== Full: Full-data vs Few-shot ==="
    for ds in $DATASETS_FULL; do
        for method in $METHODS; do
            run "$ds" "$method" ""                  # 10-shot
            run "$ds" "$method" "--disable_fewshot"  # full data
        done
    done
}

case "${1:---full}" in
    --quick) quick_mode ;;
    --full)  full_mode ;;
    *)
        echo "Usage: $0 [--quick|--full]"
        echo "  --quick: STL10, GTSRB, SVHN only (3 datasets × 3 methods × 2 settings = 18 runs)"
        echo "  --full:  All 7 datasets × 3 methods × 2 settings = 42 runs"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results saved in: $STORAGE_DIR/results/"
echo "============================================================"

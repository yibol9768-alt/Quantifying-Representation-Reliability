#!/bin/bash
# Unified experiment runner.
#
# Usage:
#   bash experiments/run_experiments.sh              # 运行所有实验
#   bash experiments/run_experiments.sh --quick       # 快速测试
#   bash experiments/run_experiments.sh --methods     # 所有融合方法对比
#   bash experiments/run_experiments.sh --scaling     # 模型数量递增实验

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
DATASET="${DATASET:-cifar100}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"

run_single() {
    echo ">>> $BASE_CMD --dataset $DATASET --model $1 ${@:2}"
    $BASE_CMD --dataset "$DATASET" --model "$1" "${@:2}"
}

run_fusion() {
    local method=$1
    local models=$2
    shift 2
    echo ">>> Fusion: method=$method models=$models $@"
    $BASE_CMD --dataset "$DATASET" --model fusion \
        --fusion_method "$method" --fusion_models "$models" "$@"
}

# ── Quick test ──
quick_test() {
    echo "=== Quick Test ==="
    run_single clip
    run_fusion concat clip,dino
    echo "=== Quick Test Done ==="
}

# ── All fusion methods comparison ──
methods_comparison() {
    echo "=== Methods Comparison (2 models: clip,dino) ==="
    for method in concat proj_concat weighted_sum gated \
                  difference_concat hadamard_concat bilinear_concat \
                  film context_gating lmf se_fusion \
                  comm mmvit \
                  topk_router moe_router attention_router; do
        run_fusion "$method" clip,dino || echo "FAILED: $method"
    done
}

# ── Model scaling experiment (original 6) ──
scaling_experiment() {
    echo "=== Model Scaling (gated fusion, original 6 models) ==="
    local method="${1:-gated}"
    run_single clip
    run_fusion "$method" clip,dino
    run_fusion "$method" clip,dino,mae
    run_fusion "$method" clip,dino,mae,siglip
    run_fusion "$method" clip,dino,mae,siglip,convnext
    run_fusion "$method" clip,dino,mae,siglip,convnext,data2vec
}

# ── Extended model scaling (use selection ordering) ──
scaling_extended() {
    echo "=== Extended Model Scaling (use --ordering to specify model order) ==="
    echo "=== Run selection first: python experiments/run_selection_comparison.py ==="
    local method="${1:-gated}"
    local ordering="${2:-clip,dino,mae,siglip,convnext,data2vec,vit,dinov2_large,clip_large,beit,resnet50}"
    IFS=',' read -ra models <<< "$ordering"
    local accum=""
    for model in "${models[@]}"; do
        if [ -z "$accum" ]; then
            accum="$model"
            run_single "$model"
        else
            accum="$accum,$model"
            run_fusion "$method" "$accum"
        fi
    done
}

# ── Multi-dataset experiment ──
multi_dataset() {
    echo "=== Multi-Dataset ==="
    local method="${1:-gated}"
    local models="${2:-clip,dino,mae,siglip}"
    for ds in cifar100 stl10 pets eurosat dtd gtsrb svhn country211; do
        DATASET="$ds" run_fusion "$method" "$models" || echo "FAILED: $ds"
    done
}

# ── Main dispatch ──
case "${1:---all}" in
    --quick)     quick_test ;;
    --methods)   methods_comparison ;;
    --scaling)   scaling_experiment "${2:-gated}" ;;
    --scaling-ext) scaling_extended "${2:-gated}" "${3:-}" ;;
    --datasets)  multi_dataset "${2:-gated}" "${3:-clip,dino,mae,siglip}" ;;
    --all)
        quick_test
        methods_comparison
        scaling_experiment
        ;;
    *)
        echo "Usage: $0 [--quick|--methods|--scaling|--scaling-ext|--datasets|--all]"
        echo "  --scaling-ext METHOD ORDERING: extended scaling with custom model order"
        exit 1
        ;;
esac

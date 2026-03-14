#!/bin/bash
# Full-data Scaling 实验
# 验证：full data 下"更多模型=更差"是否仍然成立
#
# 对比：
#   - few-shot scaling（已有结果，峰值在 4-5 模型）
#   - full-data scaling（本实验，预测峰值上移或消失）
#
# Usage:
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_fulldata_scaling.sh
#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_fulldata_scaling.sh --quick

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="${CACHE_DTYPE:-fp32}"

BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"

DATASETS_FULL="stl10 pets eurosat dtd gtsrb svhn country211"
DATASETS_QUICK="stl10 gtsrb svhn"

# 模型递增顺序（与 few-shot scaling 实验保持一致）
MODEL_STEPS=(
    "clip"
    "clip,dino"
    "clip,dino,mae"
    "clip,dino,mae,siglip"
    "clip,dino,mae,siglip,convnext"
    "clip,dino,mae,siglip,convnext,data2vec"
)

run_scaling() {
    local dataset=$1
    local fewshot_flag=$2  # "" or "--disable_fewshot"
    local tag
    if [ -z "$fewshot_flag" ]; then tag="10shot"; else tag="fulldata"; fi

    for i in "${!MODEL_STEPS[@]}"; do
        local models="${MODEL_STEPS[$i]}"
        local n_models=$((i + 1))

        echo ""
        echo "============================================================"
        echo "  $dataset | ${n_models}模型 ($models) | $tag"
        echo "============================================================"

        if [ "$n_models" -eq 1 ]; then
            # 单模型不需要 fusion
            $BASE_CMD --dataset "$dataset" --model clip \
                $fewshot_flag \
                || echo "FAILED: $dataset ${n_models}模型 $tag"
        else
            $BASE_CMD --dataset "$dataset" --model fusion \
                --fusion_method gated --fusion_models "$models" \
                $fewshot_flag \
                || echo "FAILED: $dataset ${n_models}模型 $tag"
        fi
    done
}

quick_mode() {
    echo "=== Quick: Full-data Scaling (3 datasets) ==="
    for ds in $DATASETS_QUICK; do
        run_scaling "$ds" "--disable_fewshot"
    done
    echo ""
    echo "Quick mode: 3 datasets × 6 steps = 18 runs"
}

full_mode() {
    echo "=== Full: Full-data Scaling (7 datasets) ==="
    for ds in $DATASETS_FULL; do
        run_scaling "$ds" "--disable_fewshot"
    done
    echo ""
    echo "Full mode: 7 datasets × 6 steps = 42 runs"
}

both_mode() {
    echo "=== Both: Few-shot + Full-data Scaling ==="
    for ds in $DATASETS_QUICK; do
        run_scaling "$ds" ""                  # 10-shot
        run_scaling "$ds" "--disable_fewshot"  # full data
    done
    echo ""
    echo "Both mode: 3 datasets × 6 steps × 2 settings = 36 runs"
}

case "${1:---full}" in
    --quick) quick_mode ;;
    --full)  full_mode ;;
    --both)  both_mode ;;
    *)
        echo "Usage: $0 [--quick|--full|--both]"
        echo "  --quick: 3 datasets, full-data only (18 runs)"
        echo "  --full:  7 datasets, full-data only (42 runs)"
        echo "  --both:  3 datasets, few-shot + full-data (36 runs)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results saved in: $STORAGE_DIR/results/"
echo "============================================================"

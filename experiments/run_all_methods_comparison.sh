#!/bin/bash
# ================================================================
# All Fusion Methods Comparison (6 模型, few-shot 10-shot)
# ================================================================
# 在多个数据集上横向对比所有简单融合方法
# 配置: 6 默认模型, 10-shot, 10 epochs
#
# 用法:
#   export STORAGE_DIR=/path/to/bigfiles
#   bash experiments/run_all_methods_comparison.sh
#
# 可选: 只跑指定数据集
#   DATASETS="stl10 pets" bash experiments/run_all_methods_comparison.sh

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?请先 export STORAGE_DIR}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-42}"
CACHE_DTYPE="fp32"
FUSION_MODELS="clip,dino,mae,siglip,convnext,data2vec"

# 默认数据集（可通过环境变量覆盖）
if [ -z "${DATASETS:-}" ]; then
    DATASETS_ARR=(svhn eurosat stl10 pets dtd gtsrb country211)
else
    read -ra DATASETS_ARR <<< "${DATASETS}"
fi

# 所有简单融合方法（不含 comm/mmvit 这类 token 级方法）
METHODS=(
    concat
    proj_concat
    weighted_sum
    gated
    difference_concat
    hadamard_concat
    bilinear_concat
    film
    context_gating
    lmf
    se_fusion
)

TOTAL=$(( ${#DATASETS_ARR[@]} * ${#METHODS[@]} ))
DONE=0
FAILED=0
FAIL_LIST=""

echo "============================================================"
echo "All Fusion Methods Comparison"
echo "============================================================"
echo "Models:   ${FUSION_MODELS}"
echo "Methods:  ${METHODS[*]}"
echo "Datasets: ${DATASETS_ARR[*]}"
echo "Epochs:   ${EPOCHS}   Batch: ${BATCH_SIZE}   Seed: ${SEED}"
echo "Total experiments: ${TOTAL}"
echo "============================================================"

for dataset in "${DATASETS_ARR[@]}"; do
    echo ""
    echo "━━━ Dataset: ${dataset} ━━━"
    for method in "${METHODS[@]}"; do
        DONE=$((DONE + 1))
        printf "[%d/%d] %-12s / %-18s ... " "${DONE}" "${TOTAL}" "${dataset}" "${method}"

        if python main.py \
            --dataset "${dataset}" \
            --model fusion \
            --fusion_method "${method}" \
            --fusion_models "${FUSION_MODELS}" \
            --storage_dir "${STORAGE_DIR}" \
            --seed "${SEED}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --cache_dtype "${CACHE_DTYPE}" \
            > /dev/null 2>&1; then
            echo "✓"
        else
            FAILED=$((FAILED + 1))
            FAIL_LIST="${FAIL_LIST}\n  - ${dataset} / ${method}"
            echo "✗ FAILED"
        fi
    done
done

echo ""
echo "============================================================"
echo "完成: $((TOTAL - FAILED))/${TOTAL}"
if [ "${FAILED}" -gt 0 ]; then
    echo "失败: ${FAILED}/${TOTAL}"
    echo -e "失败列表:${FAIL_LIST}"
fi
echo "============================================================"

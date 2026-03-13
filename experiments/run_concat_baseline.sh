#!/bin/bash
# ================================================================
# Concat Baseline: 6 模型 concat 融合 (few-shot 10-shot)
# ================================================================
# 为动态路由实验提供最简单的 baseline 对比
# 配置与 Round 1 路由实验一致：6 模型、10-shot、10 epochs
#
# 用法:
#   export STORAGE_DIR=/path/to/bigfiles
#   bash experiments/run_concat_baseline.sh

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?请先 export STORAGE_DIR}"
EPOCHS=10
BATCH_SIZE=128
SEED=42
CACHE_DTYPE="fp32"
FUSION_MODELS="clip,dino,mae,siglip,convnext,data2vec"

DATASETS=(svhn eurosat stl10 pets dtd gtsrb country211)

echo "========================================"
echo "Concat Baseline (6 models, 10-shot)"
echo "========================================"
echo "Models: ${FUSION_MODELS}"
echo "Datasets: ${DATASETS[*]}"
echo "Epochs: ${EPOCHS}"
echo "========================================"

TOTAL=${#DATASETS[@]}
DONE=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "── [$((DONE+1))/${TOTAL}] ${dataset} ──"

    if python main.py \
        --dataset "${dataset}" \
        --model fusion \
        --fusion_method concat \
        --fusion_models "${FUSION_MODELS}" \
        --storage_dir "${STORAGE_DIR}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --cache_dtype "${CACHE_DTYPE}"; then
        DONE=$((DONE + 1))
        echo "✓ ${dataset} done"
    else
        FAILED=$((FAILED + 1))
        DONE=$((DONE + 1))
        echo "✗ ${dataset} FAILED"
    fi
done

echo ""
echo "========================================"
echo "Concat Baseline 完成"
echo "成功: $((TOTAL - FAILED))/${TOTAL}"
if [ "${FAILED}" -gt 0 ]; then
    echo "失败: ${FAILED}/${TOTAL}"
fi
echo "========================================"

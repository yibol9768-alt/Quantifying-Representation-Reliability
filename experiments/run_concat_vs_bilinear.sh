#!/bin/bash
# ================================================================
# Concat vs Bilinear Baseline (6 模型, few-shot 10-shot)
# ================================================================
# 对比 concat 和 bilinear_concat 两种最基础的融合方法
# 配置与 Round 1 路由实验一致
#
# 用法:
#   export STORAGE_DIR=/path/to/bigfiles
#   bash experiments/run_concat_vs_bilinear.sh

set -euo pipefail

STORAGE_DIR="${STORAGE_DIR:?请先 export STORAGE_DIR}"
EPOCHS=10
BATCH_SIZE=128
SEED=42
CACHE_DTYPE="fp32"
FUSION_MODELS="clip,dino,mae,siglip,convnext,data2vec"

DATASETS=(svhn eurosat stl10 pets dtd gtsrb country211)
METHODS=(concat bilinear_concat)

TOTAL=$(( ${#DATASETS[@]} * ${#METHODS[@]} ))
DONE=0
FAILED=0

echo "========================================"
echo "Concat vs Bilinear (6 models, 10-shot)"
echo "========================================"
echo "Models: ${FUSION_MODELS}"
echo "Methods: ${METHODS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Total experiments: ${TOTAL}"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        DONE=$((DONE + 1))
        echo ""
        echo "── [${DONE}/${TOTAL}] ${dataset} / ${method} ──"

        if python main.py \
            --dataset "${dataset}" \
            --model fusion \
            --fusion_method "${method}" \
            --fusion_models "${FUSION_MODELS}" \
            --storage_dir "${STORAGE_DIR}" \
            --seed "${SEED}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --cache_dtype "${CACHE_DTYPE}"; then
            echo "✓ ${dataset} / ${method} done"
        else
            FAILED=$((FAILED + 1))
            echo "✗ ${dataset} / ${method} FAILED"
        fi
    done
done

echo ""
echo "========================================"
echo "完成: $((TOTAL - FAILED))/${TOTAL}"
[ "${FAILED}" -gt 0 ] && echo "失败: ${FAILED}/${TOTAL}"
echo "========================================"

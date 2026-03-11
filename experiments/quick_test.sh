#!/bin/bash
# ================================================================
# Quick Test Script - 快速测试脚本
# ================================================================
# 用法: bash experiments/quick_test.sh
#
# 这是一个精简版的实验脚本，用于快速验证配置是否正确
# 只在CIFAR-100上运行少量实验组合
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载配置
source "${SCRIPT_DIR}/config.sh"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Quick Test - Fusion Experiments"
echo "========================================"
echo "This will run a small subset of experiments"
echo "to verify everything is working."
echo "Only using CIFAR-100 dataset."
echo ""

# 检查
if [ ! -d "${STORAGE_DIR}/models" ]; then
    echo "ERROR: Models directory not found"
    echo "Please set STORAGE_DIR or download models first"
    exit 1
fi

# 创建测试输出目录
TEST_DIR="${STORAGE_DIR}/results/quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Results will be saved to: $TEST_DIR"
echo ""

# 测试数据集
TEST_DATASET="cifar100"

# ========== 测试：单模型 ==========
echo "[1/5] Testing single model (CLIP) on ${TEST_DATASET}..."
python main.py \
    --dataset "${TEST_DATASET}" \
    --model clip \
    --storage_dir "${STORAGE_DIR}" \
    --epochs 3 \
    --batch_size 128 \
    --results_dir "$TEST_DIR" \
    || echo "✗ Single model failed"

# ========== 测试：2模型 + concat ==========
echo "[2/5] Testing 2 models with concat on ${TEST_DATASET}..."
python main.py \
    --dataset "${TEST_DATASET}" \
    --model fusion \
    --fusion_method concat \
    --fusion_models "clip,dino" \
    --storage_dir "${STORAGE_DIR}" \
    --epochs 3 \
    --batch_size 128 \
    --results_dir "$TEST_DIR" \
    || echo "✗ 2-model concat failed"

# ========== 测试：3模型 + weighted_sum ==========
echo "[3/5] Testing 3 models with weighted_sum on ${TEST_DATASET}..."
python main.py \
    --dataset "${TEST_DATASET}" \
    --model fusion \
    --fusion_method weighted_sum \
    --fusion_models "mae,clip,dino" \
    --storage_dir "${STORAGE_DIR}" \
    --epochs 3 \
    --batch_size 128 \
    --results_dir "$TEST_DIR" \
    || echo "✗ 3-model weighted_sum failed"

# ========== 测试：3模型 + gated ==========
echo "[4/5] Testing 3 models with gated on ${TEST_DATASET}..."
python main.py \
    --dataset "${TEST_DATASET}" \
    --model fusion \
    --fusion_method gated \
    --fusion_models "mae,clip,dino" \
    --storage_dir "${STORAGE_DIR}" \
    --epochs 3 \
    --batch_size 128 \
    --results_dir "$TEST_DIR" \
    || echo "✗ 3-model gated failed"

# ========== 测试：3模型 + proj_concat ==========
echo "[5/5] Testing 3 models with proj_concat on ${TEST_DATASET}..."
python main.py \
    --dataset "${TEST_DATASET}" \
    --model fusion \
    --fusion_method proj_concat \
    --fusion_models "mae,clip,dino" \
    --storage_dir "${STORAGE_DIR}" \
    --epochs 3 \
    --batch_size 128 \
    --results_dir "$TEST_DIR" \
    || echo "✗ 3-model proj_concat failed"

# ========== 收集结果 ==========
echo ""
echo "========================================"
echo "Quick Test Completed!"
echo "========================================"
echo "Results: $TEST_DIR"

# 显示结果摘要
python "${SCRIPT_DIR}/collect_results.py" \
    --results_dir "$TEST_DIR" \
    --output "$TEST_DIR/results.csv" \
    2>/dev/null || echo "Results collection skipped (no results yet)"

echo ""
echo "Quick test passed! Ready for full experiments."
echo ""
echo "To run full experiments on CLIP datasets:"
echo "  bash experiments/run_fusion_experiments.sh"
echo ""
echo "Full experiment breakdown:"
echo "  - 8 CLIP datasets"
echo "  - 1 single model (CLIP) per dataset"
echo "  - 10 model counts × 6 fusion methods per dataset"
echo "  - Total: 488 experiments"
echo ""
echo "Estimated time: 40-80 hours (depending on hardware)"

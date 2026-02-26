#!/bin/bash
# 一键运行脚本 (Linux/Mac)
#
# 使用前:
#   conda create -n repreli python=3.10
#   conda activate repreli
#   pip install -r requirements.txt
#
# 运行:
#   ./scripts/run_all.sh

set -e

CONFIG_FILE="configs/default.yaml"

echo "=============================================="
echo "Heterogeneous Model Evaluation Pipeline"
echo "=============================================="
echo ""

# 检查 Python
python --version

# 检查依赖
echo "Checking dependencies..."
if ! python -c "import torch, transformers, numpy, scipy, sklearn" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# 创建目录
mkdir -p data features results model_cache

# ============================================================
# Step 1: 下载预训练模型 (不需要训练！)
# ============================================================
echo ""
echo "[1/3] Downloading pretrained models..."
echo "----------------------------------------------"
python scripts/download_models.py --family clip dinov2 mae --cache_dir ./model_cache

# ============================================================
# Step 2: 提取特征
# ============================================================
echo ""
echo "[2/3] Extracting features..."
echo "----------------------------------------------"

MODELS=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(c['models']))")
DATASET=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['dataset'])")

echo "Models: $MODELS"
echo "Dataset: $DATASET"

python scripts/extract_features.py \
    --models $MODELS \
    --dataset $DATASET \
    --output ./features \
    --split both

# ============================================================
# Step 3: 训练线性头 + 计算 NC + 评估
# ============================================================
echo ""
echo "[3/3] Running evaluation..."
echo "----------------------------------------------"
echo "Training linear heads and computing NC scores..."
echo ""

python scripts/run_evaluation.py --config $CONFIG_FILE

# ============================================================
# 结果
# ============================================================
echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "Results:"
ls -la results/

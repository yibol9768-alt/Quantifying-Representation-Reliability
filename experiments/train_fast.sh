#!/bin/bash
# ================================================================
# Fast Training Script - 使用预提取特征快速训练
# ================================================================
# 用法: bash experiments/train_fast.sh
#
# 前置条件: 先运行 extract_all_features.sh 提取所有特征
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载配置
source "${SCRIPT_DIR}/config.sh"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Fast Training with Pre-extracted Features"
echo "========================================"
echo "Feature cache: ${STORAGE_DIR}/features"
echo ""

# 检查特征是否已提取
FEATURE_CACHE_DIR="${STORAGE_DIR}/features"
if [ ! -d "$FEATURE_CACHE_DIR" ]; then
    echo "ERROR: Features not extracted yet!"
    echo "Please run first: bash experiments/extract_all_features.sh"
    exit 1
fi

# 输出目录
RESULTS_DIR="${STORAGE_DIR}/results/fast_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ------------------------------------------------
# 配置
# ------------------------------------------------
export NUM_EPOCHS=10  # MLP训练很快，可以多跑几轮
export BATCH_SIZE=256  # 特征很小，可以用大batch
export LEARNING_RATE=1e-3

# ------------------------------------------------
# 特征维度映射
# ------------------------------------------------
declare -A FEATURE_DIMS=(
    ["clip"]=768
    ["dino"]=768
    ["mae"]=768
    ["vit"]=768
    ["swin"]=1024
    ["beit"]=768
    ["deit"]=768
    ["convnext"]=1024
    ["eva"]=768
    ["mae_large"]=1024
    ["dino_large"]=1024
    ["clip_large"]=768
    ["openclip"]=512
)

# 数据集类别数
declare -A NUM_CLASSES=(
    ["mnist"]=10
    ["svhn"]=10
    ["dtd"]=47
    ["eurosat"]=10
    ["gtsrb"]=43
    ["country211"]=211
    ["resisc45"]=45
    ["cifar100"]=100
)

# ------------------------------------------------
# 训练函数（单模型）
# ------------------------------------------------
train_single_model() {
    local dataset=$1
    local model=$2
    local num_classes=${NUM_CLASSES[$dataset]}
    local feature_dim=${FEATURE_DIMS[$model]}
    local cache_path="${FEATURE_CACHE_DIR}/${dataset}_${model}_seed${SEED}"

    # 检查特征是否存在
    if [ ! -d "${cache_path}/train" ]; then
        echo "✗ Features not found: ${cache_path}"
        return 1
    fi

    echo ">>> Training: ${model} on ${dataset}"

    # 使用预提取的特征路径作为cache目录
    python main.py \
        --dataset "${dataset}" \
        --model "${model}" \
        --storage_dir "${STORAGE_DIR}" \
        --cache_dir "$cache_path" \
        --epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LEARNING_RATE}" \
        --seed "${SEED}" \
        --results_dir "${RESULTS_DIR}" \
        --no_precompute \
        2>&1 | tee "${RESULTS_DIR}/train_${dataset}_${model}.log"

    echo "✓ Completed: ${model} on ${dataset}"
    return 0
}

# ------------------------------------------------
# 训练函数（融合模型）
# ------------------------------------------------
train_fusion() {
    local dataset=$1
    local models=$2
    local method=$3

    # 构建cache目录路径（多个模型需要组合）
    local cache_dirs=()
    IFS=',' read -ra MODEL_ARRAY <<< "$models"
    for model in "${MODEL_ARRAY[@]}"; do
        local cache_path="${FEATURE_CACHE_DIR}/${dataset}_${model}_seed${SEED}"
        if [ ! -d "${cache_path}/train" ]; then
            echo "✗ Features not found: ${cache_path}"
            return 1
        fi
        cache_dirs+=("$cache_path")
    done

    echo ">>> Training fusion: ${method} on ${dataset} with [${models}]"

    # 对于融合，我们需要特殊处理
    # 暂时使用在线模式，因为融合需要同时加载多个模型的特征
    python main.py \
        --dataset "${dataset}" \
        --model fusion \
        --fusion_method "${method}" \
        --fusion_models "${models}" \
        --storage_dir "${STORAGE_DIR}" \
        --epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LEARNING_RATE}" \
        --seed "${SEED}" \
        --results_dir "${RESULTS_DIR}" \
        2>&1 | tee "${RESULTS_DIR}/train_${dataset}_fusion_${method}.log"

    echo "✓ Completed: ${method} fusion on ${dataset}"
    return 0
}

# ------------------------------------------------
# 运行训练
# ------------------------------------------------

# ========== Phase 1: 单模型训练 ==========
echo ""
echo "========================================"
echo "Phase 1: Single Models (Fast!)"
echo "========================================"

for dataset in "${CLIP_DATASETS[@]}"; do
    if [ ! -d "${STORAGE_DIR}/data/${dataset}" ]; then
        echo "Skipping ${dataset} (data not found)"
        continue
    fi

    echo ""
    echo "--- Dataset: ${dataset} ---"

    # 只训练关键模型（节省时间）
    for model in "clip" "dino" "mae"; do
        train_single_model "$dataset" "$model"
    done
done

# ========== Phase 2: 融合模型训练 ==========
echo ""
echo "========================================"
echo "Phase 2: Fusion Models"
echo "========================================"

for dataset in "${CLIP_DATASETS[@]}"; do
    if [ ! -d "${STORAGE_DIR}/data/${dataset}" ]; then
        continue
    fi

    echo ""
    echo "--- Dataset: ${dataset} ---"

    # 测试几个融合方法
    for method in "concat" "weighted_sum" "gated"; do
        train_fusion "$dataset" "clip,dino" "$method"
    done
done

# ------------------------------------------------
# 收集结果
# ------------------------------------------------
echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Results: ${RESULTS_DIR}"
echo ""
echo "To view results:"
echo "  cat ${RESULTS_DIR}/results_summary.txt"

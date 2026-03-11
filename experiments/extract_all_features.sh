#!/bin/bash
# ================================================================
# Feature Extraction Script - 特征预提取脚本
# ================================================================
# 用法: bash experiments/extract_all_features.sh
#
# 功能：
# 1. 提取所有模型在所有数据集上的特征
# 2. 保存到磁盘，后续训练只需加载特征
# 3. 生成特征索引，记录已提取的特征
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载配置
source "${SCRIPT_DIR}/config.sh"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Feature Extraction - All Models × All Datasets"
echo "========================================"
echo "Storage: ${STORAGE_DIR}"
echo ""

# ------------------------------------------------
# 检查依赖
# ------------------------------------------------
if [ ! -d "${STORAGE_DIR}/models" ]; then
    echo "ERROR: Models not found at ${STORAGE_DIR}/models"
    echo "Please run: python download_models.py --models --storage_dir ${STORAGE_DIR}"
    exit 1
fi

if [ ! -d "${STORAGE_DIR}/data" ]; then
    echo "ERROR: Data not found at ${STORAGE_DIR}/data"
    echo "Please run: python download_models.py --all_datasets --storage_dir ${STORAGE_DIR}"
    exit 1
fi

# ------------------------------------------------
# 配置
# ------------------------------------------------
export FEATURE_CACHE_DIR="${STORAGE_DIR}/features"
mkdir -p "${FEATURE_CACHE_DIR}"

# 索引文件
INDEX_FILE="${FEATURE_CACHE_DIR}/feature_index.txt"
echo "# Feature Extraction Index - $(date)" > "$INDEX_FILE"
echo "# Format: model|dataset|status|path" >> "$INDEX_FILE"
echo "" >> "$INDEX_FILE"

# 所有模型（用于单模型实验和融合）
ALL_MODELS=(
    "clip"            # 单模型baseline主力
    "dino"            # 自监督
    "mae"             # 自监督
    "vit"             # 标准ViT
    "swin"            # 层级ViT
    "beit"            # BEiT
    "deit"            # DeiT
    "convnext"        # 现代CNN
    "eva"             # EVA
    "mae_large"       # 大模型
    "dino_large"      # 大模型
    "clip_large"      # CLIP大模型
    "openclip"        # OpenCLIP
)

# ------------------------------------------------
# 特征提取函数
# ------------------------------------------------
extract_features() {
    local model=$1
    local dataset=$2
    local cache_name="${dataset}_${model}_seed${SEED}"
    local cache_path="${FEATURE_CACHE_DIR}/${cache_name}"

    echo ">>> Extracting: ${model} on ${dataset}"
    echo "    Cache: ${cache_path}"

    # 检查是否已存在
    if [ -d "${cache_path}/train" ] && [ -d "${cache_path}/test" ]; then
        echo "    ✓ Already exists, skipping"
        echo "${model}|${dataset}|exists|${cache_path}" >> "$INDEX_FILE"
        return 0
    fi

    # 提取特征
    python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
import torch

from src.models.extractors import get_extractor
from src.data.hf_dataset import get_dataloaders
from src.training.offline_cache import build_split_cache

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_type = '${model}'
dataset = '${dataset}'
storage_dir = '${STORAGE_DIR}'
cache_dir = '${cache_path}'

print(f'Extracting {model_type} on {dataset}')
print(f'Device: {device}')

# 加载模型
extractor = get_extractor(model_type, model_dir=storage_dir).to(device)
extractor.eval()

# 加载数据
train_loader, test_loader = get_dataloaders(
    dataset, storage_dir + '/data',
    batch_size=128, num_workers=4, model_type=model_type
)

# 构建缓存
build_split_cache(extractor, train_loader, Path(cache_dir) / 'train', 'train', device, torch.float32, False, True)
build_split_cache(extractor, test_loader, Path(cache_dir) / 'test', 'test', device, torch.float32, False, True)

print(f'✓ Saved to {cache_dir}')
" 2>&1 | tee -a "${FEATURE_CACHE_DIR}/log_${model}_${dataset}.txt"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✓ Success"
        echo "${model}|${dataset}|success|${cache_path}" >> "$INDEX_FILE"
        return 0
    else
        echo "    ✗ Failed"
        echo "${model}|${dataset}|failed|${cache_path}" >> "$INDEX_FILE"
        return 1
    fi
}

# ------------------------------------------------
# 统计信息
# ------------------------------------------------
TOTAL_MODELS=${#ALL_MODELS[@]}
TOTAL_DATASETS=${#CLIP_DATASETS[@]}
TOTAL_EXTRACTIONS=$((TOTAL_MODELS * TOTAL_DATASETS))

echo "Extraction Plan:"
echo "  Models: ${TOTAL_MODELS}"
echo "  Datasets: ${TOTAL_DATASETS}"
echo "  Total extractions: ${TOTAL_EXTRACTIONS}"
echo ""
echo "Cache directory: ${FEATURE_CACHE_DIR}"
echo "========================================"
echo ""

# ------------------------------------------------
# 执行特征提取
# ------------------------------------------------
success_count=0
failed_count=0
skipped_count=0

for dataset in "${CLIP_DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Dataset: ${dataset}"
    echo "========================================"

    # 检查数据集是否存在
    if [ ! -d "${STORAGE_DIR}/data/${dataset}" ]; then
        echo "WARNING: Dataset ${dataset} not found, skipping"
        for model in "${ALL_MODELS[@]}"; do
            echo "${model}|${dataset}|dataset_not_found|N/A" >> "$INDEX_FILE"
        done
        continue
    fi

    for model in "${ALL_MODELS[@]}"; do
        if extract_features "$model" "$dataset"; then
            ((success_count++))
        else
            ((failed_count++))
        fi
    done
done

# ------------------------------------------------
# 生成统计报告
# ------------------------------------------------
echo ""
echo "========================================"
echo "Feature Extraction Complete!"
echo "========================================"
echo "Success: ${success_count}/${TOTAL_EXTRACTIONS}"
echo "Failed: ${failed_count}/${TOTAL_EXTRACTIONS}"
echo "Skipped (datasets not found): $((TOTAL_EXTRACTIONS - success_count - failed_count))"
echo ""
echo "Cache location: ${FEATURE_CACHE_DIR}"
echo "Index file: ${INDEX_FILE}"
echo "========================================"

# 生成Python格式的索引（方便训练脚本使用）
cat > "${FEATURE_CACHE_DIR}/feature_paths.py" << EOF
# Feature paths for training
# Generated at $(date)

FEATURE_CACHE_DIR = "${FEATURE_CACHE_DIR}"

# Available features: {(model, dataset): cache_path}
FEATURE_PATHS = {
EOF

while IFS='|' read -r model dataset status path; do
    if [[ "$status" == "success" || "$status" == "exists" ]]; then
        echo "    ('${model}', '${dataset}'): '${path}'," >> "${FEATURE_CACHE_DIR}/feature_paths.py"
    fi
done < "$INDEX_FILE"

echo "}" >> "${FEATURE_CACHE_DIR}/feature_paths.py"
echo ""
echo "Feature paths saved to: ${FEATURE_CACHE_DIR}/feature_paths.py"

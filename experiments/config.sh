#!/bin/bash
# ================================================================
# Fusion Experiment Configuration
# ================================================================
# 系统融合实验配置
# 目标：测试不同模型数量和融合方法的组合效果
# 数据集：CLIP论文使用的全部数据集

# ------------------------------------------------
# 基础配置
# ------------------------------------------------
export STORAGE_DIR="${STORAGE_DIR:-/path/to/bigfiles}"
export EPOCHS=10
export BATCH_SIZE=128
export CACHE_DTYPE="fp32"
export SEED=42

# ------------------------------------------------
# 数据集配置 - CLIP论文数据集
# ------------------------------------------------
# 所有CLIP论文数据集
export CLIP_DATASETS=(
    "mnist"           # 手写数字 (10类)
    "svhn"            # 街景门牌号 (10类)
    "dtd"             # 纹理描述 (47类)
    "eurosat"         # 卫星图像 (10类)
    "gtsrb"           # 交通标志 (43类)
    "country211"      # 国家识别 (211类)
    "resisc45"        # 遥感场景 (45类)
    "cifar100"        # 基础对比 (100类)
)

# 单个数据集模式（用于快速测试）
# export CLIP_DATASETS=("cifar100")

# ------------------------------------------------
# 模型组合设计
# ------------------------------------------------
# 原则：从当前 12 个可稳定下载和加载的模型中选取代表性组合
# - 覆盖不同预训练范式（自监督、监督、对比学习）
# - 覆盖不同架构（ViT、Swin、CNN）
# - 逐步增加模型数量，观察融合效果变化

# 1个模型 - 单模型baseline（最强单模型）
export MODELS_1="clip"

# 2个模型 - 最小融合（两种不同范式）
export MODELS_2="clip,dino"

# 3个模型 - 经典三件套
export MODELS_3="mae,clip,dino"

# 4个模型 - 加入标准ViT
export MODELS_4="mae,clip,dino,vit"

# 5个模型 - 加入层级Transformer
export MODELS_5="mae,clip,dino,vit,swin"

# 6个模型 - 加入现代CNN
export MODELS_6="mae,clip,dino,vit,swin,convnext"

# 7个模型 - 加入Data-efficient ViT
export MODELS_7="mae,clip,dino,vit,swin,convnext,deit"

# 8个模型 - 加入Bootstrapped ImageText
export MODELS_8="mae,clip,dino,vit,swin,convnext,deit,beit"

# 9个模型 - 加入 OpenCLIP
export MODELS_9="mae,clip,dino,vit,swin,convnext,deit,beit,openclip"

# 10个模型 - 加入大模型
export MODELS_10="mae,clip,dino,vit,swin,convnext,deit,beit,openclip,mae_large"

# ------------------------------------------------
# 融合方法列表（只使用简单baseline方法）
# ------------------------------------------------
# 6个简单baseline方法 - 快速高效
export BASELINE_METHODS=(
    "concat"           # 原始拼接
    "proj_concat"      # 投影拼接
    "weighted_sum"     # 加权求和
    "gated"            # 门控融合
    "difference_concat"# 差异拼接
    "hadamard_concat"  # 哈达玛积拼接
)

# 所有融合方法（只用baseline，不跑复杂的comm/mmvit）
export ALL_METHODS=("${BASELINE_METHODS[@]}")

# 论文启发方法（comm, mmvit）- 已禁用，计算量大
# 如需测试，取消注释并添加到ALL_METHODS
# export PAPER_METHODS=("comm" "mmvit")

# ------------------------------------------------
# 输出目录
# ------------------------------------------------
export EXPERIMENT_NAME="fusion_$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="${STORAGE_DIR}/results/${EXPERIMENT_NAME}"
export LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Fusion Experiment Configuration"
echo "========================================"
echo "Storage Dir: ${STORAGE_DIR}"
echo "Datasets: ${CLIP_DATASETS[*]}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Results Dir: ${RESULTS_DIR}"
echo "========================================"

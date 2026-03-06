#!/bin/bash
# 所有数据集完整实验脚本
# 包含: cifar10, cifar100, flowers102, pets, stanford_cars, food101

set -e  # 遇到错误立即退出

# 环境配置
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"

# 目录配置
FEATURE_DIR="${FEATURE_DIR:-/root/autodl-tmp/features}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/checkpoints}"

# 所有数据集
ALL_DATASETS=("cifar10" "cifar100" "flowers102" "pets" "stanford_cars" "food101")

# 创建输出目录
mkdir -p "$FEATURE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "所有数据集完整实验流程"
echo "=========================================="
echo "数据集: ${ALL_DATASETS[@]}"
echo "数据根目录: $DATA_ROOT"
echo "特征目录: $FEATURE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 遍历每个数据集
for DATASET in "${ALL_DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始 $DATASET 数据集实验"
    echo "=========================================="
    
    # ===========================
    # 1. 提取单模型特征
    # ===========================
    echo ""
    echo "Step 1: 提取单模型特征 - $DATASET"
    echo "=========================================="
    
    for model in clip dino mae; do
        echo ""
        echo "提取 $model 特征 ($DATASET)..."
        
        train_file="$FEATURE_DIR/${DATASET}_${model}_train.pt"
        test_file="$FEATURE_DIR/${DATASET}_${model}_test.pt"
        
        if [ -f "$train_file" ] && [ -f "$test_file" ]; then
            echo "⊘ $model 特征已存在，跳过"
            continue
        fi
        
        python scripts/extract.py \
            --model "$model" \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 128
        
        python scripts/extract.py \
            --model "$model" \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 128
        
        echo "✓ $model 特征提取完成"
    done
    
    # ===========================
    # 2. 提取双模型融合特征
    # ===========================
    echo ""
    echo "Step 2: 提取双模型融合特征 - $DATASET"
    echo "=========================================="
    
    # CLIP + DINO
    echo "提取 CLIP + DINO ($DATASET)..."
    if [ ! -f "$FEATURE_DIR/${DATASET}_clip_dino_train.pt" ]; then
        python scripts/extract.py \
            --models clip dino \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    if [ ! -f "$FEATURE_DIR/${DATASET}_clip_dino_test.pt" ]; then
        python scripts/extract.py \
            --models clip dino \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    
    # CLIP + MAE
    echo "提取 CLIP + MAE ($DATASET)..."
    if [ ! -f "$FEATURE_DIR/${DATASET}_clip_mae_train.pt" ]; then
        python scripts/extract.py \
            --models clip mae \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    if [ ! -f "$FEATURE_DIR/${DATASET}_clip_mae_test.pt" ]; then
        python scripts/extract.py \
            --models clip mae \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    
    # DINO + MAE
    echo "提取 DINO + MAE ($DATASET)..."
    if [ ! -f "$FEATURE_DIR/${DATASET}_dino_mae_train.pt" ]; then
        python scripts/extract.py \
            --models dino mae \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    if [ ! -f "$FEATURE_DIR/${DATASET}_dino_mae_test.pt" ]; then
        python scripts/extract.py \
            --models dino mae \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    
    # ===========================
    # 3. 提取三模型融合特征
    # ===========================
    echo ""
    echo "Step 3: 提取三模型融合特征 - $DATASET"
    echo "=========================================="
    
    if [ ! -f "$FEATURE_DIR/${DATASET}_clip_dino_mae_train.pt" ]; then
        echo "提取 CLIP + DINO + MAE ($DATASET)..."
        python scripts/extract.py \
            --models clip dino mae \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    if [ ! -f "$FEATURE_DIR/${DATASET}_clip_dino_mae_test.pt" ]; then
        python scripts/extract.py \
            --models clip dino mae \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 64
    fi
    
    # ===========================
    # 4. 提取 COMM 多层特征
    # ===========================
    echo ""
    echo "Step 4: 提取 COMM 多层特征 - $DATASET"
    echo "=========================================="
    
    # COMM (CLIP + DINO)
    if [ ! -f "$FEATURE_DIR/${DATASET}_comm_train.pt" ]; then
        echo "提取 COMM ($DATASET)..."
        python scripts/extract.py \
            --method comm \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 32
    fi
    if [ ! -f "$FEATURE_DIR/${DATASET}_comm_test.pt" ]; then
        python scripts/extract.py \
            --method comm \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 32
    fi
    
    # COMM3 (三模型)
    if [ ! -f "$FEATURE_DIR/${DATASET}_comm3_train.pt" ]; then
        echo "提取 COMM3 ($DATASET)..."
        python scripts/extract.py \
            --method comm3 \
            --dataset "$DATASET" \
            --split train \
            --output-dir "$FEATURE_DIR" \
            --batch-size 32
    fi
    if [ ! -f "$FEATURE_DIR/${DATASET}_comm3_test.pt" ]; then
        python scripts/extract.py \
            --method comm3 \
            --dataset "$DATASET" \
            --split test \
            --output-dir "$FEATURE_DIR" \
            --batch-size 32
    fi
    
    # ===========================
    # 5. 训练单模型
    # ===========================
    echo ""
    echo "Step 5: 训练单模型 - $DATASET"
    echo "=========================================="
    
    for model in clip dino mae; do
        echo ""
        echo "训练单模型: $model ($DATASET)"
        
        checkpoint="$OUTPUT_DIR/${DATASET}_${model}_concat.pth"
        if [ -f "$checkpoint" ]; then
            echo "⊘ $checkpoint 已存在，跳过"
            continue
        fi
        
        python scripts/train.py \
            --model "$model" \
            --dataset "$DATASET" \
            --epochs 30 \
            --batch-size 256 \
            --feature-dir "$FEATURE_DIR"
        
        echo "✓ $model ($DATASET) 训练完成"
    done
    
    # ===========================
    # 6. 训练双模型
    # ===========================
    echo ""
    echo "Step 6: 训练双模型融合 - $DATASET"
    echo "=========================================="
    
    # CLIP + DINO (所有方法)
    echo ""
    echo "训练 CLIP + DINO ($DATASET)..."
    for method in concat weighted_sum mmvit mmvit_lite comm; do
        checkpoint="$OUTPUT_DIR/${DATASET}_clip_dino_${method}.pth"
        if [ -f "$checkpoint" ]; then
            echo "⊘ $method 已存在，跳过"
            continue
        fi
        
        echo "  方法: $method"
        python scripts/train.py \
            --models clip dino \
            --dataset "$DATASET" \
            --method "$method" \
            --epochs 30 \
            --batch-size 256 \
            --feature-dir "$FEATURE_DIR"
        
        echo "  ✓ $method 完成"
    done
    
    # CLIP + MAE
    echo ""
    echo "训练 CLIP + MAE ($DATASET)..."
    for method in concat weighted_sum mmvit mmvit_lite; do
        checkpoint="$OUTPUT_DIR/${DATASET}_clip_mae_${method}.pth"
        if [ -f "$checkpoint" ]; then
            echo "⊘ $method 已存在，跳过"
            continue
        fi
        
        echo "  方法: $method"
        python scripts/train.py \
            --models clip mae \
            --dataset "$DATASET" \
            --method "$method" \
            --epochs 30 \
            --batch-size 256 \
            --feature-dir "$FEATURE_DIR"
        
        echo "  ✓ $method 完成"
    done
    
    # DINO + MAE
    echo ""
    echo "训练 DINO + MAE ($DATASET)..."
    for method in concat weighted_sum mmvit mmvit_lite; do
        checkpoint="$OUTPUT_DIR/${DATASET}_dino_mae_${method}.pth"
        if [ -f "$checkpoint" ]; then
            echo "⊘ $method 已存在，跳过"
            continue
        fi
        
        echo "  方法: $method"
        python scripts/train.py \
            --models dino mae \
            --dataset "$DATASET" \
            --method "$method" \
            --epochs 30 \
            --batch-size 256 \
            --feature-dir "$FEATURE_DIR"
        
        echo "  ✓ $method 完成"
    done
    
    # ===========================
    # 7. 训练三模型
    # ===========================
    echo ""
    echo "Step 7: 训练三模型融合 - $DATASET"
    echo "=========================================="
    
    for method in concat weighted_sum mmvit comm3; do
        checkpoint="$OUTPUT_DIR/${DATASET}_clip_dino_mae_${method}.pth"
        if [ -f "$checkpoint" ]; then
            echo "⊘ $method 已存在，跳过"
            continue
        fi
        
        echo "训练三模型: $method ($DATASET)"
        python scripts/train.py \
            --models clip dino mae \
            --dataset "$DATASET" \
            --method "$method" \
            --epochs 30 \
            --batch-size 256 \
            --feature-dir "$FEATURE_DIR"
        
        echo "✓ $method ($DATASET) 完成"
    done
    
    echo ""
    echo "✓ $DATASET 所有实验完成！"
done

# ===========================
# 总结
# ===========================
echo ""
echo "=========================================="
echo "✓ 所有数据集实验完成！"
echo "=========================================="
echo ""
echo "数据集: ${ALL_DATASETS[@]}"
echo "每个数据集实验数: 20"
echo "总计实验数: $(( ${#ALL_DATASETS[@]} * 20 ))"
echo ""
echo "生成的模型文件："
ls -lh "$OUTPUT_DIR"/*.pth 2>/dev/null | wc -l
echo ""
echo "查看结果："
echo "  ls -lh $OUTPUT_DIR/"

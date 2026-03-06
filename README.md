# 多视图预训练模型融合

基于 [COMM](https://arxiv.org/abs/2310.08825) 和 [MMViT](https://arxiv.org/abs/2305.00104) 的多层特征融合方法。

## 快速开始

```bash
# 安装依赖
pip install torch torchvision openai-clip transformers scipy tqdm pillow

# 1. 提取特征
python scripts/extract.py --model clip --dataset cifar10 --split train
python scripts/extract.py --model clip --dataset cifar10 --split test

# 2. 训练
python scripts/train.py --model clip --dataset cifar10
```

## 一键运行所有实验

我们提供了自动化脚本，可以自动运行所有融合方法的实验。

### 方法 1: 使用 Shell 脚本（推荐）

```bash
# 设置环境变量
export DATA_ROOT="/root/autodl-tmp/data"
export HF_ENDPOINT="https://hf-mirror.com"

# 运行 CIFAR-10 的所有实验
bash scripts/run_cifar10_experiments.sh

# 实验包括：
# - 3 个单模型基线 (CLIP, DINO, MAE)
# - 5 个双模型融合 (CLIP+DINO with concat, weighted_sum, mmvit, mmvit_lite, comm)
# - 4 个三模型融合 (CLIP+DINO+MAE with concat, weighted_sum, mmvit, comm3)
```

### 方法 2: 手动运行完整流程

```bash
# Step 1: 提取单模型特征
for model in clip dino mae; do
    python scripts/extract.py --model $model --dataset cifar10 --split train --output-dir features
    python scripts/extract.py --model $model --dataset cifar10 --split test --output-dir features
done

# Step 2: 提取融合特征
python scripts/extract.py --models clip dino --dataset cifar10 --split train --output-dir features
python scripts/extract.py --models clip dino --dataset cifar10 --split test --output-dir features
python scripts/extract.py --models clip dino mae --dataset cifar10 --split train --output-dir features
python scripts/extract.py --models clip dino mae --dataset cifar10 --split test --output-dir features

# Step 3: 提取 COMM 多层特征
python scripts/extract.py --method comm --dataset cifar10 --split train --output-dir features
python scripts/extract.py --method comm --dataset cifar10 --split test --output-dir features
python scripts/extract.py --method comm3 --dataset cifar10 --split train --output-dir features
python scripts/extract.py --method comm3 --dataset cifar10 --split test --output-dir features

# Step 4: 训练单模型
for model in clip dino mae; do
    python scripts/train.py --model $model --dataset cifar10 --epochs 30 --batch-size 256 --feature-dir features
done

# Step 5: 训练双模型
for method in concat weighted_sum mmvit mmvit_lite comm; do
    python scripts/train.py --models clip dino --dataset cifar10 --method $method --epochs 30 --batch-size 256 --feature-dir features
done

# Step 6: 训练三模型
for method in concat weighted_sum mmvit comm3; do
    python scripts/train.py --models clip dino mae --dataset cifar10 --method $method --epochs 30 --batch-size 256 --feature-dir features
done
```

### 后台运行

```bash
# 使用 tmux 在后台运行（推荐）
tmux new-session -d -s experiments
tmux send-keys -t experiments "bash scripts/run_cifar10_experiments.sh 2>&1 | tee experiment.log" Enter

# 查看进度
tmux attach -t experiments

# 或者查看日志
tail -f experiment.log
```

## 融合方法

### 双模型 (CLIP + DINO)

```bash
# 提取特征
python scripts/extract.py --models clip dino --dataset cifar10 --split train
python scripts/extract.py --models clip dino --dataset cifar10 --split test

# 训练
python scripts/train.py --models clip dino --dataset cifar10 --method concat
python scripts/train.py --models clip dino --dataset cifar10 --method weighted_sum
python scripts/train.py --models clip dino --dataset cifar10 --method mmvit
python scripts/train.py --models clip dino --dataset cifar10 --method comm
```

### 三模型 (CLIP + DINO + MAE)

```bash
# 提取特征
python scripts/extract.py --models clip dino mae --dataset cifar10 --split train
python scripts/extract.py --models clip dino mae --dataset cifar10 --split test

# 训练
python scripts/train.py --models clip dino mae --dataset cifar10 --method concat
python scripts/train.py --models clip dino mae --dataset cifar10 --method mmvit
python scripts/train.py --models clip dino mae --dataset cifar10 --method comm3
```

## 方法对比

| 方法 | 说明 | 论文 |
|------|------|------|
| `concat` | 特征拼接 (基线) | - |
| `weighted_sum` | 可学习权重求和 | - |
| `comm` | 多层特征 + LLN-Layerscale | [COMM](https://arxiv.org/abs/2310.08825) |
| `comm3` | COMM 三模型扩展 | - |
| `mmvit` | 跨视图交叉注意力 | [MMViT](https://arxiv.org/abs/2305.00104) |
| `mmvit_lite` | MMViT 轻量版 | - |

## 架构图

### COMM
```
CLIP (12层) → LLN-Layerscale → v̄₁ ─┐
                                   ├→ [Concat] → Classify
DINO (6层)  → LLN-Layerscale → MLP → v̄₂ ─┘
```

### MMViT
```
CLIP ─┐
      │   Cross-Attention
DINO ─┼→ Q₁↔K₁,K₂, V₁↔V₂ → View₁' ─┐
      │   Q₂↔K₁,K₂, V₁↔V₂ → View₂' ─┼→ [Concat] → Classify
MAE  ─┘   Q₃↔K₁,K₂,K₃, V₁,V₂,V₃ → View₃' ─┘
```

## 数据集

| 数据集 | 类别数 | 自动下载 |
|--------|--------|----------|
| CIFAR-10 | 10 | ✓ |
| CIFAR-100 | 100 | ✓ |
| Flowers-102 | 102 | ✓ |
| Oxford-IIIT Pets | 37 | ✓ |

## 项目结构

```
├── scripts/
│   ├── extract.py      # 特征提取
│   └── train.py        # 训练
├── src/
│   ├── models/         # CLIP, DINO, MAE
│   ├── training/
│   │   └── fusion.py   # 所有融合方法
│   └── data/           # 数据集
├── features/           # 特征文件
└── outputs/            # 模型输出
```

## 参考文献

- **COMM**: Jiang et al., [From CLIP to DINO](https://arxiv.org/abs/2310.08825), ECCV 2024
- **MMViT**: Liu et al., [MMViT: Multiscale Multiview Vision Transformers](https://arxiv.org/abs/2305.00104), 2023
- **CLIP**: Radford et al., ICML 2021
- **DINO**: Caron et al., ICCV 2021
- **MAE**: He et al., CVPR 2022

# Quantifying Representation Reliability

量化表示可靠性的新基准研究项目。

## 核心思想

**原论文问题**：用同一模型（SimCLR/BYOL/MoCo）训练10次（仅随机种子不同），缺乏真正的多样性。

**我们的改进**：使用本质不同的预训练模型（CLIP、DINO、MAE），在架构、预训练方法、训练数据上都不同。

---

## 快速开始

### 1. 创建环境

```bash
conda create -n repreli python=3.10
conda activate repreli
pip install -r requirements.txt
```

### 2. 一键运行

```bash
# Windows
.\scripts\run_all.ps1

# Linux/Mac
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

### 3. 查看结果

```bash
ls results/
```

---

## 流程说明

```
预训练模型 (下载)  →  提取特征  →  训练线性头  →  计算 NC  →  评估相关性
    ↓                  ↓            ↓             ↓           ↓
 CLIP/DINO/MAE     冻结权重     Logistic      k-NN       Kendall's τ
                   (不训练)    Regression    Jaccard
                                (训练)
```

| 步骤 | 操作 | 训练？ |
|------|------|--------|
| 下载模型 | 下载 CLIP/DINO/MAE | ❌ |
| 提取特征 | 用预训练模型 | ❌ |
| 训练线性头 | Logistic Regression | ✅ |
| 计算 NC | k-NN Jaccard | ❌ |
| 评估 | Kendall's τ | ❌ |

---

## 分步执行

```bash
# 1. 下载模型
python scripts/download_models.py --family clip dinov2 mae

# 2. 提取特征
python scripts/extract_features.py \
    --models clip_vit_b16 dinov2_vit_b14 mae_vit_b16 \
    --dataset cifar10 \
    --output ./features

# 3. 评估 (复用 repreli 代码)
python scripts/run_evaluation.py --config configs/default.yaml
```

---

## 项目结构

```
├── repreli/              # 原论文代码 (复用)
│   ├── tasks/            # OVO 下游任务
│   ├── evaluation/       # NC 计算
│   └── utils.py
│
├── src/models/           # 新模型封装
│   ├── clip.py
│   ├── dino.py
│   └── mae.py
│
├── scripts/
│   ├── download_models.py    # 下载预训练模型
│   ├── extract_features.py   # 提取特征
│   ├── run_evaluation.py     # 评估 (复用 repreli)
│   └── run_all.ps1/sh        # 一键运行
│
└── configs/
    └── default.yaml      # 配置
```

---

## 支持的模型

| 模型 | 维度 | 方法 |
|------|------|------|
| `clip_vit_b16` | 512 | 图像-文本对比 |
| `dinov2_vit_b14` | 768 | 自蒸馏 |
| `dinov2_vit_l14` | 1024 | 自蒸馏 |
| `mae_vit_b16` | 768 | 掩码自编码 |
| `dino_vit_b16` | 768 | 自蒸馏 |

---

## 论文评估协议

按原论文设置：
- **NC**: k=100, n_ref=5000
- **下游任务**: OVO 二分类，Logistic Regression
- **评估指标**: Kendall's τ 相关系数
- **下游性能**: negative Brier score

---

## 参考文献

- 目标论文: Park et al., "Quantifying Representation Reliability in Self-Supervised Learning Models", arXiv 2024
- CLIP: Radford et al., 2021
- DINO: Caron et al., 2021
- DINOv2: Oquab et al., 2023
- MAE: He et al., 2022

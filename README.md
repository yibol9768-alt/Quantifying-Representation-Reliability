# 多视图预训练模型融合 - COMM 方法扩展

本项目实现并扩展了 COMM (CLIP and DINO with Multi-level features Merging) 融合方法，支持 CLIP、DINO、MAE 三模型融合。

---

## 项目背景

基于论文: [From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models](https://arxiv.org/abs/2310.08825)

**核心思想**: 利用 Vision Transformer 的多层特征，通过 LLN-Layerscale 融合不同深度的特征，获得更丰富的表示。

**本项目的扩展**:
- **原始 COMM**: CLIP + DINO 双模型融合
- **扩展 COMM3**: CLIP + DINO + MAE 三模型融合

---

## 支持的数据集

| 数据集 | 类别数 | 训练集 | 测试集 | 类型 |
|--------|--------|--------|--------|------|
| **CIFAR-100** | 100 | 50,000 | 10,000 | 通用对象 |
| **Flowers-102** | 102 | 1,020 | 6,149 | 细粒度（花卉） |
| **Oxford-IIIT Pets** | 37 | 3,680 | 3,669 | 细粒度（宠物） |

---

## 快速开始

### 环境要求

```bash
pip install torch torchvision openai-clip transformers scipy tqdm pillow
```

### 2模型 COMM 融合 (CLIP + DINO)

```bash
# 1. 提取多层特征
python scripts/2_extract_comm.py --dataset cifar100 --split train
python scripts/2_extract_comm.py --dataset cifar100 --split test

# 2. 训练 COMM 模型
python scripts/4_train_comm.py --dataset cifar100
```

### 3模型 COMM3 融合 (CLIP + DINO + MAE)

```bash
# 1. 提取多层特征（包含 MAE）
python scripts/2_extract_comm3.py --dataset cifar100 --split train
python scripts/2_extract_comm3.py --dataset cifar100 --split test

# 2. 训练 COMM3 模型
python scripts/4_train_comm3.py --dataset cifar100
```

---

## 模型架构

### COMM 多层融合 (原始)

```
输入图像 (224×224)
    │
    ├──→ [CLIP ViT-B/32]  ─→ 12层特征 → [LLN-Layerscale] → v̄₁ ─┐
    │   (全部 12 层)                                         │
    │                                                        │
    └──→ [DINO ViT-B/16] ─→ 6层特征 → [LLN-Layerscale] → MLP─┼→ [拼接] → 分类器
        (最后 6 层, 7-12)                                    │
                                                              │
                                        冻结预训练权重         可训练参数
```

### COMM3 扩展融合

```
输入图像 (224×224)
    │
    ├──→ [CLIP] ─→ 12层 → [LLN] → v̄₁ ─────────────────────────┐
    │   (全部 12 层)                                         │
    │                                                        │
    ├──→ [DINO] ─→ 6层 → [LLN] → v̄₂ → [MLP对齐] ────────────┤
    │   (7-12 层)                                            │
    │                                                   │
    └──→ [MAE] ─→ 6层 → [LLN] → v̄₃ → [MLP对齐] ────────────┤
        (7-12 层)                                         │
                                                         │
                                   v̄₁ + v̄₂_aligned + v̄₃_aligned
                                                   │
                                              ↓ 分类器
                                        冻结预训练权重    可训练参数
```

**层选择说明**:
- **CLIP**: 全部 12 层 - 图文对比训练使各层都有独特信息
- **DINO**: 最后 6 层 (7-12) - 深层语义特征更鲁棒
- **MAE**: 最后 6 层 (7-12) - 自监督重建，深层特征更语义化

---

## 项目结构

```
Quantifying-Representation-Reliability/
├── src/
│   ├── models/
│   │   ├── clip_multilayer.py     # CLIP 多层特征
│   │   ├── dino_multilayer.py     # DINO 多层特征
│   │   └── mae_multilayer.py      # MAE 多层特征 (新增)
│   └── training/
│       ├── comm_fusion.py         # 2模型 COMM 融合
│       └── comm3_fusion.py        # 3模型 COMM3 融合 (新增)
├── scripts/
│   ├── 2_extract_comm.py          # 提取 2模型多层特征
│   ├── 2_extract_comm3.py         # 提取 3模型多层特征 (新增)
│   ├── 4_train_comm.py            # 训练 2模型 COMM
│   └── 4_train_comm3.py           # 训练 3模型 COMM3 (新增)
├── features/                       # 特征文件目录
├── outputs/checkpoints/            # 模型权重目录
├── CLAUDE.md                       # 开发规范
└── README.md
```

---

## AutoDL 使用

### 数据盘位置

```
/root/autodl-fs/data/    # 持久化数据盘（推荐）
```

### 网络加速

```bash
# 学术网络加速
source /etc/network_turbo

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 参考文献

- **CLIP**: Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021
- **DINO**: Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021
- **MAE**: He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022
- **COMM**: Jiang et al., [From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models](https://arxiv.org/abs/2310.08825), ECCV 2024

---

## 许可证

MIT License

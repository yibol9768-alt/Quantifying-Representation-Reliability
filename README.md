# 多视图预训练模型融合实验

本项目研究如何融合多种不同的视觉预训练模型（CLIP、DINO、MAE）来提升图像分类任务的性能。通过在多个细粒度分类数据集上进行对比实验，验证多视图融合的有效性。

---

## 实验概述

### 研究问题

单一预训练模型往往只能捕捉到图像的某些特定特征。本项目通过融合多个本质不同的预训练模型，探索：

1. **互补性**：不同模型捕捉不同类型的视觉特征，融合是否能获得更好的表示？
2. **性能提升**：多模型融合相比单模型能带来多少准确率提升？
3. **数据集差异**：不同类型的分类任务中，融合效果是否一致？

### 实验设计

| 变量 | 设置 |
|------|------|
| **预训练模型** | CLIP (ViT-B/32), DINO (ViT-B/16), MAE (ViT-Base) |
| **融合策略** | 单模型 / 双模型融合 / 三模型融合 |
| **融合方式** | 早期特征融合 (Early Fusion) |
| **数据集** | CIFAR-100, Flowers-102, Oxford-IIIT Pets |
| **评估指标** | 测试集准确率 (Test Accuracy) |

### 对比实验

| 实验组 | 模型组合 | 特征维度 |
|--------|----------|----------|
| **单模型基线** | CLIP / DINO / MAE | 512 / 768 / 768 |
| **双模型融合** | CLIP+DINO / CLIP+MAE / DINO+MAE | 1280 / 1280 / 1536 |
| **三模型融合** | CLIP+DINO+MAE | 2048 |

---

## 支持的数据集

| 数据集 | 类别数 | 训练集 | 测试集 | 类型 |
|--------|--------|--------|--------|------|
| **CIFAR-100** | 100 | 50,000 | 10,000 | 通用对象 |
| **Flowers-102** | 102 | 1,020 | 6,149 | 细粒度（花卉） |
| **Oxford-IIIT Pets** | 37 | 3,680 | 3,669 | 细粒度（宠物） |

---

## 支持的预训练模型

| 模型 | 特征维度 | 预训练方法 | 特点 |
|------|----------|------------|------|
| **CLIP (ViT-B/32)** | 512 | 图文对比学习 | 丰富的语义知识，零样本能力强 |
| **DINO (ViT-B/16)** | 768 | 自监督知识蒸馏 | 细粒度纹理和形状特征 |
| **MAE (ViT-Base)** | 768 | 掩码自编码器 | 强大的全局表示能力 |

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+ with CUDA
- GPU: 推荐 8GB+ 显存

### 安装依赖

```bash
# 安装 PyTorch (CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install openai-clip transformers scipy tqdm pillow
```

### 克隆项目

```bash
git clone https://github.com/yibol9768-alt/Quantifying-Representation-Reliability.git
cd Quantifying-Representation-Reliability
```

---

## 运行实验

### 方式一：完整流程自动化（推荐）

```bash
# 在 tmux 中运行（防止 SSH 断开）
tmux new-session -s exp

# 运行所有实验
python scripts/run_experiments.py

# 分离 tmux: Ctrl+B, D
# 重新连接: tmux attach -t exp
```

自动化脚本会依次执行：
1. 特征提取（18 个任务：3 数据集 × 3 模型 × 2 分割）
2. 单模型训练（9 个任务）
3. 双模型融合训练（9 个任务）
4. 三模型融合训练（3 个任务）

### 方式二：手动分步运行

```bash
# 1. 提取特征
python scripts/1_extract_single.py --model clip --dataset cifar100 --split train
python scripts/1_extract_single.py --model dino --dataset cifar100 --split train
python scripts/1_extract_single.py --model mae --dataset cifar100 --split train

# 2. 合并特征（用于融合模型）
python scripts/2_extract_multi.py --models clip dino --dataset cifar100 --split train

# 3. 训练单模型
python scripts/3_train_single.py --model clip --dataset cifar100

# 4. 训练融合模型
python scripts/4_train_fusion.py --models clip dino --dataset cifar100
python scripts/4_train_fusion.py --models clip dino mae --dataset cifar100
```

### 监控实验进度

```bash
# 实时监控
python scripts/monitor.py

# 自动监控并上传结果
bash scripts/auto_monitor.sh
```

---

## 项目结构

```
Quantifying-Representation-Reliability/
├── src/                        # 源代码
│   ├── models/                 # 模型封装
│   │   ├── clip_model.py      # CLIP (OpenAI)
│   │   ├── dino_model.py      # DINO (Facebook)
│   │   └── mae_model.py       # MAE (Facebook)
│   ├── data/                   # 多数据集加载器
│   ├── features/               # 特征提取器
│   ├── training/               # 训练相关
│   └── utils/                  # 工具函数
├── scripts/                    # 运行脚本
│   ├── 1_extract_single.py    # 单模型特征提取
│   ├── 2_extract_multi.py     # 多模型特征合并
│   ├── 3_train_single.py      # 单模型训练
│   ├── 4_train_fusion.py      # 融合模型训练
│   ├── run_experiments.py     # 自动化实验流程
│   ├── monitor.py             # 实时监控
│   ├── auto_monitor.sh        # 自动监控并上传
│   └── start_proxy.sh         # 代理配置（国内网络）
├── configs/                    # 配置文件
├── features/                   # 特征文件目录
├── outputs/                    # 训练输出目录
├── CLAUDE.md                   # 开发规范
└── README.md
```

---

## 模型架构

### 融合策略：早期特征融合

```
输入图像 (224×224×3)
    │
    ├──→ [CLIP ViT-B/32] ───→ 512D 特征 ──┐
    │                                   │
    ├──→ [DINO ViT-B/16] ───→ 768D 特征 ──┼→ [拼接] → [MLP 分类器] → N类
    │                                   │
    └──→ [MAE ViT-Base] ───→ 768D 特征 ──┘
         (冻结预训练权重)              (可训练)
```

### MLP 分类器结构

```python
class MultiViewClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 拼接后的特征维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-3 |
| 权重衰减 | 1e-4 |
| Batch Size | 256 |
| Epochs | 30 |
| 学习率调度 | ReduceLROnPlateau |

---

## 实验结果

### CIFAR-100 (100 类)

| 模型 | 测试准确率 | 参数量 |
|------|-----------|--------|
| CLIP | - | - |
| DINO | - | - |
| MAE | - | - |
| CLIP + DINO | - | - |
| CLIP + MAE | - | - |
| DINO + MAE | - | - |
| **CLIP + DINO + MAE** | - | - |

> ⚠️ 实验正在进行中，结果即将更新...

### Flowers-102 (102 类花卉分类)

| 模型 | 测试准确率 |
|------|-----------|
| *(实验中)* | - |

### Oxford-IIIT Pets (37 类宠物分类)

| 模型 | 测试准确率 |
|------|-----------|
| *(实验中)* | - |

---

## TODO

- [x] 搭建项目框架
- [x] 集成 CLIP、DINO、MAE 预训练模型
- [x] 实现特征提取模块
- [x] 实现单模型训练
- [x] 实现多模型融合训练
- [x] 添加自动化实验脚本
- [x] 添加实时监控工具
- [ ] 完成所有数据集实验
- [ ] 整理实验结果
- [ ] 生成对比图表
- [ ] 撰写实验报告

---

## 常见问题

### Q1: 如何配置代理（国内网络）？

```bash
# 启动代理脚本
source scripts/start_proxy.sh

# 或手动设置
export http_proxy=http://127.0.0.1:1081
export https_proxy=http://127.0.0.1:1081
```

### Q2: SSH 断开后实验会停止吗？

使用 tmux 运行实验，断开 SSH 不会影响实验：

```bash
tmux new-session -s exp
python scripts/run_experiments.py
# 按 Ctrl+B 然后 D 分离会话
```

### Q3: 显存不足怎么办？

```bash
# 减小批大小
python scripts/4_train_fusion.py --models clip dino mae --dataset cifar100 --batch-size 128
```

### Q4: 如何只运行部分实验？

修改 `scripts/run_experiments.py` 中的 `DATASETS` 列表：

```python
DATASETS = ["cifar100"]  # 只运行 CIFAR-100
```

---

## 参考文献

- **CLIP**: Radford et al. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021
- **DINO**: Caron et al. [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021
- **MAE**: He et al. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022
- **CIFAR-100**: Krizhevsky et al., 2009
- **Flowers-102**: Nilsback & Zisserman, 2008
- **Oxford-IIIT Pets**: Parkhi et al., 2012

---

## 许可证

MIT License

# 多视图预训练模型融合 - Stanford Cars 细粒度分类

本项目探索如何融合多种不同的视觉预训练模型（CLIP、DINO、MAE）来提升细粒度图像分类任务的性能。通过结合不同架构、不同预训练方法和不同训练数据的模型，实现多视图学习。

---

## 项目背景

### 细粒度图像分类挑战

细粒度图像分类（Fine-grained Image Classification）是计算机视觉中的一个挑战性任务：

- **类别数量多**：Stanford Cars 包含 196 个汽车品牌/型号
- **类间差异小**：不同类别之间的视觉差异非常细微
- **类内差异大**：同一类别内存在姿态、角度、光照等变化

### 多视图学习思想

单一预训练模型往往只能捕捉到图像的某些特定特征。通过融合多个本质不同的模型：

1. **互补性**：不同模型捕捉不同类型的视觉特征
2. **鲁棒性**：减少对单一模型的依赖
3. **性能提升**：融合多个视角的表示通常优于单一模型

---

## 支持的预训练模型

| 模型 | 特征维度 | 预训练方法 | 训练数据 | 特点 |
|------|----------|------------|----------|------|
| **CLIP (ViT-B/32)** | 512 | 图文对比学习 | LAION-400M | 具有丰富的语义知识 |
| **DINO (ViT-B/16)** | 768 | 自监督蒸馏 | ImageNet-1K | 捕捉细粒度纹理和形状 |
| **MAE (ViT-Base)** | 768 | 掩码自编码 | ImageNet-1K | 强大的全局表示能力 |

---

## 项目结构

```
Quantifying-Representation-Reliability/
├── src/                        # 源代码
│   ├── models/                 # 模型封装 (CLIP/DINO/MAE)
│   │   ├── base.py             # 基类
│   │   ├── clip_model.py       # CLIP
│   │   ├── dino_model.py       # DINO
│   │   └── mae_model.py        # MAE
│   ├── data/                   # 数据加载
│   │   └── dataset.py          # Stanford Cars 数据集
│   ├── features/               # 特征提取
│   │   └── extractor.py        # 特征提取器
│   ├── training/               # 训练相关
│   │   ├── classifier.py       # 分类器
│   │   └── trainer.py          # 训练器
│   └── utils/                  # 工具函数
│       └── common.py
├── configs/                    # 配置文件
│   └── config.py
├── scripts/                    # 运行脚本
│   ├── 1_extract_single.py     # 单模型特征提取
│   ├── 2_extract_multi.py      # 多模型特征提取
│   ├── 3_train_single.py       # 单模型训练
│   ├── 4_train_fusion.py       # 融合模型训练
│   └── 5_evaluate.py           # 模型评估
├── main.py                     # 统一入口
├── CLAUDE.md                   # 开发规范
└── README.md
```

---

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n multiview python=3.10
conda activate multiview

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install clip transformers scipy tqdm pillow
```

### 2. 数据准备

```bash
# 下载 Stanford Cars 数据集
mkdir stanford_cars
cd stanford_cars

# 下载数据
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.zip
wget http://ai.stanford.edu/~jkrause/cars/cars_train.zip
wget http://ai.stanford.edu/~jkrause/cars/cars_test.zip
wget https://folk.ntnu.no/haakohu/NNIFTI/2022/cars_test_annos_withlabels.mat

# 解压
unzip car_devkit.zip && unzip cars_train.zip && unzip cars_test.zip
cd ..
```

### 3. 运行实验

```bash
# 方式一：使用统一入口（推荐）
# 完整流程：提取特征 + 训练 + 评估
python main.py --mode full --models clip dino mae

# 方式二：分步执行
# 步骤1：提取特征
python scripts/1_extract_single.py --model clip --split train
python scripts/1_extract_single.py --model dino --split train
python scripts/1_extract_single.py --model mae --split train
python scripts/1_extract_single.py --model clip --split test
python scripts/1_extract_single.py --model dino --split test
python scripts/1_extract_single.py --model mae --split test

# 步骤2：训练单模型基线
python scripts/3_train_single.py --model clip
python scripts/3_train_single.py --model dino
python scripts/3_train_single.py --model mae

# 步骤3：训练融合模型
python scripts/4_train_fusion.py --models clip dino mae

# 步骤4：评估
python scripts/5_evaluate.py --checkpoint outputs/checkpoints/clip_dino_mae_fusion.pth
```

---

## 使用说明

### 单模型实验

验证每个预训练模型的独立性能：

```bash
# 提取 CLIP 特征
python scripts/1_extract_single.py --model clip --split train
python scripts/1_extract_single.py --model clip --split test

# 训练 CLIP 分类器
python scripts/3_train_single.py --model clip

# 评估
python scripts/5_evaluate.py --checkpoint outputs/checkpoints/clip_single.pth --single
```

### 多模型融合实验

```bash
# 提取多模型特征（一次性）
python scripts/2_extract_multi.py --models clip dino mae --split train
python scripts/2_extract_multi.py --models clip dino mae --split test

# 训练融合模型
python scripts/4_train_fusion.py --models clip dino mae

# 评估
python scripts/5_evaluate.py --checkpoint outputs/checkpoints/clip_dino_mae_fusion.pth --models clip dino mae
```

### 双模型融合

```bash
# 只使用 CLIP + DINO
python scripts/2_extract_multi.py --models clip dino --split train
python scripts/2_extract_multi.py --models clip dino --split test
python scripts/4_train_fusion.py --models clip dino
```

---

## 模型架构

### 融合策略：早期特征融合

```
输入图像
    │
    ├──→ [CLIP] ───→ 512D ──┐
    │                      │
    ├──→ [DINO] ───→ 768D ──┼→ [拼接] → [MLP] → 196类
    │                      │
    └──→ [MAE] ───→ 768D ──┘
         (冻结)              (可训练)
```

### MLP 分类器结构

```
输入特征 (融合后)
    ↓
Linear(2048 → 1024)
BatchNorm1d + ReLU + Dropout(0.5)
    ↓
Linear(1024 → 512)
ReLU + Dropout(0.3)
    ↓
Linear(512 → 196)  # 196个汽车类别
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-3 |
| 权重衰减 | 1e-4 |
| Batch Size | 256 |
| Epochs | 30 |
| 损失函数 | CrossEntropyLoss |

---

## 脚本参数说明

### 1_extract_single.py - 单模型特征提取

```
--model    模型类型 (clip/dino/mae)
--split    数据划分 (train/test)
--output   输出路径 (可选)
```

### 2_extract_multi.py - 多模型特征提取

```
--models   模型列表 (空格分隔)
--split    数据划分 (train/test)
--output   输出路径 (可选)
```

### 3_train_single.py - 单模型训练

```
--model        模型类型
--feature-dir  特征目录 (默认: features)
--epochs       训练轮数 (默认: 30)
--batch-size   批大小 (默认: 256)
--lr           学习率 (默认: 1e-3)
```

### 4_train_fusion.py - 融合模型训练

```
--models       模型列表 (空格分隔)
--feature-dir  特征目录
--epochs       训练轮数
--batch-size   批大小
--lr           学习率
```

### 5_evaluate.py - 模型评估

```
--checkpoint   模型权重路径
--models       模型列表 (融合模型需要)
--single       单模型标志
```

---

## 常见问题

### Q1: 显存不足怎么办？

```bash
# 减小批大小
python scripts/4_train_fusion.py --models clip dino mae --batch-size 128
```

### Q2: 特征提取很慢？

- 确保使用了 CUDA：检查输出中的 `Device: cuda`
- 考虑分别提取不同模型的特征，然后并行

### Q3: 如何验证数据集是否正确？

```bash
python -c "from src.data import StanfordCarsDataset; ds = StanfordCarsDataset(); print(ds.verify_dataset())"
```

### Q4: 输出文件在哪里？

```
features/           # 特征文件
outputs/            # 训练输出
└── checkpoints/    # 模型权重
```

---

## 实验流程建议

1. **单模型基线** - 先运行每个模型的独立实验
2. **双模型融合** - 尝试 CLIP+DINO、CLIP+MAE、DINO+MAE
3. **三模型融合** - 使用全部三个模型
4. **结果对比** - 记录各方案的准确率

---

## 参考文献

- **CLIP**: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **DINO**: Caron et al. "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021
- **MAE**: He et al. "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **Stanford Cars**: Krause et al., 2013

---

## 许可证

MIT License

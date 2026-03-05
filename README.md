# 多视图预训练模型融合 - 多数据集验证

本项目探索如何融合多种不同的视觉预训练模型（CLIP、DINO、MAE）来提升图像分类任务的性能。支持多个数据集，验证多视图融合的有效性。

---

## 项目背景

### 多视图学习思想

单一预训练模型往往只能捕捉到图像的某些特定特征。通过融合多个本质不同的模型：

1. **互补性**：不同模型捕捉不同类型的视觉特征
2. **鲁棒性**：减少对单一模型的依赖
3. **性能提升**：融合多个视角的表示通常优于单一模型

---

## 支持的数据集

| 数据集 | 类别数 | 训练集 | 测试集 | 类型 | 说明 |
|--------|--------|--------|--------|------|------|
| **Stanford Cars** | 196 | 8,144 | 8,041 | 细粒度 | 196 个汽车品牌/型号 |
| **CIFAR-10** | 10 | 50,000 | 10,000 | 通用 | 10 个物体类别 |
| **CIFAR-100** | 100 | 50,000 | 10,000 | 通用 | 100 个物体类别 |
| **Flowers-102** | 102 | 1,020 | 6,149 | 细粒度 | 102 种花卉 |
| **Oxford-IIIT Pets** | 37 | 3,680 | 3,669 | 细粒度 | 37 种猫狗品种 |
| **Food-101** | 101 | 75,750 | 25,250 | 细粒度 | 101 种食物 |

### 数据集自动下载

除 Stanford Cars 和 Food-101 外，其他数据集会在首次使用时自动下载。

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
│   ├── data/                   # 多数据集加载
│   ├── features/               # 特征提取
│   ├── training/               # 训练相关
│   └── utils/                  # 工具函数
├── configs/                    # 配置文件
├── scripts/                    # 运行脚本 (1-5 按步骤编号)
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

### 2. 查看可用数据集

```bash
python main.py --mode list
```

### 3. 运行实验

```bash
# 在 CIFAR-10 上运行完整流程
python main.py --mode full --dataset cifar10 --models clip dino mae

# 在 Stanford Cars 上运行
python main.py --mode full --dataset stanford_cars --models clip dino mae

# 在 Flowers-102 上运行
python main.py --mode full --dataset flowers102 --models clip dino
```

---

## 使用说明

### 单数据集完整流程

```bash
# CIFAR-10 示例
# 步骤1：提取特征
python scripts/1_extract_single.py --model clip --dataset cifar10 --split train
python scripts/1_extract_single.py --model dino --dataset cifar10 --split train
python scripts/1_extract_single.py --model clip --dataset cifar10 --split test
python scripts/1_extract_single.py --model dino --dataset cifar10 --split test

# 步骤2：训练单模型
python scripts/3_train_single.py --model clip --dataset cifar10
python scripts/3_train_single.py --model dino --dataset cifar10

# 步骤3：训练融合模型
python scripts/4_train_fusion.py --models clip dino --dataset cifar10

# 步骤4：评估
python scripts/5_evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_fusion.pth --models clip dino --dataset cifar10
```

### 多数据集批量实验

```bash
# 在多个数据集上运行 CLIP 单模型
for dataset in cifar10 cifar100 flowers102 pets; do
    python main.py --mode full --dataset $dataset --models clip
done

# 对比不同融合策略
python main.py --mode full --dataset cifar10 --models clip dino
python main.py --mode full --dataset cifar10 --models clip dino mae
```

---

## 脚本参数说明

### 1_extract_single.py - 单模型特征提取

```
--model    模型类型 (clip/dino/mae)
--dataset  数据集名称
--split    数据划分 (train/test)
--output   输出路径 (可选)
```

### 2_extract_multi.py - 多模型特征提取

```
--models   模型列表 (空格分隔)
--dataset  数据集名称
--split    数据划分 (train/test)
--output   输出路径 (可选)
```

### 3_train_single.py - 单模型训练

```
--model        模型类型
--dataset      数据集名称
--epochs       训练轮数 (默认: 30)
--batch-size   批大小 (默认: 256)
--lr           学习率 (默认: 1e-3)
```

### 4_train_fusion.py - 融合模型训练

```
--models       模型列表 (空格分隔)
--dataset      数据集名称
--epochs       训练轮数
--batch-size   批大小
--lr           学习率
```

### 5_evaluate.py - 模型评估

```
--checkpoint   模型权重路径
--model        单模型类型
--models       模型列表 (融合模型)
--dataset      数据集名称
```

---

## 模型架构

### 融合策略：早期特征融合

```
输入图像
    │
    ├──→ [CLIP] ───→ 512D ──┐
    │                      │
    ├──→ [DINO] ───→ 768D ──┼→ [拼接] → [MLP] → N类
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
Linear(512 → N)  # N = 数据集类别数
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

## 数据准备详情

### Stanford Cars (需要手动下载)

```bash
mkdir -p data/stanford_cars
cd data/stanford_cars

wget http://ai.stanford.edu/~jkrause/cars/car_devkit.zip
wget http://ai.stanford.edu/~jkrause/cars/cars_train.zip
wget http://ai.stanford.edu/~jkrause/cars/cars_test.zip
wget https://folk.ntnu.no/haakohu/NNIFTI/2022/cars_test_annos_withlabels.mat

unzip car_devkit.zip && unzip cars_train.zip && unzip cars_test.zip
```

### Food-101 (需要手动下载)

从 Kaggle 下载并解压到 `data/food-101/`：
https://www.kaggle.com/datasets/dansbecker/food-101

### 其他数据集

首次运行时会自动下载到 `data/` 目录。

---

## 输出文件结构

```
features/                    # 特征文件
├── cifar10_clip_train.pt
├── cifar10_dino_train.pt
├── cifar10_clip_dino_train.pt
└── ...

outputs/                     # 训练输出
└── checkpoints/             # 模型权重
    ├── cifar10_clip_single.pth
    ├── cifar10_dino_single.pth
    ├── cifar10_clip_dino_fusion.pth
    └── ...
```

---

## 实验建议

### 对比实验流程

1. **单模型基线** - 在各数据集上运行单模型实验
2. **双模型融合** - 尝试 CLIP+DINO、CLIP+MAE、DINO+MAE
3. **三模型融合** - 使用全部三个模型
4. **跨数据集对比** - 分析不同数据集上的融合效果

### 预期观察

- **细粒度数据集** (Cars, Flowers, Pets, Food)：融合提升更明显
- **通用数据集** (CIFAR)：CLIP 单模型可能已足够好

---

## 常见问题

### Q1: 显存不足？

```bash
# 减小批大小
python scripts/4_train_fusion.py --models clip dino mae --dataset cifar10 --batch-size 128
```

### Q2: Food-101 数据集下载失败？

从 Kaggle 手动下载：
https://www.kaggle.com/datasets/dansbecker/food-101
解压到 `data/food-101/`

### Q3: 如何对比不同数据集结果？

```bash
# 运行多个数据集
for ds in cifar10 flowers102 pets; do
    python main.py --mode full --dataset $ds --models clip dino
done
```

### Q4: 输出文件在哪里？

```
features/           # 特征文件
outputs/
└── checkpoints/    # 模型权重
```

---

## 参考文献

- **CLIP**: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **DINO**: Caron et al. "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021
- **MAE**: He et al. "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **Stanford Cars**: Krause et al., 2013
- **CIFAR**: Krizhevsky et al., 2009
- **Flowers-102**: Nilsback & Zisserman, 2008
- **Oxford-IIIT Pets**: Parkhi et al., 2012
- **Food-101**: Bossard et al., 2014

---

## 许可证

MIT License

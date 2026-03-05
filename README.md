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

| 数据集 | 类别数 | 训练集 | 测试集 | 类型 | 自动下载 |
|--------|--------|--------|--------|------|----------|
| **Stanford Cars** | 196 | 8,144 | 8,041 | 细粒度 | ❌ |
| **CIFAR-10** | 10 | 50,000 | 10,000 | 通用 | ✅ |
| **CIFAR-100** | 100 | 50,000 | 10,000 | 通用 | ✅ |
| **Flowers-102** | 102 | 1,020 | 6,149 | 细粒度 | ✅ |
| **Oxford-IIIT Pets** | 37 | 3,680 | 3,669 | 细粒度 | ✅ |
| **Food-101** | 101 | 75,750 | 25,250 | 细粒度 | ❌ |

---

## 支持的预训练模型

| 模型 | 特征维度 | 预训练方法 | 特点 |
|------|----------|------------|------|
| **CLIP (ViT-B/32)** | 512 | 图文对比学习 | 丰富的语义知识 |
| **DINO (ViT-B/16)** | 768 | 自监督蒸馏 | 细粒度纹理和形状 |
| **MAE (ViT-Base)** | 768 | 掩码自编码 | 强大的全局表示 |

---

## 快速开始

### 方式一：AutoDL 平台（推荐）

1. **创建实例时选择镜像**：
   - `pytorch-2.3.0-cuda12.1-cudnn8`
   - 或 `pytorch-2.0.1-cuda11.8-cudnn8`

2. **克隆项目并安装依赖**：

```bash
# 进入项目目录
cd /root/autodl-tmp/Quantifying-Representation-Reliability

# 安装依赖
pip install openai-clip transformers scipy tqdm pillow

# 运行项目
python main.py --mode list
```

### 方式二：本地环境

```bash
# 创建虚拟环境
conda create -n multiview python=3.10
conda activate multiview

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openai-clip transformers scipy tqdm pillow
```

---

## 运行实验

### 查看可用数据集

```bash
python main.py --mode list
```

### 在 CIFAR-10 上快速测试

```bash
# 完整流程：提取 + 训练 + 评估
python main.py --mode full --dataset cifar10 --models clip
```

### 多模型融合

```bash
# CLIP + DINO 融合
python main.py --mode full --dataset cifar10 --models clip dino

# 三模型融合
python main.py --mode full --dataset cifar10 --models clip dino mae
```

### 在其他数据集上运行

```bash
# Stanford Cars (数据集需预先下载到 /root/autodl-tmp/data)
python main.py --mode full --dataset stanford_cars --models clip dino

# Flowers-102
python main.py --mode full --dataset flowers102 --models clip dino

# 多数据集批量实验
for ds in cifar10 cifar100 flowers102; do
    python main.py --mode full --dataset $ds --models clip dino
done
```

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
├── setup_env.sh                # 环境安装脚本
├── CLAUDE.md                   # 开发规范
└── README.md
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

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-3 |
| 权重衰减 | 1e-4 |
| Batch Size | 256 |
| Epochs | 30 |

---

## 数据准备

### AutoDL 数据盘位置

数据集应放在 `/root/autodl-tmp/data/`：

```bash
# 在 AutoDL 终端中
mkdir -p /root/autodl-tmp/data
cd /root/autodl-tmp/data

# Stanford Cars 数据集
mkdir stanford_cars && cd stanford_cars
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.zip
wget http://ai.stanford.edu/~jkrause/cars/cars_train.zip
wget http://ai.stanford.edu/~jkrause/cars/cars_test.zip
wget https://folk.ntnu.no/haakohu/NNIFTI/2022/cars_test_annos_withlabels.mat
unzip car_devkit.zip && unzip cars_train.zip && unzip cars_test.zip
```

### 自动下载的数据集

CIFAR-10/100、Flowers-102、Pets 会在首次运行时自动下载。

---

## 输出文件

```
features/                    # 特征文件
├── cifar10_clip_train.pt
├── cifar10_dino_train.pt
└── ...

outputs/                     # 训练输出
└── checkpoints/             # 模型权重
    ├── cifar10_clip_single.pth
    ├── cifar10_dino_single.pth
    └── cifar10_clip_dino_fusion.pth
```

---

## 常见问题

### Q1: AutoDL 环境缺少 pip？

选择带 PyTorch 的镜像，或运行：
```bash
bash setup_env.sh
```

### Q2: 数据盘在哪里？

AutoDL 数据盘：`/root/autodl-tmp/`

### Q3: 显存不足？

```bash
# 减小批大小
python scripts/4_train_fusion.py --models clip dino --dataset cifar10 --batch-size 128
```

---

## 参考文献

- **CLIP**: Radford et al., ICML 2021
- **DINO**: Caron et al., ICCV 2021
- **MAE**: He et al., CVPR 2022
- **Stanford Cars**: Krause et al., 2013

---

## 许可证

MIT License

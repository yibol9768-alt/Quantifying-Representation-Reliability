# Feature Classification with MAE/CLIP/DINO

使用 HuggingFace Transformers 的预训练模型提取特征，配合 MLP 分类器。

## 环境配置

```bash
# 创建环境
conda create -n feature_cls python=3.10
conda activate feature_cls

# 安装依赖
pip install -r requirements.txt
```

## 手动下载

### 1. 下载模型

```bash
# 创建模型目录
mkdir -p models

# MAE (ViT-Base)
huggingface-cli download facebook/vit-mae-base --local-dir models/vit-mae-base

# CLIP (ViT-B/16)
huggingface-cli download openai/clip-vit-base-patch16 --local-dir models/clip-vit-base-patch16

# DINOv2 (ViT-Base)
huggingface-cli download facebook/dinov2-base --local-dir models/dinov2-base
```

或者使用镜像：

```bash
# 设置镜像 (国内)
export HF_ENDPOINT=https://hf-mirror.com

# 然后下载
huggingface-cli download facebook/vit-mae-base --local-dir models/vit-mae-base
```

### 2. 下载数据集

数据集放到 `data/` 目录，格式如下：

```
data/
├── cifar10/
│   ├── train/
│   │   ├── airplane/
│   │   ├── automobile/
│   │   └── ...
│   └── test/
│       ├── airplane/
│       └── ...
├── cifar100/
│   ├── train/
│   └── test/
└── flowers102/
    ├── train/
    └── test/
```

#### 数据集下载链接

| 数据集 | 下载地址 | 格式转换 |
|--------|----------|----------|
| CIFAR-10 | [官网](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | 需转换为图像文件夹 |
| CIFAR-100 | [官网](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) | 需转换为图像文件夹 |
| STL-10 | [官网](https://cs.stanford.edu/~acoates/stl10/) | 需转换为图像文件夹 |
| Tiny ImageNet | [下载](http://cs231n.stanford.edu/tiny-imagenet-200.zip) | 解压即可 |
| Flowers-102 | [官网](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | 需整理为 train/test |
| Food-101 | [官网](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) | 解压即可 |
| Oxford Pets | [官网](https://www.robots.ox.ac.uk/~vgg/data/pets/) | 需整理 |
| CUB-200 | [官网](https://data.caltech.edu/records/65de2-v2p15) | 需整理 |

#### CIFAR 转换脚本

```python
# convert_cifar.py
import torchvision
from PIL import Image
from pathlib import Path

def convert_cifar(name, data_dir, output_dir):
    """Convert CIFAR to image folders."""
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=False)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=False)

    for split, dataset in [("train", train_set), ("test", test_set)]:
        split_dir = Path(output_dir) / name / split
        for i in range(len(dataset)):
            img, label = dataset[i]
            class_name = dataset.classes[label]
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img.save(class_dir / f"{i}.png")

    print(f"Converted {name} to {output_dir}/{name}")

# 使用
convert_cifar("cifar10", "./raw_data", "./data")
```

## 目录结构

```
project/
├── main.py
├── requirements.txt
├── README.md
├── configs/
│   └── config.py
├── src/
│   ├── models/
│   │   ├── extractors.py    # 模型加载 (离线)
│   │   └── mlp.py           # 分类器
│   ├── data/
│   │   └── hf_dataset.py    # 数据加载
│   └── training/
│       └── hf_trainer.py    # 训练器
├── models/                   # 预训练模型 (手动下载)
│   ├── vit-mae-base/
│   ├── clip-vit-base-patch16/
│   └── dinov2-base/
└── data/                     # 数据集 (手动下载)
    ├── cifar10/
    ├── cifar100/
    └── ...
```

## 使用方法

```bash
# 基本用法
python main.py --dataset cifar100 --model mae --epochs 50

# 预计算特征 (更快)
python main.py --dataset cifar100 --model mae --precompute --epochs 50

# 融合模型
python main.py --dataset cifar100 --model fusion --epochs 50

# 完整参数
python main.py \
    --dataset cifar100 \
    --model mae \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 128 \
    --hidden_dim 512 \
    --device cuda:0 \
    --precompute
```

## 模型说明

| 模型 | 参数 | 特征维度 | 本地路径 |
|------|------|----------|----------|
| MAE | `--model mae` | 768 | `models/vit-mae-base` |
| CLIP | `--model clip` | 512 | `models/clip-vit-base-patch16` |
| DINO | `--model dino` | 768 | `models/dinov2-base` |
| Fusion | `--model fusion` | 2048 | 拼接以上三个 |

## Fusion 实现

```python
# 简单拼接 + L2 归一化
features = []
for name in ["mae", "clip", "dino"]:
    feat = extractors[name](images)
    feat = feat / feat.norm(dim=-1, keepdim=True)  # 归一化
    features.append(feat)

fused = torch.cat(features, dim=-1)  # 768+512+768=2048
```

## 预期结果

| 数据集 | MAE | CLIP | DINO | Fusion |
|--------|-----|------|------|--------|
| CIFAR-10 | ~95% | ~96% | ~95% | ~97% |
| CIFAR-100 | ~75% | ~82% | ~78% | ~85% |
| Flowers-102 | ~85% | ~90% | ~88% | ~93% |

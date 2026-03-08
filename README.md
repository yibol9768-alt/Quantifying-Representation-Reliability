# Feature Classification with MAE/CLIP/DINO

使用 MAE、CLIP、DINO 预训练模型提取特征，配合 MLP 分类器在多个数据集上进行训练。

## 项目结构

```
├── README.md
├── main.py              # 主入口
├── requirements.txt     # 依赖
├── configs/
│   └── config.py        # 配置文件
└── src/
    ├── data/
    │   └── dataset.py   # 多数据集加载
    ├── models/
    │   ├── mae.py       # MAE 特征提取
    │   ├── clip.py      # CLIP 特征提取
    │   └── dino.py      # DINO 特征提取
    └── training/
        ├── mlp.py       # MLP 分类器
        └── trainer.py   # 训练器
```

## 环境配置

```bash
# 创建虚拟环境
conda create -n feature_cls python=3.10
conda activate feature_cls

# 安装依赖
pip install -r requirements.txt
```

## 支持的数据集

### 自动下载

| 数据集 | 参数 | 类别数 | 描述 |
|--------|------|--------|------|
| CIFAR-10 | `--dataset cifar10` | 10 | 简单图像分类 |
| CIFAR-100 | `--dataset cifar100` | 100 | 标准benchmark |
| STL-10 | `--dataset stl10` | 10 | 自监督评估 |
| Flowers-102 | `--dataset flowers102` | 102 | 花卉细粒度 |
| Food-101 | `--dataset food101` | 101 | 食物分类 |
| Oxford Pets | `--dataset pets` | 37 | 宠物细粒度 |
| Caltech-101 | `--dataset caltech101` | 101 | 物体识别 |

### 手动下载

| 数据集 | 参数 | 类别数 | 下载地址 |
|--------|------|--------|----------|
| Tiny ImageNet | `--dataset tiny_imagenet` | 200 | [下载](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |
| CUB-200 | `--dataset cub200` | 200 | [下载](https://data.caltech.edu/records/65de2-v2p15) |

手动下载的数据集请解压到 `data/` 目录：
```
data/
├── tiny-imagenet-200/
│   ├── train/
│   └── val/
└── CUB_200_2011/
    └── images/
```

## 使用方法

### 1. 查看可用数据集

```bash
python main.py --list_datasets
```

### 2. 单模型训练

```bash
# CIFAR-10 + MAE
python main.py --dataset cifar10 --model mae --epochs 50

# CIFAR-100 + CLIP
python main.py --dataset cifar100 --model clip --epochs 50

# Flowers-102 + DINO
python main.py --dataset flowers102 --model dino --epochs 50

# Food-101 + Fusion (全部特征融合)
python main.py --dataset food101 --model fusion --epochs 50
```

### 3. 完整参数

```bash
python main.py \
    --dataset cifar100 \   # 数据集
    --model mae \          # 模型: mae/clip/dino/fusion
    --epochs 50 \          # 训练轮数
    --lr 0.001 \           # 学习率
    --batch_size 128 \     # 批大小
    --hidden_dim 512 \     # MLP 隐藏层维度
    --device cuda:0        # GPU 设备
```

### 4. 批量实验脚本

```bash
# 运行所有数据集 + 所有模型
for dataset in cifar10 cifar100 stl10 flowers102 food101 pets; do
    for model in mae clip dino fusion; do
        python main.py --dataset $dataset --model $model --epochs 50
    done
done
```

## 模型说明

| 模型 | 特征维度 | 预训练来源 |
|------|----------|------------|
| MAE (ViT-B) | 768 | ImageNet |
| CLIP (ViT-B/16) | 512 | ImageNet + 文本 |
| DINO (ViT-B/16) | 768 | ImageNet |
| Fusion (拼接) | 2048 | - |

## 预期结果

| 数据集 | MAE | CLIP | DINO | Fusion |
|--------|-----|------|------|--------|
| CIFAR-10 | ~95% | ~96% | ~95% | ~97% |
| CIFAR-100 | ~75% | ~82% | ~78% | ~85% |
| STL-10 | ~97% | ~98% | ~97% | ~99% |
| Flowers-102 | ~85% | ~90% | ~88% | ~93% |
| Food-101 | ~70% | ~80% | ~75% | ~82% |
| Oxford Pets | ~85% | ~90% | ~88% | ~92% |

## 注意事项

1. 首次运行会自动下载预训练模型（约 1-2GB）
2. 需要至少 8GB 显存
3. 部分数据集（Tiny ImageNet, CUB-200）需手动下载
4. 数据集会自动下载到 `data/` 目录

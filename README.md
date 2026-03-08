# Feature Classification with MAE/CLIP/DINO

使用 HuggingFace Transformers 的预训练模型提取特征，配合 MLP 分类器。

## 快速开始 (AutoDL 服务器)

```bash
# 1. 克隆代码
git clone https://github.com/yibol9768-alt/Quantifying-Representation-Reliability.git
cd Quantifying-Representation-Reliability
git checkout test

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型和数据集 (首次运行)
python download_models.py --all

# 4. 开始训练
python main.py --dataset cifar100 --model mae --epochs 50 --precompute --fp16
```

## 环境配置

```bash
# 创建环境
conda create -n feature_cls python=3.10
conda activate feature_cls

# 安装依赖
pip install -r requirements.txt
```

## 一键下载

```bash
# 下载所有模型和数据集
python download_models.py --all

# 只下载模型
python download_models.py --models

# 只下载 CIFAR-100
python download_models.py --cifar100

# 只下载 CIFAR-10
python download_models.py --cifar10
```

## 手动下载 (可选)

### 模型

```bash
# 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download facebook/vit-mae-base --local-dir models/vit-mae-base
huggingface-cli download openai/clip-vit-base-patch16 --local-dir models/clip-vit-base-patch16
huggingface-cli download facebook/dinov2-base --local-dir models/dinov2-base
```

### 数据集格式

```
data/
├── cifar10/
│   ├── train/
│   │   ├── airplane/
│   │   └── ...
│   └── test/
├── cifar100/
│   ├── train/
│   └── test/
└── ...
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

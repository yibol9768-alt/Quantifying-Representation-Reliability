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
python main.py --dataset cifar100 --model mae --epochs 50 --fp16
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

# 在线提特征训练 (仅调试时使用)
python main.py --dataset cifar100 --model mae --no_precompute --epochs 50

# 融合: concat (2 模型)
python main.py --dataset cifar100 --model fusion \
    --fusion_method concat --fusion_models clip,dino --epochs 50

# 融合: COMM token-level (2 模型, 默认离线缓存)
python main.py --dataset cifar100 --model fusion \
    --fusion_method comm --fusion_models clip,dino --epochs 50

# 融合: MMViT token-level (3 模型, 默认离线缓存)
python main.py --dataset cifar100 --model fusion \
    --fusion_method mmvit --fusion_models mae,clip,dino --epochs 50

# 横向对比（推荐）：三种方法同维度、同训练协议、同seed
python main.py --dataset cifar100 --model fusion \
    --fusion_method concat --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 --epochs 50
python main.py --dataset cifar100 --model fusion \
    --fusion_method comm --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 --epochs 50
python main.py --dataset cifar100 --model fusion \
    --fusion_method mmvit --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 --epochs 50

# 完整参数
python main.py \
    --dataset cifar100 \
    --model fusion \
    --fusion_method mmvit \
    --fusion_models mae,clip,dino \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 128 \
    --hidden_dim 512 \
    --device cuda:0 \
    --cache_dir ./cache/offline \
    --cache_dtype fp32 \
    --cleanup_cache \
    --fp16
```

## 模型说明

| 模型 | 参数 | 特征维度 | 本地路径 |
|------|------|----------|----------|
| MAE | `--model mae` | 768 | `models/vit-mae-base` |
| CLIP | `--model clip` | 768 | `models/clip-vit-base-patch16` |
| DINO | `--model dino` | 768 | `models/dinov2-base` |
| Fusion | `--model fusion` | 取决于融合方法与模型数 | 支持 2 模型 / 3 模型 |

## Fusion 方法

| 方法 | 参数 | 粒度 | 说明 |
|------|------|------|------|
| Concat | `--fusion_method concat` | 全局特征 | L2 归一化后拼接 |
| COMM | `--fusion_method comm` | Token 级 | 严格复现 `LLN+LayerScale`（CLIP 全层、DINO 深层）+ DINO MLP 对齐 + 最终线性投影 |
| MMViT | `--fusion_method mmvit` | Token 级 | 严格复现 4-stage 16-block 结构（`[0,0,9,1]`）、cross-attn 与 scaled self-attn |

> 默认开启融合横向对比模式（harmonization）：  
> 1) 三种方法统一输出维度 `--fusion_output_dim`；  
> 2) 三种方法统一离线缓存训练协议；  
> 3) 统一随机种子 `--seed`。  
> 若需要关闭，使用 `--disable_fusion_harmonization`。

## 离线缓存

默认训练流程会先把 frozen backbone 的输出写入 `--cache_dir`，然后只从缓存训练后续模块和 MLP。

```bash
# 默认使用 fp32 离线缓存
python main.py --dataset cifar100 --model fusion \
    --fusion_method comm --fusion_models clip,dino \
    --cache_dir ./cache/offline --cache_dtype fp32

# 强制重建缓存
python main.py --dataset cifar100 --model dino --rebuild_cache

# 训练完成后删除缓存文件
python main.py --dataset cifar10 --model clip --cleanup_cache
```

## 预期结果

| 数据集 | MAE | CLIP | DINO | Fusion |
|--------|-----|------|------|--------|
| CIFAR-10 | ~95% | ~96% | ~95% | ~97% |
| CIFAR-100 | ~75% | ~82% | ~78% | ~85% |
| Flowers-102 | ~85% | ~90% | ~88% | ~93% |

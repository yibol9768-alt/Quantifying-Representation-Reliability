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
python download_models.py --all --storage_dir /path/to/bigfiles

# 4. 开始训练
python main.py --dataset cifar100 --model mae \
    --storage_dir /path/to/bigfiles \
    --epochs 50 --batch_size 128 --cache_dtype fp32
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
python download_models.py --all --storage_dir /path/to/bigfiles

# 只下载模型
python download_models.py --models --storage_dir /path/to/bigfiles

# 只下载 CIFAR-100
python download_models.py --cifar100 --storage_dir /path/to/bigfiles

# 只下载 CIFAR-10
python download_models.py --cifar10 --storage_dir /path/to/bigfiles
```

## 手动下载 (可选)

### 模型

```bash
# 国内镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
huggingface-cli download facebook/vit-mae-base --local-dir /path/to/bigfiles/models/vit-mae-base
huggingface-cli download openai/clip-vit-base-patch16 --local-dir /path/to/bigfiles/models/clip-vit-base-patch16
huggingface-cli download facebook/dinov2-base --local-dir /path/to/bigfiles/models/dinov2-base
```

### 数据集格式

```
<storage_dir>/data/
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

如果不传 `--storage_dir`，则默认回落到仓库内的 `./data`。

## 目录结构

代码仓库：

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
│   └── training/             # 训练与离线缓存
│       ├── hf_trainer.py
│       └── offline_cache.py
└── ...
```

大文件目录（推荐通过 `--storage_dir` 指定）：

```text
<storage_dir>/
├── models/
│   ├── vit-mae-base/
│   ├── clip-vit-base-patch16/
│   └── dinov2-base/
├── data/
│   ├── cifar10/
│   └── cifar100/
├── data_raw/
└── cache/
    └── offline/
```

## CIFAR-100 运行命令

默认推荐：

```bash
--storage_dir /path/to/bigfiles --epochs 50 --batch_size 128 --cache_dtype fp32
```

首次运行会先构建离线缓存；之后只要不加 `--rebuild_cache`，并且不加 `--cleanup_cache`，同配置会直接复用缓存。
默认不启用 `--fp16`，也就是全精度训练与缓存；如果显存或速度有压力，再额外加 `--fp16`。
`--storage_dir` 会统一控制大文件根目录：

```text
/path/to/bigfiles/
├── models/
├── data/
└── cache/
```

### 单模型

```bash
# MAE
python main.py --dataset cifar100 --model mae \
    --storage_dir /path/to/bigfiles \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# CLIP
python main.py --dataset cifar100 --model clip \
    --storage_dir /path/to/bigfiles \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# DINO
python main.py --dataset cifar100 --model dino \
    --storage_dir /path/to/bigfiles \
    --epochs 50 --batch_size 128 --cache_dtype fp32
```

### 多模型融合

```bash
# Concat: clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method concat --fusion_models clip,dino \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# Concat: mae + clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method concat --fusion_models mae,clip,dino \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# COMM: clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models clip,dino \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# COMM: mae + clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models mae,clip,dino \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# MMViT: clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method mmvit --fusion_models clip,dino \
    --epochs 50 --batch_size 128 --cache_dtype fp32

# MMViT: mae + clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method mmvit --fusion_models mae,clip,dino \
    --epochs 50 --batch_size 128 --cache_dtype fp32
```

### 横向对比

```bash
# 三种融合方法统一输出维度、统一 seed
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method concat --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 \
    --epochs 50 --batch_size 128 --cache_dtype fp32

python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 \
    --epochs 50 --batch_size 128 --cache_dtype fp32

python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method mmvit --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 \
    --epochs 50 --batch_size 128 --cache_dtype fp32
```

### 常用附加选项

```bash
# 强制重建缓存
--rebuild_cache

# 训练结束后删除缓存文件
--cleanup_cache

# 关闭离线缓存，改为在线提特征
--no_precompute

# 开启混合精度加速
--fp16
```

## 模型说明

| 模型 | 参数 | 特征维度 | 本地路径 |
|------|------|----------|----------|
| MAE | `--model mae` | 768 | `<storage_dir>/models/vit-mae-base` |
| CLIP | `--model clip` | 768 | `<storage_dir>/models/clip-vit-base-patch16` |
| DINO | `--model dino` | 768 | `<storage_dir>/models/dinov2-base` |
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
# 统一指定大文件目录
python main.py --dataset cifar100 --model clip \
    --storage_dir /path/to/bigfiles \
    --epochs 50 --cache_dtype fp32

# 默认使用 fp32 离线缓存
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models clip,dino \
    --cache_dtype fp32

# 强制重建缓存
python main.py --dataset cifar100 --model dino \
    --storage_dir /path/to/bigfiles --rebuild_cache

# 训练完成后删除缓存文件
python main.py --dataset cifar10 --model clip \
    --storage_dir /path/to/bigfiles --cleanup_cache
```

## 预期结果

| 数据集 | MAE | CLIP | DINO | Fusion |
|--------|-----|------|------|--------|
| CIFAR-10 | ~95% | ~96% | ~95% | ~97% |
| CIFAR-100 | ~75% | ~82% | ~78% | ~85% |
| Flowers-102 | ~85% | ~90% | ~88% | ~93% |

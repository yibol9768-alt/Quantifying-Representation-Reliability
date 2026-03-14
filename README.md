# Quantifying Representation Reliability

研究多模型特征融合中的表征冗余与互补性问题。核心发现：**更多模型 ≠ 更好性能**，存在最优模型子集，且该子集是任务依赖的。

## 快速开始

```bash
# 环境安装
conda create -n feature_cls python=3.10 -y && conda activate feature_cls
pip install -r requirements.txt

# 设置大文件目录
export STORAGE_DIR=/path/to/bigfiles

# 下载模型和数据
python download_models.py --models --storage_dir "$STORAGE_DIR"
python download_models.py --all_datasets --storage_dir "$STORAGE_DIR"
```

## 使用方法

### 单模型训练

```bash
python main.py --dataset cifar100 --model clip \
    --storage_dir "$STORAGE_DIR" --epochs 10
```

### 多模型融合

```bash
python main.py --dataset cifar100 --model fusion \
    --fusion_method gated --fusion_models clip,dino \
    --storage_dir "$STORAGE_DIR" --epochs 10
```

### 动态路由

```bash
python main.py --dataset cifar100 --model fusion \
    --fusion_method moe_router \
    --fusion_models clip,dino,mae,siglip,convnext,data2vec \
    --storage_dir "$STORAGE_DIR" --epochs 10
```

## 支持的模型 (10)

| 模型 | 参数 | 类别 |
|------|------|------|
| ViT | `vit` | Vision Transformer |
| Swin | `swin` | 层级 ViT |
| BEiT | `beit` | 自监督 ViT |
| Data2Vec | `data2vec` | 自监督 |
| MAE | `mae` | 自监督 |
| DINOv2 | `dino` | 自监督 |
| CLIP | `clip` | 图文对齐 |
| OpenCLIP | `openclip` | 图文对齐 |
| SigLIP | `siglip` | 图文对齐 |
| ConvNeXt | `convnext` | 现代 CNN |

## 融合方法 (17)

所有方法通过 `--fusion_method` 参数选择。

| 类别 | 方法 | 参数值 |
|------|------|--------|
| **简单融合** | Concat / Projected Concat / Weighted Sum / Gated | `concat` `proj_concat` `weighted_sum` `gated` |
| **交互融合** | Difference Concat / Hadamard / Bilinear | `difference_concat` `hadamard_concat` `bilinear_concat` |
| **注意力融合** | FiLM / Context Gating / LMF / SE-Fusion / Late Fusion | `film` `context_gating` `lmf` `se_fusion` `late_fusion` |
| **Token 级** | COMM / MMViT | `comm` `mmvit` |
| **动态路由** | Top-K Router / MoE Router / Attention Router | `topk_router` `moe_router` `attention_router` |

## 支持的数据集 (19)

CIFAR-10/100, STL-10, Tiny ImageNet, Caltech-101, Flowers-102, Food-101, Pets, CUB-200, MNIST, SVHN, SUN397, Stanford Cars, DTD, EuroSAT, GTSRB, Country211, Aircraft, RESISC45

## 项目结构

```
├── main.py                    # 入口：参数解析 + 调度
├── download_models.py         # 模型/数据下载
├── configs/config.py          # 数据集与实验配置
├── src/
│   ├── models/
│   │   ├── extractor.py       # 基础特征提取器
│   │   ├── classifier.py      # MLP 分类器
│   │   └── fusion/            # 17 种融合方法
│   │       ├── __init__.py    # 工厂函数 get_extractor()
│   │       ├── simple.py      # concat, proj_concat, weighted_sum, gated
│   │       ├── interaction.py # difference, hadamard, bilinear
│   │       ├── attention.py   # film, context_gating, lmf, se_fusion
│   │       ├── token.py       # COMM, MMViT
│   │       └── routing.py     # topk_router, moe_router, attention_router
│   ├── data/dataset.py        # 数据加载与 few-shot 采样
│   └── training/
│       ├── trainer.py         # 统一训练循环
│       ├── cache.py           # 离线特征缓存
│       └── results.py         # 结果记录 (JSON/CSV)
└── experiments/               # 实验脚本
```

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--storage_dir` | - | 大文件根目录 |
| `--fusion_method` | concat | 融合方法 |
| `--fusion_models` | mae,clip,dino | 融合模型列表 |
| `--epochs` | 10 | 训练轮数 |
| `--fewshot_min/max` | 10/10 | 每类训练样本数 |
| `--disable_fewshot` | - | 使用完整训练集 |
| `--fp16` | - | 混合精度训练 |
| `--cache_dtype` | fp32 | 缓存精度 |
| `--no_precompute` | - | 在线提取特征 |

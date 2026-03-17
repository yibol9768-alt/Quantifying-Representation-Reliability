# Model Zoo: 24 预训练模型下载指南

本项目使用 24 个冻结预训练视觉模型进行多模型特征融合实验。所有模型通过 HuggingFace 下载，运行时以离线模式加载。

## 快速开始

### 环境要求

```bash
pip install transformers torch huggingface_hub
```

### 一键下载全部模型

```bash
bash scripts/download_models.sh ./models
```

### 下载指定模型（按需）

```bash
# 只下载原始 6 个模型（最小实验集）
bash scripts/download_models.sh ./models --original

# 下载推荐的 15 个模型（平衡覆盖度和磁盘用量）
bash scripts/download_models.sh ./models --recommended

# 下载全部 25 个模型
bash scripts/download_models.sh ./models --all
```

---

## 模型一览表

### Vision Transformer（监督学习）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `vit` | `google/vit-base-patch16-224` | 768 | 86M | ~330MB |
| `vit_large` | `google/vit-large-patch16-224` | 1024 | 304M | ~1.2GB |

### DeiT（数据高效 ViT）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `deit_small` | `facebook/deit-small-patch16-224` | 384 | 22M | ~90MB |
| `deit_base` | `facebook/deit-base-patch16-224` | 768 | 86M | ~330MB |

### Swin Transformer

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `swin_tiny` | `microsoft/swin-tiny-patch4-window7-224` | 768 | 28M | ~110MB |
| `swin` | `microsoft/swin-base-patch4-window7-224` | 1024 | 88M | ~340MB |

### BEiT

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `beit` | `microsoft/beit-base-patch16-224-pt22k` | 768 | 86M | ~330MB |
| `beit_large` | `microsoft/beit-large-patch16-224-pt22k` | 1024 | 304M | ~1.2GB |

### Data2Vec Vision

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `data2vec` | `facebook/data2vec-vision-base` | 768 | 86M | ~330MB |

### MAE（自监督）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `mae` | `facebook/vit-mae-base` | 768 | 86M | ~330MB |
| `mae_large` | `facebook/vit-mae-large` | 1024 | 304M | ~1.2GB |

### DINOv2（自监督）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `dinov2_small` | `facebook/dinov2-small` | 384 | 22M | ~90MB |
| `dino` | `facebook/dinov2-base` | 768 | 86M | ~330MB |
| `dinov2_large` | `facebook/dinov2-large` | 1024 | 304M | ~1.2GB |

### CLIP（对比语言-图像）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `clip_base32` | `openai/clip-vit-base-patch32` | 768 | 86M | ~340MB |
| `clip` | `openai/clip-vit-base-patch16` | 768 | 86M | ~340MB |
| `clip_large` | `openai/clip-vit-large-patch14` | 1024 | 304M | ~1.2GB |
| `openclip` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 768 | 86M | ~340MB |

### SigLIP

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `siglip` | `google/siglip-base-patch16-224` | 768 | 86M | ~330MB |

### ConvNeXt（现代 CNN）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `convnext_tiny` | `facebook/convnext-tiny-224` | 768 | 28M | ~110MB |
| `convnext` | `facebook/convnext-base-224` | 1024 | 88M | ~340MB |
| `convnext_large` | `facebook/convnext-large-224` | 1536 | 197M | ~760MB |

### ResNet（经典 CNN 基线）

| 模型名 | HuggingFace ID | 特征维度 | 参数量 | 磁盘 |
|--------|----------------|---------|-------|------|
| `resnet50` | `microsoft/resnet-50` | 2048 | 25M | ~100MB |
| `resnet101` | `microsoft/resnet-101` | 2048 | 44M | ~175MB |

---

## 磁盘用量估算

| 模型集合 | 模型数 | 总磁盘 |
|---------|-------|-------|
| Original（原始 6 个） | 6 | ~2.0 GB |
| Recommended（推荐 15 个） | 15 | ~5.5 GB |
| All（全部 24 个） | 24 | ~10.5 GB |

---

## 手动下载单个模型

如果自动脚本失败或需要单独下载：

```bash
# 通用格式
huggingface-cli download <HF_ID> --local-dir ./models/<LOCAL_PATH>

# 示例
huggingface-cli download openai/clip-vit-base-patch16 --local-dir ./models/clip-vit-base-patch16
huggingface-cli download facebook/dinov2-large --local-dir ./models/dinov2-large
huggingface-cli download microsoft/resnet-50 --local-dir ./models/resnet-50
```

如果在中国大陆，可以使用镜像：

```bash
# 设置 HuggingFace 镜像（hf-mirror.com）
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常下载
huggingface-cli download openai/clip-vit-base-patch16 --local-dir ./models/clip-vit-base-patch16
```

---

## 验证下载

```bash
# 验证所有模型是否已下载
python -c "
from src.models.extractor import FeatureExtractor
import os

model_dir = './models'
for name, cfg in FeatureExtractor.MODEL_PATHS.items():
    path = os.path.join(model_dir, cfg['path'])
    status = 'OK' if os.path.isdir(path) else 'MISSING'
    print(f'  [{status:7s}] {name:16s} -> {path}')
"
```

---

## 模型分组说明

本项目的 25 个模型覆盖以下维度，确保模型选择实验的全面性：

### 按训练范式
- **监督学习**：vit, vit_large, deit_small, deit_base, swin_tiny, swin, resnet50, resnet101
- **自监督学习**：mae, mae_large, dino, dinov2_small, dinov2_large, data2vec, beit, beit_large
- **对比学习（语言-图像）**：clip, clip_base32, clip_large, openclip, siglip
- **现代 CNN**：convnext_tiny, convnext, convnext_large

### 按模型规模
- **Small（≤384d）**：dinov2_small, deit_small — 弱模型，应被选择算法过滤
- **Base（768d）**：vit, deit_base, swin_tiny, beit, data2vec, mae, dino, clip, clip_base32, openclip, siglip, convnext_tiny — 主力模型
- **Large（1024-1536d）**：vit_large, swin, beit_large, mae_large, dinov2_large, clip_large, convnext, convnext_large — 强模型
- **CNN（2048d）**：resnet50, resnet101 — 经典基线，特征维度高但信息量可能不如 ViT

### 冗余组（预期 CKA 较高）
- CLIP 家族：clip, clip_base32, clip_large, openclip — 应被冗余检测识别
- DINOv2 家族：dinov2_small, dino, dinov2_large — 同一框架不同规模
- ConvNeXt 家族：convnext_tiny, convnext, convnext_large

这种设计使得：
1. **模型选择成为必要**：24 个模型穷举 $2^{24} \approx 1600$ 万种子集，无法暴力搜索
2. **弱模型需要过滤**：Random 和 All Models 基线会被拖垮
3. **冗余检测需要生效**：同家族多个模型不应全部入选
4. **方法差异被放大**：选择策略的优劣在大模型池中更加明显

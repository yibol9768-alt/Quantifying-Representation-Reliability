# Feature Classification with Pretrained Vision Models

使用 HuggingFace Transformers 的预训练模型提取特征，配合 MLP 分类器。支持15个预训练视觉模型和19个数据集。

## 国内环境推荐流程

下面这套流程按中国大陆网络环境写，目标是：

1. 依赖安装尽量走国内镜像
2. Hugging Face 模型下载走镜像或本地代理
3. 数据、模型、缓存统一放到一个大目录里

先选一个大文件目录，下面统一用：

```bash
export STORAGE_DIR=/path/to/bigfiles
mkdir -p "$STORAGE_DIR"
```

### 第 1 步：先开代理或镜像

如果你本机或服务器已经有代理，先在当前终端导出。把下面的 `127.0.0.1:7890` 替换成你自己的代理地址：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

Hugging Face 模型建议再加一个国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果你没有代理，只想先试镜像，也可以只执行上一条 `HF_ENDPOINT`。

### 第 2 步：克隆代码

```bash
git clone --branch test --single-branch \
    https://github.com/yibol9768-alt/Quantifying-Representation-Reliability.git
cd Quantifying-Representation-Reliability
```

### 第 3 步：创建环境并安装依赖

推荐直接用下面这一组命令。`pip` 走清华镜像，通常比默认源稳。

```bash
conda create -n feature_cls python=3.10 -y
conda activate feature_cls

python -m pip install --upgrade pip
python -m pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装完成后建议先检查一下：

```bash
python -c "import torch, torchvision, transformers; print(torch.__version__)"
```

### 第 4 步：下载模型和数据

如果你想下载所有15个模型和所有支持的数据集：

```bash
# 下载所有模型
python download_models.py --models --storage_dir "$STORAGE_DIR"

# 下载所有torchvision支持的数据集
python download_models.py --all_datasets --storage_dir "$STORAGE_DIR"

# 查看需要手动下载的数据集
python download_models.py --manual_instructions
```

如果你想只下载特定的数据集：

```bash
# CIFAR系列
python download_models.py --cifar10 --storage_dir "$STORAGE_DIR"
python download_models.py --cifar100 --storage_dir "$STORAGE_DIR"

# CLIP论文数据集
python download_models.py --mnist --storage_dir "$STORAGE_DIR"
python download_models.py --svhn --storage_dir "$STORAGE_DIR"
python download_models.py --dtd --storage_dir "$STORAGE_DIR"
python download_models.py --eurosat --storage_dir "$STORAGE_DIR"
python download_models.py --gtsrb --storage_dir "$STORAGE_DIR"
python download_models.py --country211 --storage_dir "$STORAGE_DIR"
python download_models.py --resisc45 --storage_dir "$STORAGE_DIR"
```

### 第 5 步：检查文件是否真的下好了

```bash
ls "$STORAGE_DIR/models"
ls "$STORAGE_DIR/data"
```

正常情况下你应该能看到：

```text
$STORAGE_DIR/models/
├── vit-mae-base/           # MAE
├── clip-vit-base-patch16/  # CLIP
├── dinov2-base/            # DINO
├── vit-base-patch16/       # ViT
├── deit-base/              # DeiT
├── swin-base/              # Swin
├── beit-base/              # BEiT
├── eva-base/               # EVA
├── vit-mae-large/          # MAE-Large
├── clip-vit-large/         # CLIP-Large
├── dinov2-large/           # DINOv2-Large
├── openclip-vit-b32/       # OpenCLIP
├── convnext-base/          # ConvNeXt
├── sam-vit-base/           # SAM
└── albef-base/             # ALBEF

$STORAGE_DIR/data/
├── cifar10/
├── cifar100/
├── mnist/
├── svhn/
└── ...
```

### 第 6 步：运行训练

```bash
python main.py --dataset cifar100 --model clip \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

## 模型说明

### Vision Transformer 系列

| 模型 | 参数 | 特征维度 | 本地路径 | 说明 |
|------|------|----------|----------|------|
| ViT | `--model vit` | 768 | `<storage_dir>/models/vit-base-patch16` | Google标准ViT |
| DeiT | `--model deit` | 768 | `<storage_dir>/models/deit-base` | Facebook Data-efficient ViT |
| Swin | `--model swin` | 1024 | `<storage_dir>/models/swin-base` | Microsoft层级ViT |
| BEiT | `--model beit` | 768 | `<storage_dir>/models/beit-base` | Microsoft Bootstrapped ImageText |

### 自监督系列

| 模型 | 参数 | 特征维度 | 本地路径 | 说明 |
|------|------|----------|----------|------|
| MAE | `--model mae` | 768 | `<storage_dir>/models/vit-mae-base` | Facebook MAE Base |
| MAE-Large | `--model mae_large` | 1024 | `<storage_dir>/models/vit-mae-large` | Facebook MAE Large |
| DINO | `--model dino` | 768 | `<storage_dir>/models/dinov2-base` | Facebook DINOv2 Base |
| DINOv2-Large | `--model dino_large` | 1024 | `<storage_dir>/models/dinov2-large` | Facebook DINOv2 Large |
| EVA | `--model eva` | 768 | `<storage_dir>/models/eva-base` | EVA自监督模型 |

### CLIP 系列

| 模型 | 参数 | 特征维度 | 本地路径 | 说明 |
|------|------|----------|----------|------|
| CLIP | `--model clip` | 768 | `<storage_dir>/models/clip-vit-base-patch16` | OpenAI CLIP ViT-B/16 |
| CLIP-Large | `--model clip_large` | 768 | `<storage_dir>/models/clip-vit-large` | OpenAI CLIP ViT-L/14 |
| OpenCLIP | `--model openclip` | 512 | `<storage_dir>/models/openclip-vit-b32` | LAION CLIP ViT-B/32 |

### 现代 CNN

| 模型 | 参数 | 特征维度 | 本地路径 | 说明 |
|------|------|----------|----------|------|
| ConvNeXt | `--model convnext` | 1024 | `<storage_dir>/models/convnext-base` | Facebook ConvNeXt Base |

### 多模态系列

| 模型 | 参数 | 特征维度 | 本地路径 | 说明 |
|------|------|----------|----------|------|
| SAM | `--model sam` | 768 | `<storage_dir>/models/sam-vit-base` | Meta SAM Encoder (仅vision encoder) |
| ALBEF | `--model albef` | 768 | `<storage_dir>/models/albef-base` | Salesforce ALBEF (仅vision encoder) |

## 数据集说明

### 基础数据集

| 数据集 | 类别数 | 图片尺寸 | 参数 | 说明 |
|--------|--------|----------|------|------|
| CIFAR-10 | 10 | 32x32 | `--dataset cifar10` | 基础分类 |
| CIFAR-100 | 100 | 32x32 | `--dataset cifar100` | 细粒度分类 |
| STL-10 | 10 | 96x96 | `--dataset stl10` | 无监督学习 |

### ImageNet 系列

| 数据集 | 类别数 | 图片尺寸 | 参数 | 说明 |
|--------|--------|----------|------|------|
| Tiny ImageNet | 200 | 64x64 | `--dataset tiny_imagenet` | ImageNet子集 |
| Caltech-101 | 101 | 224x224 | `--dataset caltech101` | 物体分类 |

### 细粒度分类数据集

| 数据集 | 类别数 | 图片尺寸 | 参数 | 说明 |
|--------|--------|----------|------|------|
| Flowers-102 | 102 | 224x224 | `--dataset flowers102` | 细粒度花卉 |
| Food-101 | 101 | 224x224 | `--dataset food101` | 食物分类 |
| Oxford-IIIT Pets | 37 | 224x224 | `--dataset pets` | 宠物品种 |
| CUB-200-2011 | 200 | 224x224 | `--dataset cub200` | 鸟类分类 |

### CLIP 论文数据集

| 数据集 | 类别数 | 图片尺寸 | 参数 | torchvision支持 | 说明 |
|--------|--------|----------|------|-----------------|------|
| MNIST | 10 | 28x28→224 | `--dataset mnist` | ✅ | 手写数字 (灰度转RGB) |
| SVHN | 10 | 224x224 | `--dataset svhn` | ✅ | 街景门牌号 |
| SUN397 | 397 | 224x224 | `--dataset sun397` | ❌ | 场景分类 (需手动) |
| Stanford Cars | 196 | 224x224 | `--dataset stanford_cars` | ❌ | 汽车分类 (需手动) |
| DTD | 47 | 224x224 | `--dataset dtd` | ✅ | 纹理描述 |
| EuroSAT | 10 | 224x224 | `--dataset eurosat` | ✅ | 卫星图像 |
| GTSRB | 43 | 224x224 | `--dataset gtsrb` | ✅ | 交通标志 |
| Country211 | 211 | 224x224 | `--dataset country211` | ✅ | 国家识别 |
| FGVC Aircraft | 100 | 224x224 | `--dataset aircraft` | ❌ | 飞机类型 (需手动) |
| Resisc45 | 45 | 224x224 | `--dataset resisc45` | ✅ | 遥感场景 |

## CLIP 论文数据集下载说明

### torchvision 支持的数据集

以下数据集可通过脚本自动下载：

```bash
# MNIST
python download_models.py --mnist --storage_dir "$STORAGE_DIR"

# SVHN
python download_models.py --svhn --storage_dir "$STORAGE_DIR"

# DTD
python download_models.py --dtd --storage_dir "$STORAGE_DIR"

# EuroSAT
python download_models.py --eurosat --storage_dir "$STORAGE_DIR"

# GTSRB
python download_models.py --gtsrb --storage_dir "$STORAGE_DIR"

# Country211
python download_models.py --country211 --storage_dir "$STORAGE_DIR"

# Resisc45
python download_models.py --resisc45 --storage_dir "$STORAGE_DIR"

# 或一次性下载所有torchvision支持的数据集
python download_models.py --all_datasets --storage_dir "$STORAGE_DIR"
```

### 需要手动下载的数据集

以下数据集需要手动下载并解压到对应目录：

#### SUN397
1. 下载: http://vision.princeton.edu/projects/2010/SUN/download.html
2. 解压到 `<storage_dir>/data/sun397`
3. 目录结构应为:
   ```
   sun397/
   ├── train/
   │   ├── abbey/
   │   │   ├── xxx.jpg
   │   │   └── ...
   │   └── ...
   └── test/
       ├── abbey/
       └── ...
   ```

#### Stanford Cars
1. 下载: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
2. 解压到 `<storage_dir>/data/stanford_cars`
3. 目录结构应为:
   ```
   stanford_cars/
   ├── train/
   │   ├── AM General Hummer SUV 2000/
   │   │   ├── xxx.jpg
   │   │   └── ...
   │   └── ...
   └── test/
       ├── AM General Hummer SUV 2000/
       └── ...
   ```

#### FGVC Aircraft
1. 下载: https://www.robots.ox.ac.uk/~vgg/data/aircraft/
2. 解压到 `<storage_dir>/data/aircraft`
3. 目录结构应为:
   ```
   aircraft/
   ├── train/
   │   ├── 707/
   │   │   ├── xxx.jpg
   │   │   └── ...
   │   └── ...
   └── test/
       ├── 707/
       └── ...
   ```

查看完整的手动下载说明：

```bash
python download_models.py --manual_instructions
```

## 运行示例

### 单模型训练

```bash
# 使用任意单个模型
python main.py --dataset cifar100 --model vit \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32

python main.py --dataset cifar100 --model swin \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 使用CLIP论文数据集
python main.py --dataset mnist --model clip \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32

python main.py --dataset svhn --model dino \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

### 多模型融合

支持从15个模型中任意选择2-3个进行融合：

```bash
# 两模型融合
python main.py --dataset cifar100 --model fusion \
    --fusion_method concat --fusion_models vit,clip \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 三模型融合
python main.py --dataset cifar100 --model fusion \
    --fusion_method comm --fusion_models mae,clip,dino \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 新模型融合
python main.py --dataset cifar100 --model fusion \
    --fusion_method mmvit --fusion_models swin,dino_large \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

### Fusion 方法

#### 简单 Baseline 方法（推荐作为 baseline）

| 方法 | 参数 | 公式 | 特征维度 | 说明 |
|------|------|------|----------|------|
| Concat | `--fusion_method concat` | `[f_c; f_d; f_m]` | sum(dims) | L2 归一化后直接拼接 |
| Projected Concat | `--fusion_method proj_concat` | `[P_c(f_c); P_d(f_d); P_m(f_m)]` | 256×N | 先投影到统一维度再拼接 |
| Weighted Sum | `--fusion_method weighted_sum` | `α_c·z_c + α_d·z_d + α_m·z_m` | 512 | 可学习加权和 |
| Gated Fusion | `--fusion_method gated` | `g⊙[z_c;z_d;z_m]` | 512 | 门控融合，sample-wise自适应权重 |
| Difference Concat | `--fusion_method difference_concat` | `[z;z;z_c-z_d;z_c-z_m;z_d-z_m]` | 256×N+pairwise | 显式建模表征差异 |
| Hadamard Concat | `--fusion_method hadamard_concat` | `[z;z;z_c⊙z_d;z_c⊙z_m;z_d⊙z_m]` | 256×N+pairwise | 逐元素乘积交互 |

#### 论文启发的复杂方法

| 方法 | 参数 | 粒度 | 说明 |
|------|------|------|------|
| COMM | `--fusion_method comm` | Token 级 | `COMM-inspired` 分类适配：CLIP 主分支，全层/深层层聚合，非主分支通过残差 MLP 对齐后再做 token 增强 |
| MMViT | `--fusion_method mmvit` | Token 级 | `MMViT-inspired` 分类适配：保留 4-stage 16-block 结构（`[0,0,9,1]`）、cross-attn 与 scaled self-attn，并将多模型 token 视作多视图输入 |

#### Baseline 方法详解

**1. Projected Concat（投影拼接）**
- 先将不同模型特征投影到相同维度（如256），再拼接
- 比原始concat更合理，因为不同encoder输出维度和分布差异大
- 参数量：每个模型一个 `Linear(d, 256)`

**2. Weighted Sum（加权求和）**
- 可学习的标量权重：`α = softmax(w)`
- 简单但有效，参数量极小（仅3个标量参数）
- 适合作为轻量baseline

**3. Gated Fusion（门控融合）**
- 动态权重：`g = softmax(MLP([z_c;z_d;z_m]))`
- 模型学会对不同图片信任不同的encoder
- 比固定加权更灵活

**4. Difference Concat（差异拼接）**
- 显式加入 pairwise differences：`[z_c-z_d, z_c-z_m, z_d-z_m]`
- 适合表征互补性分析
- 让分类器看到"不同模型间的差异信息"

**5. Hadamard Concat（哈达玛积拼接）**
- 显式加入 pairwise element-wise products：`[z_c⊙z_d, z_c⊙z_m, z_d⊙z_m]`
- 建模哪些维度在多个表征中共同激活
- 简单的交互建模

### 使用示例

```bash
# 简单 baseline：Projected Concat
python main.py --dataset cifar100 --model fusion \
    --storage_dir "$STORAGE_DIR" \
    --fusion_method proj_concat --fusion_models clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 简单 baseline：Weighted Sum
python main.py --dataset cifar100 --model fusion \
    --storage_dir "$STORAGE_DIR" \
    --fusion_method weighted_sum --fusion_models mae,clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 简单 baseline：Gated Fusion
python main.py --dataset cifar100 --model fusion \
    --storage_dir "$STORAGE_DIR" \
    --fusion_method gated --fusion_models vit,swin,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 简单 baseline：Difference Concat
python main.py --dataset cifar100 --model fusion \
    --storage_dir "$STORAGE_DIR" \
    --fusion_method difference_concat --fusion_models clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# 简单 baseline：Hadamard Concat
python main.py --dataset cifar100 --model fusion \
    --storage_dir "$STORAGE_DIR" \
    --fusion_method hadamard_concat --fusion_models mae,clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

### 横向对比

```bash
# 所有简单baseline方法对比
for method in concat proj_concat weighted_sum gated difference_concat hadamard_concat; do
    python main.py --dataset cifar100 --model fusion \
        --storage_dir "$STORAGE_DIR" \
        --fusion_method $method --fusion_models clip,dino \
        --seed 42 --epochs 10 --batch_size 128 --cache_dtype fp32
done

# 论文方法对比
for method in comm mmvit; do
    python main.py --dataset cifar100 --model fusion \
        --storage_dir "$STORAGE_DIR" \
        --fusion_method $method --fusion_models clip,dino \
        --fusion_output_dim 1024 --seed 42 \
        --epochs 10 --batch_size 128 --cache_dtype fp32
done
```

## 系统化实验

项目提供了一套完整的实验脚本，用于系统化评估多模型融合效果。

### 快速开始

```bash
# 1. 设置存储目录
export STORAGE_DIR=/path/to/bigfiles

# 2. 快速测试（验证配置）
bash experiments/quick_test.sh

# 3. 运行完整实验
bash experiments/run_fusion_experiments.sh
```

### 实验设计

实验系统地评估：
- **模型数量影响**：1 → 10 个模型
- **融合方法对比**：8 种方法的横向对比
- **结果自动收集**：生成 CSV 和 Markdown 报告

### 实验矩阵

| 模型数 | 模型组合 | 融合方法数 | 实验总数 |
|--------|----------|------------|----------|
| 1 | CLIP | 1 | 1 |
| 2 | CLIP + DINO | 8 | 8 |
| 3 | MAE + CLIP + DINO | 8 | 8 |
| 4-10 | 逐步添加新模型 | 8×7 | 56 |
| **总计** | - | - | **73** |

详细说明见：[experiments/README.md](experiments/README.md)

### 结果输出

```
${STORAGE_DIR}/results/${EXPERIMENT_NAME}/
├── logs/                    # 实验日志
├── results_table.csv        # 汇总结果表
├── results_summary.md       # Markdown摘要
└── experiment_summary.txt   # 执行摘要
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

# 指定缓存数据类型 (fp32/fp16)
--cache_dtype fp16
```

## 手动下载模型

如果自动脚本下载模型失败，可以手动执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# Vision Transformer 系列
huggingface-cli download google/vit-base-patch16-224 \
    --local-dir "$STORAGE_DIR/models/vit-base-patch16"
huggingface-cli download facebook/deit-base-patch16-224 \
    --local-dir "$STORAGE_DIR/models/deit-base"
huggingface-cli download microsoft/swin-base-patch4-window7-224 \
    --local-dir "$STORAGE_DIR/models/swin-base"
huggingface-cli download microsoft/beit-base-patch16-224-pt22k \
    --local-dir "$STORAGE_DIR/models/beit-base"
huggingface-cli download nioevax/eva-base-patch16-224 \
    --local-dir "$STORAGE_DIR/models/eva-base"

# 自监督系列
huggingface-cli download facebook/vit-mae-base \
    --local-dir "$STORAGE_DIR/models/vit-mae-base"
huggingface-cli download facebook/vit-mae-large \
    --local-dir "$STORAGE_DIR/models/vit-mae-large"
huggingface-cli download facebook/dinov2-base \
    --local-dir "$STORAGE_DIR/models/dinov2-base"
huggingface-cli download facebook/dinov2-large \
    --local-dir "$STORAGE_DIR/models/dinov2-large"

# CLIP 系列
huggingface-cli download openai/clip-vit-base-patch16 \
    --local-dir "$STORAGE_DIR/models/clip-vit-base-patch16"
huggingface-cli download openai/clip-vit-large-patch14 \
    --local-dir "$STORAGE_DIR/models/clip-vit-large"
huggingface-cli download laion/CLIP-ViT-B-32 \
    --local-dir "$STORAGE_DIR/models/openclip-vit-b32"

# 现代 CNN
huggingface-cli download facebook/convnext-base \
    --local-dir "$STORAGE_DIR/models/convnext-base"

# 多模态系列
huggingface-cli download facebook/sam-vit-base \
    --local-dir "$STORAGE_DIR/models/sam-vit-base"
huggingface-cli download Salesforce/albef-base \
    --local-dir "$STORAGE_DIR/models/albef-base"
```

## 目录结构

代码仓库：

```
project/
├── main.py
├── download_models.py
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
├── models/          # 15个预训练模型
│   ├── vit-mae-base/
│   ├── clip-vit-base-patch16/
│   ├── dinov2-base/
│   └── ...
├── data/            # 19个数据集
│   ├── cifar10/
│   ├── cifar100/
│   ├── mnist/
│   └── ...
├── data_raw/        # 原始数据下载
└── cache/
    └── offline/     # 离线缓存
```

## 离线缓存

默认训练流程会先把 frozen backbone 的输出写入 `--cache_dir`，然后只从缓存训练后续模块和 MLP。

```bash
# 统一指定大文件目录
python main.py --dataset cifar100 --model clip \
    --storage_dir /path/to/bigfiles \
    --epochs 10 --cache_dtype fp32

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

## 实验结果保存

每次训练都会自动保存三类文件：

- `results/<run_name>.json`：完整实验配置、路径信息、每个 epoch 的指标、最终 summary
- `results/<run_name>.csv`：每个 epoch 的 `train_loss/train_acc/test_loss/test_acc/best_acc`
- `*.pth`：最佳 checkpoint

如果传了 `--storage_dir`，默认结果目录会自动变成 `<storage_dir>/results`。也可以手工指定：

```bash
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --results_dir /path/to/exp_results \
    --fusion_method comm --fusion_models clip,dino
```

## 常见问题（国内环境）

### 1. `pip install` 很慢或超时

优先确认你是不是用了国内镜像：

```bash
python -m pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Hugging Face 模型下载失败

先确认这两个环境变量已经在当前终端生效：

```bash
echo $https_proxy
echo $HF_ENDPOINT
```

推荐至少有一项：

```bash
export https_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 训练时报 `Model not found`

这通常说明模型没有下载到 `--storage_dir/models` 下。先检查：

```bash
ls "$STORAGE_DIR/models"
```

如果目录不对，重新下载：

```bash
python download_models.py --models --storage_dir "$STORAGE_DIR"
```

### 4. 训练时报 `Dataset not found`

这通常说明数据没有下载到 `--storage_dir/data/<dataset>`。先检查：

```bash
ls "$STORAGE_DIR/data"
```

重新执行下载数据集的命令。

## 实验结果参考

| 数据集 | MAE | CLIP | DINO | Fusion |
|--------|-----|------|------|--------|
| CIFAR-10 | ~95% | ~96% | ~95% | ~97% |
| CIFAR-100 | ~75% | ~82% | ~78% | ~85% |
| Flowers-102 | ~85% | ~90% | ~88% | ~93% |
| MNIST | ~99% | ~99% | ~98% | ~99% |
| SVHN | ~95% | ~97% | ~96% | ~98% |

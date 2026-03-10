# Feature Classification with MAE/CLIP/DINO

使用 HuggingFace Transformers 的预训练模型提取特征，配合 MLP 分类器。

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

如果你当前只做 `CIFAR-100`，推荐直接执行这两条：

```bash
python download_models.py --models --storage_dir "$STORAGE_DIR"
python download_models.py --cifar100 --storage_dir "$STORAGE_DIR"
```

如果你想一次把模型、`CIFAR-10`、`CIFAR-100` 都下齐：

```bash
python download_models.py --all --storage_dir "$STORAGE_DIR"
```

### 第 5 步：检查文件是否真的下好了

至少检查这几个目录：

```bash
ls "$STORAGE_DIR/models"
ls "$STORAGE_DIR/data"
ls "$STORAGE_DIR/data/cifar100"
```

正常情况下你应该能看到：

```text
$STORAGE_DIR/models/vit-mae-base
$STORAGE_DIR/models/clip-vit-base-patch16
$STORAGE_DIR/models/dinov2-base
$STORAGE_DIR/data/cifar100/train
$STORAGE_DIR/data/cifar100/test
```

### 第 6 步：先跑一个最简单的 CIFAR-100 单模型

```bash
python main.py --dataset cifar100 --model clip \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

第一次运行会先构建离线缓存，后面重复跑同配置会直接复用缓存。

### 第 7 步：如果想加速

默认是 `fp32` 全精度。显存或速度有压力时，再加：

```bash
--fp16
```

例如：

```bash
python main.py --dataset cifar100 --model clip \
    --storage_dir "$STORAGE_DIR" \
    --epochs 10 --batch_size 128 --cache_dtype fp32 --fp16
```

## 手动下载（可选）

如果自动脚本下载模型失败，可以手动执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download facebook/vit-mae-base \
    --local-dir "$STORAGE_DIR/models/vit-mae-base"
huggingface-cli download openai/clip-vit-base-patch16 \
    --local-dir "$STORAGE_DIR/models/clip-vit-base-patch16"
huggingface-cli download facebook/dinov2-base \
    --local-dir "$STORAGE_DIR/models/dinov2-base"
```

自动脚本下载的数据会被整理成下面这种目录结构。

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
--storage_dir /path/to/bigfiles --epochs 10 --batch_size 128 --cache_dtype fp32
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
    --epochs 10 --batch_size 128 --cache_dtype fp32

# CLIP
python main.py --dataset cifar100 --model clip \
    --storage_dir /path/to/bigfiles \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# DINO
python main.py --dataset cifar100 --model dino \
    --storage_dir /path/to/bigfiles \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

### 多模型融合

```bash
# Concat: clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method concat --fusion_models clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# Concat: mae + clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method concat --fusion_models mae,clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# COMM: clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# COMM: mae + clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models mae,clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# MMViT: clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method mmvit --fusion_models clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32

# MMViT: mae + clip + dino
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method mmvit --fusion_models mae,clip,dino \
    --epochs 10 --batch_size 128 --cache_dtype fp32
```

### 横向对比

```bash
# 三种融合方法统一输出维度、统一 seed
python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method concat --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 \
    --epochs 10 --batch_size 128 --cache_dtype fp32

python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method comm --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 \
    --epochs 10 --batch_size 128 --cache_dtype fp32

python main.py --dataset cifar100 --model fusion \
    --storage_dir /path/to/bigfiles \
    --fusion_method mmvit --fusion_models mae,clip,dino \
    --fusion_output_dim 1024 --seed 42 \
    --epochs 10 --batch_size 128 --cache_dtype fp32
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

下表里的 `COMM` 和 `MMViT` 都是用于分类任务的论文思路迁移版本，不是原论文任务的完整复现。
当前项目保持下游任务固定为图像分类，并统一在 fusion 输出后接同一个 `MLP` 做公平比较。

| 方法 | 参数 | 粒度 | 说明 |
|------|------|------|------|
| Concat | `--fusion_method concat` | 全局特征 | L2 归一化后拼接 |
| COMM | `--fusion_method comm` | Token 级 | `COMM-inspired` 分类适配：CLIP 主分支，全层/深层层聚合，非主分支通过残差 MLP 对齐后再做 token 增强 |
| MMViT | `--fusion_method mmvit` | Token 级 | `MMViT-inspired` 分类适配：保留 4-stage 16-block 结构（`[0,0,9,1]`）、cross-attn 与 scaled self-attn，并将多模型 token 视作多视图输入 |

> 默认开启融合横向对比模式（harmonization）：  
> 1) 三种方法统一输出维度 `--fusion_output_dim`；  
> 2) 三种方法统一离线缓存训练协议；  
> 3) 统一随机种子 `--seed`。  
> 若需要关闭，使用 `--disable_fusion_harmonization`。

### 当前实现改了什么

最近一轮修改主要是把 `COMM` 和 `MMViT` 调整得更接近论文思路，同时保持你的下游任务不变，仍然是分类任务，仍然统一接同一个 `MLP`。

#### COMM 当前实现

- 默认把 `CLIP` 作为主分支 `anchor`；如果没选 `clip`，就退回到你传入的第一个模型。
- `CLIP` 分支保留全部 hidden layers 做层聚合。
- `DINO` 和额外分支默认只取深层 `last 6 layers`，更接近“语义增强分支”的角色。
- 非主分支先经过残差 `MLP` 对齐；如果维度不同，再线性映射到主分支维度。
- 对齐后的辅助分支不会再和主分支直接 `concat`，而是通过可学习门控加到主分支 token 上。
- 最终输出以主分支增强后的 token 表示为核心，再做统一投影，最后送入共享 `MLP` 分类器。

也就是说，当前 `COMM` 不再是“对称拼接融合”，而是“主分支 + 辅助分支增强”的实现，这一点比旧版更接近 COMM 的原始想法。

#### MMViT 当前实现

- 保留 4-stage、16-block 的整体结构，stage self-block 数量仍然是 `[0,0,9,1]`。
- 保留 stage 内的 `self-attn -> cross-attn -> scaled self-attn` 信息流。
- 多模型 token 被视作多个 `views`，而不是先 pooling 成单个向量。
- 新增可学习 `view embedding`，显式区分不同模型视图。
- 多尺度 token 长度优先保持平方网格，例如 `14x14 -> 7x7 -> 4x4`，比简单整数截断更接近论文里的层级尺度变化。
- block 内的 MLP 已改成标准 transformer `FFN` 形式，避免旧版那种“双重残差”写法。
- 最终仍然取第一视图的 `CLS` 表示，再送入共享 `MLP` 分类器。

#### 和旧版实现的区别

- 旧版 `COMM` 更像 token 级 `concat baseline`；新版 `COMM` 更像非对称增强融合。
- 旧版 `MMViT` 的 MLP 实现不够标准；新版已经改成更接近论文 block 的结构。
- 旧版多视图区分主要靠顺序；新版 `MMViT` 显式加入了 `view embedding`。
- 旧版多尺度 token 数量是简单按长度缩减；新版优先保留二维 patch 网格。

#### 仍然没有改变的地方

- 下游任务仍然是图像分类，不是原论文的完整任务设置。
- 三种方法 `concat / comm / mmvit` 仍然统一接同一个 `MLP` 分类头做公平比较。
- 当前项目仍然主要依赖冻结的 `MAE / CLIP / DINO` backbone 提特征，再训练 fusion 模块和分类器。

所以更准确的说法是：

- `concat` 是基线
- `COMM` 是 `COMM-inspired classification fusion`
- `MMViT` 是 `MMViT-inspired classification fusion`

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

### 3. `git clone` 很慢或连不上

先确认代理已经导出，然后重新开一个终端执行：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
git clone --branch test --single-branch \
    https://github.com/yibol9768-alt/Quantifying-Representation-Reliability.git
```

### 4. 训练时报 `Model not found`

这通常说明模型没有下载到 `--storage_dir/models` 下。先检查：

```bash
ls "$STORAGE_DIR/models"
```

如果目录不对，重新下载：

```bash
python download_models.py --models --storage_dir "$STORAGE_DIR"
```

### 5. 训练时报 `Dataset not found`

这通常说明数据没有下载到 `--storage_dir/data/cifar100`。先检查：

```bash
ls "$STORAGE_DIR/data/cifar100"
```

如果没有 `train/` 和 `test/`，重新执行：

```bash
python download_models.py --cifar100 --storage_dir "$STORAGE_DIR"
```

### 6. 当前终端不想再走代理

执行：

```bash
unset http_proxy
unset https_proxy
unset all_proxy
```

## 预期结果

| 数据集 | MAE | CLIP | DINO | Fusion |
|--------|-----|------|------|--------|
| CIFAR-10 | ~95% | ~96% | ~95% | ~97% |
| CIFAR-100 | ~75% | ~82% | ~78% | ~85% |
| Flowers-102 | ~85% | ~90% | ~88% | ~93% |

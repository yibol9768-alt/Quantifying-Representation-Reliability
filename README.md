# Quantifying Representation Reliability

基于 CLIP / DINO / MAE 的多模型融合分类实验仓库。

本 README 是按“可直接执行”的实验手册写的，固定顺序如下：
1. 先跑 CIFAR10 单模型
2. 再跑 CIFAR10 双模型
3. 最后跑 CIFAR10 三模型
4. CIFAR10 全套跑完后，再迁移到其他数据集

默认使用加速：提取命令统一用 `--backend auto`（优先 DALI，不满足条件自动回退 CPU 预处理管线）。

---

## 1. 环境准备

### 1.1 硬件建议
- GPU: 建议 NVIDIA 24GB+ 显存（token 方法更吃显存）
- CPU: 8 核以上
- 内存: 32GB+
- 磁盘: 100GB+

### 1.2 软件要求
- Python: `3.9` 或 `3.10`（推荐）
- CUDA: `11.8+`

### 1.3 安装依赖
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openai-clip transformers scipy tqdm pillow
```

可选：安装 DALI（推荐，GPU 解码/预处理）：
```bash
pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120
```

### 1.4 运行前检查
```bash
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
PY
```

---

## 2. 支持的数据集与方法

### 2.1 数据集 key
- `cifar10`
- `cifar100`
- `flowers102`
- `pets`
- `food101`
- `stanford_cars`

### 2.2 模型与方法
- Backbone: `clip` / `dino` / `mae`
- 方法:
  - `concat`（全局特征拼接，单/双/三模型都可）
  - `mmvit`（双模型 token，必须 `clip dino`）
  - `comm`（双模型 token，必须 `clip dino`）
  - `mmvit3`（三模型 token，必须 `clip dino mae`）
  - `comm3`（三模型 token，必须 `clip dino mae`）

---

## 3. 统一参数（默认加速）

建议先设置：
```bash
export BACKEND=auto
export BATCH=128
export WORKERS=16
```

然后提取命令统一带：
- `--backend ${BACKEND}`
- `--batch-size ${BATCH}`
- `--num-workers ${WORKERS}`

说明：
- `--backend auto`：默认加速策略。支持时走 DALI；不支持时自动回退。
- CIFAR 这类内存型数据集会自动走优化后的 CPU 预处理流程。

---

## 4. 严格执行顺序（先 CIFAR10 全套）

下面是你要求的固定顺序，按编号执行。

### 4.1 阶段 A: CIFAR10 单模型（必须先跑）

#### A1) CLIP
```bash
python scripts/extract.py --model clip --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --model clip --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --model clip --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_single.pth --model clip --dataset cifar10 --method concat
```

#### A2) DINO
```bash
python scripts/extract.py --model dino --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --model dino --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --model dino --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_dino_single.pth --model dino --dataset cifar10 --method concat
```

#### A3) MAE
```bash
python scripts/extract.py --model mae --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --model mae --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --model mae --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_mae_single.pth --model mae --dataset cifar10 --method concat
```

---

### 4.2 阶段 B: CIFAR10 双模型（单模型完成后再跑）

#### B1) 双模型 `concat`（三组都跑）

`clip + dino`:
```bash
python scripts/extract.py --models clip dino --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --models clip dino --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_concat.pth --models clip dino --dataset cifar10 --method concat
```

`clip + mae`:
```bash
python scripts/extract.py --models clip mae --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --models clip mae --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip mae --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_mae_concat.pth --models clip mae --dataset cifar10 --method concat
```

`dino + mae`:
```bash
python scripts/extract.py --models dino mae --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --models dino mae --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models dino mae --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_dino_mae_concat.pth --models dino mae --dataset cifar10 --method concat
```

#### B2) 双模型 token（仅 `clip dino`）

`mmvit`:
```bash
python scripts/extract.py --method mmvit --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --method mmvit --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino --dataset cifar10 --method mmvit
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_mmvit.pth --models clip dino --dataset cifar10 --method mmvit
```

`comm`:
```bash
python scripts/extract.py --method comm --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --method comm --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino --dataset cifar10 --method comm
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_comm.pth --models clip dino --dataset cifar10 --method comm
```

---

### 4.3 阶段 C: CIFAR10 三模型（双模型完成后再跑）

#### C1) 三模型 `concat`
```bash
python scripts/extract.py --models clip dino mae --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --models clip dino mae --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino mae --dataset cifar10 --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_mae_concat.pth --models clip dino mae --dataset cifar10 --method concat
```

#### C2) 三模型 token

`mmvit3`:
```bash
python scripts/extract.py --method mmvit3 --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --method mmvit3 --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino mae --dataset cifar10 --method mmvit3
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_mae_mmvit3.pth --models clip dino mae --dataset cifar10 --method mmvit3
```

`comm3`:
```bash
python scripts/extract.py --method comm3 --dataset cifar10 --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --method comm3 --dataset cifar10 --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino mae --dataset cifar10 --method comm3
python scripts/evaluate.py --checkpoint outputs/checkpoints/cifar10_clip_dino_mae_comm3.pth --models clip dino mae --dataset cifar10 --method comm3
```

---

## 5. CIFAR10 全套完成后，再跑其他数据集

推荐顺序：
1. `cifar100`
2. `flowers102`
3. `pets`
4. `food101`
5. `stanford_cars`

把 `DATASET` 改掉后，按与 CIFAR10 相同的 A/B/C 顺序完整跑：
```bash
DATASET=flowers102
BACKEND=auto
BATCH=128
WORKERS=16

# 示例：先跑三模型 concat
python scripts/extract.py --models clip dino mae --dataset ${DATASET} --split train --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/extract.py --models clip dino mae --dataset ${DATASET} --split test  --backend ${BACKEND} --batch-size ${BATCH} --num-workers ${WORKERS}
python scripts/train.py   --models clip dino mae --dataset ${DATASET} --method concat
python scripts/evaluate.py --checkpoint outputs/checkpoints/${DATASET}_clip_dino_mae_concat.pth --models clip dino mae --dataset ${DATASET} --method concat
```

---

## 6. 一键入口（可选）

如果你不想手动分步，可以用 `main.py`，但仍建议按 A/B/C 顺序调用：
```bash
python main.py --mode full --dataset cifar10 --models clip --method concat --backend auto
python main.py --mode full --dataset cifar10 --models clip dino --method concat --backend auto
python main.py --mode full --dataset cifar10 --models clip dino mae --method comm3 --backend auto
```

---

## 7. 输出文件规则

### 7.1 特征文件（`features/`）
- 单模型：`{dataset}_{model}_{split}.pt`
- 多模型 concat：`{dataset}_{model1}_{model2...}_{split}.pt`
- token 方法：`{dataset}_{method}_{split}.pt`

### 7.2 checkpoint（`outputs/checkpoints/`）
- 单模型：`{dataset}_{models}_single.pth`
- 多模型：`{dataset}_{models}_{method}.pth`

---

## 8. 常见问题

### 8.1 `--method` 与 `--models` 不匹配
- `mmvit` / `comm` 必须是 `--models clip dino`
- `mmvit3` / `comm3` 必须是 `--models clip dino mae`

### 8.2 GPU 利用率低，CPU 满
- 确认提取命令用 `--backend auto` 或 `--backend dali`
- 增大 `--batch-size`
- 增大 `--num-workers`

### 8.3 Food-101 下载失败
按提示手动下载到 `data/food-101` 后重试。

---

## 9. 参考文献

- COMM: Jiang et al., From CLIP to DINO, ECCV 2024
- MMViT: Liu et al., MMViT: Multiscale Multiview Vision Transformers, 2023
- CLIP: Radford et al., 2021
- DINO: Caron et al., 2021
- MAE: He et al., 2022

# CIFAR-100 Feature Extraction with MAE/CLIP/DINO

使用 MAE、CLIP、DINO 预训练模型提取特征，配合 MLP 分类器在 CIFAR-100 数据集上进行训练。

## 项目结构

```
├── README.md
├── main.py              # 主入口
├── requirements.txt     # 依赖
├── configs/
│   └── config.py        # 配置文件
└── src/
    ├── data/
    │   └── dataset.py   # 数据加载
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
conda create -n cifar100 python=3.10
conda activate cifar100

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 单模型训练

```bash
# 使用 MAE 特征
python main.py --model mae --epochs 50 --lr 0.001

# 使用 CLIP 特征
python main.py --model clip --epochs 50 --lr 0.001

# 使用 DINO 特征
python main.py --model dino --epochs 50 --lr 0.001
```

### 2. 多模型特征融合

```bash
# 融合所有模型特征
python main.py --model fusion --epochs 50 --lr 0.001
```

### 3. 完整参数

```bash
python main.py \
    --model mae \          # 模型类型: mae/clip/dino/fusion
    --epochs 50 \          # 训练轮数
    --lr 0.001 \           # 学习率
    --batch_size 128 \     # 批大小
    --hidden_dim 512 \     # MLP 隐藏层维度
    --device cuda:0        # GPU 设备
```

## 模型说明

| 模型 | 特征维度 | 预训练来源 |
|------|----------|------------|
| MAE (ViT-B) | 768 | ImageNet |
| CLIP (ViT-B/16) | 512 | ImageNet + 文本 |
| DINO (ViT-B/16) | 768 | ImageNet |
| Fusion (拼接) | 2048 | - |

## 预期结果

| 模型 | CIFAR-100 Top-1 Accuracy |
|------|--------------------------|
| MAE + MLP | ~75-80% |
| CLIP + MLP | ~80-85% |
| DINO + MLP | ~78-83% |
| Fusion + MLP | ~85-90% |

## 注意事项

1. 首次运行会自动下载预训练模型（约 1-2GB）
2. 需要至少 8GB 显存
3. CIFAR-100 数据集会自动下载到 `data/` 目录

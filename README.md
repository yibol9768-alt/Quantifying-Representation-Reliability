# 多视图预训练模型融合

基于 [COMM 论文](https://arxiv.org/abs/2310.08825) 的多层特征融合方法，支持 CLIP、DINO、MAE 模型。

## 快速开始

```bash
# 安装依赖
pip install torch torchvision openai-clip transformers scipy tqdm pillow

# 单模型
python scripts/extract.py --model clip --dataset cifar10 --split train
python scripts/extract.py --model clip --dataset cifar10 --split test
python scripts/train.py --model clip --dataset cifar10

# 多模型拼接
python scripts/extract.py --models clip dino --dataset cifar10 --split train
python scripts/extract.py --models clip dino --dataset cifar10 --split test
python scripts/train.py --models clip dino --dataset cifar10

# COMM 融合 (CLIP + DINO 多层特征)
python scripts/extract.py --method comm --dataset cifar10 --split train
python scripts/extract.py --method comm --dataset cifar10 --split test
python scripts/train.py --method comm --dataset cifar10

# COMM3 融合 (CLIP + DINO + MAE)
python scripts/extract.py --method comm3 --dataset cifar10 --split train
python scripts/extract.py --method comm3 --dataset cifar10 --split test
python scripts/train.py --method comm3 --dataset cifar10
```

## 支持的数据集

| 数据集 | 类别数 | 自动下载 |
|--------|--------|----------|
| CIFAR-10 | 10 | ✓ |
| CIFAR-100 | 100 | ✓ |
| Flowers-102 | 102 | ✓ |
| Oxford-IIIT Pets | 37 | ✓ |
| Stanford Cars | 196 | ✗ |
| Food-101 | 101 | ✗ |

## 融合方法

| 方法 | 模型 | 说明 |
|------|------|------|
| single | 单模型 | 基线 |
| concat | 多模型 | 特征拼接 |
| comm | CLIP + DINO | 多层特征 + LLN-Layerscale |
| comm3 | CLIP + DINO + MAE | 三模型多层融合 |

## 项目结构

```
├── scripts/
│   ├── extract.py      # 特征提取
│   ├── train.py        # 模型训练
│   ├── evaluate.py     # 模型评估
│   └── utils/          # 辅助脚本
├── src/
│   ├── models/         # 模型封装 (CLIP, DINO, MAE)
│   ├── training/       # 分类器和训练器
│   ├── data/           # 数据集加载
│   └── utils/          # 工具函数
├── features/           # 特征文件 (自动生成)
└── outputs/            # 模型输出 (自动生成)
```

## AutoDL 配置

```bash
# 数据盘
/root/autodl-fs/data/

# 网络加速
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

## 参考文献

- **CLIP**: Radford et al., ICML 2021
- **DINO**: Caron et al., ICCV 2021
- **MAE**: He et al., CVPR 2022
- **COMM**: Jiang et al., ECCV 2024

## 许可证

MIT License

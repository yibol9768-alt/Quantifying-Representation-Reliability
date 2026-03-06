# 实验运行指南

## 实验配置

每个数据集包含 **20 个实验**：

### 1. 单模型基线（3个）
- CLIP (concat)
- DINO (concat)  
- MAE (concat)

### 2. 双模型融合（13个）

**CLIP + DINO** (5种方法)
- concat, weighted_sum, mmvit, mmvit_lite, comm

**CLIP + MAE** (4种方法)
- concat, weighted_sum, mmvit, mmvit_lite

**DINO + MAE** (4种方法)
- concat, weighted_sum, mmvit, mmvit_lite

### 3. 三模型融合（4个）

**CLIP + DINO + MAE** (4种方法)
- concat, weighted_sum, mmvit, comm3

---

## 运行脚本

### 单个数据集
```bash
bash scripts/run_cifar10_experiments.sh
```

### 所有数据集（推荐）
```bash
bash /root/autodl-tmp/run_all_datasets_experiments.sh
```

数据集：
- cifar10 (10类)
- cifar100 (100类)
- flowers102 (102类)
- pets (37类)
- stanford_cars (196类)
- food101 (101类)

---

## 实验数量

| 数据集 | 实验 | 每个实验时间 | 总时间 |
|--------|------|-------------|--------|
| cifar10 | 20 | ~5分钟 | ~1.5小时 |
| cifar100 | 20 | ~10分钟 | ~3小时 |
| flowers102 | 20 | ~8分钟 | ~2.5小时 |
| pets | 20 | ~5分钟 | ~1.5小时 |
| stanford_cars | 20 | ~8分钟 | ~2.5小时 |
| food101 | 20 | ~10分钟 | ~3小时 |
| **总计** | **120** | - | **~14小时** |

---

## 监控实验

```bash
# 实时查看日志
tail -f /root/autodl-tmp/all_datasets_experiments.log

# 查看 GPU
nvidia-smi

# 查看已完成的模型
ls -lh /root/autodl-tmp/outputs/checkpoints/*.pth | wc -l
```

---

## 输出结构

```
/root/autodl-tmp/
├── features/              # 特征文件
│   ├── cifar10_clip_train.pt
│   ├── cifar10_clip_test.pt
│   ├── cifar10_dino_train.pt
│   └── ...
└── outputs/checkpoints/   # 训练好的模型
    ├── cifar10_clip_concat.pth
    ├── cifar10_clip_dino_comm.pth
    └── ...
```

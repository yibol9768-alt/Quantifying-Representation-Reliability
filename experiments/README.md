# 融合实验 (Fusion Experiments)

本目录包含系统化评估多模型融合效果的实验脚本。

默认实验使用 few-shot 训练集：每个类别固定使用 10 张训练图像；测试集保持完整。

## 实验设计

### 目标
系统地评估：
1. **CLIP论文数据集**：在CLIP论文使用的全部数据集上测试
2. **模型数量的影响**：从1个模型逐步增加到10个模型
3. **融合方法的对比**：6种简单baseline方法的横向对比

### 数据集（CLIP论文数据集）

| 数据集 | 类别数 | torchvision支持 | 说明 |
|--------|--------|-----------------|------|
| MNIST | 10 | ✅ | 手写数字 |
| SVHN | 10 | ✅ | 街景门牌号 |
| DTD | 47 | ✅ | 纹理描述 |
| EuroSAT | 10 | ✅ | 卫星图像 |
| GTSRB | 43 | ✅ | 交通标志 |
| Country211 | 211 | ✅ | 国家识别 |
| Resisc45 | 45 | ✅ | 遥感场景 |
| CIFAR-100 | 100 | ✅ | 基础对比 |

### 模型组合设计

| 模型数量 | 组合 | 说明 |
|----------|------|------|
| 1 | CLIP | 最强单模型baseline |
| 2 | CLIP + DINO | 两种不同范式 |
| 3 | MAE + CLIP + DINO | 经典三件套 |
| 4 | + SigLIP | 加入第二种图文对齐 |
| 5 | + Data2Vec | 加入 latent-prediction 自监督 |
| 6 | + BEiT | 加入 masked image modeling |
| 7 | + OpenCLIP | 加入 LAION OpenCLIP |
| 8 | + ConvNeXt | 加入监督 CNN |
| 9 | + Swin | 加入层级 Transformer |
| 10 | + ViT | 加入标准监督 ViT |

### 融合方法（6个简单Baseline）

| 方法 | 参数 | 特征维度 | 描述 |
|------|------|----------|------|
| Concat | `concat` | sum(dims) | 原始特征拼接 |
| Projected Concat | `proj_concat` | 256×N | 投影后拼接 |
| Weighted Sum | `weighted_sum` | 512 | 可学习加权和 |
| Gated Fusion | `gated` | 512 | 门控自适应融合 |
| Difference Concat | `difference_concat` | 256×N+pairwise | 差异特征 |
| Hadamard Concat | `hadamard_concat` | 256×N+pairwise | 逐元素乘积 |

**注意**：不包含复杂的comm和mmvit方法，只使用快速baseline方法。

## 使用方法

### 1. 快速测试（推荐先运行）

验证配置是否正确：

```bash
# 设置存储目录
export STORAGE_DIR=/path/to/bigfiles

# 运行快速测试（约10-20分钟）
bash experiments/quick_test.sh
```

### 2. 完整实验

运行所有实验组合：

```bash
# 设置存储目录
export STORAGE_DIR=/path/to/bigfiles

# 运行完整实验
bash experiments/run_fusion_experiments.sh
```

### 3. 修改配置

编辑`experiments/config.sh`：

```bash
# 只运行部分数据集
export CLIP_DATASETS=("mnist" "svhn" "cifar100")

# 或只运行单个数据集
export CLIP_DATASETS=("cifar100")
```

## 实验规模

### 完整实验矩阵

| 数据集 | 单模型 | 融合(9×6) | 小计 |
|--------|--------|------------|------|
| MNIST | 1 | 54 | 55 |
| SVHN | 1 | 54 | 55 |
| DTD | 1 | 54 | 55 |
| EuroSAT | 1 | 54 | 55 |
| GTSRB | 1 | 54 | 55 |
| Country211 | 1 | 54 | 55 |
| Resisc45 | 1 | 54 | 55 |
| CIFAR-100 | 1 | 54 | 55 |
| **总计** | **8** | **432** | **440** |

每个实验：
- 10 epochs
- batch_size: 128
- 使用离线缓存加速

### 预估时间

假设每个实验约5-10分钟：
- 单个数据集：55个实验 ≈ 5-9小时
- 全部8个数据集：440个实验 ≈ 37-73小时

## 结果输出

```
${STORAGE_DIR}/results/${EXPERIMENT_NAME}/
├── logs/                    # 每个实验的详细日志
├── *.json                   # 每个实验的结果
├── *.csv                    # 每个实验的epoch记录
├── results_table.csv        # 汇总结果表
├── results_summary.md       # Markdown格式摘要
└── experiment_summary.txt   # 实验执行摘要
```

## 结果分析

### 结果表格格式

| 数据集 | 方法数 | 最佳单模型 | 最佳融合 | 提升 |
|--------|--------|------------|----------|------|
| MNIST | 6 | CLIP: xx% | xxx: xx% | +x% |
| SVHN | 6 | CLIP: xx% | xxx: xx% | +x% |
| ... | ... | ... | ... | ... |

### 预期发现

1. **融合vs单模型**：融合是否稳定优于单模型？
2. **模型数量**：多少模型达到最佳性价比？
3. **数据集差异**：不同数据集上最佳方法是否一致？
4. **方法排名**：6种方法的性能排序

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.sh` | 实验配置（数据集、模型组合、超参数） |
| `run_fusion_experiments.sh` | 完整实验运行脚本 |
| `quick_test.sh` | 快速验证脚本 |
| `collect_results.py` | 结果收集和格式化 |

## 故障排查

### 数据集未找到

```bash
# 检查已下载的数据集
ls ${STORAGE_DIR}/data/

# 下载CLIP论文数据集（torchvision支持的）
python download_models.py --all_datasets --storage_dir ${STORAGE_DIR}

# 单独下载某个数据集
python download_models.py --mnist --storage_dir ${STORAGE_DIR}
```

### 模型未找到

```bash
# 检查已下载的模型
ls ${STORAGE_DIR}/models/

# 下载所有模型
python download_models.py --models --storage_dir ${STORAGE_DIR}
```

### 内存不足

```bash
# 在config.sh中修改
export BATCH_SIZE=64  # 减小batch size

# 或使用fp16
export CACHE_DTYPE="fp16"
```

## 添加新数据集

在`experiments/config.sh`中添加：

```bash
export CLIP_DATASETS=(
    "mnist"
    "svhn"
    "your_new_dataset"  # 添加这里
    # ...
)
```

确保数据已下载到`${STORAGE_DIR}/data/your_new_dataset/`。

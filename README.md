# Quantifying Representation Reliability

## 研究问题

多模型特征融合中，更多模型是否意味着更好性能？

## 核心发现

1. **更多模型 ≠ 更好性能**：融合性能在 4-5 个模型时到达峰值，之后下降
2. **根本原因是表征冗余**：根据 Platonic Representation Hypothesis（ICML 2024），不同预训练模型趋同到相似的统计表示，融合它们等于复制信息而非增加信息
3. **互补性 > 数量**：CLIP（对比学习）+ DINO（自监督蒸馏）训练目标正交，带来最大融合收益（+3-8pp）；相似训练范式的模型（MAE、Data2Vec）带来冗余
4. **冗余结构是任务依赖的**：不同数据集需要不同的融合策略，没有万能方法
5. **Few-shot 放大了冗余问题**：复杂路由方法（MoE）在 few-shot 下因路由器欠拟合而表现不佳，full data 下应能更有效地消除冗余

## 研究方向

### 已完成

- 10 个预训练模型的 scaling 实验（模型数量 1→6 递增）
- 17 种融合方法的横向对比
- 3 种动态路由方法的有效性验证
- Top-K Router 的 k 值消融实验

### 进行中

- **Few-shot vs Full-data 对比**：验证 few-shot 是否是复杂融合方法表现不佳的根本原因
- **CKA-Guided Model Selection**：用 CKA（Centered Kernel Alignment）量化模型间表征相似度，基于互补性而非数量选择最优模型子集

详细实验结果见 [experiments/RESULTS.md](experiments/RESULTS.md)。

## 快速开始

```bash
# 环境
conda create -n feature_cls python=3.10 -y && conda activate feature_cls
pip install -r requirements.txt

# 设置路径
export STORAGE_DIR=/path/to/bigfiles

# 下载模型和数据
python download_models.py --models --storage_dir "$STORAGE_DIR"
python download_models.py --all_datasets --storage_dir "$STORAGE_DIR"
```

## 使用方法

```bash
# 单模型
python main.py --dataset cifar100 --model clip --storage_dir "$STORAGE_DIR"

# 多模型融合（17 种方法通过 --fusion_method 选择）
python main.py --dataset cifar100 --model fusion \
    --fusion_method gated --fusion_models clip,dino \
    --storage_dir "$STORAGE_DIR"

# Full data 训练（关闭 few-shot）
python main.py --dataset cifar100 --model fusion \
    --fusion_method moe_router --fusion_models clip,dino,mae,siglip,convnext,data2vec \
    --storage_dir "$STORAGE_DIR" --disable_fewshot --epochs 20
```

## 实验脚本

```bash
# Few-shot vs Full-data 对比实验（核心验证实验）
STORAGE_DIR=$STORAGE_DIR bash experiments/run_fulldata_vs_fewshot.sh --quick  # 3 数据集
STORAGE_DIR=$STORAGE_DIR bash experiments/run_fulldata_vs_fewshot.sh --full   # 7 数据集

# 其他实验
bash experiments/run_experiments.sh --scaling    # 模型数量递增
bash experiments/run_experiments.sh --methods    # 融合方法对比
```

## 模型与方法

**10 个预训练模型**：ViT, Swin, BEiT, Data2Vec, MAE, DINOv2, CLIP, OpenCLIP, SigLIP, ConvNeXt

**17 种融合方法**（`--fusion_method`）：

| 类别 | 方法 |
|------|------|
| 简单融合 | `concat` `proj_concat` `weighted_sum` `gated` |
| 交互融合 | `difference_concat` `hadamard_concat` `bilinear_concat` |
| 注意力融合 | `film` `context_gating` `lmf` `se_fusion` `late_fusion` |
| Token 级 | `comm` `mmvit` |
| 动态路由 | `topk_router` `moe_router` `attention_router` |

## 理论框架

本研究的发现可以从三个理论视角统一解释：

- **Platonic Representation Hypothesis**（Huh et al., ICML 2024）：预训练模型趋同到相似表示
- **Information Bottleneck**（Kawaguchi et al., ICML 2023）：特征维度线性增长而有用信息边际递减
- **Bias-Variance-Diversity Decomposition**（Wood et al., JMLR 2023）：高相关性模型集成几乎不降低方差

## 项目结构

```
├── main.py                    # 入口：参数解析 + 调度
├── configs/config.py          # 数据集配置
├── src/
│   ├── models/
│   │   ├── extractor.py       # 基础特征提取器
│   │   ├── classifier.py      # MLP 分类器
│   │   └── fusion/            # 17 种融合方法
│   ├── data/dataset.py        # 数据加载与 few-shot 采样
│   └── training/
│       ├── trainer.py         # 统一训练循环
│       ├── cache.py           # 离线特征缓存
│       └── results.py         # 结果记录
└── experiments/
    ├── RESULTS.md             # 实验结果与分析
    ├── run_experiments.sh     # 基础实验脚本
    └── run_fulldata_vs_fewshot.sh  # Few-shot vs Full-data 对比
```

## 参考文献

- Huh et al. *"The Platonic Representation Hypothesis"*, ICML 2024
- Kawaguchi et al. *"How Does Information Bottleneck Help Deep Learning?"*, ICML 2023
- Wood et al. *"A Unified Theory of Diversity in Ensemble Learning"*, JMLR 2023
- Kornblith et al. *"Similarity of Neural Network Representations Revisited"* (CKA), ICML 2019
- Wu et al. *"Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks"*, ICML 2022

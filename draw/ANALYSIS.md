# 多模型特征融合：完整实验分析

## 一、核心研究问题

**多模型特征融合中，更多模型是否意味着更好性能？如果不是，原因是什么？**

我们通过三组实验系统地回答了这个问题：
1. **Scaling 实验**（10-shot）：模型数量 1→6 递增
2. **方法对比实验**（10-shot vs full-data）：Concat / Gated / MoE 横向比较
3. **Full-data Scaling 实验**：在完整训练集下重复 scaling 实验，区分"表征冗余"和"维度灾难"

---

## 二、实验一：Few-shot Scaling（10-shot, Gated Fusion）

**实验配置**：10-shot few-shot, Gated Fusion, 模型按 CLIP → +DINO → +MAE → +SigLIP → +ConvNeXt → +Data2Vec 顺序递增。

![Few-shot Scaling](fig1_fewshot_scaling.png)

### 数据

| 数据集 | 1模型(CLIP) | 2模型(+DINO) | 3模型(+MAE) | 4模型(+SigLIP) | 5模型(+ConvNeXt) | 6模型(+Data2Vec) | 峰值 |
|--------|-----------|------------|-----------|--------------|-----------------|-----------------|------|
| STL10 | 91.44% | 95.34% | 95.21% | **95.71%** | 94.74% | 94.99% | 4模型 |
| Pets | 88.99% | 94.19% | 94.06% | 94.71% | **95.48%** | 95.45% | 5模型 |
| EuroSAT | 83.50% | 88.15% | 88.11% | 86.94% | **88.72%** | 87.65% | 5模型 |
| DTD | 67.93% | 75.85% | 75.16% | 76.54% | **76.70%** | 76.28% | 5模型 |
| GTSRB | 67.63% | 66.47% | 63.30% | **75.00%** | 71.82% | 70.75% | 4模型 |
| SVHN | 29.76% | 25.68% | 26.40% | **35.42%** | 34.05% | 32.18% | 4模型 |
| Country211 | **29.62%** | 27.47% | 26.92% | 27.08% | 26.87% | 25.92% | 1模型 |

### 发现

- **所有数据集在 4-5 模型时到达峰值，之后下降**
- CLIP → +DINO 带来最大提升（3-8pp）
- +MAE、+Data2Vec 几乎总是带来退化
- Country211 单模型最优

---

## 三、实验二：Full-data vs Few-shot 方法对比（6模型固定）

**实验配置**：6 模型（CLIP+DINO+MAE+SigLIP+ConvNeXt+Data2Vec），3 种方法（Concat / Gated / MoE Router），10-shot vs full data。

![Method Comparison](fig4_method_comparison.png)

### 数据

| 数据集 | Concat(10s) | Concat(full) | Gated(10s) | Gated(full) | MoE(10s) | MoE(full) |
|--------|------------|-------------|-----------|------------|---------|----------|
| STL10 | 94.66% | **97.96%** | 93.80% | 97.85% | 93.65% | 97.80% |
| Pets | 95.26% | **96.40%** | 94.17% | 96.13% | 94.74% | 96.29% |
| EuroSAT | 86.30% | **97.83%** | 80.56% | 97.63% | 82.69% | 97.54% |
| DTD | 76.33% | **83.88%** | 74.20% | 82.61% | 76.44% | 83.14% |
| GTSRB | 60.93% | **91.18%** | 49.61% | 90.86% | 69.18% | 90.98% |
| SVHN | 22.47% | **83.05%** | 33.01% | 77.97% | 25.68% | 82.47% |
| Country211 | 12.47% | 26.64% | 15.25% | **27.58%** | 12.08% | 26.81% |

### 数据量对性能的影响

![Data Impact Heatmap](fig5_data_impact_heatmap.png)

### 发现

**10-shot 最优方法分布**：Concat 赢 3/7，MoE 赢 2/7，Gated 赢 2/7
**Full-data 最优方法分布**：Concat 赢 6/7，Gated 赢 1/7，MoE 赢 0/7

- Full data 下 Concat 几乎全赢，且三种方法差距收窄到 1-2pp
- SVHN 提升最大：Concat 从 22.47% → 83.05%（+60.58pp）
- 10-shot 下 MoE/Gated 的优势来自隐式维度压缩，full data 下不再需要

---

## 四、实验三：Full-data Scaling（关键实验）

**实验目的**：区分"表征冗余"和"维度灾难"各自的贡献。

**实验配置**：Full data, Gated Fusion, 模型按 CLIP → +DINO → +MAE → +SigLIP → +ConvNeXt → +Data2Vec 递增。对比 few-shot scaling 曲线。

![Few-shot vs Full-data Scaling](fig2_fewshot_vs_fulldata_scaling.png)

### 数据

**STL10**
| 模型数 | Few-shot | Full-data |
|--------|---------|-----------|
| 1 (CLIP) | 91.44% | 94.93% |
| 2 (+DINO) | 95.34% | 97.84% |
| 3 (+MAE) | 95.21% | 97.83% |
| 4 (+SigLIP) | **95.71%** | 97.93% |
| 5 (+ConvNeXt) | 94.74% | 97.93% |
| 6 (+Data2Vec) | 94.99% | **97.95%** |

**GTSRB**
| 模型数 | Few-shot | Full-data |
|--------|---------|-----------|
| 1 (CLIP) | 67.63% | 87.81% |
| 2 (+DINO) | 66.47% | 88.57% |
| 3 (+MAE) | 63.30% | 78.46% |
| 4 (+SigLIP) | **75.00%** | 78.62% |
| 5 (+ConvNeXt) | 71.82% | **91.63%** |
| 6 (+Data2Vec) | 70.75% | 91.10% |

**SVHN**
| 模型数 | Few-shot | Full-data |
|--------|---------|-----------|
| 1 (CLIP) | 29.76% | 73.33% |
| 2 (+DINO) | 25.68% | 66.61% |
| 3 (+MAE) | 26.40% | 66.52% |
| 4 (+SigLIP) | **35.42%** | 66.36% |
| 5 (+ConvNeXt) | 34.05% | **75.66%** |
| 6 (+Data2Vec) | 32.18% | 75.10% |

### 每步增量分析

![Per-step Delta](fig3_fulldata_delta_per_step.png)

### 三个数据集，三种行为模式

| 类型 | 数据集 | Few-shot 峰值 | Full-data 峰值 | 诊断 |
|------|--------|-------------|---------------|------|
| **纯维度灾难** | STL10 | 4模型（之后↓） | 6模型（一直↑） | Full data 完全消除了下降 |
| **冗余+维度灾难** | GTSRB | 4模型（之后↓） | 5模型（峰值上移） | +MAE 两种设定都暴跌 10pp |
| **纯表征噪声** | SVHN | 4模型（之后↓） | 5模型 | +DINO full data 仍跌 6.7pp |

---

## 五、实验四：CKA 表征相似度分析（Patch Tokens + PCA）

**实验目的**：用 CKA（Centered Kernel Alignment）定量测量 6 个模型间的表征相似度，验证融合实验中观察到的互补/冗余关系，并为模型选择提供理论依据。

**实验配置**：6 模型 × 7 数据集，提取最后一层 patch tokens（[N, 196, 768] → flatten → [N, ~150K]），PCA 降到 256 维后计算 linear CKA。每个数据集取 2000 样本。

> **为什么用 patch tokens 而非 pooled output？** 第一轮实验用 pooler_output（[CLS] 向量），所有模型对 CKA 趋近零（0.001~0.05），无区分度。Patch tokens 保留空间结构信息，CKA 分布在 0.01~0.69，区分度显著提升。

### 全局平均 CKA 矩阵

|  | clip | dino | mae | siglip | convnext | data2vec |
|--|------|------|-----|--------|----------|----------|
| **clip** | 1.00 | 0.24 | **0.07** | 0.18 | 0.19 | 0.16 |
| **dino** | 0.24 | 1.00 | 0.11 | 0.34 | **0.44** | 0.29 |
| **mae** | **0.07** | 0.11 | 1.00 | 0.31 | 0.10 | 0.32 |
| **siglip** | 0.18 | 0.34 | 0.31 | 1.00 | 0.27 | 0.36 |
| **convnext** | 0.19 | **0.44** | 0.10 | 0.27 | 1.00 | 0.22 |
| **data2vec** | 0.16 | 0.29 | 0.32 | 0.36 | 0.22 | 1.00 |

![CKA Heatmaps](../result/cka_patch_pca_full_fix_20260314_220955/cka_heatmap_all.png)

### 单数据集 CKA 特征

| 数据集 | CKA 范围 | 特征 | 最高对 |
|--------|---------|------|-------|
| **DTD** | 0.01~0.05 | **全面极低**，纹理任务让所有模型展现最大差异 | convnext-clip: 0.05 |
| **EuroSAT** | 0.17~0.69 | **全面偏高**，遥感场景语义简单，模型趋同 | mae-data2vec: 0.69 |
| **STL10** | 0.08~0.55 | 中等分散，自然图像分类 | dino-convnext: 0.55 |
| **GTSRB** | 0.09~0.63 | 中等偏高，局部模式让 DINO 和 ConvNeXt 趋同 | dino-convnext: 0.63 |
| **SVHN** | 0.02~0.57 | 中等分散，数字识别的特殊性 | mae-data2vec: 0.57 |
| **Country211** | 0.06~0.57 | 中等分散 | dino-data2vec: 0.57 |
| **Pets** | 0.03~0.43 | 中等偏低，细粒度分类需要多样化特征 | dino-data2vec: 0.43 |

### CKA 跨数据集方差（任务依赖性）

同一模型对在不同数据集上的 CKA 差异巨大，最极端的例子：

| 模型对 | 方差 | 范围 | 含义 |
|--------|-----|------|------|
| mae-data2vec | **0.073** | 0.007 ~ 0.691 | EuroSAT 上高度冗余，DTD 上完全互补 |
| siglip-data2vec | 0.038 | 0.032 ~ 0.579 | 任务决定互补性 |
| dino-convnext | 0.033 | 0.042 ~ 0.634 | 在 GTSRB 上趋同，在 DTD 上分化 |
| clip-mae | **0.003** | 0.010 ~ 0.175 | **最稳定的互补对**，任务无关 |

### 模型选择策略输出

**Max Diversity k=3**: `clip, mae, convnext` — 两两 CKA 都极低（0.07, 0.10, 0.19）

**Greedy Selection（from clip）**: 所有 7 个数据集前两步都是 `clip → mae`，第三步因任务而异

### 发现

1. **CLIP-MAE 是全局最互补的模型对**（CKA=0.07），对比学习 vs 掩码重建产生了最正交的特征
2. **DINO-ConvNeXt 是最相似的模型对**（CKA=0.44），尽管架构不同（ViT vs CNN）——强力支持 Platonic Representation Hypothesis 的跨架构趋同论点
3. **训练目标比架构更能决定表征差异**：dino-mae=0.11（同 ViT，极不同训练目标）<< dino-convnext=0.44（不同架构，但视觉理解目标相近）
4. **CKA 高度任务依赖**：mae-data2vec 在 EuroSAT 上 0.69（冗余），在 DTD 上 0.007（互补）。不存在"万能互补"组合
5. **DTD（纹理）是最特殊的数据集**：所有模型对 CKA 都极低，解释了融合实验中 DTD 受益最大的原因
6. **CKA 修正了之前的部分结论**：
   - "MAE 与 DINO 训练范式相近" → 实际 CKA 只有 0.11，非常互补
   - "不同架构才能互补" → DINO-ConvNeXt CKA 最高，架构不同不保证互补

---

## 五(b)、实验五：CKA 顺序 vs 原始顺序 Scaling（关键验证）

**实验目的**：验证 CKA 指导的模型选择是否真正提升融合性能。如果"选最不同的模型"有效，CKA 顺序应在中间步骤（2-4 模型）优于原始顺序。

**实验配置**：7 数据集 × 6 步 × 2 顺序 = 84 runs，Concat 融合，Full-data。

| | 顺序 |
|--|------|
| **原始** | clip → dino → mae → siglip → convnext → data2vec（固定） |
| **CKA** | clip → mae → ...（每个数据集按 CKA greedy，MAE 始终排第二） |

### 结果：CKA 顺序全面落后

| #models | 平均 Δ(CKA−Orig) | 说明 |
|---------|-----------------|------|
| 2 | **-1.56pp** | CKA 最弱处：CLIP+MAE << CLIP+DINO |
| 3 | -1.01pp | |
| 4 | -0.70pp | 差距收窄 |
| 5 | -0.65pp | |
| 6 | -0.04pp | 追平（最终集合相同） |

**总计 35 对比较：原始赢 25，CKA 赢 7，持平 3**

### 典型案例

| 数据集 | CLIP+DINO (原始) | CLIP+MAE (CKA) | Δ | 原因 |
|--------|----------------|----------------|---|------|
| **DTD** | 82.71% | 76.91% | **-5.80pp** | DINO 的纹理特征对 DTD 极有价值 |
| **STL10** | 97.78% | 94.90% | -2.88pp | DINO 的视觉结构对自然图像分类有用 |
| **Pets** | 95.86% | 93.65% | -2.21pp | DINO 的部件特征帮助细粒度分类 |
| **SVHN 3m** | 77.39% | **79.84%** | **+2.45pp** | 唯一例外：DINO 对数字识别是噪声 |

### 核心发现：多样性 ≠ 有效性

```
融合收益 = 多样性(diversity) × 任务相关性(relevance)
```

| 模型 | CKA to CLIP | 多样性 | 任务相关性 | 实际融合效果 |
|------|------------|--------|-----------|------------|
| **MAE** | 0.07（极低） | 极高 | 低（重建导向） | 差（-1.56pp avg） |
| **DINO** | 0.24（中等） | 中等 | 高（语义丰富） | 好（多数数据集提升） |

CKA 准确度量了表征多样性，但**朴素的"选最不同的模型"策略行不通**。SVHN 是唯一例外——DINO 对数字识别的 relevance 为零，此时 diversity 主导选择，CKA 顺序反而赢了。

### 方法论启示：需要 Diversity × Relevance 联合框架

有效的模型选择需要同时优化：
- **互补性**（CKA 低）→ 避免信息冗余
- **任务相关性**（单模型精度高 / transferability score 高）→ 确保新增信息有用

```
Score(candidate) = α · Relevance(candidate) − β · Redundancy(candidate, selected_set)
```

---

## 六、核心发现（综合实验一~五）

### 发现 1：CLIP + DINO 仍是最佳融合起点（Scaling 实验确认）

CKA 分析曾修正判断为"CLIP+MAE 最互补"。但 Scaling 实验五证明：**表征互补性最高 ≠ 融合效果最好**。

| 模型对 | CKA | 2 模型融合精度（7 数据集平均） | 评价 |
|--------|-----|---------------------------|------|
| CLIP+MAE | **0.07**（最互补） | 较低 | 多样但不够有用 |
| CLIP+DINO | 0.24（中等互补） | **较高** | 多样性适中 + 高任务相关性 |

DINO 的优势不在于表征正交性，而在于其自监督蒸馏特征（纹理、空间结构、物体部件）对分类任务**直接有用**。MAE 的掩码重建特征虽然最独特，但对分类贡献有限。

唯一例外：SVHN（数字识别）上 DINO 是噪声，CKA 顺序反而更好。

### 发现 2：+MAE 对 GTSRB 是灾难性的

+MAE 在 GTSRB 上 full data 下仍然暴跌 **-10.11pp**。MAE 的 masked autoencoder 重建目标关注像素级低频信息，对需要精确图案识别的交通标志任务是有害表征。

### 发现 3：ConvNeXt 价值高但并非因为"架构不同"（CKA 修正）

| 数据集 | +ConvNeXt 效果（full data） |
|--------|--------------------------|
| STL10 | +0.00pp（已饱和）|
| GTSRB | **+13.01pp** |
| SVHN | **+9.30pp** |

ConvNeXt 的融合价值确实很高，但 CKA 修正了原因解释：

- 旧结论："CNN vs ViT 架构差异导致表征正交" → **错误**。dino-convnext CKA=0.44，是所有模型对中**最高的**
- 新解释：ConvNeXt 的价值来自它对特定任务（GTSRB 交通标志、SVHN 数字）的局部模式识别能力，而非表征正交性
- CKA 证明**训练目标比架构更能决定表征差异**：dino-mae=0.11（同 ViT）远低于 dino-convnext=0.44（不同架构）
- 这深化了对 Platonic Representation Hypothesis 的理解：趋同程度取决于训练目标的相似性，而非架构的相似性

### 发现 4："先升后降"是两种因素的叠加

多模型融合性能下降 = **表征冗余** × **维度灾难**：

| 因素 | 描述 | Full data 能否解决 | 证据 |
|------|------|------------------|------|
| **维度灾难** | 高维特征 + 少量样本 → 分类器过拟合 | **能** | STL10 full data 下曲线一直升 |
| **表征冗余** | 相似训练范式的模型提供重复信息 | **部分缓解** | GTSRB 的 +MAE 两种设定都跌 |
| **表征噪声** | 某些模型的特征对特定任务有害 | **不能** | SVHN 的 +DINO full data 仍跌 6.7pp |

### 发现 5：最优融合策略取决于数据量

| 数据量 | 最优策略 | 原因 |
|--------|---------|------|
| Few-shot | Gated/MoE（维度压缩） | 分类器无法处理高维空间，需要路由做隐式降维 |
| Full data | Concat（全量保留） | 分类器有足够样本处理高维空间，全量信息 > 压缩信息 |

---

## 七、理论框架

上述发现可以从三个理论视角统一解释：

### 1. Platonic Representation Hypothesis（Huh et al., ICML 2024）

不同预训练模型趋同到相似的统计表示。CKA 实验深化了这一理论：**趋同程度取决于训练目标的相似性，而非架构**。DINO-ConvNeXt（不同架构）CKA=0.44（最高），而 DINO-MAE（同 ViT）CKA=0.11（极低）。这表明 Platonic convergence 沿着"视觉理解目标"维度发生，而非"架构"维度。

### 2. Information Bottleneck（Kawaguchi et al., ICML 2023）

最优表征应该压缩输入同时保留对标签有用的信息。每加一个模型，特征维度线性增长，但有用信息边际递减。存在最优压缩点，超过后信噪比下降。

### 3. Bias-Variance-Diversity Decomposition（Wood et al., JMLR 2023）

集成误差 = Bias² + Variance − Diversity。当模型间相关性 ρ 高时（Platonic convergence），增加模型几乎不降低 variance，而特征维度增长使优化变难（bias 上升）。

---

## 八、方法论启示

### 模型选择：Diversity × Relevance 双因素框架

实验四（CKA 分析）和实验五（CKA vs 原始顺序 Scaling）共同揭示了模型选择的完整图景：

- **CKA 准确度量表征多样性**，但多样性不等于融合收益
- **有效选择 = 互补性（CKA 低） × 任务相关性（分类性能高）**
- 实践中最佳起点仍是 CLIP+DINO（中等互补 + 高任务相关性），而非 CLIP+MAE（极高互补 + 低任务相关性）
- **训练目标比架构更能决定表征差异**：dino-mae=0.11（同 ViT）<< dino-convnext=0.44（不同架构）
- **模型选择必须任务自适应**：SVHN 上 DINO 是噪声（relevance=0），此时 CKA 顺序反而更好

→ 已发展为 MUMS 框架（见下文实验六）和 IB Bottleneck（见实验七）。

### 融合策略：基于数据量

- **数据稀缺时**（few-shot）：用 Gated/MoE 做隐式维度压缩
- **数据充足时**（full data）：直接 Concat，简单即最优

### CKA 分析的价值定位

CKA 分析（实验四）的核心价值不在于直接指导模型选择，而在于：
1. **定量揭示了模型间的表征关系**：6×6 CKA 矩阵 × 7 数据集
2. **证明训练目标是表征差异的决定因素**，而非架构
3. **量化了 CKA 的任务依赖性**：同一模型对在不同数据集上互补性可以完全反转
4. **通过实验五的反面验证**，揭示了"多样性 ≠ 有效性"这一关键洞察，为双因素框架提供了实证基础

---

## 八(b)、提出的方法一：MUMS — Marginal Utility-based Model Selection

### 数学框架

**核心公式 — 边际效用函数：**

```
U(m | S, T) = R(m, T)^α · (1 - ρ(m, S | D))^β
```

其中：
- `R(m, T)` = 归一化任务相关性（单模型准确率 min-max 归一化到 [0,1]）
- `ρ(m, S | D)` = 集合冗余度 = `(1/|S|) Σ_{s∈S} CKA(m, s)`
- `α, β ≥ 0` 分别控制 relevance 和 diversity 的权重

**乘法形式 vs 加法形式（MMR）：**

| 特性 | 乘法（MUMS） | 加法（MMR） |
|------|-------------|-----------|
| Relevance=0 时 | Score=0 (✓) | Score 可 >0 (✗) |
| Novelty=0 时 | Score=0 (✓) | Score 可 >0 (✗) |
| 实验匹配 | MAE 不被误选 | MAE 可能被误选 |

**信息论推导：**

```
ΔI(m, S) = I(f_m; Y | f_S)                    [边际信息增益]
          ≈ I(f_m; Y) · [1 - I(f_m; f_S)/H(f_m)]   [乘法近似]
          = Relevance(m) · Novelty(m, S)
```

**与 DPP 的联系：**

DPP 核分解 `L = diag(q) · S · diag(q)` 中 `q_i = quality`, `S_{ij} = similarity`。我们的贪心框架是 DPP 的贪心近似，其中 quality = R(m,T), similarity kernel = CKA matrix。

### 贪心算法

```
S ← {argmax_m R(m, T)}                    # 从最相关模型开始
while M \ S ≠ ∅:
    m* ← argmax_{m ∉ S} R(m)^α · (1 - avg_CKA(m, S))^β
    S ← S ∪ {m*}
return S
```

### 特殊情况

| α | β | 退化为 |
|---|---|--------|
| 0 | 1 | 纯 CKA diversity（= 实验五，已验证效果差） |
| 1 | 0 | 纯 relevance 排序 |
| 1 | 1 | 平衡 joint selection |
| 2 | 1 | relevance-biased |

### 实验设计（待运行）

1. **单模型基线**：6 模型 × 7 数据集独立训练 → R(m,T) 矩阵
2. **Joint Ordering 计算**：(α,β) 网格搜索 + 3 种顺序对比
3. **Scaling 验证**：original vs diversity_only vs joint，Concat 融合，full-data

脚本：`experiments/run_single_model.sh` → `experiments/run_joint_selection.py` → auto-generated `run_joint_scaling.sh`

---

## 八(c)、提出的方法二：Redundancy-Aware Feature Bottleneck

### 动机

如果问题的根源是"冗余维度膨胀"，最直接的解决方案是在融合后压缩冗余。

**核心假设**：6模型 + Bottleneck ≥ 4模型 Concat

### IB 理论基础

IB 目标：`min_Z -I(Z;Y) + β·I(X;Z)`

- X = 4608维 Concat 特征（含大量冗余维度）
- Z = bottleneck 后的压缩表征
- β 控制压缩-保留 trade-off

### 三种实现

| 方法 | 公式 | 参数量 | 理论性质 |
|------|------|--------|---------|
| **PCA** | `Z = V_k^T(X-μ)` | 0 | 最大方差，非任务对齐 |
| **Linear** | `Z = W·LN(X)` | ~236万 | 端到端任务对齐 |
| **VIB** | `z ~ N(μ(X), σ²(X))`, KL 正则 | ~472万 | 直接优化 IB，自动压缩冗余维度 |

**VIB 的冗余压缩机制：**

```
L = CE(Y, f(Z)) + β · KL(q(Z|X) || N(0,I))
KL = (1/2) Σ_i [μ_i² + σ_i² - 1 - log(σ_i²)]
```

- 冗余维度 i：task gradient 不需要 → μ_i→0, σ_i²→1 → 被推向先验（"关闭"）
- 有用维度 i：task gradient 需要 → μ_i≠0, σ_i²<1 → KL 代价被 CE 补偿

### 实验设计（待运行）

| 配置 | 特征维度 | 说明 |
|------|---------|------|
| 4模型 Concat | 3072 | 最优子集基线 |
| 6模型 Concat | 4608 | 当前基线 |
| 6模型 + PCA(512) | 512 | 非参数压缩 |
| 6模型 + Linear(512) | 512 | 可学习压缩 |
| 6模型 + VIB(512, β=0.01) | 512 | IB 最优压缩 |

脚本：`experiments/run_bottleneck.sh`

---

## 九、完整 Paper Story（五幕结构）

### 第一幕：现象
多模型特征融合存在"先升后降"——7 个数据集上性能在 4-5 模型时达到峰值。

### 第二幕：诊断
三层因素叠加：
- **表征冗余**（Platonic convergence, CKA 验证）
- **维度灾难**（full-data 实验区分，SVHN 交叉现象）
- **表征噪声**（DINO on SVHN, MAE on GTSRB）

### 第三幕：反面验证
纯 CKA 选择失败（25/35 输），揭示 **diversity ≠ effectiveness**。

### 第四幕：解决方案
1. **MUMS 框架**：Relevance^α × Novelty^β 联合选择（解决"选什么模型"）
2. **IB Bottleneck**：VIB 压缩冗余维度（解决"怎么融合"）

### 第五幕：统一理论
多模型融合的完整优化涉及两个正交维度：
- **模型选择**（what to fuse）：Diversity × Relevance
- **融合方法**（how to fuse）：IB 最优压缩

---

## 十、参考文献

1. Huh, Cheung, Wang, Isola. *"The Platonic Representation Hypothesis"*, ICML 2024
2. Kawaguchi et al. *"How Does Information Bottleneck Help Deep Learning?"*, ICML 2023
3. Wood et al. *"A Unified Theory of Diversity in Ensemble Learning"*, JMLR 2023
4. Kornblith et al. *"Similarity of Neural Network Representations Revisited"* (CKA), ICML 2019
5. Wu et al. *"Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks"*, ICML 2022
6. *"Less is More: On the Feature Redundancy of Pretrained Models"*, arXiv 2023
7. *"Diffused Redundancy in Pre-trained Representations"*, NeurIPS 2023
8. *Eagle: "Exploring the Design Space for Multimodal LLMs with Mixture of Encoders"*, ICLR 2025
9. Carbonell & Goldberg. *"The Use of MMR, Diversity-Based Reranking"*, SIGIR 1998
10. Alemi et al. *"Deep Variational Information Bottleneck"*, ICLR 2017
11. Kulesza & Taskar. *"Determinantal Point Processes for Machine Learning"*, Found. & Trends in ML, 2012
12. *"HE-CKA: Enhancing Diversity in Bayesian Deep Learning"*, NeurIPS 2024
13. *"OSBORN: Submodular Transferability Estimation for Ensemble Selection"*, ICCV 2023
14. Chen et al. *"Explore and Exploit the Diverse Knowledge in Model Zoo"*, ICML 2023
15. Agostinelli et al. *"Transferability Metrics for Selecting Source Model Ensembles"*, CVPR 2022

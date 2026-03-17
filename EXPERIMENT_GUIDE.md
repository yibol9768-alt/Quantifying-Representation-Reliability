# 实验补充指南

## 一、环境准备

```bash
pip install torch numpy scipy
```

确认导入无报错：

```bash
python3 -c "from src.scoring import greedy_select; print('OK')"
```

---

## 二、数据准备

### 2.1 预提取特征

需要对每个数据集、每个模型提取冻结特征，保存为如下结构：

```
data/features/
├── dtd/
│   ├── clip.pt          # [N, d] float tensor
│   ├── dino.pt
│   ├── mae.pt
│   ├── siglip.pt
│   ├── convnext.pt
│   ├── data2vec.pt
│   └── labels.pt        # [N] int tensor
├── eurosat/
│   ├── clip.pt
│   └── ...
├── flowers102/
├── food101/
├── pets/
├── sun397/
└── ucf101/
```

如果已有提特征的脚本（`src/models/extractor.py`），直接跑即可。关键要求：

- 每个 `.pt` 文件是 `torch.save(features, path)`，shape 为 `[N, d]`
- `labels.pt` 是 `[N]` 的整数 tensor，和特征行对齐
- 所有模型在同一数据集上的 N 必须相同

### 2.2 LEEP 所需的 softmax 输出（可选）

如果要跑 LEEP 对比，还需要每个源模型在目标数据上的 softmax 概率：

```
data/source_probs/
├── dtd/
│   ├── clip.pt      # [N, C_source] float tensor，每行 sum=1
│   └── ...
```

> LEEP 不是核心方法，优先级低。如果来不及可以先跳过。

---

## 三、实验清单

### 实验 1：选择方法对比（核心，必做）

**目标**：比较 9 种选择方法在 7 个数据集上的模型排序 + 最终融合准确率。

**步骤**：

```bash
# Step 1: 跑选择（几分钟内完成，不需要 GPU）
python3 experiments/run_selection_comparison.py \
    --data_root data/features \
    --datasets dtd,eurosat,flowers102,food101,pets,sun397,ucf101 \
    --max_models 6 \
    --output_dir result/selection_comparison
```

这会输出每种方法的模型排序到 `result/selection_comparison/`。

```bash
# Step 2: 对每种选择方法的排序，从 k=1 累加到 k=6 跑融合训练
# 伪代码示意，需要接到现有训练脚本：
for method in Ours_LogME_CKA GBC_CKA HScore_CKA LogME_SVCCA mRMR JMI Relevance_Only Random All_Models; do
    ordering=$(python3 -c "import json; d=json.load(open('result/selection_comparison/${dataset}.json')); print(' '.join(d['${method}']['selected']))")
    for k in 1 2 3 4 5 6; do
        models=$(echo $ordering | cut -d' ' -f1-$k)
        # 用现有训练脚本跑融合，记录准确率
        # python3 train.py --models $models --dataset $dataset ...
    done
done
```

**预期产出**：

| 选择方法 | DTD | EuroSAT | Flowers | Food101 | Pets | SUN397 | UCF101 | 平均 |
|---------|-----|---------|---------|---------|------|--------|--------|------|
| Ours (LogME+CKA) | | | | | | | | |
| GBC+CKA | | | | | | | | |
| H-Score+CKA | | | | | | | | |
| LogME+SVCCA | | | | | | | | |
| mRMR | | | | | | | | |
| JMI | | | | | | | | |
| Relevance Only | | | | | | | | |
| Random | | | | | | | | |
| All Models | | | | | | | | |

每格填该方法最优 k 处的准确率。

---

### 实验 2：评分与准确率相关性（必做）

**目标**：验证 LogME/GBC/H-Score 与单模型准确率正相关。

**步骤**：

```python
from src.scoring import logme_score, gbc_score, hscore
import numpy as np, torch, json

dataset = "dtd"  # 对每个数据集重复
features = {}
for model in ["clip", "dino", "mae", "siglip", "convnext", "data2vec"]:
    features[model] = torch.load(f"data/features/{dataset}/{model}.pt").numpy()
labels = torch.load(f"data/features/{dataset}/labels.pt").numpy()

scores = {}
for model, feat in features.items():
    scores[model] = {
        "logme": logme_score(feat, labels),
        "gbc": gbc_score(feat, labels),
        "hscore": hscore(feat, labels),
        # "single_acc": <从已有结果中读取该模型在该数据集上的单模型准确率>
    }

print(json.dumps(scores, indent=2))
```

**预期产出**：
- 散点图：横轴 LogME/GBC/H-Score，纵轴单模型准确率
- Spearman 相关系数（`scipy.stats.spearmanr`）

---

### 实验 3：λ 敏感性（必做）

**目标**：验证 LogME+CKA 方法对 λ 参数不太敏感。

**步骤**：

```python
from src.scoring import greedy_select

for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
    selected, _ = greedy_select(
        features, labels,
        relevance_method="logme",
        redundancy_method="cka",
        selection_method="relevance_redundancy",
        lambda_param=lam,
        max_models=6,
    )
    print(f"lambda={lam}: {selected}")
    # 然后对每个 lambda 的排序跑融合训练，记录最优 k 处准确率
```

**预期产出**：折线图，横轴 λ，纵轴各数据集的准确率。

---

### 实验 4：选择×融合交叉（如果时间够）

**目标**：验证选择方法的效果不依赖特定融合器。

选 3 种选择方法（Ours / mRMR / Random）× 3 种融合方式（concat / attention / bottleneck），共 9 组，在 2-3 个数据集上跑。

---

### 实验 5：选择效率统计（顺手记录）

跑实验 1 的时候 `run_selection_comparison.py` 会自动输出每种方法的耗时，直接从 JSON 结果里提取即可。

---

## 四、结果汇总格式

所有数值结果放到 `result/` 下，建议结构：

```
result/
├── selection_comparison/
│   ├── dtd.json
│   ├── eurosat.json
│   └── all_results.json
├── fusion_with_selection/
│   └── {method}_{dataset}_k{1-6}.json
├── correlation/
│   └── score_vs_accuracy.json
└── lambda_sensitivity/
    └── lambda_sweep.json
```

---

## 五、优先级

1. **P0（必须有）**：实验 1（选择方法对比表）+ 实验 2（评分相关性）
2. **P1（强烈建议）**：实验 3（λ 敏感性）
3. **P2（加分项）**：实验 4（交叉实验）、实验 5（效率）

---

## 六、有问题找谁

- 选择模块代码在 `src/scoring/`，入口是 `greedy_select()`
- 理论部分在 `latex/main.tex`
- 如果特征提取有问题看 `src/models/extractor.py`

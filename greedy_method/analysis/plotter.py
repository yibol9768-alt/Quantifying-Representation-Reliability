"""
analysis/plotter.py
===================
可视化工具，生成三张图：

  图① incremental_accuracy.png  — 增量 few-shot 准确率曲线（贪心搜索结果）
  图② fusion_weight_heatmap.png — 融合权重热力图（贪心搜索结果）
  图③ few_shot_accuracy.png     — K-shot 精度条形图（最终评估结果）
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# 图① 增量 few-shot 准确率曲线
# ---------------------------------------------------------------------------
def plot_incremental_accuracy(
    selection_order: List[str],
    val_accuracies:  List[float],
    test_accuracies: List[float],
    dataset_name:    str,
    save_path:       str,
    k_shot:          int = 5,
) -> None:
    """
    以贪心步骤为 X 轴，绘制 val / test few-shot 情节准确率随模型数量的变化曲线。
    X 轴标签为每步实际加入的模型名称。
    """
    k_values = list(range(1, len(selection_order) + 1))
    val_pct  = [a * 100 for a in val_accuracies]
    test_pct = [a * 100 for a in test_accuracies]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(k_values, val_pct,  marker="o", linewidth=2,
            label=f"Val {k_shot}-shot Acc",  color="#2196F3")
    ax.plot(k_values, test_pct, marker="s", linewidth=2,
            label=f"Test {k_shot}-shot Acc", color="#4CAF50")

    ax.set_xticks(k_values)
    ax.set_xticklabels(selection_order, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Model Added (Greedy Order)", fontsize=11)
    ax.set_ylabel(f"Few-Shot Accuracy (%) [{k_shot}-shot]", fontsize=11)
    ax.set_title(
        f"Incremental Few-Shot Accuracy – {dataset_name.upper()}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    for x, y in zip(k_values, test_pct):
        ax.annotate(
            f"{y:.1f}",
            xy=(x, y), xytext=(0, 6),
            textcoords="offset points",
            ha="center", fontsize=7, color="#4CAF50",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图①] 已保存 → {save_path}")


# ---------------------------------------------------------------------------
# 图② 融合权重热力图
# ---------------------------------------------------------------------------
def plot_fusion_weight_heatmap(
    fusion_weights:  List[Dict[str, float]],
    selection_order: List[str],
    dataset_name:    str,
    save_path:       str,
) -> None:
    """
    行 = 贪心步骤 k，列 = 模型，格子值 = softmax 权重 α_i。
    尚未加入集合的格子显示为空白（NaN）。
    """
    n_steps    = len(fusion_weights)
    all_models = selection_order

    matrix = np.full((n_steps, len(all_models)), np.nan)
    for step_idx, weight_dict in enumerate(fusion_weights):
        for col_idx, model_name in enumerate(all_models):
            if model_name in weight_dict:
                matrix[step_idx, col_idx] = weight_dict[model_name]

    fig, ax = plt.subplots(
        figsize=(min(14, len(all_models) * 1.4), n_steps * 0.7 + 1.5)
    )

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels([f"k={i+1}" for i in range(n_steps)], fontsize=9)
    ax.set_title(
        f"Fusion Weight Evolution – {dataset_name.upper()}",
        fontsize=12, fontweight="bold",
    )

    plt.colorbar(im, ax=ax, label="α_i (softmax weight)")

    for r in range(n_steps):
        for c in range(len(all_models)):
            val = matrix[r, c]
            if not np.isnan(val):
                ax.text(
                    c, r, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black" if val < 0.6 else "white",
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图②] 已保存 → {save_path}")


# ---------------------------------------------------------------------------
# 图③ Few-Shot 精度条形图
# ---------------------------------------------------------------------------
def plot_few_shot_accuracy(
    k_values:             List[int],
    accuracies:           List[float],
    confidence_intervals: List[float],
    dataset_name:         str,
    model_label:          str,
    save_path:            str,
) -> None:
    """
    不同 K 值下的 few-shot 准确率条形图，附 95% 置信区间误差线。
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    x   = np.arange(len(k_values))
    pct = [a * 100 for a in accuracies]
    ci  = [c * 100 for c in confidence_intervals]

    bars = ax.bar(
        x, pct, yerr=ci,
        color="#9C27B0", alpha=0.80,
        capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#555"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}-shot" for k in k_values], fontsize=10)
    ax.set_xlabel("K (shots per class)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(
        f"Few-Shot Accuracy – {dataset_name.upper()}  [{model_label}]",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, acc, ci_val in zip(bars, pct, ci):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_val + 0.5,
            f"{acc:.1f}±{ci_val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图③] 已保存 → {save_path}")

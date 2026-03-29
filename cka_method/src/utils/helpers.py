"""
工具函数：seed 设置、日志、可视化。
"""
import os
import random
import json
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ──────────────── 随机种子 ────────────────

def set_seed(seed: int = 42):
    """设置全局随机种子以确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────── 结果保存 ────────────────

def save_results(
    output_dir: str,
    dataset: str,
    n_shot: int,
    method: str,
    ordered_models: List[str],
    accuracies: List[float],
    weight_logs: List[Dict[str, float]],
    relevance: Dict[str, float] = None,
    cka_matrix: np.ndarray = None,
    model_names: List[str] = None,
    ci_95s: List[float] = None,
):
    """保存实验结果到 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "dataset": dataset,
        "n_shot": n_shot,
        "method": method,
        "ordered_models": ordered_models,
        "accuracies": accuracies,
        "weight_logs": weight_logs,
        "best_k": int(np.argmax(accuracies)) + 1,
        "best_acc": float(max(accuracies)),
    }
    if ci_95s is not None:
        result["ci_95s"] = ci_95s
        result["best_ci_95"] = float(ci_95s[int(np.argmax(accuracies))])
    if relevance is not None:
        result["relevance_scores"] = relevance
    if cka_matrix is not None and model_names is not None:
        result["cka_matrix"] = {
            "model_names": model_names,
            "values": cka_matrix.tolist(),
        }

    path = os.path.join(output_dir, f"{dataset}_{n_shot}shot_{method}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")
    return path


# ──────────────── 可视化 ────────────────

def plot_accuracy_curve(
    results: Dict[str, List[float]],
    dataset: str,
    n_shot: int,
    output_dir: str = "./outputs",
    ci_95s: Dict[str, List[float]] = None,
):
    """
    绘制性能增量曲线 Acc(k) vs k。

    Args:
        results:  {method_name: [Acc(1), Acc(2), ..., Acc(N)]}
        dataset:  数据集名称
        n_shot:   shot 数
        output_dir: 保存路径
        ci_95s:   可选，{method_name: [ci_95(1), ..., ci_95(N)]} 用于误差棒
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for method, accs in results.items():
        ks = list(range(1, len(accs) + 1))
        if ci_95s is not None and method in ci_95s:
            errs = ci_95s[method]
            plt.errorbar(ks, accs, yerr=errs, marker="o", label=method,
                         linewidth=2, capsize=4)
        else:
            plt.plot(ks, accs, marker="o", label=method, linewidth=2)

    plt.xlabel("Number of Models (k)", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title(f"{dataset} | {n_shot}-shot | Acc(k) vs k", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max(len(v) for v in results.values()) + 1))

    path = os.path.join(output_dir, f"{dataset}_{n_shot}shot_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    model_names: List[str],
    dataset: str,
    output_dir: str = "./outputs",
):
    """绘制 CKA 相似度热力图"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    # 标注数值
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{cka_matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cka_matrix[i,j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="CKA Similarity")
    plt.title(f"CKA Similarity Matrix | {dataset}", fontsize=13)

    path = os.path.join(output_dir, f"{dataset}_cka_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CKA heatmap saved to {path}")


def plot_shot_comparison(
    all_shot_results: Dict,
    dataset: str,
    output_dir: str = "./outputs",
):
    """
    汇总图：不同 shot 的 Acc(k) 曲线画在一张图上。

    Args:
        all_shot_results: {n_shot: {"accuracies": [...], "best_k": int, ...}}
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for n_shot, r in sorted(all_shot_results.items()):
        accs = r["accuracies"]
        ks = list(range(1, len(accs) + 1))
        best_k = r["best_k"]
        plt.plot(ks, accs, marker="o", linewidth=2,
                 label=f"{n_shot}-shot (k*={best_k}, Acc={r['best_acc']:.4f})")
        # 标记最优点
        plt.scatter([best_k], [r["best_acc"]], s=120, zorder=5,
                    edgecolors="black", linewidths=1.5)

    plt.xlabel("Number of Models (k)", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title(f"{dataset} | All Shots Comparison", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 11))

    path = os.path.join(output_dir, f"{dataset}_all_shots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Shot comparison plot saved to {path}")


def save_summary(
    all_shot_results: Dict,
    dataset: str,
    output_dir: str = "./outputs",
):
    """保存所有 shot 的汇总 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    summary = {"dataset": dataset, "shots": {}}
    best_shot, best_acc = None, -1.0

    for n_shot, r in sorted(all_shot_results.items()):
        summary["shots"][n_shot] = {
            "best_k": r["best_k"],
            "best_acc": r["best_acc"],
            "all_accuracies": r["accuracies"],
            "ordered_models": r["ordered_models"],
            "optimal_models": r["ordered_models"][:r["best_k"]],
        }
        if r["best_acc"] > best_acc:
            best_acc = r["best_acc"]
            best_shot = n_shot

    summary["best_overall"] = {
        "n_shot": best_shot,
        "best_k": all_shot_results[best_shot]["best_k"],
        "best_acc": best_acc,
        "optimal_models": all_shot_results[best_shot]["ordered_models"][
            :all_shot_results[best_shot]["best_k"]
        ],
    }

    path = os.path.join(output_dir, f"{dataset}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {path}")


def plot_weight_evolution(
    weight_logs: List[Dict[str, float]],
    ordered_models: List[str],
    dataset: str,
    n_shot: int,
    output_dir: str = "./outputs",
):
    """绘制融合权重随模型数量 k 的演化"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ordered_models:
        weights = []
        for k, wlog in enumerate(weight_logs):
            weights.append(wlog.get(model, 0.0))
        ks = list(range(1, len(weight_logs) + 1))
        ax.plot(ks, weights, marker="s", label=model, linewidth=1.5)

    ax.set_xlabel("Number of Models (k)", fontsize=13)
    ax.set_ylabel("Fusion Weight α", fontsize=13)
    ax.set_title(f"{dataset} | {n_shot}-shot | Weight Evolution", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"{dataset}_{n_shot}shot_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weight evolution plot saved to {path}")

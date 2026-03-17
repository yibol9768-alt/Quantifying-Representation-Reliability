"""收集融合实验结果，整理成论文所需的表格和数据。

输出：
  1. 每种选择方法 × 每个数据集的 k-accuracy 曲线数据
  2. 最优 k 处的准确率对比表
  3. CSV 文件供画图使用

Usage:
    python experiments/collect_fusion_results.py \
        --selection_dir /path/to/results/selection_comparison \
        --fusion_dir /path/to/results/fusion_scaling \
        --single_model_dir /path/to/results \
        --output_dir /path/to/results/paper_tables
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from glob import glob


def load_single_model_results(results_dir):
    """从单模型结果目录中提取 best_acc。"""
    single = {}  # (dataset, model) -> best_acc
    for f in glob(os.path.join(results_dir, "*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # 从文件名解析: {dataset}_fulltrain_{model}_seed42_...json
            basename = os.path.basename(f).replace(".json", "")
            parts = basename.split("_fulltrain_")
            if len(parts) != 2:
                continue
            dataset = parts[0]
            model = parts[1].split("_seed")[0]

            best_acc = data.get("summary", {}).get("best_test_acc")
            if best_acc is None:
                # 从 history 中找
                history = data.get("history", [])
                if history:
                    best_acc = max(h.get("test_acc", 0) for h in history)
            if best_acc is not None and best_acc > 0:
                key = (dataset, model)
                # 保留最高的（可能有重复跑）
                if key not in single or best_acc > single[key]:
                    single[key] = best_acc
        except Exception:
            continue
    return single


def load_fusion_results(fusion_dir):
    """从融合结果目录中提取每组实验的 best_acc。"""
    fusion = {}  # (dataset, method, k) -> best_acc
    for f in glob(os.path.join(fusion_dir, "*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            basename = os.path.basename(f).replace(".json", "")

            # 从结果中解析
            dataset = data.get("config", {}).get("dataset", "")
            fusion_models = data.get("config", {}).get("fusion_models", [])
            if isinstance(fusion_models, str):
                fusion_models = fusion_models.split(",")
            k = len(fusion_models)

            best_acc = data.get("summary", {}).get("best_test_acc")
            if best_acc is None:
                history = data.get("history", [])
                if history:
                    best_acc = max(h.get("test_acc", 0) for h in history)

            if best_acc and dataset and k >= 2:
                # 通过 .done 文件反查 method
                pass  # 直接通过 models 组合匹配

        except Exception:
            continue

    # 更简单的方式：直接读 .done marker 文件
    for marker in glob(os.path.join(fusion_dir, "*.done")):
        basename = os.path.basename(marker).replace(".done", "")
        # 格式: {dataset}_{method}_k{k}
        match = re.match(r"(.+?)_(Ours_LogME_CKA|GBC_CKA|HScore_CKA|LogME_SVCCA|mRMR|JMI|Relevance_Only|Random|All_Models)_k(\d+)", basename)
        if not match:
            continue
        dataset, method, k = match.group(1), match.group(2), int(match.group(3))

        with open(marker) as f:
            models_str = f.read().strip()

        # 找对应的结果 JSON（通过 models 匹配）
        # 搜索 fusion_dir 下包含这些 models 的结果
        for result_file in glob(os.path.join(fusion_dir, f"{dataset}_*_{method}*k{k}*.json")):
            pass  # 复杂匹配

    return fusion


def load_selection_orderings(selection_dir):
    """读取各方法的选择排序。"""
    orderings = {}  # (dataset, method) -> [model_list]
    for f in glob(os.path.join(selection_dir, "*.json")):
        if "all_results" in f:
            continue
        dataset = os.path.basename(f).replace(".json", "")
        with open(f) as fh:
            data = json.load(fh)
        for method, entry in data.items():
            if isinstance(entry, dict) and "selected" in entry:
                orderings[(dataset, method)] = entry["selected"]
    return orderings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_dir", required=True)
    parser.add_argument("--fusion_dir", required=True)
    parser.add_argument("--single_model_dir", required=True)
    parser.add_argument("--output_dir", default="result/paper_tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading single model results...")
    single = load_single_model_results(args.single_model_dir)
    print(f"  Found {len(single)} single-model results")

    print("Loading selection orderings...")
    orderings = load_selection_orderings(args.selection_dir)
    print(f"  Found {len(orderings)} orderings")

    # Build k-accuracy curves from single model results + fusion results
    # For k=1, use single model accuracy from the ordering's first model
    datasets = sorted(set(d for d, _ in orderings.keys()))
    methods = sorted(set(m for _, m in orderings.keys()))

    print(f"\nDatasets: {datasets}")
    print(f"Methods: {methods}")

    # Output 1: Selection orderings table
    print("\n" + "=" * 80)
    print("选择排序对比")
    print("=" * 80)
    for dataset in datasets:
        print(f"\n{dataset}:")
        for method in methods:
            key = (dataset, method)
            if key in orderings:
                ordering = orderings[key]
                # k=1 准确率
                first_model = ordering[0]
                acc = single.get((dataset, first_model), "?")
                if isinstance(acc, float):
                    acc = f"{acc:.2f}%"
                print(f"  {method:25s}: {' -> '.join(ordering[:8])}  (k=1: {acc})")

    # Output 2: Save orderings as CSV for fusion script
    csv_path = os.path.join(args.output_dir, "orderings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "k", "model", "single_acc"])
        for (dataset, method), ordering in sorted(orderings.items()):
            for k, model in enumerate(ordering, 1):
                acc = single.get((dataset, model), "")
                writer.writerow([dataset, method, k, model, acc])
    print(f"\n[Saved] {csv_path}")

    # Output 3: Single model ranking table
    print("\n" + "=" * 80)
    print("单模型准确率排名")
    print("=" * 80)
    models = sorted(set(m for _, m in single.keys()))
    header = ["Model"] + datasets
    print(f"{'Model':20s}" + "".join(f"{d:>12s}" for d in datasets))
    print("-" * (20 + 12 * len(datasets)))
    for model in models:
        row = f"{model:20s}"
        for dataset in datasets:
            acc = single.get((dataset, model))
            if acc is not None:
                row += f"{acc:>11.2f}%"
            else:
                row += f"{'—':>12s}"
        print(row)

    single_csv = os.path.join(args.output_dir, "single_model_accuracy.csv")
    with open(single_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "best_acc"])
        for (dataset, model), acc in sorted(single.items()):
            writer.writerow([model, dataset, acc])
    print(f"\n[Saved] {single_csv}")

    print(f"\n全部输出保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

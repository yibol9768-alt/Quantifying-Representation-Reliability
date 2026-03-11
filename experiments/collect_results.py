#!/usr/bin/env python3
"""收集和整理融合实验结果

用法:
    python collect_results.py --results_dir ./results --output results_table.csv
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict


def parse_run_name(run_name: str) -> dict:
    """解析run名称，提取实验信息

    例如: cifar100_fusion-gated_clip,dino_seed42_offline-cache_20250101_120000
    """
    info = {
        "dataset": "unknown",
        "fusion_method": "none",
        "models": "unknown",
        "num_models": 0,
        "is_single": False,
    }

    parts = run_name.split("_")

    # 解析数据集
    if parts:
        info["dataset"] = parts[0]

    # 解析是否为融合
    if "fusion" in run_name:
        # 提取融合方法
        for part in parts:
            if part.startswith("fusion-"):
                info["fusion_method"] = part.replace("fusion-", "")
                break

        # 提取模型列表
        for i, part in enumerate(parts):
            if part == "fusion" and i + 1 < len(parts):
                # 下一个部分可能是模型列表或方法
                if parts[i + 1].startswith("fusion-"):
                    # 方法名，继续找模型
                    for j in range(i + 2, len(parts)):
                        if not parts[j].startswith("seed"):
                            info["models"] = parts[j]
                            break
                        else:
                            break
                else:
                    info["models"] = parts[i + 1]
                break
    else:
        # 单模型
        info["is_single"] = True
        for part in parts:
            if part in ["mae", "clip", "dino", "vit", "swin", "beit", "deit",
                       "convnext", "eva", "mae_large", "clip_large", "dino_large",
                       "openclip", "sam", "albef"]:
                info["models"] = part
                info["num_models"] = 1
                break

    # 计算模型数量
    if not info["is_single"] and info["models"] != "unknown":
        info["num_models"] = len(info["models"].split(","))

    return info


def extract_best_acc(json_file: Path) -> float:
    """从JSON结果文件中提取最佳准确率"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get("summary", {}).get("best_acc", 0.0)
    except Exception as e:
        print(f"Warning: Could not read {json_file}: {e}")
        return 0.0


def collect_results(results_dir: Path) -> list:
    """收集所有实验结果"""
    results = []

    for json_file in results_dir.glob("*.json"):
        run_name = json_file.stem
        info = parse_run_name(run_name)

        best_acc = extract_best_acc(json_file)

        result = {
            "run_name": run_name,
            "dataset": info["dataset"],
            "num_models": info["num_models"],
            "models": info["models"],
            "fusion_method": info["fusion_method"],
            "best_acc": best_acc,
            "is_single": info["is_single"],
        }
        results.append(result)

    return results


def format_results_table(results: list) -> str:
    """格式化结果表格（Markdown格式）"""
    # 按模型数量和方法分组
    single_results = [r for r in results if r["is_single"]]
    fusion_results = [r for r in results if not r["is_single"]]

    lines = []
    lines.append("# Fusion Experiments Results")
    lines.append("")
    lines.append(f"Generated at: {Path(__file__).stat().st_mtime}")
    lines.append("")

    # 单模型结果
    lines.append("## 1. Single Model Baselines")
    lines.append("")
    lines.append("| Model | Best Acc |")
    lines.append("|-------|----------|")
    for r in sorted(single_results, key=lambda x: -x["best_acc"]):
        lines.append(f"| {r['models']} | {r['best_acc']:.2f}% |")
    lines.append("")

    # 按模型数量分组
    by_num_models = defaultdict(list)
    for r in fusion_results:
        by_num_models[r["num_models"]].append(r)

    # 多模型融合结果
    lines.append("## 2. Fusion Results by Number of Models")
    lines.append("")

    for num_models in sorted(by_num_models.keys()):
        results_list = by_num_models[num_models]
        lines.append(f"### {num_models} Model(s)")
        lines.append("")
        lines.append("| Fusion Method | Models | Best Acc |")
        lines.append("|----------------|--------|----------|")

        # 按准确率排序
        sorted_results = sorted(results_list, key=lambda x: -x["best_acc"])
        for r in sorted_results:
            lines.append(f"| {r['fusion_method']} | {r['models']} | {r['best_acc']:.2f}% |")
        lines.append("")

    # 方法横向对比
    lines.append("## 3. Cross-Method Comparison (All Model Counts)")
    lines.append("")
    lines.append("| Method | 2-Models | 3-Models | 4-Models | 5-Models | 6+ Models |")
    lines.append("|--------|----------|----------|----------|----------|-----------|")

    # 收集每种方法在不同模型数量下的最佳结果
    method_by_count = defaultdict(lambda: defaultdict(float))
    for r in fusion_results:
        count = r["num_models"]
        method = r["fusion_method"]
        acc = r["best_acc"]
        if acc > method_by_count[method][count]:
            method_by_count[method][count] = acc

    methods = sorted(method_by_count.keys())
    for method in methods:
        counts = method_by_count[method]
        row = [f"| {method} |"]
        for count in [2, 3, 4, 5]:
            row.append(f" {counts.get(count, 0):.2f}% |")
        # 6+ models取最佳
        best_6plus = max([v for k, v in counts.items() if k >= 6], default=0)
        row.append(f" {best_6plus:.2f}% |")
        lines.append("".join(row))
    lines.append("")

    # 最佳结果汇总
    lines.append("## 4. Best Results Summary")
    lines.append("")
    lines.append("| Category | Best Acc | Configuration |")
    lines.append("|----------|----------|---------------|")

    categories = [
        ("Single Model", max([r["best_acc"] for r in single_results], default=0),
         "Best single model"),
        ("2 Models", max([r["best_acc"] for r in by_num_models[2]], default=0),
         "Best 2-model fusion"),
        ("3 Models", max([r["best_acc"] for r in by_num_models[3]], default=0),
         "Best 3-model fusion"),
        ("4-5 Models", max([r["best_acc"] for r in by_num_models[4] + by_num_models[5]], default=0),
         "Best 4-5 model fusion"),
        ("6+ Models", max([r["best_acc"] for k in range(6, 11) for r in by_num_models[k]], default=0),
         "Best 6+ model fusion"),
        ("Overall Best", max([r["best_acc"] for r in results], default=0),
         "Best overall"),
    ]

    for cat, acc, desc in categories:
        lines.append(f"| {cat} | {acc:.2f}% | {desc} |")
    lines.append("")

    return "\n".join(lines)


def write_csv(results: list, output_file: Path):
    """写入CSV格式结果"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Run Name", "Dataset", "Num Models", "Models", "Fusion Method", "Best Acc", "Is Single"])

        for r in sorted(results, key=lambda x: (x["num_models"], x["fusion_method"])):
            writer.writerow([
                r["run_name"],
                r["dataset"],
                r["num_models"],
                r["models"],
                r["fusion_method"],
                f"{r['best_acc']:.2f}",
                r["is_single"],
            ])


def main():
    parser = argparse.ArgumentParser(description="Collect fusion experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output", type=str, default="results_table.csv",
                        help="Output CSV file path")
    parser.add_argument("--markdown", type=str, default=None,
                        help="Output Markdown file path (optional)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Collecting results from: {results_dir}")
    results = collect_results(results_dir)
    print(f"Found {len(results)} results")

    # 写入CSV
    csv_file = Path(args.output)
    write_csv(results, csv_file)
    print(f"CSV saved to: {csv_file}")

    # 写入Markdown
    md_file = Path(args.markdown) if args.markdown else results_dir / "results_summary.md"
    md_content = format_results_table(results)
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"Markdown saved to: {md_file}")

    # 打印简要统计
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)

    single_results = [r for r in results if r["is_single"]]
    fusion_results = [r for r in results if not r["is_single"]]

    if single_results:
        best_single = max(single_results, key=lambda x: x["best_acc"])
        print(f"Best Single Model: {best_single['models']} ({best_single['best_acc']:.2f}%)")

    if fusion_results:
        best_fusion = max(fusion_results, key=lambda x: x["best_acc"])
        print(f"Best Fusion: {best_fusion['fusion_method']} with {best_fusion['num_models']} models ({best_fusion['best_acc']:.2f}%)")
        print(f"  Models: {best_fusion['models']}")

    print("="*50)


if __name__ == "__main__":
    main()

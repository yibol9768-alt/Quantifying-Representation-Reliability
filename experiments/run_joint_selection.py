#!/usr/bin/env python3
"""Joint Diversity × Relevance Model Selection Analysis.

Reads CKA matrices and single-model accuracies, then computes joint-selected
orderings for each dataset under various (α, β) settings.

Outputs:
    - joint_selection_results.json: all orderings and traces
    - joint_ordering_comparison.txt: summary table for quick inspection
    - Per-dataset ordering tables

Usage:
    python experiments/run_joint_selection.py \
        --cka_dir result/cka_patch_pca_full_fix_20260314_220955 \
        --results_dir /path/to/storage/results \
        --output_dir result/joint_selection
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.joint_selection import (
    normalize_relevance,
    joint_greedy_selection,
    compare_orderings,
    compute_cumulative_scores,
)


MODELS = ["clip", "dino", "mae", "siglip", "convnext", "data2vec"]
DATASETS = ["stl10", "pets", "eurosat", "dtd", "gtsrb", "svhn", "country211"]

ORIGINAL_ORDER = ["clip", "dino", "mae", "siglip", "convnext", "data2vec"]


def load_cka_matrices(cka_dir: str) -> dict:
    """Load per-dataset CKA matrices from CSV files."""
    matrices = {}
    model_names = None
    for ds in DATASETS:
        csv_path = os.path.join(cka_dir, f"cka_matrix_{ds}.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping {ds}")
            continue
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            names = header[1:]  # skip first empty column
            data = []
            for row in reader:
                data.append([float(x) for x in row[1:]])
            matrices[ds] = np.array(data)
            if model_names is None:
                model_names = names
    return matrices, model_names


def collect_single_model_accs(results_dir: str) -> dict:
    """Collect single-model best accuracies from result JSON files.

    Scans results_dir for files matching the pattern:
        {dataset}_fulltrain_{model}_seed42_*.json

    Returns:
        {dataset: {model: best_acc_percent}}
    """
    accs = {ds: {} for ds in DATASETS}
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"  Warning: results_dir {results_dir} not found")
        return accs

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            dataset = cfg.get("dataset", data.get("dataset", ""))
            model = cfg.get("model_type") or cfg.get("model") or data.get("model", "")

            # Skip fusion results
            if model == "fusion":
                continue
            if dataset not in accs:
                continue
            if model not in MODELS:
                continue

            # Check it's full-data (not few-shot)
            if "disable_fewshot" in cfg:
                fewshot = not bool(cfg.get("disable_fewshot"))
            else:
                fewshot = cfg.get("fewshot", True)
            if fewshot:
                continue

            summary = data.get("summary", {})
            best_acc = data.get("best_accuracy")
            if best_acc is None:
                best_acc = summary.get("best_acc", 0.0)
            # Keep the best across seeds/runs
            if model not in accs[dataset] or best_acc > accs[dataset][model]:
                accs[dataset][model] = best_acc
        except (json.JSONDecodeError, KeyError):
            continue

    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cka_dir", type=str, required=True,
                        help="Directory with CKA CSV matrices")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with single-model result JSONs")
    parser.add_argument("--output_dir", type=str, default="result/joint_selection")
    parser.add_argument("--alpha", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Relevance exponent values to sweep")
    parser.add_argument("--beta", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Diversity exponent values to sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CKA matrices
    print("Loading CKA matrices...")
    cka_matrices, model_names = load_cka_matrices(args.cka_dir)
    print(f"  Loaded {len(cka_matrices)} datasets, models: {model_names}")

    # Load single-model accuracies
    print("\nCollecting single-model accuracies...")
    single_accs = collect_single_model_accs(args.results_dir)
    for ds in DATASETS:
        if ds in single_accs and single_accs[ds]:
            items = ", ".join(f"{m}={v:.2f}%" for m, v in sorted(single_accs[ds].items()))
            print(f"  {ds}: {items}")
        else:
            print(f"  {ds}: NO DATA (need to run single-model baselines first)")

    # Check if we have enough data
    datasets_ready = [ds for ds in DATASETS if ds in cka_matrices and len(single_accs.get(ds, {})) >= 4]
    if not datasets_ready:
        print("\nERROR: No datasets have both CKA matrices and single-model accuracies.")
        print("Run 'bash experiments/run_single_model.sh' first to get single-model baselines.")
        sys.exit(1)

    print(f"\nReady datasets: {datasets_ready}")

    # --- Run joint selection for each dataset ---
    all_results = {}
    summary_lines = []

    for ds in datasets_ready:
        cka = cka_matrices[ds]
        raw_accs = single_accs[ds]

        # Normalize relevance
        rel = normalize_relevance(raw_accs)

        print(f"\n{'='*60}")
        print(f"  {ds.upper()}")
        print(f"{'='*60}")
        print(f"  Raw accuracies: {raw_accs}")
        print(f"  Normalized relevance: { {m: f'{v:.3f}' for m,v in rel.items()} }")

        ds_results = {"raw_acc": raw_accs, "normalized_relevance": rel, "orderings": {}}

        # Key orderings to compare
        key_orderings = {}

        # 1. Original order
        key_orderings["original"] = ORIGINAL_ORDER

        # 2. Pure diversity (CKA-only, from our previous experiment)
        div_order, div_trace = joint_greedy_selection(
            cka, model_names, rel, alpha=0.0, beta=1.0, start_model="clip",
        )
        key_orderings["diversity_only"] = div_order
        ds_results["orderings"]["diversity_only"] = {"order": div_order, "trace": div_trace}

        # 3. Pure relevance
        rel_order, rel_trace = joint_greedy_selection(
            cka, model_names, rel, alpha=1.0, beta=0.0,
        )
        key_orderings["relevance_only"] = rel_order
        ds_results["orderings"]["relevance_only"] = {"order": rel_order, "trace": rel_trace}

        # 4. Joint (α=1, β=1) — our proposed method
        joint_order, joint_trace = joint_greedy_selection(
            cka, model_names, rel, alpha=1.0, beta=1.0,
        )
        key_orderings["joint_a1_b1"] = joint_order
        ds_results["orderings"]["joint_a1_b1"] = {"order": joint_order, "trace": joint_trace}

        # 5. Joint (α=2, β=1) — relevance-biased
        joint2_order, joint2_trace = joint_greedy_selection(
            cka, model_names, rel, alpha=2.0, beta=1.0,
        )
        key_orderings["joint_a2_b1"] = joint2_order
        ds_results["orderings"]["joint_a2_b1"] = {"order": joint2_order, "trace": joint2_trace}

        # 6. Joint (α=1, β=2) — diversity-biased
        joint3_order, joint3_trace = joint_greedy_selection(
            cka, model_names, rel, alpha=1.0, beta=2.0,
        )
        key_orderings["joint_a1_b2"] = joint3_order
        ds_results["orderings"]["joint_a1_b2"] = {"order": joint3_order, "trace": joint3_trace}

        # Alpha-beta grid
        grid_results = {}
        for a in args.alpha:
            for b in args.beta:
                if a == 0.0 and b == 0.0:
                    continue
                label = f"a{a:.1f}_b{b:.1f}"
                order, trace = joint_greedy_selection(
                    cka, model_names, rel, alpha=a, beta=b,
                )
                grid_results[label] = {"order": order, "trace": trace}
        ds_results["orderings"]["grid"] = grid_results

        all_results[ds] = ds_results

        # Print comparison table
        print(f"\n  Ordering comparison:")
        print(f"  {'Method':<20} {'Order'}")
        print(f"  {'-'*20} {'-'*50}")
        for label, order in key_orderings.items():
            print(f"  {label:<20} {' → '.join(order)}")

        # Print joint trace
        print(f"\n  Joint (α=1,β=1) trace:")
        print(f"  {'Step':<5} {'Model':<10} {'Relevance':<10} {'Novelty':<10} {'Utility':<10}")
        for t in joint_trace:
            print(f"  {t['step']:<5} {t['model']:<10} {t['relevance']:<10.3f} {t['novelty']:<10.3f} {t['utility']:<10.4f}")

        summary_lines.append(f"\n{ds.upper()}")
        for label, order in key_orderings.items():
            summary_lines.append(f"  {label:<20} {' → '.join(order)}")

    # Save results
    results_path = os.path.join(args.output_dir, "joint_selection_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    summary_path = os.path.join(args.output_dir, "joint_ordering_comparison.txt")
    with open(summary_path, "w") as f:
        f.write("Joint Diversity × Relevance Model Selection\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_path}")

    # Generate shell script for scaling experiments with joint ordering
    _generate_scaling_script(all_results, args.output_dir)


def _generate_scaling_script(all_results: dict, output_dir: str):
    """Generate a shell script to run scaling experiments with joint-selected orderings."""
    script_lines = [
        '#!/bin/bash',
        '# Joint Selection Scaling Experiment (auto-generated)',
        '# Compares: original / diversity_only / joint_a1_b1 orderings',
        '#',
        '# Usage:',
        '#   STORAGE_DIR=/path/to/bigfiles bash experiments/run_joint_scaling.sh',
        '',
        'set -euo pipefail',
        '',
        'STORAGE_DIR="${STORAGE_DIR:?Please set STORAGE_DIR}"',
        'EPOCHS="${EPOCHS:-20}"',
        'BATCH_SIZE="${BATCH_SIZE:-128}"',
        'SEED="${SEED:-42}"',
        'CACHE_DTYPE="${CACHE_DTYPE:-fp32}"',
        '',
        'BASE_CMD="python main.py --storage_dir $STORAGE_DIR --epochs $EPOCHS '
        '--batch_size $BATCH_SIZE --seed $SEED --cache_dtype $CACHE_DTYPE"',
        '',
        'run_one() {',
        '    local dataset=$1',
        '    local models=$2',
        '    local n_models=$3',
        '    local order_tag=$4',
        '',
        '    echo ""',
        '    echo "============================================================"',
        '    echo "  $dataset | ${n_models}m ($models) | $order_tag"',
        '    echo "============================================================"',
        '',
        '    if [ "$n_models" -eq 1 ]; then',
        '        $BASE_CMD --dataset "$dataset" --model "${models}" \\',
        '            --disable_fewshot \\',
        '            || echo "FAILED: $dataset ${n_models}m $order_tag"',
        '    else',
        '        $BASE_CMD --dataset "$dataset" --model fusion \\',
        '            --fusion_method concat --fusion_models "$models" \\',
        '            --disable_fewshot \\',
        '            || echo "FAILED: $dataset ${n_models}m $order_tag"',
        '    fi',
        '}',
        '',
    ]

    # Generate per-dataset ordering functions
    for ds, ds_data in all_results.items():
        orderings = ds_data["orderings"]

        # Get key orderings
        orig = ["clip", "dino", "mae", "siglip", "convnext", "data2vec"]
        div_only = orderings.get("diversity_only", {}).get("order", orig)
        joint = orderings.get("joint_a1_b1", {}).get("order", orig)

        def _steps(order):
            """Build cumulative model strings: clip, clip,dino, clip,dino,mae, ..."""
            steps = []
            for i in range(len(order)):
                steps.append(",".join(order[:i+1]))
            return steps

        script_lines.append(f'run_{ds}() {{')
        script_lines.append(f'    echo "=== {ds.upper()} ==="')

        for tag, order in [("orig", orig), ("div", div_only), ("joint", joint)]:
            steps = _steps(order)
            for i, step in enumerate(steps):
                script_lines.append(f'    run_one "{ds}" "{step}" "{i+1}" "{tag}"')

        script_lines.append('}')
        script_lines.append('')

    # Main
    script_lines.append('echo "=== Joint Selection Scaling Experiment ==="')
    script_lines.append('')
    for ds in all_results:
        script_lines.append(f'run_{ds}')
    script_lines.append('')
    script_lines.append('echo ""')
    script_lines.append('echo "============================================================"')
    script_lines.append('echo "  Joint scaling experiments complete!"')
    script_lines.append('echo "  Results in: $STORAGE_DIR/results/"')
    script_lines.append('echo "============================================================"')

    script_path = os.path.join(output_dir, "run_joint_scaling.sh")
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines) + "\n")
    os.chmod(script_path, 0o755)
    print(f"Generated scaling script: {script_path}")


if __name__ == "__main__":
    main()

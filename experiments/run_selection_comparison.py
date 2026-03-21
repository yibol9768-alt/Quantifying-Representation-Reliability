"""Compare model selection methods across datasets.

This script runs all selection methods in the comparison matrix and
reports the selected model orderings, enabling downstream fusion
experiments to evaluate which selection strategy leads to best accuracy.

Usage:
    python experiments/run_selection_comparison.py \
        --data_root /path/to/features \
        --datasets dtd,eurosat,flowers102 \
        --max_models 6 \
        --output_dir result/selection_comparison

Expected feature directory layout:
    {data_root}/{dataset}/{split}/{model_name}.pt   # [N, d] tensor
    {data_root}/{dataset}/{split}/labels.pt         # [N] integer tensor
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.scoring.selection import greedy_select


SELECTION_CONFIGS = [
    # (name, relevance, redundancy, selection_method, lambda)
    ("Ours_LogME_CKA", "logme", "cka", "relevance_redundancy", 1.0),
    ("GBC_CKA", "gbc", "cka", "relevance_redundancy", 1.0),
    ("HScore_CKA", "hscore", "cka", "relevance_redundancy", 1.0),
    ("LogME_SVCCA", "logme", "svcca", "relevance_redundancy", 1.0),
    ("mRMR", "logme", "cka", "mrmr", 1.0),
    ("JMI", "logme", "cka", "jmi", 1.0),
    ("Relevance_Only", "logme", "cka", "relevance_only", 1.0),
    ("Random", "logme", "cka", "random", 1.0),
]


def _flatten_feature_array(feat: torch.Tensor) -> np.ndarray:
    """Normalize serialized feature tensors to [N, D] for scoring."""
    if feat.ndim <= 2:
        return feat.numpy().astype(np.float64)
    if all(dim == 1 for dim in feat.shape[2:]):
        feat = feat.flatten(1)
    else:
        feat = feat.reshape(feat.size(0), -1)
    return feat.numpy().astype(np.float64)


def load_features(data_root: str, dataset: str, split: str):
    """Load pre-extracted features and labels for one dataset split."""
    dataset_dir = os.path.join(data_root, dataset, split)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    labels_path = os.path.join(dataset_dir, "labels.pt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    labels = torch.load(labels_path, map_location="cpu").numpy()

    features = {}
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.endswith(".pt") and fname != "labels.pt":
            model_name = fname.replace(".pt", "")
            feat = torch.load(os.path.join(dataset_dir, fname), map_location="cpu")
            features[model_name] = _flatten_feature_array(feat)

    return features, labels


def run_comparison(features, labels, max_models):
    """Run all selection methods and return results."""
    results = {}
    all_models = sorted(features.keys())

    for config_name, rel, red, sel, lam in SELECTION_CONFIGS:
        print(f"  Running {config_name}...", end=" ", flush=True)
        t0 = time.time()

        try:
            selected, metadata = greedy_select(
                features=features,
                labels=labels,
                relevance_method=rel,
                redundancy_method=red,
                selection_method=sel,
                max_models=max_models,
                lambda_param=lam,
            )
            elapsed = time.time() - t0
            results[config_name] = {
                "selected": selected,
                "time_seconds": round(elapsed, 2),
                "config": {
                    "relevance": rel,
                    "redundancy": red,
                    "selection": sel,
                    "lambda": lam,
                },
            }
            print(f"done ({elapsed:.1f}s) -> {selected}")
        except Exception as e:
            print(f"FAILED: {e}")
            results[config_name] = {"error": str(e)}

    # Add "All Models" baseline
    results["All_Models"] = {
        "selected": all_models,
        "time_seconds": 0.0,
        "config": {"selection": "all"},
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare model selection methods")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing per-dataset feature dirs")
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated list of dataset names")
    parser.add_argument("--max_models", type=int, default=10,
                        help="Maximum number of models to select")
    parser.add_argument("--output_dir", type=str, default="result/selection_comparison",
                        help="Output directory for results")
    parser.add_argument("--selection_split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Dataset split used to compute selection statistics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",")]

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        try:
            features, labels = load_features(args.data_root, dataset, args.selection_split)
            print(f"  Loaded {len(features)} models, {len(labels)} samples, "
                  f"{len(np.unique(labels))} classes from split={args.selection_split}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        results = run_comparison(features, labels, args.max_models)
        results["_protocol"] = {
            "selection_split": args.selection_split,
            "selection_eval_split": "val",
            "final_eval_split": "test",
        }
        all_results[dataset] = results

        # Save per-dataset result
        out_path = os.path.join(args.output_dir, f"{dataset}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Selected model orderings")
    print(f"{'='*60}")
    for dataset in datasets:
        if dataset not in all_results:
            continue
        print(f"\n{dataset}:")
        for method, res in all_results[dataset].items():
            if "selected" in res:
                sel_str = " -> ".join(res["selected"])
                print(f"  {method:20s}: {sel_str}")


if __name__ == "__main__":
    main()

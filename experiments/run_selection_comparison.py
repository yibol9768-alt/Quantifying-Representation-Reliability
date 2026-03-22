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


BASE_SELECTION_CONFIGS = [
    {
        "name": "Ours_LogME_CKA",
        "relevance": "logme",
        "redundancy": "cka",
        "selection": "relevance_redundancy",
        "lambda": 1.0,
    },
    {
        "name": "GBC_CKA",
        "relevance": "gbc",
        "redundancy": "cka",
        "selection": "relevance_redundancy",
        "lambda": 1.0,
    },
    {
        "name": "HScore_CKA",
        "relevance": "hscore",
        "redundancy": "cka",
        "selection": "relevance_redundancy",
        "lambda": 1.0,
    },
    {
        "name": "LogME_SVCCA",
        "relevance": "logme",
        "redundancy": "svcca",
        "selection": "relevance_redundancy",
        "lambda": 1.0,
    },
    {
        "name": "mRMR",
        "relevance": "logme",
        "redundancy": "cka",
        "selection": "mrmr",
        "lambda": 1.0,
    },
    {
        "name": "JMI",
        "relevance": "logme",
        "redundancy": "cka",
        "selection": "jmi",
        "lambda": 1.0,
    },
    {
        "name": "Relevance_Only",
        "relevance": "logme",
        "redundancy": "cka",
        "selection": "relevance_only",
        "lambda": 1.0,
    },
    {
        "name": "Random",
        "relevance": "logme",
        "redundancy": "cka",
        "selection": "random",
        "lambda": 1.0,
    },
]


def build_selection_configs(args):
    configs = list(BASE_SELECTION_CONFIGS)
    if args.include_third_term:
        configs.append(
            {
                "name": "Ours_LogME_CKA_3Term",
                "relevance": "logme",
                "redundancy": "cka",
                "selection": "three_term_low_order",
                "lambda": 1.0,
                "eta_cond": args.eta_cond,
                "conditional_pca_dim": args.conditional_pca_dim,
                "conditional_min_class_samples": args.conditional_min_class_samples,
                "conditional_reg": args.conditional_reg,
            }
        )

    if not args.methods:
        return configs

    requested = {name.strip() for name in args.methods.split(",") if name.strip()}
    filtered = [cfg for cfg in configs if cfg["name"] in requested]
    missing = sorted(requested - {cfg["name"] for cfg in configs} - {"All_Models"})
    if missing:
        raise ValueError(f"Unknown methods requested: {missing}")
    return filtered


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


def run_comparison(features, labels, max_models, selection_configs, include_all_models=True):
    """Run all selection methods and return results."""
    results = {}
    all_models = sorted(features.keys())

    for cfg in selection_configs:
        config_name = cfg["name"]
        rel = cfg["relevance"]
        red = cfg["redundancy"]
        sel = cfg["selection"]
        lam = cfg["lambda"]
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
                eta_cond=cfg.get("eta_cond", 1.0),
                conditional_pca_dim=cfg.get("conditional_pca_dim", 32),
                conditional_min_class_samples=cfg.get("conditional_min_class_samples", 8),
                conditional_reg=cfg.get("conditional_reg", 1e-3),
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
                    "eta_cond": cfg.get("eta_cond"),
                    "conditional_pca_dim": cfg.get("conditional_pca_dim"),
                    "conditional_min_class_samples": cfg.get("conditional_min_class_samples"),
                    "conditional_reg": cfg.get("conditional_reg"),
                },
            }
            print(f"done ({elapsed:.1f}s) -> {selected}")
        except Exception as e:
            print(f"FAILED: {e}")
            results[config_name] = {"error": str(e)}

    if include_all_models:
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
    parser.add_argument("--methods", type=str, default=None,
                        help="Optional comma-separated subset of method names to run")
    parser.add_argument("--include_third_term", action="store_true",
                        help="Include the low-order three-term ablation variant")
    parser.add_argument("--eta_cond", type=float, default=1.0,
                        help="Weight on the third conditional term")
    parser.add_argument("--conditional_pca_dim", type=int, default=32,
                        help="PCA dimension for pairwise conditional MI")
    parser.add_argument("--conditional_min_class_samples", type=int, default=8,
                        help="Minimum class size for pairwise conditional MI")
    parser.add_argument("--conditional_reg", type=float, default=1e-3,
                        help="Covariance ridge for pairwise conditional MI")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",")]
    selection_configs = build_selection_configs(args)
    requested = None if not args.methods else {
        name.strip() for name in args.methods.split(",") if name.strip()
    }
    include_all_models = requested is None or "All_Models" in requested

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

        results = run_comparison(
            features,
            labels,
            args.max_models,
            selection_configs=selection_configs,
            include_all_models=include_all_models,
        )
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

"""CKA-guided model selection analysis.

Computes pairwise CKA matrices across models and datasets,
applies three selection strategies, and outputs visualizations and reports.

Usage:
    python experiments/run_cka_analysis.py \
        --datasets stl10,gtsrb,svhn,pets,eurosat,dtd,country211 \
        --models clip,dino,mae,siglip,convnext,data2vec \
        --model_dir ./models --data_dir ./data \
        --output_dir ./results/cka --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.extractor import FeatureExtractor
from src.data.dataset import get_dataloaders
from src.analysis.cka import compute_cka_matrix
from src.analysis.model_selection import (
    greedy_selection,
    max_diversity_selection,
    task_adaptive_selection,
)


def extract_features(
    model_type: str,
    dataloader,
    model_dir: str,
    device: str,
    use_patches: bool = True,
) -> torch.Tensor:
    """Extract features from a single model on the full dataset.

    Args:
        use_patches: If True, extract last-layer patch tokens and flatten
            to [N, num_patches*dim] for richer CKA computation.
            If False, use pooled [CLS] output [N, dim].
    """
    extractor = FeatureExtractor(model_type=model_type, model_dir=model_dir)
    extractor.eval().to(device)

    all_features = []
    for images, _ in dataloader:
        images = images.to(device)
        if use_patches:
            tokens = extractor.extract_last_tokens(images)
            # patches: [B, num_patches, dim] -> flatten to [B, num_patches*dim]
            patches = tokens["patches"]
            if patches.shape[1] == 0:
                # Fallback: model has no patch tokens (unlikely)
                feat = tokens["cls"]
            else:
                feat = patches.flatten(1)  # [B, num_patches * dim]
        else:
            feat = extractor(images)  # [B, dim]
        all_features.append(feat.cpu())

    extractor.to("cpu")
    del extractor
    torch.cuda.empty_cache() if "cuda" in device else None

    return torch.cat(all_features, dim=0)


def save_cka_heatmap(cka_matrix, model_names, title, save_path):
    """Save a single CKA heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cka_matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(model_names, fontsize=11)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            color = "white" if cka_matrix[i, j] > 0.7 else "black"
            ax.text(j, i, f"{cka_matrix[i, j]:.3f}",
                    ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8, label="CKA Similarity")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_combined_heatmap(all_matrices, model_names, datasets, save_path):
    """Save a combined heatmap with all datasets in subplots."""
    n = len(datasets)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        matrix = all_matrices[dataset]
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")

        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(model_names, fontsize=9)

        for i in range(len(model_names)):
            for j in range(len(model_names)):
                color = "white" if matrix[i, j] > 0.7 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", color=color, fontsize=8)

        ax.set_title(dataset.upper(), fontsize=12, fontweight="bold")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("CKA Similarity Matrices Across Datasets", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CKA-guided model selection analysis")
    parser.add_argument("--datasets", type=str, default="stl10,gtsrb,svhn,pets,eurosat,dtd,country211",
                        help="Comma-separated dataset names")
    parser.add_argument("--models", type=str, default="clip,dino,mae,siglip,convnext,data2vec",
                        help="Comma-separated model names")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results/cka")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_model", type=str, default="clip",
                        help="Starting model for greedy selection")
    parser.add_argument("--max_redundancy", type=float, default=0.25,
                        help="Max avg CKA of new model to existing set for "
                             "recommended subset cutoff. Full ordering is always "
                             "produced regardless.")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples per dataset for CKA (0=all). "
                             "Default 2000 to manage memory with patch tokens.")
    parser.add_argument("--use_patches", action="store_true", default=True,
                        help="Use last-layer patch tokens instead of pooled output (default: True)")
    parser.add_argument("--no_patches", dest="use_patches", action="store_false",
                        help="Use pooled [CLS] output instead of patch tokens")
    parser.add_argument("--pca_dim", type=int, default=256,
                        help="PCA target dim for patch tokens before CKA (0=no PCA)")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for dataset subsampling when --max_samples > 0")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CKA Analysis ===")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Device: {args.device}")
    print(f"Feature type: {'patch tokens (last layer)' if args.use_patches else 'pooled [CLS]'}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"PCA dim: {args.pca_dim if args.pca_dim > 0 else 'disabled'}")
    print(f"Output: {output_dir}")
    print()

    # ---- Step 1: Extract features and compute CKA per dataset ----
    all_cka_matrices = {}

    for dataset in datasets:
        print(f"--- Dataset: {dataset} ---")

        # Load full training data (no fewshot)
        train_loader, _ = get_dataloaders(
            dataset=dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_type="mae",  # transforms don't matter much for CKA
        )

        # Subsample at the dataset level before feature extraction to avoid
        # materializing huge patch-token tensors for the full training set.
        if args.max_samples > 0 and len(train_loader.dataset) > args.max_samples:
            generator = torch.Generator().manual_seed(args.sample_seed)
            subset_indices = torch.randperm(len(train_loader.dataset), generator=generator)[:args.max_samples]
            train_loader = DataLoader(
                Subset(train_loader.dataset, subset_indices.tolist()),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            print(f"  Using shared subset: {args.max_samples} samples")

        # Extract features per model
        features_dict = {}
        for model_name in models:
            print(f"  Extracting features: {model_name}...", end=" ", flush=True)
            feat = extract_features(
                model_name, train_loader, args.model_dir, args.device,
                use_patches=args.use_patches,
            )

            features_dict[model_name] = feat
            print(f"shape={feat.shape}")

        # Compute CKA matrix
        pca_dim = args.pca_dim if args.pca_dim > 0 else None
        print(f"  Computing CKA matrix (PCA dim={pca_dim})...")
        model_names, cka_matrix = compute_cka_matrix(features_dict, pca_dim=pca_dim)
        all_cka_matrices[dataset] = cka_matrix

        # Save CSV
        csv_path = output_dir / f"cka_matrix_{dataset}.csv"
        header = "," + ",".join(model_names)
        rows = []
        for i, name in enumerate(model_names):
            row = name + "," + ",".join(f"{cka_matrix[i, j]:.4f}" for j in range(len(model_names)))
            rows.append(row)
        csv_path.write_text(header + "\n" + "\n".join(rows) + "\n")
        print(f"  Saved: {csv_path}")

        # Save individual heatmap
        heatmap_path = output_dir / f"cka_heatmap_{dataset}.png"
        save_cka_heatmap(cka_matrix, model_names, f"CKA Similarity - {dataset.upper()}", heatmap_path)
        print(f"  Saved: {heatmap_path}")

        # Free memory
        del features_dict
        torch.cuda.empty_cache() if "cuda" in args.device else None
        print()

    # ---- Step 2: Combined heatmap ----
    if len(datasets) > 1:
        combined_path = output_dir / "cka_heatmap_all.png"
        save_combined_heatmap(all_cka_matrices, models, datasets, combined_path)
        print(f"Saved combined heatmap: {combined_path}")

    # ---- Step 3: Model selection strategies ----
    print("\n=== Model Selection ===")
    selection_results = {
        "greedy": {},
        "max_diversity": {},
        "task_adaptive": {},
    }

    # Strategy A: Greedy per dataset
    greedy_traces = {}
    greedy_full_orders = {}
    for dataset in datasets:
        recommended, full_order, trace = greedy_selection(
            all_cka_matrices[dataset], models, args.start_model, args.max_redundancy
        )
        selection_results["greedy"][dataset] = {
            "recommended": recommended,
            "full_order": full_order,
        }
        greedy_traces[dataset] = trace
        greedy_full_orders[dataset] = full_order
        print(f"  Greedy ({dataset}): recommended={recommended} | full={full_order}")
        for step in trace:
            marker = ">>" if step["step"] == len(recommended) + 1 and len(recommended) < len(full_order) else "  "
            print(f"   {marker} step {step['step']}: {step['model']:>12s}  "
                  f"cka_to_set={step['avg_cka_to_set']:.4f}  "
                  f"diversity={step['set_diversity']:.4f}")

    # Strategy B: Max diversity (k=3 and k=4) using average CKA across datasets
    avg_matrix = np.mean(list(all_cka_matrices.values()), axis=0)
    for k in [3, 4]:
        selected = max_diversity_selection(avg_matrix, models, k)
        selection_results["max_diversity"][f"k={k}"] = selected
        print(f"  Max Diversity (k={k}): {selected}")

    # Strategy C: Task-adaptive
    task_results = task_adaptive_selection(
        all_cka_matrices, models, args.start_model, args.max_redundancy
    )
    selection_results["task_adaptive"] = {}
    print(f"  Task Adaptive (max_redundancy={args.max_redundancy}):")
    for dataset, (recommended, full_order, trace) in task_results.items():
        selection_results["task_adaptive"][dataset] = {
            "recommended": recommended,
            "full_order": full_order,
        }
        print(f"    {dataset}: {recommended} ({len(recommended)} models)")

    # ---- Step 4: Save results ----
    # Selection results JSON
    json_path = output_dir / "selection_results.json"
    with open(json_path, "w") as f:
        json.dump(selection_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Summary text
    summary_path = output_dir / "summary.txt"
    lines = []
    lines.append("=" * 60)
    lines.append("CKA-Guided Model Selection Summary")
    lines.append("=" * 60)
    lines.append(f"\nModels analyzed: {models}")
    lines.append(f"Datasets: {datasets}")
    lines.append(f"Start model: {args.start_model}")
    lines.append(f"Max redundancy: {args.max_redundancy}")

    lines.append(f"\n{'=' * 60}")
    lines.append("Average CKA Matrix (across all datasets)")
    lines.append("=" * 60)
    header = f"{'':>12s}" + "".join(f"{m:>12s}" for m in models)
    lines.append(header)
    for i, name in enumerate(models):
        row = f"{name:>12s}" + "".join(f"{avg_matrix[i, j]:>12.4f}" for j in range(len(models)))
        lines.append(row)

    lines.append(f"\n{'=' * 60}")
    lines.append("Selection Results")
    lines.append("=" * 60)

    lines.append(f"\n--- Strategy A: Greedy Selection (max_redundancy={args.max_redundancy}) ---")
    for dataset, info in selection_results["greedy"].items():
        rec = info["recommended"]
        full = info["full_order"]
        lines.append(f"  {dataset}:")
        lines.append(f"    Recommended: {' -> '.join(rec)} ({len(rec)} models)")
        lines.append(f"    Full order:  {' -> '.join(full)}")
        if dataset in greedy_traces:
            for step in greedy_traces[dataset]:
                cutoff = " <<< cutoff" if step["step"] == len(rec) + 1 and len(rec) < len(full) else ""
                lines.append(f"      step {step['step']}: {step['model']:>12s}  "
                             f"cka_to_set={step['avg_cka_to_set']:.4f}  "
                             f"diversity={step['set_diversity']:.4f}{cutoff}")

    lines.append("\n--- Strategy B: Max Diversity ---")
    for key, selected in selection_results["max_diversity"].items():
        lines.append(f"  {key}: {', '.join(selected)}")

    lines.append(f"\n--- Strategy C: Task-Adaptive (max_redundancy={args.max_redundancy}) ---")
    for dataset, info in selection_results["task_adaptive"].items():
        rec = info["recommended"]
        full = info["full_order"]
        lines.append(f"  {dataset}: {' -> '.join(rec)} ({len(rec)} models)"
                     f"  [full: {' -> '.join(full)}]")

    # Cross-dataset analysis
    lines.append(f"\n{'=' * 60}")
    lines.append("Cross-Dataset Analysis")
    lines.append("=" * 60)

    # Find most/least similar pairs
    n_models = len(models)
    pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairs.append((models[i], models[j], avg_matrix[i, j]))
    pairs.sort(key=lambda x: x[2])

    lines.append("\nMost diverse pairs (lowest CKA):")
    for m1, m2, cka in pairs[:3]:
        lines.append(f"  {m1} - {m2}: {cka:.4f}")

    lines.append("\nMost similar pairs (highest CKA):")
    for m1, m2, cka in pairs[-3:]:
        lines.append(f"  {m1} - {m2}: {cka:.4f}")

    # Per-dataset variance
    lines.append("\nCKA variance across datasets (high = task-dependent similarity):")
    for i in range(n_models):
        for j in range(i + 1, n_models):
            vals = [all_cka_matrices[d][i, j] for d in datasets]
            var = np.var(vals)
            if var > 0.001:  # report notable variance
                lines.append(f"  {models[i]} - {models[j]}: var={var:.4f} "
                             f"(range: {min(vals):.3f}-{max(vals):.3f})")

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    print(f"Saved: {summary_path}")
    print(f"\n{'=' * 60}")
    print("CKA analysis complete!")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()

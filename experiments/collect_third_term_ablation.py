#!/usr/bin/env python3
"""Collect a two-row ablation table for the third-term selector.

Compares the existing two-term baseline against the new low-order three-term
variant by merging selection/fusion artifacts from two directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect_fusion_results import (
    build_summary,
    load_fusion_results,
    load_selection_orderings,
    load_single_model_results,
    write_best_csv,
    write_latex_table,
    write_markdown,
)


def _filter_methods(orderings: dict, allowed: set[str]) -> dict:
    return {
        (dataset, method): ordering
        for (dataset, method), ordering in orderings.items()
        if method in allowed
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect third-term ablation results")
    parser.add_argument("--base_selection_dir", required=True)
    parser.add_argument("--base_fusion_dir", required=True)
    parser.add_argument("--ablation_selection_dir", required=True)
    parser.add_argument("--ablation_fusion_dir", required=True)
    parser.add_argument("--single_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fusion_method", default="gated")
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--base_method", default="Ours_LogME_CKA")
    parser.add_argument("--ablation_method", default="Ours_LogME_CKA_3Term")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = {args.base_method, args.ablation_method}
    orderings = {}
    orderings.update(
        _filter_methods(load_selection_orderings(Path(args.base_selection_dir)), allowed)
    )
    orderings.update(
        _filter_methods(load_selection_orderings(Path(args.ablation_selection_dir)), allowed)
    )

    single = load_single_model_results(Path(args.single_model_dir))
    fusion = {}
    fusion.update(load_fusion_results(Path(args.base_fusion_dir), fusion_method=args.fusion_method))
    fusion.update(load_fusion_results(Path(args.ablation_fusion_dir), fusion_method=args.fusion_method))

    summary = build_summary(orderings, single, fusion, max_k=args.max_k)
    summary["methods"] = [
        method for method in [args.base_method, args.ablation_method]
        if method in summary["methods"]
    ]
    summary["meta"] = {
        "base_selection_dir": args.base_selection_dir,
        "base_fusion_dir": args.base_fusion_dir,
        "ablation_selection_dir": args.ablation_selection_dir,
        "ablation_fusion_dir": args.ablation_fusion_dir,
        "single_model_dir": args.single_model_dir,
        "fusion_method": args.fusion_method,
        "max_k": args.max_k,
    }

    (output_dir / "third_term_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    write_best_csv(summary, output_dir / "third_term_ablation_best.csv")
    write_markdown(summary, output_dir / "third_term_ablation_best.md")
    write_latex_table(
        summary,
        output_dir / "third_term_ablation_best.tex",
        caption=(
            "Ablation on the third term. The baseline uses the two-term "
            "LogME+CKA selector, while the ablation variant adds the low-order "
            "pairwise conditional term."
        ),
        label="tab:third_term_ablation",
    )

    print(f"[saved] {output_dir / 'third_term_ablation_summary.json'}")
    print(f"[saved] {output_dir / 'third_term_ablation_best.csv'}")
    print(f"[saved] {output_dir / 'third_term_ablation_best.md'}")
    print(f"[saved] {output_dir / 'third_term_ablation_best.tex'}")


if __name__ == "__main__":
    main()

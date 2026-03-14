"""Main entry point for feature classification experiments."""

import argparse
import random
import torch

from configs.config import Config, DATASET_CONFIGS
from src.training.trainer import train

DEFAULT_MODEL_DIR = "./models"
DEFAULT_DATA_DIR = "./data"
DEFAULT_CACHE_DIR = "./cache/offline"
DEFAULT_RESULTS_DIR = "./results"

VALID_MODELS = [
    "mae", "clip", "dino", "fusion",
    "vit", "swin", "beit", "data2vec",
    "openclip", "siglip", "convnext",
]

FUSION_METHODS = [
    "concat", "proj_concat", "weighted_sum", "gated",
    "difference_concat", "hadamard_concat", "bilinear_concat",
    "film", "context_gating", "lmf", "se_fusion", "late_fusion",
    "comm", "mmvit",
    "topk_router", "moe_router", "attention_router",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Classification")

    # Core
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", type=str, default="mae", choices=VALID_MODELS)
    parser.add_argument("--fusion_method", type=str, default="concat",
                        choices=FUSION_METHODS,
                        help="Fusion method when --model fusion")
    parser.add_argument("--fusion_models", type=str, default="mae,clip,dino",
                        help="Comma-separated model list for fusion")

    # Fusion-specific
    parser.add_argument("--fusion_output_dim", type=int, default=1024)
    parser.add_argument("--disable_fusion_harmonization", action="store_true")
    parser.add_argument("--comm_dino_mlp_blocks", type=int, default=2)
    parser.add_argument("--comm_dino_mlp_ratio", type=float, default=8.0)
    parser.add_argument("--mmvit_base_dim", type=int, default=96)
    parser.add_argument("--mmvit_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--mmvit_num_heads", type=int, default=8)
    parser.add_argument("--mmvit_max_position_tokens", type=int, default=256)
    parser.add_argument("--router_k", type=int, default=2)
    parser.add_argument("--router_aux_weight", type=float, default=0.01)
    parser.add_argument("--attention_router_heads", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    # Few-shot
    parser.add_argument("--fewshot_min", type=int, default=10)
    parser.add_argument("--fewshot_max", type=int, default=10)
    parser.add_argument("--disable_fewshot", action="store_true")

    # Paths
    parser.add_argument("--storage_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    # Cache
    parser.add_argument("--no_precompute", action="store_true")
    parser.add_argument("--cache_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--cleanup_cache", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_fusion_models(fusion_models_str: str):
    """Parse and validate fusion model list from CLI."""
    valid = set(m for m in VALID_MODELS if m != "fusion")
    models = list(dict.fromkeys(
        name.strip().lower() for name in fusion_models_str.split(",") if name.strip()
    ))
    if not models:
        raise ValueError("fusion_models is empty.")
    if bad := [m for m in models if m not in valid]:
        raise ValueError(f"Invalid model(s): {bad}. Valid: {sorted(valid)}")
    if len(models) < 2:
        raise ValueError("Fusion requires at least 2 models.")
    return models


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_storage_paths(args):
    """Resolve model/data/cache paths from a shared storage root."""
    if args.storage_dir is None:
        args.model_dir = DEFAULT_MODEL_DIR
        return
    from pathlib import Path
    root = Path(args.storage_dir)
    args.model_dir = str(root / "models")
    if args.data_dir == DEFAULT_DATA_DIR:
        args.data_dir = str(root / "data")
    if args.cache_dir == DEFAULT_CACHE_DIR:
        args.cache_dir = str(root / "cache" / "offline")
    if args.results_dir == DEFAULT_RESULTS_DIR:
        args.results_dir = str(root / "results")


def _sanitize(value: str) -> str:
    return str(value).strip().replace("/", "-").replace(" ", "_").replace(".", "p")


def _data_tag(args) -> str:
    if not args.disable_fewshot:
        return f"fs{args.fewshot_min}to{args.fewshot_max}"
    return "fulltrain"


def _use_harmonization(args) -> bool:
    return args.model == "fusion" and not args.disable_fusion_harmonization


def build_derived_names(args):
    """Pre-compute checkpoint_name, cache_name, run_basename, fusion_kwargs."""
    dt = _data_tag(args)
    harmonized = _use_harmonization(args)

    if args.model == "fusion":
        mtag = "-".join(args.fusion_model_list)
        base_parts = [args.dataset, dt, "fusion", args.fusion_method, mtag]
        if harmonized:
            base_parts.append(f"dim{args.fusion_output_dim}")
        args.checkpoint_name = "_".join(base_parts) + "_best.pth"
    else:
        base_parts = [args.dataset, dt, args.model]
        args.checkpoint_name = f"{args.dataset}_{dt}_{args.model}_best.pth"

    cache_parts = list(base_parts) + [f"seed{args.seed}", f"cache{args.cache_dtype}"]
    args.cache_name = "_".join(cache_parts).replace(".", "p")

    run_parts = list(base_parts) + [
        f"seed{args.seed}",
        "offline-cache" if not args.no_precompute else "online",
    ]
    args.run_basename = "_".join(_sanitize(p) for p in run_parts)

    # Build fusion kwargs dict for the extractor factory
    args.fusion_kwargs = {
        "comm_dino_mlp_blocks": args.comm_dino_mlp_blocks,
        "comm_dino_mlp_ratio": args.comm_dino_mlp_ratio,
        "mmvit_base_dim": args.mmvit_base_dim,
        "mmvit_mlp_ratio": args.mmvit_mlp_ratio,
        "mmvit_num_heads": args.mmvit_num_heads,
        "mmvit_max_position_tokens": args.mmvit_max_position_tokens,
        "fusion_output_dim": args.fusion_output_dim if harmonized else None,
        "num_classes": Config(dataset=args.dataset).num_classes,
        "router_k": args.router_k,
        "attention_router_heads": args.attention_router_heads,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.fewshot_min <= 0 or args.fewshot_max <= 0:
        raise ValueError("fewshot_min and fewshot_max must be positive.")
    if args.fewshot_min > args.fewshot_max:
        raise ValueError("fewshot_min must be <= fewshot_max.")

    set_random_seed(args.seed)
    resolve_storage_paths(args)

    if args.model == "fusion":
        args.fusion_model_list = parse_fusion_models(args.fusion_models)
        args.loader_model_type = "fusion"
    else:
        args.fusion_model_list = None
        args.loader_model_type = args.model

    build_derived_names(args)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = Config(
        dataset=args.dataset,
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    # Print experiment info
    print("=" * 60)
    print("Feature Classification")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({config.num_classes} classes)")
    print(f"Model: {args.model}")
    if args.model == "fusion":
        print(f"  Fusion method: {args.fusion_method}")
        print(f"  Fusion models: {args.fusion_model_list}")
        print(f"  Harmonization: {_use_harmonization(args)}")
    print(f"Seed: {args.seed}")
    fewshot = not args.disable_fewshot
    print(f"Few-shot: {fewshot}" + (f" ({args.fewshot_min}-{args.fewshot_max}/class)" if fewshot else ""))
    print(f"Cache mode: {not args.no_precompute}")
    print(f"Storage: {args.storage_dir or '(local)'}")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    best_acc = train(args, config)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Accuracy: {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

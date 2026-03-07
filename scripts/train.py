"""
Training script.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.config import MODEL_CONFIGS
from src.data import DATASET_INFO
from src.features import FeatureExtractor
from src.training import SingleViewClassifier, Trainer, create_fusion_model
from src.utils import get_device, print_model_info, set_seed


TOKEN_METHODS = {"comm", "comm3", "mmvit", "mmvit3"}
METHOD_MODEL_REQUIREMENTS = {
    "comm": ["clip", "dino"],
    "mmvit": ["clip", "dino"],
    "comm3": ["clip", "dino", "mae"],
    "mmvit3": ["clip", "dino", "mae"],
}


def _ensure_split_features(features_dict: Dict, models: List[str]) -> Dict:
    split_keys = [f"{m}_features" for m in models]
    if all(k in features_dict for k in split_keys):
        result = dict(features_dict)
        result["feature_order"] = split_keys
        return result

    if "features" not in features_dict:
        raise ValueError("Feature file missing both split keys and 'features' key")

    x = features_dict["features"]
    dims = [MODEL_CONFIGS[m]["feature_dim"] for m in models]
    if x.shape[1] != sum(dims):
        raise ValueError(
            f"Feature dim mismatch: got {x.shape[1]}, expected {sum(dims)} for models {models}"
        )

    chunks = torch.split(x, dims, dim=1)
    result = {"labels": features_dict["labels"], "feature_order": split_keys}
    for key, chunk in zip(split_keys, chunks):
        result[key] = chunk
    return result


def _infer_view_layout(feature_order: List[str]) -> List[List[int]]:
    groups = {}
    ordered_names = []
    for idx, key in enumerate(feature_order):
        if "_layer_" in key:
            name = key.split("_layer_")[0]
        elif key.endswith("_tokens_features"):
            name = key[: -len("_tokens_features")]
        elif key.endswith("_features"):
            name = key[: -len("_features")]
        else:
            name = key
        if name not in groups:
            groups[name] = []
            ordered_names.append(name)
        groups[name].append(idx)
    return [groups[name] for name in ordered_names]


def _load_feature_pair(args, models: List[str]) -> Tuple[Dict, Dict, str, str]:
    if args.method in TOKEN_METHODS and len(models) > 1:
        train_path = os.path.join(args.feature_dir, f"{args.dataset}_{args.method}_train.pt")
        test_path = os.path.join(args.feature_dir, f"{args.dataset}_{args.method}_test.pt")
    else:
        model_str = "_".join(models)
        train_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_train.pt")
        test_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_test.pt")

    for path in [train_path, test_path]:
        if not os.path.exists(path):
            if args.method in TOKEN_METHODS and len(models) > 1:
                cmd = f"python scripts/extract.py --method {args.method} --dataset {args.dataset} --split {{}}"
            elif len(models) == 1:
                cmd = f"python scripts/extract.py --model {models[0]} --dataset {args.dataset} --split {{}}"
            else:
                cmd = f"python scripts/extract.py --models {' '.join(models)} --dataset {args.dataset} --split {{}}"
            raise FileNotFoundError(
                f"{path} not found.\nPlease run:\n  {cmd.format('train')}\n  {cmd.format('test')}"
            )

    train_features = FeatureExtractor.load_features(train_path)
    test_features = FeatureExtractor.load_features(test_path)

    if args.method in TOKEN_METHODS and len(models) > 1:
        for data in [train_features, test_features]:
            if "feature_order" not in data:
                data["feature_order"] = sorted([k for k in data.keys() if k.endswith("_features")])
            if "view_layout" not in data:
                data["view_layout"] = _infer_view_layout(data["feature_order"])
    else:
        train_features = _ensure_split_features(train_features, models)
        test_features = _ensure_split_features(test_features, models)

    return train_features, test_features, train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Train classifier")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    model_group.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])

    parser.add_argument(
        "--method",
        type=str,
        default="concat",
        choices=["concat", "mmvit", "mmvit3", "comm", "comm3"],
        help="Fusion method. Token methods need token feature files from scripts/extract.py --method ...",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_INFO.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-dir", type=str, default="features")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    device = get_device()
    set_seed(42)
    num_classes = DATASET_INFO[args.dataset]["num_classes"]

    if args.model:
        models = [args.model]
    else:
        models = args.models

    is_single = len(models) == 1
    if not is_single and args.method in METHOD_MODEL_REQUIREMENTS:
        required = METHOD_MODEL_REQUIREMENTS[args.method]
        if models != required:
            print(f"\nError: --method {args.method} requires --models {' '.join(required)}")
            sys.exit(1)
    model_str = "_".join(models)

    print(f"Training {model_str.upper()} on {args.dataset}")
    print(f"Mode: {'Single' if is_single else f'Fusion ({args.method})'}")
    print(f"Device: {device}, Classes: {num_classes}")

    try:
        train_features, test_features, train_path, test_path = _load_feature_pair(args, models)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print("\nLoading features:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    if is_single:
        feature_dim = MODEL_CONFIGS[models[0]]["feature_dim"]
        model = SingleViewClassifier(feature_dim=feature_dim, num_classes=num_classes)
    else:
        if args.method == "concat":
            feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in models]
            model = create_fusion_model(
                method="concat",
                num_classes=num_classes,
                feature_dims=feature_dims,
            )
            print(f"Input feature dims: {feature_dims} (total: {sum(feature_dims)})")
        else:
            feature_order = train_features["feature_order"]
            feature_shapes = [tuple(train_features[k].shape[1:]) for k in feature_order]
            model = create_fusion_model(
                method=args.method,
                num_classes=num_classes,
                feature_shapes=feature_shapes,
                view_layout=train_features.get("view_layout"),
                feature_order=feature_order,
            )
            print(f"Token feature keys: {len(feature_order)}")
            print(f"View layout: {train_features.get('view_layout')}")

    print_model_info(model)

    trainer = Trainer(model, device=device, lr=args.lr, weight_decay=1e-4)

    output_path = args.output
    if output_path is None:
        os.makedirs("outputs/checkpoints", exist_ok=True)
        suffix = "single" if is_single else args.method
        output_path = f"outputs/checkpoints/{args.dataset}_{model_str}_{suffix}.pth"

    history = trainer.fit(
        train_features=train_features,
        test_features=test_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    trainer.save(output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"Best accuracy: {max(history['test_acc']):.2f}%")


if __name__ == "__main__":
    main()

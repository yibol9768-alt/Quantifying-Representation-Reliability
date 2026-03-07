"""
Evaluation script.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from configs.config import MODEL_CONFIGS
from src.data import DATASET_INFO
from src.features import FeatureExtractor
from src.training import FeatureDataset, SingleViewClassifier, create_fusion_model
from src.utils import get_device


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


def _load_test_features(args, models: List[str]) -> Tuple[Dict, str]:
    if args.method in TOKEN_METHODS and len(models) > 1:
        test_path = os.path.join(args.feature_dir, f"{args.dataset}_{args.method}_test.pt")
    else:
        model_str = "_".join(models)
        test_path = os.path.join(args.feature_dir, f"{args.dataset}_{model_str}_test.pt")

    if not os.path.exists(test_path):
        if args.method in TOKEN_METHODS and len(models) > 1:
            cmd = f"python scripts/extract.py --method {args.method} --dataset {args.dataset} --split test"
        elif len(models) == 1:
            cmd = f"python scripts/extract.py --model {models[0]} --dataset {args.dataset} --split test"
        else:
            cmd = f"python scripts/extract.py --models {' '.join(models)} --dataset {args.dataset} --split test"
        raise FileNotFoundError(f"{test_path} not found.\nPlease run:\n  {cmd}")

    test_features = FeatureExtractor.load_features(test_path)
    if args.method in TOKEN_METHODS and len(models) > 1:
        if "feature_order" not in test_features:
            test_features["feature_order"] = sorted([k for k in test_features.keys() if k.endswith("_features")])
        if "view_layout" not in test_features:
            test_features["view_layout"] = _infer_view_layout(test_features["feature_order"])
    else:
        test_features = _ensure_split_features(test_features, models)

    return test_features, test_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    model_group.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])

    parser.add_argument(
        "--method",
        type=str,
        default="concat",
        choices=["concat", "mmvit", "mmvit3", "comm", "comm3"],
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_INFO.keys()))
    parser.add_argument("--feature-dir", type=str, default="features")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = get_device()
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    models = [args.model] if args.model else args.models
    is_single = len(models) == 1
    if not is_single and args.method in METHOD_MODEL_REQUIREMENTS:
        required = METHOD_MODEL_REQUIREMENTS[args.method]
        if models != required:
            print(f"\nError: --method {args.method} requires --models {' '.join(required)}")
            sys.exit(1)
    model_str = "_".join(models)

    print(f"Evaluating {model_str.upper()} on {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    try:
        test_features, test_path = _load_test_features(args, models)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print(f"Features: {test_path}")

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
        else:
            feature_order = test_features["feature_order"]
            feature_shapes = [tuple(test_features[k].shape[1:]) for k in feature_order]
            model = create_fusion_model(
                method=args.method,
                num_classes=num_classes,
                feature_shapes=feature_shapes,
                view_layout=test_features.get("view_layout"),
                feature_order=feature_order,
            )

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    dataset = FeatureDataset(test_features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                features, labels = batch
                features = features.to(device).float()
                outputs = model(features)
            else:
                *features, labels = batch
                features = [f.to(device).float() for f in features]
                outputs = model(*features)

            labels = labels.to(device).long()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

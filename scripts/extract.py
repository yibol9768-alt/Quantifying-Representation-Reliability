"""
Feature extraction script.

Modes:
1) Global feature extraction (concat baseline):
   python scripts/extract.py --model clip --dataset cifar10 --split train
   python scripts/extract.py --models clip dino --dataset cifar10 --split train
   python scripts/extract.py --models clip dino --dataset flowers102 --split train --backend dali

2) Token-level paper methods:
   python scripts/extract.py --method comm --dataset cifar10 --split train
   python scripts/extract.py --method comm3 --dataset cifar10 --split train
   python scripts/extract.py --method mmvit --dataset cifar10 --split train
   python scripts/extract.py --method mmvit3 --dataset cifar10 --split train
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import tempfile
from typing import Dict, Iterator, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm


def _parse_cuda_device_id(device: str) -> int:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":")[1])
        except (ValueError, IndexError):
            return 0
    return 0


def _import_dali():
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    return fn, pipeline_def, types, LastBatchPolicy, DALIGenericIterator


def _can_use_dali(args, image_paths) -> bool:
    if args.backend == "cpu":
        return False
    if not (args.device.startswith("cuda") and torch.cuda.is_available()):
        if args.backend == "dali":
            raise RuntimeError("DALI backend requires CUDA device")
        return False
    if not image_paths or not isinstance(image_paths[0], str):
        if args.backend == "dali":
            raise RuntimeError("DALI backend currently requires file-path based datasets")
        return False
    try:
        _import_dali()
    except Exception as exc:
        if args.backend == "dali":
            raise RuntimeError(f"DALI import failed: {exc}") from exc
        return False
    return True


def _size_to_int(size_obj, default: int) -> int:
    if isinstance(size_obj, int):
        return int(size_obj)
    if isinstance(size_obj, (tuple, list)) and len(size_obj) > 0:
        return int(size_obj[0])
    if isinstance(size_obj, dict):
        for key in ("shortest_edge", "height", "width", "size"):
            if key in size_obj:
                return _size_to_int(size_obj[key], default)
    return int(default)


def _get_model_preprocess_config(model) -> Dict:
    # Reasonable defaults (ImageNet style)
    cfg = {
        "resize_shorter": 256,
        "crop_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    # MAE processor path
    if hasattr(model, "processor"):
        processor = model.processor
        cfg["mean"] = list(getattr(processor, "image_mean", cfg["mean"]))
        cfg["std"] = list(getattr(processor, "image_std", cfg["std"]))
        cfg["resize_shorter"] = _size_to_int(getattr(processor, "size", 224), 224)
        cfg["crop_size"] = _size_to_int(
            getattr(processor, "crop_size", getattr(processor, "size", 224)),
            224,
        )
        return cfg

    transform = model.get_transform()
    if not hasattr(transform, "transforms"):
        return cfg

    for t in transform.transforms:
        name = t.__class__.__name__.lower()
        if "resize" in name and hasattr(t, "size"):
            cfg["resize_shorter"] = _size_to_int(t.size, cfg["resize_shorter"])
        elif "centercrop" in name and hasattr(t, "size"):
            cfg["crop_size"] = _size_to_int(t.size, cfg["crop_size"])
        elif "normalize" in name and hasattr(t, "mean") and hasattr(t, "std"):
            cfg["mean"] = [float(x) for x in t.mean]
            cfg["std"] = [float(x) for x in t.std]

    return cfg


def _iter_dali_batches(args, image_paths, labels, preprocess_cfg):
    fn, pipeline_def, types, LastBatchPolicy, DALIGenericIterator = _import_dali()

    file_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    file_list_path = file_list.name
    try:
        for path, label in zip(image_paths, labels):
            file_list.write(f"{path} {int(label)}\n")
        file_list.close()

        @pipeline_def
        def _dali_pipe(file_list_path: str, resize_shorter: int, crop_size: int):
            jpegs, dali_labels = fn.readers.file(
                file_list=file_list_path,
                random_shuffle=False,
                name="Reader",
            )
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(images, resize_shorter=resize_shorter, interp_type=types.INTERP_CUBIC)
            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                output_layout="CHW",
                crop=(crop_size, crop_size),
                crop_pos_x=0.5,
                crop_pos_y=0.5,
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
            )
            return images, dali_labels

        device_id = args.dali_device_id
        if device_id is None:
            device_id = _parse_cuda_device_id(args.device)

        pipe = _dali_pipe(
            batch_size=args.batch_size,
            num_threads=max(1, args.dali_num_threads),
            device_id=device_id,
            file_list_path=file_list_path,
            resize_shorter=int(preprocess_cfg["resize_shorter"]),
            crop_size=int(preprocess_cfg["crop_size"]),
        )
        pipe.build()

        dali_iter = DALIGenericIterator(
            [pipe],
            output_map=["images", "labels"],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        mean = torch.tensor(preprocess_cfg["mean"], device=args.device).view(1, 3, 1, 1)
        std = torch.tensor(preprocess_cfg["std"], device=args.device).view(1, 3, 1, 1)

        for batch in dali_iter:
            images = batch[0]["images"]
            images = (images - mean) / std
            yield images
    finally:
        if os.path.exists(file_list_path):
            os.remove(file_list_path)


def _load_images_parallel(dataset, batch_paths, num_workers):
    if len(batch_paths) == 0:
        return []

    workers = max(1, min(num_workers, len(batch_paths)))
    if workers == 1:
        return [dataset.get_image(p) for p in batch_paths]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(dataset.get_image, batch_paths))


def _sanitize_labels(labels, dataset_name):
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if dataset_name == "flowers102":
        labels_tensor = torch.clamp(labels_tensor, 0, 101)
    return labels_tensor


def _trim_to_min_samples(data):
    tensor_keys = [k for k, v in data.items() if isinstance(v, torch.Tensor)]
    if not tensor_keys:
        return data
    min_size = min(data[k].shape[0] for k in tensor_keys)
    for key in tensor_keys:
        if data[key].shape[0] != min_size:
            data[key] = data[key][:min_size]
    return data


def _stack_images(pil_images, transform, device):
    batch_images = torch.stack([transform(img) for img in pil_images], dim=0)
    return batch_images.to(device)


def _iter_pil_batches(
    dataset,
    image_paths,
    batch_size: int,
    num_workers: int,
    prefetch: bool = True,
) -> Iterator[List]:
    total = len(image_paths)
    if total == 0:
        return

    if not prefetch:
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            yield _load_images_parallel(dataset, batch_paths, num_workers)
        return

    starts = list(range(0, total, batch_size))
    with ThreadPoolExecutor(max_workers=1) as pool:
        first_paths = image_paths[starts[0]:starts[0] + batch_size]
        future = pool.submit(_load_images_parallel, dataset, first_paths, num_workers)
        for i, start in enumerate(starts):
            pil_images = future.result()
            if i + 1 < len(starts):
                next_start = starts[i + 1]
                next_paths = image_paths[next_start:next_start + batch_size]
                future = pool.submit(_load_images_parallel, dataset, next_paths, num_workers)
            yield pil_images


def extract_single(args):
    from src.models import get_model
    from src.data import get_dataset

    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()

    print(f"Extracting {args.model.upper()} global features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")

    model = get_model(args.model, device=args.device)
    model.eval()
    use_dali = _can_use_dali(args, image_paths)
    if use_dali:
        print("Backend: DALI (GPU decode/resize)")
    else:
        print("Backend: CPU/PIL")

    features_list = []
    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    if use_dali:
        preprocess_cfg = _get_model_preprocess_config(model)
        for batch_images in tqdm(
            _iter_dali_batches(args, image_paths, labels, preprocess_cfg),
            total=total_batches,
            desc="Extracting",
        ):
            with torch.no_grad():
                features = model.extract_batch_features(batch_images)
            features_list.append(features.cpu())
    else:
        for pil_images in tqdm(
            _iter_pil_batches(
                dataset,
                image_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch=not args.disable_prefetch,
            ),
            total=total_batches,
            desc="Extracting",
        ):
            batch_images = _stack_images(pil_images, model.get_transform(), args.device)
            with torch.no_grad():
                features = model.extract_batch_features(batch_images)
            features_list.append(features.cpu())

    features = torch.cat(features_list, dim=0)
    labels_tensor = _sanitize_labels(labels, args.dataset)
    return {
        "features": features,
        "labels": labels_tensor,
        "model": args.model,
    }


def extract_multi(args):
    from src.models import get_model
    from src.data import get_dataset

    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()

    print(f"Extracting {'+'.join(args.models).upper()} global features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")

    models = {m: get_model(m, device=args.device) for m in args.models}
    for model in models.values():
        model.eval()
    use_dali = _can_use_dali(args, image_paths)
    if use_dali:
        print("Backend: DALI (per-model pass, GPU decode/resize)")
    else:
        print("Backend: CPU/PIL")

    features_list = {m: [] for m in args.models}
    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    if use_dali:
        for m in args.models:
            preprocess_cfg = _get_model_preprocess_config(models[m])
            for batch_images in tqdm(
                _iter_dali_batches(args, image_paths, labels, preprocess_cfg),
                total=total_batches,
                desc=f"Extracting-{m}",
            ):
                with torch.no_grad():
                    features = models[m].extract_batch_features(batch_images)
                features_list[m].append(features.cpu())
    else:
        for pil_images in tqdm(
            _iter_pil_batches(
                dataset,
                image_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch=not args.disable_prefetch,
            ),
            total=total_batches,
            desc="Extracting",
        ):
            for m in args.models:
                batch_images = _stack_images(pil_images, models[m].get_transform(), args.device)
                with torch.no_grad():
                    features = models[m].extract_batch_features(batch_images)
                features_list[m].append(features.cpu())

    all_features = torch.cat([torch.cat(features_list[m], dim=0) for m in args.models], dim=1)
    labels_tensor = _sanitize_labels(labels, args.dataset)
    return {
        "features": all_features,
        "labels": labels_tensor,
        "models": args.models,
    }


def _extract_multilayer_tokens(
    args,
    model_specs: List[Dict],
    method_name: str,
):
    """
    model_specs:
      [
        {
          "name": "clip",
          "class": CLIPMultiLayerModel,
          "layers": [0..11],
          "key_fmt": "clip_layer_{layer}_features",
        },
        ...
      ]
    """
    from src.data import get_dataset

    dataset = get_dataset(args.dataset, "data")
    image_paths, labels = dataset.load_train_data() if args.split == "train" else dataset.load_test_data()
    labels_tensor = _sanitize_labels(labels, args.dataset)

    print(f"Extracting {method_name.upper()} token features for {args.dataset} ({args.split})")
    print(f"Samples: {len(image_paths)}, Batch size: {args.batch_size}")

    runtime_models = []
    for spec in model_specs:
        model = spec["class"](device=args.device)
        model.eval()
        runtime_models.append((spec, model))
    use_dali = _can_use_dali(args, image_paths)
    if use_dali:
        print("Backend: DALI (per-model token extraction)")
    else:
        print("Backend: CPU/PIL")

    result: Dict[str, torch.Tensor] = {"labels": labels_tensor, "method": method_name}
    feature_order: List[str] = []
    view_layout: List[List[int]] = []

    for spec in model_specs:
        group = []
        for layer in spec["layers"]:
            key = spec["key_fmt"].format(layer=layer)
            result[key] = []
            feature_order.append(key)
            group.append(len(feature_order) - 1)
        view_layout.append(group)

    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    if use_dali:
        for spec, model in runtime_models:
            preprocess_cfg = _get_model_preprocess_config(model)
            for batch_images in tqdm(
                _iter_dali_batches(args, image_paths, labels, preprocess_cfg),
                total=total_batches,
                desc=f"Extracting-{spec['name']}",
            ):
                layer_outputs = model.extract_batch_multilayer_features(batch_images, spec["layers"])
                for layer in spec["layers"]:
                    key = spec["key_fmt"].format(layer=layer)
                    result[key].append(layer_outputs[layer].cpu())
    else:
        for pil_images in tqdm(
            _iter_pil_batches(
                dataset,
                image_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch=not args.disable_prefetch,
            ),
            total=total_batches,
            desc="Extracting",
        ):

            for spec, model in runtime_models:
                batch_images = _stack_images(pil_images, model.get_transform(), args.device)
                layer_outputs = model.extract_batch_multilayer_features(batch_images, spec["layers"])
                for layer in spec["layers"]:
                    key = spec["key_fmt"].format(layer=layer)
                    result[key].append(layer_outputs[layer].cpu())

    for key in feature_order:
        result[key] = torch.cat(result[key], dim=0)

    result["feature_order"] = feature_order
    result["view_layout"] = view_layout
    return _trim_to_min_samples(result)


def extract_comm(args):
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel

    model_specs = [
        {
            "name": "clip",
            "class": CLIPMultiLayerModel,
            "layers": list(range(0, 12)),
            "key_fmt": "clip_layer_{layer}_features",
        },
        {
            "name": "dino",
            "class": DINOMultiLayerModel,
            "layers": list(range(6, 12)),
            "key_fmt": "dino_layer_{layer}_features",
        },
    ]
    return _extract_multilayer_tokens(args, model_specs, "comm")


def extract_comm3(args):
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel, MAEMultiLayerModel

    model_specs = [
        {
            "name": "clip",
            "class": CLIPMultiLayerModel,
            "layers": list(range(0, 12)),
            "key_fmt": "clip_layer_{layer}_features",
        },
        {
            "name": "dino",
            "class": DINOMultiLayerModel,
            "layers": list(range(6, 12)),
            "key_fmt": "dino_layer_{layer}_features",
        },
        {
            "name": "mae",
            "class": MAEMultiLayerModel,
            "layers": list(range(6, 12)),
            "key_fmt": "mae_layer_{layer}_features",
        },
    ]
    return _extract_multilayer_tokens(args, model_specs, "comm3")


def extract_mmvit(args):
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel

    model_specs = [
        {
            "name": "clip",
            "class": CLIPMultiLayerModel,
            "layers": [11],
            "key_fmt": "clip_tokens_features",
        },
        {
            "name": "dino",
            "class": DINOMultiLayerModel,
            "layers": [11],
            "key_fmt": "dino_tokens_features",
        },
    ]
    return _extract_multilayer_tokens(args, model_specs, "mmvit")


def extract_mmvit3(args):
    from src.models import CLIPMultiLayerModel, DINOMultiLayerModel, MAEMultiLayerModel

    model_specs = [
        {
            "name": "clip",
            "class": CLIPMultiLayerModel,
            "layers": [11],
            "key_fmt": "clip_tokens_features",
        },
        {
            "name": "dino",
            "class": DINOMultiLayerModel,
            "layers": [11],
            "key_fmt": "dino_tokens_features",
        },
        {
            "name": "mae",
            "class": MAEMultiLayerModel,
            "layers": [11],
            "key_fmt": "mae_tokens_features",
        },
    ]
    return _extract_multilayer_tokens(args, model_specs, "mmvit3")


def main():
    parser = argparse.ArgumentParser(description="Feature extraction")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--model", type=str, choices=["clip", "dino", "mae"])
    mode_group.add_argument("--models", type=str, nargs="+", choices=["clip", "dino", "mae"])
    mode_group.add_argument("--method", type=str, choices=["comm", "comm3", "mmvit", "mmvit3"])

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "dali"],
        help="Image pipeline backend: auto(cpu fallback), cpu, or dali",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dali-num-threads", type=int, default=4, help="Threads used inside DALI pipeline")
    parser.add_argument("--dali-device-id", type=int, default=None, help="CUDA device id for DALI")
    parser.add_argument("--disable-prefetch", action="store_true", help="Disable async next-batch prefetch")
    parser.add_argument("--output-dir", type=str, default="features")
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.model:
        data = extract_single(args)
        name = f"{args.dataset}_{args.model}_{args.split}.pt"
    elif args.models:
        data = extract_multi(args)
        name = f"{args.dataset}_{'_'.join(args.models)}_{args.split}.pt"
    elif args.method == "comm":
        data = extract_comm(args)
        name = f"{args.dataset}_comm_{args.split}.pt"
    elif args.method == "comm3":
        data = extract_comm3(args)
        name = f"{args.dataset}_comm3_{args.split}.pt"
    elif args.method == "mmvit":
        data = extract_mmvit(args)
        name = f"{args.dataset}_mmvit_{args.split}.pt"
    elif args.method == "mmvit3":
        data = extract_mmvit3(args)
        name = f"{args.dataset}_mmvit3_{args.split}.pt"
    else:
        raise ValueError("Unknown extraction mode")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, name)
    torch.save(data, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

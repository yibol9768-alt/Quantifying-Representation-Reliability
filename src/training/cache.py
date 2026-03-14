"""Disk-backed offline cache utilities."""

from __future__ import annotations

import bisect
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm


def _apply_nested(obj: Any, fn) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {key: _apply_nested(value, fn) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_apply_nested(value, fn) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_apply_nested(value, fn) for value in obj)
    return obj


def move_to_device(obj: Any, device: torch.device) -> Any:
    """Move nested tensors to device."""
    return _apply_nested(
        obj,
        lambda tensor: tensor.to(device, non_blocking=True),
    )


def detach_to_storage(obj: Any, dtype: torch.dtype) -> Any:
    """Detach nested tensors and cast floating tensors for storage."""
    def _convert(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().cpu().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor

    return _apply_nested(obj, _convert)


def nested_batch_size(obj: Any) -> int:
    """Infer the leading batch dimension from nested tensors."""
    if torch.is_tensor(obj):
        return int(obj.size(0))
    if isinstance(obj, dict):
        for value in obj.values():
            return nested_batch_size(value)
    if isinstance(obj, (list, tuple)):
        for value in obj:
            return nested_batch_size(value)
    raise ValueError("Unable to infer batch size from cached object.")


def nested_index(obj: Any, index: int) -> Any:
    """Index a single sample from nested cached tensors."""
    if torch.is_tensor(obj):
        return obj[index]
    if isinstance(obj, dict):
        return {key: nested_index(value, index) for key, value in obj.items()}
    if isinstance(obj, list):
        return [nested_index(value, index) for value in obj]
    if isinstance(obj, tuple):
        return tuple(nested_index(value, index) for value in obj)
    return obj


def nested_bytes(obj: Any) -> int:
    """Estimate storage bytes for nested tensors."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nested_bytes(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nested_bytes(value) for value in obj)
    return 0


def clear_directory(path: Path):
    """Remove and recreate a cache directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def cleanup_cache_dir(cache_dir: Path):
    """Delete a cache directory after training if requested."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def build_split_cache(
    extractor,
    dataloader,
    split_dir: Path,
    split: str,
    device: torch.device,
    storage_dtype: torch.dtype = torch.float32,
    use_fp16: bool = False,
    recache: bool = False,
) -> Dict[str, Any]:
    """Build or load a sharded cache for one dataset split."""
    meta_path = split_dir / "meta.pt"
    if meta_path.exists() and not recache:
        meta = torch.load(meta_path, map_location="cpu")
        print(
            f"Loaded {split} cache: {split_dir} "
            f"(N={meta['num_samples']}, shards={meta['num_shards']}, {meta['storage_size_gb']:.2f} GiB)"
        )
        return meta

    clear_directory(split_dir)
    extractor.eval()

    shard_records: List[Dict[str, Any]] = []
    total_samples = 0
    total_bytes = 0

    pbar = tqdm(dataloader, desc=f"Caching {split}")
    for shard_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            if use_fp16 and device.type == "cuda":
                with autocast():
                    cached_inputs = extractor.extract_cache_batch(images)
            else:
                cached_inputs = extractor.extract_cache_batch(images)

        cached_inputs = detach_to_storage(cached_inputs, storage_dtype)
        labels = labels.detach().cpu().long().contiguous()

        shard_path = split_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({"inputs": cached_inputs, "labels": labels}, shard_path)

        num_samples = int(labels.size(0))
        shard_bytes = nested_bytes(cached_inputs) + nested_bytes(labels)
        total_samples += num_samples
        total_bytes += shard_bytes
        shard_records.append(
            {
                "file": shard_path.name,
                "num_samples": num_samples,
                "size_bytes": shard_bytes,
            }
        )
        pbar.set_postfix(
            {
                "samples": total_samples,
                "size_gb": f"{total_bytes / 1024**3:.2f}",
            }
        )

    meta = {
        "split": split,
        "num_samples": total_samples,
        "num_shards": len(shard_records),
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "storage_size_gb": total_bytes / 1024**3,
        "shards": shard_records,
    }
    torch.save(meta, meta_path)
    print(
        f"Saved {split} cache: {split_dir} "
        f"(N={total_samples}, shards={len(shard_records)}, {meta['storage_size_gb']:.2f} GiB)"
    )
    return meta


class CachedShardDataset(Dataset):
    """Dataset backed by cached shard files on disk."""

    def __init__(self, split_dir: Path):
        self.split_dir = Path(split_dir)
        self.meta = torch.load(self.split_dir / "meta.pt", map_location="cpu")
        self.shards = self.meta["shards"]
        self.cumulative: List[int] = []
        running = 0
        for shard in self.shards:
            running += int(shard["num_samples"])
            self.cumulative.append(running)

        self._loaded_shard_idx: Optional[int] = None
        self._loaded_inputs: Optional[Dict[str, Any]] = None
        self._loaded_labels: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.meta["num_samples"])

    def _load_shard(self, shard_idx: int):
        payload = torch.load(self.split_dir / self.shards[shard_idx]["file"], map_location="cpu")
        self._loaded_shard_idx = shard_idx
        self._loaded_inputs = payload["inputs"]
        self._loaded_labels = payload["labels"]

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_idx = bisect.bisect_right(self.cumulative, index)
        prev_end = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_index = index - prev_end

        if self._loaded_shard_idx != shard_idx:
            self._load_shard(shard_idx)

        assert self._loaded_inputs is not None
        assert self._loaded_labels is not None
        return nested_index(self._loaded_inputs, local_index), self._loaded_labels[local_index]

    def shard_bounds(self) -> List[range]:
        """Return global index ranges for each shard."""
        bounds = []
        start = 0
        for shard in self.shards:
            end = start + int(shard["num_samples"])
            bounds.append(range(start, end))
            start = end
        return bounds


class GroupedShardSampler(Sampler[int]):
    """Sampler that keeps samples from the same shard adjacent."""

    def __init__(self, dataset: CachedShardDataset, shuffle: bool = False, seed: int = 42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterable[int]:
        rng = random.Random(self.seed + self.epoch)
        shard_ranges = self.dataset.shard_bounds()
        shard_order = list(range(len(shard_ranges)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for shard_idx in shard_order:
            indices = list(shard_ranges[shard_idx])
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                yield index

    def __len__(self) -> int:
        return len(self.dataset)

"""Shared Torch backend helpers for scoring modules."""

from typing import Callable, TypeVar

import torch


T = TypeVar("T")


def run_with_fallback(fn: Callable[[torch.device, torch.dtype], T]) -> T:
    """Run a scoring kernel on CUDA when available, with CPU fallback."""
    backends = []
    if torch.cuda.is_available():
        backends.append((torch.device("cuda"), torch.float32))
    backends.append((torch.device("cpu"), torch.float64))

    last_error = None
    for device, dtype in backends:
        try:
            return fn(device, dtype)
        except RuntimeError as exc:
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("No available torch backend for scoring.")

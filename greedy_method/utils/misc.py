"""
utils/misc.py
=============
Shared utilities: seed fixing, logging setup, checkpoint helpers,
and device resolution.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def resolve_device(requested: str = "cuda") -> torch.device:
    """
    Return the best available device.
      "cuda" → use GPU if available, else fall back to CPU
      "mps"  → use Apple MPS if available, else fall back to CPU
      "cpu"  → always CPU
    """
    if requested == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "mps":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")

    return device


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(
    log_dir:  str,
    log_name: str = "experiment",
    level:    int = logging.INFO,
) -> logging.Logger:
    """
    Configure root logger to write to both console and a log file.
    Returns the root logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # Avoid duplicate handlers on re-runs
    if not root_logger.handlers:
        root_logger.addHandler(ch)
        root_logger.addHandler(fh)

    return root_logger


# ---------------------------------------------------------------------------
# JSON results persistence
# ---------------------------------------------------------------------------
def save_json(data: dict, path: str) -> None:
    """Save a dict as a pretty-printed JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
def human_param_count(n: int) -> str:
    """Format parameter count: 12345678 → '12.3M'."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

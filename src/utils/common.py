"""
Utility functions
"""
import os
import torch
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get available device"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)


def load_checkpoint(path: str, map_location: str = None) -> Dict[str, Any]:
    """Load checkpoint from disk"""
    if map_location is None:
        map_location = get_device()
    return torch.load(path, map_location=map_location)


def save_checkpoint(data: Dict[str, Any], path: str):
    """Save checkpoint to disk"""
    ensure_dir(os.path.dirname(path))
    torch.save(data, path)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module):
    """Print model information"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")

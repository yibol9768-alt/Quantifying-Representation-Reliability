"""Feature extractor models."""

from .mae import MAEExtractor
from .clip import CLIPExtractor
from .dino import DINOExtractor

__all__ = ["MAEExtractor", "CLIPExtractor", "DINOExtractor"]

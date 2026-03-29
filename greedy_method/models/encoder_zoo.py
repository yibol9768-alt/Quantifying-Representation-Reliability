"""
models/encoder_zoo.py
=====================
Loads and wraps all 10 pretrained visual encoders.

All encoders are **frozen** – they act as fixed feature extractors.
Only the downstream projection heads / fusion weights / classifier
receive gradient updates during training.

Supported sources
-----------------
  timm        : DeiT-Small, ConvNeXt-Tiny, ViT-Base, ConvNeXt-Base, ResNet-50
  huggingface : DINOv2-Small/Base, ViT-MAE-Base, CLIP ViT-B/16, SigLIP

Feature extraction
------------------
  All encoders output a flat 1-D feature vector per image.
  The dimensionality for each model is listed in MODEL_REGISTRY["output_dim"].
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

# ---- 确保模型缓存到数据盘，必须在导入 timm / transformers 之前设置 ----
_MODELS_DIR = '/root/autodl-tmp/models'
_HF_DIR     = os.path.join(_MODELS_DIR, 'huggingface')
os.environ.setdefault('HF_HOME',               _HF_DIR)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(_HF_DIR, 'hub'))
os.environ.setdefault('TORCH_HOME',            os.path.join(_MODELS_DIR, 'torch'))

# ---- 本地模型目录映射 ----
# key: HuggingFace model_id  /  timm model_id
# value: 已下载的本地目录路径
_LOCAL_PATHS: Dict[str, str] = {
    # HuggingFace 格式模型（HF model_id → 本地路径）
    'facebook/dinov2-small':          os.path.join(_HF_DIR, 'dinov2-small'),
    'facebook/dinov2-base':           os.path.join(_HF_DIR, 'dinov2-base'),
    'facebook/vit-mae-base':          os.path.join(_HF_DIR, 'vit-mae-base'),
    'openai/clip-vit-base-patch16':   os.path.join(_HF_DIR, 'clip-vit-base-patch16'),
    'google/siglip-base-patch16-224': os.path.join(_HF_DIR, 'siglip-base'),
    # timm 模型对应的本地 HF Transformers 格式目录（timm model_id → 本地路径）
    'deit_small_patch16_224':         os.path.join(_HF_DIR, 'deit-small'),
    'convnext_tiny':                  os.path.join(_HF_DIR, 'convnext-tiny'),
    'convnext_base':                  os.path.join(_HF_DIR, 'convnext-base'),
    'vit_base_patch16_224':           os.path.join(_HF_DIR, 'vit-base'),
    'resnet50':                       os.path.join(_HF_DIR, 'resnet50'),
}


def _has_weights(path: str) -> bool:
    """判断目录下是否有有效的权重文件。"""
    if not path or not os.path.isdir(path):
        return False
    return any(
        os.path.exists(os.path.join(path, f))
        for f in ('pytorch_model.bin', 'model.safetensors')
    )


def _resolve(model_id: str) -> str:
    """若本地路径存在权重则返回本地路径，否则返回原始 model_id（触发在线下载）。"""
    local = _LOCAL_PATHS.get(model_id, '')
    if _has_weights(local):
        return local
    return model_id


from configs.config import MODEL_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class FrozenEncoder(nn.Module):
    """Frozen pretrained visual encoder that returns a flat feature vector."""

    def __init__(self, name: str, output_dim: int) -> None:
        super().__init__()
        self.name       = name
        self.output_dim = output_dim

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train(self, mode: bool = True):
        # Always stay in eval mode – encoders are frozen
        return super().train(False)


# ---------------------------------------------------------------------------
# timm-based encoders（优先使用本地 HF Transformers 格式，回退到 timm 在线下载）
# ---------------------------------------------------------------------------

# timm model_id → HuggingFace Transformers 类名 + 特征提取方式
_TIMM_HF_MAP = {
    'deit_small_patch16_224': ('ViTModel',      lambda o: o.pooler_output),
    'convnext_tiny':          ('ConvNextModel', lambda o: o.pooler_output),
    'convnext_base':          ('ConvNextModel', lambda o: o.pooler_output),
    'vit_base_patch16_224':   ('ViTModel',      lambda o: o.pooler_output),
    'resnet50':               ('ResNetModel',   lambda o: o.pooler_output.flatten(1)),
}


class TimmEncoder(FrozenEncoder):
    """
    优先从本地 HF Transformers 格式目录加载；若无本地文件则回退到 timm 在线下载。
    """

    def __init__(self, name: str, model_id: str, output_dim: int) -> None:
        super().__init__(name, output_dim)
        local = _resolve(model_id)          # 本地路径 or 原始 model_id
        hf_info = _TIMM_HF_MAP.get(model_id)

        if hf_info and local != model_id:   # 有本地 HF 格式文件
            import transformers
            ModelCls = getattr(transformers, hf_info[0])
            self._model    = ModelCls.from_pretrained(local)
            self._extract  = hf_info[1]
            self._use_hf   = True
            logger.info(f"[TimmEncoder] {name} 从本地 HF 格式加载: {local}")
        else:                               # 回退到 timm 在线下载
            import timm
            self._model   = timm.create_model(model_id, pretrained=True, num_classes=0)
            self._extract = None
            self._use_hf  = False
            logger.info(f"[TimmEncoder] {name} 通过 timm 加载 ({model_id})")

        self._freeze()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_hf:
            return self._extract(self._model(pixel_values=x))
        return self._model(x)  # timm 直接返回特征向量


# ---------------------------------------------------------------------------
# HuggingFace DINOv2 (small and base)
# ---------------------------------------------------------------------------
class DINOv2Encoder(FrozenEncoder):
    """Uses the [CLS] token from the last hidden state."""

    def __init__(self, name: str, model_id: str, output_dim: int) -> None:
        super().__init__(name, output_dim)
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(_resolve(model_id))
        self._freeze()
        logger.info(f"[DINOv2Encoder] Loaded {name} from {_resolve(model_id)}, dim={output_dim}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(pixel_values=x)
        return out.last_hidden_state[:, 0, :]  # CLS token → (B, output_dim)


# ---------------------------------------------------------------------------
# HuggingFace ViT-MAE
# ---------------------------------------------------------------------------
class ViTMAEEncoder(FrozenEncoder):
    """Uses the [CLS] token from the MAE encoder (no decoder needed)."""

    def __init__(self, name: str, model_id: str, output_dim: int) -> None:
        super().__init__(name, output_dim)
        from transformers import ViTMAEModel
        self._model = ViTMAEModel.from_pretrained(_resolve(model_id))
        self._freeze()
        logger.info(f"[ViTMAEEncoder] Loaded {name} from {_resolve(model_id)}, dim={output_dim}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # noise=None disables masking; we want full-image features
        out = self._model(pixel_values=x, noise=None)
        return out.last_hidden_state[:, 0, :]  # CLS token → (B, 768)


# ---------------------------------------------------------------------------
# HuggingFace CLIP Vision
# ---------------------------------------------------------------------------
class CLIPVisionEncoder(FrozenEncoder):
    """
    Uses CLIPModel.get_image_features() which returns the L2-normalised
    512-dim projected visual embedding.
    """

    def __init__(self, name: str, model_id: str, output_dim: int) -> None:
        super().__init__(name, output_dim)
        from transformers import CLIPModel
        self._model = CLIPModel.from_pretrained(_resolve(model_id))
        self._freeze()
        logger.info(f"[CLIPVisionEncoder] Loaded {name} from {_resolve(model_id)}, dim={output_dim}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract pooled features and project to 512-dim
        vision_out = self._model.vision_model(pixel_values=x)
        projected = self._model.visual_projection(vision_out.pooler_output)
        return projected  # (B, 512)


# ---------------------------------------------------------------------------
# HuggingFace SigLIP Vision
# ---------------------------------------------------------------------------
class SigLIPEncoder(FrozenEncoder):
    """Uses the pooler_output (mean-pooled CLS-like token) from SigLIP."""

    def __init__(self, name: str, model_id: str, output_dim: int) -> None:
        super().__init__(name, output_dim)
        from transformers import SiglipVisionModel
        self._model = SiglipVisionModel.from_pretrained(_resolve(model_id))
        self._freeze()
        logger.info(f"[SigLIPEncoder] Loaded {name} from {_resolve(model_id)}, dim={output_dim}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(pixel_values=x)
        return out.pooler_output  # (B, 768)


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------
_HF_DINOV2   = {"dinov2_small", "dinov2_base"}
_HF_MAE      = {"vit_mae_base"}
_HF_CLIP     = {"clip_vit_base"}
_HF_SIGLIP   = {"siglip"}
_TIMM_MODELS = {"deit_small", "convnext_tiny", "vit_base", "convnext_base", "resnet50"}


def build_encoder(name: str) -> FrozenEncoder:
    """Instantiate a frozen encoder by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(MODEL_REGISTRY)}")

    meta      = MODEL_REGISTRY[name]
    model_id  = meta["model_id"]
    out_dim   = meta["output_dim"]

    if name in _TIMM_MODELS:
        return TimmEncoder(name, model_id, out_dim)
    elif name in _HF_DINOV2:
        return DINOv2Encoder(name, model_id, out_dim)
    elif name in _HF_MAE:
        return ViTMAEEncoder(name, model_id, out_dim)
    elif name in _HF_CLIP:
        return CLIPVisionEncoder(name, model_id, out_dim)
    elif name in _HF_SIGLIP:
        return SigLIPEncoder(name, model_id, out_dim)
    else:
        raise ValueError(f"No encoder class registered for '{name}'")


def build_encoder_zoo(names: Optional[list] = None) -> Dict[str, FrozenEncoder]:
    """
    Build a dict of {name: FrozenEncoder} for the requested model names.
    Defaults to all models in MODEL_REGISTRY if names is None.
    """
    if names is None:
        names = list(MODEL_REGISTRY.keys())
    return {name: build_encoder(name) for name in names}

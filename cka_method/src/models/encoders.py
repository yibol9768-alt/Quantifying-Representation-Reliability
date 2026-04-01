"""
预训练模型池：使用 transformers 从本地路径加载冻结编码器。

每个编码器通过 `get_encoder(name, model_root)` 获取，返回一个冻结的 nn.Module，
其 forward 输出 shape = (batch, feat_dim)。
"""
import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Dict, Callable

from src.config import MODEL_REGISTRY


# ─────────────────── 通用包装器 ───────────────────

class EncoderWrapper(nn.Module):
    """统一封装：冻结模型 → (B, feat_dim) 特征输出"""

    def __init__(self, backbone: nn.Module, feat_dim: int,
                 extract_fn: Callable = None):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self._extract_fn = extract_fn
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._extract_fn is not None:
            return self._extract_fn(self.backbone, x)
        return self.backbone(x)


# ─────────────────── 各 loader 实现 ───────────────────

def _load_vit(local_path: str) -> tuple:
    """加载 ViT / DeiT 模型（CLS token 特征）"""
    from transformers import ViTModel
    model = ViTModel.from_pretrained(local_path)
    model.eval()

    def extract(m, x):
        out = m(pixel_values=x)
        return out.last_hidden_state[:, 0]  # CLS token

    return model, extract


def _load_dinov2(local_path: str) -> tuple:
    """加载 DINOv2 模型（CLS token 特征）"""
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(local_path)
    model.eval()

    def extract(m, x):
        out = m(pixel_values=x)
        return out.last_hidden_state[:, 0]

    return model, extract


def _load_convnext(local_path: str) -> tuple:
    """加载 ConvNeXt 模型（全局平均池化特征）"""
    from transformers import ConvNextModel
    model = ConvNextModel.from_pretrained(local_path)
    model.eval()

    def extract(m, x):
        out = m(pixel_values=x)
        return out.pooler_output.squeeze(-1).squeeze(-1)

    return model, extract


def _load_vit_mae(local_path: str) -> tuple:
    """加载 ViT-MAE 模型（CLS token 特征）"""
    from transformers import ViTMAEModel
    model = ViTMAEModel.from_pretrained(local_path)
    model.eval()

    def extract(m, x):
        out = m(pixel_values=x)
        return out.last_hidden_state[:, 0]

    return model, extract


def _load_clip(local_path: str) -> tuple:
    """加载 CLIP 模型（视觉投影后特征，dim=512）"""
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(local_path)
    model.eval()

    def extract(m, x):
        vout = m.vision_model(pixel_values=x)
        return m.visual_projection(vout.pooler_output)

    return model, extract


def _load_siglip(local_path: str) -> tuple:
    """加载 SigLIP 模型（视觉特征）"""
    from transformers import SiglipModel
    model = SiglipModel.from_pretrained(local_path)
    model.eval()

    def extract(m, x):
        vision_out = m.vision_model(pixel_values=x)
        pooled = vision_out.pooler_output
        return pooled

    return model, extract


def _load_resnet(local_path: str) -> tuple:
    """加载 torchvision ResNet50（去掉分类头）"""
    weights = tv_models.ResNet50_Weights.DEFAULT
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model, None  # 默认 forward 已返回 2048-dim


# ─────────────────── loader 路由表 ───────────────────

_LOADERS = {
    "vit": _load_vit,
    "dinov2": _load_dinov2,
    "convnext": _load_convnext,
    "vit_mae": _load_vit_mae,
    "clip": _load_clip,
    "siglip": _load_siglip,
    "resnet": _load_resnet,
}


# ─────────────────── 公开 API ───────────────────

def get_encoder(name: str, model_root: str = "/root/autodl-tmp/models",
                device: str = "cuda") -> EncoderWrapper:
    """
    按名称加载冻结编码器。

    Args:
        name:       MODEL_REGISTRY 中的键，如 "clip_vit_base"
        model_root: 模型存储根目录
        device:     放置设备

    Returns:
        EncoderWrapper，forward 输出 (B, feat_dim)
    """
    info = MODEL_REGISTRY[name]
    loader_fn = _LOADERS[info["loader"]]

    local_path = os.path.join(model_root, info["local_dir"]) if info["local_dir"] else ""
    backbone, extract_fn = loader_fn(local_path)

    encoder = EncoderWrapper(backbone, feat_dim=info["feat_dim"],
                             extract_fn=extract_fn)
    return encoder.to(device).eval()


def get_all_encoders(
    model_names=None, model_root: str = "/root/autodl-tmp/models",
    device: str = "cuda"
) -> Dict[str, EncoderWrapper]:
    """加载多个编码器，返回 {name: encoder} 字典"""
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())
    encoders = {}
    for name in model_names:
        print(f"  Loading encoder: {name}...")
        encoders[name] = get_encoder(name, model_root, device)
    return encoders

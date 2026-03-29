"""
configs/config.py
=================
全局配置：模型注册表、数据集注册表、训练超参数。

本项目只使用 few-shot + 原型距离分类，不涉及全量数据训练或 MLP 分类头。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 模型注册表（10 个候选预训练编码器）
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "deit_small": {
        "source":      "timm",
        "model_id":    "deit_small_patch16_224",
        "output_dim":  384,
        "param_M":     22,
        "description": "DeiT-Small (22M) – 轻量级监督 ViT",
    },
    "dinov2_small": {
        "source":      "huggingface",
        "model_id":    "facebook/dinov2-small",
        "output_dim":  384,
        "param_M":     22,
        "description": "DINOv2-Small (22M) – 自监督，强局部特征",
    },
    "convnext_tiny": {
        "source":      "timm",
        "model_id":    "convnext_tiny",
        "output_dim":  768,
        "param_M":     28,
        "description": "ConvNeXt-Tiny (28M) – 现代 CNN，强纹理/边缘感知",
    },
    "vit_base": {
        "source":      "timm",
        "model_id":    "vit_base_patch16_224",
        "output_dim":  768,
        "param_M":     86,
        "description": "ViT-Base (86M) – 标准监督全局注意力基线",
    },
    "convnext_base": {
        "source":      "timm",
        "model_id":    "convnext_base",
        "output_dim":  1024,
        "param_M":     89,
        "description": "ConvNeXt-Base – 强力现代 CNN",
    },
    "vit_mae_base": {
        "source":      "huggingface",
        "model_id":    "facebook/vit-mae-base",
        "output_dim":  768,
        "param_M":     86,
        "description": "MAE ViT-Base (86M) – 掩码重建，擅长细粒度局部特征",
    },
    "dinov2_base": {
        "source":      "huggingface",
        "model_id":    "facebook/dinov2-base",
        "output_dim":  768,
        "param_M":     86,
        "description": "DINOv2-Base (86M) – 细粒度分类与密集预测",
    },
    "clip_vit_base": {
        "source":      "huggingface",
        "model_id":    "openai/clip-vit-base-patch16",
        "output_dim":  512,
        "param_M":     86,
        "description": "CLIP ViT-Base (86M) – 图文对比，零样本语义泛化",
    },
    "resnet50": {
        "source":      "timm",
        "model_id":    "resnet50",
        "output_dim":  2048,
        "param_M":     25,
        "description": "ResNet-50 (25M) – 经典 CNN 基线，稳定高效",
    },
    "siglip": {
        "source":      "huggingface",
        "model_id":    "google/siglip-base-patch16-224",
        "output_dim":  768,
        "param_M":     86,
        "description": "SigLIP (86M) – CLIP 升级版，更精细的特征表达",
    },
}

ALL_MODELS: List[str] = list(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 数据集注册表
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gtsrb":      {"num_classes": 43,   "description": "德国交通标志识别"},
    "svhn":       {"num_classes": 10,   "description": "街景门牌数字"},
    "dtd":        {"num_classes": 47,   "description": "可描述纹理数据集"},
    "eurosat":    {"num_classes": 10,   "description": "EuroSAT 卫星图像"},
    "pets":       {"num_classes": 37,   "description": "Oxford-IIIT 细粒度宠物"},
    "country211": {"num_classes": 211,  "description": "Country211 地理定位"},
    "imagenet":   {"num_classes": 1000, "description": "ImageNet-1K 通用基准"},
}


# ---------------------------------------------------------------------------
# 训练配置（仅 few-shot 范式）
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # ---- 网络架构 ----
    fusion_dim: int = 512          # 所有编码器统一投影到的公共维度

    # ---- 优化器 ----
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4

    # ---- 数据 ----
    data_root:    str   = "/root/autodl-tmp/data/raw"
    num_workers:  int   = 4
    val_fraction: float = 0.15     # 无官方 val 时从 train 切出的比例
    image_size:   int   = 224

    # ---- 特征缓存 ----
    cache_dir: str = "/root/autodl-tmp/data/feature_cache"

    # ---- 贪心搜索专用（固定 K 用于模型选择打分）----
    search_k_shot:         int = 5    # 贪心搜索阶段的 K-shot
    search_n_way:          int = 5    # 贪心搜索阶段的 N-way
    search_n_query:        int = 15   # 每类 query 样本数
    search_train_episodes: int = 100  # 每轮训练情节数
    search_val_episodes:   int = 200  # 每次评分时的验证情节数
    search_epochs:         int = 10   # 每个候选组合的训练轮数

    # ---- 最终 few-shot 评估 ----
    eval_k_values:    List[int] = field(default_factory=lambda: [1, 5, 10, 16])
    eval_n_way:       int       = None  # None = 自动使用数据集实际类别数
    eval_n_query:     int       = 15
    eval_episodes:    int       = 600  # 测试情节数（越多结果越稳定）
    eval_train_episodes: int    = 200  # 最终评估前的训练情节数/轮
    eval_epochs:      int       = 20   # 最终评估的训练轮数

    # ---- 输出路径 ----
    log_dir:    str = "/root/autodl-tmp/logs"
    results_dir: str = "/root/autodl-tmp/results"

    # ---- 硬件 ----
    device: str = "cuda"
    seed:   int = 42


# ---------------------------------------------------------------------------
# 顶层实验配置
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    dataset:          str       = "dtd"
    candidate_models: List[str] = field(default_factory=lambda: list(MODEL_REGISTRY.keys()))
    training:         TrainingConfig = field(default_factory=TrainingConfig)

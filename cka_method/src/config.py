"""
全局配置：超参数、模型池定义、数据集定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ──────────────────────────── 模型池定义 ────────────────────────────
# local_dir: 相对于 model_root 的路径
# loader: transformers 加载方式
# feat_dim: 冻结特征维度

MODEL_REGISTRY: Dict[str, dict] = {
    "deit_small": {
        "loader": "vit",
        "local_dir": "huggingface/deit-small",
        "feat_dim": 384,
        "params": "22M",
    },
    "dinov2_small": {
        "loader": "dinov2",
        "local_dir": "huggingface/dinov2-small",
        "feat_dim": 384,
        "params": "22M",
    },
    "convnext_tiny": {
        "loader": "convnext",
        "local_dir": "huggingface/convnext-tiny",
        "feat_dim": 768,
        "params": "28M",
    },
    "vit_base": {
        "loader": "vit",
        "local_dir": "huggingface/vit-base",
        "feat_dim": 768,
        "params": "86M",
    },
    "convnext_base": {
        "loader": "convnext",
        "local_dir": "huggingface/convnext-base",
        "feat_dim": 1024,
        "params": "86M",
    },
    "vit_mae_base": {
        "loader": "vit_mae",
        "local_dir": "huggingface/vit-mae-base",
        "feat_dim": 768,
        "params": "86M",
    },
    "dinov2_base": {
        "loader": "dinov2",
        "local_dir": "huggingface/dinov2-base",
        "feat_dim": 768,
        "params": "86M",
    },
    "clip_vit_base": {
        "loader": "clip",
        "local_dir": "huggingface/clip-vit-base-patch16",
        "feat_dim": 512,
        "params": "86M",
    },
    "resnet50": {
        "loader": "resnet",
        "local_dir": "",
        "feat_dim": 2048,
        "params": "25M",
    },
    "siglip": {
        "loader": "siglip",
        "local_dir": "huggingface/siglip-base",
        "feat_dim": 768,
        "params": "86M",
    },
}

# ──────────────────────────── 数据集定义 ────────────────────────────

DATASET_REGISTRY: Dict[str, dict] = {
    # 简单
    "gtsrb":      {"num_classes": 43,  "category": "simple"},
    "svhn":       {"num_classes": 10,  "category": "simple"},
    # 复杂
    "dtd":        {"num_classes": 47,  "category": "complex"},
    "eurosat":    {"num_classes": 10,  "category": "complex"},
    "pets":       {"num_classes": 37,  "category": "complex"},
    "country211": {"num_classes": 211, "category": "complex"},
    # 通用性验证
    "imagenet":   {"num_classes": 1000, "category": "general"},
}


# ──────────────────────────── 实验超参 ────────────────────────────

@dataclass
class ExpConfig:
    """一次实验的完整配置"""

    # 数据集
    dataset: str = "dtd"
    data_root: str = "/root/autodl-tmp/data/raw"

    # Few-shot 设定
    n_shot: int = 16                   # K: 每类样本数
    n_query: int = 50                  # 每类查询样本数（评估用）

    # 模型选择超参
    alpha: float = 1.0                 # 任务相关性权重指数
    beta: float = 1.0                  # 冗余惩罚权重指数

    # 融合架构超参
    d_proj: int = 512                  # 公共隐空间维度

    # Episode 训练超参
    n_way: int = 5                 # Phase 2 episode 训练的 way 数
    n_train_episodes: int = 100    # 每 epoch 训练 episode 数
    n_train_epochs: int = 10       # 训练 epoch 数
    n_eval_episodes: int = 200     # 测试 episode 数
    n_eval_query: int = 15         # 每类 query 样本数

    # 优化器超参
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # 通用
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    output_dir: str = "/root/autodl-tmp/results/cka_method"
    model_root: str = "/root/autodl-tmp/models"

    # 模型池（默认全部）
    model_names: List[str] = field(
        default_factory=lambda: list(MODEL_REGISTRY.keys())
    )

    @property
    def num_classes(self) -> int:
        return DATASET_REGISTRY[self.dataset]["num_classes"]

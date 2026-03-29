"""
Few-shot 数据集加载器。

支持数据集：GTSRB, SVHN, DTD, EuroSAT, Pets, Country211, ImageNet
提供 N-way K-shot 的 support/query 划分。
"""
import os
import random
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
import torchvision.datasets as tv_datasets

from src.config import DATASET_REGISTRY


# ──────────────────── 标准预处理（224×224） ────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = TRAIN_TRANSFORM  # few-shot 不做数据增强


# ──────────────────── 数据集构造路由 ────────────────────

def _build_dataset(name: str, root: str, split: str) -> Dataset:
    """根据名称返回 torchvision 数据集实例"""
    train = (split == "train")

    if name == "gtsrb":
        return tv_datasets.GTSRB(root, split="train" if train else "test",
                                  transform=TRAIN_TRANSFORM, download=True)
    elif name == "svhn":
        return tv_datasets.SVHN(root, split="train" if train else "test",
                                 transform=TRAIN_TRANSFORM, download=True)
    elif name == "dtd":
        return tv_datasets.DTD(root, split="train" if train else "test",
                                transform=TRAIN_TRANSFORM, download=True)
    elif name == "eurosat":
        ds = tv_datasets.EuroSAT(root, transform=TRAIN_TRANSFORM, download=True)
        return ds  # EuroSAT 无官方 split，后续统一做 few-shot 划分
    elif name == "pets":
        return tv_datasets.OxfordIIITPet(
            root, split="trainval" if train else "test",
            target_types="category", transform=TRAIN_TRANSFORM, download=True,
        )
    elif name == "country211":
        return tv_datasets.Country211(root, split="train" if train else "test",
                                       transform=TRAIN_TRANSFORM, download=True)
    elif name == "imagenet":
        # ImageNet 需要手动下载，指定路径
        split_dir = os.path.join(root, "imagenet", "train" if train else "val")
        return tv_datasets.ImageFolder(split_dir, transform=TRAIN_TRANSFORM)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ──────────────────── 按类别索引样本 ────────────────────

def _get_targets(dataset: Dataset) -> List[int]:
    """兼容不同数据集的标签获取方式"""
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if hasattr(dataset, "_labels"):
        return list(dataset._labels)
    # fallback: 遍历
    return [dataset[i][1] for i in range(len(dataset))]


def _group_by_class(dataset: Dataset) -> Dict[int, List[int]]:
    """返回 {class_id: [sample_indices]}"""
    targets = _get_targets(dataset)
    class_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(idx)
    return class_to_indices


# ──────────────────── Few-shot 划分 ────────────────────

class FewShotSplit:
    """存储一次 few-shot 划分的 support 和 query 索引"""

    def __init__(self, dataset: Dataset,
                 support_indices: List[int],
                 query_indices: List[int]):
        self.dataset = dataset
        self.support = Subset(dataset, support_indices)
        self.query = Subset(dataset, query_indices)
        self.support_indices = support_indices
        self.query_indices = query_indices
        self.all_indices = support_indices + query_indices


def build_fewshot_split(
    name: str,
    root: str,
    n_shot: int,
    n_query: int = 50,
    seed: int = 42,
) -> FewShotSplit:
    """
    构建 N-way K-shot 的 support/query 划分。

    Args:
        name:    数据集名称
        root:    数据根目录
        n_shot:  每类 support 样本数 (K)
        n_query: 每类 query 样本数
        seed:    随机种子

    Returns:
        FewShotSplit 对象
    """
    dataset = _build_dataset(name, root, split="train")
    class_to_indices = _group_by_class(dataset)

    rng = random.Random(seed)

    support_indices = []
    query_indices = []

    for cls_id, indices in sorted(class_to_indices.items()):
        rng.shuffle(indices)
        support_indices.extend(indices[:n_shot])
        query_indices.extend(indices[n_shot:n_shot + n_query])

    return FewShotSplit(dataset, support_indices, query_indices)


def build_train_dataset(name: str, root: str) -> Dataset:
    """构建全量训练集（用于 Phase 2 episode 训练）"""
    return _build_dataset(name, root, split="train")


def build_test_dataset(name: str, root: str) -> Dataset:
    """构建测试集（用于最终评估）"""
    return _build_dataset(name, root, split="test")

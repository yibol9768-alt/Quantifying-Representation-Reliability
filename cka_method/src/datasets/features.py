"""
特征提取与缓存工具。

预提取所有冻结编码器的特征，避免训练时重复前向传播。
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
from tqdm import tqdm

from src.models.encoders import EncoderWrapper


@torch.no_grad()
def extract_features(
    encoder: EncoderWrapper,
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    提取单个编码器在整个数据集上的冻结特征。

    Returns:
        features: (N_samples, feat_dim)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_feats = []
    encoder.eval()
    for images, _ in tqdm(loader, desc="Extracting", leave=False):
        images = images.to(device)
        feats = encoder(images)  # (B, feat_dim)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def extract_labels(dataset: Dataset) -> torch.Tensor:
    """提取数据集的全部标签"""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_labels = []
    for _, labels in loader:
        all_labels.append(labels)
    return torch.cat(all_labels, dim=0)


def extract_all_features(
    encoders: Dict[str, EncoderWrapper],
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    对所有编码器提取特征。

    Returns:
        {model_name: (N_samples, feat_dim)}
    """
    features = {}
    for name, encoder in encoders.items():
        print(f"  Extracting features from {name}...")
        features[name] = extract_features(
            encoder, dataset, batch_size, num_workers, device
        )
    return features

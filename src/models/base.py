"""
模型基类
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseModel(ABC):
    """预训练模型基类"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.preprocess = None
        self.feature_dim = None
        self.model_name = None

    @abstractmethod
    def load_model(self):
        """加载预训练模型"""
        pass

    @abstractmethod
    def get_transform(self):
        """获取图像预处理变换"""
        pass

    @abstractmethod
    def extract_feature(self, image) -> torch.Tensor:
        """
        从图像提取特征

        Args:
            image: PIL Image 或预处理后的 tensor

        Returns:
            torch.Tensor: 归一化后的特征向量 [1, feature_dim]
        """
        pass

    def extract_batch_features(self, images) -> torch.Tensor:
        """
        批量提取特征

        Args:
            images: 图像批次

        Returns:
            torch.Tensor: 特征矩阵 [batch_size, feature_dim]
        """
        features = self.extract_feature(images)
        return self._normalize(features)

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2 归一化"""
        return features / features.norm(dim=-1, keepdim=True)

    def to_device(self, tensor):
        """将张量移动到指定设备"""
        return tensor.to(self.device)

    def eval(self):
        """设置为评估模式"""
        if self.model is not None:
            self.model.eval()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.model_name}, dim={self.feature_dim})"

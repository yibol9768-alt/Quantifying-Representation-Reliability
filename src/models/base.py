"""
Base model class for pre-trained vision models
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseModel(ABC):
    """Base class for pre-trained vision models"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.preprocess = None
        self.feature_dim = None
        self.model_name = None

    @abstractmethod
    def load_model(self):
        """Load pre-trained model"""
        pass

    @abstractmethod
    def get_transform(self):
        """Get image preprocessing transform"""
        pass

    @abstractmethod
    def extract_feature(self, image) -> torch.Tensor:
        """
        Extract feature from image.

        Args:
            image: PIL Image or preprocessed tensor

        Returns:
            torch.Tensor: Normalized feature vector [1, feature_dim]
        """
        pass

    def extract_batch_features(self, images) -> torch.Tensor:
        """
        Extract features from a batch of images.

        Args:
            images: Batch of images

        Returns:
            torch.Tensor: Feature matrix [batch_size, feature_dim]
        """
        features = self.extract_feature(images)
        return self._normalize(features)

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features"""
        return features / features.norm(dim=-1, keepdim=True)

    def to_device(self, tensor):
        """Move tensor to device"""
        return tensor.to(self.device)

    def eval(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.model_name}, dim={self.feature_dim})"

"""
DINO model wrapper
"""
import torch
from torchvision import transforms
from typing import Optional
from PIL import Image

from .base import BaseModel


class DINOModel(BaseModel):
    """Facebook DINO model wrapper"""

    def __init__(self, device: str = "cuda", model_name: str = "dino_vitb16"):
        super().__init__(device)
        self.model_name = model_name
        self.feature_dim = 768
        self.load_model()

    def load_model(self):
        """Load DINO model from torch hub"""
        self.model = torch.hub.load('facebookresearch/dino:main', self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # DINO preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_transform(self):
        """Get DINO preprocessing transform"""
        return self.preprocess

    def extract_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Extract feature from image using DINO

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Normalized feature vector [1, 768]
        """
        if self.preprocess is None:
            self.load_model()

        # Preprocess
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(image_input)

        # L2 normalize
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature

    def extract_batch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from batch of preprocessed images

        Args:
            images: Batch of preprocessed tensors [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Feature matrix [batch_size, 768]
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        with torch.no_grad():
            features = self.model(images)

        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)

        return features

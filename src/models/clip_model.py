"""
CLIP model wrapper
"""
import torch
import clip
from typing import Optional
from PIL import Image

from .base import BaseModel


class CLIPModel(BaseModel):
    """OpenAI CLIP model wrapper"""

    def __init__(self, device: str = "cuda", model_name: str = "ViT-B/32"):
        super().__init__(device)
        self.model_name = model_name
        self.feature_dim = 512
        self.load_model()

    def load_model(self):
        """Load CLIP model from OpenAI"""
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

    def get_transform(self):
        """Get CLIP preprocessing transform"""
        return self.preprocess

    def extract_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Extract feature from image using CLIP

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Normalized feature vector [1, 512]
        """
        if self.preprocess is None:
            self.load_model()

        # Preprocess and extract
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model.encode_image(image_input)

        # L2 normalize
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature

    def extract_batch_features(self, images) -> torch.Tensor:
        """
        Extract features from batch of images

        Args:
            images: Batch of PIL Images or preprocessed tensor

        Returns:
            torch.Tensor: Feature matrix [batch_size, 512]
        """
        if isinstance(images, list):
            # List of PIL Images
            images = torch.stack([self.preprocess(img) for img in images])

        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(images)

        return self._normalize(features)

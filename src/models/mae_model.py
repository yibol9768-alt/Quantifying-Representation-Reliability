"""
MAE model wrapper
"""
import torch
import os
from transformers import ViTMAEModel, AutoImageProcessor
from typing import Optional
from PIL import Image

from .base import BaseModel


class MAEModel(BaseModel):
    """Facebook MAE model wrapper"""

    def __init__(self, device: str = "cuda", model_name: str = "facebook/vit-mae-base"):
        super().__init__(device)
        self.model_name = model_name
        self.feature_dim = 768
        self.load_model()

    def load_model(self):
        """Load MAE model from Hugging Face (with mirror for China)"""
        # Use mirror for faster download in China
        mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = mirror

        # 使用环境变量方式设置镜像
        import transformers
        # 设置 endpoints
        if hasattr(transformers, 'file_utils'):
            transformers.file_utils.HF_HUB_OFFLINE = False

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = ViTMAEModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_transform(self):
        """Get MAE preprocessing (returns processor)"""
        return self.processor

    def extract_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Extract feature from image using MAE

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Normalized feature vector [1, 768]
        """
        # Preprocess using MAE processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract CLS token
        feature = outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature

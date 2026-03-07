"""
MAE model wrapper
"""
import os
from typing import List, Union

import torch
from transformers import AutoImageProcessor, ViTMAEModel
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
        """Load MAE model from local path or Hugging Face."""
        mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = mirror

        import transformers
        if hasattr(transformers, "file_utils"):
            transformers.file_utils.HF_HUB_OFFLINE = False

        source = os.environ.get("MAE_MODEL_PATH", self.model_name)
        local_files_only = os.path.isdir(source)

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                source,
                local_files_only=local_files_only,
            )
            self.model = ViTMAEModel.from_pretrained(
                source,
                local_files_only=local_files_only,
            ).to(self.device)
        except OSError:
            if local_files_only and source != self.model_name:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = ViTMAEModel.from_pretrained(self.model_name).to(self.device)
            else:
                raise

        self.model.eval()

    def get_transform(self):
        """Get MAE preprocessing transform that returns pixel_values tensor."""

        def _transform(image: Image.Image) -> torch.Tensor:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

        return _transform

    def extract_feature(self, image: Image.Image) -> torch.Tensor:
        """
        Extract feature from image using MAE.

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Normalized feature vector [1, 768]
        """
        if isinstance(image, torch.Tensor):
            pixel_values = image
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.to(self.device)
            inputs = {"pixel_values": pixel_values}
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract CLS token
        feature = outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature

    def extract_batch_features(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract MAE features from a batch without per-image model calls.

        Args:
            images: Preprocessed batch tensor [B, C, H, W] or list of PIL images

        Returns:
            torch.Tensor: Feature matrix [B, 768]
        """
        if isinstance(images, list):
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
        else:
            pixel_values = images
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        features = outputs.last_hidden_state[:, 0, :]
        return self._normalize(features)

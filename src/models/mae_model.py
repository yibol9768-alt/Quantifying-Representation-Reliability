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
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, mirror=mirror)
        self.model = ViTMAEModel.from_pretrained(self.model_name, mirror=mirror).to(self.device)
        self.model.eval()

        # Create a torchvision transform compatible with extract.py
        # MAE uses ImageNet normalization
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_transform(self):
        """Get MAE preprocessing as torchvision transform"""
        return self.preprocess

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
            # MAE expects pixel values in [0, 1], but we have normalized tensors
            # So we need to use the processor to properly preprocess
            # Convert back to PIL first
            from torchvision import transforms
            to_pil = transforms.ToPILImage()

            batch_features = []
            for i in range(images.shape[0]):
                # Denormalize
                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)

                img_pil = to_pil(img_tensor)
                inputs = self.processor(images=img_pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                feature = outputs.last_hidden_state[:, 0, :]
                batch_features.append(feature)

            features = torch.cat(batch_features, dim=0)

        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)

        return features

"""
Feature extraction module
"""
import os
import torch
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from pathlib import Path

from src.models import get_model, MODEL_CONFIGS
from src.data import StanfordCarsDataset


class FeatureExtractor:
    """Extract features from pre-trained models"""

    def __init__(self, device: str = "cuda", data_root: str = "stanford_cars"):
        self.device = device
        self.data_root = data_root
        self.dataset = StanfordCarsDataset(data_root)
        self.models = {}

    def load_model(self, model_type: str):
        """Load a single model"""
        if model_type not in self.models:
            print(f"Loading {model_type} model...")
            self.models[model_type] = get_model(model_type, self.device)

    def load_models(self, model_types: List[str]):
        """Load multiple models"""
        for model_type in model_types:
            self.load_model(model_type)

    def extract(
        self,
        model_types: Union[str, List[str]],
        split: str = "train",
        output_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features for specified models

        Args:
            model_types: Single model type or list of model types
            split: "train" or "test"
            output_path: Path to save features (.pt file)

        Returns:
            Dict with features and labels
        """
        if isinstance(model_types, str):
            model_types = [model_types]

        # Load models
        self.load_models(model_types)

        # Get feature dimensions
        feature_dims = [MODEL_CONFIGS[m]["feature_dim"] for m in model_types]

        # Load data
        print(f"Loading {split} data...")
        if split == "train":
            image_paths, labels = self.dataset.load_train_data()
        else:
            image_paths, labels = self.dataset.load_test_data(with_labels=True)

        # Initialize storage
        features_dict = {f"{m}_features": [] for m in model_types}
        labels_list = []

        # Extract features
        print(f"Extracting features with {model_types}...")
        with torch.no_grad():
            for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
                try:
                    img = self.dataset.get_image(img_path)

                    for i, model_type in enumerate(model_types):
                        model = self.models[model_type]
                        feat = model.extract_feature(img).cpu()
                        features_dict[f"{model_type}_features"].append(feat)

                    labels_list.append(label)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        # Concatenate features
        result = {}
        for model_type in model_types:
            key = f"{model_type}_features"
            result[key] = torch.cat(features_dict[key], dim=0)

        result["labels"] = torch.tensor(labels_list, dtype=torch.long)

        # Print info
        print(f"\nExtraction complete:")
        for model_type in model_types:
            key = f"{model_type}_features"
            print(f"  {model_type}: {result[key].shape}")

        # Save if path provided
        if output_path:
            self._save_features(result, output_path)

        return result

    def _save_features(self, features: Dict, path: str):
        """Save features to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(features, path)
        print(f"Features saved to {path}")

    @staticmethod
    def load_features(path: str) -> Dict[str, torch.Tensor]:
        """Load features from disk"""
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    # Test single model extraction
    extractor = FeatureExtractor(device="cuda")

    # Extract CLIP features
    features = extractor.extract(
        model_types="clip",
        split="train",
        output_path="features/clip_train.pt"
    )

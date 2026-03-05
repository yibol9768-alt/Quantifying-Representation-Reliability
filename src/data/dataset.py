"""
Stanford Cars dataset loader
"""
import os
import scipy.io
from typing import List, Tuple, Optional
from PIL import Image


class StanfordCarsDataset:
    """Stanford Cars dataset handler"""

    def __init__(self, data_root: str = "stanford_cars"):
        self.data_root = data_root
        self.train_dir = os.path.join(data_root, "cars_train")
        self.test_dir = os.path.join(data_root, "cars_test")
        self.devkit_dir = os.path.join(data_root, "devkit")

        # Label mapping (196 car classes)
        self.num_classes = 196

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        """
        Load training data paths and labels

        Returns:
            Tuple of (image_paths, labels)
        """
        anno_path = os.path.join(self.devkit_dir, "cars_train_annos.mat")
        return self._load_annotations(anno_path, self.train_dir)

    def load_test_data(self, with_labels: bool = True) -> Tuple[List[str], List[int]]:
        """
        Load test data paths and labels

        Args:
            with_labels: If True, load from labeled test file

        Returns:
            Tuple of (image_paths, labels)
        """
        if with_labels:
            anno_path = os.path.join(self.data_root, "cars_test_annos_withlabels.mat")
        else:
            anno_path = os.path.join(self.devkit_dir, "cars_test_annos.mat")

        return self._load_annotations(anno_path, self.test_dir)

    def _load_annotations(self, anno_path: str, img_dir: str) -> Tuple[List[str], List[int]]:
        """
        Parse .mat annotation file

        Args:
            anno_path: Path to annotation .mat file
            img_dir: Directory containing images

        Returns:
            Tuple of (image_paths, labels)
        """
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        mat_data = scipy.io.loadmat(anno_path)
        annotations = mat_data['annotations']

        image_paths = []
        labels = []

        for anno in annotations[0]:
            img_name = anno['fname'][0]
            # Convert from 1-196 to 0-195
            label = int(anno['class'][0][0]) - 1

            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(label)

        print(f"Loaded {len(image_paths)} images from {anno_path}")
        return image_paths, labels

    def get_image(self, path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(path).convert("RGB")

    def verify_dataset(self) -> dict:
        """
        Verify dataset integrity

        Returns:
            Dict with verification results
        """
        result = {
            "train_exists": os.path.exists(self.train_dir),
            "test_exists": os.path.exists(self.test_dir),
            "devkit_exists": os.path.exists(self.devkit_dir),
            "train_annos_exists": os.path.exists(
                os.path.join(self.devkit_dir, "cars_train_annos.mat")
            ),
            "test_annos_with_labels": os.path.exists(
                os.path.join(self.data_root, "cars_test_annos_withlabels.mat")
            ),
        }

        # Count images
        if result["train_exists"]:
            result["train_images"] = len([f for f in os.listdir(self.train_dir) if f.endswith(('.jpg', '.png'))])
        if result["test_exists"]:
            result["test_images"] = len([f for f in os.listdir(self.test_dir) if f.endswith(('.jpg', '.png'))])

        return result


if __name__ == "__main__":
    # Test dataset loading
    dataset = StanfordCarsDataset()
    verification = dataset.verify_dataset()
    print("Dataset verification:")
    for k, v in verification.items():
        print(f"  {k}: {v}")

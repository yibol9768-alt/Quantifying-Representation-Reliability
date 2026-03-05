"""
Multiple dataset loaders for fine-grained and general classification

Supported datasets:
- Stanford Cars (original)
- CIFAR-10
- CIFAR-100
- Flowers-102
- Oxford-IIIT Pets
- Food-101
"""
import os
import torch
from torchvision import datasets
from typing import Tuple, List, Optional
from PIL import Image


class BaseDataset:
    """Base dataset class"""

    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        self.num_classes = None
        self.name = None

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        """Load training data paths and labels"""
        raise NotImplementedError

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        """Load test data paths and labels"""
        raise NotImplementedError

    def get_image(self, path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(path).convert("RGB")


class StanfordCarsDataset(BaseDataset):
    """Stanford Cars dataset handler"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.name = "stanford_cars"
        self.data_path = os.path.join(data_root, "stanford_cars")
        self.train_dir = os.path.join(self.data_path, "cars_train")
        self.test_dir = os.path.join(self.data_path, "cars_test")
        self.devkit_dir = os.path.join(self.data_path, "devkit")
        self.num_classes = 196

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        import scipy.io
        anno_path = os.path.join(self.devkit_dir, "cars_train_annos.mat")
        return self._load_annotations(anno_path, self.train_dir)

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        import scipy.io
        anno_path = os.path.join(self.data_path, "cars_test_annos_withlabels.mat")
        return self._load_annotations(anno_path, self.test_dir)

    def _load_annotations(self, anno_path: str, img_dir: str) -> Tuple[List[str], List[int]]:
        import scipy.io
        mat_data = scipy.io.loadmat(anno_path)
        annotations = mat_data['annotations']

        image_paths = []
        labels = []

        for anno in annotations[0]:
            img_name = anno['fname'][0]
            label = int(anno['class'][0][0]) - 1
            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(label)

        print(f"Loaded {len(image_paths)} images from {anno_path}")
        return image_paths, labels


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset (10 classes, 32x32 images)"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.name = "cifar10"
        self.num_classes = 10

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
        )
        return self._torchvision_to_paths(dataset)

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
        )
        return self._torchvision_to_paths(dataset)

    def _torchvision_to_paths(self, dataset) -> Tuple[List[str], List[int]]:
        # Convert PIL images to temporary saved paths
        import tempfile
        temp_dir = os.path.join(self.data_root, self.name, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        image_paths = []
        labels = []

        for idx, (img, label) in enumerate(dataset):
            img_path = os.path.join(temp_dir, f"img_{idx}.png")
            img.save(img_path)
            image_paths.append(img_path)
            labels.append(label)

        print(f"Loaded {len(image_paths)} images for {self.name}")
        return image_paths, labels


class CIFAR100Dataset(BaseDataset):
    """CIFAR-100 dataset (100 classes, 32x32 images)"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.name = "cifar100"
        self.num_classes = 100

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
        )
        return self._torchvision_to_paths(dataset)

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.CIFAR100(
            root=self.data_root,
            train=False,
            download=True,
        )
        return self._torchvision_to_paths(dataset)

    def _torchvision_to_paths(self, dataset) -> Tuple[List[str], List[int]]:
        import tempfile
        temp_dir = os.path.join(self.data_root, self.name, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        image_paths = []
        labels = []

        for idx, (img, label) in enumerate(dataset):
            img_path = os.path.join(temp_dir, f"img_{idx}.png")
            img.save(img_path)
            image_paths.append(img_path)
            labels.append(label)

        print(f"Loaded {len(image_paths)} images for {self.name}")
        return image_paths, labels


class Flowers102Dataset(BaseDataset):
    """Oxford 102 Flowers dataset (102 flower categories)"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.name = "flowers102"
        self.num_classes = 102

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        # Flowers102 has a fixed split
        dataset = datasets.Flowers102(
            root=self.data_root,
            split="train",
            download=True,
        )
        return self._flowers_to_paths(dataset)

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.Flowers102(
            root=self.data_root,
            split="test",
            download=True,
        )
        return self._flowers_to_paths(dataset)

    def _flowers_to_paths(self, dataset) -> Tuple[List[str], List[int]]:
        import tempfile
        temp_dir = os.path.join(self.data_root, self.name, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        image_paths = []
        labels = []

        for idx, (img, label) in enumerate(dataset):
            # Labels are 1-102, convert to 0-101
            label = label - 1
            img_path = os.path.join(temp_dir, f"img_{idx}.jpg")
            img.save(img_path)
            image_paths.append(img_path)
            labels.append(label)

        print(f"Loaded {len(image_paths)} images for {self.name}")
        return image_paths, labels


class PetsDataset(BaseDataset):
    """Oxford-IIIT Pets Dataset (37 pet breeds)"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.name = "pets"
        self.num_classes = 37

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.OxfordIIITPet(
            root=self.data_root,
            split="trainval",
            download=True,
        )
        return self._pets_to_paths(dataset)

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        dataset = datasets.OxfordIIITPet(
            root=self.data_root,
            split="test",
            download=True,
        )
        return self._pets_to_paths(dataset)

    def _pets_to_paths(self, dataset) -> Tuple[List[str], List[int]]:
        import tempfile
        temp_dir = os.path.join(self.data_root, self.name, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        image_paths = []
        labels = []

        for idx, (img, label) in enumerate(dataset):
            img_path = os.path.join(temp_dir, f"img_{idx}.jpg")
            img.save(img_path)
            image_paths.append(img_path)
            labels.append(label)

        print(f"Loaded {len(image_paths)} images for {self.name}")
        return image_paths, labels


class Food101Dataset(BaseDataset):
    """Food-101 dataset (101 food categories)"""

    def __init__(self, data_root: str = "data"):
        super().__init__(data_root)
        self.name = "food101"
        self.num_classes = 101
        self.data_path = os.path.join(data_root, "food-101")

    def load_train_data(self) -> Tuple[List[str], List[int]]:
        # Download if not exists
        self._download_food101()
        return self._load_food_split("train")

    def load_test_data(self) -> Tuple[List[str], List[int]]:
        self._download_food101()
        return self._load_food_split("test")

    def _download_food101(self):
        """Download Food-101 dataset using kagglehub or torchvision"""
        if os.path.exists(os.path.join(self.data_path, "images")):
            return

        print("Downloading Food-101 dataset...")
        try:
            # Try torchvision first (newer versions)
            dataset = datasets.Food101(
                root=self.data_root,
                download=True,
            )
            print("Food-101 downloaded successfully")
        except Exception as e:
            print(f"Could not auto-download Food-101: {e}")
            print("Please download manually from:")
            print("  https://www.kaggle.com/datasets/dansbecker/food-101")
            print(f"  Extract to: {self.data_path}")

    def _load_food_split(self, split: str) -> Tuple[List[str], List[int]]:
        """Load Food-101 split"""
        meta_file = os.path.join(self.data_path, "meta", f"{split}.txt")
        images_dir = os.path.join(self.data_path, "images")

        if not os.path.exists(meta_file):
            raise FileNotFoundError(
                f"Food-101 meta file not found: {meta_file}\n"
                "Please download the dataset first."
            )

        image_paths = []
        labels = []

        # Build class mapping
        classes = sorted(os.listdir(images_dir))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        with open(meta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: class_name/img_name.jpg
                parts = line.split('/')
                if len(parts) != 2:
                    continue

                class_name = parts[0]
                img_name = parts[1]
                img_path = os.path.join(images_dir, class_name, img_name)

                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])

        print(f"Loaded {len(image_paths)} images for {self.name} ({split})")
        return image_paths, labels


# Dataset registry
DATASET_REGISTRY = {
    "stanford_cars": StanfordCarsDataset,
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "flowers102": Flowers102Dataset,
    "pets": PetsDataset,
    "food101": Food101Dataset,
}


def get_dataset(dataset_name: str, data_root: str = "data") -> BaseDataset:
    """
    Get dataset instance by name

    Args:
        dataset_name: Name of the dataset
        data_root: Root directory for data

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    return DATASET_REGISTRY[dataset_name](data_root)


def list_datasets() -> List[str]:
    """List all available datasets"""
    return list(DATASET_REGISTRY.keys())


# Dataset info for documentation
DATASET_INFO = {
    "stanford_cars": {
        "name": "Stanford Cars",
        "num_classes": 196,
        "train_size": 8144,
        "test_size": 8041,
        "image_size": "variable",
        "type": "Fine-grained",
        "description": "196 car classes",
    },
    "cifar10": {
        "name": "CIFAR-10",
        "num_classes": 10,
        "train_size": 50000,
        "test_size": 10000,
        "image_size": "32x32",
        "type": "General",
        "description": "10 object categories",
    },
    "cifar100": {
        "name": "CIFAR-100",
        "num_classes": 100,
        "train_size": 50000,
        "test_size": 10000,
        "image_size": "32x32",
        "type": "General",
        "description": "100 object categories",
    },
    "flowers102": {
        "name": "Flowers-102",
        "num_classes": 102,
        "train_size": 1020,
        "test_size": 6149,
        "image_size": "variable",
        "type": "Fine-grained",
        "description": "102 flower categories",
    },
    "pets": {
        "name": "Oxford-IIIT Pets",
        "num_classes": 37,
        "train_size": 3680,
        "test_size": 3669,
        "image_size": "variable",
        "type": "Fine-grained",
        "description": "37 pet breeds (dogs/cats)",
    },
    "food101": {
        "name": "Food-101",
        "num_classes": 101,
        "train_size": 75750,
        "test_size": 25250,
        "image_size": "variable",
        "type": "Fine-grained",
        "description": "101 food categories",
    },
}


if __name__ == "__main__":
    # Test dataset loading
    print("Available datasets:")
    for name, info in DATASET_INFO.items():
        print(f"  {name}: {info['num_classes']} classes, {info['type']}")

"""Training with HuggingFace Trainer."""

import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional
import os


class FeatureTrainer(Trainer):
    """Custom Trainer for feature classification.

    Handles feature extraction before training.
    """

    def __init__(self, feature_extractor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to handle feature extraction."""
        pixel_values = inputs.pop("pixel_values")
        labels = inputs.pop("labels")

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss


class PrecomputedDataset(torch.utils.data.Dataset):
    """Dataset with pre-computed features for faster training.

    Features are extracted once at initialization.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
        }


def precompute_features(model, dataloader, device: str = "cuda:0"):
    """Pre-compute features for entire dataset.

    This speeds up training by only running the frozen backbone once.

    Args:
        model: FeatureClassifier model
        dataloader: DataLoader with (pixel_values, labels)
        device: Device to use

    Returns:
        features, labels tensors
    """
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    print("Pre-computing features...")
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch[0].to(device)
            labels = batch[1]

            features = model.extract_features(pixel_values)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Pre-computed {len(features)} features, shape: {features.shape}")
    return features, labels


def create_training_args(
    output_dir: str = "./outputs",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    device: str = "cuda:0",
) -> TrainingArguments:
    """Create HF TrainingArguments."""

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",  # Disable wandb/mlflow
        no_cuda=(device == "cpu"),
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()

    return {"accuracy": accuracy}

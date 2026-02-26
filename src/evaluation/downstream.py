"""Downstream task evaluation."""

from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


class DownstreamTask(ABC):
    """Base class for downstream tasks."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train downstream classifier on embeddings."""
        pass

    @abstractmethod
    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate downstream task performance."""
        pass


class BinaryClassificationTask(DownstreamTask):
    """
    One-vs-One binary classification tasks.

    For a C-class problem, creates C*(C-1)/2 binary classifiers.
    """

    def __init__(self, n_classes: int, normalize_emb: bool = True):
        self.n_classes = n_classes
        self.normalize_emb = normalize_emb
        self.classifiers: Dict[Tuple[int, int], LogisticRegression] = {}

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train all pairwise classifiers."""
        if self.normalize_emb:
            embeddings = normalize(embeddings)

        for class0 in range(self.n_classes):
            for class1 in range(class0 + 1, self.n_classes):
                # Get data for this pair
                idx0 = labels == class0
                idx1 = labels == class1

                if idx0.sum() == 0 or idx1.sum() == 0:
                    continue

                X = np.concatenate([embeddings[idx0], embeddings[idx1]], axis=0)
                y = np.array([0] * idx0.sum() + [1] * idx1.sum())

                # Train classifier
                clf = LogisticRegression(max_iter=100, random_state=42)
                clf.fit(X, y)

                self.classifiers[(class0, class1)] = clf
                self.classifiers[(class1, class0)] = clf

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate on all pairwise tasks.

        Returns:
            Dictionary with:
            - brier_score: Brier score per sample per class
            - pred_entropy: Prediction entropy per sample per class
            - accuracy: Binary accuracy per sample per class
        """
        if self.normalize_emb:
            embeddings = normalize(embeddings)

        n_samples = len(embeddings)

        # Per-sample metrics
        brier_score = np.full((self.n_classes, n_samples), np.nan)
        pred_entropy = np.full((self.n_classes, n_samples), np.nan)
        accuracy = np.full((self.n_classes, n_samples), np.nan)

        for class0 in range(self.n_classes):
            for class1 in range(class0 + 1, self.n_classes):
                if (class0, class1) not in self.classifiers:
                    continue

                clf = self.classifiers[(class0, class1)]

                # Get test samples for this pair
                idx0 = labels == class0
                idx1 = labels == class1

                if idx0.sum() == 0 and idx1.sum() == 0:
                    continue

                test_idx = idx0 | idx1
                X_test = embeddings[test_idx]
                y_true = np.array([0] * idx0.sum() + [1] * idx1.sum())

                # Get predictions
                probs = clf.predict_proba(X_test)
                prob_correct = probs[:, 0] if idx0.sum() > 0 else probs[:, 1]

                # Compute metrics
                brier = 2 * (1 - prob_correct) ** 2
                ent = -prob_correct * np.log(prob_correct + 1e-10) - \
                      (1 - prob_correct) * np.log(1 - prob_correct + 1e-10)
                acc = (probs.argmax(axis=1) == y_true).astype(float)

                # Store per-sample metrics
                indices = np.where(test_idx)[0]
                brier_score[class0, indices[idx0]] = brier[:idx0.sum()]
                brier_score[class1, indices[idx1]] = brier[idx0.sum():]

                pred_entropy[class0, indices[idx0]] = ent[:idx0.sum()]
                pred_entropy[class1, indices[idx1]] = ent[idx0.sum():]

                accuracy[class0, indices[idx0]] = acc[:idx0.sum()]
                accuracy[class1, indices[idx1]] = acc[idx0.sum():]

        return {
            "brier_score": brier_score,
            "pred_entropy": pred_entropy,
            "accuracy": accuracy,
        }


class MultiClassificationTask(DownstreamTask):
    """Standard multi-class classification."""

    def __init__(self, n_classes: int, normalize_emb: bool = True):
        self.n_classes = n_classes
        self.normalize_emb = normalize_emb
        self.classifier: Optional[LogisticRegression] = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train multi-class classifier."""
        if self.normalize_emb:
            embeddings = normalize(embeddings)

        self.classifier = LogisticRegression(
            max_iter=100,
            random_state=42,
            multi_class="ovr"
        )
        self.classifier.fit(embeddings, labels)

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate multi-class performance."""
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")

        if self.normalize_emb:
            embeddings = normalize(embeddings)

        probs = self.classifier.predict_proba(embeddings)

        # Brier score
        one_hot = np.zeros((len(labels), self.n_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        brier = np.sum((probs - one_hot) ** 2, axis=1)

        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # Accuracy
        preds = probs.argmax(axis=1)
        accuracy = (preds == labels).astype(float)

        return {
            "brier_score": brier,
            "pred_entropy": entropy,
            "accuracy": accuracy,
            "probs": probs,
        }

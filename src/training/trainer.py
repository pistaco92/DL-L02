"""Ciclo de entrenamiento y evaluacion de un modelo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.data.dataset import CognitiveSequenceDataset
from src.evaluation.metrics import MetricsCalculator
from src.models.factory import ModelFactory
from src.tasks.base import ClassificationTask
from src.tasks.factory import TaskFactory
from src.utils import set_seed


@dataclass
class FitResult:
    """Resultado de entrenar en un conjunto y evaluar en otro."""

    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    train_losses: list[float] = field(default_factory=list)

    @property
    def final_train_loss(self) -> float:
        return self.train_losses[-1] if self.train_losses else float("nan")


class Trainer:
    """
    Entrena un modelo con Adam usando la perdida y la regla de prediccion
    que define la tarea (SoftmaxTask o CoralTask).

    Uso tipico:
        trainer = Trainer.build("conv1d_softmax", model_cfg, train_cfg,
                                num_features=15, num_classes=3,
                                device=device, seed=42)
        result = trainer.fit_and_evaluate(X_tr, y_tr, X_ev, y_ev, seed=42)
    """

    def __init__(
        self,
        model: nn.Module,
        task: ClassificationTask,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.task = task
        self.config = config
        self.device = device

    @classmethod
    def build(
        cls,
        algorithm: str,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        num_features: int,
        num_classes: int,
        device: torch.device,
        seed: int,
    ) -> "Trainer":
        """Fija la semilla y crea modelo y tarea desde las fabricas."""

        set_seed(seed)
        model = ModelFactory.create(algorithm, num_features, num_classes, model_config)
        task = TaskFactory.create(algorithm, num_classes)
        return cls(model, task, training_config, device)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool, seed: int
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            CognitiveSequenceDataset(X, y),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
    ) -> float:
        """Una pasada por el conjunto de entrenamiento. Devuelve la perdida media."""

        self.model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            logits = self.model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(loader.dataset)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> list[float]:
        """Entrena config.epochs epocas. Devuelve la perdida de cada epoca."""

        loader = self._make_loader(X, y, shuffle=True, seed=seed)
        criterion = self.task.build_criterion(y, self.config, self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return [
            self.train_one_epoch(loader, optimizer, criterion)
            for _ in range(self.config.epochs)
        ]

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clases enteras (N,) para X (N, 15)."""

        self.model.eval()
        loader = self._make_loader(X, np.zeros(len(X), dtype=np.int64), False, 0)
        predictions = []
        for inputs, _ in loader:
            logits = self.model(inputs.to(self.device))
            predictions.append(self.task.predict(logits).cpu().numpy())
        return np.concatenate(predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        """Devuelve (metricas, y_pred)."""

        y_pred = self.predict(X)
        return MetricsCalculator.compute(y, y_pred), y_pred

    def fit_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        seed: int,
    ) -> FitResult:
        """Entrena en (X_train, y_train) y evalua en (X_eval, y_eval)."""

        losses = self.fit(X_train, y_train, seed)
        metrics, y_pred = self.evaluate(X_eval, y_eval)
        return FitResult(
            metrics=metrics,
            y_true=np.asarray(y_eval),
            y_pred=y_pred,
            train_losses=losses,
        )

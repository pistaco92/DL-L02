"""Interfaz comun de las tareas de clasificacion (Softmax u ordinal)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch

from src.config import TrainingConfig

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ClassificationTask(ABC):
    """
    Estrategia que define como se entrena y como se predice.

    Separa el "que se optimiza" del modelo y del ciclo de entrenamiento:
    el Trainer solo llama a build_criterion() y predict().

    - SoftmaxTask: CrossEntropyLoss sobre K logits, prediccion argmax.
    - CoralTask:   BCE sobre K-1 umbrales, prediccion por conteo.
    """

    name: str = "base"

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError("Se requieren al menos dos clases.")
        self.num_classes = num_classes

    @abstractmethod
    def build_criterion(
        self,
        y_train: np.ndarray,
        config: TrainingConfig,
        device: torch.device,
    ) -> LossFunction:
        """
        Devuelve la funcion de perdida loss(logits, labels).

        y_train permite calcular pesos por clase con los datos de
        entrenamiento del fold (nunca con los de evaluacion).
        """

    @abstractmethod
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Convierte logits en clases enteras (B,) en 0 .. K-1."""

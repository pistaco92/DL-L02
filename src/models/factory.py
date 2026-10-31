"""Registro de modelos por nombre de algoritmo."""

from __future__ import annotations

import torch.nn as nn

from src.config import ALGORITHM_CORAL, ALGORITHM_SOFTMAX, ModelConfig
from src.models.conv_softmax import Conv1DSoftmaxNet
from src.models.coral import Conv1DCoralNet


class ModelFactory:
    """
    Crea el modelo asociado a cada algoritmo.

    Todas las clases registradas deben aceptar los argumentos:
    num_features, num_classes, conv_channels, kernel_size, hidden_dim, dropout.

    Para probar otra implementacion (por ejemplo la solucion del profesor)
    basta con registrarla con el mismo nombre:

        ModelFactory.register("conv1d_coral", MiConv1DCoralNet)
    """

    _registry: dict[str, type[nn.Module]] = {
        ALGORITHM_SOFTMAX: Conv1DSoftmaxNet,
        ALGORITHM_CORAL: Conv1DCoralNet,
    }

    @classmethod
    def register(cls, algorithm: str, model_class: type[nn.Module]) -> None:
        cls._registry[algorithm] = model_class

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(
        cls,
        algorithm: str,
        num_features: int,
        num_classes: int,
        config: ModelConfig,
    ) -> nn.Module:
        if algorithm not in cls._registry:
            raise KeyError(
                f"Algoritmo sin modelo registrado: {algorithm}. "
                f"Disponibles: {cls.available()}."
            )
        model_class = cls._registry[algorithm]
        return model_class(
            num_features=num_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_size=config.kernel_size,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

"""
Experimento 2: Conv1D + MLP con cabeza ordinal CORAL.

Este archivo contiene esqueletos que el alumno debe completar.
El tronco Conv1D + MLP (Conv1DBackbone) ya esta implementado y es el mismo
del experimento Softmax: solo cambia la cabeza de salida.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import Conv1DBackbone


class CoralLayer(nn.Module):
    """
    Capa de salida CORAL.

    Produce K-1 logits acumulativos a partir de un vector de
    caracteristicas de tamano input_size.

    Formas esperadas:
    - x: (batch_size, input_size)
    - salida: (batch_size, num_classes - 1)
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes debe ser al menos 2.")
        self.input_size = input_size
        self.num_classes = num_classes
        self.linear = nn.Linear(input_size, 1, bias=False)
        self.initial_bias = nn.Parameter(torch.zeros(1))
        self.bias_deltas = nn.Parameter(torch.zeros(num_classes - 2))

    def get_ordered_biases(self) -> torch.Tensor:
        if self.num_classes == 2:
            return self.initial_bias
        deltas = torch.nn.functional.softplus(self.bias_deltas)
        offsets = torch.cumsum(deltas, dim=0)
        return torch.cat([self.initial_bias, self.initial_bias - offsets], dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = self.linear(inputs)  # (B, 1)
        biases = self.get_ordered_biases()  # (K-1,)
        return score + biases.unsqueeze(0)


class Conv1DCoralNet(nn.Module):
    """
    Conv1D + MLP con cabeza CORAL.

    Arquitectura:
        (B, 15) -> Conv1DBackbone -> (B, hidden_dim)
                -> CoralLayer(hidden_dim, K) -> (B, K-1)

    El forward devuelve logits de forma (batch_size, K-1).
    No aplica sigmoide en el forward: la perdida usa BCE con logits.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Conv1DBackbone(
            num_features=num_features,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = CoralLayer(self.backbone.output_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))

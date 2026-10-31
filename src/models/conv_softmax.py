"""Experimento 1: Conv1D + MLP con salida Softmax (multiclase categorica)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import Conv1DBackbone


class Conv1DSoftmaxNet(nn.Module):
    """
    Conv1D + MLP + capa lineal de K logits.

    El forward devuelve logits. No se aplica Softmax aqui porque
    CrossEntropyLoss lo hace internamente (log_softmax + NLLLoss).

    Formas:
        x: (B, 15) o (B, 1, 15)  ->  logits (B, K)
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
        self.head = nn.Linear(self.backbone.output_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))

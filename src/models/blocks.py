"""
Bloques reutilizables de la arquitectura Conv1D + MLP.

Trazado de formas con la configuracion por defecto (conv_channels=(16, 32),
kernel_size=3, hidden_dim=32):

    x            (B, 15)
    unsqueeze    (B, 1, 15)
    Conv1DBlock  (B, 16, 15)
    Conv1DBlock  (B, 32, 15)
    Flatten      (B, 480)
    MLPBlock     (B, 32)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """
    Conv1d -> BatchNorm1d -> ReLU.

    Usa padding = kernel_size // 2 (kernel impar), por lo que el largo de
    la secuencia se conserva:

        L_out = floor((L + 2p - d(k - 1) - 1) / s) + 1 = L   (s = 1, d = 1)

    Formas:
        x: (B, in_channels, L)  ->  (B, out_channels, L)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar para conservar el largo.")
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.conv(inputs)))


class Conv1DFeatureExtractor(nn.Module):
    """
    Pila de Conv1DBlock seguida de Flatten.

    Se usa Flatten (y no un pooling global) para conservar la posicion de
    cada item del test: el MLP posterior sabe que canal se activo en que
    posicion de la secuencia.

    Formas:
        x: (B, in_channels, L)  ->  (B, conv_channels[-1] * L)
    """

    def __init__(
        self,
        sequence_length: int,
        conv_channels: tuple[int, ...],
        kernel_size: int,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for out_channels in conv_channels:
            blocks.append(Conv1DBlock(current_channels, out_channels, kernel_size))
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.output_dim = current_channels * sequence_length

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.blocks(inputs))


class MLPBlock(nn.Module):
    """
    Linear -> ReLU -> Dropout.

    Formas:
        x: (B, in_features)  ->  (B, hidden_dim)
    """

    def __init__(self, in_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(inputs)))


class Conv1DBackbone(nn.Module):
    """
    Tronco comun de ambos experimentos: Conv1DFeatureExtractor + MLPBlock.

    Acepta entradas (B, L) o (B, 1, L); si llega (B, L) agrega el canal.
    La cabeza (Softmax o CORAL) se conecta sobre output_dim = hidden_dim.

    Formas:
        x: (B, L) o (B, 1, L)  ->  (B, hidden_dim)
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: tuple[int, ...],
        kernel_size: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.features = Conv1DFeatureExtractor(
            sequence_length=num_features,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
        )
        self.mlp = MLPBlock(self.features.output_dim, hidden_dim, dropout)
        self.output_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        return self.mlp(self.features(inputs))

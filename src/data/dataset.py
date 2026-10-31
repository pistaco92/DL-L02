"""Dataset de PyTorch que presenta cada fila como una secuencia 1D."""

import numpy as np
import torch
from torch.utils.data import Dataset


class CognitiveSequenceDataset(Dataset):
    """
    Dataset para la Conv1D.

    nn.Conv1d espera tensores (batch, canales, largo). Cada fila tabular
    de 15 atributos se interpreta como una senal de 1 canal y largo 15.

    Formas:
        X: (N, 15)  ->  cada muestra: (1, 15) float32
        y: (N,)     ->  cada etiqueta: escalar int64 en 0 .. K-1
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        self.X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

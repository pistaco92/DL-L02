"""Utilidades transversales: semillas y dispositivo."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch para reproducibilidad."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Convierte 'cpu', 'cuda' o 'auto' en un torch.device valido."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero no hay GPU disponible.")
    return device

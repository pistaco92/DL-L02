"""Recodificacion del objetivo y particiones estratificadas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_COLUMNS
from src.data.loader import DataFrameLoader


class TargetEncoder:
    """
    Convierte una columna objetivo a indices enteros 0 .. K-1.

    El orden de las clases originales se conserva (orden ordinal).
    Ejemplo: clases originales 1, 2, 3 -> indices 0, 1, 2.
    """

    def __init__(self) -> None:
        self.classes_: list[int] = []
        self.class_to_idx_: dict[int, int] = {}

    @property
    def num_classes(self) -> int:
        return len(self.classes_)

    def fit(self, values: pd.Series) -> "TargetEncoder":
        """Aprende las clases ordenadas presentes en la columna."""

        self.classes_ = sorted(values.astype(int).unique().tolist())
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        """Devuelve un vector (N,) int64 con indices 0 .. K-1."""

        if not self.class_to_idx_:
            raise RuntimeError("TargetEncoder debe ajustarse con fit() primero.")
        return values.astype(int).map(self.class_to_idx_).to_numpy(dtype=np.int64)

    def fit_transform(self, values: pd.Series) -> np.ndarray:
        return self.fit(values).transform(values)


class StratifiedSplitter:
    """
    Envoltorio de StratifiedKFold con validaciones del laboratorio.

    La clase 7 de GDS solo tiene 2 muestras, por lo que con 5 folds la
    estratificacion es imposible. can_split() permite detectarlo antes.
    """

    def __init__(self, n_splits: int, random_state: int = 42) -> None:
        if n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2.")
        self.n_splits = n_splits
        self.random_state = random_state

    def can_split(self, y: np.ndarray) -> bool:
        """True si cada clase tiene al menos n_splits muestras."""

        _, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
        return len(counts) >= 2 and int(counts.min()) >= self.n_splits

    def split(self, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Devuelve una lista de pares (indices_train, indices_eval)."""

        labels = np.asarray(y).ravel()
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            raise ValueError("Se requieren al menos dos clases para validar.")
        if counts.min() < self.n_splits:
            raise ValueError(
                f"No se puede estratificar con {self.n_splits} folds: la clase "
                f"menos frecuente solo tiene {counts.min()} muestras. Reduzca "
                "los folds o use un objetivo distinto de GDS."
            )

        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        dummy = np.zeros(len(labels), dtype=np.float32)
        return list(splitter.split(dummy, labels))


@dataclass
class ExperimentData:
    """
    Datos listos para un experimento (una columna objetivo).

    Atributos:
        target_name: columna objetivo activa (por ejemplo GDS_R2).
        X: matriz (N, 15) float32.
        y: vector (N,) int64 con indices 0 .. K-1.
        classes: valores originales de cada clase, en orden.
    """

    target_name: str
    X: np.ndarray
    y: np.ndarray
    classes: list[int]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])


def prepare_experiment_data(
    dataframe: pd.DataFrame,
    target_name: str,
    loader: DataFrameLoader | None = None,
) -> ExperimentData:
    """Construye X e y para una columna objetivo."""

    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"Target invalido: {target_name}. Use uno de {TARGET_COLUMNS}.")
    if target_name not in dataframe.columns:
        raise ValueError(f"La columna objetivo {target_name} no existe en el dataset.")

    loader = loader or DataFrameLoader()
    encoder = TargetEncoder()
    X = loader.feature_matrix(dataframe)
    y = encoder.fit_transform(dataframe[target_name])
    return ExperimentData(target_name=target_name, X=X, y=y, classes=encoder.classes_)

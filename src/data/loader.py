"""Lectura del archivo de datos (CSV o SAV) y extraccion de la matriz X."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS


class DataFrameLoader:
    """
    Carga el dataset del laboratorio y extrae las columnas de entrada.

    Ejemplo:
        loader = DataFrameLoader()
        dataframe = loader.load("dataset/15 atributos R0-R5.sav")
        X = loader.feature_matrix(dataframe)   # (N, 15) float32
    """

    SUPPORTED_SUFFIXES = (".csv", ".sav")

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)

    def load(self, data_path: str | Path) -> pd.DataFrame:
        """Lee un archivo .csv o .sav y devuelve un DataFrame."""

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontro el archivo: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".sav":
            return self._read_sav(path)

        raise ValueError(
            f"Formato no soportado ({suffix}). Use uno de {self.SUPPORTED_SUFFIXES}."
        )

    @staticmethod
    def _read_sav(path: Path) -> pd.DataFrame:
        try:
            import pyreadstat
        except ImportError as error:
            raise ImportError(
                "Para leer archivos .sav debe instalar pyreadstat."
            ) from error

        dataframe, _ = pyreadstat.read_sav(path)
        return dataframe

    def validate(self, dataframe: pd.DataFrame) -> None:
        """Verifica que todas las columnas de entrada existan."""

        missing = [c for c in self.feature_columns if c not in dataframe.columns]
        if missing:
            raise ValueError(
                "Faltan columnas de entrada en el dataset: " + ", ".join(missing)
            )

    def feature_matrix(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Extrae X respetando el orden de FEATURE_COLUMNS.

        Salida: arreglo (N, num_features) en float32.
        """

        self.validate(dataframe)
        return dataframe[self.feature_columns].astype("float32").to_numpy()

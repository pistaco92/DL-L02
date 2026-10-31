"""Metricas multiclase y ordinales (mismas del Laboratorio 01)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mae_ordinal",
    "qwk",
    "accuracy_pm1",
    "errores_graves",
]


class MetricsCalculator:
    """
    Calcula todas las metricas del laboratorio sobre etiquetas enteras.

    Multiclase (no usan el orden):
        accuracy, balanced_accuracy, precision/recall/f1 macro.
    Ordinales (usan la distancia |y - y_hat|):
        mae_ordinal (menor es mejor), qwk (mayor es mejor),
        accuracy_pm1, errores_graves (|y - y_hat| >= 2, menor es mejor).
    """

    @staticmethod
    def _as_labels(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true e y_pred deben tener la misma forma.")
        if len(y_true) == 0:
            raise ValueError("No se puede calcular una metrica sobre un arreglo vacio.")
        return y_true, y_pred

    @classmethod
    def accuracy(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraccion de aciertos exactos."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(np.mean(y_true == y_pred))

    @classmethod
    def balanced_accuracy(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Promedio del recall por clase."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(balanced_accuracy_score(y_true, y_pred))

    @classmethod
    def precision_macro(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    @classmethod
    def recall_macro(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    @classmethod
    def f1_macro(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    @classmethod
    def mae_ordinal(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Media de |y - y_hat| sobre las clases enteras."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(mean_absolute_error(y_true, y_pred))

    @classmethod
    def quadratic_weighted_kappa(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Kappa de Cohen con penalizacion cuadratica (maximo 1)."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if len(labels) < 2:
            return 1.0 if np.array_equal(y_true, y_pred) else 0.0
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels))

    @classmethod
    def accuracy_pm1(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraccion con |y - y_hat| <= 1."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred) <= 1))

    @classmethod
    def errores_graves(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraccion con |y - y_hat| >= 2."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred) >= 2))

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Diccionario con todas las metricas de METRIC_KEYS."""

        return {
            "accuracy": cls.accuracy(y_true, y_pred),
            "balanced_accuracy": cls.balanced_accuracy(y_true, y_pred),
            "precision_macro": cls.precision_macro(y_true, y_pred),
            "recall_macro": cls.recall_macro(y_true, y_pred),
            "f1_macro": cls.f1_macro(y_true, y_pred),
            "mae_ordinal": cls.mae_ordinal(y_true, y_pred),
            "qwk": cls.quadratic_weighted_kappa(y_true, y_pred),
            "accuracy_pm1": cls.accuracy_pm1(y_true, y_pred),
            "errores_graves": cls.errores_graves(y_true, y_pred),
        }

    @classmethod
    def classification_report(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list | None = None,
    ) -> str:
        """Reporte por clase de precision, recall y F1."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        target_names = None
        if class_names is not None and len(class_names) >= int(labels.max()) + 1:
            target_names = [str(class_names[int(label)]) for label in labels]
        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            digits=4,
        )

    @classmethod
    def confusion_matrix(cls, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Matriz de confusion: filas = real, columnas = predicho."""

        y_true, y_pred = cls._as_labels(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        return confusion_matrix(y_true, y_pred, labels=labels)

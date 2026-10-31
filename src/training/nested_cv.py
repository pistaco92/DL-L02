"""Validacion cruzada anidada: el loop interno elige, el externo reporta."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig, describe_configs
from src.data.preprocessing import ExperimentData, StratifiedSplitter
from src.evaluation.metrics import METRIC_KEYS, MetricsCalculator
from src.training.search import CandidateScore, GridSearch
from src.training.trainer import Trainer


@dataclass
class OuterFoldResult:
    """Resultado de un fold externo."""

    outer_fold: int
    selected: CandidateScore
    inner_skipped: bool
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    final_train_loss: float


@dataclass
class ExperimentResult:
    """
    Resultado de un experimento completo (algoritmo x objetivo).

    summary contiene mean_<metrica> y std_<metrica> sobre los folds externos:
    es el unico numero que se reporta en el informe.
    """

    algorithm: str
    target_name: str
    classes: list[int]
    num_samples: int
    num_features: int
    outer_folds: int
    inner_folds: int
    use_grid: bool
    device: str
    epochs: int
    folds: list[OuterFoldResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)

    @property
    def last_fold(self) -> OuterFoldResult:
        return self.folds[-1]

    def last_fold_report(self) -> str:
        return MetricsCalculator.classification_report(
            self.last_fold.y_true, self.last_fold.y_pred, class_names=self.classes
        )

    def last_fold_confusion(self) -> np.ndarray:
        return MetricsCalculator.confusion_matrix(
            self.last_fold.y_true, self.last_fold.y_pred
        )

    def most_selected(self) -> CandidateScore:
        """Configuracion elegida con mas frecuencia entre los folds externos."""

        labels = [
            describe_configs(f.selected.model_config, f.selected.training_config)
            for f in self.folds
        ]
        winner, _ = Counter(labels).most_common(1)[0]
        return self.folds[labels.index(winner)].selected


class NestedCrossValidator:
    """
    Validacion anidada outer x inner con busqueda de hiperparametros.

    Para cada fold externo:
    1. GridSearch sobre folds internos del entrenamiento externo.
    2. Reentrenar la configuracion ganadora con todo el entrenamiento externo.
    3. Evaluar una sola vez en el fold externo de prueba.
    """

    def __init__(
        self,
        algorithm: str,
        base_model_config: ModelConfig,
        base_training_config: TrainingConfig,
        outer_folds: int,
        inner_folds: int,
        grid: list[dict[str, Any]] | None,
        seed: int,
        device: torch.device,
        verbose: bool = True,
    ) -> None:
        self.algorithm = algorithm
        self.base_model_config = base_model_config
        self.base_training_config = base_training_config
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.use_grid = bool(grid)
        self.candidates = list(grid) if grid else [{}]
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def _search(self) -> GridSearch:
        return GridSearch(
            algorithm=self.algorithm,
            candidates=self.candidates,
            base_model_config=self.base_model_config,
            base_training_config=self.base_training_config,
            inner_folds=self.inner_folds,
            device=self.device,
        )

    def run(self, data: ExperimentData) -> ExperimentResult:
        """Ejecuta la validacion anidada sobre un objetivo."""

        X, y = data.X, data.y
        outer_splits = StratifiedSplitter(self.outer_folds, self.seed).split(y)
        search = self._search()
        result = ExperimentResult(
            algorithm=self.algorithm,
            target_name=data.target_name,
            classes=data.classes,
            num_samples=len(y),
            num_features=data.num_features,
            outer_folds=self.outer_folds,
            inner_folds=self.inner_folds,
            use_grid=self.use_grid,
            device=str(self.device),
            epochs=self.base_training_config.epochs,
        )

        for outer_index, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            search_result = search.run(
                X_train,
                y_train,
                num_classes=data.num_classes,
                seed=self.seed + outer_index * 100,
            )
            if search_result.skipped:
                print(
                    f"Aviso: el fold externo {outer_index} de {data.target_name} no "
                    f"admite {self.inner_folds} folds internos estratificados. "
                    "Se usa la configuracion base sin busqueda."
                )

            selected = search_result.best
            final_seed = self.seed + outer_index * 1000
            trainer = Trainer.build(
                self.algorithm,
                selected.model_config,
                selected.training_config,
                num_features=data.num_features,
                num_classes=data.num_classes,
                device=self.device,
                seed=final_seed,
            )
            fit = trainer.fit_and_evaluate(X_train, y_train, X_test, y_test, final_seed)
            result.folds.append(
                OuterFoldResult(
                    outer_fold=outer_index,
                    selected=selected,
                    inner_skipped=search_result.skipped,
                    metrics=fit.metrics,
                    y_true=fit.y_true,
                    y_pred=fit.y_pred,
                    final_train_loss=fit.final_train_loss,
                )
            )

            if self.verbose:
                print(
                    f"  fold {outer_index}/{self.outer_folds}: "
                    f"{describe_configs(selected.model_config, selected.training_config)} "
                    f"| MAE interno={selected.mae_mean:.4f} "
                    f"| test F1_m={fit.metrics['f1_macro']:.4f} "
                    f"MAE={fit.metrics['mae_ordinal']:.4f}"
                )

        for metric in METRIC_KEYS:
            values = np.asarray([f.metrics[metric] for f in result.folds], dtype=float)
            result.summary[f"mean_{metric}"] = float(values.mean())
            result.summary[f"std_{metric}"] = float(values.std())
        return result
